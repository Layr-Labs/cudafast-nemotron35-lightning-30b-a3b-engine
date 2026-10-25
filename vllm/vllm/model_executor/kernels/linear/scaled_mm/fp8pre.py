# SPDX-License-Identifier: Apache-2.0
"""LOCAL PATCH (cudafast-nemotron35): the Mamba FP8 projections at prefill row counts
(257..512 rows, where CUTLASS takes its default 128x128x128 configuration) through a
same-arithmetic kernel (harness/kernels/fp8pre) that reads the original weight.

Guarded at load. The kernel library holds a few tile/stage variants per shape with
the same per-output arithmetic. On the first layer of each shape, every variant's BF16
output bits must equal the production `ops.cutlass_scaled_mm` call on that layer's
weight for several row counts, activation sets and scales; a CUDA graph of each exact
variant at 496 rows is then timed against the same graph of CUTLASS, and the variants
below MIN_GAIN of CUTLASS are ranked fastest first. Every layer of that shape (the
first included) takes the first ranked variant that passes the same bit check on its
own weight. A shape with no ranked variant, or a layer with no exact one, keeps
CUTLASS. FP8PRE=0 disables it.
"""

import os

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger

logger = init_logger(__name__)
ENABLED = os.environ.get("FP8PRE", "1") != "0"
M_LO, M_HI = 257, 512
CHECK_M = (496, 512, 257, 383)
TIME_M = 496
MIN_GAIN = 0.97
# weight data_ptr -> (the admitted weight itself, so its address is never reused; variant)
_admitted: dict[int, tuple[torch.Tensor, int]] = {}
_ranked: dict[tuple[int, int], list[int]] = {}  # (N, K) -> variants faster than CUTLASS, fastest first
_logged = False


def _cutlass(a, b, sa, sb):
    return ops.cutlass_scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)


def _activations(m: int, k: int, dev, gen):
    """Random bytes, small, large and every byte pattern, as E4M3 (NaN codes zeroed:
    the static activation quantization saturates and never emits them)."""
    raw = torch.randint(0, 256, (m, k), device=dev, dtype=torch.uint8, generator=gen)
    raw[(raw & 0x7F) == 0x7F] = 0
    small = (torch.randn((m, k), device=dev, generator=gen) * 0.05).to(torch.float8_e4m3fn)
    large = (torch.randn((m, k), device=dev, generator=gen) * 200.0).clamp(-448, 448)
    pattern = ((torch.arange(m, device=dev).view(m, 1) * 7 + torch.arange(k, device=dev).view(1, k))
               % 256).to(torch.uint8)
    pattern[(pattern & 0x7F) == 0x7F] = 0
    return (raw.view(torch.float8_e4m3fn), small, large.to(torch.float8_e4m3fn),
            pattern.view(torch.float8_e4m3fn))


@torch.no_grad()
def _exact(fp8pre, b: torch.Tensor, variant: int) -> bool:
    k, n = b.shape
    dev = b.device
    gen = torch.Generator(device=dev).manual_seed(20261025 + n)
    scales = [(torch.tensor([sa], device=dev), torch.tensor([sb], device=dev))
              for sa, sb in ((0.0123457, 0.00731), (3.3333333, 0.1428571))]
    for m in CHECK_M:
        for a in _activations(m, k, dev, gen):
            a = a.contiguous()
            for sa, sb in scales:
                got, want = fp8pre.gemm(a, b, sa, sb, variant), _cutlass(a, b, sa, sb)
                torch.cuda.synchronize(dev)
                if not torch.equal(got.view(torch.int16), want.view(torch.int16)):
                    return False
    return True


def _graph_us(fn, reps: int = 12, rounds: int = 7) -> float:
    """Median GPU time per call in microseconds from a CUDA graph of `reps` calls."""
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    ms = []
    for _ in range(rounds):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        e.synchronize()
        ms.append(s.elapsed_time(e))
    return sorted(ms)[rounds // 2] * 1000.0 / reps


@torch.no_grad()
def _rank(fp8pre, b: torch.Tensor, ready: list[tuple[int, str]]) -> list[int]:
    """The exact variants that time below MIN_GAIN of CUTLASS at TIME_M rows, fastest first."""
    k, n = b.shape
    dev = b.device
    gen = torch.Generator(device=dev).manual_seed(496)
    a = (torch.randn((TIME_M, k), device=dev, generator=gen) * 4.0).to(torch.float8_e4m3fn)
    sa, sb = torch.tensor([0.02], device=dev), torch.tensor([0.003], device=dev)
    cutlass_us = _graph_us(lambda: _cutlass(a, b, sa, sb))
    timed = []
    for v, label in ready:
        try:
            if not _exact(fp8pre, b, v):
                logger.warning("fp8pre %dx%d %s: output bits differ from CUTLASS; variant off",
                               n, k, label)
                continue
            us = _graph_us(lambda v=v: fp8pre.gemm(a, b, sa, sb, v))
        except Exception as exc:  # noqa: BLE001  (a failed variant is dropped)
            logger.warning("fp8pre %dx%d %s: guard failed (%r); variant off", n, k, label, exc)
            continue
        logger.info("fp8pre %dx%d %s at M=%d: %.1f us (CUTLASS %.1f us)",
                    n, k, label, TIME_M, us, cutlass_us)
        if us < MIN_GAIN * cutlass_us:
            timed.append((us, v))
    ranked = [v for _, v in sorted(timed)]
    labels = dict(ready)
    logger.info("fp8pre %dx%d: %s", n, k,
                "on, " + labels[ranked[0]] if ranked else "off, CUTLASS stays")
    return ranked


def prepare_weight(weight: torch.Tensor) -> bool:
    """Admit one layer's final column-major [K, N] weight, or leave it on CUTLASS."""
    if not ENABLED:
        return False
    try:
        import fp8pre
    except ImportError:
        return False
    try:
        ready = fp8pre.variants(weight)
        if not ready:
            return False
        k, n = weight.shape
        if (n, k) not in _ranked:
            _ranked[(n, k)] = _rank(fp8pre, weight, ready)
        variant = next((v for v in _ranked[(n, k)] if _exact(fp8pre, weight, v)), None)
        if variant is None:
            if _ranked[(n, k)]:
                logger.warning("fp8pre %dx%d: no ranked variant matches CUTLASS bits on this "
                               "layer; it keeps CUTLASS", n, k)
            return False
    except Exception as exc:  # noqa: BLE001  (any failed guard keeps CUTLASS)
        logger.warning("fp8pre: guard failed (%r); this layer keeps CUTLASS", exc)
        return False
    _admitted[weight.data_ptr()] = (weight, variant)
    return True


def maybe_gemm(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor):
    """The kernel's output for an admitted weight at a prefill row count, else None."""
    global _logged
    if (
        not _admitted
        or not M_LO <= a.shape[0] <= M_HI
        or b.data_ptr() not in _admitted
        or _admitted[b.data_ptr()][0].shape != b.shape
        or b.stride() != (1, b.shape[0])
        or a.dtype != torch.float8_e4m3fn
        or b.dtype != torch.float8_e4m3fn
        or not a.is_contiguous()
        or a.data_ptr() % 16
        or a.shape[1] != b.shape[0]
        or a.device != b.device
        or sa.device != a.device
        or sb.device != a.device
        or sa.dtype != torch.float32
        or sb.dtype != torch.float32
        or sa.numel() != 1
        or sb.numel() != 1
    ):
        return None
    import fp8pre

    if not _logged:
        _logged = True
        logger.info("fp8pre active: %d Mamba FP8 projections at %d..%d rows; FP8PRE=0 disables",
                    len(_admitted), M_LO, M_HI)
    return fp8pre.gemm(a, b, sa, sb, _admitted[b.data_ptr()][1])

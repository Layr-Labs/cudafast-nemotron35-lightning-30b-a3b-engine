"""marlinx: Marlin-exact NVFP4 W4A16 GEMM for 1 <= M <= 8 bf16 rows.

libmarlinx.so computes exactly what vLLM v0.27.1's ``ops.marlin_gemm`` computes
for the NVFP4 W4A16 dense layers of Nemotron-3.5 (shared experts and lm_head)
when M <= 8, from the same Marlin-format tensors (no repack), with a schedule
that streams the weights faster. See README.md for the arithmetic it preserves.

``MARLINX=0`` in the environment disables it; every call outside the guard
(shape allowlist, 1 <= M <= 8, bf16, alignment, no bias, fp32 reduce, not
inside torch.compile tracing) returns None and the caller runs stock Marlin.
"""

from __future__ import annotations

import ctypes
import logging
import os

import torch

__all__ = ["LIB_PATH", "available", "prepare", "maybe_apply", "gemm", "hog", "regs", "SHAPES", "maybe_apply_relu2"]

log = logging.getLogger(__name__)
_skip_logged = False

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libmarlinx.so")

# (N, K) of the layers the kernel was verified on: shared expert up, down; lm_head.
SHAPES = frozenset({(3712, 2688), (2688, 3712), (131072, 2688)})
# Shapes whose activations can be zero in whole k16 blocks: the shared expert's down
# projection only (its input is the up projection's relu^2, which is 0 where the up
# output is <= 0). The skip drops those blocks' weight/scale loads and MMA.
SKIP_SHAPES = frozenset({(2688, 3712)})
MAX_M = 8
# Register-ring depth in k16 blocks (csrc/marlinx.cu supports 4 and 8; same bits).
DEPTH = int(os.environ.get("MARLINX_DEPTH", "8"))
ENABLED = os.environ.get("MARLINX", "1") != "0"
SKIP_ENABLED = os.environ.get("MARLINX_ZSKIP", "1") != "0"

_lib = None
_load_error: str | None = None
_ready = False
_sms: dict[int, int] = {}


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    try:
        lib = ctypes.CDLL(LIB_PATH)
        lib.marlinx_init.argtypes = []
        lib.marlinx_init.restype = ctypes.c_int
        lib.marlinx_regs.argtypes = [ctypes.c_int]
        lib.marlinx_regs.restype = ctypes.c_int
        lib.marlinx_gemm.argtypes = [
            ctypes.c_void_p, ctypes.c_longlong, ctypes.c_int,  # A lda M
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # B S GS
            ctypes.c_void_p, ctypes.c_longlong,  # C ldc
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # N K sms depth
            ctypes.c_void_p,  # stream
        ]
        lib.marlinx_gemm.restype = ctypes.c_int
        if hasattr(lib, "marlinx_gemm_relu2"):
            lib.marlinx_gemm_relu2.argtypes = lib.marlinx_gemm.argtypes
            lib.marlinx_gemm_relu2.restype = ctypes.c_int
        lib.marlinx_hog.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                    ctypes.c_longlong, ctypes.c_void_p, ctypes.c_void_p]
        lib.marlinx_hog.restype = ctypes.c_int
        if hasattr(lib, "marlinx_gemm_skip"):
            lib.marlinx_gemm_skip.argtypes = lib.marlinx_gemm.argtypes
            lib.marlinx_gemm_skip.restype = ctypes.c_int
        _lib = lib
    except OSError as exc:  # library missing or unloadable: callers fall back
        _load_error = str(exc)
    return _lib


def available() -> bool:
    return _load() is not None


def _init() -> bool:
    """Loads the kernels into the context (must run outside CUDA graph capture)."""
    global _ready
    if _ready:
        return True
    lib = _load()
    if lib is None or torch.cuda.is_current_stream_capturing():
        return False
    rc = lib.marlinx_init()
    if rc != 0:
        raise RuntimeError(f"marlinx_init failed: {rc}")
    _ready = True
    return True


def regs(depth: int = DEPTH) -> int:
    lib = _load()
    assert lib is not None, _load_error
    return lib.marlinx_regs(depth)


def _sm_count(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    n = _sms.get(idx)
    if n is None:
        n = torch.cuda.get_device_properties(idx).multi_processor_count
        _sms[idx] = n
    return n


def prepare(layer) -> bool:
    """Marks a Marlin NVFP4 linear layer (after prepare_fp4_layer_for_marlin) as eligible."""
    layer._marlinx_ok = False
    layer._marlinx_skip_ok = False
    if not ENABLED or _load() is None:
        return False
    n = layer.output_size_per_partition
    k = layer.input_size_per_partition
    w, s, gs = layer.weight, layer.weight_scale, getattr(layer, "weight_global_scale", None)
    ok = (
        (n, k) in SHAPES
        and gs is not None
        and gs.dtype == torch.float32
        and gs.numel() == 1
        and w.dtype == torch.int32
        and tuple(w.shape) == (k // 16, 2 * n)
        and w.is_contiguous()
        and s.dtype == torch.float8_e4m3fn
        and tuple(s.shape) == (k // 16, n)
        and s.is_contiguous()
        and w.is_cuda
        and getattr(layer, "bias", None) is None
    )
    if ok:
        ok = _init()
    layer._marlinx_ok = ok
    if ok and (n, k) in SKIP_SHAPES and SKIP_ENABLED and hasattr(_lib, "marlinx_gemm_skip"):
        layer._marlinx_skip_ok = _skip_self_check(layer, n, k)
    return ok


def _skip_self_check(layer, n: int, k: int) -> bool:
    """Load-time guard for the zero-k16-block skip: bitwise against stock Marlin.

    Every k16 block whose 16 activations are zero in every row multiplies zero weights
    only, so it adds +-0 to an accumulator that starts at +0 and can only be changed by
    a nonzero product; dropping it leaves each output's fp32 chain and Marlin's segment
    fold identical. This runs the layer's real Marlin tensors and six synthetic sparse
    activations (M = 1, 4, 8, each with +0 and with -0 in the zeroed positions) and
    requires bit equality, which also covers any NaN scale in the weights (0 * NaN
    would otherwise become +0 instead of NaN, so a NaN scale must reject the skip).

    The zeros are placed per element, not only per whole block: the mask must be clear
    only when all 16 activations are zero, because the mma's B operand takes elements
    0..7 of the block from one activation word pair and 8..15 from the next. So the
    input mixes scattered element zeros, blocks zero in their first 8 elements only,
    blocks zero in their last 8 only, and blocks zero in all 16 elements of every row
    (the ones the skip may drop). Four blocks per row are forced to the first 8 zero /
    last 8 nonzero pattern in every row, with nonzero values in the last 8, so a mask
    that reads only the first half of a block fails this check on every run.
    """
    global _skip_logged
    try:
        if bool(((layer.weight_scale.view(torch.uint8) & 0x7F) == 0x7F).any()):
            log.warning("marlinx: NaN e4m3 scale in the shared down weights; zero-block skip off")
            return False
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            apply_fp4_marlin_linear,
        )

        dev = layer.weight.device
        gen = torch.Generator(device=dev)
        nblk = k // 16
        for m in (1, 4, MAX_M):
            for neg_zero in (False, True):
                gen.manual_seed(20240 + m + 100 * int(neg_zero))
                x = torch.randn((m, k), generator=gen, device=dev).relu().square()
                # zero[.,.,e] is element e of a k16 block, so the three kinds of block
                # below are distinguished at element granularity
                zero = torch.rand((m, nblk, 16), generator=gen, device=dev) < 0.10
                half = torch.rand((m, nblk), generator=gen, device=dev) < 0.30
                first = torch.rand((m, nblk), generator=gen, device=dev) < 0.5
                zero[:, :, :8] |= (half & first)[:, :, None]
                zero[:, :, 8:] |= (half & ~first)[:, :, None]
                # whole blocks, per row and shared, so the mask's any-row test and the
                # skipped path itself are exercised; the zeros are -0 in the second pass
                whole = (torch.rand((m, nblk), generator=gen, device=dev) < 0.12) | (
                    torch.rand((1, nblk), generator=gen, device=dev) < 0.15
                )
                zero |= whole[:, :, None]
                # forced adversarial blocks: first 8 elements zero in every row, last 8
                # nonzero in every row, one k16 block every 32 so all four mask warps
                # see one
                forced = list(range(0, min(nblk, 128), 32))
                for b in forced:
                    zero[:, b, :8] = True
                    zero[:, b, 8:] = False
                x = x.masked_fill(zero.reshape(m, k), 0)
                for b in forced:
                    x[:, 16 * b + 8 : 16 * b + 16] = 1.0 + b
                if neg_zero:
                    x = torch.where(x == 0, torch.full_like(x, -0.0), x)
                x = x.to(torch.bfloat16)
                ref = apply_fp4_marlin_linear(
                    input=x,
                    weight=layer.weight,
                    weight_scale=layer.weight_scale,
                    weight_global_scale=layer.weight_global_scale,
                    workspace=layer.workspace,
                    size_n=n,
                    size_k=k,
                    bias=None,
                )
                out = gemm(x, layer.weight, layer.weight_scale, layer.weight_global_scale, n, k, skip=True)
                if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                    log.warning(
                        "marlinx: zero-block skip differs from stock Marlin at M=%d (neg_zero=%d); "
                        "dense marlinx keeps this layer", m, int(neg_zero)
                    )
                    return False
    except Exception as exc:  # noqa: BLE001
        log.warning("marlinx: zero-block skip self-check failed (%r); dense marlinx keeps this layer", exc)
        return False
    if not _skip_logged:
        _skip_logged = True
        log.info("marlinx: shared-expert down projection skips zero-activation k16 blocks")
    return True


def gemm(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
         global_scale: torch.Tensor, n: int, k: int, depth: int = DEPTH,
         relu2: bool = False, skip: bool = False) -> torch.Tensor:
    """out[M, n] for a 2-D bf16 x[M, k] (1 <= M <= 8); no guard beyond the C side's."""
    lib = _load()
    assert lib is not None, _load_error
    m = x.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    op = lib.marlinx_gemm_relu2 if relu2 else (lib.marlinx_gemm_skip if skip else lib.marlinx_gemm)
    rc = op(
        x.data_ptr(), x.stride(0), m,
        weight.data_ptr(), weight_scale.data_ptr(), global_scale.data_ptr(),
        out.data_ptr(), n, n, k, _sm_count(x.device), depth,
        torch.cuda.current_stream(x.device).cuda_stream,
    )
    if rc != 0:
        raise RuntimeError(f"marlinx_gemm failed: {rc} (M={m} N={n} K={k} depth={depth})")
    return out


def maybe_apply(layer, x: torch.Tensor) -> torch.Tensor | None:
    """Marlin-exact output for an eligible layer and input, or None (caller runs Marlin)."""
    if not ENABLED or not getattr(layer, "_marlinx_ok", False):
        return None
    if torch.compiler.is_compiling():
        return None
    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    if (
        m < 1
        or m > MAX_M
        or x2.dtype != torch.bfloat16
        or x2.stride(-1) != 1
        or x2.stride(0) % 8
        or x2.data_ptr() % 16
    ):
        return None
    if not _ready and not _init():
        return None
    n = layer.output_size_per_partition
    out = gemm(x2, layer.weight, layer.weight_scale, layer.weight_global_scale, n, k,
               skip=bool(getattr(layer, "_marlinx_skip_ok", False)))
    return out.reshape(*x.shape[:-1], n)


def hog(src: torch.Tensor, blocks: int, threads: int, smem: int, sink: torch.Tensor) -> None:
    """Streams src with a simple read kernel (bandwidth probe / concurrent load)."""
    lib = _load()
    assert lib is not None, _load_error
    rc = lib.marlinx_hog(blocks, threads, smem, src.data_ptr(), src.numel() * src.element_size(),
                         sink.data_ptr(), torch.cuda.current_stream(src.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"marlinx_hog failed: {rc}")


def maybe_apply_relu2(x, weight, weight_scale, global_scale, n, k):
    """Opaque-op runtime path for a prepared single-device shared up projection."""
    if not ENABLED or not _ready or _load() is None:
        return None
    if not hasattr(_lib, "marlinx_gemm_relu2") or (n, k) != (3712, 2688):
        return None
    x2 = x.reshape(-1, k)
    if (
        not 1 <= x2.shape[0] <= MAX_M
        or x2.dtype != torch.bfloat16
        or not x2.is_cuda
        or x.shape[-1] != k
        or x2.stride(-1) != 1
        or x2.stride(0) % 8
        or x2.data_ptr() % 16
        or weight.dtype != torch.int32
        or weight.shape != (k // 16, 2 * n)
        or not weight.is_contiguous()
        or weight_scale.dtype != torch.float8_e4m3fn
        or weight_scale.shape != (k // 16, n)
        or not weight_scale.is_contiguous()
        or global_scale.dtype != torch.float32
        or global_scale.numel() != 1
        or not global_scale.is_contiguous()
        or weight.device != x.device
        or weight_scale.device != x.device
        or global_scale.device != x.device
    ):
        return None
    out = gemm(x2, weight, weight_scale, global_scale, n, k, relu2=True)
    return out.reshape(*x.shape[:-1], n)

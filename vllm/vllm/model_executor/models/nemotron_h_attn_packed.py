"""Lossless 12-bit storage of the target's BF16 attention projections for the 5-row verify.

Each of the six attention layers streams its QKV (4608x2688) and output (2688x4096)
projection from DRAM once per verify: 46.7 MB of BF16 per layer, about 1.25 ms of a
17.8 ms cycle. The weights are repacked once at load with the drafter's exact codec
(nemotron_h_mtp_packed.pack_bf16: sign+mantissa byte, 4-bit exponent code relative to a
per-128-block base, raw side plane for blocks whose exponents do not fit) and the
verify's M = 5 call decodes them in registers into the same BF16 MMA operands
(nemotron_h_attn_packed_kernel.py).

Exactness: the kernel is one tl.dot chain over K tiles of 128 into an fp32 accumulator,
rounded once to BF16. That is the reduction the cuBLAS call produces for M = 2..8 on
these shapes (measured bit-identical on the real weights by an earlier submission,
b9787b2). It is not assumed here: the boot guard compares every packed layer against
F.linear on random inputs and the whole path stays off on a single differing bit.
It also stays off unless a cold-cache timing of the chosen configuration beats
F.linear, and unless a child process has compiled and run every configuration on
synthetic planes first (a compiler abort must not take the engine with it).
Every other row count (prefill, M = 1) calls F.linear on the BF16 weight.

The planes live in a registry the custom op reads, so the compiled graph only sees
(x, weight, n, k, id). ATTN_PACK12=0 disables it.
"""

import os
import subprocess
import sys

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from .nemotron_h_attn_packed_kernel import CONFIGS, ROWS, run

logger = init_logger(__name__)

ENABLED = os.environ.get("ATTN_PACK12", "1") != "0"
SHAPES = frozenset({(4608, 2688), (2688, 4096)})
MIN_GAIN = 0.96
KERNEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nemotron_h_attn_packed_kernel.py")

_chosen: dict[tuple[int, int], tuple[int, int]] = {}
_registry: dict[int, tuple[torch.Tensor, ...]] = {}

_PROBE = """
import importlib.util, sys, torch
spec = importlib.util.spec_from_file_location("apk_kernel", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
dev = torch.device("cuda")
for n, k in ((4608, 2688), (2688, 4096)):
    nb = n * k // 128
    sm = torch.randint(0, 256, (n * k,), dtype=torch.uint8, device=dev)
    ex = torch.randint(0, 256, (n * k // 2,), dtype=torch.uint8, device=dev)
    base = torch.randint(60, 120, (nb,), dtype=torch.uint8, device=dev)
    ri = torch.full((nb,), -1, dtype=torch.int32, device=dev)
    ri[0] = 0
    raw = torch.randint(-32768, 32767, (128,), dtype=torch.int16, device=dev)
    x = torch.randn(m.ROWS, k, device=dev, dtype=torch.bfloat16)
    for cfg in m.CONFIGS:
        y = m.run(x, sm, ex, base, ri, raw, n, k, cfg)
        torch.cuda.synchronize()
        assert y.shape == (m.ROWS, n)
print("probe-ok")
"""


def _linear_impl(x: torch.Tensor, w: torch.Tensor, n: int, k: int, pid: int) -> torch.Tensor:
    cfg = _chosen.get((n, k))
    planes = _registry.get(pid)
    if (
        cfg is None
        or planes is None
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.dim() != 2
        or x.shape[0] != ROWS
        or x.shape[1] != k
        or x.stride(1) != 1
    ):
        return F.linear(x, w)
    return run(x, *planes, n, k, cfg)


def _linear_fake(x: torch.Tensor, w: torch.Tensor, n: int, k: int, pid: int) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], n))


direct_register_custom_op(
    op_name="nemo_attn_pk_linear", op_func=_linear_impl, fake_impl=_linear_fake,
)


def apply(proj: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """proj(x)[0] for an attention projection, packed when prepare() enabled it."""
    nk = getattr(proj, "_apk_nk", None)
    if nk is None:
        return proj(x)[0]
    return torch.ops.vllm.nemo_attn_pk_linear(x, proj.weight, nk[0], nk[1], proj._apk_id)


def _probe() -> bool:
    try:
        out = subprocess.run(
            [sys.executable, "-c", _PROBE, KERNEL_PATH],
            capture_output=True, text=True, timeout=240,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("attention pack12: probe did not run (%r); off", exc)
        return False
    if out.returncode != 0 or "probe-ok" not in out.stdout:
        logger.warning(
            "attention pack12: probe failed (rc %s): %s", out.returncode, out.stderr[-400:]
        )
        return False
    return True


def _exact(items, n: int, k: int, cfg: tuple[int, int]) -> bool:
    dev = items[0][0].weight.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(20261025)
    for proj, planes in items:
        for scale in (1.0, 30.0, 1e-3):
            for _ in range(8):
                x = torch.randn(ROWS, k, device=dev, dtype=torch.float32, generator=gen)
                x = (x * scale).to(torch.bfloat16)
                if not torch.equal(run(x, *planes, n, k, cfg), F.linear(x, proj.weight)):
                    return False
    return True


def _op_us(fn, flush: torch.Tensor, reps: int = 20) -> float:
    """Median cold-weight GPU time of fn in microseconds. The L2 flush keeps the GPU busy
    long enough for the host to queue the event pair and fn behind it, so the events
    bracket fn alone."""
    fn()
    torch.cuda.synchronize()
    spans = []
    for _ in range(reps):
        flush.zero_()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        spans.append((s, e))
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in spans)
    return ms[len(ms) // 2] * 1000.0


def _choose(items, n: int, k: int, flush: torch.Tensor):
    proj, planes = items[0]
    x = torch.randn(ROWS, k, device=proj.weight.device, dtype=torch.bfloat16)
    linear_us = _op_us(lambda: F.linear(x, proj.weight), flush)
    best = None
    for cfg in CONFIGS:
        if n % cfg[0]:
            continue
        try:
            us = _op_us(lambda: run(x, *planes, n, k, cfg), flush)
        except Exception as exc:  # noqa: BLE001  (a configuration this Triton cannot build)
            logger.warning("attention pack12 %dx%d: config %s failed (%r)", n, k, cfg, exc)
            continue
        if best is None or us < best[0]:
            best = (us, cfg)
    logger.info("attention pack12 %dx%d: linear %.1f us, packed %s", n, k, linear_us, best)
    if best is None or best[0] > MIN_GAIN * linear_us:
        return None
    return best[1]


@torch.no_grad()
def prepare(model: torch.nn.Module) -> int:
    """Pack the target's attention projections and enable the shapes that pass the
    guards. Returns the number of projections enabled."""
    if not ENABLED or not torch.cuda.is_available():
        return 0
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    projs = []
    for module in model.modules():
        if type(module).__name__ != "NemotronHAttention":
            continue
        for proj in (module.qkv_proj, module.o_proj):
            w = getattr(proj, "weight", None)
            if (
                w is not None
                and w.dtype == torch.bfloat16
                and w.is_cuda
                and w.is_contiguous()
                and tuple(w.shape) in SHAPES
                and isinstance(getattr(proj, "quant_method", None), UnquantizedLinearMethod)
            ):
                projs.append(proj)
    if not projs or not _probe():
        return 0

    from .nemotron_h_mtp_packed import pack_bf16

    groups: dict[tuple[int, int], list] = {}
    for proj in projs:
        packed = pack_bf16(proj.weight)
        if packed is None:
            continue
        sm, ex, base, ri, raw = packed.planes
        groups.setdefault(tuple(proj.weight.shape), []).append(
            (proj, (sm, ex, base, ri, raw.view(torch.int16)))
        )
    if not groups:
        return 0
    dev = projs[0].weight.device
    flush = torch.empty(96 << 20, dtype=torch.uint8, device=dev)
    enabled = 0
    for (n, k), items in groups.items():
        cfg = None
        try:
            cfg = _choose(items, n, k, flush)
            if cfg is not None and not _exact(items, n, k, cfg):
                logger.warning("attention pack12 %dx%d: output differs from F.linear; off", n, k)
                cfg = None
        except Exception as exc:  # noqa: BLE001
            logger.warning("attention pack12 %dx%d: guard failed (%r); off", n, k, exc)
            cfg = None
        if cfg is None:
            continue
        for proj, planes in items:
            pid = len(_registry)
            _registry[pid] = planes
            proj._apk_planes = planes
            proj._apk_nk = (n, k)
            proj._apk_id = pid
        _chosen[(n, k)] = cfg
        enabled += len(items)
    return enabled

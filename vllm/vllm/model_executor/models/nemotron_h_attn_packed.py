"""Lossless 12-bit storage of the target's BF16 attention projections for the 5-row verify.

Each of the six attention layers streams its QKV (4608x2688) and output (2688x4096)
projection from DRAM once per verify: 46.7 MB of BF16 per layer, about 1.25 ms of a
17.8 ms cycle. The weights are repacked once at load with the drafter's exact codec
(nemotron_h_mtp_packed.pack_bf16: sign+mantissa byte, 4-bit exponent code relative to a
per-128-block base, raw side plane for blocks whose exponents do not fit) and the
verify's M = 5 call decodes them in registers into the same BF16 MMA operands.

Exactness: the kernel is one tl.dot chain over K tiles of 128 into an fp32 accumulator,
rounded once to BF16. That is the reduction the cuBLAS call produces for M = 2..8 on
these shapes (measured bit-identical on the real weights by an earlier submission,
b9787b2). It is not assumed here: the boot guard compares every packed layer against
F.linear on random inputs and the whole path stays off on a single differing bit.
It also stays off unless a cold-cache timing of the chosen configuration beats
F.linear. Every other row count (prefill, M = 1) calls F.linear on the BF16 weight.

ATTN_PACK12=0 disables it.
"""

import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

ENABLED = os.environ.get("ATTN_PACK12", "1") != "0"
ROWS = 5
SHAPES = frozenset({(4608, 2688), (2688, 4096)})
# (columns per program, warps), tried in this order; BN is a multiple of 16.
CONFIGS = ((32, 4), (16, 4), (64, 4), (32, 2))
MIN_GAIN = 0.96
PLANES = ("sm", "ex", "base", "ri", "raw")

_chosen: dict[tuple[int, int], tuple[int, int]] = {}


@triton.jit
def _packed_gemm(
    x_ptr, sm_ptr, ex_ptr, base_ptr, ri_ptr, raw_ptr, y_ptr,
    M, N, KB, stride_xm, stride_ym,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    offs_m = tl.arange(0, BM)
    offs_n = tl.program_id(0) * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    row = offs_m < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for ki in range(0, KB):
        blk = offs_n * KB + ki
        rid = tl.load(ri_ptr + blk)
        bs = tl.load(base_ptr + blk).to(tl.int32)
        flat = blk[None, :] * BK + offs_k[:, None]
        sm = tl.load(sm_ptr + flat).to(tl.int32)
        nib = tl.load(ex_ptr + (flat >> 1)).to(tl.int32)
        code = (nib >> ((flat & 1) * 4)) & 15
        e = tl.where(code == 0, 0, bs[None, :] + code - 1)
        bits = ((sm & 128) << 8) | (e << 7) | (sm & 127)
        esc = rid >= 0
        rw = tl.load(
            raw_ptr + rid[None, :].to(tl.int64) * BK + offs_k[:, None],
            mask=esc[None, :],
            other=0,
        ).to(tl.int32)
        bits = tl.where(esc[None, :], rw, bits)
        b = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
        a = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (ki * BK + offs_k)[None, :],
            mask=row[:, None],
            other=0.0,
        )
        acc = tl.dot(a, b, acc)
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :],
        acc.to(y_ptr.dtype.element_ty),
        mask=row[:, None],
    )


def _run(x, sm, ex, base, ri, raw, n: int, k: int, cfg: tuple[int, int]) -> torch.Tensor:
    bn, warps = cfg
    y = x.new_empty((x.shape[0], n))
    _packed_gemm[(n // bn,)](
        x, sm, ex, base, ri, raw, y,
        x.shape[0], n, k // 128, x.stride(0), y.stride(0),
        BM=16, BN=bn, BK=128, num_warps=warps, num_stages=2,
    )
    return y


def _linear_impl(
    x: torch.Tensor, sm: torch.Tensor, ex: torch.Tensor, base: torch.Tensor,
    ri: torch.Tensor, raw: torch.Tensor, n: int, k: int, w: torch.Tensor,
) -> torch.Tensor:
    cfg = _chosen.get((n, k))
    if (
        cfg is None
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.dim() != 2
        or x.shape[0] != ROWS
        or x.shape[1] != k
        or x.stride(1) != 1
    ):
        return F.linear(x, w)
    return _run(x, sm, ex, base, ri, raw, n, k, cfg)


def _linear_fake(
    x: torch.Tensor, sm: torch.Tensor, ex: torch.Tensor, base: torch.Tensor,
    ri: torch.Tensor, raw: torch.Tensor, n: int, k: int, w: torch.Tensor,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], n))


direct_register_custom_op(
    op_name="nemo_attn_pk_linear", op_func=_linear_impl, fake_impl=_linear_fake,
)


def apply(proj: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """proj(x)[0] for an attention projection, packed when prepare() enabled it."""
    nk = getattr(proj, "_apk_nk", None)
    if nk is None:
        return proj(x)[0]
    return torch.ops.vllm.nemo_attn_pk_linear(
        x, proj._apk_sm, proj._apk_ex, proj._apk_base, proj._apk_ri, proj._apk_raw,
        nk[0], nk[1], proj.weight,
    )


def _exact(items, n: int, k: int, cfg: tuple[int, int]) -> bool:
    dev = items[0][0].weight.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(20261025)
    for proj, planes in items:
        for scale in (1.0, 30.0, 1e-3):
            for _ in range(8):
                x = torch.randn(ROWS, k, device=dev, dtype=torch.float32, generator=gen)
                x = (x * scale).to(torch.bfloat16)
                if not torch.equal(_run(x, *planes, n, k, cfg), F.linear(x, proj.weight)):
                    return False
    return True


def _op_us(fn, flush: torch.Tensor, reps: int = 16, rounds: int = 7) -> float:
    """Cold-weight time of fn in microseconds: a graph of `reps` (L2 flush, fn) pairs
    minus a graph of the flushes alone."""
    fn()
    flush.zero_()
    torch.cuda.synchronize()

    def timed(with_fn: bool) -> float:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(reps):
                flush.zero_()
                if with_fn:
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
        return sorted(ms)[rounds // 2]

    return (timed(True) - timed(False)) * 1000.0 / reps


def _choose(items, n: int, k: int, flush: torch.Tensor):
    proj, planes = items[0]
    x = torch.randn(ROWS, k, device=proj.weight.device, dtype=torch.bfloat16)
    linear_us = _op_us(lambda: F.linear(x, proj.weight), flush)
    best = None
    for cfg in CONFIGS:
        if n % cfg[0]:
            continue
        try:
            us = _op_us(lambda: _run(x, *planes, n, k, cfg), flush)
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
    boot guard. Returns the number of projections enabled."""
    if not ENABLED or not torch.cuda.is_available():
        return 0
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from .nemotron_h_mtp_packed import pack_bf16

    groups: dict[tuple[int, int], list] = {}
    for module in model.modules():
        if type(module).__name__ != "NemotronHAttention":
            continue
        for proj in (module.qkv_proj, module.o_proj):
            w = getattr(proj, "weight", None)
            if (
                w is None
                or w.dtype != torch.bfloat16
                or not w.is_cuda
                or not w.is_contiguous()
                or tuple(w.shape) not in SHAPES
                or type(getattr(proj, "quant_method", None)) is not UnquantizedLinearMethod
            ):
                continue
            packed = pack_bf16(w)
            if packed is None:
                continue
            groups.setdefault(tuple(w.shape), []).append((proj, packed.planes))
    if not groups:
        return 0
    dev = next(iter(groups.values()))[0][0].weight.device
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
            for name, plane in zip(PLANES, planes):
                proj.register_buffer("_apk_" + name, plane, persistent=False)
            proj._apk_nk = (n, k)
        _chosen[(n, k)] = cfg
        enabled += len(items)
    return enabled

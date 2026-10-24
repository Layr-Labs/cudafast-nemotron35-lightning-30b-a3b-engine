# SPDX-License-Identifier: Apache-2.0
"""Lossless 12-bit storage for the target attention projections at 2..8 rows.

LOCAL PATCH (cudafast-nemotron35). The target's only BF16 weights are the six
attention layers' fused QKV (4608 x 2688) and output projection (2688 x 4096):
281 MB read by every depth-4 verify (M=5), on cuBLAS at about 222 GB/s.

Storage: ArthurOstapenko b7a45bfa's MTP_PACK12 codec, unchanged. Per 128-value
row block, an 8-bit sign/mantissa plane, a 4-bit exponent code relative to a
per-block base byte (code 0 keeps exponent 0, so zeros and subnormals keep
their bits), and a raw 16-bit side plane for blocks whose nonzero exponents
span more than 15 values. The pack is verified at boot against every original
BF16 bit pattern.

Arithmetic: b7a45bfa's small-M kernel (`packed_small`): BM 16, BN 64, BK 128,
four warps, MMA v2 m16n8k16. Every output starts at fp32 +0 and adds its K in
ascending 128-wide tiles, then rounds once to BF16. That is the persistent
Triton tile that ivanfioravanti b9787b23 measured bit-identical to cuBLAS at
M=2..8 on these two shapes (97.5M outputs, eager, compiled and graph replay;
M=1 differs, so it is excluded). The weight enters the MMA as the same BF16
values the decoder restores bit for bit.

Guard: before any projection uses the packed path, a boot check compares the
packed kernel with `torch.nn.functional.linear` on the original weight, bit
for bit, at every M in 2..8 over several synthetic input distributions (fixed
seed, no prompt data). A single differing bit in any projection turns the
packed path off for all of them. The original BF16 weight stays resident, so
M=1, M>8 (the 496-token prefill) and every fallback run the record's cuBLAS
call unchanged. ATTN_PACK12=0 restores the record.
"""

import os

import torch
import torch.nn.functional as F
import triton
import triton.language as pl
from triton.experimental import gluon
from triton.experimental.gluon import language as tl
from triton.experimental.gluon.language import (
    BlockedLayout,
    DotOperandLayout,
    NVMMADistributedLayout,
    SliceLayout,
)
from triton.experimental.gluon.language.nvidia.ampere import mma_v2

from vllm.utils.torch_utils import direct_register_custom_op

SHAPES = ((4608, 2688), (2688, 4096))
MIN_ROWS, MAX_ROWS = 2, 8
ENABLED = os.environ.get("ATTN_PACK12", "1") != "0"
_SUFFIXES = ("sm", "ex", "base", "ri", "raw")


@gluon.jit
def load_packed(SM, EX, BASE, RI, RAW, index, mask):
    # K is a multiple of 128, so plane and escape blocks share the source
    # matrix's flattened row-major block index.
    bid = index // 128
    sm = tl.load(SM + index, mask=mask, other=0, eviction_policy='evict_first').to(tl.uint16)
    nib = tl.load(EX + index // 2, mask=mask, other=0, eviction_policy='evict_first').to(tl.uint16)
    code = (nib >> ((index % 2)*4)) & 15
    base = tl.load(BASE + bid, mask=mask, other=1, eviction_policy='evict_first').to(tl.uint16)
    exp = tl.where(code == 0, 0, base + code - 1)
    bits = ((sm & 128) << 8) | (exp << 7) | (sm & 127)
    rawid = tl.load(RI + bid, mask=mask, other=-1, eviction_policy='evict_first')
    raw = tl.load(RAW + rawid.to(tl.int64)*128 + index % 128,
                  mask=mask & (rawid >= 0), other=0, eviction_policy='evict_first')
    bits = tl.where(rawid >= 0, raw, bits).to(tl.uint16)
    return bits.to(tl.bfloat16, bitcast=True)


@gluon.jit
def verify_decoded(SM, EX, BASE, RI, RAW, W, COUNTER, N: tl.constexpr, B: tl.constexpr):
    ix = tl.program_id(0)*B + tl.arange(0, B, layout=BlockedLayout([1], [32], [4], [0]))
    actual = load_packed(SM, EX, BASE, RI, RAW, ix, ix < N).to(tl.uint16, bitcast=True)
    expected = tl.load(W + ix, mask=ix < N, other=0)
    bad = tl.sum(((actual != expected) & (ix < N)).to(tl.int32), axis=0)
    tl.atomic_add(COUNTER, bad, sem='relaxed')


@triton.jit
def _pack_stats(W, BASE, FIT, NB: pl.constexpr, B: pl.constexpr):
    blocks = pl.program_id(0) * B + pl.arange(0, B)
    k = pl.arange(0, 128)
    bits = pl.load(W + blocks[:, None] * 128 + k[None, :],
                   blocks[:, None] < NB, other=0).to(pl.uint16)
    exp = (bits >> 7) & 255
    low = pl.min(pl.where(exp == 0, 256, exp).to(pl.int32), 1)
    high = pl.max(exp.to(pl.int32), 1)
    base = pl.where(high == 0, 1, low)
    fit = (high == 0) | (high - low <= 14)
    pl.store(BASE + blocks, base, blocks < NB)
    pl.store(FIT + blocks, ~fit, blocks < NB)


@gluon.jit
def packed_small(x, SM, EX, BASE, RI, RAW, y, M, N, K, sxm, sym,
                 BM: tl.constexpr = 16, BN: tl.constexpr = 64, BK: tl.constexpr = 128):
    ld: tl.constexpr = BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0])
    mma: tl.constexpr = NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 4], instr_shape=[16, 8])
    pid = tl.program_id(0)
    nn = pid*BN + tl.arange(0, BN, layout=SliceLayout(1, ld))
    kk0 = tl.arange(0, BK, layout=SliceLayout(0, ld))
    mm = tl.arange(0, BM, layout=SliceLayout(1, ld))
    acc = tl.full((BM, BN), 0, tl.float32, mma)
    for k0 in range(0, K, BK):
        kk = k0 + kk0
        w = load_packed(SM, EX, BASE, RI, RAW, nn[:, None]*K + kk[None, :],
                        (nn[:, None] < N) & (kk[None, :] < K))
        a = tl.load(x + mm[:, None]*sxm + kk[None, :],
                    mask=(mm[:, None] < M) & (kk[None, :] < K), other=0.)
        a = tl.convert_layout(a, DotOperandLayout(0, mma, 2))
        b = tl.convert_layout(tl.permute(w, (1, 0)), DotOperandLayout(1, mma, 2))
        acc = mma_v2(a, b, acc)
    outlayout: tl.constexpr = BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    acc = tl.convert_layout(acc, outlayout)
    om = tl.arange(0, BM, layout=SliceLayout(1, outlayout))
    on = pid*BN + tl.arange(0, BN, layout=SliceLayout(0, outlayout))
    tl.store(y + om[:, None]*sym + on[None, :], acc.to(y.dtype.element_ty),
             mask=(om[:, None] < M) & (on[None, :] < N))


@torch.no_grad()
def _pack(w: torch.Tensor):
    """b7a45bfa's pack_bf16 on one contiguous [N, K] BF16 matrix; None if any
    decoded bit differs (the caller then keeps F.linear)."""
    n = w.numel()
    nb = n // 128
    base = torch.empty(nb, dtype=torch.uint8, device=w.device)
    escape = torch.empty(nb, dtype=torch.int32, device=w.device)
    _pack_stats[(triton.cdiv(nb, 8),)](w.view(torch.uint16), base, escape, nb, 8, num_warps=4)
    prefix = torch.cumsum(escape, 0, dtype=torch.int32)
    count = int(prefix[-1].item())
    ri = torch.where(escape != 0, prefix - 1, -1)
    del escape, prefix
    sm = torch.empty(n, dtype=torch.uint8, device=w.device)
    ex = torch.empty(n // 2, dtype=torch.uint8, device=w.device)
    raw = torch.empty(max(1, count) * 128, dtype=torch.uint16, device=w.device)
    flat = w.view(torch.int16).reshape(-1)
    for first in range(0, nb, 16384):
        last = min(first + 16384, nb)
        bits = flat[first * 128:last * 128].to(torch.int32) & 65535
        exp = (bits >> 7) & 255
        b = base[first:last].to(torch.int32).repeat_interleave(128)
        code = torch.where(exp == 0, 0, exp - b + 1) & 15
        sm[first * 128:last * 128].copy_(((bits >> 8) & 128) | (bits & 127))
        ex[first * 64:last * 64].copy_(code[::2] | (code[1::2] << 4))
        select = torch.nonzero(ri[first:last] >= 0).reshape(-1)
        raw.view(torch.int16).reshape(-1, 128).index_copy_(
            0, ri[first:last].index_select(0, select).to(torch.int64),
            flat[first * 128:last * 128].reshape(-1, 128).index_select(0, select))
    del bits, exp, b, code, select
    planes = (sm, ex, base, ri, raw)
    counter = torch.zeros(1, dtype=torch.int32, device=w.device)
    verify_decoded[(triton.cdiv(n, 4096),)](
        *planes, w.view(torch.uint16), counter, n, 4096, num_warps=4)
    if int(counter.item()):
        return None
    return planes, count


def _small(x: torch.Tensor, planes, n: int, k: int) -> torch.Tensor:
    if x.stride(1) != 1:
        x = x.contiguous()
    y = x.new_empty((x.shape[0], n))
    packed_small[(triton.cdiv(n, 64),)](
        x, *planes, y, x.shape[0], n, k, x.stride(0), y.stride(0),
        num_warps=4, num_stages=4)
    return y


def _probe_inputs(m: int, k: int, g: torch.Generator, device):
    """Synthetic activations (fixed seed): normal, heavy, small, and raw BF16
    bit patterns with exponents 100..140 (random signs and mantissas; products
    stay normal fp32, so flush-to-zero choices cannot fail the guard)."""
    for scale in (1.0, 30.0, 1e-3, 1.0, 1.0):
        yield (torch.randn((m, k), generator=g, device=device, dtype=torch.float32)
               * scale).to(torch.bfloat16)
    r = torch.randint(0, 1 << 30, (m, k), generator=g, device=device, dtype=torch.int64)
    bits = ((r & 1) << 15) | ((100 + (r >> 1) % 41) << 7) | ((r >> 9) & 127)
    yield bits.to(torch.int32).to(torch.int16).view(torch.bfloat16)


@torch.no_grad()
def _matches_linear(planes, w: torch.Tensor) -> bool:
    n, k = w.shape
    g = torch.Generator(device=w.device)
    g.manual_seed(20261025)
    for m in range(MIN_ROWS, MAX_ROWS + 1):
        for x in _probe_inputs(m, k, g, w.device):
            if not torch.equal(_small(x, planes, n, k).view(torch.int16),
                               F.linear(x, w).view(torch.int16)):
                return False
    return True


_PACKED: list[torch.nn.Module] = []
_OFF = {"why": ""}


def _disable(why: str) -> None:
    _OFF["why"] = why
    for layer in _PACKED:
        for suffix in _SUFFIXES:
            layer._buffers.pop("_apk12_" + suffix, None)
    _PACKED.clear()
    print(f"ATTN_PACK12 off: {why}; every attention projection keeps F.linear", flush=True)


@torch.no_grad()
def maybe_pack(layer: torch.nn.Module) -> None:
    """Called from UnquantizedLinearMethod.process_weights_after_loading, i.e.
    after the original weights are loaded and before compile and capture."""
    if not ENABLED or _OFF["why"]:
        return
    prefix = getattr(layer, "prefix", "") or ""
    w = getattr(layer, "weight", None)
    if (w is None or "mtp" in prefix or not prefix.endswith((".qkv_proj", ".o_proj"))
            or tuple(w.shape) not in SHAPES or w.dtype != torch.bfloat16
            or not w.is_cuda or not w.is_contiguous()
            or getattr(layer, "bias", None) is not None):
        return
    try:
        packed = _pack(w.data)
        if packed is None:
            _disable(f"{prefix}: decoded bits differ")
            return
        planes, escapes = packed
        if not _matches_linear(planes, w.data):
            _disable(f"{prefix}: packed MMA differs from F.linear")
            return
    except Exception as e:  # JIT, CUDA or memory failure: keep F.linear
        _disable(f"{prefix}: {type(e).__name__}: {e}")
        return
    for suffix, t in zip(_SUFFIXES, planes):
        layer.register_buffer("_apk12_" + suffix, t, persistent=False)
    _PACKED.append(layer)
    nbytes = sum(t.numel() * t.element_size() for t in planes)
    print(f"ATTN_PACK12 {prefix} shape={tuple(w.shape)} bf16_bytes={w.numel() * 2} "
          f"packed_bytes={nbytes} escapes={escapes} rows={MIN_ROWS}..{MAX_ROWS} bitwise=F.linear",
          flush=True)


def _pk12_linear_impl(
    x: torch.Tensor, sm: torch.Tensor, ex: torch.Tensor, base: torch.Tensor,
    ri: torch.Tensor, raw: torch.Tensor, weight: torch.Tensor,
) -> torch.Tensor:
    n, k = weight.shape
    if (x.dim() == 2 and MIN_ROWS <= x.shape[0] <= MAX_ROWS and x.shape[1] == k
            and x.dtype == torch.bfloat16 and x.is_cuda):
        return _small(x, (sm, ex, base, ri, raw), n, k)
    return F.linear(x, weight)


def _pk12_linear_fake(
    x: torch.Tensor, sm: torch.Tensor, ex: torch.Tensor, base: torch.Tensor,
    ri: torch.Tensor, raw: torch.Tensor, weight: torch.Tensor,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="attn_pk12_linear", op_func=_pk12_linear_impl, fake_impl=_pk12_linear_fake,
)

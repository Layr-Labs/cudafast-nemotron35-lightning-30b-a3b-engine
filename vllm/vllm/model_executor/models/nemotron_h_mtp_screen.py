# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy draft-token head for the Nemotron-H MTP drafter: cascaded partial-K
screens plus a full-K rescore of the survivors.

DRAFTER ONLY. The target's lm_head, logits and verification are untouched;
the drafter's lm_head is the target's shared NVFP4 head (131072 x 2688, about
198 MB), and the drafter reads all of it three times per depth-3 cycle only to
take one argmax per draft. This replaces that argmax:

1. screen every vocabulary row on the 16 most important whole NVFP4
   16-element groups, chosen once from weight-only importance, in fp32, and
   keep the top PRE_PER rows of every SCREEN_CHUNK-row chunk;
2. add the next 16 groups for those rows (their score on the 32 most
   important groups) and keep the top CASCADE_PER rows of every chunk;
3. rescore those candidates on the original 64 importance groups and keep
   the top SCREEN_PER candidates of each original chunk;
4. rescore those rows on the full K in fp32, round to bf16 as the stock head
   does, and take the maximum with the lowest token id among equals (the rule
   torch.argmax applies to the stock bf16 logits).

Importance is sum over k of final_norm_gamma[k]^2 * ||W[:, k]||^2. Both group
sets are selected by importance rank before being sorted back into K order.
The screen and the full copy are built once from the pinned checkpoint's own
lm_head tensors (raw NVFP4 bytes, fp8 group scales, fp32 global scale): an
input-independent load-time repack, no format change, no new weights. The
proposal can differ when either screen removes a candidate or the rescore's
different fp32 accumulation changes rounded logits. The target's verification decides
every committed token, and the timed window is still checked against its oracle.

VLLM_MTP_SCREEN_PRE=0 screens every row on the 32 groups (steps 1-2 in one).
VLLM_MTP_SCREEN_CASCADE=0 restores the original 64-group single screen.
VLLM_MTP_SCREEN=0 restores the stock head.
"""

import json
import os

import torch

# l2pf (harness/kernels/l2pf); absent, nothing changes.
try:
    import l2pf as _l2pf
except ImportError:
    _l2pf = None

from vllm.logger import init_logger

# LOCAL PATCH (cudafast-nemotron35): the three-level cascade's per-chunk top-k selections as
# one CUDA kernel each (harness/kernels/sctopk), used only after a load-time comparison with
# _chunk_topk; the same ids in the same order. VLLM_MTP_SCREEN_SCTOPK=0 keeps _chunk_topk.
try:
    import sctopk as _sctopk
except ImportError:
    _sctopk = None
_SCTOPK = os.environ.get("VLLM_MTP_SCREEN_SCTOPK", "1") != "0"
# LOCAL PATCH (cudafast-nemotron35): the three-level cascade after its first streaming dot
# (the three per-chunk selections, the mid and rest dots and the survivors' full-K dot) as one
# CUDA kernel, one CTA per chunk (harness/kernels/scrtail), used only after a load-time bitwise
# comparison with the cascade below. MTP_SCREEN_TAIL=0 keeps the cascade.
try:
    import scrtail as _scrtail
except ImportError:
    _scrtail = None
from vllm.triton_utils import tl, triton

from .nemotron_h_mtp_kernels import (
    DEPTH_CUMULATIVE, DEPTH_THETA, _gate_args, depth_gate_init, mtp_pdl,
)

logger = init_logger(__name__)

SCREEN_FRAC_GROUPS = 64  # of 168 groups: 1024 of 2688 hidden dims (512 bytes per row)
SCREEN_CHUNK = 2048
SCREEN_PER = 32
CASCADE_FRAC_GROUPS = 32
CASCADE_PER = 256
# First stage of the three-level cascade: every row on its 16 most important
# groups, keeping the top PRE_PER of each chunk for the 32-group stage.
PRE_FRAC_GROUPS = 16
PRE_PER = 1024
# LOCAL PATCH (drafter3): shares of the next-pass prefetch queued after each of the
# cascade's three streaming dots and after its final dot (one per top-k window).
NEXT_PF_SHARES = (0.3, 0.35, 0.2, 0.15)
# LOCAL PATCH (drafter3): the screen's fp8 group scales are read once per pass like its
# weights: load them evict-first too, so they do not displace the evict-last eh hidden
# half or prefetched lines (MTP_SCREEN_SCALE_EF=0 keeps default loads).
SCREEN_SCALE_EF = os.environ.get("MTP_SCREEN_SCALE_EF", "1") != "0"


def _radix_for(radix, n: int):
    """LOCAL PATCH (drafter3): 8-bit digits (radix 2) only where they measured faster,
    selections of at most 1024 keys; 2048-key chunks keep 4-bit digits."""
    return 1 if radix == 2 and n > 1024 else radix


def _split_ranges(ranges, shares):
    """Cut [(tensor, bytes)] (each range a prefix of its tensor) into consecutive pieces
    of about shares[i] of the total bytes, as [(tensor_view, bytes)] lists; every piece
    starts 16-byte aligned. Proportional within each range, so every piece covers the
    same leading fraction of every plane (the planes of the same rows)."""
    pieces = [[] for _ in shares]
    for t, nbytes in ranges:
        flat = t.reshape(-1).view(torch.uint8)
        nbytes = min(int(nbytes), flat.numel())
        off = 0
        acc = 0.0
        for i, share in enumerate(shares):
            acc += share
            end = nbytes if i == len(shares) - 1 else min(nbytes, int(nbytes * acc) // 256 * 256)
            if end > off:
                pieces[i].append((flat[off:end], end - off))
            off = max(off, end)
    return pieces


def screen_enabled() -> bool:
    return os.environ.get("VLLM_MTP_SCREEN", "1") != "0"


def screen_graph_enabled() -> bool:
    """Kill switch for the per-draft-row capture: VLLM_MTP_SCREEN_GRAPH=0."""
    return os.environ.get("VLLM_MTP_SCREEN_GRAPH", "1") != "0"


@triton.jit
def _e2m1(nib):
    # fp4 e2m1: s ee m -> +-{0, .5, 1, 1.5, 2, 3, 4, 6}
    e = (nib >> 1) & 3
    m = (nib & 1).to(tl.float32)
    p2 = tl.where(e == 1, 1.0, tl.where(e == 2, 2.0, 4.0))
    mag = tl.where(e == 0, m * 0.5, (1.0 + m * 0.5) * p2)
    return tl.where((nib & 8) != 0, -mag, mag)


@triton.jit
def _e2m1_pair(byte):
    # Exact hardware decode of both nibbles; FP4 values all fit in FP16/FP32.
    lo, hi = tl.inline_asm_elementwise(
        """{
        .reg .b8 a, b, c, d;
        .reg .b32 pair;
        .reg .b16 lo, hi;
        mov.b32 {a, b, c, d}, $2;
        cvt.rn.f16x2.e2m1x2 pair, a;
        mov.b32 {lo, hi}, pair;
        cvt.f32.f16 $0, lo;
        cvt.f32.f16 $1, hi;
        }""",
        constraints="=f,=f,r", args=[byte],
        dtype=(tl.float32, tl.float32), is_pure=True, pack=1,
    )
    # The software decoder canonicalizes E2M1 -0 to +0 in FP32. Hardware
    # preserves -0; match the reference even for the two zero encodings.
    return tl.where(lo == 0.0, 0.0, lo), tl.where(hi == 0.0, 0.0, hi)


@triton.jit
def _nvfp4_rows_dot(
    h_ptr,  # [2 * NBYTES] fp32
    w_ptr,  # [N, NBYTES] uint8, element 2j in the low nibble of byte j
    s_ptr,  # [N, NBYTES // 8] fp8 e4m3 group scales
    rows_ptr,  # [R] int64 row ids, or unused when GATHER is False
    out_ptr,  # [R] fp32
    R,
    gscale,
    NBYTES: tl.constexpr,
    GATHER: tl.constexpr,
    BR: tl.constexpr,
    BB: tl.constexpr,
    INDEX32: tl.constexpr = False,
    gate_ptr=None,
    GATED: tl.constexpr = False,
    launch_pdl: tl.constexpr = False,
    hcols_ptr=None,  # [2 * NBYTES] int64 columns of h to read, when HCOLS
    HCOLS: tl.constexpr = False,
    ref_ptr=None,  # [R] int64 second-stage top-k indices, when RMODE == 2
    RMODE: tl.constexpr = 0,
    CHUNK: tl.constexpr = 1,
    CPER: tl.constexpr = 1,
    SPER: tl.constexpr = 1,
    base_ptr=None,  # [N] fp32 per-row scores already computed over other groups, when BASE
    BASE: tl.constexpr = False,
    scatter_ptr=None,  # [N] fp32: the result is also stored at its row, when SCATTER
    SCATTER: tl.constexpr = False,
    SCALE_EF: tl.constexpr = False,
):
    """RMODE 0: rows from GATHER as before. 1: rows = rows_ptr[r] + (r // CPER) * CHUNK
    (first-stage top-k indices within their chunk). 2: rows = rows_ptr[c * CPER + ref_ptr[r]]
    + c * CHUNK with c = r // SPER (the second stage's picks among them). HCOLS reads
    h[hcols_ptr[j]] where h[j] was read: the same elements a gathered copy holds."""
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    rmask = r < R
    if GATED:
        if tl.load(gate_ptr) != 0:
            # Stopped draft pass: the +0 sums its masked weight and scale loads
            # would give, without reading rows, weights or h.
            tl.store(out_ptr + r, tl.zeros((BR,), dtype=tl.float32) * gscale, mask=rmask)
            return
    wr = rmask
    if RMODE == 1:
        rows = tl.load(rows_ptr + r, mask=rmask, other=0) + (r // CPER) * CHUNK
        if INDEX32:
            rows = rows.to(tl.int32)
        else:
            rows = rows.to(tl.int64)
    elif RMODE == 2:
        c = r // SPER
        sel = tl.load(ref_ptr + r, mask=rmask, other=0)
        rows = tl.load(rows_ptr + c * CPER + sel, mask=rmask, other=0) + c * CHUNK
        if INDEX32:
            rows = rows.to(tl.int32)
        else:
            rows = rows.to(tl.int64)
    elif GATHER:
        if INDEX32:
            rows = tl.load(rows_ptr + r, mask=rmask, other=0).to(tl.int32)
        else:
            rows = tl.load(rows_ptr + r, mask=rmask, other=0).to(tl.int64)
    else:
        if INDEX32:
            rows = r.to(tl.int32)
        else:
            rows = r.to(tl.int64)
    acc = tl.zeros((BR,), dtype=tl.float32)
    for b0 in range(0, NBYTES, BB):
        ob = b0 + tl.arange(0, BB)
        wb = tl.load(
            w_ptr + rows[:, None] * NBYTES + ob[None, :],
            mask=wr[:, None],
            other=0,
            eviction_policy="evict_first",
        ).to(tl.int32)
        og = (b0 // 8) + tl.arange(0, BB // 8)
        sc = tl.load(
            s_ptr + rows[:, None] * (NBYTES // 8) + og[None, :],
            mask=wr[:, None],
            other=0.0,
            eviction_policy="evict_first" if SCALE_EF else "",
        ).to(tl.float32)
        if HCOLS:
            hl = tl.load(h_ptr + tl.load(hcols_ptr + 2 * ob))
            hh = tl.load(h_ptr + tl.load(hcols_ptr + 2 * ob + 1))
        else:
            hl = tl.load(h_ptr + 2 * ob)
            hh = tl.load(h_ptr + 2 * ob + 1)
        lo, hi = _e2m1_pair(wb)
        part = lo * hl[None, :] + hi * hh[None, :]
        part = tl.reshape(part, (BR, BB // 8, 8))
        acc += tl.sum(tl.sum(part, axis=2) * sc, axis=1)
    if BASE:
        # LOCAL PATCH (cudafast-nemotron35): base[row] (the coarse stage's score over
        # its own groups) plus this kernel's dot over the remaining groups.
        b = tl.load(base_ptr + rows, mask=wr, other=0.0)
        tl.store(out_ptr + r, b + acc * gscale, mask=rmask)
        if SCATTER:
            tl.store(scatter_ptr + rows, b + acc * gscale, mask=rmask)
    else:
        tl.store(out_ptr + r, acc * gscale, mask=rmask)


@triton.jit
def _nvfp4_rows_dot_part(
    h_ptr, w_ptr, s_ptr, rows_ptr, part_ptr, R,
    NBYTES: tl.constexpr, BR: tl.constexpr, BB: tl.constexpr, INDEX32: tl.constexpr,
    gate_ptr=None, GATED: tl.constexpr = False, launch_pdl: tl.constexpr = False,
    SCALE_EF: tl.constexpr = False,
):
    """LOCAL PATCH (cudafast-nemotron35): one BB-byte block (program axis 1) of
    _nvfp4_rows_dot's gathered rows: the exact term that kernel's loop adds to its
    accumulator in that iteration, stored at part[kb, r]. _nvfp4_rows_dot_sum adds
    the terms in the loop's order, so the result is the same bits with BB-block
    parallelism instead of a serial BB loop per row block."""
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if GATED:
        if tl.load(gate_ptr) != 0:
            return
    pid = tl.program_id(0)
    kb = tl.program_id(1)
    r = pid * BR + tl.arange(0, BR)
    rmask = r < R
    if INDEX32:
        rows = tl.load(rows_ptr + r, mask=rmask, other=0).to(tl.int32)
    else:
        rows = tl.load(rows_ptr + r, mask=rmask, other=0).to(tl.int64)
    wr = rmask
    b0 = kb * BB
    ob = b0 + tl.arange(0, BB)
    wb = tl.load(
        w_ptr + rows[:, None] * NBYTES + ob[None, :],
        mask=wr[:, None],
        other=0,
        eviction_policy="evict_first",
    ).to(tl.int32)
    og = (b0 // 8) + tl.arange(0, BB // 8)
    sc = tl.load(
        s_ptr + rows[:, None] * (NBYTES // 8) + og[None, :],
        mask=wr[:, None],
        other=0.0,
        eviction_policy="evict_first" if SCALE_EF else "",
    ).to(tl.float32)
    hl = tl.load(h_ptr + 2 * ob)
    hh = tl.load(h_ptr + 2 * ob + 1)
    lo, hi = _e2m1_pair(wb)
    part = lo * hl[None, :] + hi * hh[None, :]
    part = tl.reshape(part, (BR, BB // 8, 8))
    acc = tl.zeros((BR,), dtype=tl.float32)
    acc += tl.sum(tl.sum(part, axis=2) * sc, axis=1)
    tl.store(part_ptr + kb * R + r, acc, mask=rmask)


@triton.jit
def _nvfp4_rows_dot_sum(
    part_ptr, out_ptr, R, gscale, NK: tl.constexpr, BR: tl.constexpr,
    gate_ptr=None, GATED: tl.constexpr = False, launch_pdl: tl.constexpr = False,
):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    rmask = r < R
    if GATED:
        if tl.load(gate_ptr) != 0:
            tl.store(out_ptr + r, tl.zeros((BR,), dtype=tl.float32) * gscale, mask=rmask)
            return
    acc = tl.zeros((BR,), dtype=tl.float32)
    for k in range(NK):
        acc += tl.load(part_ptr + k * R + r, mask=rmask, other=0.0)
    tl.store(out_ptr + r, acc * gscale, mask=rmask)


def screen_ksplit_enabled() -> bool:
    """The final full-row dot as BB-block partial terms + an in-order sum (same
    bits): VLLM_MTP_SCREEN_KSPLIT=0 keeps the single serial-loop kernel."""
    return os.environ.get("VLLM_MTP_SCREEN_KSPLIT", "1") != "0"


def _rows_dot_ksplit(h, w, s, rows, gscale, out, part, gated=False):
    n_out = out.numel()
    nbytes = w.shape[1]
    br = 16
    nk = nbytes // 64
    max_rows = max(w.shape[0], n_out) + br - 1
    index32 = (
        max_rows * nbytes < 2**31
        and max_rows * (nbytes // 8) < 2**31
        and n_out + br - 1 < 2**31
    )
    gate, ok = _gate_args(out)
    gated = gated and ok
    _nvfp4_rows_dot_part[(triton.cdiv(n_out, br), nk)](
        h, w, s, rows, part, n_out,
        NBYTES=nbytes, BR=br, BB=64, INDEX32=index32,
        gate_ptr=gate, GATED=gated, launch_pdl=mtp_pdl(),
        num_warps=4, num_stages=2, SCALE_EF=SCREEN_SCALE_EF,
    )
    _nvfp4_rows_dot_sum[(triton.cdiv(n_out, 128),)](
        part, out, n_out, gscale, NK=nk, BR=128,
        gate_ptr=gate, GATED=gated, launch_pdl=mtp_pdl(), num_warps=4,
    )


def _rows_dot(h, w, s, rows, gscale, out, gather, gated=False, hcols=None, ref=None,
              rmode=0, cper=1, sper=1, base=None, scatter=None):
    n_out = out.numel()
    nbytes = w.shape[1]
    gather = gather or rmode != 0
    br = 64 if not gather else 16
    max_rows = max(w.shape[0], n_out) + br - 1
    index32 = (
        max_rows * nbytes < 2**31
        and max_rows * (nbytes // 8) < 2**31
        and n_out + br - 1 < 2**31
    )
    gate, ok = _gate_args(out)
    gated = gated and ok
    _nvfp4_rows_dot[(triton.cdiv(n_out, br),)](
        h, w, s, rows if gather else out, out, n_out, gscale,
        NBYTES=nbytes, GATHER=gather, BR=br, BB=64, INDEX32=index32,
        gate_ptr=gate, GATED=gated,
        num_warps=4, num_stages=2,
        launch_pdl=mtp_pdl(),
        hcols_ptr=hcols, HCOLS=hcols is not None,
        ref_ptr=ref, RMODE=rmode, CHUNK=SCREEN_CHUNK, CPER=cper, SPER=sper,
        base_ptr=base, BASE=base is not None,
        scatter_ptr=scatter, SCATTER=scatter is not None,
        SCALE_EF=SCREEN_SCALE_EF,
    )


def screen_topk_enabled() -> bool:
    """Kill switch for the fused exact top-k: VLLM_MTP_SCREEN_TOPK=0."""
    return os.environ.get("VLLM_MTP_SCREEN_TOPK", "1") != "0"


def screen_reuse_enabled() -> bool:
    """Refine stage = coarse score + dot over the refine-only groups (the coarse
    groups are a subset of the refine groups). VLLM_MTP_SCREEN_REUSE=0 restores the
    full 64-group refine dot."""
    return os.environ.get("VLLM_MTP_SCREEN_REUSE", "1") != "0"


@triton.jit
def _topk_key(v):
    # torch's TopKTypeConfig<float>::convert, widened to int64: an order-preserving
    # unsigned key (-0 below +0) with every NaN mapped to the largest key.
    x = v.to(tl.int32, bitcast=True)
    xu = x.to(tl.int64) & 0xFFFFFFFF
    key = tl.where(x < 0, xu ^ 0xFFFFFFFF, xu ^ 0x80000000)
    return tl.where(v != v, 0xFFFFFFFF, key)


@triton.jit
def _chunk_topk(
    scores_ptr,  # [C, N] fp32
    src_ptr,  # [C, N] int64 ids (GATHER) or unused
    out_ptr,  # [C, K] int64
    N: tl.constexpr,
    K: tl.constexpr,
    GATHER: tl.constexpr,
    launch_pdl: tl.constexpr = False,
    gate_ptr=None,
    GATED: tl.constexpr = False,
    RADIX: tl.constexpr = False,
):
    """The indices torch.topk(scores.view(C, N), K, dim=-1, sorted=False) returns,
    in its order (CUDA sbtopk::gatherTopK: every key above the K-th largest in
    ascending index order, then the K-th largest's ties in ascending index order
    until K are taken), written as row-major ids (c * N + i) or, with GATHER, as
    src[c, i]: the id lists the screen built with topk + chunk offset / gather."""
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    c = tl.program_id(0)
    if GATED:
        if tl.load(gate_ptr) != 0:
            # Stopped draft pass: its scores are all +0 (the gated rows_dot), so
            # every key ties and the first K ids in ascending index order are taken.
            j = tl.arange(0, K)
            if GATHER:
                first = tl.load(src_ptr + c * N + j)
            else:
                first = c.to(tl.int64) * N + j
            tl.store(out_ptr + c * K + j, first)
            return
    i = tl.arange(0, N)
    key = _topk_key(tl.load(scores_ptr + c * N + i))
    if RADIX:
        # Exact 4-bit radix selection: eight histograms replace 32 serial
        # whole-block threshold reductions. Keys include NaNs and signed zeros
        # in the original torch order; survivor scattering below is unchanged.
        low = tl.min(key, axis=0)
        high = tl.max(key, axis=0)
        if low == high:
            # Stopped draft rows normally produce uniform scores. Finish exact
            # tie selection without any histogram, while still writing all K ids.
            t = low
        else:
            active = tl.full((N,), True, tl.int1)
            left = tl.full((), K, tl.int32)
            t = tl.full((), 0, tl.int64)
            if RADIX == 2:
                # LOCAL PATCH (drafter3): the same exact radix selection with 8-bit
                # digits: four histogram rounds instead of eight (the rounds, not
                # the elements, set the latency of these small selections).
                bins8 = tl.arange(0, 256)
                for shift in tl.static_range(24, -1, -8):
                    digit = ((key >> shift) & 255).to(tl.int32)
                    hist = tl.histogram(digit, 256, mask=active)
                    suffix = tl.sum(hist, axis=0) - tl.cumsum(hist, axis=0) + hist
                    pick = tl.max(tl.where(suffix >= left, bins8, 0), axis=0)
                    left -= tl.sum(tl.where(bins8 > pick, hist, 0), axis=0)
                    t |= pick.to(tl.int64) << shift
                    active = active & (digit == pick)
            else:
                bins = tl.arange(0, 16)
                for shift in tl.static_range(28, -1, -4):
                    digit = ((key >> shift) & 15).to(tl.int32)
                    hist = tl.histogram(digit, 16, mask=active)
                    suffix = tl.sum(hist, axis=0) - tl.cumsum(hist, axis=0) + hist
                    pick = tl.max(tl.where(suffix >= left, bins, 0), axis=0)
                    left -= tl.sum(tl.where(bins > pick, hist, 0), axis=0)
                    t |= pick.to(tl.int64) << shift
                    active = active & (digit == pick)
    else:
        # Original binary radix selection, retained for remote admission/fallback.
        t = tl.sum(key * 0, axis=0)
        for b in tl.static_range(31, -1, -1):
            trial = t | (1 << b)
            cnt = tl.sum((key >= trial).to(tl.int32), axis=0)
            t = tl.where(cnt >= K, trial, t)
    gt = (key > t).to(tl.int32)
    eq = (key == t).to(tl.int32)
    n_gt = tl.sum(gt, axis=0)
    gt_rank = tl.cumsum(gt, axis=0) - gt
    eq_rank = tl.cumsum(eq, axis=0) - eq
    take = (gt != 0) | ((eq != 0) & (eq_rank < K - n_gt))
    pos = tl.where(gt != 0, gt_rank, n_gt + eq_rank)
    if GATHER:
        val = tl.load(src_ptr + c * N + i)
    else:
        val = c.to(tl.int64) * N + i
    tl.store(out_ptr + c * K + pos, val, mask=take)


@triton.jit
def _gather_three(
    h_ptr, ia_ptr, oa_ptr, ib_ptr, ob_ptr, ic_ptr, oc_ptr,
    NA: tl.constexpr, NB: tl.constexpr, NC: tl.constexpr,
    launch_pdl: tl.constexpr = False,
):
    """oa = h[ia], ob = h[ib], oc = h[ic] in one launch; pure copies."""
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    a = tl.arange(0, NA)
    tl.store(oa_ptr + a, tl.load(h_ptr + tl.load(ia_ptr + a)))
    b = tl.arange(0, NB)
    tl.store(ob_ptr + b, tl.load(h_ptr + tl.load(ib_ptr + b)))
    c = tl.arange(0, NC)
    tl.store(oc_ptr + c, tl.load(h_ptr + tl.load(ic_ptr + c)))


@triton.jit
def _gather_two(
    h_ptr, ia_ptr, oa_ptr, ib_ptr, ob_ptr,
    NA: tl.constexpr, NB: tl.constexpr,
    launch_pdl: tl.constexpr = False,
):
    """oa = h[ia], ob = h[ib] (the two index_select gathers of the cascade) in one
    launch; pure copies."""
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    a = tl.arange(0, NA)
    tl.store(oa_ptr + a, tl.load(h_ptr + tl.load(ia_ptr + a)))
    b = tl.arange(0, NB)
    tl.store(ob_ptr + b, tl.load(h_ptr + tl.load(ib_ptr + b)))


@triton.jit
def _select_token(
    logits, ids, out, N: tl.constexpr, BLOCK: tl.constexpr,
    gate_ptr=None,
    GATED: tl.constexpr = False,
    THETA: tl.constexpr = 0.0,
    CUM: tl.constexpr = False,
    launch_pdl: tl.constexpr = False,
    ref=None,
    TWO_LEVEL: tl.constexpr = False,
    CHUNK: tl.constexpr = 1,
    CPER: tl.constexpr = 1,
    SPER: tl.constexpr = 1,
):
    """Stock BF16 maximum, then minimum token id among ties, in one launch.
    TWO_LEVEL: the token ids are ids[c * CPER + ref[i]] + c * CHUNK, c = i // SPER."""
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    i = tl.arange(0, BLOCK)
    # Round before reduction: FP32 logit differences can disappear in BF16.
    v = tl.load(logits + i, mask=i < N, other=-float("inf"))
    v = v.to(tl.bfloat16).to(tl.float32)
    if TWO_LEVEL:
        c = i // SPER
        sel = tl.load(ref + i, mask=i < N, other=0)
        token = tl.load(ids + c * CPER + sel, mask=i < N, other=0) + c * CHUNK
        token = tl.where(i < N, token, 1 << 30)
    else:
        token = tl.load(ids + i, mask=i < N, other=1 << 30)
    best = tl.max(v, axis=0)
    # torch.max propagates NaNs; its subsequent equality tests all fail.
    any_nan = tl.max(((i < N) & (v != v)).to(tl.int32), axis=0)
    chosen = tl.min(tl.where((i < N) & (v == best), token, 1 << 30), axis=0)
    tl.store(out, tl.where(any_nan != 0, 1 << 30, chosen))
    if GATED:
        # Depth gate (nemotron_h_mtp_kernels.py): while a draft loop is active and
        # not stopped, count this draft as kept, then stop the loop if its top-1
        if (tl.load(gate_ptr + 2) != 0) & (tl.load(gate_ptr) == 0):
            tl.store(gate_ptr + 1, tl.load(gate_ptr + 1) + 1)
            denom = tl.sum(tl.where(i < N, tl.exp(v - best), 0.0), axis=0)
            conf = 1.0 / denom
            if CUM:
                conf = tl.load(gate_ptr + 3).to(tl.float32, bitcast=True) * conf
                tl.store(gate_ptr + 3, conf.to(tl.int32, bitcast=True))
            if conf < THETA:
                tl.store(gate_ptr, 1)


class DraftHeadScreen:
    """Load-time buffers plus the greedy screen for one shared NVFP4 head."""

    def __init__(self, snapshot_dir: str, device: torch.device, depth: int | None = None):
        from safetensors import safe_open

        idx = json.load(open(os.path.join(snapshot_dir, "model.safetensors.index.json")))
        wmap = idx["weight_map"]

        def get(name):
            with safe_open(os.path.join(snapshot_dir, wmap[name]), "pt", device="cpu") as f:
                return f.get_tensor(name)

        w = get("lm_head.weight")  # [N, K/2] uint8
        s = get("lm_head.weight_scale")  # [N, K/16] fp8 e4m3
        g = float(get("lm_head.weight_scale_2").float().reshape(-1)[0])
        gamma = None
        for name in wmap:
            if name.startswith("mtp.") and name.endswith("final_layernorm.weight"):
                gamma = get(name).float()
        n, kb = w.shape
        k = kb * 2
        ng = k // 16
        assert w.dtype == torch.uint8 and s.shape == (n, ng), (w.shape, s.shape)
        assert n % SCREEN_CHUNK == 0
        self.cascade = os.environ.get("VLLM_MTP_SCREEN_CASCADE", "1") != "0"
        # VLLM_MTP_SCREEN_FOLD=0 keeps the cascade's separate gather kernels.
        self.fold = os.environ.get("VLLM_MTP_SCREEN_FOLD", "1") != "0"
        if self.cascade:
            assert 0 < CASCADE_FRAC_GROUPS < SCREEN_FRAC_GROUPS <= ng
            assert SCREEN_PER <= CASCADE_PER <= SCREEN_CHUNK
        # Weight-only importance of each 16-element group.
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
        )
        colsq = torch.zeros(k, dtype=torch.float64)
        for r0 in range(0, n, 16384):
            wb = w[r0:r0 + 16384].long()
            vals = torch.stack([lut[wb & 15], lut[wb >> 4]], -1).reshape(wb.shape[0], k)
            vals = vals * s[r0:r0 + 16384].float().repeat_interleave(16, 1)
            colsq += (vals.double() ** 2).sum(0)
        imp = colsq if gamma is None else colsq * gamma.double() ** 2
        ranked_groups = imp.view(ng, 16).sum(1).argsort(descending=True)
        groups = ranked_groups[:SCREEN_FRAC_GROUPS].sort().values
        cols = (groups[:, None] * 16 + torch.arange(16)[None, :]).reshape(-1)
        self.gscale = g
        self.n = n
        self.w_full = w.to(device).contiguous()
        self.s_full = s.to(device).contiguous()
        self.w_scr = w.view(n, ng, 8)[:, groups, :].reshape(n, -1).to(device).contiguous()
        self.s_scr = s[:, groups].to(device).contiguous()
        self.cols = cols.to(device)
        if self.cascade:
            # Select by importance rank, not the first half of K-sorted groups.
            coarse_groups = ranked_groups[:CASCADE_FRAC_GROUPS].sort().values
            coarse_cols = (
                coarse_groups[:, None] * 16 + torch.arange(16)[None, :]
            ).reshape(-1)
            self.w_coarse = (
                w.view(n, ng, 8)[:, coarse_groups, :]
                .reshape(n, -1).to(device).contiguous()
            )
            self.s_coarse = s[:, coarse_groups].to(device).contiguous()
            self.coarse_cols = coarse_cols.to(device)
            self.refine_scores = torch.empty(
                (n // SCREEN_CHUNK) * CASCADE_PER,
                dtype=torch.float32, device=device,
            )

        def _pow2(v: int) -> bool:
            return v > 0 and (v & (v - 1)) == 0

        self.fused_topk = (
            self.cascade
            and device.type == "cuda"
            and screen_topk_enabled()
            and all(
                _pow2(v)
                for v in (SCREEN_CHUNK, CASCADE_PER, SCREEN_PER,
                          self.coarse_cols.numel(), cols.numel())
            )
        )
        if self.fused_topk:
            self._coarse_cand = torch.empty(
                (n // SCREEN_CHUNK) * CASCADE_PER, dtype=torch.int64, device=device
            )
            self._cand = torch.empty(
                (n // SCREEN_CHUNK) * SCREEN_PER, dtype=torch.int64, device=device
            )
            self._hc = torch.empty(
                self.coarse_cols.numel(), dtype=torch.float32, device=device
            )
            self._hs = torch.empty(cols.numel(), dtype=torch.float32, device=device)
        # LOCAL PATCH (cudafast-nemotron35): the refine groups minus the coarse groups.
        self.reuse = self.fused_topk and screen_reuse_enabled()
        if self.reuse:
            rest_groups = ranked_groups[CASCADE_FRAC_GROUPS:SCREEN_FRAC_GROUPS].sort().values
            rest_cols = (rest_groups[:, None] * 16 + torch.arange(16)[None, :]).reshape(-1)
            self.reuse = _pow2(rest_cols.numel())
        if self.reuse:
            self.w_rest = (
                w.view(n, ng, 8)[:, rest_groups, :].reshape(n, -1).to(device).contiguous()
            )
            self.s_rest = s[:, rest_groups].to(device).contiguous()
            self.rest_cols = rest_cols.to(device)
            self._hr = torch.empty(rest_cols.numel(), dtype=torch.float32, device=device)
        # LOCAL PATCH (cudafast-nemotron35): three-level cascade. Every row is
        # screened on the 16 most important groups only; the top PRE_PER of each
        # chunk add the next 16 groups (the coarse 32-group score, stored back at
        # its row for the refine stage), and the top CASCADE_PER of those go on to
        # the refine stage as before. VLLM_MTP_SCREEN_PRE=0 keeps two levels.
        self.pre = (
            self.reuse
            and os.environ.get("VLLM_MTP_SCREEN_PRE", "1") != "0"
            and 0 < PRE_FRAC_GROUPS < CASCADE_FRAC_GROUPS
            and CASCADE_PER <= PRE_PER <= SCREEN_CHUNK
            and _pow2(PRE_PER)
            and _pow2(PRE_FRAC_GROUPS * 16)
            and _pow2((CASCADE_FRAC_GROUPS - PRE_FRAC_GROUPS) * 16)
        )
        if self.pre:
            pre_groups = ranked_groups[:PRE_FRAC_GROUPS].sort().values
            mid_groups = ranked_groups[PRE_FRAC_GROUPS:CASCADE_FRAC_GROUPS].sort().values
            pre_cols = (pre_groups[:, None] * 16 + torch.arange(16)[None, :]).reshape(-1)
            mid_cols = (mid_groups[:, None] * 16 + torch.arange(16)[None, :]).reshape(-1)
            self.w_pre = (
                w.view(n, ng, 8)[:, pre_groups, :].reshape(n, -1).to(device).contiguous()
            )
            self.s_pre = s[:, pre_groups].to(device).contiguous()
            self.w_mid = (
                w.view(n, ng, 8)[:, mid_groups, :].reshape(n, -1).to(device).contiguous()
            )
            self.s_mid = s[:, mid_groups].to(device).contiguous()
            self.pre_cols = pre_cols.to(device)
            self.mid_cols = mid_cols.to(device)
            self._hp = torch.empty(pre_cols.numel(), dtype=torch.float32, device=device)
            self._hm = torch.empty(mid_cols.numel(), dtype=torch.float32, device=device)
            self._pre_cand = torch.empty(
                (n // SCREEN_CHUNK) * PRE_PER, dtype=torch.int64, device=device
            )
            self.mid_scores = torch.empty(
                (n // SCREEN_CHUNK) * PRE_PER, dtype=torch.float32, device=device
            )
        self.radix = self.fused_topk and os.environ.get("VLLM_MTP_RADIX_TOPK", "1") != "0"
        if self.radix:
            self.radix = self._radix_matches(device)
        # LOCAL PATCH (drafter3): 8-bit digits, admitted by the same load-time comparison
        # against the binary selection (MTP_SCREEN_RADIX8=0 keeps 4-bit digits).
        if self.radix and os.environ.get("MTP_SCREEN_RADIX8", "1") != "0":
            if self._radix_matches(device, mode=2):
                self.radix = 2
        self.sctopk = (
            self.radix and self.pre and _SCTOPK and _sctopk is not None
            and device.type == "cuda" and _sctopk.available()
        )
        if self.sctopk:
            self.sctopk = self._sctopk_matches(device)
        self.chunk_off = (
            torch.arange(n // SCREEN_CHUNK, device=device) * SCREEN_CHUNK
        )[:, None]
        self.scores = torch.empty(n, dtype=torch.float32, device=device)
        self.cand_logits = torch.empty(
            (n // SCREEN_CHUNK) * SCREEN_PER, dtype=torch.float32, device=device
        )
        # LOCAL PATCH (cudafast-nemotron35): partial terms of the final full-row dot.
        self._full_part = None
        if (screen_ksplit_enabled() and device.type == "cuda"
                and self.w_full.shape[1] % 64 == 0):
            self._full_part = torch.empty(
                (self.w_full.shape[1] // 64) * self.cand_logits.numel(),
                dtype=torch.float32, device=device,
            )
        self.big = torch.tensor(1 << 30, dtype=torch.int32, device=device)
        # l2pf: the coarse matrix is registered for the MTP MoE's down
        # projection window (nemotron_h_mtp_kernels.py), which prefetches its head.
        if _l2pf is not None and self.cascade:
            _l2pf.DRAFT_SCREEN[:] = [self.w_pre if self.pre else self.w_coarse]
        self.graph = None
        # LOCAL PATCH (drafter3): next-pass L2 prefetch (harness/kernels/l2pf DRAFT_NEXT):
        # the head of what the next draft pass reads first, cut into one piece per
        # top-k window of the cascade (see _greedy_row).
        self._next_pf = []
        self._next_max_kept = (1 << 30) if not depth else max(1, int(depth) - 1)
        if (_l2pf is not None and _l2pf.ENABLED and device.type == "cuda"
                and getattr(_l2pf, "DRAFT_NEXT_BYTES", 0) > 0
                and hasattr(_l2pf, "GatedTable")):
            left = _l2pf.DRAFT_NEXT_BYTES
            names = ("eh", "qkv") if os.environ.get("L2PF_DRAFT_NEXT_EH", "0") == "1" else ("qkv",)
            ranges = []
            for name in names:
                planes = _l2pf.DRAFT_NEXT.get(name)
                if planes is None or left <= 0:
                    continue
                part = _l2pf.packed_ranges(planes, left)
                ranges += part
                left -= sum(int(b) for _, b in part)
            for piece in _split_ranges(ranges, NEXT_PF_SHARES):
                table = _l2pf.GatedTable(piece, device, name="next")
                if table.nranges:
                    self._next_pf.append(table)
            if self._next_pf:
                logger.info("MTP draft screen: next-pass L2 prefetch %s MB in %d pieces, "
                            "max_kept=%d", [round(t.nbytes / 2**20, 2) for t in self._next_pf],
                            len(self._next_pf), self._next_max_kept)
        self._h = torch.empty(k, dtype=torch.float32, device=device)
        self._out = torch.empty(1, dtype=torch.int32, device=device)
        self.tail = False
        self._tail_debug = False
        if device.type == "cuda" and self.cascade and self.fold:
            self.fold = self._fold_matches(k, device)
        if (device.type == "cuda" and self.pre and self.sctopk and not self._next_pf
                and _scrtail is not None and _scrtail.available()):
            self.tail = self._tail_matches(k, device)
            print(f"MTP_SCREEN_TAIL admitted={self.tail}", flush=True)
        if device.type == "cuda" and self.tail:
            from .nemotron_h_mtp_gather_first import prepare as prepare_gather_first

            prepare_gather_first(self)
        if device.type == "cuda" and screen_graph_enabled():
            # The captured one-row screen is the gated one (depth gate): allocate
            # the gate first so the graph captures its address.
            gate = depth_gate_init(device)
            try:
                self._h.zero_()
                for _ in range(2):
                    self._greedy_row(self._h, gated=True, out=self._out)
                torch.cuda.synchronize(device)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    self.graph, pool=torch.cuda.graph_pool_handle()
                ):
                    self._greedy_row(self._h, gated=True, out=self._out)
                torch.cuda.synchronize(device)
            except Exception as exc:
                self.graph = None
                logger.warning("MTP draft-head screen graph disabled: %s", exc)
            if gate is not None:
                gate.zero_()  # the warm-up rows above may have counted or stopped
        logger.info(
            "MTP draft-head screen: %d of %d groups, %d candidates, cascade=%s, "
            "fused_topk=%s, graph=%s",
            SCREEN_FRAC_GROUPS, ng, self.cand_logits.numel(), self.cascade,
            self.fused_topk, self.graph is not None,
        )

    def _radix_matches(self, device: torch.device, mode: int = 1) -> bool:
        """Official remote model-load admission, no request data or local execution."""
        try:
            pairs = [(SCREEN_CHUNK, CASCADE_PER, False), (CASCADE_PER, SCREEN_PER, True)]
            if getattr(self, "pre", False):
                pairs += [(SCREEN_CHUNK, PRE_PER, False), (PRE_PER, CASCADE_PER, True)]
            for n, k, gather in pairs:
                index = torch.arange(2 * n, device=device, dtype=torch.int32)
                src = (torch.arange(2 * n, device=device, dtype=torch.int64) + 10000).flip(0)
                for case in range(4):
                    if case == 0:
                        scores = (index * 1103515245 + 12345).view(torch.float32)
                    elif case == 1:
                        bits = torch.where(index % 2 == 0, 0, -2147483648).to(torch.int32)
                        scores = bits.view(torch.float32)
                    elif case == 2:
                        scores = (index % 19).float() - 9
                        scores[:n // 16] = float("nan")
                        scores[n // 16:n // 8] = float("inf")
                        scores[n:n + n // 8] = -float("inf")
                    else:
                        scores = torch.full((2 * n,), 42.0, device=device)
                    a = torch.empty(2 * k, dtype=torch.int64, device=device)
                    b = torch.empty_like(a)
                    for radix, out in ((False, a), (mode, b)):
                        _chunk_topk[(2,)](scores, src, out, N=n, K=k, GATHER=gather,
                                          RADIX=radix, num_warps=4 if radix or n != SCREEN_CHUNK else 8,
                                          launch_pdl=mtp_pdl())
                    if not torch.equal(a, b):
                        logger.warning("MTP radix top-k differs; original selection retained")
                        return False
            return True
        except Exception as exc:
            logger.warning("MTP radix top-k unavailable (%r); original selection retained", exc)
            return False

    def _sctopk_matches(self, device: torch.device) -> bool:
        """sctopk against _chunk_topk (RADIX) on the cascade's three shapes, gated and not,
        with ties, signed zeros and NaNs: identical id lists or sctopk stays off."""
        try:
            gen = torch.Generator(device=device).manual_seed(20261026)
            c = 4
            stop = torch.zeros(3, dtype=torch.int32, device=device)
            for n, k, gather in ((SCREEN_CHUNK, PRE_PER, False), (PRE_PER, CASCADE_PER, True),
                                 (CASCADE_PER, SCREEN_PER, True)):
                for trial in range(6):
                    s = torch.randn(c * n, device=device, generator=gen) * 3
                    if trial == 1:
                        s = torch.round(s * 4) / 4
                    elif trial == 2:
                        s[::7] = float("nan")
                        s[::5] = -0.0
                    elif trial == 3:
                        s = torch.zeros_like(s)
                    src = torch.randperm(1 << 20, device=device, generator=gen)[: c * n]
                    for flag in (0, 1):
                        stop.fill_(flag)
                        a = torch.full((c * k,), -1, dtype=torch.int64, device=device)
                        b = torch.full((c * k,), -2, dtype=torch.int64, device=device)
                        _chunk_topk[(c,)](s, src if gather else s, a, N=n, K=k, GATHER=gather,
                                          RADIX=True, num_warps=4, gate_ptr=stop, GATED=True)
                        if not _sctopk.chunk_topk(s, src, b, n, k, gather, gate=stop, pdl=False):
                            return False
                        if not torch.equal(a, b):
                            logger.warning("MTP screen sctopk differs; _chunk_topk retained")
                            return False
            torch.cuda.synchronize(device)
            return True
        except Exception as e:  # noqa: BLE001 - any failure keeps _chunk_topk
            logger.warning("MTP screen sctopk unavailable: %r", e)
            return False

    def _topk(self, scores, src, out, n, k, gather, tgate, tgated, num_warps):
        if self.sctopk and _sctopk.chunk_topk(
            scores, src, out, n, k, gather, gate=tgate if tgated else None, pdl=mtp_pdl()
        ):
            return
        _chunk_topk[(self.n // SCREEN_CHUNK,)](
            scores, src if gather else scores, out,
            N=n, K=k, GATHER=gather, RADIX=_radix_for(self.radix, n),
            num_warps=num_warps, launch_pdl=mtp_pdl(), gate_ptr=tgate, GATED=tgated,
        )

    def _tail_matches(self, k: int, device: torch.device) -> bool:
        """Load time, before capture: scrtail against the three-level cascade on random rows
        (several scales, signed zeros, ties) and on the stopped (gated) pass, bitwise on the
        candidate ids, their logits, every intermediate buffer and the token."""
        gate = depth_gate_init(device)
        try:
            gen = torch.Generator(device=device)
            gen.manual_seed(20261027)
            out = torch.empty((), dtype=torch.int32, device=device)
            names = ("_pre_cand", "mid_scores", "scores", "_coarse_cand", "refine_scores",
                     "_cand", "cand_logits")
            self._tail_debug = True
            for trial in range(10):
                h = torch.randn(k, generator=gen, device=device) * (0.25 * 2 ** (trial % 6))
                if trial == 6:
                    h[::3] = 0.0
                    h[1::7] = -0.0
                elif trial == 7:
                    h = torch.round(h * 4) / 4
                elif trial == 8:
                    h = h * 1e-20
                h = h.float().contiguous()
                for stop in ((0, 1) if gate is not None else (0,)):
                    outs = []
                    for tail in (False, True):
                        if gate is not None:
                            gate.zero_()
                            gate[0] = stop
                        for name in names:
                            buf = getattr(self, name)
                            buf.fill_(-7 if buf.dtype == torch.int64 else float("nan"))
                        self.tail = tail
                        tok = self._greedy_row(h, gated=gate is not None, out=out)
                        torch.cuda.synchronize(device)
                        outs.append([tok.clone()] + [getattr(self, n).clone() for n in names])
                    for a, b in zip(*outs):
                        if a.dtype == torch.float32:
                            a, b = a.view(torch.int32), b.view(torch.int32)
                        if not torch.equal(a, b):
                            logger.warning("MTP screen: scrtail differs from the cascade; "
                                           "cascade kept")
                            return False
            logger.info("MTP screen: scrtail admitted (cascade after the first dot in one "
                        "kernel per chunk)")
            return True
        except Exception as exc:  # noqa: BLE001 - any failure keeps the cascade
            logger.warning("MTP screen: scrtail unavailable (%r); cascade kept", exc)
            return False
        finally:
            self.tail = False
            self._tail_debug = False
            if gate is not None:
                gate.zero_()

    def _fold_matches(self, k: int, device: torch.device) -> bool:
        """Load time, before capture: the folded cascade against the unfolded one on
        random rows, bitwise on every stage's scores and on the token."""
        try:
            gen = torch.Generator(device=device)
            gen.manual_seed(1234)
            for _ in range(3):
                h = torch.randn(k, generator=gen, device=device) * 2.0
                outs = []
                for fold in (True, False):
                    self.fold = fold
                    tok = self._greedy_row(h)
                    outs.append([tok.clone(), self.scores.clone(),
                                 self.refine_scores.clone(), self.cand_logits.clone()])
                for a, b in zip(*outs):
                    if not torch.equal(a.view(torch.int32), b.view(torch.int32)):
                        logger.warning("MTP screen: folded gathers differ; unfolded path kept")
                        return False
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("MTP screen: folded gathers unavailable (%r)", exc)
            return False
        finally:
            self.fold = True

    def _greedy_row(
        self, h: torch.Tensor, gated: bool = False, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """One row h [K] float32: the cascade (or the single screen) and the
        argmax tie rule, written to out (a fresh int32 scalar when None). Every
        buffer it touches is fixed-address, so the body is capturable as-is."""
        if out is None:
            out = torch.empty((), dtype=torch.int32, device=h.device)
        if self.cascade and self.fused_topk and self.pre:
            # LOCAL PATCH (cudafast-nemotron35): the three-level cascade
            # (see __init__); the refine stage and everything after it are
            # the two-level cascade's.
            from .nemotron_h_mtp_gather_first import maybe_first as gather_first
            if not gather_first(self, h, gated):
                _gather_three[(1,)](
                    h, self.pre_cols, self._hp, self.mid_cols, self._hm,
                    self.rest_cols, self._hr,
                    NA=self.pre_cols.numel(), NB=self.mid_cols.numel(),
                    NC=self.rest_cols.numel(), launch_pdl=mtp_pdl(),
                )
                _rows_dot(
                    self._hp, self.w_pre, self.s_pre, None,
                    self.gscale, self.scores, False, gated,
                )
            nchunk = self.n // SCREEN_CHUNK
            tgate, tok = _gate_args(self.scores)
            tgated = gated and tok
            if self.tail and _scrtail.run(
                self.scores, self._hm, self._hr, h, self.w_mid, self.s_mid, self.w_rest,
                self.s_rest, self.w_full, self.s_full, self._cand, self.cand_logits,
                self.gscale, gate=tgate if tgated else None, win=0, pdl=mtp_pdl(),
                debug=(self._pre_cand, self.mid_scores, self.scores, self._coarse_cand,
                       self.refine_scores) if self._tail_debug else None,
            ):
                gate, ok = _gate_args(out)
                _select_token[(1,)](
                    self.cand_logits, self._cand, out, N=self.cand_logits.numel(),
                    BLOCK=triton.next_power_of_2(self.cand_logits.numel()),
                    gate_ptr=gate, GATED=gated and ok, THETA=DEPTH_THETA,
                    CUM=DEPTH_CUMULATIVE, launch_pdl=mtp_pdl(),
                )
                return out
            # LOCAL PATCH (drafter3): each top-k selection below leaves DRAM idle: a
            # piece of what the next draft pass reads first is queued as each streaming
            # dot ends (skipped on the device for a stopped loop and for the last pass);
            # all are joined after the token.
            next_pf = gated and tok and len(self._next_pf) == len(NEXT_PF_SHARES)
            if next_pf:
                self._next_pf[0].launch(tgate, self._next_max_kept)
            self._topk(
                self.scores, self.scores, self._pre_cand, SCREEN_CHUNK, PRE_PER, False,
                tgate, tgated, 4 if self.radix else 8,
            )
            # The coarse score: pre score + the next 16 groups, also stored at
            # its row for the refine stage's base.
            _rows_dot(
                self._hm, self.w_mid, self.s_mid, self._pre_cand,
                self.gscale, self.mid_scores, True, gated,
                base=self.scores, scatter=self.scores,
            )
            if next_pf:
                self._next_pf[1].launch(tgate, self._next_max_kept)
            self._topk(
                self.mid_scores, self._pre_cand, self._coarse_cand, PRE_PER, CASCADE_PER, True,
                tgate, tgated, 4,
            )
            _rows_dot(
                self._hr, self.w_rest, self.s_rest, self._coarse_cand,
                self.gscale, self.refine_scores, True, gated, base=self.scores,
            )
            if next_pf:
                self._next_pf[2].launch(tgate, self._next_max_kept)
            self._topk(
                self.refine_scores, self._coarse_cand, self._cand, CASCADE_PER, SCREEN_PER, True,
                tgate, tgated, 4,
            )
            cand = self._cand
            if next_pf:
                self._finish_next_pf = True
        elif self.cascade and self.fused_topk:
            # LOCAL PATCH (cudafast-nemotron35): the same cascade with the two
            # index_select gathers in one launch and each segmented torch.topk
            # (+ chunk offset / + gather) in one launch that writes the id list
            # torch.topk(sorted=False) returns, in torch's order (_chunk_topk).
            # Same candidate ids in the same order, so every rows_dot, the
            # selection and the depth gate see identical inputs.
            if self.reuse:
                _gather_two[(1,)](
                    h, self.coarse_cols, self._hc, self.rest_cols, self._hr,
                    NA=self.coarse_cols.numel(), NB=self.rest_cols.numel(),
                    launch_pdl=mtp_pdl(),
                )
            else:
                _gather_two[(1,)](
                    h, self.coarse_cols, self._hc, self.cols, self._hs,
                    NA=self.coarse_cols.numel(), NB=self.cols.numel(),
                    launch_pdl=mtp_pdl(),
                )
            _rows_dot(
                self._hc, self.w_coarse, self.s_coarse, None,
                self.gscale, self.scores, False, gated,
            )
            nchunk = self.n // SCREEN_CHUNK
            tgate, tok = _gate_args(self.scores)
            tgated = gated and tok
            _chunk_topk[(nchunk,)](
                self.scores, self.scores, self._coarse_cand,
                N=SCREEN_CHUNK, K=CASCADE_PER, GATHER=False, RADIX=_radix_for(self.radix, SCREEN_CHUNK),
                num_warps=4 if self.radix else 8, launch_pdl=mtp_pdl(),
                gate_ptr=tgate, GATED=tgated,
            )
            if self.reuse:
                # refine score = coarse score (its 32 groups) + the other 32 refine
                # groups: the same 64-group sum, added in a different fp32 order, so
                # it reads 256 of the 512 bytes per candidate row (the coarse groups
                # are a subset of the refine groups). VLLM_MTP_SCREEN_REUSE=0 keeps
                # the full 64-group dot.
                _rows_dot(
                    self._hr, self.w_rest, self.s_rest, self._coarse_cand,
                    self.gscale, self.refine_scores, True, gated, base=self.scores,
                )
            else:
                _rows_dot(
                    self._hs, self.w_scr, self.s_scr, self._coarse_cand,
                    self.gscale, self.refine_scores, True, gated,
                )
            _chunk_topk[(nchunk,)](
                self.refine_scores, self._coarse_cand, self._cand,
                N=CASCADE_PER, K=SCREEN_PER, GATHER=True, RADIX=_radix_for(self.radix, CASCADE_PER),
                num_warps=4, launch_pdl=mtp_pdl(), gate_ptr=tgate, GATED=tgated,
            )
            cand = self._cand
        elif self.cascade and self.fold:
            # The same arithmetic as the unfolded cascade below, with the column
            # gathers of h, the chunk offsets and the candidate gather done as
            # index arithmetic inside the dot and select kernels.
            _rows_dot(
                h, self.w_coarse, self.s_coarse, None,
                self.gscale, self.scores, False, gated, hcols=self.coarse_cols,
            )
            coarse_idx = self.scores.view(-1, SCREEN_CHUNK).topk(
                CASCADE_PER, dim=-1, sorted=False
            ).indices
            _rows_dot(
                h, self.w_scr, self.s_scr, coarse_idx,
                self.gscale, self.refine_scores, True, gated, hcols=self.cols,
                rmode=1, cper=CASCADE_PER,
            )
            refined = self.refine_scores.view(-1, CASCADE_PER).topk(
                SCREEN_PER, dim=-1, sorted=False
            ).indices
            _rows_dot(
                h, self.w_full, self.s_full, coarse_idx, self.gscale, self.cand_logits,
                True, gated, ref=refined, rmode=2, cper=CASCADE_PER, sper=SCREEN_PER,
            )
            gate, ok = _gate_args(out)
            _select_token[(1,)](
                self.cand_logits, coarse_idx, out, N=self.cand_logits.numel(),
                BLOCK=triton.next_power_of_2(self.cand_logits.numel()),
                gate_ptr=gate, GATED=gated and ok, THETA=DEPTH_THETA,
                CUM=DEPTH_CUMULATIVE, launch_pdl=mtp_pdl(),
                ref=refined, TWO_LEVEL=True, CHUNK=SCREEN_CHUNK, CPER=CASCADE_PER,
                SPER=SCREEN_PER,
            )
            return out
        elif self.cascade:
            hc = h.index_select(0, self.coarse_cols)
            _rows_dot(
                hc, self.w_coarse, self.s_coarse, None,
                self.gscale, self.scores, False, gated,
            )
            # Every consumer below is order-independent (index-aligned gathers,
            # then the max with the lowest-id tie rule), so the candidate sets
            # need not be sorted: sorted=False skips a segmented sort per stage.
            coarse_cand = (
                self.scores.view(-1, SCREEN_CHUNK)
                .topk(CASCADE_PER, dim=-1, sorted=False).indices + self.chunk_off
            )
            hs = h.index_select(0, self.cols)
            _rows_dot(
                hs, self.w_scr, self.s_scr, coarse_cand.reshape(-1),
                self.gscale, self.refine_scores, True, gated,
            )
            # Scores retain the [original chunk, coarse candidate] layout.
            # Gather global IDs; adding chunk_off here would offset twice.
            refined = self.refine_scores.view(-1, CASCADE_PER).topk(
                SCREEN_PER, dim=-1, sorted=False
            ).indices
            cand = coarse_cand.gather(1, refined).reshape(-1)
        else:
            hs = h.index_select(0, self.cols)
            _rows_dot(
                hs, self.w_scr, self.s_scr, None, self.gscale, self.scores, False, gated
            )
            cand = (
                self.scores.view(-1, SCREEN_CHUNK)
                .topk(SCREEN_PER, dim=-1, sorted=False).indices
                + self.chunk_off
            ).reshape(-1)
        if self._full_part is not None and cand.numel() == self.cand_logits.numel():
            _rows_dot_ksplit(
                h, self.w_full, self.s_full, cand, self.gscale, self.cand_logits,
                self._full_part, gated,
            )
            if getattr(self, "_finish_next_pf", False):
                tgate, _ = _gate_args(self.scores)
                self._next_pf[3].launch(tgate, self._next_max_kept)
        else:
            _rows_dot(
                h, self.w_full, self.s_full, cand, self.gscale, self.cand_logits, True, gated
            )
        gate, ok = _gate_args(out)
        _select_token[(1,)](
            self.cand_logits, cand, out, N=self.cand_logits.numel(),
            BLOCK=triton.next_power_of_2(self.cand_logits.numel()),
            gate_ptr=gate, GATED=gated and ok, THETA=DEPTH_THETA,
            CUM=DEPTH_CUMULATIVE, launch_pdl=mtp_pdl(),
        )
        if getattr(self, "_finish_next_pf", False):
            self._finish_next_pf = False
            _l2pf.join(out.device)
        return out

    def greedy(self, hidden: torch.Tensor) -> torch.Tensor:
        """Draft ids [B] for hidden [B, K] (B is small; one pass per row)."""
        if self.graph is not None and hidden.shape[0] == 1:
            self._h.copy_(hidden[0])
            self.graph.replay()
            return self._out.clone()
        out = []
        for i in range(hidden.shape[0]):
            out.append(self._greedy_row(hidden[i].float().contiguous()))
        return torch.stack(out)

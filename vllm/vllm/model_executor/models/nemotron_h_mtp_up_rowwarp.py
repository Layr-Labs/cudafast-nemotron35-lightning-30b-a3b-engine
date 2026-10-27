# SPDX-License-Identifier: Apache-2.0
"""Optional native BF16 packed-up row-owned warp schedule; remote admission only."""
from __future__ import annotations

import os
import statistics

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as tl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout
from vllm.logger import init_logger
from vllm.triton_utils import triton

from .nemotron_h_mtp_packed import (
    load_packed_group8, packed_moe_up, pdl_prologue, stopped_zero,
    word_load_enabled,
)

logger = init_logger(__name__)
_OK = {}
_SENTINEL = 0x2BAD
I, K = 1856, 2688


@gluon.jit
def _row_payload(SM, EX, BASE, RI, RAW, SSM, SEX, SBASE, SRI, SRAW,
                 ids, slot, group_n, group_k, lane, k0,
                 TOPK: tl.constexpr, EVICT: tl.constexpr,
                 layout: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 I: tl.constexpr = 1856, K: tl.constexpr = 2688):
    gk = k0 + group_k
    if slot < TOPK:
        index = tl.load(ids + slot).to(tl.int64) * (I * K) + group_n.to(tl.int64) * K + gk
    else:
        index = ((slot - TOPK) * I + group_n).to(tl.int64) * K + gk
    mask = (group_n < I) & (gk < K)
    value_mask = mask[:, None] & (gk[:, None] + lane[None, :] < K)
    if slot < TOPK:
        w8 = load_packed_group8(SM, EX, BASE, RI, RAW, index, lane, mask, value_mask, EVICT=EVICT)
    else:
        w8 = load_packed_group8(SSM, SEX, SBASE, SRI, SRAW, index, lane, mask, value_mask, EVICT=EVICT)
    return tl.convert_layout(tl.reshape(w8, (BN, BK)), layout)


@gluon.jit
def _row_warp(x, SM, EX, BASE, RI, RAW, SSM, SEX, SBASE, SRI, SRAW,
              ids, h, gate, TOPK: tl.constexpr, EVICT: tl.constexpr,
              GATED: tl.constexpr, launch_pdl: tl.constexpr,
              BN: tl.constexpr = 8, BK: tl.constexpr = 1024,
              I: tl.constexpr = 1856, K: tl.constexpr = 2688):
    if launch_pdl:
        pdl_prologue()
    # One physical warp owns a row's mainloop. Its 32 lanes each carry four
    # original logical K-warp stripes, keeping the same logical [N,K] terms.
    layout: tl.constexpr = BlockedLayout([1, 8], [1, 32], [8, 1], [1, 0])
    reference_layout: tl.constexpr = BlockedLayout([1, 8], [1, 32], [2, 4], [1, 0])
    group_layout: tl.constexpr = BlockedLayout([1, 8], [32, 1], [8, 1], [1, 0])
    row = tl.program_id(0) * BN + tl.arange(0, BN, layout=SliceLayout(1, layout))
    kk = tl.arange(0, BK, layout=SliceLayout(0, layout))
    slot = tl.program_id(1)
    if GATED:
        if tl.load(gate) != 0:
            stopped_zero(h + slot * I + row, row < I, SliceLayout(1, layout), BN)
            return
    group = tl.arange(0, BN * (BK // 8), layout=SliceLayout(1, group_layout))
    lane = tl.arange(0, 8, layout=SliceLayout(0, group_layout))
    group_n = tl.program_id(0) * BN + group // (BK // 8)
    group_k = (group % (BK // 8)) * 8
    current = _row_payload(SM, EX, BASE, RI, RAW, SSM, SEX, SBASE, SRI, SRAW,
                           ids, slot, group_n, group_k, lane, 0,
                           TOPK, EVICT, layout, BN, BK)
    current_x = tl.load(x + kk, mask=kk < K, other=0.)
    acc = tl.full((BN, BK), 0., tl.float32, layout)
    for start in range(BK, K, BK):
        # Independent next weight decode and input load precede the current
        # logical element FMA. No reassociation across K chunks is requested.
        following = _row_payload(SM, EX, BASE, RI, RAW, SSM, SEX, SBASE, SRI, SRAW,
                                 ids, slot, group_n, group_k, lane, start,
                                 TOPK, EVICT, layout, BN, BK)
        following_x = tl.load(x + start + kk, mask=start + kk < K, other=0.)
        acc += current.to(tl.float32) * current_x[None, :].to(tl.float32)
        current = following
        current_x = following_x
    acc += current.to(tl.float32) * current_x[None, :].to(tl.float32)
    # Relayout values (no arithmetic) before the ORIGINAL four-K-warp
    # reduction layout. Compiled reduction conformance still needs admission.
    acc = tl.convert_layout(acc, reference_layout)
    result = tl.maximum(tl.sum(acc, axis=1), 0.)
    result = result * result
    out_row = tl.program_id(0) * BN + tl.arange(0, BN, layout=SliceLayout(1, reference_layout))
    tl.store(h + slot * I + out_row, result.to(h.dtype.element_ty), mask=out_row < I)


def _bits(x):
    return x.contiguous().reshape(-1).view(torch.uint8)


def _same(a, b):
    return torch.equal(_bits(a), _bits(b))


def _guarded(shape, dtype, device):
    n = 1
    for s in shape:
        n *= s
    pad = 128 if dtype == torch.bfloat16 else 64
    allocation = torch.empty(n + pad * 2, dtype=dtype, device=device)
    allocation.view(torch.int16 if dtype == torch.bfloat16 else torch.int32).fill_(_SENTINEL)
    return allocation, allocation[pad:-pad].view(shape), pad


def _guard_ok(a, pad):
    b = a.view(torch.int16 if a.dtype == torch.bfloat16 else torch.int32)
    return bool(torch.all(b[:pad] == _SENTINEL)) and bool(torch.all(b[-pad:] == _SENTINEL))


def _launch(pu, ps, x, ids, h, gate, gated, pdl, evict):
    _row_warp[(triton.cdiv(I, 8), ids.numel() + 2)](
        x, *pu.planes, *ps.planes, ids, h, gate,
        TOPK=ids.numel(), EVICT=evict, GATED=gated, launch_pdl=pdl,
        num_warps=8, num_stages=3)


def maybe_up(pu, ps, x, ids, h, gate, gated, pdl):
    if (ids.dtype != torch.int32 or x.dtype != torch.bfloat16 or x.numel() != K
            or not x.is_contiguous() or not ids.is_contiguous() or not h.is_contiguous()
            or tuple(h.shape) != (ids.numel() + 2, I) or h.dtype != torch.bfloat16):
        return False
    entry = _OK.get((id(pu), id(ps), ids.numel(), word_load_enabled(x.device), bool(gated)))
    if entry is None or entry[0] is not pu or entry[1] is not ps:
        return False
    _launch(pu, ps, x, ids, h, gate, gated, pdl, entry[2])
    return True


@torch.no_grad()
def prepare(pu, ps):
    """Private actual packed-up reference, guards, changed graphs and timing.

    No request activations/graphs are retained. Any optional failure keeps the
    original production launch, including its actual selected word setting.
    """
    if (os.environ.get("MTP_UP_ROW_WARP", "1") == "0"
            or tuple(pu.shape) != (128, I, K) or tuple(ps.shape) != (2 * I, K)
            or torch.cuda.is_current_stream_capturing()):
        return
    device = pu.planes[0].device
    if torch.cuda.get_device_capability(device) != (12, 1):
        return
    from .nemotron_h_mtp_kernels import (
        _EVICT_W, _MOE_UP_BK, _MOE_UP_STAGES, _PK_UP_BN, _PK_UP_WN, mtp_pdl,
    )
    # The relayout above binds to this exact current selected reference.
    if (_MOE_UP_BK, _MOE_UP_STAGES, _PK_UP_BN, _PK_UP_WN) != (1024, 3, 2, 2):
        return
    word, pdl, evict = word_load_enabled(device), mtp_pdl(), _EVICT_W
    immutable = [(a, a.clone()) for a in (*pu.planes, *ps.planes)]
    gen = torch.Generator(device=device).manual_seed(271028)
    pressure = torch.empty(64 << 20, dtype=torch.uint8, device=device)

    def owners_unchanged():
        if any(not _same(a, b) for a, b in immutable):
            raise ValueError("row-warp changed packed owner")

    try:
        for topk in (3, 6):
            xb, x, xp = _guarded((K,), torch.bfloat16, device)
            ib, ids, ip = _guarded((topk,), torch.int32, device)
            gb, gate, gp = _guarded((1,), torch.int32, device)
            ob, old, op = _guarded((topk + 2, I), torch.bfloat16, device)
            nb, new, np = _guarded((topk + 2, I), torch.bfloat16, device)

            def case(first, mode, stopped=0):
                ids.copy_((torch.arange(topk, device=device, dtype=torch.int32) + first) % 128)
                gate.fill_(stopped)
                x.copy_(torch.randn((K,), device=device, generator=gen))
                x.mul_((1., .001, 32., 1., 1., 1., 1.)[mode])
                x.view(torch.int16)[:4] = torch.tensor([0, -32768, 1, -32767], device=device, dtype=torch.int16)
                if mode == 3:
                    x.zero_()
                    x[::2] = -0.
                if mode == 4:
                    x.copy_((x.float() * 4).round() / 4)
                if mode == 5:
                    x.view(torch.int16)[4:6] = torch.tensor([32640, -128], device=device, dtype=torch.int16)
                if mode == 6:
                    x.view(torch.int16)[4:6] = torch.tensor([32704, -64], device=device, dtype=torch.int16)

            for gated in (False, True):
                def run(which):
                    if which:
                        _launch(pu, ps, x, ids, new, gate, gated, pdl, evict)
                    else:
                        packed_moe_up[(triton.cdiv(I, _PK_UP_BN), topk + 2)](
                            x, *pu.planes, *ps.planes, ids, old, I, K,
                            I * K, K, K, gate, TOPK=topk, BN=_PK_UP_BN,
                            BK=_MOE_UP_BK, EVICT=evict, GATED=gated, WN=_PK_UP_WN,
                            WORD=word, num_warps=4 * _PK_UP_WN,
                            num_stages=_MOE_UP_STAGES, launch_pdl=pdl)

                def compare(graphs=None):
                    copies = tuple(a.clone() for a in (xb, ib, gb))
                    ob.view(torch.int16).fill_(_SENTINEL)
                    nb.view(torch.int16).fill_(_SENTINEL)
                    for which in (0, 1):
                        graphs[which].replay() if graphs else run(which)
                    if (not _same(ob, nb)
                            or any(not _same(a, b) for a, b in zip((xb, ib, gb), copies))
                            or not all(_guard_ok(a, pad) for a, pad in
                                       ((xb, xp), (ib, ip), (gb, gp), (ob, op), (nb, np)))):
                        raise ValueError("row-warp output/fullguard/input mismatch")

                for first in range(0, 128, topk):
                    for mode in range(7 if first == 0 else 5):
                        case(first, mode)
                        compare()
                    if gated:
                        case(first, 0, 1)
                        compare()
                owners_unchanged()
                case(0, 0)
                graphs = []
                for which in (0, 1):
                    for _ in range(2):
                        run(which)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
                        run(which)
                    graphs.append(graph)
                for first, mode, stopped in ((125, 2, 0), (43, 1, 0), (17, 3, 0),
                                               (93, 4, 1 if gated else 0), (0, 0, 0)):
                    case(first, mode, stopped)
                    compare(graphs)
                    compare(graphs)
                times, ratios = [], []
                for first in (17, 93, 125):
                    case(first, 0)
                    compare(graphs)
                    for cold in (False, True):
                        samples = [[], []]
                        for repeat in range(5):
                            for which in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                                if cold:
                                    pressure.fill_(repeat + 1)
                                else:
                                    for _ in range(8):
                                        graphs[which].replay()
                                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                reps = 1 if cold else 16
                                start.record()
                                for _ in range(reps):
                                    graphs[which].replay()
                                end.record()
                                end.synchronize()
                                value = start.elapsed_time(end) / reps
                                if not value > 0:
                                    raise ValueError("invalid private row-warp timing")
                                samples[which].append(value)
                        a, b = (statistics.median(s) for s in samples)
                        times.append((a, b))
                        ratios.append(b / a)
                        compare(graphs)
                owners_unchanged()
                admitted = (statistics.median(ratios) < .995
                            and sum(b for a, b in times) / sum(a for a, b in times) < .995
                            and max(ratios) < 1.01)
                if admitted:
                    _OK[(id(pu), id(ps), topk, word, gated)] = (pu, ps, evict)
                logger.info("MTP up: row-owned warp admitted=%s topk=%d gated=%s", admitted, topk, gated)
    except Exception as exc:
        # A failure after an earlier selection must not leave it active.
        for key in [k for k in _OK if k[:2] == (id(pu), id(ps))]:
            del _OK[key]
        logger.warning("MTP up: optional row-warp unavailable (%r); original retained", exc)
    finally:
        for actual, saved in immutable:
            actual.copy_(saved)

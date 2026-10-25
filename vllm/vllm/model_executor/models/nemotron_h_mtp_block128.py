"""Optional block128 decoder for the native MTP's original packed projections.

Only immutable planes and fixed admission metadata are retained. All probes run
at official remote weight load; ordinary projection arithmetic is unchanged.
"""
import os
import statistics
import weakref

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as tl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout

from vllm.logger import init_logger
from .nemotron_h_mtp_packed import (
    load_packed, load_packed_group8, pdl_prologue, stopped_zero,
)

logger = init_logger(__name__)
_ENABLED = os.environ.get("MTP_PACK12_BLOCK128", "1") != "0"
_LINEAR = {}
_UP = {}


@gluon.jit
def _load128(SM, EX, BASE, RI, RAW, index, mask,
             B: tl.constexpr, WG: tl.constexpr, EVICT: tl.constexpr):
    # (128-value block, exponent word, value within that word). Each
    # exponent word belongs to one thread; its eight values stay in registers.
    lay: tl.constexpr = BlockedLayout([1, 1, 8], [2, 16, 1], [WG, 1, 1], [2, 1, 0])
    word_lane = tl.arange(0, 16, layout=SliceLayout(0, SliceLayout(2, lay)))
    value_lane = tl.arange(0, 8, layout=SliceLayout(0, SliceLayout(0, lay)))
    ix2 = tl.expand_dims(index, 1)
    ix3 = tl.expand_dims(ix2, 2)
    word3 = tl.expand_dims(tl.expand_dims(word_lane, 0), 2)
    value3 = tl.expand_dims(tl.expand_dims(value_lane, 0), 0)
    idx = ix3 + word3 * 8 + value3
    m3 = tl.expand_dims(tl.expand_dims(mask, 1), 2)
    rawid = tl.load(RI + index // 128, mask=mask, other=-1,
                    eviction_policy="evict_first" if EVICT else "")
    coded = mask & (rawid < 0)
    exponent_base = tl.load(BASE + index // 128, mask=coded, other=1,
                            eviction_policy="evict_first" if EVICT else "").to(tl.uint16)
    word = tl.load(EX.to(tl.pointer_type(tl.uint32)) + ix2 // 8 + word_lane[None, :],
                   mask=coded[:, None], other=0,
                   eviction_policy="evict_first" if EVICT else "")
    code = ((tl.expand_dims(word, 2) >> (value3.to(tl.uint32) * 4)) & 15).to(tl.uint16)
    raw3 = tl.expand_dims(tl.expand_dims(rawid, 1), 2)
    base3 = tl.expand_dims(tl.expand_dims(exponent_base, 1), 2)
    sm = tl.load(SM + idx, mask=m3 & (raw3 < 0), other=0,
                 eviction_policy="evict_first" if EVICT else "").to(tl.uint16)
    exp = tl.where(code == 0, 0, base3 + code - 1)
    bits = ((sm & 128) << 8) | (exp << 7) | (sm & 127)
    raw = tl.load(RAW + raw3.to(tl.int64) * 128 + idx % 128,
                  mask=m3 & (raw3 >= 0), other=0,
                  eviction_policy="evict_first" if EVICT else "")
    bits = tl.where(raw3 >= 0, raw, bits)
    bits = tl.where(m3, bits, 0).to(tl.uint16)
    return bits.to(tl.bfloat16, bitcast=True), idx


@gluon.jit
def _decode_bits(SM, EX, BASE, RI, RAW, W, bad, N: tl.constexpr,
                 B: tl.constexpr, WG: tl.constexpr):
    lay: tl.constexpr = BlockedLayout([1, 1, 8], [2, 16, 1], [WG, 1, 1], [2, 1, 0])
    block = tl.program_id(0) * B + tl.arange(0, B, layout=SliceLayout(1, SliceLayout(2, lay)))
    weight, idx = _load128(SM, EX, BASE, RI, RAW, block * 128, block * 128 < N,
                           B, WG, True)
    want = tl.load(W + idx, mask=idx < N, other=0)
    diff = (weight.to(tl.uint16, bitcast=True) != want).to(tl.int32)
    count = tl.sum(tl.sum(tl.sum(diff, axis=2), axis=1), axis=0)
    if count:
        tl.atomic_add(bad, count)


@gluon.jit
def _project(x, SM, EX, BASE, RI, RAW, SSM, SEX, SBASE, SRI, SRAW,
              ids, out, gate, raw_out, N, K, stride_we, stride_wn, stride_wsn,
              TOPK: tl.constexpr, MOE: tl.constexpr, BN: tl.constexpr,
              BK: tl.constexpr, WN: tl.constexpr, WG: tl.constexpr,
              EVICT: tl.constexpr, GATED: tl.constexpr,
              launch_pdl: tl.constexpr, NEW: tl.constexpr,
              WORD: tl.constexpr, GROUP8: tl.constexpr,
              SAVE_RAW: tl.constexpr = False):
    if launch_pdl:
        pdl_prologue()
    warp_n: tl.constexpr = WN if MOE else 2
    warp_k: tl.constexpr = 4 if MOE else 2
    lay: tl.constexpr = BlockedLayout([1, 8], [1, 32], [warp_n, warp_k], [1, 0])
    n = tl.program_id(0) * BN + tl.arange(0, BN, layout=SliceLayout(1, lay))
    k = tl.arange(0, BK, layout=SliceLayout(0, lay))
    slot = tl.program_id(1) if MOE else 0
    nmask = n < N
    if GATED:
        if tl.load(gate) != 0:
            stopped_zero(out + slot * N + n, nmask, SliceLayout(1, lay), BN)
            if SAVE_RAW:
                tl.store(raw_out + slot * N + n,
                         tl.full((BN,), 0., tl.float32, SliceLayout(1, lay)), mask=nmask)
            return
    if NEW:
        dl: tl.constexpr = BlockedLayout([1, 1, 8], [2, 16, 1], [WG, 1, 1], [2, 1, 0])
        block = tl.arange(0, BN * (BK // 128), layout=SliceLayout(1, SliceLayout(2, dl)))
        group_n = tl.program_id(0) * BN + block // (BK // 128)
        group_k = (block % (BK // 128)) * 128
    elif GROUP8:
        group_warps: tl.constexpr = WN * 4 if MOE else 4
        gl: tl.constexpr = BlockedLayout([1, 8], [32, 1], [group_warps, 1], [1, 0])
        group = tl.arange(0, BN * (BK // 8), layout=SliceLayout(1, gl))
        lane = tl.arange(0, 8, layout=SliceLayout(0, gl))
        group_n = tl.program_id(0) * BN + group // (BK // 8)
        group_k = (group % (BK // 8)) * 8
    if MOE:
        if slot < TOPK:
            expert = tl.load(ids + slot).to(tl.int64)
            wbase = expert * stride_we
        else:
            wbase = (slot - TOPK).to(tl.int64) * N * stride_wsn
    else:
        wbase = 0
    acc = tl.full((BN, BK), 0., tl.float32, lay)
    for k0 in range(0, K, BK):
        kk = k0 + k
        if NEW or GROUP8:
            gk = k0 + group_k
            if MOE:
                if slot < TOPK:
                    stride_n = stride_wn
                else:
                    stride_n = stride_wsn
            else:
                stride_n = stride_wn
            index = wbase + group_n.to(tl.int64) * stride_n + gk
            mask = (group_n < N) & (gk < K)
            if NEW:
                if MOE and slot >= TOPK:
                    wb, _ = _load128(SSM, SEX, SBASE, SRI, SRAW, index, mask,
                                     BN * (BK // 128), WG, EVICT)
                else:
                    wb, _ = _load128(SM, EX, BASE, RI, RAW, index, mask,
                                     BN * (BK // 128), WG, EVICT)
                weight = tl.convert_layout(tl.reshape(wb, (BN, BK)), lay)
            else:
                value_mask = mask[:, None] & (gk[:, None] + lane[None, :] < K)
                if MOE and slot >= TOPK:
                    wb = load_packed_group8(SSM, SEX, SBASE, SRI, SRAW,
                                            index, lane, mask, value_mask, EVICT=EVICT)
                else:
                    wb = load_packed_group8(SM, EX, BASE, RI, RAW,
                                            index, lane, mask, value_mask, EVICT=EVICT)
                weight = tl.convert_layout(tl.reshape(wb, (BN, BK)), lay)
        else:
            index = wbase + n[:, None].to(tl.int64) * stride_wn + kk[None, :]
            weight = load_packed(SM, EX, BASE, RI, RAW, index,
                                 nmask[:, None] & (kk[None, :] < K), EVICT=EVICT, WORD=WORD)
        value = tl.load(x + kk, mask=kk < K, other=0.)
        acc += weight.to(tl.float32) * value[None, :].to(tl.float32)
    raw_sum = tl.sum(acc, axis=1)
    if SAVE_RAW:
        tl.store(raw_out + slot * N + n, raw_sum, mask=nmask)
    result = raw_sum
    if MOE:
        result = tl.maximum(result, 0.)
        result = result * result
    tl.store(out + slot * N + n, result.to(out.dtype.element_ty), mask=nmask)


@gluon.jit
def _producer(x, gate, seed, K: tl.constexpr, B: tl.constexpr, launch_pdl: tl.constexpr):
    if launch_pdl:
        pdl_prologue()
    i = tl.program_id(0) * B + tl.arange(0, B, layout=BlockedLayout([1], [32], [4], [0]))
    s = tl.load(seed)
    val = (((i * 17 + s * 37) % 257).to(tl.float32) - 128.) / 128.
    val = tl.where(i % 17 == 0, -0., val)
    tl.store(x + i, val.to(x.dtype.element_ty), mask=i < K)
    if tl.program_id(0) == 0:
        tl.store(gate, (s % 3 == 0).to(tl.int32))


def _planes_key(p):
    return tuple(t.data_ptr() for t in p.planes)


def _refs(p, ps=None):
    return tuple(weakref.ref(t) for t in (p.planes if ps is None else (*p.planes, *ps.planes)))


def _matches(choice, p, ps=None):
    tensors = p.planes if ps is None else (*p.planes, *ps.planes)
    return bool(choice) and len(choice) == len(tensors) and all(r() is t for r, t in zip(choice, tensors))


def _launch(p, ps, x, ids, out, gate, mode, *, new, raw=None):
    moe, n, k, topk, bn, bk, wn, wg, evict, gated, pdl, word, g8 = mode
    _project[(triton.cdiv(n, bn), topk + 2 if moe else 1)](
        x, *p.planes, *ps.planes, ids, out, gate, raw, n, k,
        n * k if moe else 0, k, k, TOPK=topk, MOE=moe, BN=bn, BK=bk,
        WN=wn, WG=wg, EVICT=evict, GATED=gated, launch_pdl=pdl,
        NEW=new, WORD=word, GROUP8=g8, SAVE_RAW=raw is not None,
        num_warps=wg, num_stages=3)


def _original(p, ps, x, ids, out, gate, mode):
    from .nemotron_h_mtp_packed import packed_gemv, packed_moe_up
    moe, n, k, topk, bn, bk, wn, wg, evict, gated, pdl, word, g8 = mode
    if moe:
        packed_moe_up[(triton.cdiv(n, bn), topk + 2)](
            x, *p.planes, *ps.planes, ids, out, n, k, n * k, k, k, gate,
            TOPK=topk, BN=bn, BK=bk, WN=wn, EVICT=evict,
            GATED=gated, launch_pdl=pdl, WORD=word, GROUP8=g8,
            num_warps=wg, num_stages=3)
    else:
        packed_gemv[(triton.cdiv(n, bn),)](
            x, *p.planes, out, n, k, k, gate, BN=bn, BK=bk,
            EVICT=evict, GATED=gated, launch_pdl=pdl, WORD=word, GROUP8=g8,
            num_warps=wg, num_stages=3)


def _same(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _guarded(length, device, dtype, fill):
    backing = torch.full((length + 16,), fill, device=device, dtype=dtype)
    return backing, backing[8:-8]


def _capture(call, device):
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
        call()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    pool = torch.cuda.graph_pool_handle()
    outer = torch.cuda.current_stream()
    captured = False
    try:
        with torch.cuda.graph(graph, stream=stream, pool=pool):
            call()
        captured = True
    finally:
        torch.cuda.set_stream(outer)
        if not captured:
            try:
                torch._C._cuda_endAllocateToPool(device.index, pool)
            except Exception:
                pass
    return graph


def _all_bits(packed, original, wg, b):
    count = torch.zeros(1, device=original.device, dtype=torch.int32)
    _decode_bits[(triton.cdiv(original.numel(), b * 128),)](
        *packed.planes, original.view(torch.uint16), count,
        N=original.numel(), B=b, WG=wg, num_warps=wg)
    return int(count.item()) == 0


def _check(p, ps, mode, gen, ids_dtype=torch.int32):
    moe, n, k, topk, bn, bk, wn, wg, evict, gated, pdl, word, g8 = mode
    dev = p.planes[0].device
    slots = topk + 2 if moe else 1
    yb = [_guarded(slots * n, dev, torch.bfloat16, .0625) for _ in range(4)]
    rb = [_guarded(slots * n, dev, torch.float32, .09375) for _ in range(2)]
    xb, xv = _guarded(k, dev, torch.bfloat16, .125)
    x = xv.view(1, k)
    ids = torch.arange(topk, device=dev, dtype=ids_dtype) if moe else None
    gate = torch.zeros(3, device=dev, dtype=torch.int32)
    starts = range(0, 128, topk) if moe else (0,)
    for first in starts:
        if moe:
            ids.copy_((torch.arange(topk, device=dev, dtype=ids_dtype) + first) % 128)
        for case, scale in enumerate((0., .001, 1., 32.)):
            x.copy_((torch.randn(x.shape, device=dev, generator=gen) * scale).to(torch.bfloat16))
            x[:, ::17] = -0.0
            if case == 1:
                x.view(torch.int16)[:, :4] = torch.tensor([1, -32767, 128, -32640], device=dev, dtype=torch.int16)
            if first == 0 and case == 3:
                x.view(torch.int16)[:, :4] = torch.tensor([32640, -128, 32704, -64], device=dev, dtype=torch.int16)
            for stop in ((0, 1) if gated else (0,)):
                gate.fill_(stop)
                for a, _ in yb:
                    a.fill_(.0625)
                for a, _ in rb:
                    a.fill_(.09375)
                _original(p, ps, x, ids, yb[0][1], gate, mode)
                _launch(p, ps, x, ids, yb[1][1], gate, mode, new=False, raw=rb[0][1])
                _launch(p, ps, x, ids, yb[2][1], gate, mode, new=True, raw=rb[1][1])
                _launch(p, ps, x, ids, yb[3][1], gate, mode, new=True)
                torch.cuda.synchronize(dev)
                if not (_same(yb[0][0], yb[1][0]) and _same(yb[0][0], yb[2][0]) and _same(yb[0][0], yb[3][0])
                        and _same(rb[0][0], rb[1][0])):
                    return False
                # Independent fixed canaries, not only old/new agreement.
                for a, fill in [(z[0], .0625) for z in yb] + [(z[0], .09375) for z in rb] + [(xb, .125)]:
                    if not bool(torch.all(a[:8] == fill) & torch.all(a[-8:] == fill)):
                        return False
    # Changing device producer in captured graphs; same seed, separate outputs.
    seed = torch.ones(1, device=dev, dtype=torch.int32)
    def ref():
        _producer[(triton.cdiv(k, 256),)](x, gate, seed, K=k, B=256, launch_pdl=pdl, num_warps=4)
        _original(p, ps, x, ids, yb[0][1], gate, mode)
        _launch(p, ps, x, ids, yb[1][1], gate, mode, new=False, raw=rb[0][1])
    def candidate():
        _producer[(triton.cdiv(k, 256),)](x, gate, seed, K=k, B=256, launch_pdl=pdl, num_warps=4)
        _launch(p, ps, x, ids, yb[2][1], gate, mode, new=True, raw=rb[1][1])
        _launch(p, ps, x, ids, yb[3][1], gate, mode, new=True)
    graphs = [_capture(ref, dev), _capture(candidate, dev)]
    for case in range(12):
        seed.fill_(case)
        if moe:
            ids.copy_((torch.arange(topk, device=dev, dtype=ids_dtype) + case * 11) % 128)
        for graph in graphs:
            graph.replay()
        torch.cuda.synchronize(dev)
        if not (_same(yb[0][0], yb[1][0]) and _same(yb[0][0], yb[2][0]) and _same(yb[0][0], yb[3][0]) and _same(rb[0][0], rb[1][0])):
            return False
        if not bool(torch.all(xb[:8] == .125) & torch.all(xb[-8:] == .125)):
            return False
    return True


def _time(p, ps, mode, gen, ids_dtype=torch.int32):
    moe, n, k, topk, bn, bk, wn, wg, evict, gated, pdl, word, g8 = mode
    dev = p.planes[0].device
    slots = topk + 2 if moe else 1
    x = torch.randn((1, k), device=dev, generator=gen).to(torch.bfloat16)
    ids = torch.arange(topk, device=dev, dtype=ids_dtype) if moe else None
    gate = torch.zeros(3, device=dev, dtype=torch.int32)
    outputs = [torch.empty((slots, n), device=dev, dtype=torch.bfloat16) for _ in range(2)]
    graphs = {}
    for which in (0, 1):
        for repeats in (1, 16):
            def call():
                for _ in range(repeats):
                    if which:
                        _launch(p, ps, x, ids, outputs[which], gate, mode, new=True)
                    else:
                        _original(p, ps, x, ids, outputs[which], gate, mode)
            graphs[(which, repeats)] = _capture(call, dev)
    flush = torch.empty(64 << 20, device=dev, dtype=torch.uint8)
    times = {(w, r): [] for w in (0, 1) for r in (1, 16)}
    starts = (0, 41, 83, 127) if moe else (0,)
    for first in starts:
        if moe:
            ids.copy_((torch.arange(topk, device=dev, dtype=ids_dtype) + first) % 128)
        for repeats in (1, 16):
            samples = {0: [], 1: []}
            for trial in range(7):
                for which in ((0, 1) if trial % 2 == 0 else (1, 0)):
                    if repeats == 1:
                        flush.zero_()
                    else:
                        graphs[(which, repeats)].replay()
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    graphs[(which, repeats)].replay()
                    b.record()
                    b.synchronize()
                    samples[which].append(a.elapsed_time(b) / repeats)
            for which in (0, 1):
                median = statistics.median(samples[which])
                if median <= 0:
                    return False
                times[(which, repeats)].append(median)
    ratios = [b / a for r in (1, 16) for a, b in zip(times[(0, r)], times[(1, r)])]
    totals = [sum(times[(1, r)]) / sum(times[(0, r)]) for r in (1, 16)]
    if max(totals) >= .97 or statistics.median(ratios) >= .97 or max(ratios) >= 1.03:
        return False
    if gated:
        gate.fill_(1)
        stopped = {0: [], 1: []}
        for trial in range(7):
            for which in ((0, 1) if trial % 2 == 0 else (1, 0)):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                graphs[(which, 16)].replay()
                b.record()
                b.synchronize()
                stopped[which].append(a.elapsed_time(b) / 16)
        ref, cand = statistics.median(stopped[0]), statistics.median(stopped[1])
        if cand > ref * 1.02 + .001:  # milliseconds: at most1us additive overhead
            return False
    logger.info("MTP block128 whole projection admission mode=%s cold/warm=%s", mode, totals)
    return True


def _eligible(p, original):
    if (not _ENABLED or not original.is_cuda or original.dtype != torch.bfloat16
            or not original.is_contiguous() or original.shape[-1] % 128
            or torch.cuda.is_current_stream_capturing()
            or torch.cuda.get_device_capability(original.device) != (12, 1)
            or any(t.device != original.device or not t.is_contiguous() for t in p.planes)
            or p.planes[1].data_ptr() % 4):
        return False
    return True


@torch.no_grad()
def prepare_linear(p, original):
    if tuple(original.shape) not in ((4608, 2688), (2688, 4096)):
        return
    from .nemotron_h_mtp_packed import word_load_enabled, group8_enabled
    from .nemotron_h_mtp_kernels import mtp_pdl
    try:
        if not _eligible(p, original) or not word_load_enabled(original.device):
            return
        n, k = original.shape
        if not _all_bits(p, original, 4, 32):
            return
        word, g8, pdl = word_load_enabled(original.device), group8_enabled(original.device, n, k), mtp_pdl()
        gen = torch.Generator(device=original.device).manual_seed(2026128)
        for gated in (False, True):
            mode = (False, n, k, 0, 8, 512, 2, 4, True, gated, pdl, word, g8)
            key = (_planes_key(p), mode)
            _LINEAR[key] = False
            if _check(p, p, mode, gen) and _time(p, p, mode, gen):
                _LINEAR[key] = _refs(p)
    except Exception as exc:
        # No partially qualified choice survives a layer's admission failure.
        for key in list(_LINEAR):
            if key[0] == _planes_key(p):
                _LINEAR.pop(key, None)
        logger.warning("MTP block128 linear retained original: %r", exc)


@torch.no_grad()
def prepare_up(p, ps, original, shared):
    from .nemotron_h_mtp_packed import word_load_enabled
    from .nemotron_h_mtp_kernels import mtp_pdl, _EVICT_W, _PK_UP_BN, _PK_UP_WN, _MOE_UP_BK, _MOE_UP_STAGES
    try:
        if (tuple(original.shape) != (128, 1856, 2688) or tuple(shared.shape) != (3712, 2688)
                or not _eligible(p, original) or not _eligible(ps, shared)
                or not word_load_enabled(original.device)
                or (_PK_UP_BN, _PK_UP_WN, _MOE_UP_BK, _MOE_UP_STAGES) != (2, 2, 1024, 3)):
            return
        if not (_all_bits(p, original, 8, 16) and _all_bits(ps, shared, 8, 16)):
            return
        gen = torch.Generator(device=original.device).manual_seed(20261281856)
        for topk in (3, 6):
            for gated in (False, True):
                mode = (True, 1856, 2688, topk, 2, 1024, 2, 8, _EVICT_W,
                        gated, mtp_pdl(), word_load_enabled(original.device), True)
                key = (_planes_key(p), _planes_key(ps), mode)
                _UP[key] = False
                if all(_check(p, ps, mode, gen, ids_dtype=dt)
                       and _time(p, ps, mode, gen, ids_dtype=dt)
                       for dt in (torch.int32, torch.int64)):
                    _UP[key] = _refs(p, ps)
    except Exception as exc:
        for key in list(_UP):
            if key[:2] == (_planes_key(p), _planes_key(ps)):
                _UP.pop(key, None)
        logger.warning("MTP block128 up retained original: %r", exc)


def maybe_linear(p, x, y, gate, gated, pdl, word, g8):
    if not _ENABLED or p.shape not in ((4608, 2688), (2688, 4096)):
        return False
    n, k = p.shape
    mode = (False, n, k, 0, 8, 512, 2, 4, True, gated, pdl, word, g8)
    if (not _matches(_LINEAR.get((_planes_key(p), mode)), p)
            or x.shape != (1, k) or y.shape != (1, n)
            or x.dtype != y.dtype or x.dtype != torch.bfloat16
            or not x.is_contiguous() or not y.is_contiguous()
            or x.device != y.device or x.device != p.planes[0].device):
        return False
    _launch(p, p, x, None, y, gate, mode, new=True)
    return True


def maybe_up(p, ps, x, ids, out, gate, gated, pdl, word, evict, bn, bk, wn, stages):
    topk = ids.numel()
    mode = (True, 1856, 2688, topk, bn, bk, wn, 4 * wn, evict, gated, pdl, word, True)
    if (not _ENABLED or stages != 3 or topk not in (3, 6)
            or not _matches(_UP.get((_planes_key(p), _planes_key(ps), mode)), p, ps)
            or tuple(p.shape) != (128, 1856, 2688) or tuple(ps.shape) != (3712, 2688)
            or x.shape != (1, 2688) or out.shape != (topk + 2, 1856)
            or x.dtype != out.dtype or x.dtype != torch.bfloat16
            or ids.dtype not in (torch.int32, torch.int64)
            or not all(t.device == x.device and t.is_contiguous() for t in (x, ids, out))
            or x.device != p.planes[0].device or x.device != ps.planes[0].device):
        return False
    _launch(p, ps, x, ids, out, gate, mode, new=True)
    return True

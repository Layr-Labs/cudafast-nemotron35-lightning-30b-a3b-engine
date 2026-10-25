# SPDX-License-Identifier: Apache-2.0
"""Optional grouped-plane and indirect-read pipeline for compact MTP down.

Static development only. All compilation, actual-weight bit admission, changing
graph replay and cold whole-path timing execute on the official remote worker.
Original BF16 and admitted packed slots remain the production fallback.
"""
import os
import statistics

import torch
from vllm.triton_utils import tl, triton

_ENABLED = os.environ.get("MTP_DOWN_PIPELINE", "1") != "0"
_CACHE = {}


@triton.jit
def _grouped_weight(SM, EX, BASE, RI, RAW, index0, live, J: tl.constexpr):
    # H, expert stride and each BN tile are multiples of 128/8. Thus all
    # eight adjacent output columns share one codec block and exponent word.
    bid = index0 // 128
    rawid = tl.load(RI + bid, mask=live, other=-1, eviction_policy="evict_first")
    coded = live & (rawid < 0)
    exponent = tl.load(EX.to(tl.pointer_type(tl.uint32)) + index0 // 8,
                       mask=coded, other=0, eviction_policy="evict_first")
    base = tl.load(BASE + bid, mask=coded, other=1,
                   eviction_policy="evict_first").to(tl.uint16)
    j = tl.arange(0, J)
    index = index0[:, :, None] + j[None, None, :]
    sm = tl.load(SM + index, mask=coded[:, :, None], other=0,
                 eviction_policy="evict_first").to(tl.uint16)
    code = ((exponent[:, :, None] >> (4 * j[None, None, :])) & 15).to(tl.uint16)
    exp = tl.where(code == 0, 0, base[:, :, None] + code - 1)
    bits = ((sm & 128) << 8) | (exp << 7) | (sm & 127)
    raw = tl.load(RAW + rawid[:, :, None].to(tl.int64) * 128
                  + index % 128,
                  mask=(live & (rawid >= 0))[:, :, None], other=0,
                  eviction_policy="evict_first")
    return tl.where(rawid[:, :, None] >= 0, raw, bits).to(tl.uint16).to(
        tl.bfloat16, bitcast=True)


@triton.jit
def _chunk(indices, values, count, slot, offset, n0, lane, start,
           SM, EX, BASE, RI, RAW, H: tl.constexpr, I: tl.constexpr,
           BK: tl.constexpr, BN: tl.constexpr):
    pos = start + lane
    live = pos < count
    k = tl.load(indices + slot * I + pos, mask=live, other=0)
    value = tl.load(values + slot * I + pos, mask=live, other=0.0)
    index0 = offset + k[:, None].to(tl.int64) * H + n0[None, :]
    weight = _grouped_weight(SM, EX, BASE, RI, RAW, index0,
                             live[:, None] & (n0[None, :] < H), J=8)
    # Return to precisely the original [BK, BN] per-lane arithmetic shape.
    return value, tl.reshape(weight, (BK, BN))


@triton.jit
def _slots(indices, values, counts,
           RSM, REX, RBASE, RRI, RRAW, SSM, SEX, SBASE, SRI, SRAW,
           ids, part, H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
           BN: tl.constexpr, BK: tl.constexpr, LOOK: tl.constexpr,
           launch_pdl: tl.constexpr = False):
    tl.static_assert(H % 128 == 0 and BN % 8 == 0 and 128 % BN == 0)
    tl.static_assert(LOOK == 0 or LOOK == 1 or LOOK == 2)
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    n0 = tl.program_id(0) * BN + 8 * tl.arange(0, BN // 8)
    slot = tl.program_id(1)
    lane = tl.arange(0, BK)
    count = tl.load(counts + slot)
    routed = slot < TOPK
    if routed:
        offset = tl.load(ids + slot).to(tl.int64) * I * H
    else:
        offset = (slot - TOPK).to(tl.int64) * I * H
    acc = tl.zeros((BK, BN), tl.float32)
    # Only reads move ahead. There is a single original FP32 accumulator;
    # every product is added at its original ascending compacted-K position.
    if LOOK:
        if routed:
            v0, w0 = _chunk(indices, values, count, slot, offset, n0, lane, 0,
                            RSM, REX, RBASE, RRI, RRAW, H=H, I=I, BK=BK, BN=BN)
        else:
            v0, w0 = _chunk(indices, values, count, slot, offset, n0, lane, 0,
                            SSM, SEX, SBASE, SRI, SRAW, H=H, I=I, BK=BK, BN=BN)
        if LOOK == 2:
            if routed:
                v1, w1 = _chunk(indices, values, count, slot, offset, n0, lane, BK,
                                RSM, REX, RBASE, RRI, RRAW, H=H, I=I, BK=BK, BN=BN)
            else:
                v1, w1 = _chunk(indices, values, count, slot, offset, n0, lane, BK,
                                SSM, SEX, SBASE, SRI, SRAW, H=H, I=I, BK=BK, BN=BN)
    for start in range(0, count, BK):
        if LOOK:
            if routed:
                vn, wn = _chunk(indices, values, count, slot, offset, n0, lane,
                                start + LOOK * BK, RSM, REX, RBASE, RRI, RRAW,
                                H=H, I=I, BK=BK, BN=BN)
            else:
                vn, wn = _chunk(indices, values, count, slot, offset, n0, lane,
                                start + LOOK * BK, SSM, SEX, SBASE, SRI, SRAW,
                                H=H, I=I, BK=BK, BN=BN)
            acc += w0.to(tl.float32) * v0[:, None].to(tl.float32)
            if LOOK == 2:
                v0, w0 = v1, w1
                v1, w1 = vn, wn
            else:
                v0, w0 = vn, wn
        else:
            if routed:
                value, weight = _chunk(indices, values, count, slot, offset, n0,
                                       lane, start, RSM, REX, RBASE, RRI, RRAW,
                                       H=H, I=I, BK=BK, BN=BN)
            else:
                value, weight = _chunk(indices, values, count, slot, offset, n0,
                                       lane, start, SSM, SEX, SBASE, SRI, SRAW,
                                       H=H, I=I, BK=BK, BN=BN)
            acc += weight.to(tl.float32) * value[:, None].to(tl.float32)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    tl.store(part + slot * H + n, tl.sum(acc, axis=0), mask=n < H)


def _run(pw, ps, indices, values, counts, ids, part, config, pdl):
    bn, look = config
    _slots[(triton.cdiv(2688, bn), ids.numel() + 2)](
        indices, values, counts, *pw.planes, *ps.planes, ids, part,
        H=2688, I=1856, TOPK=ids.numel(), BN=bn, BK=64, LOOK=look,
        num_warps=4, num_stages=3, launch_pdl=pdl)


def _bits(a, b):
    return torch.equal(a.contiguous().view(torch.uint8),
                       b.contiguous().view(torch.uint8))


@torch.no_grad()
def prepare(packed):
    if not _ENABLED or torch.cuda.is_current_stream_capturing():
        return
    from . import nemotron_h_mtp_down_packed as original
    from .nemotron_h_mtp_compact import _compact, _down_slots, _down_combine
    from .nemotron_h_mtp_kernels import mtp_pdl
    w, sd = packed
    key = (w.data_ptr(), sd.data_ptr())
    if key in _CACHE:
        return
    _CACHE[key] = (w, sd, None, None, {})
    # Reuse only the existing immutable pack; no second encoded copy or
    # replacement of the entire highest source's original admission.
    entry = original._PACKS.get(key)
    if (entry is None or entry[0] is not w or entry[1] is not sd
            or entry[2] is None or entry[3] is None
            or tuple(w.shape) != (128, 1856, 2688)
            or tuple(sd.shape) != (2, 1856, 2688)):
        return
    pw, ps = entry[2], entry[3]
    pdl = mtp_pdl()
    choices = [(bn, look) for bn in (32, 64) for look in (0, 1, 2)]
    flags = {}
    try:
        gen = torch.Generator(device=w.device).manual_seed(6408128)
        for topk in (3, 6):
            slots = topk + 2
            h = torch.empty((slots, 1856), dtype=torch.bfloat16, device=w.device)
            indices = torch.empty(h.shape, dtype=torch.int32, device=w.device)
            values = torch.empty_like(h)
            counts = torch.empty(slots, dtype=torch.int32, device=w.device)
            ids = torch.empty(topk, dtype=torch.int32, device=w.device)
            old = torch.empty((slots, 2688), dtype=torch.float32, device=w.device)
            new = torch.empty_like(old)
            allowed = set(choices)

            def compact():
                _compact[(slots,)](h, indices, values, counts, I=1856, BLOCK=2048,
                                    num_warps=4, launch_pdl=pdl)

            def reference():
                _down_slots[(84, slots)](indices, values, counts, w, sd, ids, old,
                    H=2688, I=1856, TOPK=topk, BN=32, BK=64, num_warps=4,
                    num_stages=3, launch_pdl=pdl)

            def current():
                if not original.maybe_slots(packed, indices, values, counts,
                                             ids, old, pdl):
                    reference()

            def set_case(first, density, scale, special=0):
                ids.copy_((torch.arange(topk, device=w.device, dtype=torch.int32)
                           + first) % 128)
                h.copy_((torch.randn(h.shape, device=w.device, generator=gen)
                         .relu().square() * scale).to(torch.bfloat16))
                h.mul_(torch.rand(h.shape, device=w.device, generator=gen) < density)
                h.view(torch.int16)[:, :4] = torch.tensor(
                    [0, -32768, 1, -32767], dtype=torch.int16, device=w.device)
                if special == 1:
                    h.zero_()
                elif special == 2:
                    h.view(torch.int16)[:, 4:6] = torch.tensor(
                        [32640, -128], dtype=torch.int16, device=w.device)
                elif special == 3:
                    h.view(torch.int16)[:, 4:6] = torch.tensor(
                        [32704, -64], dtype=torch.int16, device=w.device)

            # Every routed expert and both shared tensors, every output FP32
            # byte, with finite scales/densities and separate nonfinite cases.
            for first in range(0, 128, topk):
                for density in (0.015625, 0.125, 0.5, 1.0):
                    for scale in (0.001, 1.0, 32.0):
                        set_case(first, density, scale)
                        compact()
                        reference()
                        for cfg in tuple(allowed):
                            _run(pw, ps, indices, values, counts, ids, new, cfg, pdl)
                            if not _bits(old, new):
                                allowed.remove(cfg)
                if not allowed:
                    break
            for special in (1, 2, 3):
                set_case(121, 0.5, 1.0, special)
                compact()
                reference()
                for cfg in tuple(allowed):
                    _run(pw, ps, indices, values, counts, ids, new, cfg, pdl)
                    if not _bits(old, new):
                        allowed.remove(cfg)
            if not allowed:
                continue

            weights = torch.full((topk,), 1.0 / topk, dtype=torch.float32,
                                 device=w.device)
            residual = torch.zeros((5, 2688), dtype=torch.bfloat16, device=w.device)
            y = torch.empty_like(residual)
            expected = torch.empty_like(y)

            def combine(part, out):
                _down_combine[(11,)](part, weights, out, M=5, H=2688,
                    ROUTED_SCALE=2.5, TOPK=topk, S=2, BN=256, num_warps=4,
                    launch_pdl=pdl, residual_ptr=residual, ADD_RES=True)

            # One full compact->gather/decode->fold/Add graph for each choice.
            # Compaction's PDL early signal exercises current mutable counts,
            # indices and values. Replay changes every route/input/residual.
            graphs = {}
            for cfg in (None, *sorted(allowed)):
                set_case(0, 0.5, 1.0)
                for _ in range(2):
                    compact()
                    if cfg is None:
                        current()
                        combine(old, y)
                    else:
                        _run(pw, ps, indices, values, counts, ids, new, cfg, pdl)
                        combine(new, y)
                torch.cuda.synchronize(w.device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
                    compact()
                    if cfg is None:
                        current()
                        combine(old, y)
                    else:
                        _run(pw, ps, indices, values, counts, ids, new, cfg, pdl)
                        combine(new, y)
                valid = True
                for case in range(12):
                    set_case((11 * case) % 128, (0.0, 0.015625, 0.5, 1.0)[case % 4],
                             (0.001, 1.0, 32.0)[case % 3], case % 4)
                    residual.copy_(torch.randn(residual.shape, device=w.device,
                                               generator=gen).to(torch.bfloat16))
                    weights.copy_(torch.rand(topk, device=w.device, generator=gen))
                    weights.div_(weights.sum())
                    graph.replay()
                    actual = y.clone()
                    compact()
                    reference()
                    combine(old, expected)
                    if not _bits(actual, expected):
                        valid = False
                        break
                if valid:
                    graphs[cfg] = graph
            if None not in graphs or len(graphs) < 2:
                continue
            flush = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=w.device)
            times = {cfg: [] for cfg in graphs}
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for density in (0.015625, 0.125, 0.5, 1.0):
                set_case(59, density, 1.0)
                for trial in range(7):
                    order = list(graphs)
                    if trial % 2:
                        order.reverse()
                    for cfg in order:
                        flush.zero_()
                        begin.record()
                        graphs[cfg].replay()
                        end.record()
                        end.synchronize()
                        times[cfg].append(begin.elapsed_time(end))
            fastest = min(times, key=lambda cfg: statistics.median(times[cfg]))
            if (fastest is not None and statistics.median(times[fastest])
                    < 0.96 * statistics.median(times[None])):
                flags[topk] = fastest
        if flags:
            _CACHE[key] = (w, sd, pw, ps, flags)
        print(f"MTP_DOWN_PIPELINE admitted={flags}; original fallback retained", flush=True)
    except Exception as exc:
        _CACHE[key] = (w, sd, None, None, {})
        print(f"MTP_DOWN_PIPELINE original slots retained: {exc!r}", flush=True)


def maybe_slots(packed, indices, values, counts, ids, part, pdl):
    if not _ENABLED:
        return False
    w, sd = packed
    entry = _CACHE.get((w.data_ptr(), sd.data_ptr()))
    if entry is None or entry[0] is not w or entry[1] is not sd:
        return False
    config = entry[4].get(ids.numel())
    if config is None:
        return False
    _run(entry[2], entry[3], indices, values, counts, ids, part, config, pdl)
    return True

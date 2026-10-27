# SPDX-License-Identifier: Apache-2.0
"""Official-remote-only actual-pkslots admission for split metadata/payload."""
from __future__ import annotations

import statistics

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
_SENTINEL = 0x2BAD
I, H = 1856, 2688


def _bits(t):
    return t.contiguous().reshape(-1).view(torch.uint8)


def _same(a, b):
    return torch.equal(_bits(a), _bits(b))


def _guarded(shape, dtype, device):
    size = 1
    for x in shape:
        size *= x
    pad = 256 // torch.empty((), dtype=dtype).element_size()
    b = torch.empty(size + 2 * pad, dtype=dtype, device=device)
    if dtype == torch.bfloat16:
        b.view(torch.int16).fill_(_SENTINEL)
    else:
        b.view(torch.int32).fill_(_SENTINEL)
    return b, b[pad:-pad].view(shape), pad


def _guards(b, pad):
    v = b.view(torch.int16 if b.dtype == torch.bfloat16 else torch.int32)
    return bool(torch.all(v[:pad] == _SENTINEL)) and bool(torch.all(v[-pad:] == _SENTINEL))


@torch.no_grad()
def qualify(api, pw, ps, topk, pdl):
    """Only after the original actual C kernel passed its original _slots gate.

    Every real routed expert and shared half is exercised with full rows, plus
    partial/empty boundaries. All inputs here are independently generated.
    """
    if (not getattr(api, "raw_available", lambda: False)()
            or topk not in (3, 6) or torch.cuda.is_current_stream_capturing()
            or tuple(pw.shape) != (128, I, H) or tuple(ps.shape) != (2, I, H)):
        return False
    device = pw.planes[0].device
    if torch.cuda.get_device_capability(device) != (12, 1):
        return False
    immutable = [(x, x.clone()) for x in (*pw.planes, *ps.planes)]
    slots = topk + 2
    ib, indices, ip = _guarded((slots, I), torch.int32, device)
    vb, values, vp = _guarded((slots, I), torch.bfloat16, device)
    cb, counts, cp = _guarded((slots,), torch.int32, device)
    db, ids, dp = _guarded((topk,), torch.int32, device)
    ob, old, op = _guarded((slots, H), torch.float32, device)
    nb, new, np = _guarded((slots, H), torch.float32, device)
    gen = torch.Generator(device=device).manual_seed(271856)
    pressure = torch.empty(64 << 20, device=device, dtype=torch.uint8)
    timing, ratios = [], []

    def case(first, count, mode=0):
        ids.copy_((torch.arange(topk, device=device, dtype=torch.int32) + first) % 128)
        # Full rows in both dense modes cover EVERY actual raw/coded block
        # for every expert and both shared halves, not only a sampled prefix.
        order = torch.arange(I, device=device, dtype=torch.int32)
        if mode not in (1, 2):
            order = (order + 91 + first) % I
        indices.copy_(order.view(1, I).expand(slots, I))
        if mode == 1:
            values.fill_(1.)
        else:
            values.copy_(torch.randn((slots, I), device=device, generator=gen))
            values.view(torch.int16)[:, :4] = torch.tensor(
                [0, -32768, 1, -32767], device=device, dtype=torch.int16)
            if mode == 3:
                values.view(torch.int16)[:, 4:6] = torch.tensor(
                    [32640, -128], device=device, dtype=torch.int16)
            if mode == 4:
                values.view(torch.int16)[:, 4:6] = torch.tensor(
                    [32704, -64], device=device, dtype=torch.int16)
        counts.fill_(count)
        if mode == 0 and count not in (0, I):
            counts.copy_((count - torch.arange(slots, device=device, dtype=torch.int32) * 7)
                         .clamp(min=0))

    def run(which):
        reached = api.run(indices, values, counts, pw.planes, ps.planes, ids,
                          old if which == 0 else new, pdl, raw_pipeline=bool(which))
        if not reached:
            raise ValueError("actual original/new pkslots contract not reached")

    def compare(graphs=None):
        snapshots = tuple(x.clone() for x in (ib, vb, cb, db))
        ob.view(torch.int32).fill_(_SENTINEL)
        nb.view(torch.int32).fill_(_SENTINEL)
        for which in (0, 1):
            graphs[which].replay() if graphs else run(which)
        if (not _same(ob, nb)
                or any(not _same(a, b) for a, b in zip((ib, vb, cb, db), snapshots))
                or not all(_guards(b, pad) for b, pad in
                           ((ib, ip), (vb, vp), (cb, cp), (db, dp), (ob, op), (nb, np)))):
            raise ValueError("raw pipeline changed full part/guards/read-only private inputs")

    def unchanged():
        if any(not _same(a, b) for a, b in immutable):
            raise ValueError("raw pipeline changed packed owners")

    try:
        for first in range(0, 128, topk):
            for count in (0, 1, 63, 64, 65, 511, 1855):
                case(first, count)
                compare()
            for mode in (1, 2):
                case(first, I, mode)
                compare()
            if first == 0:
                for mode in (3, 4):
                    case(first, I, mode)
                    compare()
                    compare()
        unchanged()
        case(0, 928)
        graphs = []
        for which in (0, 1):
            for _ in range(2):
                run(which)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
                run(which)
            graphs.append(graph)
        # Fixed pointers, changed expert IDs, row order, values and counts.
        for first, count, mode in ((121, 0, 0), (17, 1, 0), (125, 64, 0),
                                   (43, 65, 0), (3, I, 1), (91, 688, 2), (0, I, 4)):
            case(first, count, mode)
            compare(graphs)
            compare(graphs)
        for count in (256, 688, 928, I):
            case(37, count, 2)
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
                        repetitions = 1 if cold else 16
                        start.record()
                        for _ in range(repetitions):
                            graphs[which].replay()
                        end.record()
                        end.synchronize()
                        value = start.elapsed_time(end) / repetitions
                        if not value > 0:
                            return False
                        samples[which].append(value)
                a, b = (statistics.median(s) for s in samples)
                timing.append((a, b))
                ratios.append(b / a)
                compare(graphs)
        unchanged()
        admitted = (bool(ratios) and statistics.median(ratios) < .995
                    and sum(b for a, b in timing) / sum(a for a, b in timing) < .995
                    and max(ratios) < 1.01)
        logger.info("MTP down: metadata/raw pipeline admitted=%s topk=%d", admitted, topk)
        return admitted
    except Exception as exc:
        logger.warning("MTP down: metadata/raw pipeline unavailable (%r); original pkslots kept", exc)
        return False
    finally:
        for actual, saved in immutable:
            actual.copy_(saved)

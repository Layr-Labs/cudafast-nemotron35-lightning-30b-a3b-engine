"""Private, exact decoder broadcast for the native MTP compact down slots.

Same original lossless planes and FP32 slot reduction. Qualification is solely
at official remote weight load; no request data or output is retained.
"""
import os
import statistics

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)
_ENABLED = os.environ.get("MTP_DOWN_BROADCAST", "1") != "0"
_CHOICES = {}


@triton.jit
def _weights(SM, EX, BASE, RI, RAW, base, k, valid, n0,
             H: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # H=21*128, BN=32/64 and n0 is BN-aligned: every output tile
    # stays within one 128-value metadata block per selected K row.
    row = base + k.to(tl.int64) * H + n0
    bid = row // 128
    rawid = tl.load(RI + bid, mask=valid, other=-1, eviction_policy="evict_first")
    coded = valid & (rawid < 0)
    exponent_base = tl.load(BASE + bid, mask=coded, other=1,
                            eviction_policy="evict_first").to(tl.uint16)
    group = tl.arange(0, BN // 8) * 8
    sublane = tl.arange(0, 8)
    words = EX.to(tl.pointer_type(tl.uint32))
    word = tl.load(words + (row[:, None] + group[None, :]) // 8,
                   mask=coded[:, None], other=0, eviction_policy="evict_first")
    codes = ((word[:, :, None] >> (sublane[None, None, :].to(tl.uint32) * 4)) & 15)
    code = tl.reshape(codes, (BK, BN)).to(tl.uint16)
    col = tl.arange(0, BN)
    index = row[:, None] + col[None, :]
    mask = valid[:, None] & (n0 + col[None, :] < H)
    sm = tl.load(SM + index, mask=mask & (rawid[:, None] < 0), other=0,
                 eviction_policy="evict_first").to(tl.uint16)
    exponent = tl.where(code == 0, 0, exponent_base[:, None] + code - 1)
    bits = ((sm & 128) << 8) | (exponent << 7) | (sm & 127)
    raw = tl.load(RAW + rawid[:, None].to(tl.int64) * 128 + index % 128,
                  mask=mask & (rawid[:, None] >= 0), other=0,
                  eviction_policy="evict_first")
    bits = tl.where(rawid[:, None] >= 0, raw, bits)
    bits = tl.where(mask, bits, 0).to(tl.uint16)
    return bits.to(tl.bfloat16, bitcast=True)


@triton.jit
def _decode_bits(SM, EX, BASE, RI, RAW, original, bad,
                 ROWS: tl.constexpr, H: tl.constexpr, BN: tl.constexpr,
                 BK: tl.constexpr):
    k = tl.program_id(1) * BK + tl.arange(0, BK)
    n0 = tl.program_id(0) * BN
    n = n0 + tl.arange(0, BN)
    weight = _weights(SM, EX, BASE, RI, RAW, 0, k, k < ROWS, n0, H, BN, BK)
    expected = tl.load(original + k[:, None].to(tl.int64) * H + n[None, :],
                       mask=(k[:, None] < ROWS) & (n[None, :] < H), other=0)
    wrong = (weight.to(tl.uint16, bitcast=True) != expected)
    count = tl.sum(tl.sum(wrong.to(tl.int32), axis=1), axis=0)
    if count:
        tl.atomic_add(bad, count)


@triton.jit
def _slots(indices, values, counts,
           RSM, REX, RBASE, RRI, RRAW, SSM, SEX, SBASE, SRI, SRAW, ids, part,
           H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
           BN: tl.constexpr, BK: tl.constexpr,
           launch_pdl: tl.constexpr = False):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    n0 = tl.program_id(0) * BN
    n = n0 + tl.arange(0, BN)
    slot = tl.program_id(1)
    lane = tl.arange(0, BK)
    count = tl.load(counts + slot)
    if slot < TOPK:
        base = tl.load(ids + slot).to(tl.int64) * I * H
    else:
        base = (slot - TOPK).to(tl.int64) * I * H
    acc = tl.zeros((BK, BN), tl.float32)
    for start in range(0, count, BK):
        pos = start + lane
        k = tl.load(indices + slot * I + pos, mask=pos < count, other=0)
        value = tl.load(values + slot * I + pos, mask=pos < count, other=0.0)
        if slot < TOPK:
            weight = _weights(RSM, REX, RBASE, RRI, RRAW, base, k,
                              pos < count, n0, H, BN, BK)
        else:
            weight = _weights(SSM, SEX, SBASE, SRI, SRAW, base, k,
                              pos < count, n0, H, BN, BK)
        # Exactly the original lane accumulation and original axis-zero tree.
        acc += weight.to(tl.float32) * value[:, None].to(tl.float32)
    tl.store(part + slot * H + n, tl.sum(acc, axis=0), mask=n < H)


def _launch(pw, ps, indices, values, counts, ids, part, bn, pdl):
    _slots[(triton.cdiv(2688, bn), ids.numel() + 2)](
        indices, values, counts, *pw.planes, *ps.planes, ids, part,
        H=2688, I=1856, TOPK=ids.numel(), BN=bn, BK=64,
        num_warps=4, num_stages=3, launch_pdl=pdl)


def _equal(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _reference(w, sd, pw, ps, flags, word, indices, values, counts, ids, part, pdl):
    from .nemotron_h_mtp_compact import _down_slots
    from .nemotron_h_mtp_down_packed import _new
    bn = flags.get((ids.numel(), word))
    if bn:
        _new(pw, ps, indices, values, counts, ids, part, bn, word, pdl)
    else:
        _down_slots[(84, ids.numel() + 2)](
            indices, values, counts, w, sd, ids, part,
            H=2688, I=1856, TOPK=ids.numel(), BN=32, BK=64,
            num_warps=4, num_stages=3, launch_pdl=pdl)


def _buffers(device, topk):
    slots = topk + 2
    # All writable tensors have independent outer guards and untouched rows.
    ib = torch.full((slots * 1856 + 16,), -123, device=device, dtype=torch.int32)
    vb = torch.full((slots * 1856 + 16,), .0625, device=device, dtype=torch.bfloat16)
    cb = torch.full((slots + 16,), -123, device=device, dtype=torch.int32)
    pb = torch.full((slots * 2688 + 16,), .09375, device=device, dtype=torch.float32)
    yb = torch.full((5 * 2688 + 16,), .125, device=device, dtype=torch.bfloat16)
    return dict(ib=ib, vb=vb, cb=cb, pb=pb, yb=yb,
                indices=ib[8:-8].view(slots, 1856),
                values=vb[8:-8].view(slots, 1856), counts=cb[8:-8],
                part=pb[8:-8].view(slots, 2688), y=yb[8:-8].view(5, 2688))


def _chain(b, h, ids, tw, residual, w, sd, pw, ps, flags, word, bn, pdl):
    from .nemotron_h_mtp_compact import _compact, _down_combine
    _compact[(ids.numel() + 2,)](
        h, b["indices"], b["values"], b["counts"], I=1856, BLOCK=2048,
        num_warps=4, launch_pdl=pdl)
    if bn:
        _launch(pw, ps, b["indices"], b["values"], b["counts"], ids, b["part"], bn, pdl)
    else:
        _reference(w, sd, pw, ps, flags, word, b["indices"], b["values"],
                   b["counts"], ids, b["part"], pdl)
    _down_combine[(11,)](
        b["part"], tw, b["y"], M=5, H=2688, ROUTED_SCALE=2.5,
        TOPK=ids.numel(), S=2, BN=256, num_warps=4, launch_pdl=pdl,
        residual_ptr=residual, ADD_RES=residual is not None)


def _check(w, sd, pw, ps, flags, word, topk, bn, pdl, gen):
    dev = w.device
    bs = [_buffers(dev, topk) for _ in range(2)]
    slots = topk + 2
    for first in range(0, 128, topk):
        ids = ((torch.arange(topk, device=dev, dtype=torch.int32) + first) % 128).contiguous()
        tw = torch.arange(1, topk + 1, device=dev, dtype=torch.float32)
        tw /= tw.sum()
        for case, count in enumerate((0, 1, 63, 64, 65, 1855, 1856)):
            # Sorted K compaction, exact zeros including -0, odd chunk tails.
            h = torch.randn((slots, 1856), device=dev, generator=gen).to(torch.bfloat16)
            h[:, count:] = 0
            if count < 1856:
                h[:, count::2] = -0.0
            h *= (.001, 1., 32.)[case % 3]
            # Include tiny BF16 values without turning every finite sum to NaN.
            if count:
                h.view(torch.int16)[:, 0] = 1
            if first == 0 and count == 1856:
                h.view(torch.int16)[:, :4] = torch.tensor(
                    [32640, -128, 32704, -64], device=dev, dtype=torch.int16)
            for residual in (None, torch.randn((5, 2688), device=dev, generator=gen).to(torch.bfloat16)):
                for which, b in enumerate(bs):
                    # Compare identical complete backing including inactive data.
                    for name in ("ib", "cb"):
                        b[name].fill_(-123)
                    b["vb"].fill_(.0625)
                    b["pb"].fill_(.09375)
                    b["yb"].fill_(.125)
                    _chain(b, h, ids, tw, residual, w, sd, pw, ps, flags,
                           word, bn if which else 0, pdl)
                torch.cuda.synchronize(dev)
                if not all(_equal(bs[0][k], bs[1][k]) for k in ("ib", "vb", "cb", "pb", "yb")):
                    return False
    return True


def _time(w, sd, pw, ps, flags, word, topk, bn, pdl, gen):
    dev = w.device
    bs = [_buffers(dev, topk) for _ in range(2)]
    slots = topk + 2
    h = torch.randn((slots, 1856), device=dev, generator=gen).relu().square().to(torch.bfloat16)
    ids = torch.arange(topk, device=dev, dtype=torch.int32)
    tw = torch.full((topk,), 1 / topk, device=dev, dtype=torch.float32)
    residual = torch.randn((5, 2688), device=dev, generator=gen).to(torch.bfloat16)
    graphs = {}
    stream = torch.cuda.Stream(device=dev)
    stream.wait_stream(torch.cuda.current_stream())
    for which in (0, 1):
        for repeats in (1, 12):
            def call():
                _chain(bs[which], h, ids, tw, residual, w, sd, pw, ps,
                       flags, word, bn if which else 0, pdl)
            with torch.cuda.stream(stream):
                call()
                call()
            torch.cuda.synchronize(dev)
            graph = torch.cuda.CUDAGraph()
            pool = torch.cuda.graph_pool_handle()
            outer = torch.cuda.current_stream()
            captured = False
            try:
                with torch.cuda.graph(graph, stream=stream, pool=pool):
                    for _ in range(repeats):
                        call()
                captured = True
                graphs[(which, repeats)] = graph
            finally:
                torch.cuda.set_stream(outer)
                if not captured:
                    try:
                        torch._C._cuda_endAllocateToPool(dev.index, pool)
                    except Exception:
                        pass
    flush = torch.empty(64 << 20, device=dev, dtype=torch.uint8)
    times = {(which, repeats): [] for which in (0, 1) for repeats in (1, 12)}
    for first, density in ((0, .015625), (41, .125), (83, .5), (127, 1.)):
        ids.copy_((torch.arange(topk, device=dev, dtype=torch.int32) + first) % 128)
        value = torch.randn(h.shape, device=dev, generator=gen).relu().square()
        value *= torch.rand(h.shape, device=dev, generator=gen) < density
        h.copy_(value)
        for repeats in (1, 12):
            samples = {0: [], 1: []}
            for trial in range(7):
                for which in ((0, 1) if trial % 2 == 0 else (1, 0)):
                    if repeats == 1:
                        flush.zero_()
                    else:
                        graphs[(which, repeats)].replay()
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                    graphs[(which, repeats)].replay()
                    end.record()
                    end.synchronize()
                    samples[which].append(begin.elapsed_time(end) / repeats)
            for which in (0, 1):
                value = statistics.median(samples[which])
                if value <= 0:
                    return False
                times[(which, repeats)].append(value)
    ratios = [b / a for repeats in (1, 12)
              for a, b in zip(times[(0, repeats)], times[(1, repeats)])]
    totals = [sum(times[(1, r)]) / sum(times[(0, r)]) for r in (1, 12)]
    logger.info("MTP down broadcast bn=%s topk=%s cold/warm=%s ratios=%s", bn, topk, totals, ratios)
    return max(totals) < .97 and statistics.median(ratios) < .97 and max(ratios) < 1.03


@torch.no_grad()
def prepare(w, sd, pw, ps, flags, word, pdl):
    key = (w.data_ptr(), sd.data_ptr())
    _CHOICES.pop(key, None)
    if (not _ENABLED or not word or torch.cuda.is_current_stream_capturing()
            or torch.cuda.get_device_capability(w.device) != (12, 1)):
        return
    try:
        if (w.shape != (128, 1856, 2688) or sd.shape != (2, 1856, 2688)
                or w.dtype != torch.bfloat16 or sd.dtype != torch.bfloat16
                or not w.is_contiguous() or not sd.is_contiguous()
                or w.device != sd.device or pw is None or ps is None
                or any(t.device != w.device or not t.is_contiguous()
                       for p in (pw, ps) for t in p.planes)
                or any(p.planes[1].data_ptr() % 4 for p in (pw, ps))):
            return
        for packed, original in ((pw, w), (ps, sd)):
            for bn in (32, 64):
                bad = torch.zeros(1, device=w.device, dtype=torch.int32)
                _decode_bits[(triton.cdiv(2688, bn), triton.cdiv(original.numel() // 2688, 64))](
                    *packed.planes, original.view(torch.uint16), bad,
                    ROWS=original.numel() // 2688, H=2688, BN=bn, BK=64,
                    num_warps=4)
                if int(bad.item()):
                    return
        gen = torch.Generator(device=w.device).manual_seed(202610262688)
        choices = {}
        for topk in (3, 6):
            for bn in (32, 64):
                if _check(w, sd, pw, ps, flags, word, topk, bn, pdl, gen) and _time(
                        w, sd, pw, ps, flags, word, topk, bn, pdl, gen):
                    choices[topk] = bn
                    break
        if choices:
            _CHOICES[key] = (w, sd, pw, ps, pdl, choices)
        logger.info("MTP down metadata broadcast fixed choices=%s", choices)
    except Exception as exc:
        _CHOICES.pop(key, None)
        logger.warning("MTP down metadata broadcast retained original: %r", exc)


def maybe_slots(packed, indices, values, counts, ids, part, pdl):
    if not _ENABLED:
        return False
    w, sd = packed
    entry = _CHOICES.get((w.data_ptr(), sd.data_ptr()))
    if (entry is None or entry[0] is not w or entry[1] is not sd or entry[4] != pdl
            or ids.numel() not in entry[5]
            or part.shape != (ids.numel() + 2, 2688)
            or part.dtype != torch.float32 or not part.is_contiguous()
            or not all(t.device == w.device and t.is_contiguous()
                       for t in (indices, values, counts, ids, part))
            or indices.shape != (ids.numel() + 2, 1856)
            or values.shape != indices.shape or counts.shape != (ids.numel() + 2,)
            or indices.dtype != torch.int32 or values.dtype != torch.bfloat16
            or counts.dtype != torch.int32 or ids.dtype not in (torch.int32, torch.int64)):
        return False
    _launch(entry[2], entry[3], indices, values, counts, ids, part, entry[5][ids.numel()], pdl)
    return True

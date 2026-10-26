"""Optional lossless native-MTP compact-down weight storage.

Uses the inherited BF16 codec (3d51cab6-7079-4fb2-98e4-f2a238077897) with the
original compact slot FP32 tree (526d6e53-7b8b-4ce8-b4a1-b3c7a53be9d9).
All admission executes solely in the official remote resident at weight load.
"""
import os
import statistics

import torch
from vllm.triton_utils import tl, triton

_ENABLED = os.environ.get("MTP_DOWN_PACK12", "1") != "0"
_FORCE = os.environ.get("MTP_DOWN_PACK12_FORCE", "1") != "0"
_PACKS = {}


@triton.jit
def _load_packed(SM, EX, BASE, RI, RAW, index, mask, EVICT:tl.constexpr=True,
                WORD:tl.constexpr=False):
    bid = index // 128
    # Escape blocks are supplied entirely by RAW. Read their metadata first,
    # so they do not also stream unused sign/mantissa and exponent planes.
    rawid = tl.load(RI + bid, mask=mask, other=-1, eviction_policy='evict_first' if EVICT else '')
    coded = mask & (rawid < 0)
    # LOCAL PATCH (cudafast-nemotron35): plane loads use the bounds mask only (issued with RI).
    sm = tl.load(SM + index, mask=mask, other=0, eviction_policy='evict_first' if EVICT else '').to(tl.uint16)
    if WORD:
        words = EX.to(tl.pointer_type(tl.uint32))
        word = tl.load(words + index // 8, mask=mask, other=0,
                       eviction_policy='evict_first' if EVICT else '')
        code = ((word >> ((index % 8).to(tl.uint32) * 4)) & 15).to(tl.uint16)
    else:
        nib = tl.load(EX + index // 2, mask=mask, other=0,
                      eviction_policy='evict_first' if EVICT else '').to(tl.uint16)
        code = (nib >> ((index % 2)*4)) & 15
    base = tl.load(BASE + bid, mask=mask, other=1, eviction_policy='evict_first' if EVICT else '').to(tl.uint16)
    exp = tl.where(code == 0, 0, base + code - 1)
    bits = ((sm & 128) << 8) | (exp << 7) | (sm & 127)
    raw = tl.load(RAW + rawid.to(tl.int64)*128 + index % 128,
                  mask=mask & (rawid >= 0), other=0, eviction_policy='evict_first' if EVICT else '')
    bits = tl.where(rawid >= 0, raw, bits).to(tl.uint16)
    return bits.to(tl.bfloat16, bitcast=True)


@triton.jit
def _slots(indices, values, counts,
           RSM, REX, RBASE, RRI, RRAW, SSM, SEX, SBASE, SRI, SRAW, ids, part,
           H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
           BN: tl.constexpr, BK: tl.constexpr, WORD: tl.constexpr,
           launch_pdl: tl.constexpr = False):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    n = tl.program_id(0) * BN + tl.arange(0, BN)
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
        index = base + k[:, None].to(tl.int64) * H + n[None, :]
        mask = (pos[:, None] < count) & (n[None, :] < H)
        if slot < TOPK:
            weight = _load_packed(RSM, REX, RBASE, RRI, RRAW, index, mask, WORD=WORD)
        else:
            weight = _load_packed(SSM, SEX, SBASE, SRI, SRAW, index, mask, WORD=WORD)
        acc += weight.to(tl.float32) * value[:, None].to(tl.float32)
    tl.store(part + slot * H + n, tl.sum(acc, axis=0), mask=n < H)


def _new(pw, ps, indices, values, counts, ids, part, bn, word, pdl=False):
    _slots[(triton.cdiv(2688, bn), ids.numel() + 2)](
        indices, values, counts, *pw.planes, *ps.planes, ids, part,
        H=2688, I=1856, TOPK=ids.numel(), BN=bn, BK=64, WORD=word,
        num_warps=4, num_stages=3, launch_pdl=pdl)


@torch.no_grad()
def prepare(packed):
    if not _ENABLED or torch.cuda.is_current_stream_capturing():
        return
    w, sd = packed
    key = (w.data_ptr(), sd.data_ptr())
    if key in _PACKS:
        return
    flags = {}
    _PACKS[key] = (w, sd, None, None, flags)
    try:
        from .nemotron_h_mtp_packed import pack_bf16, word_load_enabled
        from .nemotron_h_mtp_compact import _compact, _down_slots
        from .nemotron_h_mtp_compact import _ENABLED as compact_enabled, _SLOTS
        from .nemotron_h_mtp_kernels import mtp_pdl
        pdl = mtp_pdl()
        if not compact_enabled or not _SLOTS:
            return
        if tuple(w.shape) != (128, 1856, 2688) or tuple(sd.shape) != (2, 1856, 2688):
            return
        pw, ps = pack_bf16(w), pack_bf16(sd)
        if pw is None or ps is None:
            return
        word = word_load_enabled(w.device)
        device = w.device
        gen = torch.Generator(device=device).manual_seed(12641856)
        admitted = {32: True, 64: True}
        for topk in (3, 6):
            slots = topk + 2
            indices = torch.empty((slots, 1856), dtype=torch.int32, device=device)
            values = torch.empty((slots, 1856), dtype=torch.bfloat16, device=device)
            counts = torch.empty(slots, dtype=torch.int32, device=device)
            old = torch.empty((slots, 2688), dtype=torch.float32, device=device)
            new = torch.empty_like(old)
            for first in range(0, 128, topk):
                ids = ((torch.arange(topk, device=device, dtype=torch.int32) + first) % 128).contiguous()
                for density in (0.015625, 0.125, 0.5, 1.0):
                    for scale in (0.001, 1., 32.):
                        h = (torch.randn((slots, 1856), device=device, generator=gen).relu().square() * scale).to(torch.bfloat16)
                        h *= (torch.rand(h.shape, device=device, generator=gen) < density)
                        # Finite cases remain finite: inserting NaN into every
                        # probe would hide all ordinary arithmetic differences.
                        h.view(torch.int16)[:, :4] = torch.tensor(
                            [0, -32768, 1, -32767], dtype=torch.int16, device=device)
                        if first == 0 and density == 1.0 and scale == 1.:
                            h.view(torch.int16)[:, 4:6] = torch.tensor(
                                [32640, -128], dtype=torch.int16, device=device)
                        if first == 0 and density == 1.0 and scale == 32.:
                            h.view(torch.int16)[:, 4:6] = torch.tensor(
                                [32704, -64], dtype=torch.int16, device=device)
                        _compact[(slots,)](h, indices, values, counts, I=1856, BLOCK=2048, num_warps=4)
                        _down_slots[(84, slots)](
                            indices, values, counts, w, sd, ids, old,
                            H=2688, I=1856, TOPK=topk, BN=32, BK=64, num_warps=4, num_stages=3, launch_pdl=pdl)
                        for bn in (32, 64):
                            if admitted[bn]:
                                _new(pw, ps, indices, values, counts, ids, new, bn, word, pdl)
                                if not torch.equal(old.view(torch.int32), new.view(torch.int32)):
                                    admitted[bn] = False
                if not any(admitted.values()):
                    break
            if not any(admitted.values()):
                break
        if not any(admitted.values()):
            print("MTP_DOWN_PACK12 original slots retained: private all-expert bits differ", flush=True)
            return
        flush = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for topk in (3, 6):
            slots = topk + 2
            indices = torch.empty((slots, 1856), dtype=torch.int32, device=device)
            values = torch.empty((slots, 1856), dtype=torch.bfloat16, device=device)
            counts = torch.empty(slots, dtype=torch.int32, device=device)
            old = torch.empty((slots, 2688), dtype=torch.float32, device=device)
            new = torch.empty_like(old)
            times = {0: [], **{bn: [] for bn, ok in admitted.items() if ok}}
            for density in (0.015625, 0.125, 0.5, 1.0):
                ids = torch.randperm(128, device=device, generator=gen)[:topk].to(torch.int32).contiguous()
                h = torch.randn((slots, 1856), device=device, generator=gen).relu().square().to(torch.bfloat16)
                h *= (torch.rand(h.shape, device=device, generator=gen) < density)
                _compact[(slots,)](h, indices, values, counts, I=1856, BLOCK=2048, num_warps=4)
                for trial in range(7):
                    order = list(times)
                    if trial % 2:
                        order.reverse()
                    for bn in order:
                        flush.zero_()
                        start.record()
                        if bn:
                            _new(pw, ps, indices, values, counts, ids, new, bn, word, pdl)
                        else:
                            _down_slots[(84, slots)](
                                indices, values, counts, w, sd, ids, old,
                                H=2688, I=1856, TOPK=topk, BN=32, BK=64, num_warps=4, num_stages=3, launch_pdl=pdl)
                        end.record()
                        end.synchronize()
                        times[bn].append(start.elapsed_time(end))
            best = min(times, key=lambda bn: statistics.median(times[bn]))
            if best and statistics.median(times[best]) < 0.96 * statistics.median(times[0]):
                flags[(topk, word)] = best
            elif _FORCE and any(bn for bn in times):
                # LOCAL PATCH (cudafast-nemotron35): MTP_DOWN_PACK12_FORCE admits the
                # bit-identical packed slots even when this private timing (sparse
                # probe densities) does not show the 4% margin.
                flags[(topk, word)] = min((bn for bn in times if bn),
                                          key=lambda bn: statistics.median(times[bn]))
        if flags:
            _PACKS[key] = (w, sd, pw, ps, flags)
        print(f"MTP_DOWN_PACK12 admitted={flags} original_weights_retained=True", flush=True)
    except Exception as exc:
        flags.clear()
        print(f"MTP_DOWN_PACK12 original slots retained: {exc!r}", flush=True)


def maybe_slots(packed, indices, values, counts, ids, part, pdl):
    if not _ENABLED:
        return False
    w, sd = packed
    entry = _PACKS.get((w.data_ptr(), sd.data_ptr()))
    if entry is None or entry[0] is not w or entry[1] is not sd:
        return False
    from .nemotron_h_mtp_packed import word_load_enabled
    word = word_load_enabled(w.device)
    bn = entry[4].get((ids.numel(), word))
    if bn is None:
        return False
    _new(entry[2], entry[3], indices, values, counts, ids, part, bn, word, pdl)
    return True

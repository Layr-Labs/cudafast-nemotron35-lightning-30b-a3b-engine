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
# LOCAL PATCH (cudafast-nemotron35): pkslots (harness/kernels/pkslots) runs the (64, 4) tile's
# exact arithmetic in CUDA; admitted per (weights, top-k, word) by prepare's bitwise comparison.
try:
    import pkslots as _pkslots
except ImportError:
    _pkslots = None
_PKSLOTS_OK = {}
_PKRAW_OK = {}


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
    # LOCAL PATCH (cudafast-nemotron35): software-pipelined gather. Chunk c + 1's row ids,
    # values and weight loads are issued before chunk c's products are added, so its gathered
    # rows are in flight during the math; the products are added in the same chunk order to
    # the same per-lane accumulators, so every output bit is unchanged.
    pos = lane
    k = tl.load(indices + slot * I + pos, mask=pos < count, other=0)
    value = tl.load(values + slot * I + pos, mask=pos < count, other=0.0)
    index = base + k[:, None].to(tl.int64) * H + n[None, :]
    mask = (pos[:, None] < count) & (n[None, :] < H)
    if slot < TOPK:
        weight = _load_packed(RSM, REX, RBASE, RRI, RRAW, index, mask, WORD=WORD)
    else:
        weight = _load_packed(SSM, SEX, SBASE, SRI, SRAW, index, mask, WORD=WORD)
    for start in range(BK, count + BK, BK):
        posn = start + lane
        kn = tl.load(indices + slot * I + posn, mask=posn < count, other=0)
        valuen = tl.load(values + slot * I + posn, mask=posn < count, other=0.0)
        indexn = base + kn[:, None].to(tl.int64) * H + n[None, :]
        maskn = (posn[:, None] < count) & (n[None, :] < H)
        if slot < TOPK:
            weightn = _load_packed(RSM, REX, RBASE, RRI, RRAW, indexn, maskn, WORD=WORD)
        else:
            weightn = _load_packed(SSM, SEX, SBASE, SRI, SRAW, indexn, maskn, WORD=WORD)
        acc += weight.to(tl.float32) * value[:, None].to(tl.float32)
        weight = weightn
        value = valuen
    tl.store(part + slot * H + n, tl.sum(acc, axis=0), mask=n < H)


# LOCAL PATCH (drafter3): column tiles (BN, warps) the load-time admission times. BN only
# changes which columns a program owns: every column keeps the same BK=64 chunk order and
# the same within-chunk reduction, so every tile gives the same bits (checked below). The
# original tiles were 32 and 64 columns (64-128 B of sign/mantissa per gathered row), too
# narrow for the gathered rows to stream at DRAM speed. MTP_DOWN_PACK12_WIDE=0 keeps them.
_TILES = ((32, 4), (64, 4)) + (
    ((128, 4), (128, 8), (256, 8)) if os.environ.get("MTP_DOWN_PACK12_WIDE", "1") != "0" else ())


# LOCAL PATCH (cudafast-nemotron35): wide tiles are admitted on bit equality only (default
# 1). The drafts feed the verify batch, whose FP8 activation scales are computed over all of
# its rows, so a draft that differs from the record drafter's can move a near-tie in the
# timed window; MTP_DOWN_PACK12_WIDE_EXACT=0 restores the 1e-5 relative admission.
_WIDE_EXACT = os.environ.get("MTP_DOWN_PACK12_WIDE_EXACT", "1") != "0"


def _same(old, new, tile):
    """The original tiles must match the original slots bit for bit. A wider tile owns
    more columns per program, which can change the fp32 tree of tl.sum over the 64-row
    chunk (Triton's layout); with MTP_DOWN_PACK12_WIDE_EXACT=0 a wide tile is admitted
    within 1e-5 relative (same NaN/inf pattern), by default only on bit equality."""
    if torch.equal(old.view(torch.int32), new.view(torch.int32)):
        return True
    bn = tile[0] if isinstance(tile, tuple) else tile
    if bn <= 64 or _WIDE_EXACT:
        return False
    fin_o, fin_n = torch.isfinite(old), torch.isfinite(new)
    if not torch.equal(fin_o, fin_n):
        return False
    if not torch.equal(old[~fin_o].nan_to_num(0.0, 1.0, -1.0), new[~fin_n].nan_to_num(0.0, 1.0, -1.0)):
        return False
    o, n = old[fin_o].double(), new[fin_n].double()
    scale = o.abs().max().clamp_min(1e-30)
    return bool(((o - n).abs() <= 1e-5 * (o.abs() + 1e-3 * scale)).all())


def _new(pw, ps, indices, values, counts, ids, part, tile, word, pdl=False):
    bn, warps = tile if isinstance(tile, tuple) else (tile, 4)
    _slots[(triton.cdiv(2688, bn), ids.numel() + 2)](
        indices, values, counts, *pw.planes, *ps.planes, ids, part,
        H=2688, I=1856, TOPK=ids.numel(), BN=bn, BK=64, WORD=word,
        num_warps=warps, num_stages=3, launch_pdl=pdl)


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
        admitted = {tile: True for tile in _TILES}
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
                        for tile in _TILES:
                            if admitted[tile]:
                                _new(pw, ps, indices, values, counts, ids, new, tile, word, pdl)
                                if not _same(old, new, tile):
                                    admitted[tile] = False
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
            times = {0: [], **{tile: [] for tile, ok in admitted.items() if ok}}
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
                        if bn:  # a (BN, warps) tile; 0 is the original BF16 slots kernel
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
            _admit_pkslots(key, w, sd, pw, ps, flags, word, pdl)
        print(f"MTP_DOWN_PACK12 admitted={flags} original_weights_retained=True", flush=True)
    except Exception as exc:
        flags.clear()
        print(f"MTP_DOWN_PACK12 original slots retained: {exc!r}", flush=True)


@torch.no_grad()
def _admit_pkslots(key, w, sd, pw, ps, flags, word, pdl):
    """pkslots for each top-k whose admitted tile is (64, 4), when its fp32 output words equal
    _slots' on every probe: every expert in groups of top-k, four densities, three scales,
    signed zeros and subnormals in every probe, infinities and NaN in two, and empty slots."""
    if _pkslots is None or not _pkslots.available():
        return
    from .nemotron_h_mtp_compact import _compact
    try:
        device = w.device
        gen = torch.Generator(device=device).manual_seed(64464)
        for (topk, wd), tile in list(flags.items()):
            if tile != (64, 4) or wd != word:
                continue
            slots = topk + 2
            indices = torch.empty((slots, 1856), dtype=torch.int32, device=device)
            values = torch.empty((slots, 1856), dtype=torch.bfloat16, device=device)
            counts = torch.empty(slots, dtype=torch.int32, device=device)
            old = torch.empty((slots, 2688), dtype=torch.float32, device=device)
            new = torch.empty_like(old)
            ok = True
            for first in range(0, 128, topk):
                ids = ((torch.arange(topk, device=device, dtype=torch.int32) + first) % 128).contiguous()
                for density in (0.0, 0.015625, 0.125, 0.5, 1.0):
                    for scale in (0.001, 1., 32.):
                        h = (torch.randn((slots, 1856), device=device, generator=gen).relu().square() * scale).to(torch.bfloat16)
                        h *= (torch.rand(h.shape, device=device, generator=gen) < density)
                        h.view(torch.int16)[:, :4] = torch.tensor(
                            [0, -32768, 1, -32767], dtype=torch.int16, device=device)
                        if first == 0 and density == 1.0 and scale == 1.:
                            h.view(torch.int16)[:, 4:6] = torch.tensor(
                                [32640, -128], dtype=torch.int16, device=device)
                        if first == 0 and density == 1.0 and scale == 32.:
                            h.view(torch.int16)[:, 4:6] = torch.tensor(
                                [32704, -64], dtype=torch.int16, device=device)
                        if density == 0.0:
                            h.zero_()
                        _compact[(slots,)](h, indices, values, counts, I=1856, BLOCK=2048, num_warps=4)
                        old.fill_(float("nan"))
                        new.fill_(float("nan"))
                        _new(pw, ps, indices, values, counts, ids, old, tile, word, pdl)
                        if not _pkslots.run(indices, values, counts, pw.planes, ps.planes, ids, new, pdl):
                            ok = False
                            break
                        if not torch.equal(old.view(torch.int32), new.view(torch.int32)):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                _PKSLOTS_OK[(key, topk, word)] = True
                if os.environ.get("MTP_DOWN_PKRAW_PIPELINE", "1") != "0":
                    try:
                        from .nemotron_h_mtp_pkraw import qualify

                        if qualify(_pkslots, pw, ps, topk, pdl):
                            _PKRAW_OK[(key, topk, word)] = (pw, ps)
                    except Exception as exc:
                        print(f"MTP_DOWN_PKRAW optional gate unavailable: {exc!r}", flush=True)
        print(f"MTP_DOWN_PKSLOTS admitted={sorted(k[1:] for k in _PKSLOTS_OK if k[0] == key)}", flush=True)
    except Exception as exc:  # noqa: BLE001 - any failure keeps _slots
        for k in [k for k in _PKSLOTS_OK if k[0] == key]:
            del _PKSLOTS_OK[k]
        print(f"MTP_DOWN_PKSLOTS unavailable, _slots kept: {exc!r}", flush=True)


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
    selected = ((w.data_ptr(), sd.data_ptr()), ids.numel(), word)
    if _PKSLOTS_OK.get(selected):
        raw = _PKRAW_OK.get(selected)
        if (raw is not None and raw[0] is entry[2] and raw[1] is entry[3]
                and _pkslots.run(indices, values, counts, entry[2].planes, entry[3].planes,
                                 ids, part, pdl, raw_pipeline=True)):
            return True
        if _pkslots.run(indices, values, counts, entry[2].planes, entry[3].planes, ids, part, pdl):
            return True
    _new(entry[2], entry[3], indices, values, counts, ids, part, bn, word, pdl)
    return True

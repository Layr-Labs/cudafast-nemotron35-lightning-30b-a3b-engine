"""Optional compact-down 11-bit exponent planes; remote qualification only.

The encoding and generic loader derive from public submission
896465fa-a80c-4000-a968-f527c04fafff. No local candidate execution.
The candidate and its private admission remain unverified until an official run.
"""
import os
import statistics

import torch
from vllm.triton_utils import tl, triton
from .nemotron_h_mtp_packed import PackedBF16

_ENABLED = os.environ.get("MTP_DOWN_CODE11", "1") != "0"
_ADMITTED = {}


@triton.jit
def _popc(x):
    return tl.inline_asm_elementwise("popc.b32 $0, $1;", "=r,r", [x],
                                     dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _load_code11(SM, EX, REC, SB, RAW, index, mask):
    """Port of 896's generic arbitrary-index decoder to Triton language."""
    words = EX.to(tl.pointer_type(tl.uint32)) + (index // 32) * 3
    rec = REC + (index // 128) * 3
    sm = tl.load(SM + index, mask=mask, other=0, eviction_policy="evict_first").to(tl.uint32)
    w0 = tl.load(words, mask=mask, other=0, eviction_policy="evict_first")
    w1 = tl.load(words + 1, mask=mask, other=0, eviction_policy="evict_first")
    w2 = tl.load(words + 2, mask=mask, other=0, eviction_policy="evict_first")
    hdr = tl.load(rec, mask=mask, other=0, eviction_policy="evict_first")
    i0 = tl.load(rec + 1, mask=mask, other=0, eviction_policy="evict_first").to(tl.uint32)
    i1 = tl.load(rec + 2, mask=mask, other=0, eviction_policy="evict_first").to(tl.uint32)
    j = (index % 32).to(tl.uint32)
    code = ((w0 >> j) & 1) | (((w1 >> j) & 1) << 1) | (((w2 >> j) & 1) << 2)
    rank = (((hdr >> (8 + 4 * ((index // 32) % 4)).to(tl.int32)) & 15).to(tl.uint32)
            + _popc(~(w0 | w1 | w2) & (((j * 0 + 1) << j) - 1)).to(tl.uint32)) & 7
    exc = (tl.where(rank < 4, i0, i1) >> ((rank % 4) * 8)) & 255
    exp = tl.where(code == 0, exc, (hdr & 255).to(tl.uint32) + code - 1)
    bits = ((sm & 128) << 8) | (exp << 7) | (sm & 127)
    sbp = SB + (index // 4096) * 2
    raw = mask & (hdr < 0)
    sbm = tl.load(sbp, mask=raw, other=0).to(tl.uint32)
    sbs = tl.load(sbp + 1, mask=raw, other=0)
    blk = ((index // 128) % 32).to(tl.uint32)
    ridx = sbs + _popc(sbm & (((blk * 0 + 1) << blk) - 1))
    rv = tl.load(RAW + ridx.to(tl.int64) * 128 + index % 128,
                 mask=raw, other=0).to(tl.uint32)
    return tl.where(mask, tl.where(raw, rv, bits), 0).to(tl.uint16)


@triton.jit
def _verify_code11(SM, EX, REC, SB, RAW, original, mismatches,
                   N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < N
    bits = _load_code11(SM, EX, REC, SB, RAW, index, mask)
    want = tl.load(original + index, mask=mask, other=0).to(tl.uint16)
    tl.atomic_add(mismatches, tl.sum((mask & (bits != want)).to(tl.int32), axis=0))


@triton.jit
def _slots_code11(indices, values, counts,
                  RSM, REX, RREC, RSB, RRAW,
                  SSM, SEX, SREC, SSB, SRAW, ids, part,
                  H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
                  BN: tl.constexpr, BK: tl.constexpr,
                  launch_pdl: tl.constexpr = False):
    """Same ascending chunk and reduction tree as packed compact-down slots."""
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
    pos = lane
    k = tl.load(indices + slot * I + pos, mask=pos < count, other=0)
    value = tl.load(values + slot * I + pos, mask=pos < count, other=0.0)
    index = base + k[:, None].to(tl.int64) * H + n[None, :]
    mask = (pos[:, None] < count) & (n[None, :] < H)
    if slot < TOPK:
        bits = _load_code11(RSM, REX, RREC, RSB, RRAW, index, mask)
    else:
        bits = _load_code11(SSM, SEX, SREC, SSB, SRAW, index, mask)
    weight = bits.to(tl.bfloat16, bitcast=True)
    for start in range(BK, count + BK, BK):
        posn = start + lane
        kn = tl.load(indices + slot * I + posn, mask=posn < count, other=0)
        valuen = tl.load(values + slot * I + posn, mask=posn < count, other=0.0)
        indexn = base + kn[:, None].to(tl.int64) * H + n[None, :]
        maskn = (posn[:, None] < count) & (n[None, :] < H)
        if slot < TOPK:
            bitsn = _load_code11(RSM, REX, RREC, RSB, RRAW, indexn, maskn)
        else:
            bitsn = _load_code11(SSM, SEX, SREC, SSB, SRAW, indexn, maskn)
        weightn = bitsn.to(tl.bfloat16, bitcast=True)
        acc += weight.to(tl.float32) * value[:, None].to(tl.float32)
        weight = weightn
        value = valuen
    tl.store(part + slot * H + n, tl.sum(acc, axis=0), mask=n < H)


@torch.no_grad()
def encode_code11(w):
    """Input-independent GPU encode of BF16 w (rows a multiple of 128 values) into the
    planes (sm, ex, rec, sb, raw) read by load_code11. Lossless by construction;
    pack_code11 verifies every value."""
    n = w.numel()
    nb = n // 128
    nsb = (nb + 31) // 32
    flat = w.view(torch.int16).reshape(-1)
    dev = w.device
    sm = torch.empty(n, dtype=torch.uint8, device=dev)
    ex = torch.empty(nb * 12, dtype=torch.int32, device=dev)
    rec = torch.zeros((nb, 3), dtype=torch.int64, device=dev)
    israw = torch.empty(nb, dtype=torch.bool, device=dev)
    shifts = torch.arange(32, dtype=torch.int64, device=dev)
    step = 16384
    for first in range(0, nb, step):
        last = min(first + step, nb)
        bits = flat[first * 128:last * 128].to(torch.int32).reshape(-1, 128) & 65535
        exp = ((bits >> 7) & 255).to(torch.int64)
        hist = torch.zeros((last - first, 256), dtype=torch.int32, device=dev)
        hist.scatter_add_(1, exp, torch.ones_like(exp, dtype=torch.int32))
        csum = torch.nn.functional.pad(hist.cumsum(1), (1, 0))
        base = (csum[:, 7:257] - csum[:, 0:250]).argmax(1)
        inwin = (exp >= base[:, None]) & (exp <= base[:, None] + 6)
        out = ~inwin
        count = out.sum(1)
        raw = count > 8
        code = torch.where(inwin | raw[:, None], exp - base[:, None] + 1, 0)
        code = torch.where(raw[:, None], 1, code)
        sm[first * 128:last * 128].copy_((((bits >> 8) & 128) | (bits & 127)).reshape(-1))
        planes = ((code.reshape(-1, 32, 1) >> torch.arange(3, device=dev)) & 1)
        words = (planes << shifts[None, :, None]).sum(1)
        words = torch.where(words >= 2 ** 31, words - 2 ** 32, words)
        ex[first * 12:last * 12].copy_(words.reshape(-1).to(torch.int32))
        group = (out & ~raw[:, None]).reshape(-1, 4, 32).sum(2).to(torch.int64)
        pre = group.cumsum(1)
        hdr = base | (pre[:, 0] << 12) | (pre[:, 1] << 16) | (pre[:, 2] << 20)
        hdr = torch.where(raw, -2 ** 31, hdr)
        # listed exponents in element order, at most 8 per coded block
        rank = (out.to(torch.int64).cumsum(1) - 1).clamp(min=0)
        keep = out & ~raw[:, None]
        slot = torch.full_like(exp, 8)
        slot = torch.where(keep, rank, slot)
        lst = torch.zeros((last - first, 9), dtype=torch.int64, device=dev)
        lst.scatter_(1, slot, torch.where(keep, exp, 0))
        lo = lst[:, 0] | (lst[:, 1] << 8) | (lst[:, 2] << 16) | (lst[:, 3] << 24)
        hi = lst[:, 4] | (lst[:, 5] << 8) | (lst[:, 6] << 16) | (lst[:, 7] << 24)
        rec[first:last, 0] = hdr
        rec[first:last, 1] = torch.where(lo >= 2 ** 31, lo - 2 ** 32, lo)
        rec[first:last, 2] = torch.where(hi >= 2 ** 31, hi - 2 ** 32, hi)
        israw[first:last] = raw
    nraw = int(israw.sum().item())
    padded = torch.zeros(nsb * 32, dtype=torch.int64, device=dev)
    padded[:nb] = israw.to(torch.int64)
    sbmask = (padded.reshape(nsb, 32) << shifts).sum(1)
    sbmask = torch.where(sbmask >= 2 ** 31, sbmask - 2 ** 32, sbmask)
    sbcount = padded.reshape(nsb, 32).sum(1)
    sb = torch.stack([sbmask, sbcount.cumsum(0) - sbcount], 1).reshape(-1).to(torch.int32)
    raws = torch.zeros((max(1, nraw), 128), dtype=torch.int16, device=dev)
    if nraw:
        sel = torch.nonzero(israw).reshape(-1)
        raws.index_copy_(0, torch.arange(nraw, device=dev),
                         flat.reshape(-1, 128).index_select(0, sel))
    return PackedBF16(tuple(w.shape), (sm, ex, rec.reshape(-1).to(torch.int32), sb,
                                       raws.reshape(-1).view(torch.uint16))), nraw


def _encode_checked(w, existing):
    """Remote startup only: exhaustive real-owner bits and actual plane size."""
    if (w.dtype != torch.bfloat16 or not w.is_cuda or not w.is_contiguous()
            or w.numel() % 128 or existing is None):
        raise ValueError("unsupported real compact-down weight owner")
    packed, nraw = encode_code11(w)
    if packed.nbytes >= existing.nbytes:
        raise ValueError("code11 is not smaller than the actual 12-bit planes")
    count = torch.zeros(1, device=w.device, dtype=torch.int32)
    _verify_code11[(triton.cdiv(w.numel(), 4096),)](
        *packed.planes, w.view(torch.uint16), count, w.numel(), 4096,
        num_warps=4)
    if int(count.item()) != 0:
        raise ValueError("code11 real-weight bit reconstruction differs")
    return packed, nraw


def _qualify_codec_fixture(device):
    """Remote synthetic zero, exponent-window, escape and raw-block coverage."""
    i = torch.arange(512, device=device, dtype=torch.int32)
    group = i // 128
    lane = i % 128
    exp = torch.where(group == 0, torch.zeros_like(i),
                      torch.where(group == 1, 120 + lane % 7,
                                  torch.where(group == 2,
                                              torch.where(lane < 8, lane * 20, 124),
                                              (lane * 37) % 256)))
    bits = ((i % 2) << 15) | (exp << 7) | (lane % 128)
    fixture = torch.empty((4, 128), device=device, dtype=torch.bfloat16)
    fixture.view(torch.int16).copy_(bits.reshape(4, 128).to(torch.int16))
    packed, nraw = encode_code11(fixture)
    if nraw < 1:
        raise ValueError("code11 fixture did not exercise a raw block")
    count = torch.zeros(1, device=device, dtype=torch.int32)
    _verify_code11[(1,)](*packed.planes, fixture.view(torch.uint16), count,
                         fixture.numel(), 4096, num_warps=4)
    if int(count.item()) != 0:
        raise ValueError("code11 edge-block fixture differs")


def _run11(pw, ps, indices, values, counts, ids, part, tile, pdl):
    bn, warps = tile
    _slots_code11[(triton.cdiv(2688, bn), ids.numel() + 2)](
        indices, values, counts, *pw.planes, *ps.planes, ids, part,
        H=2688, I=1856, TOPK=ids.numel(), BN=bn, BK=64,
        num_warps=warps, num_stages=3, launch_pdl=pdl)


def _run_reference(key, entry, indices, values, counts, ids, part, word, tile, pdl):
    """Match the actual already-admitted packed12/pkslots dispatch."""
    from . import nemotron_h_mtp_down_packed as old
    if (old._PKSLOTS_OK.get((key, ids.numel(), word)) and old._pkslots is not None
            and old._pkslots.run(indices, values, counts, entry[2].planes,
                                 entry[3].planes, ids, part, pdl)):
        return
    old._new(entry[2], entry[3], indices, values, counts, ids, part, tile, word, pdl)


def _guarded_part(slots, device):
    owner = torch.empty(slots * 2688 + 128, device=device, dtype=torch.float32)
    return owner, owner[64:-64].view(slots, 2688)


def _word_equal(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _guard_ok(owner):
    return bool(torch.all(owner[:64].view(torch.uint8) == 0xA5).item()
                and torch.all(owner[-64:].view(torch.uint8) == 0xA5).item())


@torch.no_grad()
def _qualify_slots(key, entry, pw, ps, topk, word, tile, pdl):
    """Authored for official worker startup; no local execution or proof yet."""
    from .nemotron_h_mtp_compact import _compact
    device = entry[0].device
    gen = torch.Generator(device=device).manual_seed(117520261028 + topk)
    slots = topk + 2
    h = torch.empty((slots, 1856), dtype=torch.bfloat16, device=device)
    indices = torch.empty((slots, 1856), dtype=torch.int32, device=device)
    values = torch.empty_like(h)
    counts = torch.empty(slots, dtype=torch.int32, device=device)
    ids = torch.empty(topk, dtype=torch.int32, device=device)
    old_owner, old = _guarded_part(slots, device)
    new_owner, new = _guarded_part(slots, device)

    def compare():
        old_owner.view(torch.uint8).fill_(0xA5)
        new_owner.view(torch.uint8).fill_(0xA5)
        _compact[(slots,)](h, indices, values, counts, I=1856, BLOCK=2048,
                           num_warps=4, launch_pdl=pdl)
        owners = tuple(t.clone() for t in (h, indices, values, counts, ids))
        _run_reference(key, entry, indices, values, counts, ids, old, word, tile, pdl)
        _run11(pw, ps, indices, values, counts, ids, new, tile, pdl)
        if (not _word_equal(old, new) or not _guard_ok(old_owner) or not _guard_ok(new_owner)
                or not all(_word_equal(a, b) for a, b in zip((h, indices, values, counts, ids), owners))):
            raise ValueError("code11 slot output, guard or mutable owner differs")

    # Every expert appears, with mixed, dense, sparse and empty compact rows.
    for first in range(0, 128, topk):
        ids.copy_(((torch.arange(topk, device=device, dtype=torch.int32) + first) % 128))
        cases = ([(density, scale) for density in (0.0, 0.015625, 0.125, 0.5, 1.0)
                  for scale in (0.001, 1.0, 32.0)]
                 if first == 0 or first + topk >= 128 else [(0.5, 1.0)])
        for density, scale in cases:
            sample = torch.randn(h.shape, device=device, generator=gen).relu().square() * scale
            sample *= (torch.rand(h.shape, device=device, generator=gen) < density)
            h.copy_(sample.to(torch.bfloat16))
            if first == 0 and density == 1.0 and scale == 1.0:
                h.view(torch.int16)[:, :6] = torch.tensor(
                    [0, -32768, 1, -32767, 32640, -128], dtype=torch.int16, device=device)
            if first == 0 and density == 1.0 and scale == 32.0:
                h.view(torch.int16)[:, :2] = torch.tensor(
                    [32704, -64], dtype=torch.int16, device=device)
            compare()

    # Fixed addresses, changing the compact indices, counts and experts
    # between replays. No Python branch may select a stale admission outcome.
    torch.cuda.synchronize(device)
    old_graph, new_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(old_graph, pool=torch.cuda.graph_pool_handle()):
        _run_reference(key, entry, indices, values, counts, ids, old, word, tile, pdl)
    with torch.cuda.graph(new_graph, pool=torch.cuda.graph_pool_handle()):
        _run11(pw, ps, indices, values, counts, ids, new, tile, pdl)
    for case, density in enumerate((0.0, 1.0, 0.125, 0.5, 0.015625, 1.0)):
        ids.copy_(((torch.arange(topk, device=device, dtype=torch.int32)
                    + case * 19) % 128))
        sample = torch.randn(h.shape, device=device, generator=gen).relu().square()
        sample *= (torch.rand(h.shape, device=device, generator=gen) < density)
        h.copy_(sample.to(torch.bfloat16))
        _compact[(slots,)](h, indices, values, counts, I=1856, BLOCK=2048,
                           num_warps=4, launch_pdl=pdl)
        saved = tuple(t.clone() for t in (h, indices, values, counts, ids))
        old_owner.view(torch.uint8).fill_(0xA5)
        new_owner.view(torch.uint8).fill_(0xA5)
        old_graph.replay()
        new_graph.replay()
        if (not _word_equal(old, new) or not _guard_ok(old_owner) or not _guard_ok(new_owner)
                or not all(_word_equal(a, b) for a, b in zip((h, indices, values, counts, ids), saved))):
            raise ValueError("code11 changed-metadata graph output, guard or owner differs")


@torch.no_grad()
def _time_full_chain(key, entry, pw, ps, topk, word, tile, pdl, routed_scale):
    """Compare compaction, selected slot implementation and residual combine."""
    from .nemotron_h_mtp_compact import _compact, _down_combine
    device = entry[0].device
    gen = torch.Generator(device=device).manual_seed(775008 + topk)
    slots = topk + 2
    h = torch.empty((slots, 1856), dtype=torch.bfloat16, device=device)
    indices = torch.empty((slots, 1856), dtype=torch.int32, device=device)
    values = torch.empty_like(h)
    counts = torch.empty(slots, dtype=torch.int32, device=device)
    ids = torch.empty(topk, dtype=torch.int32, device=device)
    weights = torch.full((topk,), 1.0 / topk, dtype=torch.float32, device=device)
    residual = torch.randn((8, 2688), device=device, generator=gen).to(torch.bfloat16)
    part = torch.empty((slots, 2688), dtype=torch.float32, device=device)
    out = torch.empty_like(residual)

    def chain(arm, rows, add):
        _compact[(slots,)](h, indices, values, counts, I=1856, BLOCK=2048,
                           num_warps=4, launch_pdl=pdl)
        if arm:
            _run11(pw, ps, indices, values, counts, ids, part, tile, pdl)
        else:
            _run_reference(key, entry, indices, values, counts, ids, part, word, tile, pdl)
        _down_combine[(triton.cdiv(2688, 256),)](
            part, weights, out, M=rows, H=2688, ROUTED_SCALE=routed_scale,
            TOPK=topk, S=2, BN=256, residual_ptr=residual if add else None, ADD_RES=add,
            num_warps=4, launch_pdl=pdl)

    pressure = torch.empty(64 * (1 << 20), dtype=torch.uint8, device=device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ratios, totals = [], [0.0, 0.0]
    for rows, add in ((1, False), (1, True), (5, False), (5, True), (8, False), (8, True)):
        # Capture each chain with fixed addresses after warmup, then change h
        # and ids in place for every density and trial.
        h.zero_()
        ids.zero_()
        for arm in (0, 1):
            chain(arm, rows, add)
        torch.cuda.synchronize(device)
        graphs = []
        for arm in (0, 1):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
                chain(arm, rows, add)
            graphs.append(graph)
        for density in (0.125, 0.5, 1.0):
            sample = torch.randn(h.shape, device=device, generator=gen).relu().square()
            sample *= (torch.rand(h.shape, device=device, generator=gen) < density)
            h.copy_(sample.to(torch.bfloat16))
            ids.copy_(torch.randperm(128, device=device, generator=gen)[:topk].to(torch.int32))
            graphs[0].replay()
            expected = out[:rows].clone()
            graphs[1].replay()
            if not _word_equal(expected, out[:rows]):
                raise ValueError("code11 complete graph output differs")
            for cold in (False, True):
                times = [[], []]
                for trial in range(5):
                    for arm in ((0, 1) if trial % 2 == 0 else (1, 0)):
                        if cold:
                            pressure.fill_(trial)
                        start.record()
                        for _ in range(8):
                            graphs[arm].replay()
                        end.record()
                        end.synchronize()
                        times[arm].append(start.elapsed_time(end) / 8)
                old, new = (statistics.median(x) for x in times)
                if min(old, new) <= 0:
                    raise ValueError("code11 nonpositive full-chain interval")
                ratios.append(new / old)
                totals[0] += old
                totals[1] += new
    if (not ratios or statistics.median(ratios) >= 0.99
            or totals[1] / totals[0] >= 0.99 or max(ratios) >= 1.02):
        raise ValueError(f"code11 full-chain timing refused {ratios}")
    return tuple(ratios)


@torch.no_grad()
def prepare_optional(key, entry, word, pdl, routed_scale):
    """Remote-only optional owner-specific admission; no output change on refusal."""
    if not _ENABLED or torch.cuda.is_current_stream_capturing() or routed_scale is None:
        return
    if key in _ADMITTED or entry[2] is None or entry[3] is None:
        return
    try:
        free_bytes, _ = torch.cuda.mem_get_info(entry[0].device)
        # Both old and new packed planes must coexist for the fallback; bound
        # temporary encoder storage too, otherwise leave the original path.
        if free_bytes < 2 * (entry[0].numel() + entry[1].numel()) + (1 << 30):
            raise ValueError("insufficient memory for optional code11 owners")
        _qualify_codec_fixture(entry[0].device)
        pw, routed_raw = _encode_checked(entry[0], entry[2])
        ps, shared_raw = _encode_checked(entry[1], entry[3])
        admitted, reasons = {}, {}
        for topk in (3, 6):
            tile = entry[4].get((topk, word))
            if not isinstance(tile, tuple) or len(tile) != 2:
                continue
            try:
                _qualify_slots(key, entry, pw, ps, topk, word, tile, pdl)
                ratios = _time_full_chain(
                    key, entry, pw, ps, topk, word, tile, pdl, float(routed_scale))
                admitted[(topk, word, pdl)] = (tile, ratios)
            except Exception as exc:
                reasons[topk] = repr(exc)
        if admitted:
            _ADMITTED[key] = (entry[0], entry[1], pw, ps, admitted)
        print(f"MTP_DOWN_CODE11 admitted={list(admitted)} refused={reasons} "
              f"raw_blocks={(routed_raw, shared_raw)}", flush=True)
    except Exception as exc:
        print(f"MTP_DOWN_CODE11 original packed slots retained: {exc!r}", flush=True)


def maybe_slots(packed, indices, values, counts, ids, part, word, tile, pdl, rows):
    if not _ENABLED or rows not in (1, 5, 8):
        return False
    w, sd = packed
    key = (w.data_ptr(), sd.data_ptr())
    entry = _ADMITTED.get(key)
    if (entry is None or entry[0] is not w or entry[1] is not sd
            or (ids.numel(), word, pdl) not in entry[4]
            or entry[4][(ids.numel(), word, pdl)][0] != tile):
        return False
    _run11(entry[2], entry[3], indices, values, counts, ids, part, tile, pdl)
    return True

# SPDX-License-Identifier: Apache-2.0
"""Local ordered compaction + persistent native-MTP down, optional at boot.

Loaded-weight qualification and actual packed/unpacked policy participate in
remote startup admission. No local import, syntax execution, compile or test
was run. Source alone does not establish admission, correctness or speed.

Base: ENTIRE f0fcab75-b0b3-4a4c-893d-850ef447c110. Existing compact slot tree:
526d6e53-7b8b-4ce8-b4a1-b3c7a53be9d9; packed slot source/codec:
3d51cab6-7079-4fb2-98e4-f2a238077897.
"""

import os
import statistics

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

from .nemotron_h_mtp_down_packed import _load_packed

logger = init_logger(__name__)
_ENABLED = os.environ.get("MTP_INLINE_COMPACT", "1") != "0"
_ADMITTED = {}


@triton.jit
def _inline_slots(
    h, w2, sd, ids, part,
    RSM, REX, RBASE, RRI, RRAW, SSM, SEX, SBASE, SRI, SRAW,
    H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
    BN: tl.constexpr, BK: tl.constexpr, GROUP_COLS: tl.constexpr,
    WORD: tl.constexpr, PACKED: tl.constexpr, BLOCK: tl.constexpr = 2048,
    launch_pdl: tl.constexpr = False,
):
    tl.static_assert(H == 2688 and I == 1856 and BK == 64)
    tl.static_assert(BN == 32 or BN == 64)
    tl.static_assert(GROUP_COLS == 4)
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    slot = tl.program_id(1)
    original_k = tl.arange(0, BLOCK)
    activation = tl.load(h + slot * I + original_k, mask=original_k < I, other=0.0)
    live = (original_k < I) & (activation != 0.0)
    count = tl.sum(live.to(tl.int32), axis=0)
    # Sort only original integer indices: every live index is unique, hence
    # the prefix is exactly _compact's ascending K order. No FP keys or
    # activation values are reordered by a floating-point comparison.
    ordered = original_k
    if (count > 0) & (count < I):
        ordered = tl.sort(tl.where(live, original_k, I), descending=False)
    if slot < TOPK:
        offset = tl.load(ids + slot).to(tl.int64) * I * H
    else:
        offset = (slot - TOPK).to(tl.int64) * I * H
    lane = tl.arange(0, BK)
    col = tl.arange(0, BN)
    # Reuse one local ordered list across output column blocks. This cuts
    # duplicated local compaction relative to one compactor per BN block.
    # Keeping original BN/BK/num_warps is required; generated reduction layout
    # still needs a remote bitwise check rather than a source-only claim.
    for q in range(GROUP_COLS):
        n = (tl.program_id(0) * GROUP_COLS + q) * BN + col
        acc = tl.zeros((BK, BN), tl.float32)
        for start in range(0, count, BK):
            pos = start + lane
            selected = tl.gather(ordered, tl.minimum(pos, BLOCK - 1), axis=0)
            k = tl.where(pos < count, selected, 0)
            value = tl.load(h + slot * I + k, mask=pos < count, other=0.0)
            index = offset + k[:, None].to(tl.int64) * H + n[None, :]
            mask = (pos[:, None] < count) & (n[None, :] < H)
            if PACKED:
                if slot < TOPK:
                    weight = _load_packed(RSM, REX, RBASE, RRI, RRAW, index, mask, WORD=WORD)
                else:
                    weight = _load_packed(SSM, SEX, SBASE, SRI, SRAW, index, mask, WORD=WORD)
            else:
                if slot < TOPK:
                    weight = tl.load(w2 + index, mask=mask, other=0.0, eviction_policy="evict_first")
                else:
                    weight = tl.load(sd + index, mask=mask, other=0.0, eviction_policy="evict_first")
            acc += weight.to(tl.float32) * value[:, None].to(tl.float32)
        tl.store(part + slot * H + n, tl.sum(acc, axis=0), mask=n < H)


def _policy(packed, topk):
    """Resolve precisely the policy maybe_slots would use now, not a surrogate."""
    from . import nemotron_h_mtp_down_packed as codec
    from .nemotron_h_mtp_packed import word_load_enabled

    w, sd = packed
    word = word_load_enabled(w.device)
    entry = codec._PACKS.get((w.data_ptr(), sd.data_ptr()))
    if (codec._ENABLED and entry is not None and entry[0] is w and entry[1] is sd
            and entry[4].get((topk, word)) is not None):
        return (w, sd, entry[2], entry[3], entry[4][(topk, word)], word)
    return (w, sd, None, None, 32, word)


def _same_policy(a, b):
    return all(a[i] is b[i] for i in range(4)) and a[4:] == b[4:]


def _launch(h, ids, part, policy, pdl):
    w, sd, pw, ps, bn, word = policy
    # Unused planes stay real tensors, with no synthetic storage or new codec.
    planes = (*pw.planes, *ps.planes) if pw is not None else (w,) * 5 + (sd,) * 5
    _inline_slots[(triton.cdiv(2688, bn * 4), ids.numel() + 2)](
        h, w, sd, ids, part, *planes,
        H=2688, I=1856, TOPK=ids.numel(), BN=bn, BK=64, GROUP_COLS=4,
        WORD=word, PACKED=pw is not None, num_warps=4, num_stages=3,
        launch_pdl=pdl,
    )


def _guarded(shape, dtype, device):
    count = 1
    for n in shape:
        count *= n
    width = torch.empty((), dtype=dtype).element_size()
    store = torch.full((count * width + 512,), 173, dtype=torch.uint8, device=device)
    return store[256:-256].view(dtype).view(shape), store


def _guards_ok(store):
    return bool(torch.all(store[:256] == 173) and torch.all(store[-256:] == 173))


def _qualify(packed, topk, policy, routed_scale, pdl):
    """Official remote startup only. Compare entire old/new down pipelines,
    including the actual admitted packed slots and the unchanged combine.
    """
    from .nemotron_h_mtp_compact import _compact, _down_slots, _down_combine
    from .nemotron_h_mtp_down_packed import maybe_slots

    w, sd = packed
    device = w.device
    slots = topk + 2
    gen = torch.Generator(device=device).manual_seed(973826 + topk)
    h, hg = _guarded((slots, 1856), torch.bfloat16, device)
    oldpart, pg0 = _guarded((slots, 2688), torch.float32, device)
    newpart, pg1 = _guarded((slots, 2688), torch.float32, device)
    ids = torch.arange(topk, dtype=torch.int32, device=device)
    weights = torch.rand(topk, generator=gen, device=device, dtype=torch.float32)
    weights /= weights.sum()
    residual, rg = _guarded((8, 2688), torch.bfloat16, device)
    old, yg0 = _guarded((8, 2688), torch.bfloat16, device)
    new, yg1 = _guarded((8, 2688), torch.bfloat16, device)
    indices = torch.empty(h.shape, dtype=torch.int32, device=device)
    values = torch.empty_like(h)
    counts = torch.empty(slots, dtype=torch.int32, device=device)

    def old_slots():
        _compact[(slots,)](h, indices, values, counts, I=1856, BLOCK=2048,
                          num_warps=4, launch_pdl=pdl)
        if not maybe_slots(packed, indices, values, counts, ids, oldpart, pdl):
            _down_slots[(84, slots)](
                indices, values, counts, w, sd, ids, oldpart,
                H=2688, I=1856, TOPK=topk, BN=32, BK=64,
                num_warps=4, num_stages=3, launch_pdl=pdl,
            )

    def new_slots():
        _launch(h, ids, newpart, policy, pdl)

    def combine(part, y, m=8, add=True):
        _down_combine[(triton.cdiv(2688, 256),)](
            part, weights, y, M=m, H=2688, ROUTED_SCALE=routed_scale,
            TOPK=topk, S=2, BN=256, num_warps=4, launch_pdl=pdl,
            residual_ptr=residual, ADD_RES=add,
        )

    residual.copy_(torch.randn((8, 2688), generator=gen, device=device).to(torch.bfloat16))
    # Every routed expert participates. Different densities/magnitudes exercise
    # different compact positions rather than timing only an artificially sparse
    # surrogate. Value tests use the loaded original or admitted packed planes.
    for first in range(0, 128, topk):
        ids.copy_((torch.arange(topk, device=device, dtype=torch.int32) + first) % 128)
        for case, density in enumerate((0.0, 0.015625, 0.125, 0.5, 1.0)):
            scale = (0.001, 1.0, 32.0)[(first + case) % 3]
            probe = torch.randn(h.shape, generator=gen, device=device).relu().square() * scale
            probe *= (torch.rand(h.shape, generator=gen, device=device) < density)
            h.copy_(probe.to(torch.bfloat16))
            h.view(torch.int16)[h == 0] = -32768
            expected = h.clone()
            old_slots()
            new_slots()
            if (not torch.equal(oldpart.view(torch.int32), newpart.view(torch.int32))
                    or not torch.equal(h.view(torch.int16), expected.view(torch.int16))):
                return False
    # Partial final chunks and empty/singleton lists, plus finite BF16 extremes.
    # Keep the old slot reduction tree as the
    # oracle instead of deriving correctness from index-order reasoning alone.
    for count in (0, 1, 63, 64, 65, 1856):
        h.zero_()
        h[:, :count] = (torch.randn((slots, count), generator=gen, device=device) * 0.25).to(torch.bfloat16)
        if count > 0:
            h[:, 0] = torch.finfo(torch.bfloat16).tiny
        if count > 63:
            h[:, 63] = torch.finfo(torch.bfloat16).max
        old_slots()
        new_slots()
        if not torch.equal(oldpart.view(torch.int32), newpart.view(torch.int32)):
            return False
        for m in (1, 5, 8):
            for add in (False, True):
                combine(oldpart, old, m, add)
                combine(newpart, new, m, add)
                if not torch.equal(old[:m].view(torch.int16), new[:m].view(torch.int16)):
                    return False
        if not all(_guards_ok(s) for s in (hg, pg0, pg1, rg, yg0, yg1)):
            return False

    # Captures include the full original compact/actual-slots/combine or the
    # new inline/combine pair. Replays change h, ids and residual in place.
    h.copy_(torch.randn(h.shape, generator=gen, device=device).relu().square().to(torch.bfloat16))
    old_slots()
    new_slots()
    combine(oldpart, old)
    combine(newpart, new)
    torch.cuda.synchronize(device)
    graphs = (torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph())
    with torch.cuda.graph(graphs[0]):
        old_slots()
        combine(oldpart, old)
    with torch.cuda.graph(graphs[1]):
        new_slots()
        combine(newpart, new)
    for replay in range(12):
        h.copy_((torch.randn(h.shape, generator=gen, device=device).relu().square()
                 * (replay % 4 != 0)).to(torch.bfloat16))
        ids.copy_((torch.arange(topk, device=device, dtype=torch.int32) + replay * 11) % 128)
        residual.neg_()
        graphs[0].replay()
        graphs[1].replay()
        torch.cuda.synchronize(device)
        if (not torch.equal(oldpart.view(torch.int32), newpart.view(torch.int32))
                or not torch.equal(old.view(torch.int16), new.view(torch.int16))
                or not all(_guards_ok(s) for s in (hg, pg0, pg1, rg, yg0, yg1))):
            return False

    # Time the common production output-row counts, not just the eight-row
    # correctness probe. Other output sizes retain the original dispatcher.
    performance_pairs = []
    for m in (1, 5):
        pair = (torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph())
        with torch.cuda.graph(pair[0]):
            old_slots()
            combine(oldpart, old, m)
        with torch.cuda.graph(pair[1]):
            new_slots()
            combine(newpart, new, m)
        performance_pairs.append(pair)
    pressure = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    ratios = []
    for density in (0.0, 0.015625, 0.125, 0.5, 1.0):
        probe = torch.randn(h.shape, generator=gen, device=device).relu().square()
        probe *= torch.rand(h.shape, generator=gen, device=device) < density
        h.copy_(probe.to(torch.bfloat16))
        for pair in performance_pairs:
            for cold in (False, True):
                samples = ([], [])
                for repeat in range(5):
                    for which in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                        graph = pair[which]
                        graph.replay()
                        if cold:
                            pressure.fill_(repeat)
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record()
                        for _ in range(1 if cold else 4):
                            graph.replay()
                        end.record()
                        end.synchronize()
                        samples[which].append(begin.elapsed_time(end))
                ratios.append(statistics.median(samples[1]) / statistics.median(samples[0]))
    median, worst = statistics.median(ratios), max(ratios)
    logger.info("MTP_INLINE_COMPACT topk=%d packed=%s BN=%d median=%.6f worst=%.6f",
                topk, policy[2] is not None, policy[4], median, worst)
    return median < 0.98 and worst < 1.01 and all(
        _guards_ok(s) for s in (hg, pg0, pg1, rg, yg0, yg1)
    )


def prepare(packed, routed_scale):
    if not _ENABLED or torch.cuda.is_current_stream_capturing():
        return
    from .nemotron_h_mtp_compact import _ENABLED as compact_enabled, _SLOTS
    from .nemotron_h_mtp_kernels import mtp_pdl

    w, sd = packed
    if (not compact_enabled or not _SLOTS or w.shape != (128, 1856, 2688)
            or sd.shape != (2, 1856, 2688) or w.dtype != torch.bfloat16
            or sd.dtype != torch.bfloat16 or not w.is_contiguous() or not sd.is_contiguous()):
        return
    for topk in (3, 6):
        key = (w.data_ptr(), sd.data_ptr(), topk, float(routed_scale))
        if key in _ADMITTED:
            continue
        _ADMITTED[key] = None
        try:
            policy = _policy(packed, topk)
            if _qualify(packed, topk, policy, float(routed_scale), mtp_pdl()):
                _ADMITTED[key] = policy
                logger.info("MTP_INLINE_COMPACT_ADMITTED topk=%d", topk)
        except Exception as exc:  # original actual policy remains on any refusal
            logger.warning_once("MTP inline compact unavailable %r; original retained", exc)


def maybe_down(h, packed, ids, weights, y, routed_scale, residual):
    if (not _ENABLED or torch.compiler.is_compiling() or ids.numel() not in (3, 6)
            or y.shape[0] not in (1, 5)):
        return False
    w, sd = packed
    topk = ids.numel()
    key = (w.data_ptr(), sd.data_ptr(), topk, float(routed_scale))
    policy = _ADMITTED.get(key)
    if policy is None or not _same_policy(policy, _policy(packed, topk)):
        return False
    from .nemotron_h_mtp_compact import _down_combine
    from .nemotron_h_mtp_kernels import mtp_pdl

    part = torch.empty((topk + 2, 2688), dtype=torch.float32, device=h.device)
    # Caller already checked shape/format/contiguity and residual eligibility.
    # Keep all request buffers local; errors after the first launch propagate.
    pdl = mtp_pdl()
    _launch(h, ids, part, policy, pdl)
    _down_combine[(triton.cdiv(2688, 256),)](
        part, weights, y, M=y.shape[0], H=2688, ROUTED_SCALE=routed_scale,
        TOPK=topk, S=2, BN=256, num_warps=4, launch_pdl=pdl,
        residual_ptr=residual, ADD_RES=residual is not None,
    )
    return True

# SPDX-License-Identifier: Apache-2.0
"""Optional CTA-local compaction/down fusion, actual-owner remote admission only."""
import math
import os
import statistics
from types import SimpleNamespace

import torch
from vllm.triton_utils import triton

_ENABLED = os.environ.get("MTP_DOWN_LOCAL_COMPACT", "1") != "0"
_ENTRIES = {}


def _sig(t):
    return id(t), t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device


def _prefetch_sig(k):
    p = k._l2pf
    return None if p is None else (id(p), p.DRAFT_BYTES, k._PF_GATED,
                                   tuple(_sig(t) for t in p.DRAFT_SCREEN))


def _owner(layer):
    from . import nemotron_h_mtp_kernels as k
    from . import nemotron_h_mtp_down_packed as d
    from . import nemotron_h_mtp_compact as c
    from .nemotron_h_mtp_packed import word_load_enabled
    w = layer._fast_moe_weights
    if not (_ENABLED and w and c._ENABLED and c._SLOTS and d._ENABLED and k._RELU2_MODE == 2):
        return None
    _, down, _, sd, inter, ns = w
    if (tuple(down.shape) != (128, 2688, 1856) or tuple(sd.shape) != (2688, 3712)
            or (inter, ns) != (1856, 2) or not down.is_cuda
            or torch.cuda.get_device_capability(down.device) != (12, 1)):
        return None
    packed = k._RELU2_PACKED.get((down.data_ptr(), sd.data_ptr()))
    if not packed:
        return None
    key = tuple(t.data_ptr() for t in packed)
    entry = d._PACKS.get(key)
    if (entry is None or entry[2] is None or entry[3] is None
            or any(a is not b for a, b in zip(entry[:2], packed))):
        return None
    word = word_load_enabled(down.device)
    topks = tuple(n for n in (3, 6) if entry[4].get((n, word)) is not None)
    if not topks:
        return None
    tensors = (down, sd, *packed, *entry[2].planes, *entry[3].planes)
    if any(not t.is_cuda or t.device != down.device or not t.is_contiguous() for t in tensors):
        return None
    scale = float(layer.mixer.routed_scaling_factor)
    if not math.isfinite(scale):
        return None
    return SimpleNamespace(layer=layer, weights=w, down=down, sd=sd,
                           packed=packed, key=key, entry=entry,
                           flags=tuple(sorted(entry[4].items())), topks=topks,
                           tensors=tensors, signatures=tuple(map(_sig, tensors)),
                           scale=scale, device=down.device, word=word,
                           pdl=k.mtp_pdl(), prefetch=_prefetch_sig(k))


def _can(spec, h, packed, ids, weights, y, scale, residual):
    from . import nemotron_h_mtp_kernels as k
    from . import nemotron_h_mtp_compact as c
    from . import nemotron_h_mtp_down_packed as d
    from .nemotron_h_mtp_packed import word_load_enabled
    topk = ids.numel()
    return (_ENABLED and c._ENABLED and c._SLOTS and d._ENABLED and k._RELU2_MODE == 2
            and packed is spec.packed and d._PACKS.get(spec.key) is spec.entry
            and spec.layer._fast_moe_weights is spec.weights
            and tuple(sorted(spec.entry[4].items())) == spec.flags
            and all(_sig(t) == s for t, s in zip(spec.tensors, spec.signatures))
            and _prefetch_sig(k) == spec.prefetch and k.mtp_pdl() == spec.pdl
            and word_load_enabled(spec.device) == spec.word
            and float(scale) == spec.scale and topk in spec.topks
            and h.dtype == torch.bfloat16 and h.shape == (topk + 2, 1856)
            and ids.dtype == torch.int32 and ids.shape == (topk,)
            and weights.dtype == torch.float32 and weights.shape == (topk,)
            and y.dtype == torch.bfloat16 and y.ndim == 2
            and y.shape[0] in (1, 5, 8) and y.shape[1] == 2688
            and all(t.is_cuda and t.device == spec.device and t.is_contiguous()
                    for t in (h, ids, weights, y))
            and (residual is None or c.can_add_residual(residual, y, scale, topk)))


def _core(spec, h, ids, weights, y, residual):
    import pkcompact
    from .nemotron_h_mtp_compact import _down_combine
    topk = ids.numel()
    part = torch.empty((topk + 2, 2688), dtype=torch.float32, device=h.device)
    if not pkcompact.run(h, spec.entry[2].planes, spec.entry[3].planes, ids, part, spec.pdl):
        raise RuntimeError("admitted local compact ABI changed after dispatch")
    _down_combine[(11,)](part, weights, y, M=y.shape[0], H=2688,
                         ROUTED_SCALE=spec.scale, TOPK=topk, S=2, BN=256,
                         num_warps=4, launch_pdl=spec.pdl,
                         residual_ptr=residual, ADD_RES=residual is not None)


def _run(spec, h, packed, ids, weights, y, scale, residual):
    if not _can(spec, h, packed, ids, weights, y, scale, residual):
        return False  # No new GPU work was launched.
    _core(spec, h, ids, weights, y, residual)
    return True


def maybe_down(h, packed, ids, weights, y, scale, residual):
    spec = _ENTRIES.get(tuple(t.data_ptr() for t in packed))
    return False if spec is None else _run(spec, h, packed, ids, weights, y, scale, residual)


def _words(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _guarded(shape, dtype, device):
    n = math.prod(shape)
    owner = torch.empty(n + 128, dtype=dtype, device=device)
    owner.view(torch.uint8).fill_(0xA5)
    return owner, owner[64:-64].view(shape)


def _guard(pair):
    return bool(torch.all(pair[0][:64].view(torch.uint8) == 0xA5).item()
                and torch.all(pair[0][-64:].view(torch.uint8) == 0xA5).item())


def _old_part(spec, h, ids, part):
    from .nemotron_h_mtp_compact import _compact
    from .nemotron_h_mtp_down_packed import maybe_slots
    indices = torch.empty(h.shape, dtype=torch.int32, device=h.device)
    values = torch.empty_like(h)
    counts = torch.empty(h.shape[0], dtype=torch.int32, device=h.device)
    _compact[(h.shape[0],)](h, indices, values, counts, I=1856, BLOCK=2048,
                           num_warps=4, launch_pdl=spec.pdl)
    if not maybe_slots(spec.packed, indices, values, counts, ids, part, spec.pdl):
        raise RuntimeError("actual original selected packed down unavailable")


def _old_full(spec, h, ids, weights, y, residual):
    from .nemotron_h_mtp_kernels import relu2_down
    added = relu2_down(h, spec.down, spec.sd, ids, weights, y, spec.scale, residual=residual)
    if residual is not None and not added:
        raise RuntimeError("actual original residual fold unavailable")


def _new_full(spec, h, ids, weights, y, residual):
    from . import nemotron_h_mtp_kernels as k
    pf = False
    if k._l2pf is not None and k._l2pf.DRAFT_BYTES > 0 and bool(k._l2pf.DRAFT_SCREEN):
        pf = k.gated_prefetch([(k._l2pf.DRAFT_SCREEN[0], k._l2pf.DRAFT_BYTES)], h.device, name="shead")
        if pf is None:
            pf = k._l2pf.prefetch(k._l2pf.DRAFT_SCREEN[0], max_bytes=k._l2pf.DRAFT_BYTES)
    if not _run(spec, h, spec.packed, ids, weights, y, spec.scale, residual):
        raise RuntimeError("private production local compact guard refused")
    if pf:
        k._l2pf.join(h.device)


@torch.no_grad()
def _qualify(spec, topk):
    import pkcompact
    from . import nemotron_h_mtp_kernels as k
    device, slots = spec.device, topk + 2
    gen = torch.Generator(device=device).manual_seed(2026102813 + topk)
    hb = _guarded((slots, 1856), torch.bfloat16, device)
    ib = _guarded((topk,), torch.int32, device)
    wb = _guarded((topk,), torch.float32, device)
    h, ids, weights = hb[1], ib[1], wb[1]
    gate = k._GATE
    oldpart = _guarded((slots, 2688), torch.float32, device)
    newpart = _guarded((slots, 2688), torch.float32, device)
    residuals = {m: torch.empty((m, 2688), dtype=torch.bfloat16, device=device) for m in (1, 5, 8)}

    def change(case, first=0):
        density = (0.0, .015625, .125, .37, .9, 1.0)[case % 6]
        scale = (.001, 1.0, 32.0)[case % 3]
        h.copy_((torch.randn(h.shape, device=device, generator=gen).relu().square() * scale).to(h.dtype))
        h.mul_(torch.rand(h.shape, device=device, generator=gen) < density)
        if density == 0.0:
            h.zero_()
        else:
            h.view(torch.int16)[:, :4] = torch.tensor([0, -32768, 1, -32767], dtype=torch.int16, device=device)
        if case == 6:
            h.view(torch.int16)[:, 4:8] = torch.tensor([32640, -128, 32704, -64], dtype=torch.int16, device=device)
        ids.copy_((torch.arange(topk, device=device, dtype=torch.int32) + first) % 128)
        weights.copy_(torch.rand(weights.shape, device=device, generator=gen))
        weights.div_(weights.sum())
        for r in residuals.values():
            r.copy_((torch.randn(r.shape, device=device, generator=gen) * scale).to(r.dtype))
            r.view(torch.int16)[:, :4] = torch.tensor([0, -32768, 1, -32767], dtype=torch.int16, device=device)
        gate.zero_()
        if density == 0.0:
            gate[0] = 1

    def check_parts():
        saved = tuple(t.clone() for t in (h, ids, weights, gate))
        _old_part(spec, h, ids, oldpart[1])
        if not pkcompact.run(h, spec.entry[2].planes, spec.entry[3].planes, ids, newpart[1], spec.pdl):
            raise RuntimeError("private local compact ABI unsupported")
        if not (_words(oldpart[1], newpart[1]) and all(_guard(x) for x in (hb, ib, wb, oldpart, newpart))
                and all(_words(a, b) for a, b in zip((h, ids, weights, gate), saved))):
            raise RuntimeError("local compact actual-selected part/guard/read-only differs")

    # Every routed expert and both shared halves at every density, including
    # zero/empty and a nonmultiple128 tail, against actual selected old tile.
    for first in range(0, 128, topk):
        for case in range(7):
            change(case, first)
            check_parts()
    # Actual original packed-up generated activations on private changed inputs.
    from .nemotron_h_mtp_packed import packed_moe_up
    up, _, shared, _, _, _ = spec.weights
    for case in range(3):
        change(case + 1, case * 43)
        x = (torch.randn((1, 2688), device=device, generator=gen) * (.001, 1., 32.)[case]).to(h.dtype)
        packed_moe_up[(triton.cdiv(1856, k._PK_UP_BN), slots)](
            x, *up.planes, *shared.planes, ids, h, 1856, 2688, 1856 * 2688, 2688, 2688,
            gate, TOPK=topk, BN=k._PK_UP_BN, BK=k._MOE_UP_BK,
            EVICT=k._EVICT_W, GATED=True, num_warps=4 * k._PK_UP_WN,
            num_stages=k._MOE_UP_STAGES, launch_pdl=spec.pdl, WN=k._PK_UP_WN,
            WORD=spec.word)
        check_parts()
    ratios = []
    for m in (1, 5, 8):
        residual = residuals[m]
        old = [_guarded((m, 2688), torch.bfloat16, device) for _ in range(8)]
        new = [_guarded((m, 2688), torch.bfloat16, device) for _ in range(8)]
        for case in range(7):
            change(case, case * 19)
            for r in (None, residual):
                saved = tuple(t.clone() for t in (h, ids, weights, gate, residual))
                _old_full(spec, h, ids, weights, old[0][1], r)
                _new_full(spec, h, ids, weights, new[0][1], r)
                if not (_words(old[0][1], new[0][1]) and _guard(old[0]) and _guard(new[0])
                        and all(_words(a, b) for a, b in zip((h, ids, weights, gate, residual), saved))):
                    raise RuntimeError("local compact complete down/residual differs")
        change(3)
        for pair in old:
            _old_full(spec, h, ids, weights, pair[1], residual)
        for pair in new:
            _new_full(spec, h, ids, weights, pair[1], residual)
        torch.cuda.synchronize(device)
        graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
        for arm, pairs in enumerate((old, new)):
            with torch.cuda.graph(graphs[arm], pool=torch.cuda.graph_pool_handle()):
                for pair in pairs:
                    (_old_full if arm == 0 else _new_full)(spec, h, ids, weights, pair[1], residual)
            if len({pair[1].data_ptr() for pair in pairs}) != 8:
                raise RuntimeError("local compact graph full outputs alias")
        for case in range(7):
            change(case, case * 19)
            saved = tuple(t.clone() for t in (h, ids, weights, gate, residual))
            for _ in range(2):
                graphs[0].replay(); graphs[1].replay()
                if not (all(_words(a[1], b[1]) and _guard(a) and _guard(b) for a, b in zip(old, new))
                        and all(_guard(x) for x in (hb, ib, wb))
                        and all(_words(a, b) for a, b in zip((h, ids, weights, gate, residual), saved))):
                    raise RuntimeError("local compact changed graph words/guards/read-only differ")
        pressure = torch.empty(64 << 20, dtype=torch.uint8, device=device)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for case in range(6):
            change(case)
            for cold in (False, True):
                samples = [[], []]
                for trial in range(5):
                    for arm in ((0, 1) if trial % 2 == 0 else (1, 0)):
                        elapsed = 0.0
                        for repeat in range(4):
                            if cold:
                                pressure.fill_(case + trial + repeat)
                            begin.record(); graphs[arm].replay(); end.record(); end.synchronize()
                            elapsed += begin.elapsed_time(end)
                        if not elapsed > 0:
                            raise RuntimeError("local compact invalid event interval")
                        samples[arm].append(elapsed)
                a, b = (statistics.median(v) for v in samples)
                ratios.append(b / a)
    if statistics.median(ratios) >= .995 or max(ratios) >= 1.01:
        raise RuntimeError(f"local compact complete cost refused topk={topk} ratios={ratios}")
    return statistics.median(ratios), max(ratios)


@torch.no_grad()
def prepare_all(module):
    """After actual screen and original packed owners exist; remote startup only."""
    if not _ENABLED or torch.cuda.is_current_stream_capturing():
        return
    try:
        import pkcompact
        if not pkcompact.available():
            return
    except ImportError:
        return
    from . import nemotron_h_mtp_kernels as k
    for layer in module.modules():
        if not getattr(layer, "_fast_moe", False):
            continue
        spec = _owner(layer)
        if spec is None:
            print("MTP_DOWN_LOCAL_COMPACT unsupported actual owner; original kept", flush=True)
            continue
        original_gate = k._GATE
        k._GATE = torch.zeros(4, dtype=torch.int32, device=spec.device)
        saved_owners = tuple(t.clone() for t in spec.tensors)
        safe, metrics = [], {}
        try:
            for topk in spec.topks:
                try:
                    metrics[topk] = _qualify(spec, topk)
                    safe.append(topk)
                except Exception as exc:
                    print(f"MTP_DOWN_LOCAL_COMPACT topk={topk} refused: {exc!r}", flush=True)
        finally:
            try:
                # A failed private probe can have queued a side-stream prefetch.
                # Keep its private gate alive until all such readers have ended.
                torch.cuda.synchronize(spec.device)
            finally:
                k._GATE = original_gate
        if not all(_words(a, b) for a, b in zip(spec.tensors, saved_owners)):
            raise RuntimeError("local compact immutable actual owner weights modified")
        if safe:
            spec.topks = tuple(safe)
            _ENTRIES[spec.key] = spec
        print(f"MTP_DOWN_LOCAL_COMPACT admitted topks={safe} actualoldtiles={spec.flags} ratios={metrics}", flush=True)

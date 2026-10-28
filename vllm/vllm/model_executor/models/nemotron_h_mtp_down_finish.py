"""Owner-qualified compact-down finish for an explicitly private full-step graph."""

import triton
import triton.language as tl


@triton.jit
def compact_and_clear(h, indices, values, counts, counters,
                      I: tl.constexpr, BLOCK: tl.constexpr,
                      launch_pdl: tl.constexpr = False):
    # This duplicates the original _compact body and adds the slot-zero clear.
    # The same graph replay must own counters exclusively and execute this node
    # before the dependent pkfinish grid. This contract is not yet proven.
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    slot = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    value = tl.load(h + slot * I + k, mask=k < I, other=0.0)
    live = (k < I) & (value != 0.0)
    position = tl.cumsum(live.to(tl.int32), axis=0) - 1
    tl.store(indices + slot * I + position, k, mask=live)
    tl.store(values + slot * I + position, value, mask=live)
    tl.store(counts + slot, tl.sum(live.to(tl.int32), axis=0))
    tl.store(counters + k, 0, mask=(slot == 0) & (k < 42))


# The following functions are for official remote admission only. Only
# immutable owner identities and booleans may survive admission.
import math
import statistics
from types import SimpleNamespace
import torch


def _same(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _guarded(shape, dtype, device, byte):
    size = 1
    for n in shape:
        size *= n
    owner = torch.empty(size + 128, dtype=dtype, device=device)
    owner.view(torch.uint8).fill_(byte)
    return owner, owner[64:-64].view(shape), byte


def _guard_ok(box):
    owner, _, byte = box
    return bool(torch.all(owner[:64].view(torch.uint8) == byte).item()
                and torch.all(owner[-64:].view(torch.uint8) == byte).item())


def _launch_ref(h, ids, weights, residual, pw, ps, pdl, routed_scale,
                ref_meta, ref_part, ref_y):
    from .nemotron_h_mtp_compact import _compact, _down_combine
    import pkslots
    slots = ids.numel() + 2
    _compact[(slots,)](h, *ref_meta, I=1856, BLOCK=2048, num_warps=4, launch_pdl=pdl)
    if not pkslots.run(*ref_meta, pw.planes, ps.planes, ids, ref_part[1], pdl):
        raise RuntimeError("original pkslots refused private qualification")
    _down_combine[(triton.cdiv(2688, 256),)](
        ref_part[1], weights, ref_y[1], M=ref_y[1].shape[0], H=2688,
        ROUTED_SCALE=routed_scale, TOPK=ids.numel(), S=2, BN=256, num_warps=4,
        launch_pdl=pdl, residual_ptr=residual, ADD_RES=residual is not None)


def _launch_new(h, ids, weights, residual, pw, ps, pdl, routed_scale,
                new_meta, new_part, new_y, counters):
    import pkfinish
    slots = ids.numel() + 2
    compact_and_clear[(slots,)](h, *new_meta, counters, I=1856, BLOCK=2048,
                               num_warps=4, launch_pdl=pdl)
    if not pkfinish.run(*new_meta, pw.planes, ps.planes, ids, new_part[1], counters,
                        weights, residual, new_y[1], routed_scale, pdl):
        raise RuntimeError("pkfinish refused private qualification")


def _synthetic(device, shared, raw):
    n = (2 if shared else 1) * 1856 * 2688
    pattern = torch.arange(256, device=device, dtype=torch.int32).to(torch.uint8)
    sm = pattern.repeat(n // 256)
    ex = pattern.repeat((n // 2) // 256)
    base = torch.full((n // 128,), 80, dtype=torch.uint8, device=device)
    ri = torch.full((n // 128,), -1, dtype=torch.int32, device=device)
    raw_words = torch.tensor([0, -32768, 1, -32767, 32640, -128, 32704, -64,
                              16256, -16512, 16384, -16384, 128, -32640, 32767, -1],
                             dtype=torch.int16, device=device).repeat(8).view(torch.uint16)
    if raw:
        ri[::2] = 0
    return sm, ex, base, ri, raw_words


@torch.no_grad()
def qualify(w, sd, pw, ps, topk, rows, add_residual, pdl, routed_scale):
    device = w.device
    slots = topk + 2
    gen = torch.Generator(device=device).manual_seed(202610289)
    h = torch.empty((slots, 1856), device=device, dtype=torch.bfloat16)
    ids = torch.empty(topk, device=device, dtype=torch.int32)
    weights = torch.empty(topk, device=device, dtype=torch.float32)
    residual = (torch.empty((rows, 2688), device=device, dtype=torch.bfloat16)
                if add_residual else None)
    def meta():
        return (torch.empty((slots, 1856), dtype=torch.int32, device=device),
                torch.empty((slots, 1856), dtype=torch.bfloat16, device=device),
                torch.empty(slots, dtype=torch.int32, device=device))
    ref_meta, new_meta = meta(), meta()
    ref_part = _guarded((slots, 2688), torch.float32, device, 0x7f)
    new_part = _guarded((slots, 2688), torch.float32, device, 0xa5)
    ref_y = _guarded((rows, 2688), torch.bfloat16, device, 0x7f)
    new_y = _guarded((rows, 2688), torch.bfloat16, device, 0xa5)
    counters_box = _guarded((42,), torch.int32, device, 0x5a)
    counters = counters_box[1]
    boxes = (ref_part, new_part, ref_y, new_y, counters_box)
    plane_samples = []
    for plane in (*pw.planes, *ps.planes):
        flat = plane.view(torch.uint8).reshape(-1)
        n = flat.numel()
        for offset in (0, max(0, n // 2 - 128), max(0, n - 256)):
            view = flat[offset:min(offset + 256, n)]
            plane_samples.append((view, view.clone()))

    def change(case, first):
        ids.copy_((torch.arange(topk, device=device, dtype=torch.int32) + first) % 128)
        density = (0.0, 0.015625, 0.125, 0.5, 1.0)[case % 5]
        h.copy_((torch.randn(h.shape, device=device, generator=gen).relu().square()
                 * (torch.rand(h.shape, device=device, generator=gen) < density))
                .to(torch.bfloat16))
        h.view(torch.int16)[:, :4] = torch.tensor([0, -32768, 1, -32767],
                                                  dtype=torch.int16, device=device)
        if density == 0.0:
            h.zero_()
        weights.copy_(torch.rand(weights.shape, device=device, generator=gen))
        if residual is not None:
            residual.copy_(torch.randn(residual.shape, device=device, generator=gen).to(torch.bfloat16))

    def launch_ref():
        _launch_ref(h, ids, weights, residual, pw, ps, pdl, routed_scale,
                    ref_meta, ref_part, ref_y)

    def launch_new():
        _launch_new(h, ids, weights, residual, pw, ps, pdl, routed_scale,
                    new_meta, new_part, new_y, counters)

    def poison():
        for owner, _, byte in boxes:
            owner.view(torch.uint8).fill_(byte)

    def check():
        if not (_same(ref_part[1], new_part[1]) and _same(ref_y[1], new_y[1])
                and _same(ref_meta[2], new_meta[2]) and all(_guard_ok(x) for x in boxes)
                and bool(torch.all(counters == slots).item())):
            raise RuntimeError("pkfinish private output, counts or guards differ")

    # One rotating sample per expert. Input owners must remain byte-identical.
    for case, first in enumerate(range(0, 128, topk)):
        change(case, first)
        inputs = (h, ids, weights) + (() if residual is None else (residual,))
        saved = tuple(t.clone() for t in inputs)
        poison()
        launch_ref()
        launch_new()
        check()
        if not all(_same(a, b) for a, b in zip(inputs, saved)):
            raise RuntimeError("pkfinish altered private input")

    # Mixed per-slot count tails, including empty and full slots.
    for offset in range(3):
        h.zero_()
        for slot in range(slots):
            count = (0, 1, 63, 64, 65, 127, 128, 1855, 1856)[(slot + 3 * offset) % 9]
            h[slot, :count] = 1.0
        poison()
        launch_ref()
        launch_new()
        check()

    # Explicit BF16 nonfinite values, signed zeros and subnormals.
    h.fill_(1.0)
    h.view(torch.int16)[:, :8] = torch.tensor(
        [0, -32768, 1, -32767, 32640, -128, 32704, -64],
        dtype=torch.int16, device=device)
    weights.copy_(torch.linspace(0.0, 1.0, topk, device=device, dtype=torch.float32))
    if residual is not None:
        residual.view(torch.int16)[:, :8] = torch.tensor(
            [0, -32768, 1, -32767, 32640, -128, 32704, -64],
            dtype=torch.int16, device=device)
    poison()
    launch_ref()
    launch_new()
    check()

    # The synthetic routed owner has one expert; all ids are zero. Exercise
    # mixed raw/coded blocks and compare all planes against preserved copies.
    actual_pw, actual_ps = pw, ps
    ids.zero_()
    for raw in (False, True):
        routed, shared = _synthetic(device, False, raw), _synthetic(device, True, raw)
        saved_planes = tuple(t.clone() for t in (*routed, *shared))
        pw, ps = SimpleNamespace(planes=routed), SimpleNamespace(planes=shared)
        for offset in range(3):
            h.zero_()
            for slot in range(slots):
                count = (0, 1, 63, 64, 65, 127, 128, 1855, 1856)[(slot + 3 * offset) % 9]
                h[slot, :count] = 1.0
            poison()
            launch_ref()
            launch_new()
            check()
        if not all(_same(a, b) for a, b in zip((*routed, *shared), saved_planes)):
            raise RuntimeError("pkfinish changed synthetic weight planes")
        del routed, shared, saved_planes
    pw, ps = actual_pw, actual_ps
    change(1, 0)

    torch.cuda.synchronize(device)
    graphs = []
    for launch in (launch_ref, launch_new):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
            launch()
        graphs.append(graph)
    for case in range(10):
        change(case, 17 * case)
        poison()
        for graph in graphs:
            graph.replay()
        check()

    pressure = torch.empty(64 * (1 << 20), dtype=torch.uint8, device=device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ratios, old_total, new_total = [], 0.0, 0.0
    for case in (0, 1, 3, 4, 7):
        change(case, 13 * case)
        for cold in (False, True):
            times = [[], []]
            for trial in range(3):
                for arm in ((0, 1) if trial % 2 == 0 else (1, 0)):
                    graphs[arm].replay()
                    elapsed = 0.0
                    for repeat in range(8):
                        if cold:
                            pressure.fill_(case + trial + repeat)
                        start.record()
                        graphs[arm].replay()
                        end.record()
                        end.synchronize()
                        elapsed += start.elapsed_time(end)
                    if elapsed <= 0:
                        raise RuntimeError("pkfinish invalid event timing")
                    times[arm].append(elapsed / 8.0)
            old_m, new_m = (statistics.median(t) for t in times)
            ratios.append(new_m / old_m)
            old_total += old_m
            new_total += new_m
    metrics = (statistics.median(ratios), new_total / old_total, max(ratios))
    if not (metrics[0] < 0.99 and metrics[1] < 0.99 and metrics[2] < 1.01):
        raise RuntimeError(f"pkfinish full-chain timing refused {metrics}")
    # A second captured candidate uses entirely independent mutable owners.
    # Replay both on different streams, after independent reference results are
    # recorded, to detect unintended counter/part/output sharing.
    h2, ids2, weights2 = h.clone(), ids.clone(), weights.clone()
    residual2 = None if residual is None else residual.clone()
    meta2 = meta()
    part2 = _guarded((slots, 2688), torch.float32, device, 0x36)
    y2 = _guarded((rows, 2688), torch.bfloat16, device, 0x63)
    counter2 = _guarded((42,), torch.int32, device, 0x96)
    def launch_new2():
        _launch_new(h2, ids2, weights2, residual2, pw, ps, pdl, routed_scale,
                    meta2, part2, y2, counter2[1])
    launch_new2()
    torch.cuda.synchronize(device)
    graph2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph2, pool=torch.cuda.graph_pool_handle()):
        launch_new2()
    stream1, stream2 = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    current = torch.cuda.current_stream(device)
    for case in range(5):
        change(case, 7 * case)
        h2.copy_(h.roll(1, dims=1))
        ids2.copy_((ids + 41) % 128)
        weights2.copy_(weights.roll(1, dims=0))
        if residual is not None:
            residual2.copy_(residual.roll(1, dims=1))
        launch_ref()
        expected_part1, expected_y1 = ref_part[1].clone(), ref_y[1].clone()
        _launch_ref(h2, ids2, weights2, residual2, pw, ps, pdl, routed_scale,
                    ref_meta, ref_part, ref_y)
        expected_part2, expected_y2 = ref_part[1].clone(), ref_y[1].clone()
        poison()
        for owner, _, byte in (part2, y2, counter2):
            owner.view(torch.uint8).fill_(byte)
        stream1.wait_stream(current)
        stream2.wait_stream(current)
        with torch.cuda.stream(stream1):
            graphs[1].replay()
        with torch.cuda.stream(stream2):
            graph2.replay()
        stream1.synchronize()
        stream2.synchronize()
        if not (_same(new_part[1], expected_part1) and _same(new_y[1], expected_y1)
                and _same(part2[1], expected_part2) and _same(y2[1], expected_y2)
                and all(_guard_ok(x) for x in (*boxes, part2, y2, counter2))
                and bool(torch.all(counters == slots).item())
                and bool(torch.all(counter2[1] == slots).item())):
            raise RuntimeError("pkfinish concurrent independent graph owners differ")
    if not all(_same(view, saved) for view, saved in plane_samples):
        raise RuntimeError("pkfinish altered sampled immutable owner bytes")
    return metrics

# Narrow use to the explicitly private full-step graph capture. Generic vLLM
# wrappers may share a global graph pool, so they do not enter this scope.
from contextlib import contextmanager
from contextvars import ContextVar

_PRIVATE_CAPTURE = ContextVar("mtp_pkfinish_private_capture", default=False)
_DECISIONS = {}


@contextmanager
def private_full_step_capture():
    token = _PRIVATE_CAPTURE.set(True)
    try:
        yield
    finally:
        _PRIVATE_CAPTURE.reset(token)


@torch.no_grad()
def prepare_for_owner(packed, routed_scale):
    import pkfinish
    import pkslots
    from .nemotron_h_mtp_down_packed import _PACKS, _PKSLOTS_OK
    from .nemotron_h_mtp_packed import word_load_enabled
    from .nemotron_h_mtp_kernels import mtp_pdl
    w, sd = packed
    if not (math.isfinite(float(routed_scale)) and pkfinish.available() and pkslots.available()
            and torch.cuda.get_device_capability(w.device) == (12, 1)):
        return
    owner = (w.data_ptr(), sd.data_ptr())
    entry = _PACKS.get(owner)
    if entry is None or entry[0] is not w or entry[1] is not sd:
        return
    pw, ps, flags = entry[2:]
    word, pdl = word_load_enabled(w.device), bool(mtp_pdl())
    for topk in (3, 6):
        if flags.get((topk, word)) != (64, 4) or not _PKSLOTS_OK.get((owner, topk, word)):
            continue
        for add_residual in (False, True):
            key = (*owner, topk, 1, add_residual, word, pdl, float(routed_scale))
            if key in _DECISIONS:
                continue
            _DECISIONS[key] = (w, sd, pw, ps, False)
            try:
                metrics = qualify(w, sd, pw, ps, topk, 1, add_residual, pdl, float(routed_scale))
                _DECISIONS[key] = (w, sd, pw, ps, True)
                print(f"MTP_DOWN_PKFINISH admitted topk={topk} residual={add_residual} ratios={metrics}", flush=True)
            except Exception as exc:
                print(f"MTP_DOWN_PKFINISH refused topk={topk} residual={add_residual}: {exc!r}", flush=True)


def maybe_finish(h, packed, ids, weights, y, routed_scale, residual, indices, values, counts, pdl):
    if not (_PRIVATE_CAPTURE.get() and torch.cuda.is_current_stream_capturing()):
        return False
    import pkfinish
    from .nemotron_h_mtp_down_packed import _PACKS
    from .nemotron_h_mtp_packed import word_load_enabled
    w, sd = packed
    entry = _PACKS.get((w.data_ptr(), sd.data_ptr()))
    if entry is None or entry[0] is not w or entry[1] is not sd:
        return False
    pw, ps = entry[2:4]
    word = word_load_enabled(w.device)
    key = (w.data_ptr(), sd.data_ptr(), ids.numel(), y.shape[0], residual is not None,
           word, bool(pdl), float(routed_scale))
    decision = _DECISIONS.get(key)
    if decision is None or not decision[4] or any(a is not b for a, b in zip(decision[:4], (w, sd, pw, ps))):
        return False
    # Every mutable output/scratch must be private relative to h and residual.
    if not all(pkfinish._disjoint(y, tensor) for tensor in (h, w, sd)):
        return False
    part = torch.empty((ids.numel() + 2, 2688), dtype=torch.float32, device=y.device)
    counters = torch.empty((42,), dtype=torch.int32, device=y.device)
    compact_and_clear[(ids.numel() + 2,)](h, indices, values, counts, counters,
                                         I=1856, BLOCK=2048, num_warps=4, launch_pdl=pdl)
    return pkfinish.run(indices, values, counts, pw.planes, ps.planes, ids, part,
                        counters, weights, residual, y, routed_scale, pdl)

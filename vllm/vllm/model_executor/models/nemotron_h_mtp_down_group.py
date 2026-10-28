"""Official-remote-only qualification of four-lane pkslots metadata broadcasting.

Original pkslots must first be admitted for the actual immutable owner. This
module never replaces its existing gates, packing, fallback or arithmetic.
"""
import os
import statistics

import torch

_ENABLED = os.environ.get("MTP_DOWN_PKGROUP", "1") != "0"
_DECISIONS = {}


def _words(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _guarded(slots, device):
    owner = torch.empty(slots * 2688 + 128, dtype=torch.float32, device=device)
    view = owner[64:-64].view(slots, 2688)
    return owner, view


def _poison(pair, byte):
    pair[0].view(torch.uint8).fill_(byte)


def _guard(pair, byte):
    return bool(torch.all(pair[0][:64].view(torch.uint8) == byte).item()
                and torch.all(pair[0][-64:].view(torch.uint8) == byte).item())


def _run(lib, indices, values, counts, rp, sp, ids, output, pdl):
    if not lib.run(indices, values, counts, rp, sp, ids, output, pdl):
        raise RuntimeError("pkgroup qualification launch refused its private contract")


def _case(oldlib, newlib, indices, values, counts, rp, sp, ids, old, new, pdl):
    inputs = (indices, values, counts, ids)
    saved = tuple(t.clone() for t in inputs)
    _poison(old, 0x7F)
    _poison(new, 0xA5)
    _run(oldlib, *inputs[:3], rp, sp, ids, old[1], pdl)
    _run(newlib, *inputs[:3], rp, sp, ids, new[1], pdl)
    if not (_words(old[1], new[1]) and _guard(old, 0x7F) and _guard(new, 0xA5)
            and all(_words(a, b) for a, b in zip(inputs, saved))):
        raise RuntimeError("pkgroup full owner/read-only/guard words differ")


def _synthetic(device, shared=False, raw=False):
    # Private decoder oracle, not a production weight or cached activation.
    # Only routed id0 is used; shared has both original halves. RAW is a
    # repeated exact128-word block, including signed zero/subnormal/Inf/NaN.
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
def _qualify(w, sd, pw, ps, topk, pdl, oldlib, newlib):
    from .nemotron_h_mtp_compact import _compact
    device = w.device
    gen = torch.Generator(device=device).manual_seed(202610278)
    slots = topk + 2
    indices = torch.empty((slots, 1856), dtype=torch.int32, device=device)
    values = torch.empty((slots, 1856), dtype=torch.bfloat16, device=device)
    counts = torch.empty(slots, dtype=torch.int32, device=device)
    ids = torch.empty(topk, dtype=torch.int32, device=device)
    old, new = _guarded(slots, device), _guarded(slots, device)
    rp, sp = pw.planes, ps.planes
    # Sample original planes without duplicating potentially multi-GiB owners.
    plane_samples = []
    for plane in (*rp, *sp):
        flat = plane.view(torch.uint8).reshape(-1)
        n = flat.numel()
        for offset in (0, max(0, n // 2 - 128), max(0, n - 256)):
            view = flat[offset:min(offset + 256, n)]
            plane_samples.append((view, view.clone()))

    def compact(h):
        _compact[(slots,)](h, indices, values, counts, I=1856,
                          BLOCK=2048, num_warps=4)

    # Entire actual routed owner, both shared halves, sparse/dense/empty/tails,
    # finite scales plus signed zeros, subnormal, Inf and NaN activations.
    for first in range(0, 128, topk):
        ids.copy_((torch.arange(topk, device=device, dtype=torch.int32) + first) % 128)
        for density in ((0.0, 0.015625, 0.125, 0.5, 1.0)[(first // topk) % 5],):
            for scale in ((0.001, 1.0, 32.0)[(first // topk) % 3],):
                h = (torch.randn((slots, 1856), device=device, generator=gen)
                     .relu().square() * scale).to(torch.bfloat16)
                h *= (torch.rand(h.shape, device=device, generator=gen) < density)
                h.view(torch.int16)[:, :4] = torch.tensor(
                    [0, -32768, 1, -32767], dtype=torch.int16, device=device)
                if density == 0.0:
                    h.zero_()
                compact(h)
                _case(oldlib, newlib, indices, values, counts, rp, sp, ids, old, new, pdl)

    # Explicit special values on real weights; the rotating density samples
    # above cover all experts, but do not guarantee a nonfinite case.
    ids.zero_()
    h = torch.ones((slots, 1856), device=device, dtype=torch.bfloat16)
    h.view(torch.int16)[:, :8] = torch.tensor(
        [0, -32768, 1, -32767, 32640, -128, 32704, -64],
        dtype=torch.int16, device=device)
    compact(h)
    _case(oldlib, newlib, indices, values, counts, rp, sp, ids, old, new, pdl)

    ids.zero_()
    for mixed_raw in (False, True):
        sr, ss = _synthetic(device, raw=mixed_raw), _synthetic(device, True, mixed_raw)
        synthetic_saved = tuple(t.clone() for t in (*sr, *ss))
        for count in (0, 1, 63, 64, 65, 127, 128, 129, 1855, 1856):
            indices.copy_(torch.arange(1856, device=device, dtype=torch.int32)
                          .view(1, -1).expand(slots, -1))
            values.copy_(torch.randn(values.shape, device=device, generator=gen)
                         .to(torch.bfloat16))
            counts.fill_(count)
            _case(oldlib, newlib, indices, values, counts, sr, ss, ids, old, new, pdl)
        if not all(_words(a, b) for a, b in zip((*sr, *ss), synthetic_saved)):
            raise RuntimeError("pkgroup mutates private decoder planes")
        del sr, ss, synthetic_saved

    # Capture compact plus slot execution together; h_graph is updated before
    # replay while its address and the dependent metadata buffers stay fixed.
    h_graph = torch.empty((slots, 1856), device=device, dtype=torch.bfloat16)
    h_graph.zero_()

    # Fixed-address input metadata changes between repeated graph replays.
    # Eight distinct complete output owners in each graph detect missing stores.
    old_outputs = [_guarded(slots, device) for _ in range(8)]
    new_outputs = [_guarded(slots, device) for _ in range(8)]
    for pair in old_outputs:
        _poison(pair, 0x7F)
    for pair in new_outputs:
        _poison(pair, 0xA5)
    for lib, outputs in ((oldlib, old_outputs), (newlib, new_outputs)):
        for pair in outputs:
            _run(lib, indices, values, counts, rp, sp, ids, pair[1], pdl)
    torch.cuda.synchronize(device)
    graphs = []
    for lib, outputs in ((oldlib, old_outputs), (newlib, new_outputs)):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
            for pair in outputs:
                compact(h_graph)
                _run(lib, indices, values, counts, rp, sp, ids, pair[1], pdl)
        graphs.append(graph)

    def change(case):
        ids.copy_((torch.arange(topk, device=device, dtype=torch.int32)
                   + 17 * case) % 128)
        h = (torch.randn((slots, 1856), device=device, generator=gen)
             .relu().square()).to(torch.bfloat16)
        h *= (torch.rand(h.shape, device=device, generator=gen)
              < (0.0, 0.015625, 0.125, 0.5, 1.0)[case % 5])
        h_graph.copy_(h)
        compact(h_graph)

    for case in range(10):
        change(case)
        snapshots = tuple(t.clone() for t in (indices, values, counts, ids))
        for _ in range(2):
            graphs[0].replay()
            graphs[1].replay()
            for a, b in zip(old_outputs, new_outputs):
                if not (_words(a[1], b[1]) and _guard(a, 0x7F) and _guard(b, 0xA5)):
                    raise RuntimeError("pkgroup changed-graph output/guard differs")
            if not all(_words(a, b) for a, b in zip((indices, values, counts, ids), snapshots)):
                raise RuntimeError("pkgroup changed-graph inputs mutated")

    pressure = torch.empty(64 * (1 << 20), dtype=torch.uint8, device=device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ratios, old_total, new_total = [], 0.0, 0.0
    for case in (0, 1, 3, 4, 7):
        change(case)
        for cold in (False, True):
            times = [[], []]
            for trial in range(3):
                for arm in ((0, 1) if trial % 2 == 0 else (1, 0)):
                    graphs[arm].replay()
                    elapsed_sum = 0.0
                    for repeat in range(8):
                        if cold:
                            pressure.fill_(trial + case + repeat)
                        start.record()
                        graphs[arm].replay()
                        end.record()
                        end.synchronize()
                        elapsed_sum += start.elapsed_time(end)
                    elapsed = elapsed_sum / 64.0
                    if not elapsed > 0.0:
                        raise RuntimeError("pkgroup nonpositive CUDA event interval")
                    times[arm].append(elapsed)
            old_m, new_m = (statistics.median(t) for t in times)
            ratios.append(new_m / old_m)
            old_total += old_m
            new_total += new_m
    if not all(_words(view, saved) for view, saved in plane_samples):
        raise RuntimeError("pkgroup altered sampled immutable owner bytes")
    metrics = (statistics.median(ratios), new_total / old_total, max(ratios))
    if not (metrics[0] < 0.99 and metrics[1] < 0.99 and metrics[2] < 1.01):
        raise RuntimeError(f"pkgroup full-work warm/pressure ratio refused {metrics}")
    return metrics


@torch.no_grad()
def prepare(w, sd, pw, ps, flags, word, pdl, original_ok):
    if not _ENABLED or torch.cuda.is_current_stream_capturing():
        return
    import pkslots
    import pkgroup
    if (not pkslots.available() or not pkgroup.available()
            or tuple(w.shape) != (128, 1856, 2688) or tuple(sd.shape) != (2, 1856, 2688)
            or torch.cuda.get_device_capability(w.device) != (12, 1)):
        print("MTP_DOWN_PKGROUP not prepared: library/owner/capability unsupported", flush=True)
        return
    owner = (w.data_ptr(), sd.data_ptr())
    eligible = 0
    for (topk, wd), tile in flags.items():
        if tile != (64, 4) or wd != word or not original_ok.get((owner, topk, word)):
            continue
        eligible += 1
        key = (*owner, topk, word, bool(pdl))
        if key in _DECISIONS:
            continue
        _DECISIONS[key] = (w, sd, pw, ps, False)
        try:
            metrics = _qualify(w, sd, pw, ps, topk, pdl, pkslots, pkgroup)
            _DECISIONS[key] = (w, sd, pw, ps, True)
            print(f"MTP_DOWN_PKGROUP admitted topk={topk} pdl={pdl} ratios={metrics}", flush=True)
        except Exception as exc:
            print(f"MTP_DOWN_PKGROUP refused topk={topk}: {exc!r}; original kept", flush=True)

    if not eligible:
        print("MTP_DOWN_PKGROUP not prepared: no original admitted64x4 owner", flush=True)


def maybe_run(w, sd, pw, ps, word, indices, values, counts, ids, part, pdl):
    if not _ENABLED:
        return False
    entry = _DECISIONS.get((w.data_ptr(), sd.data_ptr(), ids.numel(), word, bool(pdl)))
    if entry is None or not entry[4] or any(a is not b for a, b in zip(entry[:4], (w, sd, pw, ps))):
        return False
    import pkgroup
    # A real runtime launch error propagates. It is not a successful fallback.
    return pkgroup.run(indices, values, counts, pw.planes, ps.planes, ids, part, pdl)

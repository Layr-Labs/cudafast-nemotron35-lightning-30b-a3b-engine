# SPDX-License-Identifier: Apache-2.0
"""Official-remote-only qualification of an exact per-output MMA scheduling twin.

No request tensor is used here. The production config changes only at the end,
from actually selected653 to674 after byte/guard/graph/complete-call gates.
"""
import statistics

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
REFERENCE, CANDIDATE = 653, 674
_SENTINEL = 0x2BAD


def _bits(t):
    return t.view(torch.int16)


def _equal(a, b):
    return torch.equal(_bits(a), _bits(b))


def _guarded(rows, cols, device):
    backing = torch.empty((rows + 2, cols + 16), dtype=torch.bfloat16, device=device)
    _bits(backing).fill_(_SENTINEL)
    return backing, backing[1:rows + 1, 8:cols + 8]


def _production_guarded(rows, cols, device):
    # 256B front/back guards, with actual contiguous production row strides.
    backing = torch.empty(rows * cols + 256, dtype=torch.bfloat16, device=device)
    _bits(backing).fill_(_SENTINEL)
    return backing, backing[128:128 + rows * cols].view(rows, cols)


def _guards(backing, cols):
    b = _bits(backing)
    if backing.ndim == 1:
        return bool(torch.all(b[:128] == _SENTINEL)) and bool(torch.all(b[-128:] == _SENTINEL))
    return (bool(torch.all(b[0] == _SENTINEL))
            and bool(torch.all(b[-1] == _SENTINEL))
            and bool(torch.all(b[1:-1, :8] == _SENTINEL))
            and bool(torch.all(b[1:-1, cols + 8:] == _SENTINEL)))


def prepare(layer):
    from vllm.model_executor.layers.fused_moe.experts import humming_zskip as h

    z = h._zskip
    if (z is None or getattr(layer, "zskip_pair_config", None) != REFERENCE
            or torch.cuda.is_current_stream_capturing()
            or torch.cuda.get_device_capability() != (12, 1)):
        return
    packed = layer.zskip_pair_w2
    if packed.layout != "swap2":
        return
    old_prefix = bool(getattr(layer, "zskip_pair_prefix_ok", False))
    old_pdl = bool(getattr(z, "_pdl_on", False))
    admitted = False
    try:
        ref_info, new_info = z.config_info(REFERENCE), z.config_info(CANDIDATE)
        if ref_info["shape"] != (1, 3, 274) or new_info["shape"] != (1, 3, 338):
            return
        # Existing per-layer actual-weight/humming bit qualification, including
        # M1..8, independent/crowded routing and signed/empty halves.
        if not h._pair_check(layer, packed, CANDIDATE, ms=range(1, 9), ref_config=REFERENCE):
            return
        if old_prefix and not h._pair_check(
                layer, packed, CANDIDATE, prefix=True, ref_config=REFERENCE):
            return
        if bool(getattr(z, "_pdl_on", False)) != old_pdl:
            return
        dev = packed.w.device
        rows, k, n = 30, h.INTER, h.HIDDEN
        ab, act = _guarded(rows, k, dev)
        cb, candidate = _guarded(rows, n, dev)
        rb, reference = _guarded(rows, n, dev)
        count = torch.tensor([5], dtype=torch.int32, device=dev)
        prefix_api = z.prefix_available()
        gen = torch.Generator(device=dev)
        gen.manual_seed(19037)
        counters = [torch.zeros(4, dtype=torch.int32, device=dev) for _ in range(2)]
        metadata = None

        def route(shift, crowded):
            nonlocal metadata
            if crowded:
                ids = (torch.arange(6, dtype=torch.int32, device=dev) + shift) % 128
                ids = ids.view(1, 6).expand(5, 6).contiguous()
            else:
                ids = ((torch.arange(rows, dtype=torch.int32, device=dev) + shift) % 128).view(5, 6)
            metadata = h._align(ids, 16)

        def fill(case):
            if case == "edge":
                # Include signed zeros, BF16 subnormals, infinities and NaNs;
                # compare integer payloads against the actual selected kernel.
                v = torch.tensor([0, -32768, 1, -32767, 16256, -16512,
                                  32640, -128, 32705, -63, 127, -32641],
                                 dtype=torch.int16, device=dev)
                _bits(act).copy_(v.repeat((k + 11) // 12)[:k].view(1, k).expand(rows, k))
                return
            x = torch.randn((rows, k), generator=gen, device=dev)
            if case == "dense":
                x = x.relu().square()
            elif case == "signed":
                x = x * (torch.rand((rows, k), generator=gen, device=dev) < .5)
            else:
                x = x.relu().square() * (torch.rand((rows, k), generator=gen, device=dev) < .1)
            act.copy_(x)
            act[:, :16].zero_()
            act[:, 16:24].zero_()
            act[:, 40:48].zero_()
            act[0].zero_()
            b = _bits(act)
            b[b == 0] = -32768
            if case == "zero":
                b.fill_(-32768)

        def run(which, cap, stats=False):
            ids, experts, ntp = metadata
            z.down(act, ids, experts, ntp, packed,
                   reference if which == 0 else candidate,
                   mode=z.SKIP, config=REFERENCE if which == 0 else CANDIDATE,
                   max_blocks=rows, verify_rows=count if cap is not None else None,
                   stats=counters[which] if stats else None)
            if bool(getattr(z, "_pdl_on", False)) != old_pdl:
                raise ValueError("new config changed actual selected PDL policy")

        def compare(cap, stats=False):
            before = ab.clone()
            meta_before = tuple(t.clone() for t in metadata)
            for backing in (rb, cb):
                _bits(backing).fill_(_SENTINEL)
            if cap is not None:
                count.fill_(cap)
            for s in counters:
                s.zero_()
            run(0, cap, stats)
            run(1, cap, stats)
            torch.cuda.synchronize(dev)
            if (not _equal(rb, cb) or not _equal(ab, before)
                    or not _guards(rb, n) or not _guards(cb, n) or not _guards(ab, k)
                    or any(not torch.equal(a, b) for a, b in zip(metadata, meta_before))
                    or (stats and not torch.equal(counters[0], counters[1]))):
                raise ValueError("streamed MMA changed complete private backing, guards or fallback counts")

        # Every expert appears in the independent30-row batches. Separate
        # crowded/edge cases exercise all8MMA output tiles, empty halves and
        # negative-zero/dense-retry behavior; no captured request data.
        for shift in (0, 30, 60, 90, 120):
            route(shift, False)
            fill("sparse")
            compare(None, True)
        for crowded in (False, True):
            route(11, crowded)
            for case in ("zero", "dense", "signed", "edge"):
                fill(case)
                compare(None, True)
                if prefix_api:
                    for cap in (0, 1, 2, 3, 4, 5, 2147483647):
                        compare(cap, True)

        # Semantic guard cases above deliberately vary row padding. Timing
        # uses contiguous production strides1856/2688, not the padded fixtures.
        ab, act = _production_guarded(rows, k, dev)
        cb, candidate = _production_guarded(rows, n, dev)
        rb, reference = _production_guarded(rows, n, dev)
        pressure = torch.empty(64 << 20, dtype=torch.uint8, device=dev)
        ratios, totals = [], []
        cases = ((False, "sparse", None), (True, "sparse", 1),
                 (False, "signed", 3), (True, "dense", 5))
        for crowded, case, cap in cases:
            if cap is not None and not prefix_api:
                continue
            route(37, crowded)
            fill(case)
            compare(cap)
            graphs = []
            for which in (0, 1):
                for _ in range(2):
                    run(which, cap)
                torch.cuda.synchronize(dev)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
                    run(which, cap)
                graphs.append(graph)
            # Same fixed pointers, changed activation and count metadata. Both
            # graphs must overwrite the same complete output backing again.
            fill("signed" if case != "signed" else "dense")
            if cap is not None:
                count.fill_(2 if cap != 2 else 4)
            for backing in (rb, cb):
                _bits(backing).fill_(_SENTINEL)
            before = ab.clone()
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize(dev)
            if (not _equal(rb, cb) or not _equal(ab, before)
                    or not _guards(rb, n) or not _guards(cb, n)):
                raise ValueError("streamed MMA changed inputs/outputs under graph replay")
            fill(case)
            if cap is not None:
                count.fill_(cap)
            for cold in (False, True):
                samples = [[], []]
                for trial in range(5):
                    for which in ((0, 1) if trial % 2 == 0 else (1, 0)):
                        graph = graphs[which]
                        if cold:
                            pressure.fill_(trial + 1)
                        else:
                            for _ in range(20):
                                graph.replay()
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        repeats = 1 if cold else 32
                        begin.record()
                        for _ in range(repeats):
                            graph.replay()
                        end.record()
                        end.synchronize()
                        ms = begin.elapsed_time(end) / repeats
                        if not ms > 0:
                            raise ValueError("invalid complete-call timing")
                        samples[which].append(ms)
                med_ref, med_new = statistics.median(samples[0]), statistics.median(samples[1])
                ratios.append(med_new / med_ref)
                totals.append((med_new, med_ref))
        if (not ratios or statistics.median(ratios) >= .98 or max(ratios) >= 1.03
                or sum(t[0] for t in totals) / sum(t[1] for t in totals) >= .98):
            return
        if bool(getattr(z, "_pdl_on", False)) != old_pdl:
            return
        # Mutation only after the complete guard. Prefix policy stays exactly
        # as selected by the original finalizer; only M5 uses this config.
        layer.zskip_pair_config = CANDIDATE
        admitted = True
        logger.info("zskip streamed per-output MMA admitted: ref=%s new=%s ratios=%s prefix=%s",
                    ref_info, new_info, tuple(ratios), old_prefix)
    finally:
        # A failed optional PDL launch must not alter the retained baseline's
        # process-wide policy. This setter is the existing zskip C ABI.
        if bool(getattr(z, "_pdl_on", False)) != old_pdl:
            lib = z._load()
            if hasattr(lib, "zskip_set_pdl") and lib.zskip_set_pdl(int(old_pdl)) == 0:
                z._pdl_on = old_pdl
            else:
                raise RuntimeError("cannot restore original PDL launch policy")
        if not admitted:
            layer.zskip_pair_config = REFERENCE

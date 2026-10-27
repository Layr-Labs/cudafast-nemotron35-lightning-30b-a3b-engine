# SPDX-License-Identifier: Apache-2.0
"""Remote-only original lean653 versus immutable prepared-scale admission."""
from __future__ import annotations

import os
import statistics

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
_SENTINEL = 0x2BAD

I, H, TOPK = 1856, 2688, 6


def _bytes(x):
    return x.contiguous().reshape(-1).view(torch.uint8)


def _same(a, b):
    return torch.equal(_bytes(a), _bytes(b))


def _guarded(shape, dtype, device):
    n = 1
    for s in shape:
        n *= s
    pad = 128 if dtype == torch.bfloat16 else 64
    b = torch.empty(n + 2 * pad, dtype=dtype, device=device)
    b.view(torch.int16 if dtype == torch.bfloat16 else torch.int32).fill_(_SENTINEL)
    return b, b[pad:-pad].view(shape), pad


def _guards(b, pad):
    v = b.view(torch.int16 if b.dtype == torch.bfloat16 else torch.int32)
    return bool(torch.all(v[:pad] == _SENTINEL)) and bool(torch.all(v[-pad:] == _SENTINEL))


@torch.no_grad()
def qualify(layer, api, align):
    """Only on the actual old checked653 owner, after old prefix admission.

    No private graph, output or activation survives. Only an exact immutable
    derived scale plane and original packed owner remain after successful admission.
    """
    packed = getattr(layer, "zskip_pair_w2", None)
    if (os.environ.get("ZSKIP_SCALE_READY", "1") == "0"
            or getattr(layer, "zskip_pair_config", None) != 653
            or packed is None or packed.layout != "swap2"
            or torch.cuda.is_current_stream_capturing()):
        return None
    device = packed.w.device
    if torch.cuda.get_device_capability(device) != (12, 1):
        return None
    import scaleready
    if not scaleready.prepare(device):
        return None
    # Guard the new input-independent pack, compare complete original owners.
    scale_back = torch.full((2 * packed.s.numel() + 256,), 173, dtype=torch.uint8, device=device)
    scales = scale_back[128:-128]
    original_snapshots = [(x, x.clone()) for x in (packed.w, packed.s, packed.gs)]
    scaleready.pack(packed, scales)
    torch.cuda.current_stream(device).synchronize()
    if (not bool(torch.all(scale_back[:128] == 173)) or not bool(torch.all(scale_back[-128:] == 173))
            or any(not _same(a, b) for a, b in original_snapshots)):
        raise ValueError("prepared scale pack changed original owner or guards")
    owners = original_snapshots + [(scale_back, scale_back.clone())]
    gen = torch.Generator(device=device).manual_seed(2711673)
    pressure = torch.empty(64 << 20, device=device, dtype=torch.uint8)
    timings, ratios = [], []

    def readonly():
        if any(not _same(a, b) for a, b in owners):
            raise ValueError("prepared-scale scheduling changed packed owner")

    # The new production path is strictly M5. Other row counts keep original.
    original_lib = api._load()
    workspace, queue = api.scratch(device)
    pdl = api._pdl_on

    def original_down(act, sorted_ids, expert_ids, ntp, dst, mode, stats, prefix):
        rows = act.shape[0]
        args = (act.data_ptr(), act.stride(0), rows, sorted_ids.data_ptr(), expert_ids.data_ptr(),
                ntp.data_ptr(), rows, packed.w.data_ptr(), packed.s.data_ptr(), packed.gs.data_ptr(),
                dst.data_ptr(), dst.stride(0), packed.w_expert, packed.w_tile, packed.w_k,
                packed.s_expert, packed.s_tile, packed.s_g, mode, stats.data_ptr(), 653,
                workspace.data_ptr(), queue.data_ptr(), 0, None,
                torch.cuda.current_stream(device).cuda_stream)
        launch = original_lib.zskip_down
        if prefix is not None:
            launch = original_lib.zskip_down_prefix
            args += (prefix.data_ptr(),)
        rc = launch(*args)
        if rc:
            # Never mutate original global PDL state for this optional admission.
            raise RuntimeError("private original653 launch failed: " + str(rc))

    try:
        for m in (5,):
            rows = m * TOPK
            ab, act, ap = _guarded((rows, I), torch.bfloat16, device)
            sb, sorted_ids, sp = _guarded((rows * 16,), torch.int32, device)
            eb, expert_ids, ep = _guarded((rows,), torch.int32, device)
            nb, ntp, np = _guarded((1,), torch.int32, device)
            pb, prefix, pp = _guarded((1,), torch.int32, device)
            ob, old, op = _guarded((rows, H), torch.bfloat16, device)
            cb, new, cp = _guarded((rows, H), torch.bfloat16, device)
            stats_backing = []
            stats = []
            for _ in range(2):
                backing, view, pad = _guarded((4,), torch.int32, device)
                stats_backing.append((backing, pad))
                stats.append(view)
            metadata = ((ab, ap), (sb, sp), (eb, ep), (nb, np), (pb, pp))
            current_mode, with_prefix = api.SKIP, False

            def case(first, crowded, mode, density=.1, count=5):
                # Each token routes to six distinct experts; a crowded expert
                # has at most m<=8 live rows, the actual paired-kernel contract.
                ids = ((torch.arange(m, device=device, dtype=torch.int32)[:, None]
                        * (0 if crowded else 17)
                        + torch.arange(TOPK, device=device, dtype=torch.int32)[None, :]
                        + first) % 128).contiguous()
                si, ei, padded = align(ids, 16)
                live = int(padded.item())
                if live > rows * 16 or live % 16:
                    raise ValueError("private aligned metadata outside original contract")
                sorted_ids.fill_(rows)
                expert_ids.fill_(-1)
                sorted_ids[:live].copy_(si[:live])
                expert_ids[:live // 16].copy_(ei[:live // 16])
                ntp.copy_(padded)
                prefix.fill_(count)
                if mode == 1:
                    act.fill_(1.)
                elif mode == 2:
                    act.zero_()
                    act[:, ::2] = -0.
                else:
                    v = torch.randn((rows, I), device=device, generator=gen).relu().square()
                    v *= torch.rand(v.shape, device=device, generator=gen) < density
                    act.copy_(v * (.001 if mode == 3 else 32. if mode == 4 else 1.))
                    act[:, :16] = 0.
                    act[:, 16:24] = 0.
                    act[:, 40:48] = 0.
                    act[0].zero_()
                    bits = act.view(torch.int16)
                    bits[bits == 0] = -32768
                    bits[:, 48:50] = torch.tensor([1, -32767], dtype=torch.int16, device=device)
                    if mode == 5:
                        bits[:, 50:52] = torch.tensor([32640, -128], dtype=torch.int16, device=device)
                    if mode == 6:
                        bits[:, 50:52] = torch.tensor([32704, -64], dtype=torch.int16, device=device)

            graph_outputs = None

            def run(which, destination=None):
                live = prefix if with_prefix else None
                destination = (new if which else old) if destination is None else destination
                if which:
                    scaleready.down(act, sorted_ids, expert_ids, ntp, packed, scales,
                                    destination, mode=current_mode, max_blocks=rows,
                                    stats=stats[which], verify_rows=live, pdl=pdl)
                else:
                    original_down(act, sorted_ids, expert_ids, ntp, destination,
                                  current_mode, stats[which], live)

            def compare(graphs=None):
                saved = [x.clone() for x, _ in metadata]
                ob.view(torch.int16).fill_(_SENTINEL)
                cb.view(torch.int16).fill_(_SENTINEL)
                for x in stats:
                    x.zero_()
                if graphs:
                    for outputs in graph_outputs:
                        for backing, view, pad in outputs:
                            backing.view(torch.int16).fill_(_SENTINEL)
                for which in (0, 1):
                    graphs[which].replay() if graphs else run(which)
                if (not _same(ob, cb) or not _same(stats[0], stats[1])
                        or any(not _same(x, y) for (x, _), y in zip(metadata, saved))
                        or not all(_guards(x, pad) for x, pad in
                                   (*metadata, (ob, op), (cb, cp), *stats_backing))):
                    raise ValueError("prepared-scale scheduling full-output/stats/guard/read-only mismatch")
                if graphs:
                    for old_item, new_item in zip(*graph_outputs):
                        if (not _same(old_item[0], new_item[0])
                                or not _guards(old_item[0], old_item[2])
                                or not _guards(new_item[0], new_item[2])):
                            raise ValueError("prepared-scale scheduling changed-graph full destination mismatch")

            # All experts/all actual columns/all K groups, for both dense loads
            # and sparse masks; full activation rows exercise every raw nibble.
            for first in range(0, 128, TOPK):
                for crowded in (False, True):
                    case(first, crowded, 1)
                    compare()
                    case(first, crowded, 0, density=.5)
                    compare()
                if first == 0:
                    for mode in (0, 2, 3, 4, 5, 6):
                        case(first, True, mode)
                        compare()
                    current_mode = api.DENSE
                    case(first, False, 2)
                    compare()
                    current_mode = api.SKIP
            readonly()
            # Actual M5 can use original full rows or original confidence prefix.
            modes = (False, True) if getattr(layer, "zskip_pair_prefix_ok", False) else (False,)
            for with_prefix in modes:
                if with_prefix:
                    for count in (-1, 0, 1, 2, 3, 4, 5, 2147483647):
                        case(121, True, 0, count=count)
                        compare()
                case(0, False, 0)
                graphs = []
                graph_outputs = [[_guarded((rows, H), torch.bfloat16, device)
                                  for _ in range(8)] for _ in (0, 1)]
                for which in (0, 1):
                    for _ in range(2):
                        run(which)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
                        for backing, destination, pad in graph_outputs[which]:
                            run(which, destination)
                    graphs.append(graph)
                for first, crowded, mode, count in (
                    (125, False, 0, 1), (17, True, 1, 3), (93, False, 2, 0),
                    (43, True, 4, 5), (0, False, 6, 2147483647), (7, True, 0, 2)):
                    case(first, crowded, mode, count=count)
                    compare(graphs)
                    compare(graphs)
                for first, crowded, density, count in (
                    (17, False, .1, 5), (43, True, .5, 3), (93, False, 1., 1)):
                    case(first, crowded, 0, density, count)
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
                                reps = 1 if cold else 16
                                start.record()
                                for _ in range(reps):
                                    graphs[which].replay()
                                end.record()
                                end.synchronize()
                                elapsed = start.elapsed_time(end) / (reps * 8)
                                if not elapsed > 0:
                                    return None
                                samples[which].append(elapsed)
                        a, b = (statistics.median(s) for s in samples)
                        timings.append((a, b))
                        ratios.append(b / a)
                        compare(graphs)
            readonly()
        admitted = (bool(ratios) and statistics.median(ratios) < .995
                    and sum(b for a, b in timings) / sum(a for a, b in timings) < .995
                    and max(ratios) < 1.01)
        if api._pdl_on != pdl:
            raise ValueError("original PDL state changed during private admission")
        logger.info("zskip: immutable prepared-scale scheduling admitted=%s layer=%s",
                    admitted, getattr(layer, "layer_name", "?"))
        return (packed, scales, bool(getattr(layer, "zskip_pair_prefix_ok", False)), scaleready) if admitted else None
    except Exception as exc:
        logger.warning("zskip: prepared-scale scheduling unavailable (%r); original653 retained", exc)
        return None


def maybe_down(layer, api, a, sorted_ids, expert_ids, ntp, packed, out, verify_rows):
    """Only immutable per-owner admission; every other production call stays old."""
    admitted = getattr(layer, "_zskip_scale_ready", None)
    if (admitted is None or getattr(layer, "zskip_pair_config", None) != 653
            or a.shape != (30, I) or out.shape != (30, H)
            or admitted[0] is not packed or (verify_rows is not None and not admitted[2])):
        return False
    original, scales, prefix_ok, module = admitted
    module.down(a, sorted_ids, expert_ids, ntp, original, scales, out,
                mode=api.SKIP, max_blocks=30, verify_rows=verify_rows, pdl=api._pdl_on)
    return True

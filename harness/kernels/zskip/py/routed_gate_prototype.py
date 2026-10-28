"""Remote-startup qualification for routed_host_prototype.

Do not execute locally. Call only outside capture on the official worker,
using real layer weights.
Any exception or failed comparison must retain the original path.
"""
from __future__ import annotations

import statistics

import torch

from . import routed_host_prototype as fused


def _bits_equal(a, b):
    return torch.equal(a.contiguous().view(torch.int16),
                       b.contiguous().view(torch.int16))


def check_layer(layer, zskip, align, combine):
    """Return False unless complete fast/fallback outputs and workspaces match.

    The caller must additionally preserve the actual side-stream join.
    """
    if getattr(layer, "zskip_pair_config", None) != 663:
        return False
    packed = layer.zskip_pair_w2
    dev = packed.gs.device
    lib = zskip._load()
    if lib is None:
        return False
    for m in (5, 6):
        rows = m * 6
        for crowded in (False, True):
            ids = torch.tensor([[(29 * i + (0 if crowded else 17 * t)) % 128
                                 for i in range(6)] for t in range(m)],
                               dtype=torch.int32, device=dev)
            sorted_ids, expert_ids, padded = align(ids, 16)
            gen = torch.Generator(device=dev).manual_seed(7953 + m + int(crowded))
            raw = torch.randn((rows, 1856), generator=gen, device=dev)
            mask = torch.rand((rows, 1856), generator=gen, device=dev) < 0.1
            seed = (raw.relu().square() * mask).to(torch.bfloat16).contiguous()
            seed[:, :16] = 0
            seed[:, 16:24] = 0
            seed[:, 40:48] = 0
            seed[0].zero_()
            bits = seed.view(torch.int16)
            bits[bits == 0] = -32768
            # Distinguish NaN and infinity behavior from the zero-mask path.
            seed[1, 32] = float("nan")
            seed[2, 48] = float("inf")
            w = torch.rand((m, 6), generator=gen, device=dev, dtype=torch.float32)
            initial = torch.randn((rows, 2688), generator=gen, device=dev).to(torch.bfloat16)
            for count in (None, *range(1, m + 1)):
                prefix = None if count is None else torch.tensor([count], dtype=torch.int32, device=dev)
                for force in (False, True):
                    for graph_mode in (False, True):
                        a_ref, a_new = seed.clone(), seed.clone()
                        c_ref, c_new = initial.clone(), initial.clone()
                        # The actual caller aliases output and input workspace.
                        out_ref = a_ref.view(-1)[:m * 2688].view(m, 2688)
                        out_new = a_new.view(-1)[:m * 2688].view(m, 2688)
                        held = []

                        def reference():
                            zskip.down(a_ref, sorted_ids, expert_ids, padded, packed,
                                       c_ref, mode=zskip.SKIP, config=663,
                                       max_blocks=rows, verify_rows=prefix)
                            combine(inputs=c_ref.view(m, 6, 2688), topk_weights=w,
                                    topk_ids=ids, expert_map=None, outputs=out_ref)

                        def candidate():
                            h = fused.start(lib, a_new, sorted_ids, expert_ids, padded,
                                            packed, 663, c_new, w, out_new, prefix,
                                            force_fallback=force)
                            if h is None:
                                raise RuntimeError("prototype guards declined test case")
                            held.append(h)
                            fused.finish(h, out_new)

                        # Warm Triton's original combine before graph capture.
                        if graph_mode:
                            warm_a, warm_c = seed.clone(), initial.clone()
                            warm_out = torch.empty((m, 2688), dtype=torch.bfloat16, device=dev)
                            zskip.down(warm_a, sorted_ids, expert_ids, padded, packed,
                                       warm_c, mode=zskip.SKIP, config=663,
                                       max_blocks=rows, verify_rows=prefix)
                            combine(inputs=warm_c.view(m, 6, 2688), topk_weights=w,
                                    topk_ids=ids, expert_map=None, outputs=warm_out)
                            torch.cuda.synchronize(dev)
                            g_ref, g_new = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
                            with torch.cuda.graph(g_ref):
                                reference()
                            with torch.cuda.graph(g_new):
                                candidate()
                            # Reinitialize after capture; each graph runs once here.
                            a_ref.copy_(seed); a_new.copy_(seed)
                            c_ref.copy_(initial); c_new.copy_(initial)
                            g_ref.replay(); g_new.replay()
                        else:
                            reference(); candidate()
                        torch.cuda.synchronize(dev)
                        if not _bits_equal(out_ref, out_new) or not _bits_equal(c_ref, c_new):
                            return False
                        flag = int(held[-1][1][2].view(torch.int32).item())
                        if force and flag != 1:
                            return False
                        if not force and flag != 0:
                            return False
    return timing_layer(layer, zskip, align, combine)


def timing_layer(layer, zskip, align, combine):
    """Require full-chain graph wins on both verify shapes and routing regimes."""
    packed = layer.zskip_pair_w2
    dev = packed.gs.device
    lib = zskip._load()
    for m in (5, 6):
        rows = m * 6
        for crowded in (False, True):
            ids = torch.tensor([[(29 * i + (0 if crowded else 17 * t)) % 128
                                 for i in range(6)] for t in range(m)],
                               dtype=torch.int32, device=dev)
            sorted_ids, expert_ids, padded = align(ids, 16)
            gen = torch.Generator(device=dev).manual_seed(3781 + m + int(crowded))
            x = torch.randn((rows, 1856), generator=gen, device=dev)
            keep = torch.rand((rows, 1856), generator=gen, device=dev) < 0.1
            seed = (x.relu().square() * keep).to(torch.bfloat16).contiguous()
            initial = torch.randn((rows, 2688), generator=gen, device=dev).to(torch.bfloat16)
            weights = torch.rand((m, 6), generator=gen, device=dev, dtype=torch.float32)
            a0, a1, c0, c1 = seed.clone(), seed.clone(), initial.clone(), initial.clone()
            o0 = a0.view(-1)[:m * 2688].view(m, 2688)
            o1 = a1.view(-1)[:m * 2688].view(m, 2688)
            prefix = torch.tensor([m - 1], dtype=torch.int32, device=dev)
            held = []

            def reference():
                # Resets are deliberately in both measured graphs because output
                # aliases input; they prevent later replays from changing inputs.
                a0.copy_(seed); c0.copy_(initial)
                zskip.down(a0, sorted_ids, expert_ids, padded, packed, c0,
                           mode=zskip.SKIP, config=663, max_blocks=rows,
                           verify_rows=prefix)
                combine(inputs=c0.view(m, 6, 2688), topk_weights=weights,
                        topk_ids=ids, expert_map=None, outputs=o0)

            def candidate():
                a1.copy_(seed); c1.copy_(initial)
                h = fused.start(lib, a1, sorted_ids, expert_ids, padded, packed,
                                663, c1, weights, o1, prefix)
                if h is None:
                    raise RuntimeError("prototype timing guards declined case")
                held.append(h)
                fused.finish(h, o1)

            # Compile/warm all paths before capture on the official worker.
            reference(); candidate(); torch.cuda.synchronize(dev)
            held.clear()
            g0, g1 = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            with torch.cuda.graph(g0):
                reference()
            with torch.cuda.graph(g1):
                candidate()
            g0.replay(); g1.replay(); torch.cuda.synchronize(dev)
            if (not _bits_equal(o0, o1) or not _bits_equal(c0, c1) or
                int(held[-1][1][2].view(torch.int32).item()) != 0 or
                not graph_timing_admits(g0, g1, dev)):
                return False
    return True


def graph_timing_admits(reference_graph, candidate_graph, device):
    """Compare full captured chains; the caller must build both with identical inputs.

    This tests complete work including metadata, fallback launch and final copy.
    A wider end-to-end official score is still needed before adopting the bundle.
    """
    for _ in range(4):
        reference_graph.replay(); candidate_graph.replay()
    torch.cuda.synchronize(device)
    def measure(graph):
        times = []
        for _ in range(7):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(25):
                graph.replay()
            end.record(); end.synchronize()
            times.append(begin.elapsed_time(end))
        return statistics.median(times)
    old, new = measure(reference_graph), measure(candidate_graph)
    return old > 0 and new < old * 0.98

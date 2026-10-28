# SPDX-License-Identifier: Apache-2.0
"""Optional remote-only admission for the actual lm_head scale encoding.

The fixtures are private synthetic activations.  A pass is limited evidence for
these cases, not a claim of complete numerical equivalence or end-to-end gain.
"""
import statistics

import torch


def _same(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@torch.no_grad()
def qualify(layer, byte_su, codes, table):
    import marlinx as api

    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    if ((n, k) != (131072, 2688) or torch.cuda.is_current_stream_capturing()
            or getattr(layer, "_marlinx_zs", None) is not None
            or getattr(layer, "_marlinx_lean", False)):
        return False
    w, s, gs = layer.weight, layer.weight_scale, layer.weight_global_scale
    device = w.device
    old_pdl = api._pdl_active
    try:
        # These immutable owners are checked across the complete private gate.
        protected = (w, s, gs, byte_su, codes, table)
        copies = [x.clone() for x in protected]
        pressure = torch.empty(64 << 20, dtype=torch.uint8, device=device)
        gen = torch.Generator(device=device).manual_seed(784310)
        ratios, timings = [], []
        for m in range(1, 9):
            x = torch.randn((m, k), dtype=torch.bfloat16, device=device, generator=gen)
            owners = [torch.full((m * n + 256,), 3, dtype=torch.bfloat16, device=device)
                      for _ in range(2)]
            out = [v[128:-128].view(m, n) for v in owners]

            def run(which):
                return api.gemm(x, w, s, gs, n, k, su=codes if which else byte_su,
                                sut=table if which else None, su_fallback=byte_su,
                                _out=out[which])

            def compare(graphs=None):
                xcopy = x.clone()
                for v in owners:
                    v.fill_(3)
                for which in (0, 1):
                    graphs[which].replay() if graphs is not None else run(which)
                torch.cuda.synchronize(device)
                if (not _same(owners[0], owners[1]) or not _same(x, xcopy)
                    or not all(bool((v[:128] == 3).all()) and bool((v[-128:] == 3).all())
                               for v in owners) or api._pdl_active != old_pdl):
                    raise ValueError("SU6 output, owner, input or PDL mismatch")

            compare()
            # Exercise bit-pattern edge behavior separately from the finite
            # timing inputs; a mismatch simply retains the original SU path.
            saved_x = x.clone()
            x[0, :4] = torch.tensor([0.0, -0.0, float("inf"), float("nan")],
                                     dtype=x.dtype, device=device)
            compare()
            x.copy_(saved_x)
            if m in (1, 5, 6, 8):
                graphs = []
                for which in (0, 1):
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        run(which)
                    graphs.append(g)
                # Both captures keep fixed inputs and output owners; content
                # changes after capture so replay cannot rely on warmup values.
                for scale in (0.5, -2.0):
                    x.mul_(scale)
                    compare(graphs)
                for pressured in (False, True):
                    samples = [[], []]
                    for turn in range(5):
                        for which in ((0, 1) if turn % 2 == 0 else (1, 0)):
                            if pressured:
                                pressure.fill_(turn + 1)
                            else:
                                for _ in range(4):
                                    graphs[which].replay()
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            count = 1 if pressured else 8
                            start.record()
                            for _ in range(count):
                                graphs[which].replay()
                            end.record()
                            end.synchronize()
                            samples[which].append(start.elapsed_time(end) / count)
                    old, new = (statistics.median(values) for values in samples)
                    if not old > 0:
                        return False
                    ratios.append(new / old)
                    timings.append((old, new))
                    compare(graphs)
        if any(not _same(x, saved) for x, saved in zip(protected, copies)):
            return False
        return (bool(ratios) and statistics.median(ratios) < 0.995
                and sum(b for _, b in timings) / sum(a for a, _ in timings) < 0.995
                and max(ratios) < 1.01)
    except Exception:
        return False
    finally:
        api._pdl_active = old_pdl

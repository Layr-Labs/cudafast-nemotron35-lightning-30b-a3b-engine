#!/usr/bin/env python3
"""BB64 byte-slab repack: real-weight FP32-bit differential and head replay.

MTP_ORIGINAL_SOURCE and MTP_REPACK_SOURCE name explicit reference/candidate
sources. Retired repacks need not replace the current runtime to reproduce. The reference source is explicit. Every failed bit or graph check makes this
process exit nonzero; token agreement alone is not an exactness certificate.
Timings are diagnostic CUDA-graph microbenchmarks, not full-engine scores.
"""
import importlib.util
import json
import os
from pathlib import Path
import statistics

import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timed(fn, repeats=40):
    for _ in range(3):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(4):
            fn()
    begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    begin.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / (repeats * 4)


def main():
    torch.manual_seed(783617)
    torch.set_num_threads(8)
    source = Path(os.environ['MTP_REPACK_SOURCE'])
    ref = load('screen_repack_reference', os.environ['MTP_ORIGINAL_SOURCE'])
    new = load('screen_repack_candidate', source)
    for name in ['SCREEN_FRAC_GROUPS', 'SCREEN_CHUNK', 'SCREEN_PER',
                 'CASCADE_FRAC_GROUPS', 'CASCADE_PER']:
        assert getattr(ref, name) == getattr(new, name), name
    snapshot = os.environ['NEMO_SNAPSHOT']
    old = ref.DraftHeadScreen(snapshot, torch.device('cuda'))
    fast = new.DraftHeadScreen(snapshot, torch.device('cuda'))
    assert old.graph is not None and fast.graph is not None
    weights = [('coarse', old.w_coarse, old.s_coarse, fast.w_coarse, fast.s_coarse),
               ('screen', old.w_scr, old.s_scr, fast.w_scr, fast.s_scr),
               ('full', old.w_full, old.s_full, fast.w_full, fast.s_full)]
    for name, w, s, wp, sp in weights:
        # Inverse physical mapping must preserve every raw byte, including -0.
        if wp.shape[-1] == 72:  # interleaved raw weights + scales
            wi = wp[..., :64].permute(0, 2, 1, 3).contiguous().reshape(w.shape)
            si = sp[..., 64:72].contiguous().view(torch.uint8)
        else:  # separate weight and scale planes
            assert wp.shape[-1] == 64 and sp.shape[-1] == 8
            wi = wp.permute(0, 2, 1, 3).contiguous().reshape(w.shape)
            si = sp.contiguous().view(torch.uint8)
        si = si.permute(0, 2, 1, 3).contiguous().reshape(s.shape)
        assert torch.equal(wi, w), name
        assert torch.equal(si.view(torch.uint8), s.view(torch.uint8)), name
    cases = []
    modes = [('coarse', old.w_coarse, old.s_coarse, fast.w_coarse, fast.s_coarse,
              old.coarse_cols, False, old.n),
             ('refine', old.w_scr, old.s_scr, fast.w_scr, fast.s_scr,
              old.cols, True, old.refine_scores.numel()),
             ('full', old.w_full, old.s_full, fast.w_full, fast.s_full,
              None, True, old.cand_logits.numel())]
    for name, w, s, wp, sp, cols, gather, nout in modes:
        for trial in range(12):
            h = torch.randn(2688, device='cuda')
            if trial == 0:
                h.zero_()
            elif trial == 1:
                h = -torch.zeros_like(h)
            elif trial % 4 == 0:
                h = h.to(torch.bfloat16).float()
            elif trial % 4 == 1:
                h *= torch.pow(2.0, torch.randint(-14, 14, h.shape, device='cuda'))
            elif trial % 4 == 2:
                h = (h * 3).round()
            else:
                h *= 16
            if cols is not None:
                h = h.index_select(0, cols)
            ids = None
            if gather:
                if trial % 3 == 0:
                    ids = torch.arange(nout, device='cuda', dtype=torch.int64) % old.n
                elif trial % 3 == 1:
                    ids = torch.randint(old.n, (nout,), device='cuda')
                    ids[:4] = torch.tensor([0, 63, 64, old.n-1], device='cuda')
                else:
                    ids = torch.zeros(nout, device='cuda', dtype=torch.int64)
            expected = torch.empty(nout, dtype=torch.float32, device='cuda')
            actual = torch.empty_like(expected)
            def baseline():
                ref._rows_dot(h, w, s, ids, old.gscale, expected, gather)
            def candidate():
                new._rows_dot(h, wp, sp, ids, old.gscale, actual, gather)
            baseline()
            candidate()
            mismatch = int((actual.view(torch.int32) != expected.view(torch.int32)).sum().item())
            row = dict(mode=name, trial=trial, elements=nout, mismatches=mismatch)
            if trial == 3 and mismatch == 0:
                a, b, c, d = timed(baseline), timed(candidate), timed(candidate), timed(baseline)
                row.update(baseline_us=(a+d)/2, candidate_us=(b+c)/2, speedup=(a+d)/(b+c))
            cases.append(row)
        print('STAGE', name, json.dumps(cases[-12:]), flush=True)
    graph_checks = greedy_mismatches = 0
    hidden = [torch.zeros(1, 2688, device='cuda', dtype=torch.bfloat16)]
    hidden += [torch.randn(m, 2688, device='cuda', dtype=torch.bfloat16) * scale
               for m in [1, 2, 4] for scale in [0.1, 1, 16]]
    live_path = os.environ.get('MTP_HIDDEN_RESULT')
    if live_path:
        live = torch.load(live_path, map_location='cuda', weights_only=True)
        hidden += [row[None] for row in live]
    retained = []
    for h in hidden:
        expected = old.greedy(h)
        actual = fast.greedy(h)
        greedy_mismatches += int((actual != expected).sum().item())
        retained.append((actual, actual.clone()))
    for actual, saved in retained:
        assert torch.equal(actual, saved), 'returned output lifetime alias'
    # Both graph and eager paths read fresh inputs and preserve outputs.
    for m in [1, 2, 4]:
        h = torch.randn(m, 2688, device='cuda', dtype=torch.bfloat16)
        # greedy(B=1) replays an already captured graph and cannot itself be
        # nested in a new capture. Capture its arithmetic body directly here;
        # live-input owned-buffer replay was exercised above for every B=1 row.
        def body():
            return torch.stack([fast._greedy_row(h[i].float().contiguous())
                                for i in range(m)])
        body()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replay = body()
        for _ in range(4):
            h.copy_(torch.randn_like(h))
            expected = old.greedy(h)
            graph.replay()
            greedy_mismatches += int((replay != expected).sum().item())
            graph_checks += 1
    h = torch.randn(1, 2688, device='cuda', dtype=torch.bfloat16)
    def time_head(head):
        samples = []
        for _ in range(7):
            begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            begin.record()
            for _ in range(64):
                head.greedy(h)
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) * 1000 / 64)
        return samples
    a, b, c, d = time_head(old), time_head(fast), time_head(fast), time_head(old)
    base_us = statistics.median(a+d)
    candidate_us = statistics.median(b+c)
    result = dict(passed=not greedy_mismatches and all(not c['mismatches'] for c in cases),
                  byte_inverse_checks=6, bitwise_cases=len(cases),
                  elements=sum(c['elements'] for c in cases),
                  mismatches=sum(c['mismatches'] for c in cases),
                  hidden_rows=sum(h.shape[0] for h in hidden),
                  greedy_mismatches=greedy_mismatches, graph_replays=graph_checks,
                  head_baseline_us=base_us, head_candidate_us=candidate_us,
                  head_speedup=base_us/candidate_us, cases=cases)
    Path(os.environ['NEMO_RESULT']).write_text(json.dumps(result, indent=2)+'\n')
    print('REPACK_RESULT', json.dumps({k:v for k,v in result.items() if k != 'cases'}), flush=True)
    assert result['passed'], 'FP32 bits or greedy graph mismatch; repack is ineligible'


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Exact final-only top-k cascade and boot-graph differential on the promoted carrier."""
import importlib.util
import json
import os
from pathlib import Path
import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def timed(fn, repeats=100):
    for _ in range(3): fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(8): fn()
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    torch.cuda.synchronize()
    start.record()
    for _ in range(repeats): graph.replay()
    end.record(); end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats / 8


def main():
    torch.manual_seed(563251)
    torch.set_num_threads(8)
    src = Path(__file__).resolve().parents[2] / 'vllm/vllm/model_executor/models/nemotron_h_mtp_screen.py'
    ref = load('screen_sorted_ref', os.environ['MTP_ORIGINAL_SOURCE'])
    new = load('screen_unsorted_candidate', src)
    membership = []
    for width, count in [(2048, 256), (256, 32), (2048, 32)]:
        for trial in range(32):
            v = torch.randn(64, width, device='cuda')
            if trial % 8 == 0: v.zero_()
            if trial % 8 == 1: v = v.round()
            if trial % 8 == 2: v[:, ::2] = float('inf')
            if trial % 8 == 3: v.fill_(-float('inf'))
            if trial % 8 == 4: v[:, ::4] = float('nan')
            a = v.topk(count, dim=-1, sorted=True).indices.sort(dim=-1).values
            b = v.topk(count, dim=-1, sorted=False).indices.sort(dim=-1).values
            mismatches = int((a != b).sum().item())
            membership.append(dict(width=width, count=count, trial=trial, mismatches=mismatches))
    cascade = []
    for trial in range(48):
        coarse = torch.randn(64, 2048, device='cuda')
        refine = torch.randn_like(coarse)
        if trial % 4 == 0: refine.zero_()
        if trial % 4 == 1: refine = refine.round()
        if trial % 4 == 2: refine = (refine * 100).round() / 100
        picked = []
        for sorted_flag in [True, False]:
            ids = coarse.topk(256, dim=-1, sorted=True).indices
            val = refine.gather(1, ids)
            ix = val.topk(32, dim=-1, sorted=sorted_flag).indices
            picked.append(ids.gather(1, ix).sort(dim=-1).values)
        cascade.append(dict(trial=trial, mismatches=int((picked[0]!=picked[1]).sum().item())))
    print('CASCADE_BOUNDARY', json.dumps(dict(cases=len(cascade), mismatches=sum(r['mismatches'] for r in cascade))), flush=True)
    head = ref.DraftHeadScreen(os.environ['NEMO_SNAPSHOT'], torch.device('cuda'))
    fast = new.DraftHeadScreen(os.environ['NEMO_SNAPSHOT'], torch.device('cuda'))
    assert head.graph is not None and fast.graph is not None, 'boot capture failed'
    held = []
    graph_cases = 0
    # Use each constructor's OWN graph, not a copy of the reference graph.
    for trial in range(12):
        h = torch.randn(1, head.w_full.shape[1] * 2, device='cuda', dtype=torch.bfloat16)
        h *= [0.0, 0.1, 1.0, 10.0][trial % 4]
        a, b = head.greedy(h), fast.greedy(h)
        assert torch.equal(a, b), ('boot graph', trial, a, b)
        held.append((b, a.clone()))
        for previous, want in held:
            assert torch.equal(previous, want), 'aliased output across replays'
        graph_cases += 1
    # Eager row path records final IDs/logits before the BF16 selector.
    # Sorting by token ID compares the same work despite output permutations.
    captures = {ref: [], new: []}
    for mod in [ref, new]:
        original = mod._rows_dot
        def record(*args, _mod=mod, _orig=original):
            _orig(*args)
            h, w, s, ids, scale, out, gather = args
            if gather:
                order = ids.argsort()
                captures[_mod].append((ids[order].clone(), out[order].view(torch.int32).clone()))
        mod._rows_dot = record
    real = []
    for trial in range(16):
        h = torch.randn(1, head.w_full.shape[1] * 2, device='cuda', dtype=torch.bfloat16)
        h *= [0.0, 0.1, 1.0, 10.0][trial % 4]
        for v in captures.values(): v.clear()
        a = head._greedy_row(h[0].float())
        b = fast._greedy_row(h[0].float())
        assert len(captures[ref]) == len(captures[new]) == 2
        scores_equal = all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1])
                           for x, y in zip(captures[ref], captures[new]))
        real.append(dict(trial=trial, token_equal=torch.equal(a,b), per_id_bits_equal=scores_equal))
    # Stable coarse membership/order must be retained before the refinement.
    assert all(r['mismatches']==0 for r in cascade), 'final-unsorted cascade changed ties'
    # Remove diagnostic wrappers before timing/capture.
    ref = load('screen_sorted_timing', os.environ['MTP_ORIGINAL_SOURCE'])
    new = load('screen_unsorted_timing', src)
    head.__class__ = ref.DraftHeadScreen
    fast.__class__ = new.DraftHeadScreen
    # The final selector must round to BF16 before comparison and break ties by ID.
    selector_cases = 0
    big = torch.tensor(1 << 30, device='cuda', dtype=torch.int64)
    for n in [1, 3, 32, 2048, 2051]:
        ids = torch.randperm(131072, device='cuda')[:n].contiguous()
        for trial in range(48):
            v = torch.randn(n, device='cuda') * [1e-30, 1., 1000., 1e30][trial % 4]
            if trial % 12 == 0: v.fill_(0.)
            if trial % 12 == 1: v.fill_(-0.)
            if trial % 12 == 2: v.fill_(-float('inf'))
            if trial % 12 == 3: v[-1] = float('inf')
            if trial % 12 == 4: v[-1] = float('nan')
            if trial % 12 == 5: v[:] = 1.001; v[-1] = 1.002
            bf = v.to(torch.bfloat16)
            want = torch.where(bf == bf.max(), ids, big).min()
            out = torch.empty(1, device='cuda', dtype=torch.int64)
            new._select_token[(1,)](v, ids, out, N=n, BLOCK=new.triton.next_power_of_2(n))
            assert torch.equal(out[0], want), ('selector', n, trial)
            selector_cases += 1
    pairs = []
    for _ in range(3):
        a = timed(lambda: head._greedy_row(h[0].float()))
        b = timed(lambda: fast._greedy_row(h[0].float()))
        c = timed(lambda: fast._greedy_row(h[0].float()))
        d = timed(lambda: head._greedy_row(h[0].float()))
        pairs.append(dict(baseline_us=(a+d)/2, candidate_us=(b+c)/2, speedup=(a+d)/(b+c)))
    result = dict(graph_cases=graph_cases, selector_cases=selector_cases, cascade=cascade, passed=all(r["mismatches"]==0 for r in cascade) and all(r['mismatches']==0 for r in membership) and
                  all(r['token_equal'] and r['per_id_bits_equal'] for r in real),
                  membership=membership, real=real, pairs=pairs)
    Path(os.environ['NEMO_RESULT']).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'], graph_cases=graph_cases, selector_cases=selector_cases, membership_mismatches=sum(r['mismatches'] for r in membership),
                         real_mismatches=sum(not r['token_equal'] or not r['per_id_bits_equal'] for r in real), pairs=pairs)), flush=True)
    assert result['passed'], 'unsorted topk changed membership or per-id logits'


if __name__ == '__main__': main()

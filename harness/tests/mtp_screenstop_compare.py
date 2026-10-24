"""Parent-bound screen score bitwise and changed-input/gate graph diagnostic."""
import copy
import importlib.util
import json
import os
import statistics
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
MOD = 'vllm/vllm/model_executor/models/nemotron_h_mtp_screen.py'


def load(name, root):
    spec = importlib.util.spec_from_file_location('vllm.model_executor.models.' + name, root / MOD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def capture(fn, repeats=1):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(repeats):
            fn()
    return g


def elapsed(g, repeats):
    vals = []
    for _ in range(5):
        a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        a.record(); g.replay(); b.record(); b.synchronize()
        vals.append(a.elapsed_time(b) * 1000 / repeats)
    return statistics.median(vals)


def main():
    parent = load('parent_screenstop_probe', Path(os.environ['NEMO_CONTROL']))
    candidate = load('candidate_screenstop_probe', ROOT)
    torch.manual_seed(975169)
    screen = parent.DraftHeadScreen(os.environ['NEMO_SNAPSHOT'], torch.device('cuda'))
    gate = parent.depth_gate_init(torch.device('cuda'))
    result = {'cases': 0, 'elements': 0, 'graph_replays': 0, 'rows': [], 'all_passed': False}
    configs = [
        ('coarse', screen.w_coarse, screen.s_coarse, None, screen.n),
        ('refine', screen.w_scr, screen.s_scr,
         torch.randperm(screen.n, device='cuda')[:64 * parent.CASCADE_PER], 64 * parent.CASCADE_PER),
        ('full', screen.w_full, screen.s_full,
         torch.randperm(screen.n, device='cuda')[:64 * parent.SCREEN_PER], 64 * parent.SCREEN_PER),
    ]
    for name, w, s, ids, rows in configs:
        h = torch.randn(w.shape[1] * 2, device='cuda', dtype=torch.float32)
        ref = torch.empty(rows, device='cuda', dtype=torch.float32)
        out = torch.empty_like(ref)
        def run(mod, dest):
            mod._rows_dot(h, w, s, ids, screen.gscale, dest, ids is not None, True)
        gate.zero_()
        g = capture(lambda: run(candidate, out))
        for mode in ('finite', 'zero', 'nan', 'inf', 'finite'):
            h.normal_()
            if mode == 'zero':
                h.zero_(); h[::2] = -0.0
            if mode == 'nan':
                h[3] = float('nan')
            if mode == 'inf':
                h[5] = float('inf'); h[6] = -float('inf')
            for stop in (0, 1, 0, 1):
                gate[0] = stop
                run(parent, ref); g.replay(); torch.cuda.synchronize()
                # Canonicalize NaN payloads only; finite values include signed zero.
                mismatch = (ref.view(torch.int32) != out.view(torch.int32)) & ~(ref.isnan() & out.isnan())
                assert not bool(mismatch.any()), (name, mode, stop, int(mismatch.sum()))
                result['cases'] += 1
                result['elements'] += rows
                result['graph_replays'] += 1
        h.normal_()
        graphs = {label: capture(lambda mod=mod, dest=dest: run(mod, dest), 8)
                  for label, mod, dest in [('parent', parent, ref), ('candidate', candidate, out)]}
        for stop in (0, 1):
            gate[0] = stop
            ts = {label: elapsed(graph, 8) for label, graph in graphs.items()}
            row = {'stage': name, 'stop': stop, 'parent_us': ts['parent'], 'candidate_us': ts['candidate']}
            result['rows'].append(row)
            print(json.dumps(row), flush=True)
    # Full selector, including both torch.topk calls, tie IDs, and gate counters.
    other = copy.copy(screen)
    other.__class__ = candidate.DraftHeadScreen
    h = torch.randn(2688, device='cuda')
    token = torch.empty((), device='cuda', dtype=torch.int32)
    def select():
        token.copy_(other._greedy_row(h, gated=True))
    gate.zero_()
    cg = capture(select)
    for mode in ('finite', 'zero', 'finite'):
        h.normal_() if mode == 'finite' else h.zero_()
        for stop in (0, 1, 0, 1):
            gate.copy_(torch.tensor([stop, 2, 1], device='cuda', dtype=torch.int32))
            expected = screen._greedy_row(h, gated=True).clone()
            expected_gate = gate.clone()
            gate.copy_(torch.tensor([stop, 2, 1], device='cuda', dtype=torch.int32))
            cg.replay(); torch.cuda.synchronize()
            assert torch.equal(token, expected), (mode, stop, token, expected)
            assert torch.equal(gate, expected_gate), (mode, stop, gate, expected_gate)
            result['cases'] += 1
            result['graph_replays'] += 1
    result['all_passed'] = True
    Path(os.environ['NEMO_RESULT']).write_text(json.dumps(result, indent=2) + '\n')
    print('SCREENSTOP_BITWISE_AND_GRAPH_PASS', result['cases'], result['elements'], result['graph_replays'], flush=True)


if __name__ == '__main__':
    main()

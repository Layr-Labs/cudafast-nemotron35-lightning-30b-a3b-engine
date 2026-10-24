"""Fast live-checkpoint paired zskip library/config screen; no fixture changes."""
import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'harness/tests'))
import humming_moe_exact as hme
import zskip


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'harness/kernels/zskip/py/__init__.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.LIB_PATH = str(Path(path).resolve())
    assert mod.available(), mod.load_error()
    assert mod.init() is None or mod._ready
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--control', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    control = load('zcontrol', args.control)
    candidate = load('zcandidate', args.candidate)
    snap = hme.Snapshot(args.weights)
    emb = snap.load(['backbone.embeddings.weight'], 'cuda')['backbone.embeddings.weight']
    layers = []
    for li in [1, 27, 51]:
        raw = snap.raw_experts(li, 'cuda')
        ex = hme.build_variant(raw, f'backbone.layers.{li}.mixer.experts', True)
        packed = zskip.repack(raw['w2_weight'], raw['w2_weight_scale'], ex.layer.w2_global_scale.detach().float().reshape(-1), 'swap')
        layers.append((li, ex, packed, snap.router(li, 'cuda')))
        del raw
        torch.cuda.empty_cache()
    variants = [('control', control, 100), ('prmt', candidate, 100)]
    variants += [(f'config{c}', control, c) for c in [101, 102, 103, 104, 105, 106, 107]]
    result = {'cases': 0, 'elements': 0, 'replays': 0, 'rows': []}
    timing = {}
    for m in [1, 4]:
        for li, ex, packed, router in layers:
            for ki, kind in enumerate(['emb', 'rand', 'rand_uniform_route', 'edge']):
                x, ids, tw = hme.make_inputs(kind, m, li*104729 + m*37 + ki, emb, router, 'cuda')
                stages = hme.run_stages(ex, x, ids, tw)
                _, kw = ex.prepare_humming_moe_kwargs(ids, None, None)
                act = stages['act'].clone()
                original_act = stages['act']
                ref = stages['w2']
                for name, mod, config in variants:
                    act.copy_(original_act)
                    out = torch.empty_like(ref)
                    def run():
                        mod.down(act, kw['sorted_ids'], kw['expert_ids'], kw['num_tokens_padded'], packed, out, mode=mod.SKIP, config=config)
                    run()
                    torch.cuda.synchronize()
                    assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), (li,m,kind,name)
                    result['cases'] += 1
                    result['elements'] += out.numel()
                    if kind == 'emb':
                        timing.setdefault((m, name), []).append((mod, config, packed, act, kw, out))
                    if kind == 'edge':
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            run()
                        for rep in range(5):
                            act.copy_(original_act if rep % 2 else torch.zeros_like(act))
                            reference = torch.empty_like(ref)
                            control.down(act, kw['sorted_ids'], kw['expert_ids'], kw['num_tokens_padded'], packed, reference, mode=control.SKIP, config=100)
                            graph.replay()
                            torch.cuda.synchronize()
                            assert torch.equal(out.view(torch.int16), reference.view(torch.int16)), (li,m,kind,name,rep)
                            result['replays'] += 1
                print('EXACT',li,m,kind,flush=True)
        def make_run(calls):
            def run():
                for mod, cfg, packed, act, kw, out in calls:
                    mod.down(act, kw['sorted_ids'], kw['expert_ids'], kw['num_tokens_padded'], packed, out, mode=mod.SKIP, config=cfg)
            return run
        funcs = {name: make_run(timing[m, name]) for name, _, _ in variants}
        for name, _, _ in variants[1:]:
            pairs=[]
            for rep in range(3):
                order=['control',name] if rep%2==0 else [name,'control']
                times={key:statistics.median(hme.graph_time(funcs[key], 8))*1000/len(layers) for key in order}
                pairs.append({'control_us':times['control'],'candidate_us':times[name],'speedup':times['control']/times[name]})
            row={'m':m,'variant':name,'speedup':statistics.median(p['speedup'] for p in pairs),'pairs':pairs}
            result['rows'].append(row)
            Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(row),flush=True)
    print('ZSKIP_EXACT_PAIRED_PASS',result['cases'],result['elements'],result['replays'],flush=True)


if __name__ == '__main__':
    main()

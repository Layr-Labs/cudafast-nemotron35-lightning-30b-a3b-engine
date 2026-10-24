# SPDX-License-Identifier: Apache-2.0
"""Compare byte-identical sparse-down libraries on real weights and graph replay.

Run in the pinned GPU container after setup. Libraries share the public C ABI.
No official score is produced. Candidate arithmetic is checked independently
against direct Humming stages before paired rotating-layer timings.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import statistics

os.environ['VLLM_HUMMING_BATCH_INVARIANT'] = '1'
import torch
import humming_moe_exact as check
import routed_sparse


def load(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    lib.routed_sparse_init.argtypes = []
    lib.routed_sparse_init.restype = ctypes.c_int
    lib.routed_sparse_run.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int, ctypes.c_void_p]
    lib.routed_sparse_run.restype = ctypes.c_int
    assert lib.routed_sparse_init() == 0
    return lib


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', required=True)
    p.add_argument('--libraries', required=True, help='JSON object: variant -> library path')
    p.add_argument('--output', required=True)
    p.add_argument('--layers', default='1,13,27,40,51')
    p.add_argument('--ms', default='1,3,4,8')
    a = p.parse_args()
    torch.set_num_threads(8)
    paths = json.loads(a.libraries)
    libs = {k: load(v) for k, v in paths.items()}
    assert 'control' in libs
    snap = check.Snapshot(a.weights)
    emb = snap.load(['backbone.embeddings.weight'], 'cuda')['backbone.embeddings.weight']
    layers = check.MOE_LAYERS if a.layers == 'all' else list(map(int, a.layers.split(',')))
    ctx = {'pairs': [], 'routers': [], 'emb': emb}
    cases = elements = replays = 0
    for li in layers:
        routed_sparse._lib = libs['control']
        ex = check.build_variant(snap.raw_experts(li, 'cuda'), f'backbone.layers.{li}.mixer.experts', True)
        assert ex.layer.routed_sparse_down is not None
        ctx['pairs'].append((ex, ex))
        router = snap.router(li, 'cuda')
        ctx['routers'].append(router)
        for m in map(int, a.ms.split(',')):
            for kind in ['emb', 'rand', 'rand_uniform_route', 'edge']:
                x, ids, weights = check.make_inputs(kind, m, 1000 + li + m, emb, router, 'cuda')
                expected = check.run_stages(ex, x, ids, weights)
                for name, lib in libs.items():
                    routed_sparse._lib = lib
                    actual = check.run_apply(ex, x, ids, weights)
                    diff = check.diff_count(expected['out'], actual)
                    assert diff == 0, (name, li, m, kind, diff)
                    cases += 1
                    elements += actual.numel()
        # Capture each library separately, mutate data/routing between replays.
        m = 4
        x = torch.zeros((m * 6, 1856), dtype=torch.bfloat16, device='cuda')
        ids = torch.arange(6, device='cuda', dtype=torch.int32).repeat(m, 1)
        y = torch.empty((m * 6, 2688), dtype=torch.bfloat16, device='cuda')
        for name, lib in libs.items():
            routed_sparse._lib = lib
            for _ in range(3):
                assert routed_sparse.forward(x, ids, y, ex.layer.routed_sparse_down)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                assert routed_sparse.forward(x, ids, y, ex.layer.routed_sparse_down)
            from vllm.utils.humming import HummingMethod
            for i, kind in enumerate(['dense', 'sparse', 'tiny', 'zero', 'dense']):
                torch.manual_seed(1030 + li + i)
                x.copy_(torch.randn_like(x))
                if kind == 'sparse':
                    x.mul_((torch.rand_like(x) > .9).to(x.dtype))
                elif kind == 'tiny':
                    x.mul_(1e-38)
                elif kind == 'zero':
                    x.zero_()
                ids.copy_(torch.stack([torch.randperm(128, device='cuda')[:6] for _ in range(m)]).to(torch.int32))
                _, kw = ex.prepare_humming_moe_kwargs(ids, None, None)
                ref = torch.empty_like(y)
                HummingMethod.forward_layer(layer=ex.layer, inputs=x, input_scale=None,
                    outputs=ref, sublayer_name='w2', **kw)
                g.replay()
                diff = check.diff_count(ref, y)
                assert diff == 0, (name, li, kind, diff)
                replays += 1
            del g
        print(json.dumps(dict(layer=li, exact_cases=cases, elements=elements, graph_replays=replays)), flush=True)
    rows = []
    result = dict(reported=True, libraries=paths, cases=cases, elements=elements,
                  graph_replays=replays, layers=layers, rows=rows)
    for m in map(int, a.ms.split(',')):
        bench = check.OpBench(ctx, m)
        fn = bench.full_op(0)
        for name, lib in libs.items():
            if name == 'control':
                continue
            pairs = []
            for rep in range(6):
                values = {}
                order = ['control', name] if rep % 2 == 0 else [name, 'control']
                for variant in order:
                    routed_sparse._lib = libs[variant]
                    values[variant] = statistics.median(check.graph_time(fn, 17)) * 1000 / len(layers)
                pairs.append(dict(rep=rep, control_us=values['control'], candidate_us=values[name],
                                  speedup=values['control'] / values[name]))
            row = dict(m=m, variant=name, speedup=statistics.median(x['speedup'] for x in pairs), pairs=pairs)
            rows.append(row)
            Path(a.output).write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(row), flush=True)
    print('EXACT_AND_PAIRED_COMPLETE', cases, elements, replays, flush=True)


if __name__ == '__main__':
    main()

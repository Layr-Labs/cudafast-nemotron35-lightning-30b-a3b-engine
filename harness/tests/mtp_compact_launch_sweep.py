"""Live native BF16 MTP compact-down launch sweep, direct paired graph timing."""
import importlib.util
import json
import os
import statistics
from pathlib import Path

import torch
from safetensors import safe_open

root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('compact_probe', root/'vllm/vllm/model_executor/models/nemotron_h_mtp_compact.py')
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)


def graph_time(fn):
    fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(12):
            fn()
    samples = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b)*1000/12)
    return statistics.median(samples)


def main():
    snap = Path(os.environ['NEMO_SNAPSHOT'])
    index = json.loads((snap/'model.safetensors.index.json').read_text())['weight_map']
    def load(key):
        with safe_open(str(snap/index[key]), framework='pt', device='cuda') as f:
            return f.get_tensor(key)
    prefix = 'mtp.layers.1.mixer.'
    weights = torch.stack([load(f'{prefix}experts.{e}.down_proj.weight') for e in range(128)])
    shared = load(prefix+'shared_experts.down_proj.weight')
    h, inter = weights.shape[1:]
    wp = weights.transpose(1, 2).contiguous()
    sp = shared.reshape(h, 2, inter).permute(1, 2, 0).contiguous()
    del weights, shared
    torch.manual_seed(147033)
    variants = [(32, 4), (16, 2), (16, 4), (32, 2), (32, 8), (64, 4), (64, 8), (128, 4)]
    result = {'cases': 0, 'elements': 0, 'replays': 0, 'rows': []}
    for density in [.03, .10, .25, .5, 1.0]:
        original = torch.randn(8, inter, device='cuda', dtype=torch.bfloat16).square()
        original.masked_fill_(torch.rand_like(original.float()) > density, 0.)
        ids = torch.randperm(128, device='cuda')[:6].to(torch.int32)
        tw = torch.rand(6, device='cuda', dtype=torch.float32)
        tw /= tw.sum()
        indices = torch.empty((8, inter), device='cuda', dtype=torch.int32)
        values = torch.empty_like(original)
        counts = torch.empty(8, device='cuda', dtype=torch.int32)
        act = original.clone()
        def comp():
            k._compact[(8,)](act, indices, values, counts, I=inter, BLOCK=2048, num_warps=4)
        def run(out, bn, warps):
            comp()
            k._down[(k.triton.cdiv(h, bn),)](indices, values, counts, wp, sp, ids, tw, out,
                M=1, H=h, I=inter, ROUTED_SCALE=1., TOPK=6, S=2, BN=bn, BK=64,
                num_warps=warps, num_stages=2)
        ref = torch.empty((1, h), device='cuda', dtype=torch.bfloat16)
        outputs = {v: torch.empty_like(ref) for v in variants}
        funcs = {v: (lambda v=v: run(outputs[v], *v)) for v in variants}
        run(ref, 32, 4)
        for v in variants:
            funcs[v]()
            torch.cuda.synchronize()
            equal = torch.equal(outputs[v].view(torch.int16), ref.view(torch.int16))
            if not equal:
                print('MISMATCH',density,v,int((outputs[v].view(torch.int16)!=ref.view(torch.int16)).sum()),flush=True)
                continue
            result['cases'] += 1
            result['elements'] += h
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                funcs[v]()
            for rep in range(4):
                act.copy_(original if rep%2 else torch.zeros_like(original))
                run(ref, 32, 4)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(outputs[v].view(torch.int16), ref.view(torch.int16)), (density,v,rep)
                result['replays'] += 1
            act.copy_(original)
            pairs=[]
            for rep in range(3):
                order=[(32,4),v] if rep%2==0 else [v,(32,4)]
                times={z:graph_time(funcs[z]) for z in order}
                pairs.append({'control_us':times[(32,4)],'candidate_us':times[v], 'speedup':times[(32,4)]/times[v]})
            row={'density':density,'bn':v[0],'warps':v[1], 'speedup':statistics.median(p['speedup'] for p in pairs),'pairs':pairs}
            result['rows'].append(row)
            Path(os.environ['NEMO_RESULT']).write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(row),flush=True)
    print('COMPACT_LAUNCH_SWEEP_PASS',result['cases'],result['elements'],result['replays'],flush=True)


if __name__ == '__main__':
    main()

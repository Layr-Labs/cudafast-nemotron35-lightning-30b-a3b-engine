"""Diagnostic native pinned-BF16 MTP up/ReLU2 launch selection (no target edits)."""
import importlib.util
import json
import os
import statistics
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
with patch('vllm.utils.torch_utils.direct_register_custom_op'):
    spec = importlib.util.spec_from_file_location('mtp_up_probe', ROOT/'vllm/vllm/model_executor/models/nemotron_h_mtp_kernels.py')
    k = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(k)


def graph(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(12):
            fn()
    return g


def timing(g):
    samples=[]
    for _ in range(5):
        a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
        a.record(); g.replay(); b.record(); b.synchronize()
        samples.append(a.elapsed_time(b)*1000/12)
    return statistics.median(samples)


def main():
    torch.manual_seed(791211)
    snap=Path(os.environ['NEMO_SNAPSHOT'])
    index=json.loads((snap/'model.safetensors.index.json').read_text())['weight_map']
    def load(key):
        with safe_open(str(snap/index[key]),framework='pt',device='cuda') as f:
            return f.get_tensor(key)
    prefix='mtp.layers.1.mixer.'
    w=torch.stack([load(f'{prefix}experts.{e}.up_proj.weight') for e in range(128)])
    shared=load(prefix+'shared_experts.up_proj.weight')
    E,I,H=w.shape
    ids=torch.randperm(E,device='cuda')[:6].to(torch.int32)
    x=torch.randn((1,H),device='cuda',dtype=torch.bfloat16)
    outputs={}; variants=[(8,1024,8,3),(8,1024,4,3),(8,1024,4,2),(8,1024,4,1),(8,1024,8,1),(4,1024,4,2),(16,1024,8,2),(16,1024,4,2),(32,1024,8,2),(8,512,4,2),(16,512,4,2),(16,512,8,2),(32,512,8,2),(8,2048,8,2),(16,2048,8,2)]
    funcs={}
    for v in variants:
        out=torch.empty((8,I),device='cuda',dtype=torch.bfloat16)
        outputs[v]=out
        def run(v=v,out=out):
            bn,bk,warps,stages=v
            k._mtp_moe_up_kernel[(k.triton.cdiv(I,bn),8)](x,w,shared,ids,out,I,H,w.stride(0),w.stride(1),shared.stride(0),TOPK=6,BN=bn,BK=bk,EVICT=True,num_warps=warps,num_stages=stages)
        funcs[v]=run
    graphs={v:graph(fn) for v,fn in funcs.items()}
    ref=outputs[variants[0]]
    result={'rows':[]}
    for v in variants:
        funcs[v](); funcs[variants[0]](); torch.cuda.synchronize()
        equal=torch.equal(outputs[v].view(torch.int16),ref.view(torch.int16))
        diff=int((outputs[v].view(torch.int16)!=ref.view(torch.int16)).sum())
        pairs=[]
        for r in range(3):
            order=[variants[0],v] if r%2==0 else [v,variants[0]]
            ts={q:timing(graphs[q]) for q in order}
            pairs.append({'control_us':ts[variants[0]],'candidate_us':ts[v],'speedup':ts[variants[0]]/ts[v]})
        row={'bn':v[0],'bk':v[1],'warps':v[2],'stages':v[3],'exact':equal,'differences':diff,'speedup':statistics.median(p['speedup'] for p in pairs),'pairs':pairs}
        result['rows'].append(row)
        Path(os.environ['NEMO_RESULT']).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    print('MTP_UP_SWEEP_DONE',len(result['rows']),flush=True)


if __name__=='__main__':
    main()

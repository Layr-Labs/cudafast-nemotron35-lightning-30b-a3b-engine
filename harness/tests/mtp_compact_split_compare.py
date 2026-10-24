"""Native BF16 MTP down expert-parallel experiment vs live serial carrier."""
import importlib.util
import json
import os
from pathlib import Path
import statistics
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
def source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
control = source('compact_control', ROOT/'.local-run/mtp-compact-control.py')
candidate = source('compact_split', ROOT/'vllm/vllm/model_executor/models/nemotron_h_mtp_compact.py')

def elapsed(fn):
    fn()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(16):
            fn()
    vals=[]
    for _ in range(5):
        a,b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record();g.replay();b.record();b.synchronize()
        vals.append(a.elapsed_time(b)*1000/16)
    return statistics.median(vals)

def main():
    torch.manual_seed(714209)
    snap=Path(os.environ['NEMO_SNAPSHOT'])
    index=json.loads((snap/'model.safetensors.index.json').read_text())['weight_map']
    def load(key):
        with safe_open(str(snap/index[key]),framework='pt',device='cuda') as f:return f.get_tensor(key)
    prefix='mtp.layers.1.mixer.'
    w=torch.stack([load(f'{prefix}experts.{e}.down_proj.weight') for e in range(128)])
    sd=load(prefix+'shared_experts.down_proj.weight')
    H,I=w.shape[1:]
    packed=w.transpose(1,2).contiguous(),sd.reshape(H,2,I).permute(1,2,0).contiguous()
    del w,sd
    rows=[];elements=cases=replays=0
    for density in [0.,.01,.03,.10,.25,.50,1.]:
        original=torch.randn((8,I),device='cuda',dtype=torch.bfloat16).square()
        original.masked_fill_(torch.rand((8,I),device='cuda')>density,0.)
        ids=torch.randperm(128,device='cuda')[:6].to(torch.int32)
        tw=torch.rand(6,device='cuda',dtype=torch.float32);tw/=tw.sum()
        for M in [1,4]:
            h=original.clone()
            out={name:torch.empty((M,H),device='cuda',dtype=torch.bfloat16) for name in ['control','split']}
            def run(mod,name):
                assert mod.maybe_down(h,packed,ids,tw,out[name],1.)
            funcs={'control':lambda:run(control,'control'),'split':lambda:run(candidate,'split')}
            funcs['control']();funcs['split']()
            diff=int((out['control'].view(torch.int16)!=out['split'].view(torch.int16)).sum())
            max_abs=float((out['control'].float()-out['split'].float()).abs().max())
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):funcs['split']()
            for rep in range(4):
                h.copy_(original if rep%2 else torch.zeros_like(h))
                funcs['control']();graph.replay();torch.cuda.synchronize()
                assert torch.isfinite(out['split']).all(),(density,M,rep)
                if density==0 or rep%2==0:
                    assert torch.equal(out['control'].view(torch.int16),out['split'].view(torch.int16)),(density,M,rep)
                replays+=1
            h.copy_(original)
            pairs=[]
            for rep in range(3):
                order=['control','split'] if rep%2==0 else ['split','control']
                times={name:elapsed(funcs[name]) for name in order}
                pairs.append({'control_us':times['control'],'candidate_us':times['split'],'speedup':times['control']/times['split']})
            cases+=1;elements+=M*H
            row={'density':density,'M':M,'differences':diff,'max_abs':max_abs,'speedup':statistics.median(x['speedup'] for x in pairs),'pairs':pairs}
            rows.append(row);print(json.dumps(row),flush=True)
            Path(os.environ['NEMO_RESULT']).write_text(json.dumps({'cases':cases,'elements':elements,'replays':replays,'rows':rows},indent=2)+'\n')
    print('MTP_COMPACT_SPLIT_SCREEN_DONE',cases,elements,replays,flush=True)
if __name__=='__main__':main()

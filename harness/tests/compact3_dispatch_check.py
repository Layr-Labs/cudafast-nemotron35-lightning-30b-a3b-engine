"""Generalized wrapper vs unchanged direct compact kernels; changing CUDA graphs."""
import json,os
from pathlib import Path
import torch
from safetensors import safe_open
from vllm.model_executor.models import nemotron_h_mtp_compact as k

def main():
 torch.manual_seed(991753)
 snap=Path(os.environ['NEMO_SNAPSHOT']);index=json.loads((snap/'model.safetensors.index.json').read_text())['weight_map']
 def load(key):
  with safe_open(str(snap/index[key]),framework='pt',device='cuda') as f:return f.get_tensor(key)
 prefix='mtp.layers.1.mixer.'
 w=torch.stack([load(f'{prefix}experts.{e}.down_proj.weight') for e in range(128)])
 sd=load(prefix+'shared_experts.down_proj.weight');H,I=w.shape[1:]
 packed=w.transpose(1,2).contiguous(),sd.reshape(H,2,I).permute(1,2,0).contiguous();del w,sd
 assert all(torch.isfinite(w).all() for w in packed)
 rows=[];elems=replays=0
 for topk in [1,2,3,4,5,6]:
  slots=topk+2
  ids=torch.randperm(128,device='cuda')[:topk].to(torch.int32)
  weights=torch.rand(topk,device='cuda');weights/=weights.sum()
  for density in [0.,.03,.10,.50,1.]:
   h=torch.randn((slots,I),device='cuda',dtype=torch.bfloat16).square()
   h.masked_fill_(torch.rand_like(h.float())>density,0.)
   for M in [1,4,6]:
    y=torch.empty((M,H),device='cuda',dtype=torch.bfloat16);ref=torch.empty_like(y)
    idx=torch.empty_like(h,dtype=torch.int32);val=torch.empty_like(h);cnt=torch.empty(slots,device='cuda',dtype=torch.int32)
    def reference():
     k._compact[(slots,)](h,idx,val,cnt,I=I,BLOCK=2048,num_warps=4,launch_pdl=k.mtp_pdl())
     k._down[(k.triton.cdiv(H,32),)](idx,val,cnt,*packed,ids,weights,ref,M=M,H=H,I=I,ROUTED_SCALE=1.,TOPK=topk,S=2,BN=32,BK=64,num_warps=4,num_stages=2,launch_pdl=k.mtp_pdl())
    reference();assert k.maybe_down(h,packed,ids,weights,y,1.)
    assert torch.equal(y.view(torch.int16),ref.view(torch.int16)),(topk,density,M)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): assert k.maybe_down(h,packed,ids,weights,y,1.)
    original=h.clone()
    for rep in range(3):
     h.copy_(original if rep%2 else torch.zeros_like(h));ids.copy_(torch.randperm(128,device='cuda')[:topk]);weights.uniform_();weights/=weights.sum()
     reference();g.replay();torch.cuda.synchronize()
     assert torch.equal(y.view(torch.int16),ref.view(torch.int16)),(topk,density,M,rep)
     replays+=1;elems+=M*H
    rows.append({'topk':topk,'density':density,'M':M,'dispatch_active':True})
 result={'cases':len(rows),'bitwise_elements':elems,'changing_graph_replays':replays,'all_passed':True,'arithmetic_reference':'unchanged compact kernels, direct TOPK=1..6; wrapper selects existing path','note':'Not bitwise equivalence to zero-load fallback; draft proposal can differ; target verifier required'}
 Path('.local-run/compact3-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
 print('COMPACT3_GENERALIZED_DISPATCH_EXACT_PASS',flush=True)
if __name__=='__main__':main()

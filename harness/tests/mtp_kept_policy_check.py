"""Focused correctness test for stop-trigger exclusion or a two-attempt floor."""
import json
import os
import torch
from vllm.model_executor.models.nemotron_h_mtp_screen import _select_token

policy = os.environ['KEPT_POLICY']
assert policy in ('dropfail','min2')
torch.manual_seed(62417)
cases=0
for n in (1,2,7,32,128,768,1024):
 for scale in (0.1,1.,5.):
  logits=(torch.randn(n,device='cuda')*scale).to(torch.bfloat16).float()
  ids=torch.randperm(n,device='cuda').int()
  gate=torch.zeros(3,device='cuda',dtype=torch.int32)
  out=torch.empty(1,device='cuda',dtype=torch.int64)
  def call():
   _select_token[(1,)](logits,ids,out,n,1<<(n-1).bit_length(),gate_ptr=gate,GATED=True,THETA=.4,launch_pdl=True)
  call()
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph): call()
  for stop,kept,active in ((0,0,1),(0,1,1),(0,3,1),(1,2,1),(0,0,0),(1,4,0)):
   gate.copy_(torch.tensor([stop,kept,active],device='cuda',dtype=torch.int32));graph.replay()
   best=logits.max();chosen=ids[logits==best].min().item();prob=1/torch.exp(logits-best).sum().item();want=[stop,kept,active]
   if not stop and active:
    want[1]=kept+1
    if prob<.4 and (policy=='dropfail' or kept>=1):
     want[0]=1
     if policy=='dropfail':want[1]=max(kept,1)
   assert out.item()==chosen
   assert gate.tolist()==want,(policy,n,prob,[stop,kept,active],gate.tolist(),want)
   cases+=1
print(json.dumps({'passed':True,'policy':policy,'cases':cases,'PDL':True,'graph_replay':True}))

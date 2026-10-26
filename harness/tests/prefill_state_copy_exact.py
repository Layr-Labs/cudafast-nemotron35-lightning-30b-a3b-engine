#!/usr/bin/env python3
"""GPU bit-identity check for the imported prefill final-state copy."""
import argparse,json,pathlib,torch
from vllm.model_executor.layers.mamba.ops.prefill_state_copy import final_state_copy
parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args()
torch.manual_seed(519);cases=0
for dtype,view in [(torch.float32,torch.int32),(torch.bfloat16,torch.int16)]:
 for shape in [(64,64,128),(3,7,19),(1,1,1)]:
  for batch in [0,1,4]:
   for padded in [False,True]:
    src=torch.empty((9,*shape),device='cuda',dtype=dtype);src.view(view).random_(-32000,32000)
    cache=torch.empty((12*(2 if padded else 1),*shape),device='cuda',dtype=dtype);cache.view(view).random_(-32000,32000)
    if padded:cache=cache[::2]
    ref=cache.clone();si=torch.tensor([0,-1,3,7][:batch],device='cuda',dtype=torch.int64);di=torch.tensor([1,8,4,11][:batch],device='cuda',dtype=torch.int64);ref[di]=src[si]
    assert final_state_copy(cache,di,src,si)
    assert torch.equal(cache.view(view),ref.view(view)),(dtype,shape,batch,padded)
    cases+=1
print('state copy exact cases',cases,flush=True)
if args.output:pathlib.Path(args.output).write_text(json.dumps({'cases':cases,'bit_exact':True})+'\n')

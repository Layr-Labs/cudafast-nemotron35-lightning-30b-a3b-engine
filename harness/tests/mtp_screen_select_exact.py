#!/usr/bin/env python3
"""Exact final-token selection tests, real-weight head and paired CUDA timings."""
import importlib.util
import json
import os
from pathlib import Path
import torch


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def elapsed(fn, repeats=200):
    for _ in range(3): fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(8): fn()
    start,end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    torch.cuda.synchronize()
    start.record()
    for _ in range(repeats): graph.replay()
    end.record(); end.synchronize()
    return start.elapsed_time(end)*1000/repeats/8


def main():
    torch.manual_seed(11323)
    torch.set_num_threads(8)
    source=Path(__file__).resolve().parents[2]/'vllm/vllm/model_executor/models/nemotron_h_mtp_screen.py'
    base=module('screen_select_base',os.environ['MTP_ORIGINAL_SOURCE'])
    new=module('screen_select_candidate',source)
    cases=0
    big=torch.tensor(1<<30,device='cuda',dtype=torch.int64)
    for n in [1,3,32,2048,2051]:
        ids=torch.randperm(131072,device='cuda')[:n].contiguous()
        for trial in range(48):
            v=torch.randn(n,device='cuda',dtype=torch.float32)*[1e-30,1.,1000.,1e30][trial%4]
            if trial%12==0: v.fill_(0.)
            if trial%12==1: v.fill_(-0.)
            if trial%12==2: v.fill_(-float('inf'))
            if trial%12==3: v[-1]=float('inf')
            if trial%12==4: v[-1]=float('nan')
            if trial%12==5: v[:]=1.001;v[-1]=1.002
            bf=v.to(torch.bfloat16)
            expected=torch.where(bf==bf.max(),ids,big).min()
            out=torch.empty(1,device='cuda',dtype=torch.int64)
            new._select_token[(1,)](v,ids,out,N=n,BLOCK=new.triton.next_power_of_2(n))
            assert torch.equal(out[0],expected),(n,trial,out,expected)
            cases+=1
    n=2048
    v=torch.randn(n,device='cuda');ids=torch.randperm(131072,device='cuda')[:n].contiguous();out=torch.empty(1,device='cuda',dtype=torch.int64)
    def ref():
        bf=v.to(torch.bfloat16)
        return torch.where(bf==bf.max(),ids,big).min()
    def cand():new._select_token[(1,)](v,ids,out,N=n,BLOCK=2048)
    a,b,c,d=elapsed(ref),elapsed(cand),elapsed(cand),elapsed(ref)
    isolated=dict(baseline_us=(a+d)/2,candidate_us=(b+c)/2,speedup=(a+d)/(b+c))
    head=base.DraftHeadScreen(os.environ['NEMO_SNAPSHOT'],torch.device('cuda'))
    fast=new.DraftHeadScreen.__new__(new.DraftHeadScreen)
    fast.__dict__.update(head.__dict__)
    fast.scores=torch.empty_like(head.scores);fast.cand_logits=torch.empty_like(head.cand_logits)
    if getattr(head, "cascade", False): fast.refine_scores=torch.empty_like(head.refine_scores)
    fast._greedy_graph = None
    fast._greedy_input = None
    fast._greedy_output = None
    greedy_cases=replays=0
    held=[]
    for m in [1,2,4]:
        for trial in range(4):
            h=torch.randn(m,2688,device='cuda',dtype=torch.bfloat16)
            expected=head.greedy(h);actual=fast.greedy(h)
            assert torch.equal(expected,actual),(m,trial,expected,actual)
            held.append((actual,expected.clone()))
            for old,want in held: assert torch.equal(old,want), "aliased output"
            assert torch.equal(head.scores.view(torch.int32),fast.scores.view(torch.int32))
            assert torch.equal(head.cand_logits.view(torch.int32),fast.cand_logits.view(torch.int32))
            if getattr(head, "cascade", False):
                assert torch.equal(head.refine_scores.view(torch.int32),fast.refine_scores.view(torch.int32))
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): replay=fast.greedy(h)
            for _ in range(3):
                g.replay();assert torch.equal(replay,expected)
                replays+=1
            greedy_cases+=1
    h=torch.randn(1,2688,device='cuda',dtype=torch.bfloat16)
    a,b,c,d=elapsed(lambda:head.greedy(h),40),elapsed(lambda:fast.greedy(h),40),elapsed(lambda:fast.greedy(h),40),elapsed(lambda:head.greedy(h),40)
    whole=dict(baseline_us=(a+d)/2,candidate_us=(b+c)/2,speedup=(a+d)/(b+c))
    import time
    wall=[]
    for _ in range(3):
        pair=[]
        for obj in [head, fast, fast, head]:
            torch.cuda.synchronize(); start=time.perf_counter()
            for _ in range(100): obj.greedy(h)
            torch.cuda.synchronize(); pair.append((time.perf_counter()-start)*1e4)
        wall.append(dict(baseline_us=(pair[0]+pair[3])/2,candidate_us=(pair[1]+pair[2])/2,speedup=(pair[0]+pair[3])/(pair[1]+pair[2])))
    assert fast._greedy_graph is not None
    result=dict(passed=True,selection_cases=cases,real_head_cases=greedy_cases,graph_replays=replays,isolated=isolated,whole=whole,wall=wall)
    Path(os.environ['NEMO_RESULT']).write_text(json.dumps(result,indent=2)+'\n')
    print('VERIFIER PASS: '+json.dumps(result),flush=True)


if __name__=='__main__':main()

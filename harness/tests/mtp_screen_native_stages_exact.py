#!/usr/bin/env python3
"""Real-weight bitwise differential and paired timing for native E2M1 decode.

The fixed 64-byte K chunk, group arithmetic and survivor count are unchanged.
Every comparison uses FP32 bit patterns (not approximate/allclose).
"""
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


def timed(fn, repeats=80):
    for _ in range(3):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(8):
            fn()
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    torch.cuda.synchronize()
    begin.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / repeats / 8


def main():
    torch.manual_seed(68311)
    torch.set_num_threads(8)
    source = Path(__file__).resolve().parents[2] / 'vllm/vllm/model_executor/models/nemotron_h_mtp_screen.py'
    ref = module('screen_layout_reference', os.environ['MTP_ORIGINAL_SOURCE'])
    new = module('screen_layout_candidate', source)
    head = ref.DraftHeadScreen(os.environ['NEMO_SNAPSHOT'], torch.device('cuda'))
    from vllm.triton_utils import triton, tl
    @triton.jit
    def decode_bytes(src, a, b, c, d):
        i=tl.arange(0,256)
        v=tl.load(src+i)
        x,y=new._e2m1_pair(v)
        tl.store(a+i,x);tl.store(b+i,y)
        tl.store(c+i,ref._e2m1(v&15));tl.store(d+i,ref._e2m1(v>>4))
    packed=torch.arange(256,device="cuda",dtype=torch.int32)
    outputs=[torch.empty(256,device="cuda",dtype=torch.float32) for _ in range(4)]
    decode_bytes[(1,)](packed,*outputs)
    assert torch.equal(outputs[0].view(torch.int32),outputs[2].view(torch.int32))
    assert torch.equal(outputs[1].view(torch.int32),outputs[3].view(torch.int32))
    rows = []
    bitwise_cases = elements = 0
    modes = [("screen", head.w_scr, head.s_scr, head.cols, False, head.n),
             ("full", head.w_full, head.s_full, None, True, head.cand_logits.numel())]
    if getattr(head, "cascade", False):
        modes += [("coarse", head.w_coarse, head.s_coarse, head.coarse_cols, False, head.n),
                  ("refine", head.w_scr, head.s_scr, head.cols, True, head.refine_scores.numel())]
    for mode, w, s, cols, gather, nout in modes:
        ids = torch.randperm(head.n, device='cuda')[:nout].contiguous() if gather else None
        out = torch.empty(nout, device='cuda', dtype=torch.float32)
        expected = torch.empty_like(out)
        hs = []
        for i in range(8):
            h = torch.randn(2688, device='cuda', dtype=torch.float32) * [0.1, 1., 4., 16.][i % 4]
            hs.append(h if cols is None else h.index_select(0, cols))
        for stages in [int(v) for v in os.environ.get("MTP_SCREEN_TEST_STAGES", "2").split(",")]:
            br, warps = (16 if gather else 64),4
            def baseline(h=hs[0]):
                ref._rows_dot(h, w, s, ids, head.gscale, expected, gather)
            def candidate(h=hs[0]):
                new._nvfp4_rows_dot[((nout+br-1)//br,)](
                    h,w,s,ids if gather else out,out,nout,head.gscale,
                    NBYTES=w.shape[1],GATHER=gather,BR=br,BB=64,num_warps=warps,num_stages=stages)
            mismatches = cases = 0
            for h in hs:
                baseline(h)
                candidate(h)
                diff = int((out.view(torch.int32) != expected.view(torch.int32)).sum().item())
                mismatches += diff
                cases += 1
            row = dict(mode=mode,stages=stages,gather=gather,br=br,warps=warps,bitwise_cases=cases,elements=cases*nout,mismatches=mismatches,passed=mismatches==0)
            if not mismatches:
                a=timed(baseline); b=timed(candidate); c=timed(candidate); d=timed(baseline)
                row.update(baseline_us=(a+d)/2,candidate_us=(b+c)/2,speedup=(a+d)/(b+c))
            bitwise_cases += cases
            elements += cases*nout
            rows.append(row)
            print(json.dumps(row),flush=True)
    # Exercise actual source dispatch and CUDA-graph replay, including topk/ties.
    candidate_head = new.DraftHeadScreen.__new__(new.DraftHeadScreen)
    candidate_head.__dict__.update(head.__dict__)
    candidate_head._greedy_graph = None
    candidate_head._greedy_input = None
    candidate_head._greedy_output = None
    candidate_head.scores = torch.empty_like(head.scores)
    candidate_head.cand_logits = torch.empty_like(head.cand_logits)
    if getattr(head, "cascade", False):
        candidate_head.refine_scores = torch.empty_like(head.refine_scores)
    greedy_mismatches = 0
    for m in [1,2,4]:
        hidden = torch.randn(m,2688,device='cuda',dtype=torch.bfloat16)
        expected = head.greedy(hidden)
        actual = candidate_head.greedy(hidden)
        greedy_mismatches += int((expected != actual).sum().item())
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            replay = candidate_head.greedy(hidden)
        for _ in range(3):
            g.replay()
            assert torch.equal(replay, expected), (m, expected, replay)
    result=dict(passed=greedy_mismatches==0 and all(r["passed"] for r in rows),greedy_mismatches=greedy_mismatches,bitwise_cases=bitwise_cases,elements=elements,rows=rows)
    Path(os.environ['NEMO_RESULT']).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    main()

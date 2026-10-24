"""Drafter-only full BF16 tensor-core up/ReLU2 screen vs current scalar tree.

Reads pinned MTP checkpoint tensors. No weight/activation format conversion and
no target-model changes. Accumulation differs: BF16 differences are reported,
not hidden behind a tolerance. Timings include the complete up/ReLU2 operation.
"""
import importlib.util
import json
import os
from pathlib import Path
import statistics
from unittest.mock import patch

import torch
from safetensors import safe_open
from vllm.triton_utils import tl, triton

ROOT = Path(__file__).resolve().parents[2]
with patch('vllm.utils.torch_utils.direct_register_custom_op'):
    spec = importlib.util.spec_from_file_location('mtp_up_control', ROOT/'vllm/vllm/model_executor/models/nemotron_h_mtp_kernels.py')
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)


@triton.jit
def up_dot(x, routed, shared, ids, out, I: tl.constexpr, H: tl.constexpr,
           BN: tl.constexpr, BK: tl.constexpr):
    # Replicate the exact single input row for a legal BF16 tensor-core tile;
    # each expert/output/K element is still read, in the original BF16 format.
    slot = tl.program_id(1)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    rows = tl.arange(0, 16)
    if slot < 6:
        expert = tl.load(ids + slot).to(tl.int64)
        weight = routed + expert * I * H
    else:
        weight = shared + (slot - 6).to(tl.int64) * I * H
    acc = tl.zeros((16, BN), tl.float32)
    for start in range(0, H, BK):
        k = start + kk
        a = tl.load(x + k, mask=k < H, other=0.0)
        # Explicit replication avoids loading duplicate activation bytes.
        a = tl.broadcast_to(a[None, :], (16, BK))
        b = tl.load(weight + n[None, :].to(tl.int64) * H + k[:, None],
                    mask=(n[None, :] < I) & (k[:, None] < H), other=0.0,
                    eviction_policy='evict_first')
        acc = tl.dot(a, b, acc)
    value = tl.sum(tl.where(rows[:, None] == 0, acc, 0.0), axis=0)
    value = tl.maximum(value, 0.0)
    value = value * value
    tl.store(out + slot * I + n, value.to(out.dtype.element_ty), mask=n < I)


def capture(fn):
    fn(); fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(12):
            fn()
    return g


def elapsed(g):
    samples = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize()
        samples.append(a.elapsed_time(b)*1000/12)
    return statistics.median(samples)


def main():
    torch.manual_seed(861271)
    snap = Path(os.environ['NEMO_SNAPSHOT'])
    index = json.loads((snap/'model.safetensors.index.json').read_text())['weight_map']
    def load(key):
        with safe_open(str(snap/index[key]), framework='pt', device='cuda') as f:
            return f.get_tensor(key)
    prefix = 'mtp.layers.1.mixer.'
    w = torch.stack([load(f'{prefix}experts.{e}.up_proj.weight') for e in range(128)])
    shared = load(prefix+'shared_experts.up_proj.weight')
    _, I, H = w.shape
    x = torch.randn((1,H), device='cuda', dtype=torch.bfloat16)
    ids = torch.randperm(128, device='cuda')[:6].to(torch.int32)
    ref = torch.empty((8,I), device='cuda', dtype=torch.bfloat16)
    def original():
        control._mtp_moe_up_kernel[(triton.cdiv(I,8),8)](
            x,w,shared,ids,ref,I,H,w.stride(0),w.stride(1),shared.stride(0),
            TOPK=6, BN=8, BK=1024, EVICT=True, num_warps=8, num_stages=3)
    original_graph = capture(original)
    result = {'draft_only': True, 'target_unchanged': True, 'rows': []}
    for bn,bk,warps,stages in [(32,32,4,2),(32,64,4,2),(64,32,4,2),(64,64,4,2),
                              (64,128,4,2),(64,64,4,3),(128,32,4,2),(128,64,4,2),
                              (128,64,8,2),(128,128,4,2),(128,128,8,2)]:
        out = torch.empty_like(ref)
        def candidate():
            up_dot[(triton.cdiv(I,bn),8)](x,w,shared,ids,out,I,H,BN=bn,BK=bk,
                                         num_warps=warps,num_stages=stages)
        try:
            g = capture(candidate)
            original(); candidate(); torch.cuda.synchronize()
            diff = int((ref.view(torch.int16)!=out.view(torch.int16)).sum())
            maximum = float((ref.float()-out.float()).abs().max())
            assert torch.isfinite(out).all()
            replay_equal = 0
            for _ in range(3):
                x.copy_(torch.randn_like(x));ids.copy_(torch.randperm(128,device='cuda')[:6].to(torch.int32))
                g.replay(); expected = out.clone();candidate();torch.cuda.synchronize()
                assert torch.equal(out.view(torch.int16),expected.view(torch.int16))
                replay_equal += 1
            pairs=[]
            for r in range(3):
                gs = {'control':original_graph,'candidate':g}
                order = ['control','candidate'] if r%2==0 else ['candidate','control']
                times = {k:elapsed(gs[k]) for k in order}
                pairs.append({'control_us':times['control'],'candidate_us':times['candidate'],
                              'speedup':times['control']/times['candidate']})
            row = {'bn':bn,'bk':bk,'warps':warps,'stages':stages,'elements':ref.numel(),
                   'bf16_differences':diff,'max_abs':maximum,'changing_replays':replay_equal,
                   'speedup':statistics.median(p['speedup'] for p in pairs),'pairs':pairs}
        except Exception as e:
            row = {'bn':bn,'bk':bk,'warps':warps,'stages':stages,'error':str(e)}
        result['rows'].append(row);print(json.dumps(row),flush=True)
        Path(os.environ['NEMO_RESULT']).write_text(json.dumps(result,indent=2)+'\n')
    assert any('speedup' in r for r in result['rows']), 'no candidate compiled'
    print('MTP_UP_DOT_SCREEN_DONE',flush=True)

if __name__ == '__main__':
    main()

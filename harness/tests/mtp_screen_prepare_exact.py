#!/usr/bin/env python3
"""Differential test for fused live-hidden conversion and screen column gathers."""
import importlib.util
import json
import os
from pathlib import Path

import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def bits_equal(a, b):
    assert torch.equal(a.view(torch.int32), b.view(torch.int32)), "FP32 bits differ"


def main():
    torch.manual_seed(69211)
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[2]
    base = load('screen_prepare_base', os.environ['MTP_ORIGINAL_SOURCE'])
    new = load('screen_prepare_new', root / 'vllm/vllm/model_executor/models/nemotron_h_mtp_screen.py')
    gathers = calls = replays = 0
    for cascade in ['1', '0']:
        os.environ['VLLM_MTP_SCREEN_CASCADE'] = cascade
        ref = base.DraftHeadScreen(os.environ['NEMO_SNAPSHOT'], torch.device('cuda'))
        fast = new.DraftHeadScreen(os.environ['NEMO_SNAPSHOT'], torch.device('cuda'))
        assert ref.graph is not None and fast.graph is not None, 'boot graph not captured'
        k = fast._h.numel()
        held = []
        for trial in range(18):
            h = torch.randn(1, k, device='cuda', dtype=torch.bfloat16)
            h *= [0., 0.01, 1., 10.][trial % 4]
            if trial % 6 == 0:
                h = torch.randn(1, k * 2, device='cuda', dtype=torch.bfloat16)[:, ::2]
            want = ref.greedy(h)
            got = fast.greedy(h)
            assert torch.equal(want, got), ('graph token', cascade, trial, want, got)
            bits_equal(ref._h, fast._h)
            bits_equal(fast._hs, h[0].float().index_select(0, ref.cols))
            if fast.cascade:
                bits_equal(fast._hc, h[0].float().index_select(0, ref.coarse_cols))
                bits_equal(ref.refine_scores, fast.refine_scores)
            bits_equal(ref.scores, fast.scores)
            bits_equal(ref.cand_logits, fast.cand_logits)
            held.append((got, want.clone()))
            for old, expected in held:
                assert torch.equal(old, expected), 'replayed output aliased retained draft'
            calls += 1
        # Test conversion/gather bits directly on signed zero, infinities and NaNs.
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            for trial in range(4):
                h = torch.randn(k * 2, device='cuda', dtype=dtype)[::2]
                h[:6] = torch.tensor([0., -0., float('inf'), -float('inf'), float('nan'), 1.001], device='cuda', dtype=dtype)
                hf = torch.empty(k, device='cuda')
                hs = torch.empty_like(fast._hs)
                hc = torch.empty_like(fast._hc)
                # A contiguous source has the same address layout as stable input.
                h = h.contiguous()
                new._prepare_screen_hidden[(new.triton.cdiv(k, 256),)](
                    h, hf, fast.cols, hs,
                    fast.coarse_cols if fast.cascade else fast.cols, hc,
                    K=k, S=fast.cols.numel(),
                    C=fast.coarse_cols.numel() if fast.cascade else 0,
                    CASCADE=fast.cascade, BLOCK=256,
                )
                bits_equal(hf, h.float())
                bits_equal(hs, h.float().index_select(0, ref.cols))
                if fast.cascade:
                    bits_equal(hc, h.float().index_select(0, ref.coarse_cols))
                gathers += 1
        for rows in [1, 2, 4]:
            for dtype in [torch.float32, torch.bfloat16]:
                h = torch.randn(rows, k, device='cuda', dtype=dtype)
                want = torch.stack([ref._greedy_row(row.float()) for row in h])
                got = fast.greedy(h)
                assert torch.equal(want, got), ('fallback token', cascade, rows, dtype)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    replayed = fast.greedy(h)
                for _ in range(3):
                    graph.replay()
                    assert torch.equal(want, replayed), 'outer capture fallback'
                    replays += 1
                calls += 1
    result = dict(passed=True, head_cases=calls, gather_cases=gathers, outer_replays=replays,
                  variants=['cascade', 'single-stage'], output_lifetime='independent cloned storage')
    Path(os.environ['NEMO_RESULT']).write_text(json.dumps(result, indent=2) + '\n')
    print('PREPARE VERIFIER PASS ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

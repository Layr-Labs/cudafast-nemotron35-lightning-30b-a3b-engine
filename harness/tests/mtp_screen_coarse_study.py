#!/usr/bin/env python3
"""Unranked coarse-stage bandwidth/recall tradeoff on captured live MTP rows.

Does not change target logits, checkpoint bytes, or the native-MTP weights.
Agreement is with promoted screening proposals, not with a hidden oracle.
"""
import importlib.util
import json
import os
from pathlib import Path
import time
import torch


def load(name, path, groups=32, survivors=256):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CASCADE_FRAC_GROUPS = groups
    module.CASCADE_PER = survivors
    return module


def measure(head, hidden):
    # Live input copies + graph replay + fresh output lifetime, same as runtime.
    for i in range(16):
        head.greedy(hidden[i:i+1])
    times = []
    for rep in range(7):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(64):
            head.greedy(hidden[i:i+1])
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 64)
    return times


def main():
    source = os.environ['MTP_ORIGINAL_SOURCE']
    snapshot = os.environ['NEMO_SNAPSHOT']
    hidden = torch.load(os.environ['MTP_HIDDEN_RESULT'], map_location='cuda', weights_only=True)
    assert hidden.ndim == 2 and hidden.shape[1] == 2688
    assert hidden.dtype == torch.bfloat16 and hidden.shape[0] >= 600
    device = torch.device('cuda')
    results = []
    baseline = None
    variants = json.loads(os.environ.get('MTP_COARSE_VARIANTS', '[[32,256],[24,256],[24,384],[24,512],[16,512]]'))
    for groups, survivors in variants:
        module = load(f'screen_g{groups}_s{survivors}', source, groups, survivors)
        head = module.DraftHeadScreen(snapshot, device)
        assert head.graph is not None, 'Per-row graph was not captured'
        assert head.w_coarse.shape == (131072, groups*8)
        assert head.refine_scores.numel() == 64*survivors
        t0 = time.monotonic()
        outputs = []
        with torch.inference_mode():
            for row in hidden:
                outputs.append(head.greedy(row.unsqueeze(0)))
        output = torch.cat(outputs)
        torch.cuda.synchronize()
        if baseline is None:
            baseline = output.clone()
        mismatch = (output != baseline).nonzero().reshape(-1).cpu().tolist()
        samples = measure(head, hidden)
        result = dict(coarse_groups=groups, survivors=survivors,
                      rows=len(hidden), matches=len(hidden)-len(mismatch),
                      mismatch_rows=mismatch, match_fraction=1-len(mismatch)/len(hidden),
                      graph_ms_samples=samples,
                      approx_weight_and_scale_bytes=(131072*groups + 64*survivors*64 + 2048*168)*9,
                      evaluation_wall_seconds=time.monotonic()-t0)
        results.append(result)
        print('COARSE_STUDY_RESULT', json.dumps(result), flush=True)
        del head
        torch.cuda.empty_cache()
    result = dict(diagnostic_only=True, source='live native MTP hidden rows from generic programming proxies',
                  comparison='6f120 promoted drafter proposals, not target oracle',
                  hidden_shape=list(hidden.shape), variants=results)
    Path(os.environ['NEMO_RESULT']).write_text(json.dumps(result, indent=2)+'\n')
    print('COARSE_STUDY_VERIFIER_PASS', len(results), 'variants', len(hidden), 'rows', flush=True)


if __name__ == '__main__':
    main()

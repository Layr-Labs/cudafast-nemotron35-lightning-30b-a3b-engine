"""GPU differential checks for neutral greedy speculative token-ID sampling.

Run in the pinned container with the staged overlay on PYTHONPATH.
The reference explicitly disables the new fast-path predicate, exercising the
unchanged vocabulary-copy path on the same logits and metadata.
"""
import json
from dataclasses import replace
from types import SimpleNamespace

import torch
import triton.testing

from vllm.v1.sample.logits_processor.builtin import (
    LogitBiasLogitsProcessor,
    MinTokensLogitsProcessor,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


class Reference(RejectionSampler):
    def _can_use_greedy_token_ids(self, sm):
        return False


def sampling():
    minimum = object.__new__(MinTokensLogitsProcessor)
    minimum.min_toks = {}
    bias = object.__new__(LogitBiasLogitsProcessor)
    bias.biases = {}
    return SamplingMetadata(
        temperature=None, all_greedy=True, all_random=False,
        top_p=None, top_k=None, generators={}, max_num_logprobs=None,
        no_penalties=True, prompt_token_ids=None,
        frequency_penalties=torch.zeros(1, device='cuda'),
        presence_penalties=torch.zeros(1, device='cuda'),
        repetition_penalties=torch.ones(1, device='cuda'),
        output_token_ids=[[]], allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=SimpleNamespace(non_argmax_invariant=[minimum, bias]),
    )


def case(lengths, vocab, dtype, mode, seed):
    torch.manual_seed(seed)
    n = sum(lengths) + len(lengths)
    logits = torch.randn(n, vocab, device='cuda', dtype=dtype)
    if mode == 'ties':
        logits.fill_(-float('inf'))
        logits[:, 1] = 3
        logits[:, -1] = 3
    elif mode == 'nan':
        logits[::2, 2] = float('nan')
        logits[::3, 4] = float('nan')
    elif mode == 'strided':
        logits = torch.randn(n, vocab * 2, device='cuda', dtype=dtype)[:, ::2]
    perm = torch.randperm(n, device='cuda')
    targets, bonus, at = [], [], 0
    for length in lengths:
        targets.extend(range(at, at + length))
        bonus.append(at + length)
        at += length + 1
    ti = perm[torch.tensor(targets, device='cuda', dtype=torch.long)]
    bi = perm[torch.tensor(bonus, device='cuda', dtype=torch.long)]
    drafts = logits.float().argmax(-1)[ti].to(torch.int32)
    if seed % 2 and len(drafts):
        drafts[seed % len(drafts)] = (drafts[seed % len(drafts)] + 1) % vocab
    cu = torch.tensor(lengths, device='cuda', dtype=torch.int32).cumsum(0).int()
    cs = torch.tensor([x + 1 for x in lengths], device='cuda', dtype=torch.int32).cumsum(0).int()
    meta = SpecDecodeMetadata(drafts, lengths, cu, cs, ti, bi, perm)
    return meta, logits


def main():
    sm = sampling()
    fast = RejectionSampler(Sampler(logprobs_mode='raw_logits'))
    ref = Reference(Sampler(logprobs_mode='raw_logits'))
    assert fast._can_use_greedy_token_ids(sm)
    guards = [
        {'all_greedy': False}, {'max_num_logprobs': 0},
        {'logprob_token_ids': {0: [1]}}, {'no_penalties': False},
        {'allowed_token_ids_mask': torch.zeros(1, 17, device='cuda', dtype=torch.bool)},
        {'bad_words_token_ids': {0: [[1]]}},
        {'thinking_budget_state_holder': object()},
        {'logitsprocs': SimpleNamespace(non_argmax_invariant=[object()])},
    ]
    for kwargs in guards:
        assert not fast._can_use_greedy_token_ids(replace(sm, **kwargs)), kwargs
    for processor, attr in zip(sm.logitsprocs.non_argmax_invariant, ['min_toks', 'biases']):
        setattr(processor, attr, {0: object()})
        assert not fast._can_use_greedy_token_ids(sm)
        setattr(processor, attr, {})
    fast.synthetic_mode = True
    assert not fast._can_use_greedy_token_ids(sm)
    fast.synthetic_mode = False
    checks = 0
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        for lengths in [[0], [1], [2], [3], [4], [5], [6], [0, 1, 3, 6]]:
            for mode in ['random', 'ties', 'nan', 'strided']:
                meta, logits = case(lengths, 257, dtype, mode, checks)
                actual = fast(meta, None, logits, sm)
                expected = ref(meta, None, logits, sm)
                torch.testing.assert_close(actual.sampled_token_ids, expected.sampled_token_ids, rtol=0, atol=0)
                assert actual.logprobs_tensors is None
                checks += 1
    for dtype in [torch.float32, torch.bfloat16]:
        meta, logits = case([3], 131072, dtype, 'random', checks)
        torch.testing.assert_close(fast(meta, None, logits, sm).sampled_token_ids,
                                   ref(meta, None, logits, sm).sampled_token_ids, rtol=0, atol=0)
        checks += 1
    # Also check the unmodified fallback when top logits are requested.
    with_lps = replace(sm, max_num_logprobs=4)
    actual = fast(meta, None, logits.clone(), with_lps)
    expected = ref(meta, None, logits.clone(), with_lps)
    torch.testing.assert_close(actual.sampled_token_ids, expected.sampled_token_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual.logprobs_tensors.logprobs, expected.logprobs_tensors.logprobs, rtol=0, atol=0)
    checks += 1
    meta, logits = case([3], 131072, torch.float32, 'random', 8)
    base_ms = triton.testing.do_bench(lambda: ref(meta, None, logits, sm), warmup=100, rep=500)
    fast_ms = triton.testing.do_bench(lambda: fast(meta, None, logits, sm), warmup=100, rep=500)
    report = {'device': torch.cuda.get_device_name(), 'exact_cases': checks,
              'guard_checks': len(guards) + 3, 'reference_ms': base_ms,
              'candidate_ms': fast_ms, 'speedup': base_ms / fast_ms}
    print(json.dumps(report, indent=2), flush=True)
    with open('.local-run/sampler-exact.json', 'w') as f:
        json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()

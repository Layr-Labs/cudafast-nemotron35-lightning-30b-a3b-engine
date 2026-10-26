"""The V1 runner skips the drafter only for a batch of max_tokens == 1 requests.

Needs the staged overlay (vLLM importable); skipped elsewhere. In the pinned
container:
  PYTHONPATH=.build/vllm-overlay python3 -m unittest harness/vllm_resident/test_drafter_skip.py -v
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

try:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
except ImportError:  # no vLLM on this host
    GPUModelRunner = None


def runner(*max_tokens):
    ids = [f"r{i}" for i in range(len(max_tokens))]
    reqs = {
        rid: SimpleNamespace(sampling_params=SamplingParams(max_tokens=m))
        for rid, m in zip(ids, max_tokens)
    }
    return SimpleNamespace(input_batch=SimpleNamespace(req_ids=ids), requests=reqs)


@unittest.skipIf(GPUModelRunner is None, "vLLM is not importable here")
class DrafterSkip(unittest.TestCase):
    def check(self, expected, *max_tokens):
        self.assertIs(GPUModelRunner._drafts_can_be_verified(runner(*max_tokens)), expected)

    def test_single_token_request_skips_the_drafter(self):
        self.check(False, 1)

    def test_longer_request_keeps_the_drafter(self):
        self.check(True, 7695)
        self.check(True, 2)

    def test_mixed_batch_keeps_the_drafter(self):
        self.check(True, 1, 64)

    def test_empty_batch_keeps_the_default(self):
        self.check(True)


if __name__ == "__main__":
    unittest.main()

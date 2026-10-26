"""CPU routing checks; these do not compile or execute the GPU sampler.

Load the actual changed Python methods with AST, use NumPy for tensor indexing,
and a scalar oracle for the unchanged acceptance kernel. No vLLM import needed.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

try:
    import numpy as np
except ImportError:
    np = None


class Fallback(Exception):
    pass


class Sampler:
    def __call__(self, **kwargs):
        raise Fallback("original bonus sampler reached")


class MinTokensLogitsProcessor:
    def __init__(self):
        self.min_toks = {}


class LogitBiasLogitsProcessor:
    def __init__(self):
        self.biases = {}


class Tensor:
    def __init__(self, value):
        self.data = np.asarray(value)
        self.dtype = self.data.dtype
        self.device = "cpu"

    def __getitem__(self, index):
        return Tensor(self.data[index.data if isinstance(index, Tensor) else index])

    def argmax(self, dim):
        return Tensor(self.data.argmax(axis=dim))

    def contiguous(self):
        return self

    def to(self, dtype):
        return Tensor(self.data.astype(dtype))

    def unsqueeze(self, dim):
        return Tensor(np.expand_dims(self.data, dim))


class AcceptanceOracle:
    """Scalar specification of exact greedy acceptance, not a GPU kernel mockup."""

    def __getitem__(self, grid):
        def run(out, cu, drafts, targets, bonus, greedy, depth, uniform, rates,
                *, SYNTHETIC_MODE):
            assert not SYNTHETIC_MODE
            assert greedy is None and uniform is None and rates is None
            assert out.dtype == np.int32 and bonus.dtype == np.int32
            start = 0
            for request, end in enumerate(cu.data):
                for position in range(end - start):
                    out.data[request, position] = targets.data[start + position]
                    if drafts.data[start + position] != targets.data[start + position]:
                        break
                else:
                    out.data[request, end - start] = bonus.data[request, 0]
                start = end
            assert grid == (len(cu.data),)
        return run


def load_methods():
    path = Path(__file__).resolve().parents[2] / (
        "vllm/vllm/v1/sample/rejection_sampler.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "RejectionSampler")
    names = {"forward", "_can_use_greedy_argmax", "_greedy_argmax_sample"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in names]
    torch = NS(float16=np.dtype("float16"), bfloat16="bfloat16",
               float32=np.dtype("float32"), int32=np.dtype("int32"),
               full=lambda shape, fill, dtype, device: Tensor(
                   np.full(shape, fill, dtype=dtype)))
    namespace = dict(torch=torch, Sampler=Sampler, SamplerOutput=NS,
                     MinTokensLogitsProcessor=MinTokensLogitsProcessor,
                     LogitBiasLogitsProcessor=LogitBiasLogitsProcessor,
                     rejection_greedy_sample_kernel=AcceptanceOracle(),
                     PLACEHOLDER_TOKEN_ID=-1, MAX_SPEC_LEN=128, replace=replace)
    source = "from __future__ import annotations\n" + "\n".join(
        ast.unparse(method) for method in methods
    )
    exec(compile(source, str(path), "exec"), namespace)
    return type("ExtractedRejectionSampler", (), {
        name: namespace[name] for name in names
    })


def sampling_metadata():
    return NS(all_greedy=True, all_random=False, max_num_logprobs=None,
              logprob_token_ids=None, no_penalties=True,
              allowed_token_ids_mask=None, bad_words_token_ids={},
              thinking_budget_state_holder=None,
              logitsprocs=NS(non_argmax_invariant=[MinTokensLogitsProcessor(),
                                                  LogitBiasLogitsProcessor()]))


@unittest.skipIf(np is None, "NumPy unavailable; GPU verification still required")
class GreedyArgmaxRouting(unittest.TestCase):
    def setUp(self):
        self.engine = load_methods()()
        self.engine.sampler = Sampler()
        self.engine.synthetic_mode = False

    def test_default_installed_but_inactive_processors_allow_fast_path(self):
        self.assertTrue(self.engine._can_use_greedy_argmax(sampling_metadata()))

    def test_every_active_constraint_falls_back(self):
        changes = {
            "all_greedy": False, "all_random": True, "max_num_logprobs": 0,
            "logprob_token_ids": {0: [1]}, "no_penalties": False,
            "allowed_token_ids_mask": object(), "bad_words_token_ids": {0: [[1]]},
            "thinking_budget_state_holder": NS(has_tracked_requests=lambda: True),
        }
        for name, value in changes.items():
            with self.subTest(constraint=name):
                md = sampling_metadata()
                setattr(md, name, value)
                self.assertFalse(self.engine._can_use_greedy_argmax(md))
        md = sampling_metadata()
        md.thinking_budget_state_holder = NS(has_tracked_requests=lambda: False)
        self.assertTrue(self.engine._can_use_greedy_argmax(md))

    def test_processors_synthetic_and_sampler_subclasses_fall_back(self):
        for index, attribute in [(0, "min_toks"), (1, "biases")]:
            md = sampling_metadata()
            setattr(md.logitsprocs.non_argmax_invariant[index], attribute, {0: 1})
            self.assertFalse(self.engine._can_use_greedy_argmax(md))
        for processor in [object(), type("CustomMin", (MinTokensLogitsProcessor,), {})()]:
            md = sampling_metadata()
            md.logitsprocs.non_argmax_invariant.append(processor)
            self.assertFalse(self.engine._can_use_greedy_argmax(md))
        self.engine.synthetic_mode = True
        self.assertFalse(self.engine._can_use_greedy_argmax(sampling_metadata()))
        self.engine.synthetic_mode = False
        self.engine.sampler = type("CustomSampler", (Sampler,), {})()
        self.assertFalse(self.engine._can_use_greedy_argmax(sampling_metadata()))

    def test_depths_zero_to_six_and_every_rejection_position(self):
        for dtype in [np.float16, np.float32]:
            for depth in range(7):
                for rejected in range(depth + 1):
                    with self.subTest(dtype=dtype, depth=depth, rejected=rejected):
                        # All rows tie at tokens 1 and 2: the lowest ID must win.
                        logits = Tensor(np.tile([0, 7, 7, -3], (depth + 1, 1)).astype(dtype))
                        drafts = [1] * depth
                        if rejected < depth:
                            drafts[rejected] = 2
                        md = NS(max_spec_len=depth, num_draft_tokens=[depth],
                                target_logits_indices=Tensor(np.arange(depth)),
                                bonus_logits_indices=Tensor([depth]),
                                cu_num_draft_tokens=Tensor([depth]),
                                draft_token_ids=Tensor(drafts))
                        before = logits.data.copy()
                        result = self.engine.forward(md, None, logits, sampling_metadata())
                        expected = [1] * (rejected + 1) + [-1] * (depth - rejected)
                        self.assertEqual(result.sampled_token_ids.data.tolist(), [expected])
                        self.assertIsNone(result.logprobs_tensors)
                        np.testing.assert_array_equal(before, logits.data)

    def test_permuted_rows_heterogeneous_batch_and_masked_logits(self):
        logits = Tensor(np.array([[3, 1, 0], [-np.inf, -np.inf, -np.inf],
                                  [0, 0, 9], [1, 8, 1], [2, 4, 1]], dtype=np.float32))
        md = NS(max_spec_len=2, num_draft_tokens=[0, 2],
                target_logits_indices=Tensor([2, 0]),
                bonus_logits_indices=Tensor([1, 3]),
                cu_num_draft_tokens=Tensor([0, 2]), draft_token_ids=Tensor([2, 0]))
        result = self.engine.forward(md, None, logits, sampling_metadata())
        self.assertEqual(result.sampled_token_ids.data.tolist(), [[0, -1, -1], [2, 0, 1]])

    def test_float64_and_requested_logprobs_reach_original_sampler(self):
        md = NS(max_spec_len=0, num_draft_tokens=[0],
                target_logits_indices=Tensor(np.array([], dtype=np.int64)),
                bonus_logits_indices=Tensor([0]))
        # replace() is reached on fallback, so use a dataclass metadata instance.
        from dataclasses import make_dataclass
        for dtype, logprobs in [(np.float64, None), (np.float32, 8)]:
            values = vars(sampling_metadata())
            values["max_num_logprobs"] = logprobs
            meta_type = make_dataclass("Metadata", [(k, object) for k in values])
            with self.assertRaises(Fallback):
                self.engine.is_processed_logprobs_mode = False
                self.engine.forward(md, None, Tensor(np.ones((1, 3), dtype=dtype)),
                                    meta_type(**values))


if __name__ == "__main__":
    unittest.main()

"""Greedy fast-path routing and independent GPU acceptance checks."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

SOURCE = Path(
    os.environ.get(
        "FUSED_SAMPLER_SOURCE",
        Path(__file__).resolve().parents[2]
        / "vllm/vllm/v1/sample/rejection_sampler.py",
    )
)


class Sampler:
    pass


class MinTokensLogitsProcessor:
    min_toks = {}


class LogitBiasLogitsProcessor:
    biases = {}


def metadata():
    return NS(
        all_greedy=True,
        all_random=False,
        max_num_logprobs=None,
        logprob_token_ids=None,
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        thinking_budget_state_holder=None,
        logitsprocs=NS(
            non_argmax_invariant=[
                MinTokensLogitsProcessor(),
                LogitBiasLogitsProcessor(),
            ]
        ),
    )


class Routing(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(SOURCE.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "RejectionSampler"
        )
        fn = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_can_use_greedy_argmax"
        )
        ns = dict(
            Sampler=Sampler,
            MinTokensLogitsProcessor=MinTokensLogitsProcessor,
            LogitBiasLogitsProcessor=LogitBiasLogitsProcessor,
        )
        exec("from __future__ import annotations\n" + ast.unparse(fn), ns)
        self.engine = NS(sampler=Sampler(), synthetic_mode=False)
        self.allowed = lambda md: ns["_can_use_greedy_argmax"](self.engine, md)

    def test_only_unconstrained_greedy_sampling_is_allowed(self):
        self.assertTrue(self.allowed(metadata()))
        for key, value in dict(
            all_greedy=False,
            all_random=True,
            max_num_logprobs=0,
            logprob_token_ids={0: [1]},
            no_penalties=False,
            allowed_token_ids_mask=object(),
            bad_words_token_ids={0: [[1]]},
            thinking_budget_state_holder=NS(has_tracked_requests=lambda: True),
        ).items():
            with self.subTest(key=key):
                md = metadata()
                setattr(md, key, value)
                self.assertFalse(self.allowed(md))

    def test_active_or_unknown_processors_fall_back(self):
        for proc in [
            NS(),
            type("Custom", (MinTokensLogitsProcessor,), {})(),
            type("ActiveMin", (), {"min_toks": {0: 1}})(),
        ]:
            md = metadata()
            md.logitsprocs.non_argmax_invariant.append(proc)
            self.assertFalse(self.allowed(md))
        md = metadata()
        md.logitsprocs.non_argmax_invariant[0].min_toks = {0: 1}
        self.assertFalse(self.allowed(md))
        md = metadata()
        md.logitsprocs.non_argmax_invariant[1].biases = {0: 1}
        self.assertFalse(self.allowed(md))

    def test_custom_sampler_and_synthetic_mode_fall_back(self):
        self.engine.synthetic_mode = True
        self.assertFalse(self.allowed(metadata()))
        self.engine.synthetic_mode = False
        self.engine.sampler = type("CustomSampler", (Sampler,), {})()
        self.assertFalse(self.allowed(metadata()))


try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


@unittest.skipUnless(HAS_CUDA, "requires pinned vLLM container and GPU")
class GpuAcceptance(unittest.TestCase):
    def test_all_rejection_positions_and_mixed_batch_lengths(self):
        import importlib.util, itertools
        from vllm.v1.sample.sampler import Sampler as RealSampler
        from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

        spec = importlib.util.spec_from_file_location(
            "fused_sampler_test_target", SOURCE
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        engine = mod.RejectionSampler(RealSampler())
        torch.manual_seed(2917)
        cases = 0
        for dtype, depth, batch in itertools.product(
            [torch.float16, torch.bfloat16, torch.float32],
            [0, 1, 2, 3, 4, 5, 6, 16, 128],
            [1, 3],
        ):
            counts = [depth] + ([depth // 2, 0] if batch == 3 else [])
            rows = sum(counts) + batch
            vocab = 131072 if depth <= 6 else 257
            logits = torch.randn((rows, vocab), device="cuda", dtype=dtype)
            logits[:, 2] = 17
            logits[:, 7] = 17
            if depth % 3 == 0:
                logits[0, 11] = float("nan")
            if depth % 3 == 1:
                logits[0, 11] = float("inf")
            order = torch.randperm(rows, device="cuda")
            targets = []
            bonuses = []
            offset = 0
            for n in counts:
                targets.append(order[offset : offset + n])
                bonuses.append(order[offset + n])
                offset += n + 1
            target = torch.cat(targets).int()
            bonus = torch.stack(bonuses).int()
            row_ids = logits.argmax(-1)
            ids = row_ids[target.long()].int()
            target_ids = ids.cpu().tolist()
            bonus_ids = row_ids[bonus.long()].cpu().tolist()
            cu = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(
                0, dtype=torch.int32
            )
            for reject in range(depth + 1):
                draft = ids.clone()
                draft_cpu = target_ids.copy()
                if reject < depth:
                    draft[reject] = (draft[reject] + 1) % vocab
                    draft_cpu[reject] = (draft_cpu[reject] + 1) % vocab
                md = SpecDecodeMetadata(
                    draft,
                    counts,
                    cu,
                    cu + torch.arange(1, batch + 1, device="cuda", dtype=torch.int32),
                    target,
                    bonus,
                    torch.arange(rows, device="cuda"),
                )
                expected = []
                start = 0
                for request, n in enumerate(counts):
                    accepted = []
                    for j in range(n):
                        accepted.append(target_ids[start + j])
                        if draft_cpu[start + j] != target_ids[start + j]:
                            break
                    else:
                        accepted.append(bonus_ids[request])
                    expected.append(accepted + [-1] * (depth + 1 - len(accepted)))
                    start += n
                actual = (
                    engine._greedy_argmax_sample(logits, md)
                    .sampled_token_ids.cpu()
                    .tolist()
                )
                self.assertEqual(actual, expected, (dtype, depth, batch, reject))
                cases += 1
        self.assertEqual(cases, 1044)


if __name__ == "__main__":
    unittest.main()

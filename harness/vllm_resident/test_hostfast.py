"""GPU-free tests of vllm/vllm/v1/worker/hostfast.py (the host-side trims).

The module needs only numpy and torch, so it is loaded straight from its file
(no vLLM import). Skipped where torch is missing.

Run: python3 -m unittest discover -s harness/vllm_resident -v
"""

from __future__ import annotations

import gc
import importlib.util
import os
import random
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "..", "..", "vllm", "vllm", "v1", "worker", "hostfast.py")

try:
    import numpy as np
    import torch

    _spec = importlib.util.spec_from_file_location("hostfast_under_test", PATH)
    hostfast = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(hostfast)
except ImportError:  # no torch on this host
    hostfast = None


def kernel_batch_change(gpu_prev, prev_positions, valid, prev_drafts, cpu_num_computed):
    """update_num_computed_tokens_for_batch_change (vllm/v1/spec_decode/utils.py)
    on CPU tensors, line for line."""
    gather = prev_positions.clamp(min=0)
    valid_counts = valid[gather]
    prev_computed = gpu_prev[gather]
    drafts = prev_drafts[gather]
    participating = (prev_positions >= 0) & (drafts > 0)
    corrected = prev_computed + valid_counts.int()
    return torch.where(participating, corrected, cpu_num_computed)


@unittest.skipIf(hostfast is None, "torch is not importable here")
class ExactVerifySeqLens(unittest.TestCase):
    """Simulate the async scheduler + runner + GPU correction kernel over many
    steps and check the CPU result against the kernel's (and the truth)."""

    def run_sim(self, seed: int, depth: int, steps: int = 400, max_reqs: int = 3):
        rng = random.Random(seed)
        # Per request: true computed count at the start of its next step, the
        # scheduler's num_computed_tokens, the rejections the scheduler has not
        # applied yet, and the previous step's (drafts, valid) for the runner.
        reqs: dict[str, dict] = {}
        next_id = 0
        prev_order: list[str] = []  # batch order of the previous step
        prev_info: dict[str, tuple[int, int]] = {}  # rid -> (drafts, valid)
        gpu_prev = torch.zeros(max_reqs, dtype=torch.int32)  # GPU num_computed
        pending_update: list[tuple[str, int]] = []  # (rid, rejected) of step N-1
        checked = 0
        for _ in range(steps):
            # Requests join (prefill) or leave at random; order may change.
            if len(reqs) < max_reqs and rng.random() < 0.08:
                rid = f"r{next_id}"
                next_id += 1
                reqs[rid] = {"true": 0, "sched": 0, "prompt": rng.randint(5, 600), "new": True}
            if reqs and rng.random() < 0.03:
                gone = rng.choice(sorted(reqs))
                del reqs[gone]
            if not reqs:
                continue
            order = sorted(reqs)
            if rng.random() < 0.2:
                rng.shuffle(order)
            # schedule(N): output num_computed, then advance by the scheduled count.
            plan = {}
            for rid in order:
                r = reqs[rid]
                if r["new"]:
                    n_sched, drafts = r["prompt"], 0
                else:
                    drafts = rng.randint(0, depth)
                    n_sched = 1 + drafts
                plan[rid] = (r["sched"], n_sched, drafts)
                r["sched"] += n_sched
            # update_from_output(N-1) runs after schedule(N).
            for rid, rejected in pending_update:
                if rid in reqs:
                    reqs[rid]["sched"] -= rejected
            # Runner inputs for step N.
            n = len(order)
            optimistic = torch.zeros(max_reqs, dtype=torch.int32)
            cpu_nc = torch.zeros(n, dtype=torch.int32)
            prev_positions = np.full(max_reqs, -1, dtype=np.int64)
            prev_drafts = np.zeros(max_reqs, dtype=np.int32)
            valid = torch.zeros(max_reqs, dtype=torch.int32)
            for j, rid in enumerate(prev_order):
                if rid in prev_info:
                    valid[j] = prev_info[rid][1]
            for i, rid in enumerate(order):
                s, n_sched, _ = plan[rid]
                optimistic[i] = s + n_sched
                cpu_nc[i] = s
                if rid in prev_order:
                    p = prev_order.index(rid)
                    prev_positions[i] = p
                    prev_drafts[p] = prev_info[rid][0]
            kernel_ran = bool(prev_order)
            if kernel_ran:
                gpu_now = kernel_batch_change(
                    gpu_prev,
                    torch.from_numpy(prev_positions[:n]),
                    valid,
                    torch.from_numpy(prev_drafts),
                    cpu_nc,
                )
                fast = hostfast.exact_verify_seq_lens(
                    optimistic[:n], n, prev_positions, prev_drafts, valid.tolist()
                )
            else:
                gpu_now = cpu_nc.clone()
                fast = optimistic[:n].clone()
            gpu_seq = gpu_now + torch.tensor([plan[rid][1] for rid in order], dtype=torch.int32)
            truth = torch.tensor(
                [reqs[rid]["true"] + plan[rid][1] for rid in order], dtype=torch.int32
            )
            self.assertIsNotNone(fast)
            self.assertTrue(torch.equal(fast, gpu_seq), (fast, gpu_seq))
            self.assertTrue(torch.equal(gpu_seq, truth), (gpu_seq, truth))
            checked += 1
            # Execute step N: accept a random prefix of the drafts.
            gpu_prev = torch.zeros(max_reqs, dtype=torch.int32)
            gpu_prev[:n] = gpu_now
            pending_update, prev_info = [], {}
            for rid in order:
                r = reqs[rid]
                _, n_sched, drafts = plan[rid]
                if r["new"]:
                    r["true"] += n_sched
                    r["new"] = False
                    prev_info[rid] = (0, 1)
                    pending_update.append((rid, 0))
                    continue
                accepted = rng.randint(0, drafts)
                r["true"] += 1 + accepted
                prev_info[rid] = (drafts, 1 + accepted)
                pending_update.append((rid, drafts - accepted))
            prev_order = order
        return checked

    def test_matches_the_gpu_kernel_at_every_depth(self):
        total = 0
        for depth in range(1, 7):
            for seed in range(6):
                total += self.run_sim(seed * 31 + depth, depth)
        self.assertGreater(total, 10000)

    def test_padding_rows_stay_zero(self):
        optimistic = torch.tensor([500, 0, 0], dtype=torch.int32)
        out = hostfast.exact_verify_seq_lens(
            optimistic, 1, np.array([0, -1, -1]), np.array([3, 0, 0]), [2]
        )
        self.assertEqual(out.tolist(), [498, 0, 0])
        self.assertEqual(optimistic.tolist(), [500, 0, 0], "the input is not mutated")

    def test_a_missing_valid_count_falls_back(self):
        out = hostfast.exact_verify_seq_lens(
            torch.tensor([10, 20], dtype=torch.int32), 2, np.array([0, 1]), np.array([2, 2]), [3]
        )
        self.assertIsNone(out)

    def test_cpu_corrections_preserve_strided_inputs_and_padding(self):
        old = hostfast.CPU_INTS
        try:
            for enabled in (False, True):
                hostfast.CPU_INTS = enabled
                for dtype in (torch.int32, torch.int64):
                    storage = torch.tensor([500, 999, 0, 888], dtype=dtype)
                    out = hostfast.exact_verify_seq_lens(
                        storage[::2], 1, np.array([0]), np.array([3]), [2]
                    )
                    self.assertEqual(out.tolist(), [498, 0])
                    self.assertEqual(out.dtype, dtype)
                    out[0] += 1
                    self.assertEqual(storage.tolist(), [500, 999, 0, 888])
        finally:
            hostfast.CPU_INTS = old


@unittest.skipIf(hostfast is None, "torch is not importable here")
class DraftSeqLens(unittest.TestCase):
    def test_rejected_matches_the_padded_prepare_kernel(self):
        # eagle_prepare_inputs_padded_kernel: num_draft + 1 - valid, 0 without drafts.
        rej = hostfast.num_rejected_tokens([3, 0, 2, 1], [1, 1, 3, 2])
        self.assertEqual(rej.tolist(), [3, 0, 0, 0])
        self.assertEqual(rej.dtype, torch.int32)
        self.assertIsNone(hostfast.num_rejected_tokens([3, 3], [1]))

    def test_draft_seq_lens_is_verify_minus_rejected(self):
        verify = torch.tensor([600, 41], dtype=torch.int32)
        rej = torch.tensor([2, 0], dtype=torch.int32)
        out = hostfast.draft_seq_lens(verify, rej, 2, 8192, 3)
        self.assertEqual(out.tolist(), [598, 41])
        self.assertEqual(verify.tolist(), [600, 41], "a fresh tensor: the loop adds to it")
        out += 1
        self.assertEqual(verify.tolist(), [600, 41])

    def test_falls_back_on_missing_input_shape_or_context_end(self):
        verify = torch.tensor([600], dtype=torch.int32)
        rej = torch.tensor([1], dtype=torch.int32)
        self.assertIsNone(hostfast.draft_seq_lens(None, rej, 1, 8192, 3))
        self.assertIsNone(hostfast.draft_seq_lens(verify, None, 1, 8192, 3))
        self.assertIsNone(hostfast.draft_seq_lens(verify, rej, 2, 8192, 3))
        # 599 + 3 draft positions reach max_model_len 602: the GPU step kernel
        # clamps there and the CPU += 1 would not.
        self.assertIsNone(hostfast.draft_seq_lens(verify, rej, 1, 602, 3))
        self.assertIsNotNone(hostfast.draft_seq_lens(verify, rej, 1, 603, 3))

    def test_check_raises_on_a_mismatch(self):
        hostfast.check_seq_lens("draft", torch.tensor([5]), torch.tensor([5]))
        with self.assertRaises(RuntimeError):
            hostfast.check_seq_lens("draft", torch.tensor([5]), torch.tensor([6]))


@unittest.skipIf(hostfast is None, "torch is not importable here")
class FastCpus(unittest.TestCase):
    def sysfs(self, values: dict[int, dict[str, int]]) -> str:
        root = tempfile.mkdtemp()
        for cpu, leaves in values.items():
            for leaf, value in leaves.items():
                path = os.path.join(root, f"cpu{cpu}", leaf)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as fp:
                    fp.write(f"{value}\n")
        return root

    def test_gb10_capacities_pick_the_x925_cores(self):
        cap = {c: 718 for c in range(5)}
        cap.update({c: 997 for c in range(5, 10)})
        cap.update({c: 731 for c in range(10, 15)})
        cap.update({c: 1017 for c in range(15, 19)})
        cap[19] = 1024
        root = self.sysfs({c: {"cpu_capacity": v} for c, v in cap.items()})
        fast = hostfast.fast_cpus(set(range(20)), sysfs=root)
        self.assertEqual(fast, set(range(5, 10)) | set(range(15, 20)))
        # Within a narrower allowed set, only those.
        self.assertEqual(hostfast.fast_cpus({0, 1, 5, 19}, sysfs=root), {5, 19})

    def test_frequency_is_the_fallback(self):
        root = self.sysfs(
            {c: {"cpufreq/cpuinfo_max_freq": 3900000 if c % 2 else 2808000} for c in range(4)}
        )
        self.assertEqual(hostfast.fast_cpus(set(range(4)), sysfs=root), {1, 3})

    def test_uniform_or_unreadable_leaves_the_affinity_alone(self):
        root = self.sysfs({c: {"cpu_capacity": 1024} for c in range(4)})
        self.assertEqual(hostfast.fast_cpus(set(range(4)), sysfs=root), set())
        root = self.sysfs({0: {"cpu_capacity": 1024}, 1: {"cpu_capacity": 500}})
        self.assertEqual(hostfast.fast_cpus({0, 1, 2}, sysfs=root), set())
        self.assertEqual(hostfast.fast_cpus({0, 1}, sysfs=tempfile.mkdtemp()), set())


@unittest.skipIf(hostfast is None, "torch is not importable here")
class GcHooks(unittest.TestCase):
    def setUp(self):
        self.was_enabled = gc.isenabled()
        gc.enable()

    def tearDown(self):
        hostfast.gc_idle()
        if self.was_enabled:
            gc.enable()
        else:
            gc.disable()

    def test_busy_pauses_and_idle_restores(self):
        old = hostfast.GC
        hostfast.GC = True
        try:
            hostfast.gc_busy()
            self.assertFalse(gc.isenabled())
            hostfast.gc_in_shadow()  # collects; must not re-enable
            self.assertFalse(gc.isenabled())
            hostfast.gc_idle()
            self.assertTrue(gc.isenabled())
        finally:
            hostfast.GC = old

    def test_switch_off_is_a_no_op(self):
        old = hostfast.GC
        hostfast.GC = False
        try:
            hostfast.gc_busy()
            self.assertTrue(gc.isenabled())
        finally:
            hostfast.GC = old

    def test_a_gc_someone_else_disabled_stays_disabled(self):
        old = hostfast.GC
        hostfast.GC = True
        gc.disable()
        try:
            hostfast.gc_busy()
            hostfast.gc_idle()
            self.assertFalse(gc.isenabled())
        finally:
            hostfast.GC = old


if __name__ == "__main__":
    unittest.main()

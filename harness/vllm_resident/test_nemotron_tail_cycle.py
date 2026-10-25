"""CPU guard/lifecycle tests for the production tail-cycle helper.

These execute its actual Python functions. CUDA events and graph replay are
explicit control-flow doubles; neither GPU numerics nor throughput is claimed.
The real startup admission still compares every listed state/output byte.
"""
from __future__ import annotations

from contextlib import nullcontext
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch

import torch

PATH = Path(__file__).resolve().parents[2] / "vllm/vllm/v1/spec_decode/nemotron_tail_cycle.py"
spec = importlib.util.spec_from_file_location("tail_cycle_under_test", PATH)
tail = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tail)


def common(length=30):
    return NS(num_reqs=1, num_actual_tokens=1, max_seq_len=length,
              seq_lens=torch.tensor([length], dtype=torch.int32),
              _seq_lens_cpu=torch.tensor([length], dtype=torch.int32),
              _num_computed_tokens_cpu=torch.tensor([length - 1], dtype=torch.int32),
              seq_lens_cpu_upper_bound=torch.tensor([length + 2], dtype=torch.int32),
              block_table_tensor=torch.arange(8, dtype=torch.int32).view(1, -1))


def runtime_candidate(trace):
    candidate = tail.SamePageTail.__new__(tail.SamePageTail)
    p = NS(device=torch.device("cpu"), num_speculative_tokens=4,
           block_size=16, max_model_len=128,
           input_ids=torch.tensor([2], dtype=torch.int32),
           positions=torch.tensor([29], dtype=torch.long),
           hidden_states=torch.zeros(1, 4), _slot_mapping_buffer=torch.tensor([13]))
    p._get_positions = lambda n: p.positions[:n]
    candidate.p, candidate.padded = p, 1
    candidate.seq = torch.zeros(1, dtype=torch.int32)
    candidate.block_table = torch.zeros(1, 8, dtype=torch.int32)
    candidate.tokens = torch.tensor([[5, 7, 11]], dtype=torch.int32)
    candidate.signature = "plan"
    candidate.admitted = True
    candidate.input_signature = tuple((k, tail.tensor_signature(v) if v is not None else None)
                                      for k, v in sorted(dict(input_ids=p.input_ids[:1],
                                          positions=p.positions[:1], inputs_embeds=None,
                                          hidden_states=p.hidden_states[:1]).items()))
    def replay():
        trace.append("replay")
        candidate.seq += 2
    candidate.graph = NS(replay=replay)
    candidate.admission_cases = []
    candidate.hidden_steps = [torch.zeros(1, 4) for _ in range(3)]
    return candidate


class TailCycleTests(unittest.TestCase):
    def test_only_the_pinned_flashinfer_planner_is_eligible(self):
        with patch.object(tail, "version", return_value="0.6.16.post3"):
            self.assertTrue(tail.supported_flashinfer())
        with patch.object(tail, "version", return_value="0.6.16.post4"):
            self.assertFalse(tail.supported_flashinfer())
        with patch.object(tail, "version", side_effect=tail.PackageNotFoundError):
            self.assertFalse(tail.supported_flashinfer())

    def test_actual_plan_signature_checks_native_types_scalars_and_buffers(self):
        class FIDecode(NS):
            pass
        class FlashInferMetadata(NS):
            pass
        name = "vllm.v1.attention.backends.flashinfer"
        module = ModuleType(name)
        module.FIDecode, module.FlashInferMetadata = FIDecode, FlashInferMetadata
        w = NS(_backend="fa2", use_tensor_cores=True, is_cuda_graph_enabled=True,
               _q_len_per_req=1, _fixed_batch_size=1, _jit_module=None,
               _cached_module=object(), _plan_info=[1, 2, 3], _window_left=-1,
               _logits_soft_cap=0.0, _sm_scale=0.5, _pos_encoding_mode="NONE",
               _rope_scale=1.0, _rope_theta=10000.0, _kv_layout="HND")
        for attr in ("_float_workspace_buffer", "_int_workspace_buffer", "_paged_kv_indptr_buf",
                     "_paged_kv_indices_buf", "_paged_kv_last_page_len_buf", "_qo_indptr_buf"):
            setattr(w, attr, torch.zeros(4))
        builder = NS(page_size=16, num_qo_heads=1, num_kv_heads=1, head_dim=4,
                     decode_fixed_split_size=-1, disable_split_kv=False)
        for attr in ("paged_kv_indptr", "paged_kv_indices", "paged_kv_last_page_len"):
            setattr(builder, attr, NS(gpu=torch.zeros(4)))
        p = NS(draft_attn_groups=[NS(get_metadata_builder=lambda: builder)],
               compilation_config=NS(static_forward_context={"draft": NS(kv_cache=torch.zeros(4, 1, 16, 4))}))
        md = FlashInferMetadata(decode=FIDecode(wrapper=w), num_actual_tokens=1,
            num_decodes=1, num_decode_tokens=1, num_prefills=0, use_cascade=False,
            causal=True, q_data_type_decode=torch.bfloat16, slot_mapping=torch.zeros(1))
        with patch.dict(sys.modules, {name: module}):
            original = tail.plan_signature(p, {"draft": md})
            self.assertIsNotNone(original)
            for attr, bad in (("_backend", "fa3"), ("use_tensor_cores", False),
                              ("is_cuda_graph_enabled", False), ("_q_len_per_req", 2),
                              ("_fixed_batch_size", 2), ("_jit_module", object())):
                saved = getattr(w, attr)
                setattr(w, attr, bad)
                self.assertIsNone(tail.plan_signature(p, {"draft": md}), attr)
                setattr(w, attr, saved)
            for attr in ("_prefix_len_ptr", "_custom_mask_buf", "_token_pos_in_items_ptr"):
                setattr(w, attr, torch.ones(1))
                self.assertIsNone(tail.plan_signature(p, {"draft": md}), attr)
                setattr(w, attr, None)
            w._plan_info = [1, 2, 4]
            self.assertNotEqual(tail.plan_signature(p, {"draft": md}), original)
            w._plan_info = [1, 2, 3]
            w._qo_indptr_buf = w._qo_indptr_buf.clone()
            self.assertNotEqual(tail.plan_signature(p, {"draft": md}), original)
            del w._qo_indptr_buf
            self.assertIsNone(tail.plan_signature(p, {"draft": md}))

    def test_same_page_exhaustive_boundaries_and_invalid_types(self):
        for page in (1, 8, 16, 32, 64):
            for length in range(8195):
                expected = (length > 0 and length + 2 <= 8192
                            and len({(x - 1) // page for x in range(length, length + 3)}) == 1)
                self.assertEqual(tail.same_page(length, page, 8192), expected)
        for args in ((True, 16, 8192), (1.0, 16, 8192), (1, 0, 8192), (1, -1, 8192)):
            self.assertFalse(tail.same_page(*args))

    def test_exact_shadow_rejects_upper_bound_only_alias_and_wrong_dtype(self):
        cm = common()
        self.assertEqual(tail.exact_cpu_length(cm), 30)
        cm._seq_lens_cpu = None
        self.assertIsNone(tail.exact_cpu_length(cm))
        cm._seq_lens_cpu = cm.seq_lens_cpu_upper_bound
        self.assertIsNone(tail.exact_cpu_length(cm))
        cm._seq_lens_cpu = torch.tensor([30], dtype=torch.int64)
        self.assertIsNone(tail.exact_cpu_length(cm))
        cm._seq_lens_cpu = torch.tensor([30, 31], dtype=torch.int32)
        self.assertIsNone(tail.exact_cpu_length(cm))

    def test_cpu_shadow_updates_preserve_original_alias_order_and_none(self):
        cm = common(127)
        cm._num_computed_tokens_cpu = None
        cm.seq_lens_cpu_upper_bound = cm._seq_lens_cpu
        tail._cpu_shadow_advance(cm, 128)
        self.assertEqual(cm.max_seq_len, 128)
        self.assertEqual(cm._seq_lens_cpu.item(), 131)
        self.assertIsNone(cm._num_computed_tokens_cpu)

    def test_replay_waits_for_first_native_plan_and_updates_each_shadow(self):
        trace = []
        candidate = runtime_candidate(trace)
        cm = common(30)
        ready = object()
        metadata = {"draft": NS(plan_ready_event=ready)}
        stream = NS(wait_event=lambda event: trace.append(("wait", event)))
        with patch.object(tail, "plan_signature", return_value="plan"), \
             patch.object(torch.cuda, "current_stream", return_value=stream):
            result = candidate.maybe_run(cm, metadata, 30, 29, NS(all_greedy=True))
        self.assertIs(result, candidate.tokens)
        self.assertEqual(trace, [("wait", ready), "replay"])
        self.assertEqual(cm.seq_lens.item(), 32)
        self.assertEqual(cm._seq_lens_cpu.item(), 32)
        self.assertEqual(cm._num_computed_tokens_cpu.item(), 31)
        self.assertEqual(cm.seq_lens_cpu_upper_bound.item(), 34)
        self.assertEqual(cm.max_seq_len, 32)

    def test_boundary_plan_dtype_shape_or_admission_mismatch_never_replays(self):
        for failure in ("boundary", "plan", "dtype", "shape", "not-admitted", "non-greedy"):
            trace = []
            candidate = runtime_candidate(trace)
            cm = common(30)
            length, greedy, signature = 30, True, "plan"
            if failure == "boundary":
                length = 31
            elif failure == "plan":
                signature = "different"
            elif failure == "dtype":
                cm.seq_lens = cm.seq_lens.long()
            elif failure == "shape":
                cm.block_table_tensor = cm.block_table_tensor[:, :4]
            elif failure == "not-admitted":
                candidate.admitted = False
            else:
                greedy = False
            with patch.object(tail, "plan_signature", return_value=signature):
                self.assertIsNone(candidate.maybe_run(cm, {}, length, length - 1,
                                                      NS(all_greedy=greedy)), failure)
            self.assertFalse(trace, failure)
            self.assertEqual(cm.seq_lens.item(), 30)

    def test_input_address_change_and_position_cap_fall_back(self):
        trace = []
        candidate = runtime_candidate(trace)
        candidate.p.hidden_states = candidate.p.hidden_states.clone()
        with patch.object(tail, "plan_signature", return_value="plan"):
            self.assertIsNone(candidate.maybe_run(common(), {}, 30, 29, NS(all_greedy=True)))
        candidate = runtime_candidate(trace)
        with patch.object(tail, "plan_signature", return_value="plan"):
            self.assertIsNone(candidate.maybe_run(common(), {}, 30, 127, NS(all_greedy=True)))
        self.assertFalse(trace)

    def test_top_level_requires_current_gate_exact_length_and_consumes_views(self):
        trace = []
        candidate = runtime_candidate(trace)
        p = candidate.p
        gate = torch.tensor([0, 1, 1], dtype=torch.int32)
        candidate.gate_signature = tail.tensor_signature(gate)
        p._nemo_tail = [candidate]
        stream = NS(wait_event=lambda event: None)
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=False), \
             patch.object(torch.cuda, "current_stream", return_value=stream), \
             patch.object(tail, "plan_signature", return_value="plan"):
            self.assertIsNone(tail.maybe_run(p, common(), {}, 1, 1, NS(all_greedy=True), None))
            self.assertIsNone(tail.maybe_run(p, common(), {}, 1, 1, NS(all_greedy=True), gate.clone()))
            result = tail.maybe_run(p, common(), {}, 1, 1, NS(all_greedy=True), gate)
        self.assertEqual([r.item() for r in result], [5, 7, 11])
        self.assertEqual(len({r.data_ptr() for r in result}), 3)
        proposal = torch.stack(result, dim=1)
        candidate.tokens.fill_(9)
        self.assertEqual(proposal.tolist(), [[5, 7, 11]])

    def test_admission_checks_every_mutable_key_and_always_reinstalls_fixture(self):
        keys = {"tokens", "hidden_steps", "gate", "kv_pages", "kv_sentinels", "positions",
                "seq_lens", "slot_mapping", "input_ids", "hidden_states", "last_page",
                "cpu_shadows", "max_seq_len"}
        for bad_key in (None, *sorted(keys)):
            candidate = runtime_candidate([])
            installed = []
            phase = {"actual": False}
            def install():
                installed.append(1)
                phase["actual"] = False
                return common(), {}
            def replay():
                phase["actual"] = True
            candidate.graph = NS(replay=replay)
            def observe(tokens, hidden):
                return {k: torch.tensor([1 if phase["actual"] and k == bad_key else 0]) for k in keys}
            with patch.object(tail, "plan_signature", return_value="plan"), \
                 patch.object(torch.cuda, "synchronize", lambda *a: None):
                ok = candidate.admit_private_case("case", install,
                    lambda: (candidate.tokens, candidate.hidden_steps), observe)
            self.assertEqual(ok, bad_key is None, bad_key)
            self.assertEqual(len(installed), 3)
            self.assertFalse(candidate.admitted)
            self.assertEqual(candidate.seal_admission(("case",)), bad_key is None)
            self.assertFalse(candidate.seal_admission(("case", "missing")))

    def test_capture_exception_cannot_publish_and_installer_runs_finally(self):
        candidate = runtime_candidate([])
        installed = []
        candidate._body = lambda: (_ for _ in ()).throw(RuntimeError("private kernel failed"))
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=False), \
             patch.object(torch.cuda, "synchronize", lambda *a: None), \
             patch.object(tail, "plan_signature", return_value="plan"):
            def install():
                installed.append(1)
                return common(), {}
            with self.assertRaisesRegex(RuntimeError, "private kernel failed"):
                candidate.capture_private(install)
        self.assertIsNone(candidate.graph)
        self.assertFalse(candidate.admitted)
        self.assertEqual(len(installed), 2)

    def test_save_restore_handles_aliases_and_private_page_sentinels(self):
        pages = torch.arange(64, dtype=torch.float32).reshape(4, 16)
        initial = pages.clone()
        gate = torch.tensor([0, 1, 1], dtype=torch.int32)
        snapshots = tail._save([pages[:4], pages[:4], gate])
        self.assertEqual(len(snapshots), 2)
        pages[:2].fill_(99)
        gate.fill_(0)
        tail._restore(snapshots)
        self.assertTrue(torch.equal(pages, initial))
        self.assertEqual(gate.tolist(), [0, 1, 1])


if __name__ == "__main__":
    unittest.main()

"""CPU control-flow/admission tests; no CUDA numerical or timing claim.

The production helper is imported directly. A CPU tensor subclass advertises
the CUDA eligibility bit, while a graph double runs a small deterministic model
that writes KV pages. This exercises restoration, admission failure and dispatch
without importing vLLM's unavailable native extensions.
"""
from __future__ import annotations

from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, replace
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch


PATH = Path(__file__).resolve().parents[2] / "vllm/vllm/v1/spec_decode/nemotron_first_pass.py"
_spec = importlib.util.spec_from_file_location("first_pass_under_test", PATH)
helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helper)


class CudaTensor(torch.Tensor):
    @property
    def is_cuda(self):
        return True


def cuda_tensor(value):
    return value.as_subclass(CudaTensor)


@dataclass
class FIPrefill:
    wrapper: object


@dataclass
class FlashInferMetadata:
    prefill: object
    slot_mapping: object
    num_actual_tokens: int = 5
    num_prefills: int = 1
    num_prefill_tokens: int = 5
    num_decodes: int = 0
    num_decode_tokens: int = 0
    decode: object = None
    use_cascade: bool = False
    causal: bool = True
    q_data_type_prefill: object = torch.bfloat16


@dataclass(frozen=True)
class BatchDescriptor:
    num_tokens: int


class Graph:
    fail = False
    corrupt = False
    corrupt_kv = False

    def __init__(self, model, config, mode, options):
        self.model = model
        self.concrete_cudagraph_entries = {}
        self.calls = 0

    def __call__(self, **kwargs):
        if self.fail:
            raise RuntimeError("synthetic capture failure")
        self.concrete_cudagraph_entries[BatchDescriptor(5)] = object()
        self.calls += 1
        result = self.model(**kwargs)
        if self.corrupt:
            result = result.clone()
            result[0, 0] += 1
        if self.corrupt_kv:
            cache = self.model.p.compilation_config.static_forward_context["draft"].kv_cache
            cache[0, 0, 8, 0] += 1
        return result


class NemotronHMTP:
    def __init__(self, proposer):
        self.p = proposer
        self.rows = torch.arange(8, dtype=torch.long)

    def _moe_row_src_buffers(self):
        return [self.rows]

    def set_moe_row_src(self, count, source):
        self.rows[:count].copy_(source.expand(count))

    def __call__(self, input_ids, positions, hidden_states, inputs_embeds):
        cache = self.p.compilation_config.static_forward_context["draft"].kv_cache
        for i, slot in enumerate(self.p._slot_mapping_buffer[:5].tolist()):
            cache[slot // 16, 0, slot % 16].copy_(hidden_states[i] + input_ids[i])
        return (hidden_states + hidden_states.index_select(0, self.rows[:5]) * 0.25
                + cache[0, 0, 0].as_subclass(torch.Tensor))


class FlashInferMetadataBuilder:
    def __init__(self):
        self._spec_prefill_cudagraph = True
        self._spec_q_len = 5
        self.use_dcp = self.use_trtllm_decode_attention = self.is_kvcache_nvfp4 = False
        for name, size in (("paged_kv_indptr", 2), ("paged_kv_indices", 8),
                           ("paged_kv_last_page_len", 1)):
            setattr(self, name, NS(cpu=torch.ones(size, dtype=torch.int32),
                                  gpu=cuda_tensor(torch.ones(size, dtype=torch.int32))))
        self.paged_kv_indptr_cpu_buffer = torch.ones(2, dtype=torch.int32)
        w = NS(_vllm_spec_cg=True, is_cuda_graph_enabled=True, _backend="fa2",
               _cached_module=object(), _plan_info=[0, 1], _kv_layout="HND")
        for name in ("_float_workspace_buffer", "_int_workspace_buffer", "_qo_indptr_buf",
                     "_paged_kv_indptr_buf", "_paged_kv_indices_buf",
                     "_paged_kv_last_page_len_buf", "_kv_lens_buffer"):
            setattr(w, name, cuda_tensor(torch.ones(8, dtype=torch.int32)))
        w._pin_memory_int_workspace_buffer = torch.ones(8, dtype=torch.int32)
        self.wrapper = w
        self._decode_wrappers_cudagraph = {}
        self._spec_prefill_wrappers_cudagraph = {1: w}

    def build_for_drafting(self, private, draft_index):
        length = int(private.seq_lens[0])
        self.wrapper._plan_info = [0, length // 16]
        self.paged_kv_indptr.cpu.fill_(length)
        self.paged_kv_indptr.gpu.fill_(length)
        self.wrapper._pin_memory_int_workspace_buffer.fill_(length)
        return FlashInferMetadata(FIPrefill(self.wrapper), private.slot_mapping)


def fixture():
    builder = FlashInferMetadataBuilder()
    config = NS(parallel_config=NS(tensor_parallel_size=1, data_parallel_size=1,
                                   pipeline_parallel_size=1), lora_config=None)
    p = NS(vllm_config=config, method="mtp",
           speculative_config=NS(num_speculative_tokens=4, enforce_eager=False),
           parallel_drafting=False, pass_hidden_states_to_model=True,
           supports_mm_inputs=False, use_heterogeneous_vocab=False, uses_mrope=False,
           uses_xdrope_dim=0, eplb_state=None, _share_mtp_indices=False,
           draft_attn_groups=[NS(get_metadata_builder=lambda: builder)],
           _draft_attn_layer_names={"draft"}, block_size=16, max_model_len=128,
           device=torch.device("cpu"),
           input_ids=torch.arange(8, dtype=torch.int32),
           positions=torch.arange(8, dtype=torch.long),
           hidden_states=torch.arange(32, dtype=torch.float32).reshape(8, 4) / 8,
           _slot_mapping_buffer=torch.arange(8, dtype=torch.long))
    # Extra physical pages are sentinels: admission may touch only pages 0/1.
    cache = cuda_tensor(torch.arange(256, dtype=torch.float32).reshape(4, 1, 16, 4))
    p.compilation_config = NS(static_forward_context={"draft": NS(kv_cache=cache)})
    p.model = NemotronHMTP(p)
    p._determine_batch_execution_and_padding = lambda n: ("piecewise", n, None)
    p._get_positions = lambda n: p.positions[:n]
    p._get_slot_mapping = lambda n: {"draft": p._slot_mapping_buffer[:n]}
    p.gate = torch.tensor([0, 3, 0], dtype=torch.int32)
    return p, builder


def modules(p):
    leaves = {
        "vllm.compilation.cuda_graph": dict(CUDAGraphOptions=lambda **kw: NS(**kw),
                                             CUDAGraphWrapper=Graph),
        "vllm.config": dict(CUDAGraphMode=NS(FULL="full")),
        "vllm.forward_context": dict(BatchDescriptor=BatchDescriptor,
                                     set_forward_context=lambda *a, **kw: nullcontext()),
        "vllm.model_executor.models.nemotron_h_mtp_kernels": dict(depth_gate_init=lambda d: p.gate),
        "vllm.v1.attention.backend": dict(CommonAttentionMetadata=lambda **kw: NS(**kw)),
        "vllm.v1.attention.backends.flashinfer": dict(FIPrefill=FIPrefill,
                                                     FlashInferMetadata=FlashInferMetadata),
    }
    result = {}
    for name in leaves:
        parts = name.split(".")
        for index in range(1, len(parts) + 1):
            key = ".".join(parts[:index])
            if key not in result:
                result[key] = ModuleType(key)
                result[key].__path__ = []
            if index > 1:
                setattr(result[".".join(parts[:index - 1])], parts[index - 1], result[key])
    for name, attrs in leaves.items():
        result[name].__dict__.update(attrs)
    return result


class FirstPassGraph(unittest.TestCase):
    def environment(self, p):
        stack = ExitStack()
        stack.enter_context(patch.dict(sys.modules, modules(p)))
        stack.enter_context(patch.dict("os.environ", {"NEMO_MTP_FIRST_PASS": "1"}))
        stack.enter_context(patch.object(torch.cuda, "synchronize", lambda *a: None))
        stack.enter_context(patch.object(torch.cuda, "is_current_stream_capturing", lambda: False))
        stack.enter_context(patch.object(torch.cuda, "graph_pool_handle", lambda: object()))
        stack.enter_context(patch.object(Graph, "fail", False))
        stack.enter_context(patch.object(Graph, "corrupt", False))
        stack.enter_context(patch.object(Graph, "corrupt_kv", False))
        return stack

    def snapshots(self, p, b):
        return helper._save([p.input_ids, p.positions, p.hidden_states,
                             p._slot_mapping_buffer, p.model.rows, p.gate,
                             p.compilation_config.static_forward_context["draft"].kv_cache,
                             b.paged_kv_indptr.cpu, b.paged_kv_indptr.gpu,
                             b.wrapper._pin_memory_int_workspace_buffer])

    def assert_restored(self, snapshots):
        for actual, expected in snapshots:
            self.assertTrue(torch.equal(actual, expected))

    def test_admission_compares_all_rows_kv_and_restores_startup_buffers(self):
        p, b = fixture()
        saved = self.snapshots(p, b)
        with self.environment(p):
            helper.prepare(p)
            self.assertTrue(p._nemo_first)
            self.assertLessEqual(len(p._nemo_first), 4)
            self.assertTrue(all(e["graph"].calls >= 6 for e in p._nemo_first))
            self.assert_restored(saved)
            original_bank = p._nemo_first
            helper.prepare(p)
            self.assertIs(p._nemo_first, original_bank)

    def test_corrupt_graph_is_not_published_and_kv_inputs_are_restored(self):
        p, b = fixture()
        saved = self.snapshots(p, b)
        with self.environment(p), patch.object(Graph, "corrupt", True):
            helper.prepare(p)
            self.assertIsNone(p._nemo_first)
            self.assert_restored(saved)

    def test_capture_error_preserves_original_path_and_restores_buffers(self):
        p, b = fixture()
        saved = self.snapshots(p, b)
        with self.environment(p), patch.object(Graph, "fail", True):
            with self.assertLogs(helper.logger, level="WARNING"):
                helper.prepare(p)
            self.assertIsNone(p._nemo_first)
            self.assert_restored(saved)

    def test_kv_only_mismatch_rejects_graph_even_when_hidden_outputs_match(self):
        p, b = fixture()
        saved = self.snapshots(p, b)
        with self.environment(p), patch.object(Graph, "corrupt_kv", True):
            helper.prepare(p)
            self.assertIsNone(p._nemo_first)
            self.assert_restored(saved)

    def test_exact_signature_replays_changed_inputs_and_misses_fall_back(self):
        p, b = fixture()
        with self.environment(p):
            helper.prepare(p)
            b.wrapper._plan_info = list(p._nemo_first[0]["signature"][0][0][3])
            md = {"draft": FlashInferMetadata(FIPrefill(b.wrapper), p._slot_mapping_buffer[:5])}
            kwargs = dict(input_ids=p.input_ids[:5], positions=p.positions[:5],
                          hidden_states=p.hidden_states[:5], inputs_embeds=None)
            p.model.rows[:5].fill_(0)
            p.hidden_states[:5].fill_(0.25)
            first = helper.maybe_run(p, kwargs, md, 5, 1).clone()
            p.hidden_states[:5].fill_(0.5)
            second = helper.maybe_run(p, kwargs, md, 5, 1)
            self.assertFalse(torch.equal(first, second))
            self.assertIsNone(helper.maybe_run(p, kwargs, md, 1, 1))
            self.assertIsNone(helper.maybe_run(p, kwargs, md, 5, 2))
            changed = dict(kwargs, hidden_states=kwargs["hidden_states"].clone())
            self.assertIsNone(helper.maybe_run(p, changed, md, 5, 1))
            for field, value in (("num_actual_tokens", 4), ("num_prefills", 2),
                                 ("num_decodes", 1), ("causal", False)):
                bad = {"draft": replace(md["draft"], **{field: value})}
                self.assertIsNone(helper.maybe_run(p, kwargs, bad, 5, 1))
            b.wrapper._jit_module = object()
            self.assertIsNone(helper.maybe_run(p, kwargs, md, 5, 1))
            b.wrapper._jit_module = None
            b.wrapper._plan_info = [999]
            self.assertIsNone(helper.maybe_run(p, kwargs, md, 5, 1))

    def test_unsupported_models_and_configuration_do_not_prepare(self):
        p, _ = fixture()
        with self.environment(p):
            for name, value in (("parallel_drafting", True), ("supports_mm_inputs", True),
                                ("_share_mtp_indices", True), ("uses_mrope", True)):
                old = getattr(p, name)
                setattr(p, name, value)
                self.assertFalse(helper._eligible(p))
                setattr(p, name, old)
            p.speculative_config.num_speculative_tokens = 3
            self.assertFalse(helper._eligible(p))
            helper.prepare(p)
            self.assertFalse(hasattr(p, "_nemo_first_attempted"))


if __name__ == "__main__":
    unittest.main()

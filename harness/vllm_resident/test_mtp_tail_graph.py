"""CPU execution/ownership tests, not CUDA numerics or GPU graph validation."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "vllm/vllm/v1/spec_decode/mtp_tail_graph.py"
PROPOSER = ROOT / "vllm/vllm/v1/spec_decode/llm_base_proposer.py"


class Tensor:
    def __init__(self, data, cuda, dtype="bfloat16", device="cuda:0"):
        self.data = np.array(data, copy=True)
        self.cuda, self.dtype, self.device = cuda, dtype, device
        self.is_cuda = device.startswith("cuda:")
        self.contiguous = True

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def stride(self):
        return tuple(s // self.data.itemsize for s in self.data.strides)

    def data_ptr(self):
        return self.data.ctypes.data

    def is_contiguous(self):
        return self.contiguous

    def clone(self):
        return Tensor(self.data, self.cuda, self.dtype, self.device)

    def __getitem__(self, key):
        return Tensor(self.data[key], self.cuda, self.dtype, self.device)

    def argmax(self, dim):
        out = Tensor(self.data.argmax(axis=dim), self.cuda, "int64")
        self.cuda.operation(
            lambda: out.data.__setitem__(Ellipsis, self.data.argmax(axis=dim))
        )
        return out


class Stream:
    def __init__(self, cuda):
        self.cuda = cuda

    def wait_stream(self, other):
        self.cuda.waits.append((self, other))


class Graph:
    def __init__(self, cuda):
        self.cuda, self.steps, self.replays = cuda, [], 0

    def replay(self):
        self.replays += 1
        for operation in self.steps:
            operation()


class Cuda:
    def __init__(self):
        self.waits, self.graphs = [], []
        self.current = Stream(self)
        self.active, self.outer, self.fail = None, False, False

    def operation(self, fn):
        fn()
        if self.active is not None:
            self.active.steps.append(fn)

    def current_stream(self, device):
        return self.current

    def is_current_stream_capturing(self):
        return self.outer or self.active is not None

    def Stream(self, device):
        return Stream(self)

    def CUDAGraph(self):
        graph = Graph(self)
        self.graphs.append(graph)
        return graph

    @contextmanager
    def device(self, device):
        yield

    @contextmanager
    def stream(self, stream):
        previous = self.current
        self.current = stream
        try:
            yield
        finally:
            self.current = previous

    @contextmanager
    def graph(self, graph, stream):
        if self.fail:
            raise RuntimeError("capture failure")
        with self.stream(stream):
            self.active = graph
            try:
                yield
            finally:
                self.active = None


class FlashInferMetadata(NS):
    pass


class FIDecode(NS):
    pass


class Model:
    def __init__(self, cuda, wrapper):
        self.cuda, self.wrapper, self.calls = cuda, wrapper, 0

    def __call__(self, **kwargs):
        self.calls += 1
        out = Tensor(np.zeros((1, 4)), self.cuda)

        def compute():
            values = kwargs["hidden_states"].data + kwargs["input_ids"].data[:, None]
            shift = int(kwargs["positions"].data[0] + self.wrapper.seq_len) % 4
            out.data[...] = np.roll(values, shift, axis=-1)

        self.cuda.operation(compute)
        return out

    def compute_logits(self, hidden):
        return hidden


class TailGraphTests(unittest.TestCase):
    def setUp(self):
        self.cuda = Cuda()
        self.grad = False
        self.contexts = []

        @contextmanager
        def context(metadata, config, **kwargs):
            self.contexts.append(kwargs)
            yield

        self.context = context
        self.torch = NS(
            Tensor=Tensor,
            int32="int32",
            bfloat16="bfloat16",
            cuda=self.cuda,
            is_grad_enabled=lambda: self.grad,
        )
        tree = ast.parse(SOURCE.read_text())
        tree.body = [
            n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))
        ]
        self.namespace = {
            "torch": self.torch,
            "set_forward_context": context,
            "CUDAGraphMode": NS(NONE="NONE"),
        }
        exec(compile(tree, str(SOURCE), "exec"), self.namespace)
        fake_backend = NS(FIDecode=FIDecode, FlashInferMetadata=FlashInferMetadata)
        self.modules = patch.dict(
            sys.modules,
            {
                "vllm.vllm.v1.attention.backends.flashinfer": fake_backend,
                "vllm.v1.attention.backends.flashinfer": fake_backend,
            },
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.helper = self.namespace["MTPTailGraph"]()
        self.wrapper = NS(is_cuda_graph_enabled=True, seq_len=31)
        self.model = Model(self.cuda, self.wrapper)
        self.config = NS()
        self.kwargs = {
            "input_ids": Tensor([2], self.cuda, "int32"),
            "hidden_states": Tensor([[0, 4, 1, 2]], self.cuda),
            "positions": Tensor([30], self.cuda, "int64"),
            "inputs_embeds": None,
        }
        slot = Tensor([30], self.cuda, "int64")
        self.slot_mapping = {"layer": slot}
        self.md = FlashInferMetadata(
            num_actual_tokens=1,
            num_decodes=1,
            num_decode_tokens=1,
            num_prefills=0,
            num_prefill_tokens=0,
            prefill=None,
            use_cascade=False,
            causal=True,
            decode=FIDecode(wrapper=self.wrapper),
            slot_mapping=slot,
            q_data_type_decode="bfloat16",
            q_data_type_prefill="float8",
        )
        self.metadata = {"layer": self.md}

    def run_graph(self, index=1):
        return self.helper.run(
            self.model,
            self.config,
            index,
            self.kwargs,
            self.metadata,
            self.slot_mapping,
        )

    def test_replays_refresh_inputs_plan_and_request_without_token_alias(self):
        first = self.run_graph()[1]
        expected_first = first.data.copy()
        for length in (32, 33, 127, 128, 129, 4095, 4096):
            self.wrapper.seq_len = length
            self.kwargs["positions"].data[0] = length - 1
            self.kwargs["hidden_states"].data[...] = [[6, 0, 2, 1]]
            _, token = self.run_graph()
            expected = np.roll([6, 0, 2, 1], (2 * length - 1) % 4).argmax()
            self.assertEqual(token.data[0], expected)
            np.testing.assert_array_equal(first.data, expected_first)
        self.assertEqual(self.model.calls, 4)
        self.assertEqual(len(self.cuda.graphs), 1)
        self.assertEqual(self.cuda.graphs[0].replays, 8)
        self.assertEqual(len(self.cuda.waits), 3)
        self.assertTrue(
            all(c["cudagraph_runtime_mode"] == "NONE" for c in self.contexts)
        )

    def test_depths_one_to_six_have_bounded_stage_storage_and_ties(self):
        self.kwargs["hidden_states"].data[...] = [[3, 3, 3, 3]]
        for depth in range(1, 7):
            for index in range(1, depth):
                self.assertEqual(self.run_graph(index)[1].data[0], 0)
        self.assertEqual(len(self.helper._entries), 5)
        self.assertIsNone(self.run_graph(0))
        self.assertIsNone(self.run_graph(6))

    def test_signature_and_stream_change_fall_back_without_new_storage(self):
        self.run_graph()
        old_ids = self.kwargs["input_ids"]
        self.kwargs["input_ids"] = old_ids.clone()
        self.assertIsNone(self.run_graph())
        self.kwargs["input_ids"] = old_ids
        self.md.decode.wrapper = NS(is_cuda_graph_enabled=True, seq_len=4)
        self.assertIsNone(self.run_graph())
        self.md.decode.wrapper = self.wrapper
        self.cuda.current = Stream(self.cuda)
        self.assertIsNone(self.run_graph())
        self.assertEqual(len(self.helper._entries), 1)
        self.assertEqual(self.cuda.graphs[0].replays, 1)

    def test_capture_failure_is_not_published(self):
        self.cuda.fail = True
        with self.assertRaisesRegex(RuntimeError, "capture failure"):
            self.run_graph()
        self.assertFalse(self.helper._entries)

    def test_grad_nested_capture_and_unsupported_tensor_fall_back(self):
        self.grad = True
        self.assertIsNone(self.run_graph())
        self.grad = False
        self.cuda.outer = True
        self.assertIsNone(self.run_graph())
        self.cuda.outer = False
        for key, attr, value in [
            ("input_ids", "dtype", "int64"),
            ("hidden_states", "dtype", "float32"),
            ("positions", "contiguous", False),
            ("hidden_states", "is_cuda", False),
            ("input_ids", "device", "cuda:1"),
        ]:
            tensor = self.kwargs[key]
            old = getattr(tensor, attr)
            setattr(tensor, attr, value)
            self.assertIsNone(self.run_graph())
            setattr(tensor, attr, old)
        self.assertFalse(self.helper._entries)

    def test_non_native_prefill_cascade_shape_and_disabled_wrapper_fall_back(self):
        for key, value in [
            ("num_actual_tokens", 2),
            ("num_decodes", 2),
            ("num_prefills", 1),
            ("causal", False),
            ("use_cascade", True),
            ("prefill", object()),
            ("decode", NS(wrapper=self.wrapper)),
        ]:
            old = getattr(self.md, key)
            setattr(self.md, key, value)
            self.assertIsNone(self.run_graph())
            setattr(self.md, key, old)
        self.wrapper.is_cuda_graph_enabled = False
        self.assertIsNone(self.run_graph())
        self.wrapper.is_cuda_graph_enabled = True
        self.metadata["extra"] = self.md
        self.assertIsNone(self.run_graph())
        self.assertFalse(self.helper._entries)

    def test_real_config_rewrite_and_all_enable_guards(self):
        config_tree = ast.parse((ROOT / "vllm/vllm/config/speculative.py").read_text())
        rewrite = next(
            n
            for n in ast.walk(config_tree)
            if isinstance(n, ast.If)
            and any(
                isinstance(x, ast.Assign)
                and isinstance(x.value, ast.Constant)
                and x.value.value == "nemotron_h_mtp"
                for x in ast.walk(n)
            )
        )
        hf = NS(model_type="nemotron_h", num_nextn_predict_layers=1)
        exec(
            compile(ast.Module(body=[rewrite], type_ignores=[]), "<rewrite>", "exec"),
            {"hf_config": hf},
        )
        self.assertEqual(hf.model_type, "nemotron_h_mtp")

        def proposer():
            return NS(
                method="mtp",
                draft_model_config=NS(hf_config=hf),
                speculative_config=NS(
                    draft_sample_method="greedy", enforce_eager=False
                ),
                vllm_config=NS(
                    model_config=NS(enforce_eager=False),
                    parallel_config=NS(
                        tensor_parallel_size=1,
                        pipeline_parallel_size=1,
                        data_parallel_size=1,
                    ),
                    offload_config=NS(
                        uva=NS(cpu_offload_gb=0), prefetch=NS(offload_group_size=0)
                    ),
                ),
                parallel_drafting=False,
                supports_mm_inputs=False,
                use_heterogeneous_vocab=False,
                use_local_argmax_reduction=False,
                uses_mrope=False,
                draft_uses_xdrope_dim=0,
                eplb_state=None,
                pass_hidden_states_to_model=True,
                model_returns_tuple=lambda: False,
            )

        enabled = self.namespace["tail_graph_enabled"]
        self.assertTrue(enabled(proposer()))
        for path, value in [
            ("method", "eagle"),
            ("speculative_config.draft_sample_method", "probabilistic"),
            ("speculative_config.enforce_eager", True),
            ("vllm_config.model_config.enforce_eager", True),
            ("vllm_config.parallel_config.tensor_parallel_size", 2),
            ("vllm_config.parallel_config.pipeline_parallel_size", 2),
            ("vllm_config.parallel_config.data_parallel_size", 2),
            ("vllm_config.offload_config.uva.cpu_offload_gb", 1),
            ("vllm_config.offload_config.prefetch.offload_group_size", 1),
            ("parallel_drafting", True),
            ("supports_mm_inputs", True),
            ("use_heterogeneous_vocab", True),
            ("use_local_argmax_reduction", True),
            ("uses_mrope", True),
            ("draft_uses_xdrope_dim", 1),
            ("eplb_state", object()),
            ("pass_hidden_states_to_model", False),
            ("model_returns_tuple", lambda: True),
        ]:
            p = proposer()
            owner = p
            keys = path.split(".")
            for key in keys[:-1]:
                owner = getattr(owner, key)
            setattr(owner, keys[-1], value)
            self.assertFalse(enabled(p), path)
        # Confirm the configuration fields used by the gate exist in source.
        offload = (ROOT / "vllm/vllm/config/offload.py").read_text()
        for field in ("cpu_offload_gb:", "offload_group_size:", "uva:", "prefetch:"):
            self.assertIn(field, offload)

    def test_proposer_tail_routes_graph_and_original_fallback(self):
        proposer_tree = ast.parse(PROPOSER.read_text())
        propose = next(
            n
            for n in ast.walk(proposer_tree)
            if isinstance(n, ast.FunctionDef) and n.name == "propose"
        )
        loop = next(
            n
            for n in ast.walk(propose)
            if isinstance(n, ast.For)
            and isinstance(n.target, ast.Name)
            and n.target.id == "token_index"
        )
        start = next(
            i
            for i, n in enumerate(loop.body)
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "tail_slot_mapping"
        )
        code = compile(
            ast.Module(body=loop.body[start:], type_ignores=[]),
            "<tail integration>",
            "exec",
        )
        for mode, rows, graph_accept in [
            ("PIECEWISE", 1, True),
            ("PIECEWISE", 1, False),
            ("NONE", 1, False),
            ("PIECEWISE", 2, False),
        ]:
            calls = []
            output, token = (
                Tensor([[1, 4, 2, 0]], self.cuda),
                Tensor([1], self.cuda, "int64"),
            )

            class DummyModel:
                def __call__(self, **kwargs):
                    calls.append("model")
                    return output

            def sample(*args):
                calls.append("sample")
                return token, None

            def graph_run(*args):
                calls.append("graph")
                return (output, token) if graph_accept else None

            p = NS(
                _get_slot_mapping=lambda n: self.slot_mapping,
                model=DummyModel(),
                vllm_config=self.config,
                model_returns_tuple=lambda: False,
                _sample_draft_tokens=sample,
                _mtp_tail_graph=NS(run=graph_run),
            )
            namespace = dict(
                self.namespace,
                self=p,
                batch_size=rows,
                input_batch_size=rows,
                cudagraph_runtime_mode=mode,
                tail_graph_enabled=lambda p: True,
                model_kwargs=self.kwargs,
                per_layer_attn_metadata=self.metadata,
                batch_size_across_dp=None,
                token_index=0,
                sampling_metadata=None,
                draft_probs_list=None,
                draft_token_ids_list=[],
            )
            exec(code, namespace)
            self.assertEqual(len(namespace["draft_token_ids_list"]), 1)
            self.assertEqual("model" in calls, not graph_accept)
            self.assertEqual("sample" in calls, not graph_accept)
            if mode == "NONE" or rows != 1:
                self.assertNotIn("graph", calls)


if __name__ == "__main__":
    unittest.main()

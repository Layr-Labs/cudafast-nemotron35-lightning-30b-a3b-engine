"""CPU control-flow tests; these do not run or emulate CUDA kernel numerics."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "vllm/vllm/v1/spec_decode/greedy_head_graph.py"
PROPOSER = ROOT / "vllm/vllm/v1/spec_decode/llm_base_proposer.py"
CONFIG = ROOT / "vllm/vllm/config/speculative.py"


class Tensor:
    def __init__(self, data, cuda, dtype="bfloat16", device="cuda:0", contiguous=True):
        self.data = np.array(data, copy=True)
        self.cuda = cuda
        self.dtype = dtype
        self.device = device
        self.is_cuda = device.startswith("cuda:")
        self.contiguous = contiguous

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def is_contiguous(self):
        return self.contiguous

    def copy_(self, other):
        self.cuda.events.append("input-copy")
        self.data[...] = other.data
        return self

    def clone(self):
        self.cuda.events.append("output-clone")
        return Tensor(self.data, self.cuda, self.dtype, self.device)

    def argmax(self, dim):
        result = Tensor(self.data.argmax(axis=dim), self.cuda, "int64", self.device)
        if self.cuda.active_graph is not None:
            self.cuda.active_graph.update = lambda: result.data.__setitem__(
                slice(None), self.data.argmax(axis=dim)
            )
        return result


class Stream:
    def __init__(self, cuda):
        self.cuda = cuda

    def wait_stream(self, other):
        self.cuda.events.append(("wait", self, other))


class Graph:
    def __init__(self, cuda):
        self.cuda = cuda
        self.update = None
        self.replays = 0

    def replay(self):
        self.cuda.events.append("replay")
        self.replays += 1
        self.update()


class Cuda:
    def __init__(self):
        self.events = []
        self.current = Stream(self)
        self.active_graph = None
        self.outer_capture = False
        self.fail_capture = False
        self.graphs = []

    def is_current_stream_capturing(self):
        return self.outer_capture or self.active_graph is not None

    def current_stream(self, device):
        return self.current

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
        if self.fail_capture:
            raise RuntimeError("synthetic capture failure")
        with self.stream(stream):
            self.active_graph = graph
            try:
                yield
            finally:
                self.active_graph = None


class GreedyHeadGraphTests(unittest.TestCase):
    def setUp(self):
        self.cuda = Cuda()
        self.grad = False
        self.calls = []
        self.torch = SimpleNamespace(
            cuda=self.cuda,
            Tensor=Tensor,
            float16="float16",
            bfloat16="bfloat16",
            float32="float32",
            is_grad_enabled=lambda: self.grad,
            empty_like=lambda t: Tensor(np.empty_like(t.data), self.cuda, t.dtype, t.device),
        )
        tree = ast.parse(SOURCE.read_text())
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        # Execute the actual helper class with NumPy tensor/control-flow doubles.
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), klass],
            type_ignores=[],
        )
        namespace = {"torch": self.torch}
        exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
        self.klass = namespace["GreedyHeadGraph"]

    def tensor(self, data, **kwargs):
        return Tensor(data, self.cuda, **kwargs)

    def logits(self, hidden):
        self.calls.append((hidden, self.cuda.current))
        return hidden

    def test_replay_overwrites_input_and_keeps_prior_tokens_independent(self):
        helper = self.klass(self.logits)
        first = helper.sample(self.tensor([[1, 5, 2]]))
        second = helper.sample(self.tensor([[9, 1, 2]]))
        np.testing.assert_array_equal(first.data, [1])
        np.testing.assert_array_equal(second.data, [0])
        self.assertIsNot(first, second)
        self.assertFalse(np.shares_memory(first.data, helper._output.data))
        self.assertEqual(len(self.calls), 4)  # Three warm-ups, one capture.
        self.assertEqual(helper._graph.replays, 2)
        self.assertEqual(self.cuda.events.count("input-copy"), 2)
        self.assertEqual(self.cuda.events.count("output-clone"), 2)

    def test_warmup_stream_is_ordered_and_current_stream_restored(self):
        current = self.cuda.current
        helper = self.klass(self.logits)
        helper.sample(self.tensor([[2, 1]]))
        self.assertIs(self.cuda.current, current)
        self.assertIs(helper._stream, current)
        self.assertTrue(all(s is not current for _, s in self.calls))
        waits = [e for e in self.cuda.events if isinstance(e, tuple)]
        self.assertEqual(len(waits), 3)
        self.assertIs(waits[0][2], current)
        self.assertIs(waits[1][1], current)

    def test_depths_ties_and_later_request_do_not_reuse_results(self):
        helper = self.klass(self.logits)
        for depth in range(1, 7):
            tokens = []
            for i in range(depth):
                row = [[3, 3, 1]] if i % 2 else [[0, 1, 4]]
                tokens.append(helper.sample(self.tensor(row)))
            self.assertEqual([int(t.data[0]) for t in tokens], [0 if i % 2 else 2 for i in range(depth)])
        np.testing.assert_array_equal(helper.sample(self.tensor([[8, 2, 0]])).data, [0])
        self.assertEqual(len(self.cuda.graphs), 1)

    def test_unsupported_inputs_remain_eager(self):
        for kwargs, data in [
            ({"device": "cpu"}, [[2, 1]]),
            ({"dtype": "float64"}, [[2, 1]]),
            ({"contiguous": False}, [[2, 1]]),
            ({}, [[2, 1], [1, 2]]),
            ({}, [2, 1]),
        ]:
            with self.subTest(kwargs=kwargs, data=data):
                helper = self.klass(self.logits)
                value = self.tensor(data, **kwargs)
                np.testing.assert_array_equal(helper.sample(value).data, np.argmax(data, axis=-1))
                self.assertIsNone(helper._graph)

    def test_training_and_outer_capture_remain_eager(self):
        helper = self.klass(self.logits)
        self.grad = True
        helper.sample(self.tensor([[1, 2]]))
        self.grad = False
        self.cuda.outer_capture = True
        helper.sample(self.tensor([[1, 2]]))
        self.assertIsNone(helper._graph)
        self.assertEqual(len(self.calls), 2)

    def test_changed_shape_dtype_device_or_stream_does_not_grow_cache(self):
        helper = self.klass(self.logits)
        helper.sample(self.tensor([[1, 3, 2]]))
        graph = helper._graph
        for value in [
            self.tensor([[4, 2]]),
            self.tensor([[4, 2, 1]], dtype="float32"),
            self.tensor([[4, 2, 1]], device="cuda:1"),
        ]:
            np.testing.assert_array_equal(helper.sample(value).data, [0])
        self.cuda.current = Stream(self.cuda)
        np.testing.assert_array_equal(helper.sample(self.tensor([[0, 1, 5]])).data, [2])
        self.assertIs(helper._graph, graph)
        self.assertEqual(graph.replays, 1)
        self.assertEqual(len(self.cuda.graphs), 1)

    def test_capture_failure_is_visible_and_not_published(self):
        helper = self.klass(self.logits)
        self.cuda.fail_capture = True
        with self.assertRaisesRegex(RuntimeError, "capture failure"):
            helper.sample(self.tensor([[1, 2]]))
        self.assertIsNone(helper._graph)
        self.assertIsNone(helper._input)
        self.assertIsNone(helper._output)

    def test_proposer_model_parallelism_and_eager_guards(self):
        tree = ast.parse(PROPOSER.read_text())
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SpecDecodeBaseProposer")
        function = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "_greedy_sample")
        captures = []
        def factory(callback):
            captures.append(callback)
            return SimpleNamespace(sample=lambda h: "graph-result")
        namespace = {"torch": self.torch, "GreedyHeadGraph": factory}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(PROPOSER), "exec"), namespace)
        sample = namespace["_greedy_sample"]
        # Derive the draft type from the real config rewrite, not a hand-made
        # type that could make an unreachable fast-path guard appear to pass.
        config_tree = ast.parse(CONFIG.read_text())
        override = next(n for n in ast.walk(config_tree) if isinstance(n, ast.FunctionDef) and n.name == "hf_config_override")
        rewrite = next(n for n in override.body if isinstance(n, ast.If) and any(
            isinstance(x, ast.Assign) and isinstance(x.value, ast.Constant)
            and x.value.value == "nemotron_h_mtp" for x in ast.walk(n)
        ))
        hf_config = SimpleNamespace(model_type="nemotron_h", num_nextn_predict_layers=1)
        exec(compile(ast.Module(body=[rewrite], type_ignores=[]), str(CONFIG), "exec"), {"hf_config": hf_config})
        self.assertEqual(hf_config.model_type, "nemotron_h_mtp")
        def proposer():
            return SimpleNamespace(
                method="mtp", use_local_argmax_reduction=False,
                use_heterogeneous_vocab=False,
                draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=hf_config.model_type)),
                speculative_config=SimpleNamespace(enforce_eager=False),
                vllm_config=SimpleNamespace(
                    model_config=SimpleNamespace(enforce_eager=False),
                    parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
                ),
                model=SimpleNamespace(compute_logits=self.logits, get_top_tokens=lambda h: "local-result"),
            )
        p = proposer()
        value = self.tensor([[1, 2]])
        self.assertEqual(sample(p, value), "graph-result")
        self.assertEqual(sample(p, value), "graph-result")
        self.assertEqual(len(captures), 1)
        for path, changed in [
            (("method",), "eagle"),
            (("draft_model_config", "hf_config", "model_type"), "other"),
            (("speculative_config", "enforce_eager"), True),
            (("vllm_config", "model_config", "enforce_eager"), True),
            (("vllm_config", "parallel_config", "tensor_parallel_size"), 2),
            (("vllm_config", "parallel_config", "pipeline_parallel_size"), 2),
        ]:
            p = proposer()
            target = p
            for key in path[:-1]:
                target = getattr(target, key)
            setattr(target, path[-1], changed)
            np.testing.assert_array_equal(sample(p, value).data, [1])
            self.assertFalse(hasattr(p, "_greedy_head_graph"))
        p = proposer()
        p.use_local_argmax_reduction = True
        self.assertEqual(sample(p, value), "local-result")
        p = proposer()
        p.use_heterogeneous_vocab = True
        p.vocab_mapping = SimpleNamespace(
            constrain_draft_logits=lambda h: h,
            map_draft_to_target_ids=lambda t: "mapped-result",
        )
        self.assertEqual(sample(p, value), "mapped-result")
        self.assertFalse(hasattr(p, "_greedy_head_graph"))


if __name__ == "__main__":
    unittest.main()

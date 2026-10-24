"""CPU lifetime/order tests, without importing torch or running CUDA."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from harness.vllm_resident.test_small_output_parse import load_parser


ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "vllm/vllm/v1/worker/token_copy_pool.py"
RUNNER = ROOT / "vllm/vllm/v1/worker/gpu_model_runner.py"
INPUT_BATCH = ROOT / "vllm/vllm/v1/worker/gpu_input_batch.py"


class Device:
    def __init__(self, name):
        self.name = name
        self.type = name.split(":")[0]
    def __eq__(self, other):
        return self.name == other.name


class Tensor:
    def __init__(self, values, cuda, dtype=np.int32, device="cuda:0", copy=True):
        self.values = np.array(values, dtype=dtype, copy=True) if copy else values
        self.cuda = cuda
        self.dtype = self.values.dtype
        self.device = Device(device)
        self.is_cuda = self.device.type == "cuda"
        self.ndim = self.values.ndim
        self.shape = self.values.shape
    def __getitem__(self, key):
        return Tensor(self.values[key], self.cuda, device=self.device.name, copy=False)
    def copy_(self, source, non_blocking=False):
        assert non_blocking
        self.cuda.events.append("copy")
        self.cuda.pending.append(lambda: self.values.__setitem__(Ellipsis, source.values))
        return self
    def to(self, device, non_blocking=False):
        result = Tensor(np.zeros_like(self.values), self.cuda, dtype=self.dtype, device=device)
        return result.copy_(self, non_blocking)
    def cpu(self):
        return self
    def numpy(self):
        return self.values
    def tolist(self):
        self.cuda.events.append("tolist")
        return self.values.tolist()


class Event:
    def __init__(self, cuda, blocking):
        assert blocking
        self.cuda = cuda
        self.actions = []
        self.fail = False
    def record(self):
        self.cuda.events.append("record")
        self.actions = list(self.cuda.pending)
        self.cuda.pending.clear()
    def synchronize(self):
        self.cuda.events.append("synchronize")
        if self.fail:
            raise RuntimeError("copy wait failed")
        for action in self.actions:
            action()
        self.actions.clear()


class Cuda:
    def __init__(self):
        self.events = []
        self.pending = []
        self.created = []
    def Event(self, blocking):
        event = Event(self, blocking)
        self.created.append(event)
        return event
    def current_stream(self):
        return self
    def wait_stream(self, stream):
        self.events.append("wait-stream")
    @contextmanager
    def stream(self, stream):
        yield


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


class TokenCopyPoolTests(unittest.TestCase):
    def setUp(self):
        self.cuda = Cuda()
        self.allocations = []
        def empty(shape, device, dtype, pin_memory):
            self.assertEqual(device, "cpu")
            self.assertTrue(pin_memory)
            self.allocations.append((shape, dtype))
            return Tensor(np.full(shape, -999, dtype=dtype), self.cuda, dtype=dtype, device=device)
        self.torch = SimpleNamespace(cuda=self.cuda, empty=empty, int32=np.dtype("int32"), int64=np.dtype("int64"))
        self.ns = {"torch": self.torch, "AsyncModelRunnerOutput": object,
                   "RejectionSampler": SimpleNamespace(parse_output=load_parser(self.torch))}
        definitions(POOL, ["_TokenCopySlot", "TokenCopyPool"], self.ns)
        definitions(RUNNER, ["AsyncGPUModelRunnerOutput"], self.ns)
        self.pool = self.ns["TokenCopyPool"]()

    def tensor(self, values, **kwargs):
        return Tensor(values, self.cuda, **kwargs)

    def output(self, values, **kwargs):
        return self.ns["AsyncGPUModelRunnerOutput"](
            model_runner_output=SimpleNamespace(), sampled_token_ids=self.tensor(values),
            logprobs_tensors=kwargs.pop("logprobs_tensors", None),
            invalid_req_indices=kwargs.pop("invalid_req_indices", []),
            async_output_copy_stream=self.cuda, vocab_size=100,
            token_copy_pool=kwargs.pop("token_copy_pool", self.pool), **kwargs,
        )

    def test_slots_are_bounded_and_never_shared_while_busy(self):
        a = self.pool.acquire(self.tensor([[1, 2]]))
        b = self.pool.acquire(self.tensor([[3]]))
        self.assertIsNot(a, b)
        self.assertIsNone(self.pool.acquire(self.tensor([[4]])))
        a.in_use = False
        self.assertIs(self.pool.acquire(self.tensor([[5] * 7])), a)
        self.assertEqual(len(self.allocations), 2)

    def test_view_widths_and_dtype_device_fallbacks(self):
        for dtype in [np.int32, np.int64]:
            for width in range(1, 8):
                slot = self.pool.acquire(self.tensor([[1] * width], dtype=dtype))
                self.assertEqual(slot.view(width).shape, (1, width))
                self.assertIs(slot.view(width), slot.view(width))
                self.assertEqual(slot.view(width).dtype, np.dtype(dtype))
                slot.in_use = False
        for tensor in [self.tensor([[1]], device="cpu"), self.tensor([[1]], device="cuda:1"),
                       self.tensor([[1]], dtype=np.float32), self.tensor([[]]),
                       self.tensor([[1] * 8]), self.tensor([[1], [2]]), self.tensor([1])]:
            self.assertIsNone(self.pool.acquire(tensor))
        self.assertEqual(len(self.allocations), 2)

    def test_output_waits_for_copy_and_result_survives_reuse(self):
        first = self.output([[3, -1, 7, 100]])
        slot = first._token_copy_slot
        self.assertTrue(slot.in_use)
        self.assertEqual(first.sampled_token_ids_cpu.values[0, 0], -999)
        result = first.get_output()
        self.assertEqual(result.sampled_token_ids, [[3, 7]])
        self.assertFalse(slot.in_use)
        second = self.output([[9, 8, -1]])
        self.assertIs(second._token_copy_slot, slot)
        self.assertEqual(second.get_output().sampled_token_ids, [[9, 8]])
        self.assertEqual(result.sampled_token_ids, [[3, 7]])
        self.assertEqual(len(self.cuda.created), 1)
        self.assertLess(self.cuda.events.index("synchronize"), self.cuda.events.index("tolist"))
        self.assertEqual(self.cuda.events[:3], ["wait-stream", "copy", "record"])

    def test_two_pending_outputs_and_exhaustion_fallback(self):
        a = self.output([[1, -1]])
        b = self.output([[2, -1]])
        c = self.output([[3, -1]])
        self.assertIsNone(c._token_copy_slot)
        self.assertEqual(c.get_output().sampled_token_ids, [[3]])
        self.assertEqual(b.get_output().sampled_token_ids, [[2]])
        self.assertEqual(a.get_output().sampled_token_ids, [[1]])
        self.assertEqual(len(self.allocations), 2)

    def test_serial_and_discard_keep_existing_outputs(self):
        self.assertEqual(self.output([[100]]).get_output().sampled_token_ids, [[100]])
        self.assertEqual(self.output([[9]], invalid_req_indices=[0]).get_output().sampled_token_ids, [[]])
        self.assertEqual(self.output([[9, 8]], invalid_req_indices=[0]).get_output().sampled_token_ids, [[]])

    def test_parse_failure_releases_after_completed_copy(self):
        output = self.output([[2, -1]])
        slot = output._token_copy_slot
        def fail(*args, **kwargs):
            raise ValueError("parse failed")
        self.ns["RejectionSampler"].parse_output = fail
        with self.assertRaisesRegex(ValueError, "parse failed"):
            output.get_output()
        self.assertFalse(slot.in_use)
        self.assertIsNone(output._token_copy_slot)

    def test_failed_wait_and_abandoned_outputs_keep_their_leases(self):
        output = self.output([[2, -1]])
        slot = output._token_copy_slot
        output.async_copy_ready_event.fail = True
        with self.assertRaisesRegex(RuntimeError, "copy wait failed"):
            output.get_output()
        self.assertTrue(slot.in_use)
        self.assertTrue(hasattr(output, "_sampled_token_ids"))
        abandoned = self.output([[5, -1]])
        self.assertIsNone(self.output([[8, -1]])._token_copy_slot)
        self.assertTrue(abandoned._token_copy_slot.in_use)

    def test_logprobs_and_routed_experts_are_preserved(self):
        calls = []
        class Payload:
            def to_cpu_nonblocking(self):
                calls.append("copy-payload")
                return self
            def filter(self, mask):
                calls.append(mask.tolist())
                return self
            def tolists(self, offsets=None):
                return ["payload", offsets]
        output = self.output([[4, -1]], logprobs_tensors=Payload(), routed_experts=Payload())
        result = output.get_output()
        self.assertEqual(result.sampled_token_ids, [[4]])
        self.assertEqual(result.logprobs, ["payload", [0, 1]])
        self.assertEqual(result.routed_experts, ["payload", None])
        self.assertEqual(calls, ["copy-payload", "copy-payload", [True, False]])

    def test_input_batch_second_owner_disables_pool_at_real_callsite(self):
        tree = ast.parse(RUNNER.read_text())
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "AsyncGPUModelRunnerOutput")
        guard = next(k.value for k in call.keywords if k.arg == "token_copy_pool")
        batch_tree = ast.parse(INPUT_BATCH.read_text())
        method = next(n for n in ast.walk(batch_tree) if isinstance(n, ast.FunctionDef) and n.name == "set_async_sampled_token_ids")
        scope = {}
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(INPUT_BATCH), "exec"), scope)
        for ids in [[], [[]], [[7]]]:
            batch = SimpleNamespace(sampling_metadata=SimpleNamespace(output_token_ids=ids))
            runner = SimpleNamespace(input_batch=batch, _token_copy_pool=self.pool)
            selected = eval(compile(ast.Expression(guard), str(RUNNER), "eval"), {"self": runner})
            output = self.output([[2, -1]], token_copy_pool=selected)
            scope["set_async_sampled_token_ids"](batch, output.sampled_token_ids_cpu, output.async_copy_ready_event)
            self.assertEqual(output._token_copy_slot is not None, not bool(ids))
            if ids:
                self.assertIs(batch.sampled_token_ids_cpu, output.sampled_token_ids_cpu)
            else:
                self.assertIsNone(batch.sampled_token_ids_cpu)
            output.get_output()


if __name__ == "__main__":
    unittest.main()

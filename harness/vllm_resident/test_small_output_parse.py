"""CPU differential checks of the actual short-row output parser."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[2] / "vllm/vllm/v1/sample/rejection_sampler.py"


class Tensor:
    def __init__(self, values, dtype=np.int32, device="cpu"):
        self.values = np.asarray(values, dtype=dtype)
        self.dtype = self.values.dtype
        self.shape = self.values.shape
        self.ndim = self.values.ndim
        self.device = SimpleNamespace(type=device)
        self.cpu_calls = self.numpy_calls = self.tolist_calls = 0

    def cpu(self):
        self.cpu_calls += 1
        return self

    def numpy(self):
        self.numpy_calls += 1
        return self.values

    def tolist(self):
        self.tolist_calls += 1
        return self.values.tolist()


def load_parser(torch):
    tree = ast.parse(SOURCE.read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RejectionSampler")
    method = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "parse_output")
    method.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    namespace = {"torch": torch, "PLACEHOLDER_TOKEN_ID": -1}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace["parse_output"]


class SmallOutputParseTests(unittest.TestCase):
    def setUp(self):
        self.parse = load_parser(SimpleNamespace(int32=np.dtype("int32"), int64=np.dtype("int64")))

    def reference(self, values, vocab=100, discard=()):
        mask = (values != -1) & (values < vocab)
        if len(discard):
            mask[discard] = False
        return [row[mask[i]].tolist() for i, row in enumerate(values)]

    def test_widths_dtypes_and_discard_masks_match_reference(self):
        rng = np.random.default_rng(1907)
        for dtype in [np.int32, np.int64]:
            for width in range(8):
                for discard in [(), [], [0], [-1], [0, -1, 0]]:
                    for _ in range(8):
                        values = rng.integers(-3, 103, size=(1, width), dtype=dtype)
                        values[:, :min(width, 2)] = -1
                        tensor = Tensor(values, dtype=dtype)
                        result, logs = self.parse(tensor, 100, discard)
                        self.assertEqual(result, self.reference(values, discard=discard))
                        self.assertIsNone(logs)
                        self.assertEqual(tensor.numpy_calls, 0)
                        self.assertEqual(tensor.cpu_calls, 0)

    def test_exact_predicate_negative_values_and_bounds(self):
        tensor = Tensor([[-2, -1, 0, 99, 100, 101, 9]])
        self.assertEqual(self.parse(tensor, 100), ([[-2, 0, 99, 9]], None))
        self.assertEqual(tensor.tolist_calls, 1)
        np.testing.assert_array_equal(tensor.values, [[-2, -1, 0, 99, 100, 101, 9]])

    def test_invalid_discard_indices_keep_original_errors(self):
        for discard in [[1], [-2], [0.0]]:
            with self.subTest(discard=discard):
                tensor = Tensor([[1, 2, -1]])
                with self.assertRaises(IndexError):
                    self.parse(tensor, 100, discard)
                self.assertEqual(tensor.numpy_calls, 1)

    def test_tuple_and_numpy_indexing_semantics_stay_general(self):
        for discard in [(0, -1), (0,), [True], np.array([0])]:
            tensor = Tensor([[1, 2, -1]])
            expected = self.reference(tensor.values, discard=discard)
            self.assertEqual(self.parse(tensor, 100, discard), (expected, None))
            self.assertEqual(tensor.numpy_calls, 1)

    def test_shapes_dtype_and_device_fall_back(self):
        for values, dtype, device in [
            ([[1, -1], [2, -1]], np.int32, "cpu"),
            ([list(range(8))], np.int32, "cpu"),
            ([[1, -1]], np.float32, "cpu"),
            ([[1, -1]], np.int32, "cuda"),
        ]:
            tensor = Tensor(values, dtype=dtype, device=device)
            self.assertEqual(self.parse(tensor, 100), (self.reference(tensor.values), None))
            self.assertEqual(tensor.numpy_calls, 1)

    def test_logprobs_keep_filter_order_before_discard(self):
        calls = []
        class Logs:
            def filter(self, mask):
                calls.append(mask.tolist())
                return self
            def tolists(self, offsets):
                calls.append(offsets)
                return "logs"
        tensor = Tensor([[2, -1, 7, 100]])
        self.assertEqual(self.parse(tensor, 100, [0], Logs()), ([[]], "logs"))
        self.assertEqual(calls, [[True, False, True, False], [0, 2]])
        self.assertEqual(tensor.numpy_calls, 1)


if __name__ == "__main__":
    unittest.main()

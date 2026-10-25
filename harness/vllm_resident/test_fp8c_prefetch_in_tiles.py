"""CPU checks of input-projection tiled prefetch; no CUDA timing claim."""

import ast
import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
import unittest


ROOT = Path(__file__).resolve().parents[2]
L2PF = ROOT / "harness/kernels/l2pf/py/__init__.py"
MOE = ROOT / "vllm/vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py"
MIXER = ROOT / "vllm/vllm/model_executor/layers/mamba/mamba_mixer2.py"
TILES, HEADER, MAIN, CHUNKS = 322, 256, 3584, 21
TAIL_OFFSET = HEADER + MAIN * CHUNKS


def function(path, name, namespace):
    node = copy.deepcopy(next(
        node for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == name
    ))
    node.decorator_list = []
    node.returns = None
    for arg in (*node.args.args, *node.args.kwonlyargs):
        arg.annotation = None
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[name]


@dataclass(frozen=True)
class Device:
    type: str = "cuda"
    index: int = 0


class Storage:
    def __init__(self, pointer, size):
        self.pointer, self.size = pointer, size

    def data_ptr(self):
        return self.pointer

    def nbytes(self):
        return self.size


class Tensor:
    def __init__(self, tile_bytes=77824):
        self.shape = (TILES, tile_bytes)
        self.ndim = 2
        self.dtype = "uint8"
        self.device = Device()
        self.is_cuda = True
        self.contiguous = True
        self.pointer = 0x100000
        self.strides = (tile_bytes, 1)
        self.storage = Storage(self.pointer, TILES * tile_bytes)
        self._fp8c_prefetch_layout = (
            1, 10304, 2688, HEADER, MAIN, CHUNKS, self.pointer,
        )

    def data_ptr(self):
        return self.pointer

    def is_contiguous(self):
        return self.contiguous

    def untyped_storage(self):
        return self.storage

    def numel(self):
        return self.shape[0] * self.shape[1]

    def element_size(self):
        return 1

    def stride(self, axis=None):
        return self.strides if axis is None else self.strides[axis]


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.ns = dict(torch=NS(uint8="uint8"))
        self.plan = function(L2PF, "_fp8c_in_plan", self.ns)

    def check_ranges(self, tensor, budget, plan):
        tiles, stride, prefix, tail_offset, tail = plan
        self.assertEqual((tiles, stride), tensor.shape)
        self.assertEqual(tail_offset, TAIL_OFFSET)
        self.assertEqual(tail_offset + tail, stride)
        self.assertGreaterEqual(prefix, HEADER + 6 * MAIN)
        self.assertEqual((prefix - HEADER) % MAIN, 0)
        self.assertLessEqual(prefix, tail_offset)
        self.assertLessEqual(tiles * (prefix + tail), budget)
        for tile in range(tiles):
            head_begin = tile * stride
            head_end = head_begin + prefix
            tail_begin = head_begin + tail_offset
            tail_end = tail_begin + tail
            self.assertTrue(0 <= head_begin < head_end <= tail_begin < tail_end)
            self.assertLessEqual(tail_end, tensor.storage.nbytes())
            for start, length in ((head_begin, prefix), (tail_begin, tail)):
                self.assertEqual(start % 16, 0)
                self.assertEqual(length % 16, 0)
                for offset in range(0, length, 65536):
                    size = min(65536, length - offset)
                    self.assertTrue(0 < size <= 65536)
                    self.assertEqual(size % 16, 0)
                    self.assertLessEqual(start + offset + size, start + length)

    def test_all_legal_strides_and_budget_boundaries(self):
        checked = 0
        for stride in range(75648, 79872 + 1, 128):
            tensor = Tensor(stride)
            tail = stride - TAIL_OFFSET
            minimum = TILES * (HEADER + 6 * MAIN + tail)
            self.assertIsNone(self.plan(tensor, minimum - 1))
            for chunks in range(6, CHUNKS + 1):
                budget = TILES * (HEADER + chunks * MAIN + tail)
                with self.subTest(stride=stride, chunks=chunks):
                    plan = self.plan(tensor, budget)
                    self.assertIsNotNone(plan)
                    self.assertEqual(plan[2], HEADER + chunks * MAIN)
                    self.check_ranges(tensor, budget, plan)
                    checked += 1
                    before = self.plan(tensor, budget - 1)
                    if chunks == 6:
                        self.assertIsNone(before)
                    else:
                        self.assertEqual(before[2], HEADER + (chunks - 1) * MAIN)
            over = self.plan(tensor, tensor.numel() * 2)
            self.assertEqual(over[2], TAIL_OFFSET)
            self.check_ranges(tensor, tensor.numel() * 2, over)
        self.assertEqual(checked, 34 * 16)

    def test_existing_twelve_mib_window_covers_initial_pipeline(self):
        budget = 12 << 20
        for stride in range(75648, 79872 + 1, 128):
            tensor = Tensor(stride)
            plan = self.plan(tensor, budget)
            self.assertIsNotNone(plan)
            self.assertIn((plan[2] - HEADER) // MAIN, (9, 10))
            self.check_ranges(tensor, budget, plan)

    def test_rejects_wrong_projection_and_unproven_storage(self):
        mutations = (
            lambda t: setattr(t, "dtype", "float8"),
            lambda t: setattr(t, "is_cuda", False),
            lambda t: setattr(t, "ndim", 1),
            lambda t: setattr(t, "shape", (321, t.shape[1])),
            lambda t: setattr(t, "contiguous", False),
            lambda t: setattr(t, "pointer", t.pointer + 16),
            lambda t: setattr(t.storage, "pointer", t.pointer + 128),
            lambda t: setattr(t.storage, "size", t.storage.size + 128),
            lambda t: delattr(t, "_fp8c_prefetch_layout"),
            lambda t: setattr(t, "_fp8c_prefetch_layout", ()),
            lambda t: setattr(t, "_fp8c_prefetch_layout", (2, 10304, 2688, HEADER, MAIN, CHUNKS, t.pointer)),
            lambda t: setattr(t, "_fp8c_prefetch_layout", (1, 2688, 4096, 384, MAIN, 32, t.pointer)),
            lambda t: setattr(t, "_fp8c_prefetch_layout", (1, 10304, 2688, HEADER, MAIN, CHUNKS, t.pointer + 128)),
        )
        for index, mutate in enumerate(mutations):
            tensor = Tensor()
            mutate(tensor)
            with self.subTest(mutation=index):
                self.assertIsNone(self.plan(tensor, 12 << 20))
        for stride in (75520, 75649, 80000):
            self.assertIsNone(self.plan(Tensor(stride), 12 << 20))
        for budget in (0, -1, 1, 5.5, "12582912"):
            self.assertIsNone(self.plan(Tensor(), budget))
        self.assertIsNone(self.plan(None, 12 << 20))


class DispatchTests(unittest.TestCase):
    def setup_dispatch(self, native=True, enabled=True):
        self.events, self.fallbacks, self.rc = [], [], 0
        main = object()
        side = NS(cuda_stream=8322,
                  wait_stream=lambda stream: self.events.append(("wait", stream)))

        def launch(*args):
            self.events.append(("launch", args))
            return self.rc

        def fallback(tensor, max_bytes=0):
            self.fallbacks.append((tensor, max_bytes))
            return False

        lib = NS(l2pf_prefetch_fp8c_in=launch) if native else NS()
        ns = dict(torch=NS(uint8="uint8", cuda=NS(current_stream=lambda device: main)),
                  _FP8C_IN_TILE_PREFETCH=enabled, ENABLED=True, _lib=lib,
                  CHUNK=65536, _side_stream=lambda device: side, prefetch=fallback)
        function(L2PF, "_fp8c_in_plan", ns)
        return function(L2PF, "prefetch_fp8c_in", ns), main, ns

    def test_native_abi_and_stream_order_preserve_existing_budget(self):
        run, main, _ = self.setup_dispatch()
        tensor = Tensor()
        budget = 12 << 20
        self.assertTrue(run(tensor, max_bytes=budget))
        self.assertEqual(self.events[0], ("wait", main))
        self.assertEqual(self.events[1][0], "launch")
        args = self.events[1][1]
        self.assertEqual(len(args), 8)
        self.assertEqual(args[:2], (tensor.pointer, tensor.shape[1]))
        prefix, tail_offset, tail, tiles, chunk, stream = args[2:]
        self.assertEqual((tail_offset, tiles, chunk, stream), (TAIL_OFFSET, TILES, 65536, 8322))
        self.assertLessEqual(tiles * (prefix + tail), budget)
        self.assertFalse(self.fallbacks)

    def test_old_library_flags_and_unmarked_tensor_keep_exact_fallback(self):
        for native, enabled, bad_marker, global_enabled in (
            (False, True, False, True), (True, False, False, True),
            (True, True, True, True), (True, True, False, False),
        ):
            run, _, ns = self.setup_dispatch(native=native, enabled=enabled)
            ns["ENABLED"] = global_enabled
            tensor = Tensor()
            if bad_marker:
                del tensor._fp8c_prefetch_layout
            self.assertFalse(run(tensor, max_bytes=12 << 20))
            self.assertEqual(self.fallbacks, [(tensor, 12 << 20)])
            self.assertEqual(self.events, [])

    def test_native_error_does_not_trigger_second_launch(self):
        run, _, _ = self.setup_dispatch()
        self.rc = 999
        with self.assertRaises(RuntimeError):
            run(Tensor(), max_bytes=12 << 20)
        self.assertFalse(self.fallbacks)
        self.assertEqual([event[0] for event in self.events], ["wait", "launch"])

    def test_moe_hook_selects_only_actual_compressed_input_weight(self):
        for compressed, small, helper, rows, registered in (
            (True, True, True, 0, True),
            (True, True, True, 1, True), (True, True, True, 5, True),
            (True, True, True, 8, True), (True, True, True, 496, True),
            (False, True, True, 5, True), (False, False, True, 5, True),
            (True, True, False, 5, True), (True, True, True, 5, False),
        ):
            with self.subTest(compressed=compressed, rows=rows, helper=helper,
                              small=small, registered=registered):
                calls = []
                raw, packed, compact = object(), object(), object()
                nxt = NS(weight=raw)
                if small:
                    nxt._fp8small_weight = packed
                if compressed:
                    nxt._fp8c_weight = compact
                pf = NS(MOE_BYTES=12 << 20,
                        AHEAD_MOE={"test": nxt} if registered else {},
                        prefetch=lambda tensor, **kw: calls.append(("ordinary", tensor, kw)) or True)
                if helper:
                    pf.prefetch_fp8c_in = lambda tensor, **kw: calls.append(("tiled", tensor, kw)) or True
                run = function(MOE, "_l2pf_next_layer", dict(_l2pf=pf))
                result = run(NS(layer_name="test"), rows)
                if rows > 8 or not registered:
                    self.assertFalse(result)
                    self.assertEqual(calls, [])
                else:
                    selected = "tiled" if compressed and helper and rows >= 1 else "ordinary"
                    weight = compact if compressed else packed if small else raw
                    self.assertTrue(result)
                    self.assertEqual(calls, [(selected, weight, dict(max_bytes=12 << 20))])

    def test_mamba_output_hook_retains_original_prefetch(self):
        node = next(node for node in ast.walk(ast.parse(MIXER.read_text()))
                    if isinstance(node, ast.FunctionDef) and node.name == "mamba_mixer2")
        calls = [node.func.attr for node in ast.walk(node)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertIn("prefetch_many", calls)
        self.assertNotIn("prefetch_fp8c_out", calls)
        self.assertNotIn("prefetch_fp8c_in", calls)


if __name__ == "__main__":
    unittest.main()

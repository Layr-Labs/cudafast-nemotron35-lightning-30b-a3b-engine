"""CPU validation of fp8c prefetch ranges and dispatch; no GPU timing claim."""

import ast
import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
import unittest


ROOT = Path(__file__).resolve().parents[2]
L2PF = ROOT / "harness/kernels/l2pf/py/__init__.py"
MIXER = ROOT / "vllm/vllm/model_executor/layers/mamba/mamba_mixer2.py"
TILES, HEADER, MAIN, CHUNKS = 84, 384, 3584, 32
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
    def __init__(self, tile_bytes=117760):
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
            1, 2688, 4096, HEADER, MAIN, CHUNKS, self.pointer,
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
        self.plan = function(L2PF, "_fp8c_out_plan", self.ns)

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
                # Every split request used by the existing bulk-prefetch model
                # remains within its declared tile region.
                for offset in range(0, length, 65536):
                    size = min(65536, length - offset)
                    self.assertTrue(0 < size <= 65536)
                    self.assertEqual(size % 16, 0)
                    self.assertLessEqual(start + offset + size, start + length)

    def test_all_legal_strides_and_chunk_budget_boundaries(self):
        checked = 0
        for stride in range(115200, 121856 + 1, 128):
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
        self.assertEqual(checked, 53 * 27)

    def test_actual_window_preserves_router_reservation(self):
        total = 5 << 20
        router = 128 * 2688 * 2
        budget = total - router
        for stride in range(115200, 121856 + 1, 128):
            plan = self.plan(Tensor(stride), budget)
            self.assertIsNotNone(plan)
            self.assertGreaterEqual((plan[2] - HEADER) // MAIN, 13)
            self.assertLessEqual(plan[0] * (plan[2] + plan[4]) + router, total)

    def test_rejects_unproven_layouts_and_invalid_budgets(self):
        mutations = (
            lambda t: setattr(t, "dtype", "float8"),
            lambda t: setattr(t, "is_cuda", False),
            lambda t: setattr(t, "ndim", 1),
            lambda t: setattr(t, "shape", (83, t.shape[1])),
            lambda t: setattr(t, "contiguous", False),
            lambda t: setattr(t, "pointer", t.pointer + 16),
            lambda t: setattr(t.storage, "pointer", t.pointer + 128),
            lambda t: setattr(t.storage, "size", t.storage.size + 128),
            lambda t: delattr(t, "_fp8c_prefetch_layout"),
            lambda t: setattr(t, "_fp8c_prefetch_layout", ()),
            lambda t: setattr(t, "_fp8c_prefetch_layout", (2, 2688, 4096, HEADER, MAIN, CHUNKS, t.pointer)),
            lambda t: setattr(t, "_fp8c_prefetch_layout", (1, 10304, 2688, 256, MAIN, 21, t.pointer)),
            lambda t: setattr(t, "_fp8c_prefetch_layout", (1, 2688, 4096, HEADER, MAIN, CHUNKS, t.pointer + 128)),
        )
        for index, mutate in enumerate(mutations):
            tensor = Tensor()
            mutate(tensor)
            with self.subTest(mutation=index):
                self.assertIsNone(self.plan(tensor, 5 << 20))
        for stride in (115072, 115201, 121984):
            self.assertIsNone(self.plan(Tensor(stride), 5 << 20))
        for budget in (0, -1, 1, 5.5, "5242880"):
            with self.subTest(budget=budget):
                self.assertIsNone(self.plan(Tensor(), budget))
        self.assertIsNone(self.plan(None, 5 << 20))


class DispatchTests(unittest.TestCase):
    def setup_dispatch(self, native=True, enabled=True):
        self.events = []
        self.fallbacks = []
        self.rc = 0
        main = object()
        side = NS(cuda_stream=8128,
                  wait_stream=lambda stream: self.events.append(("wait", stream)))

        def launch(*args):
            self.events.append(("launch", args))
            return self.rc

        lib = NS(l2pf_prefetch_fp8c_out=launch) if native else NS()

        def fallback(ranges):
            self.fallbacks.append(ranges)
            return False

        ns = dict(torch=NS(uint8="uint8", cuda=NS(current_stream=lambda device: main)),
                  _FP8C_TILE_PREFETCH=enabled, ENABLED=True, _lib=lib,
                  CHUNK=65536, _side_stream=lambda device: side,
                  prefetch_many=fallback)
        function(L2PF, "_fp8c_out_plan", ns)
        run = function(L2PF, "prefetch_fp8c_out", ns)
        return run, main, ns

    def test_optional_native_abi_budget_and_stream_order(self):
        run, main, _ = self.setup_dispatch()
        tensor, gate = Tensor(), Tensor()
        gate.pointer = gate.storage.pointer = 0x3000000
        gate.storage.size = 128 * 2688 * 2
        budget = (5 << 20) - gate.storage.size
        self.assertTrue(run(tensor, gate, gate.storage.size, budget))
        self.assertEqual(self.events[0], ("wait", main))
        self.assertEqual(self.events[1][0], "launch")
        args = self.events[1][1]
        self.assertEqual(len(args), 10)
        self.assertEqual(args[:4], (gate.pointer, gate.storage.size, tensor.pointer, tensor.shape[1]))
        prefix, tail_offset, tail, tiles, chunk, stream = args[4:]
        self.assertEqual((tail_offset, tiles, chunk, stream), (TAIL_OFFSET, TILES, 65536, 8128))
        self.assertLessEqual(tiles * (prefix + tail) + gate.storage.size, 5 << 20)
        self.assertFalse(self.fallbacks)

    def test_old_library_disabled_feature_and_bad_marker_keep_exact_fallback(self):
        for native, enabled, bad_marker in ((False, True, False), (True, False, False), (True, True, True)):
            run, _, _ = self.setup_dispatch(native=native, enabled=enabled)
            tensor, gate = Tensor(), Tensor()
            if bad_marker:
                del tensor._fp8c_prefetch_layout
            self.assertFalse(run(tensor, gate, 1024, 4 << 20))
            self.assertEqual(self.fallbacks, [((gate, 1024), (tensor, 4 << 20))])
            self.assertEqual(self.events, [])

    def test_native_error_is_not_silently_double_launched(self):
        run, _, _ = self.setup_dispatch()
        self.rc = 999
        with self.assertRaises(RuntimeError):
            run(Tensor(), max_bytes=5 << 20)
        self.assertFalse(self.fallbacks)
        self.assertEqual([event[0] for event in self.events], ["wait", "launch"])

    def test_incompatible_router_range_keeps_original_fallback(self):
        for change in (
            lambda t: setattr(t, "is_cuda", False),
            lambda t: setattr(t, "device", Device(index=1)),
            lambda t: setattr(t.storage, "pointer", t.pointer + 1),
        ):
            run, _, _ = self.setup_dispatch()
            tensor, gate = Tensor(), Tensor()
            change(gate)
            self.assertFalse(run(tensor, gate, 1024, 4 << 20))
            self.assertEqual(self.fallbacks, [((gate, 1024), (tensor, 4 << 20))])
            self.assertEqual(self.events, [])
        run, _, _ = self.setup_dispatch()
        tensor, gate = Tensor(), Tensor()
        self.assertFalse(run(tensor, gate, 1023, 4 << 20))
        self.assertEqual(self.events, [])

    def test_mamba_hook_selects_only_compressed_decode_and_preserves_join(self):
        for compressed, has_helper, rows, queued in (
            (True, True, 5, True),
            (True, True, 5, False),
            (False, True, 5, True),
            (True, False, 5, True),
            (True, True, 496, True),
        ):
            with self.subTest(compressed=compressed, helper=has_helper, rows=rows,
                              queued=queued):
                events = []
                raw, compact, gate = object(), object(), object()
                projection = NS(weight=raw, _fp8small_weight=raw)
                if compressed:
                    projection._fp8c_weight = compact

                def conv(**kwargs):
                    events.append(("conv", kwargs))

                mixer = NS(out_proj=projection, conv_ssm_forward=conv)
                reserved, budget = 688128, 5 << 20
                pf = NS(MAMBA_BYTES=budget,
                        gate_weight=lambda name, cap: (gate, reserved),
                        prefetch_many=lambda ranges: events.append(("ordinary", ranges)) or queued,
                        join=lambda device: events.append(("join", device)))
                if has_helper:
                    pf.prefetch_fp8c_out = lambda weight, **kwargs: (
                        events.append(("tiled", weight, kwargs)) or queued
                    )
                ns = dict(_l2pf=pf, _resolve_layer_name=lambda name: name,
                          get_forward_context=lambda: NS(no_compile_layers={"test": mixer}))
                function(MIXER, "_out_proj_weight", ns)
                run = function(MIXER, "mamba_mixer2", ns)
                projected = NS(shape=(rows, 10304), device=Device())
                output = object()
                run(projected, output, "test")
                if rows > 8:
                    self.assertEqual([event[0] for event in events], ["conv"])
                    continue
                selected = "tiled" if compressed and has_helper else "ordinary"
                expected = [selected, "conv"] + (["join"] if queued else [])
                self.assertEqual([event[0] for event in events], expected)
                if selected == "tiled":
                    self.assertIs(events[0][1], compact)
                    self.assertEqual(events[0][2], dict(
                        gate=gate, gate_bytes=reserved, max_bytes=budget - reserved,
                    ))
                else:
                    weight = compact if compressed else raw
                    self.assertEqual(events[0][1], ((gate, reserved), (weight, budget - reserved)))
                self.assertEqual(events[1][1], dict(projected_states=projected, output=output))


if __name__ == "__main__":
    unittest.main()

"""Execute the kernel's integer expressions using NumPy, without Triton/GPU.

The AST-extracted kernel is checked against an independent stable-grouping
oracle, including buffer bounds and overlapping writes. This does not compile
Triton or prove CUDA layout, graph capture, or floating-point output equality.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

try:
    import numpy as np
except ImportError:
    np = None


SOURCE = Path(__file__).resolve().parents[2] / (
    "vllm/vllm/model_executor/layers/fused_moe/moe_align_block_size.py"
)


def extract(name, namespace):
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    source = "from __future__ import annotations\n" + ast.unparse(node)
    exec(compile(source, str(SOURCE), "exec"), namespace)
    return namespace[name]


def tensor(rows=4, columns=6, dtype="int32", cuda=True, contiguous=True):
    return NS(ndim=2, shape=(rows, columns), dtype=dtype, is_cuda=cuda,
              device="cuda" if cuda else "cpu", numel=lambda: rows * columns,
              is_contiguous=lambda: contiguous)


class DispatchTests(unittest.TestCase):
    def guard(self, capability=121, cuda=True):
        return extract("_use_small_moe_align", dict(
            torch=NS(int32="int32", int64="int64"),
            current_platform=NS(is_cuda=lambda: cuda,
                is_device_capability=lambda value: value == capability)))

    def test_serial_and_all_six_depths_on_supported_tiles(self):
        guard = self.guard()
        for rows in range(1, 8):
            for block in (16, 32, 64):
                for dtype in ("int32", "int64"):
                    self.assertTrue(guard(tensor(rows=rows, dtype=dtype), block, 128, None))

    def test_promoted_humming_table_keeps_supported_identical_block_m(self):
        # Read the actual promoted tuning table, without importing torch or
        # humming. N/warp/stage changes must not silently bypass our guard or
        # make the two expert GEMMs disagree about the alignment block size.
        path = SOURCE.parent / "experts/humming_tuned_moe.py"
        nodes = []
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name == "_cfg":
                nodes.append(node)
            elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_BIG" for t in node.targets
            ):
                nodes.append(node)
            elif (isinstance(node, ast.AnnAssign)
                  and isinstance(node.target, ast.Name)
                  and node.target.id == "TUNED_CONFIGS"):
                nodes.append(node)
        namespace = {}
        source = "from __future__ import annotations\n" + ast.unparse(
            ast.Module(body=nodes, type_ignores=[]))
        exec(compile(source, str(path), "exec"), namespace)
        table = namespace["TUNED_CONFIGS"]
        for rows in range(1, 8):
            configs = [next(cfg for limit, cfg in table[key] if rows * 6 <= limit)
                       for key in (("w13", 1856, 2688, 128),
                                   ("w2", 2688, 1856, 128))]
            self.assertEqual([c["block_shape"][0] for c in configs], [16, 16])
            self.assertTrue(self.guard()(tensor(rows=rows), 16, 128, None))

    def test_unsupported_inputs_keep_general_path(self):
        guard = self.guard()
        cases = [(tensor(rows=0), 16, 128, None),
                 (tensor(rows=8), 16, 128, None),
                 (tensor(columns=5), 16, 128, None),
                 (tensor(dtype="float32"), 16, 128, None),
                 (tensor(cuda=False), 16, 128, None),
                 (tensor(contiguous=False), 16, 128, None),
                 (tensor(), 8, 128, None), (tensor(), 128, 128, None),
                 (tensor(), 16, 127, None), (tensor(), 16, 128, object())]
        for args in cases:
            self.assertFalse(guard(*args))
        for capability in (90, 100, 120):
            self.assertFalse(self.guard(capability)(tensor(), 16, 128, None))
        self.assertFalse(self.guard(cuda=False)(tensor(), 16, 128, None))
        one_dimensional = tensor()
        one_dimensional.ndim = 1
        self.assertFalse(guard(one_dimensional, 16, 128, None))

    def test_wrapper_allocates_and_launches_once_without_general_op(self):
        calls = []

        class Launch:
            def __getitem__(self, grid):
                def run(*args, **kwargs):
                    calls.append((grid, args, kwargs))
                return run

        def empty(shape, dtype, device):
            if isinstance(shape, int):
                shape = (shape,)
            return NS(shape=shape, dtype=dtype, device=device)

        def general(*args):
            calls.append(("general", args))

        namespace = dict(
            torch=NS(empty=empty, int32="int32"),
            triton=NS(cdiv=lambda a, b: (a + b - 1) // b,
                      next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
            round_up=lambda a, b: ((a + b - 1) // b) * b,
            _use_small_moe_align=self.guard(), _small_moe_align_kernel=Launch(),
            ops=NS(moe_align_block_size=general),
        )
        wrapper = extract("moe_align_block_size", namespace)
        for rows in range(1, 8):
            for block in (16, 32, 64):
                for pad in (False, True):
                    calls.clear()
                    inp = tensor(rows=rows)
                    result = wrapper(inp, block, 128, pad_sorted_ids=pad,
                                     ignore_invalid_experts=True)
                    self.assertEqual(len(calls), 1)
                    grid, args, kw = calls[0]
                    self.assertEqual(grid, (1,))
                    self.assertIs(args[0], inp)
                    self.assertEqual(result, args[1:])
                    self.assertEqual([v.shape for v in result],
                                     [(rows * 6 * block,), (rows * 6,), (1,)])
                    self.assertEqual(kw["N"], rows * 6)
                    self.assertEqual(kw["BLOCK_SIZE"], block)
                    self.assertEqual(kw["MAX_PADDED"], rows * 6 * block)
                    self.assertEqual(kw["num_warps"], 4)
        calls.clear()
        mapping = object()
        wrapper(tensor(), 16, 128, expert_map=mapping, ignore_invalid_experts=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "general")
        self.assertIs(calls[0][1][-1], mapping)


if np is not None:
    class Vector(np.ndarray):
        def to(self, dtype):
            return self.astype(dtype)

    def vector(value):
        return np.asarray(value).view(Vector)

    class Pointer:
        def __init__(self, data, offset=0, writes=None):
            self.data, self.offset = data, offset
            self.writes = np.zeros(len(data), dtype=np.int32) if writes is None else writes

        def __add__(self, offset):
            return Pointer(self.data, self.offset + offset, self.writes)

    def load(ptr, mask=True, other=0):
        offset, mask = np.broadcast_arrays(ptr.offset, mask)
        selected = offset[mask]
        assert np.all(selected >= 0) and np.all(selected < len(ptr.data))
        out = np.full(offset.shape, other, dtype=ptr.data.dtype)
        out[mask] = ptr.data[selected]
        return vector(out)

    def store(ptr, value, mask=True):
        offset, value, mask = np.broadcast_arrays(ptr.offset, value, mask)
        indices = offset[mask]
        assert np.all(indices >= 0) and np.all(indices < len(ptr.data))
        assert np.unique(indices).size == indices.size, "overlapping writes in one store"
        assert np.all(ptr.writes[indices] == 0), "overlapping writes across stores"
        ptr.data[indices] = value[mask]
        ptr.writes[indices] += 1

    TL = NS(arange=lambda a, b: vector(np.arange(a, b)), int32=np.int32,
            load=load, store=store,
            sum=lambda a, axis: vector(np.sum(a, axis=axis, dtype=np.int32)),
            where=lambda p, a, b: vector(np.where(p, a, b)),
            cdiv=lambda a, b: (a + b - 1) // b)


@unittest.skipIf(np is None, "NumPy is needed for the CPU expression interpreter")
class KernelIntegerTests(unittest.TestCase):
    def check_routes(self, routes, block):
        kernel = extract("_small_moe_align_kernel", {"tl": TL})
        n = len(routes)
        inp = Pointer(np.asarray(routes, dtype=np.int64))
        sorted_ids = Pointer(np.full(n * block, -999, dtype=np.int32))
        expert_ids = Pointer(np.full(n, -999, dtype=np.int32))
        total = Pointer(np.full(1, -999, dtype=np.int32))
        power = lambda x: 1 << (x - 1).bit_length()
        kernel(inp, sorted_ids, expert_ids, total, N=n, BLOCK_SIZE=block,
               ROUTE_BLOCK=power(n), PAD_BLOCK=power(((n + block - 1) // block) * block),
               MAX_PADDED=n * block, TAIL_BLOCK=power(n * block))

        want_ids, want_experts = [], []
        for expert in sorted(set(routes)):
            positions = [i for i, e in enumerate(routes) if e == expert]
            padded = ((len(positions) + block - 1) // block) * block
            want_ids.extend(positions + [n] * (padded - len(positions)))
            want_experts.extend([expert] * (padded // block))
        self.assertEqual(int(total.data[0]), len(want_ids))
        want_ids += [n] * (n * block - len(want_ids))
        want_experts += [-1] * (n - len(want_experts))
        np.testing.assert_array_equal(sorted_ids.data, want_ids)
        np.testing.assert_array_equal(expert_ids.data, want_experts)
        for ptr in (sorted_ids, expert_ids, total):
            np.testing.assert_array_equal(ptr.writes, np.ones_like(ptr.writes))

    def test_router_shaped_random_cases(self):
        rng = np.random.default_rng(20261024)
        for rows in range(1, 8):
            for block in (16, 32, 64):
                for _ in range(16):
                    routes = np.concatenate([rng.choice(128, 6, replace=False)
                                             for _ in range(rows)]).tolist()
                    self.check_routes(routes, block)

    def test_concentrated_and_multiblock_experts(self):
        # Repeated IDs within a row are not generated by top-k, but they expose
        # padding and rank bugs when an expert spans more than one GEMM tile.
        for rows in range(1, 8):
            for block in (16, 32, 64):
                for routes in ([0] * (rows * 6), [127] * (rows * 6),
                               [127, 0, 127, 0, 64, 64] * rows,
                               list(range(rows * 6))[::-1]):
                    self.check_routes(routes, block)

    def test_last_slot_and_exact_block_boundaries(self):
        for block in (16, 32, 64):
            for n in (6, 12, 18, 24, 30, 36, 42):
                # A split at block-1, block and block+1 exercises all padding
                # boundaries while keeping the wrapper's route-count shapes.
                for split in (block - 1, block, block + 1):
                    split = min(n, split)
                    self.check_routes([127] * split + [0] * (n - split), block)


if __name__ == "__main__":
    unittest.main()

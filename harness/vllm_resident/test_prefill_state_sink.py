"""CPU AST/pointer checks only; not a Triton compile or GPU numerical test."""

import ast
import copy
import inspect
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "vllm/vllm/model_executor/layers/mamba/ops"
PASSING, COMBINED = OPS / "ssd_state_passing.py", OPS / "ssd_combined.py"


def extract(path, name, ns):
    tree = ast.parse(path.read_text())
    fn = copy.deepcopy(
        next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
    )
    fn.decorator_list, fn.returns = [], None
    for arg in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs):
        arg.annotation = None
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
            str(path),
            "exec",
        ),
        ns,
    )
    return ns[name]


class Value(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype).view(Value)


def value(x):
    return np.asarray(x).view(Value)


class Pointer:
    def __init__(self, data, offset=0):
        self.data, self.offset = data.reshape(-1), np.asarray(offset)

    def __add__(self, offset):
        return Pointer(self.data, self.offset + offset)

    def __sub__(self, offset):
        return self + (-offset)


class TinyTL:
    float32, int64 = np.float32, np.int64
    pid = (0, 0, 0)

    def program_id(self, axis):
        return self.pid[axis]

    def arange(self, start, end):
        return value(np.arange(start, end))

    def zeros(self, shape, dtype):
        return value(np.zeros(shape, dtype=dtype))

    def where(self, condition, x, y):
        return value(np.where(condition, x, y))

    def load(self, ptr, mask=True, other=0):
        offsets, mask = np.broadcast_arrays(ptr.offset, np.asarray(mask))
        live = offsets[mask]
        assert not np.any(live < 0) and not np.any(live >= ptr.data.size), (
            "unmasked OOB load"
        )
        result = np.full(offsets.shape, other, dtype=ptr.data.dtype)
        result[mask] = ptr.data[live]
        return value(result)

    def store(self, ptr, x, mask=True):
        offsets, x, mask = np.broadcast_arrays(ptr.offset, np.asarray(x), mask)
        live = offsets[mask]
        assert not np.any(live < 0) and not np.any(live >= ptr.data.size), (
            "unmasked OOB store"
        )
        ptr.data[live] = x[mask]


class Tensor:
    def __init__(self, shape, dtype=np.float32, device="cuda:0", strides=None):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.ndim, self.is_cuda = len(shape), device.startswith("cuda")
        self.strides = strides or tuple(
            math.prod(shape[i + 1 :]) for i in range(len(shape))
        )
        self.gathers = []

    def stride(self, dim):
        return self.strides[dim]

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        self.gathers.append(index)
        return self


TORCH = NS(
    float32=np.float32,
    int32=np.int32,
    int64=np.int64,
    empty=lambda shape, **kw: Tensor(shape, **kw),
)


class KernelTests(unittest.TestCase):
    def run_case(self, chunks, initial=False, sink=True, cache_index=2):
        tl = TinyTL()
        kernel = extract(
            PASSING,
            "_state_passing_fwd_kernel",
            {"tl": tl, "fast_exp": lambda x: value(np.exp(x).astype(np.float32))},
        )
        rng = np.random.default_rng(191)
        heads, dim, block, chunk_size = 2, 12, 8, 4
        states = rng.standard_normal((chunks, heads, dim)).astype(np.float32)
        da = -rng.random((heads, chunks, chunk_size)).astype(np.float32)
        init = rng.standard_normal((1, heads, dim)).astype(np.float32)
        out = np.full_like(states, np.nan)
        cache = np.full((5, heads, dim + 4), -917.0, dtype=np.float32)
        index_storage = np.array([cache_index, 99, 99, 99], dtype=np.int32)
        kwargs = dict(
            states_ptr=Pointer(states),
            out_ptr=Pointer(out),
            dA_cs_ptr=Pointer(da),
            initstates_ptr=Pointer(init),
            last_chunk_indices_ptr=Pointer(np.array([chunks - 1])),
            final_states_ptr=Pointer(cache),
            final_indices_ptr=Pointer(index_storage),
            dim=dim,
            chunk_size=chunk_size,
            stride_states_chunk=heads * dim,
            stride_states_head=dim,
            stride_states_dim=1,
            stride_out_chunk=heads * dim,
            stride_out_head=dim,
            stride_out_dim=1,
            stride_dA_cs_head=chunks * chunk_size,
            stride_dA_cs_chunk=chunk_size,
            stride_dA_cs_csize=1,
            stride_initstates_batch=heads * dim,
            stride_initstates_head=dim,
            stride_initstates_dim=1,
            stride_final_batch=heads * (dim + 4),
            stride_final_head=dim + 4,
            stride_final_indices=4,
            final_cache_size=5,
            HAS_INITSTATES=initial,
            STORE_FINAL=sink,
            BLOCK_SIZE=block,
        )
        for head in range(heads):
            for tile in range(math.ceil(dim / block)):
                tl.pid = (tile, 0, head)
                kernel(**kwargs)
        expected = np.empty_like(states)
        state = init[0].copy() if initial else np.zeros((heads, dim), np.float32)
        for chunk in range(chunks):
            state = np.exp(da[:, chunk, -1])[:, None] * state + states[chunk]
            expected[chunk] = state
        if sink:
            np.testing.assert_array_equal(
                out[:-1].view(np.uint32), expected[:-1].view(np.uint32)
            )
            self.assertTrue(np.isnan(out[-1]).all())
            target = cache_index % 5
            np.testing.assert_array_equal(
                cache[target, :, :dim].view(np.uint32), expected[-1].view(np.uint32)
            )
            untouched = np.ones(cache.shape, dtype=bool)
            untouched[target, :, :dim] = False
            self.assertTrue((cache[untouched] == -917).all())
        else:
            np.testing.assert_array_equal(out.view(np.uint32), expected.view(np.uint32))
            self.assertTrue((cache == -917).all())

    def test_four_chunks_and_partial_row_tile(self):
        self.run_case(4)

    def test_initial_state(self):
        self.run_case(4, initial=True)

    def test_one_chunk_no_terminal_scratch_write(self):
        self.run_case(1)

    def test_five_chunks_and_initial_state(self):
        self.run_case(5, initial=True)

    def test_legal_negative_cache_index(self):
        self.run_case(2, cache_index=-1)

    def test_zero_cache_index_like_original_indexing(self):
        self.run_case(2, cache_index=0)

    def test_disabled_sink_preserves_all_boundaries(self):
        self.run_case(4, initial=True, sink=False)


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.guard = extract(COMBINED, "prefill_state_sink_supported", {"torch": TORCH})
        self.cache = Tensor((8, 64, 64, 128))
        self.indices = Tensor((1,), dtype=np.int32, strides=(4,))

    def check(self, cache=None, indices=None, batch=1):
        return self.guard(
            cache or self.cache, indices or self.indices, batch, (64, 64, 128)
        )

    def test_valid_single_request_strided_indices(self):
        self.assertTrue(self.check())
        self.indices.dtype = np.int64
        self.assertTrue(self.check())

    def test_wrong_dtype_device_shape(self):
        for attr, item in (
            ("dtype", np.float16),
            ("is_cuda", False),
            ("shape", (8, 64, 64, 64)),
            ("ndim", 3),
        ):
            with self.subTest(attr=attr):
                old = getattr(self.cache, attr)
                setattr(self.cache, attr, item)
                self.assertFalse(self.check())
                setattr(self.cache, attr, old)

    def test_noncontiguous_cache_or_mismatched_indices(self):
        self.cache.strides = (64 * 64 * 128, 64 * 128, 256, 2)
        self.assertFalse(self.check())
        self.cache = Tensor((8, 64, 64, 128))
        for index in (
            Tensor((1, 1), np.int32),
            Tensor((1,), np.float32),
            Tensor((1,), np.int32, "cpu"),
        ):
            self.assertFalse(self.check(indices=index))
        self.assertFalse(self.guard(self.cache, None, 1, (64, 64, 128)))

    def test_empty_and_multiple_requests(self):
        self.assertFalse(self.check(batch=0))
        self.assertFalse(self.check(batch=2))


class RoutingTests(unittest.TestCase):
    def test_combined_runs_scan_without_final_gather(self):
        for sink in (False, True):
            with self.subTest(sink=sink):
                states = Tensor((4, 2, 4, 3))
                scans, passes = [], []

                def passing(*args, **kwargs):
                    passes.append(kwargs)
                    return states

                ns = {
                    "torch": TORCH,
                    "is_int_pow_2": lambda x: x == 4,
                    "rearrange": lambda t, *a, **kw: t,
                    "_chunk_cumsum_fwd": lambda *a, **kw: (
                        Tensor((2, 4, 4)),
                        Tensor((2, 4, 4)),
                    ),
                    "_chunk_state_fwd": lambda *a, **kw: states,
                    "_state_passing_fwd": passing,
                    "_bmm_chunk_fwd": lambda *a, **kw: Tensor((4, 1, 4, 4)),
                    "_chunk_scan_fwd": lambda *a, **kw: scans.append((a, kw)),
                }
                fn = extract(COMBINED, "_mamba_chunk_scan_combined_fwd", ns)
                cache, indices = Tensor((5, 2, 4, 3)), Tensor((1,), np.int32)
                result = fn(
                    Tensor((16, 2, 4)),
                    Tensor((16, 2)),
                    Tensor((2,)),
                    Tensor((16, 1, 3)),
                    Tensor((16, 1, 3)),
                    4,
                    Tensor((16, 2, 4)),
                    cu_seqlens=Tensor((2,), np.int32),
                    cu_chunk_seqlens=Tensor((5,), np.int32),
                    last_chunk_indices=indices,
                    seq_idx=Tensor((4,), np.int32),
                    state_dtype=np.float32,
                    final_states_out=cache if sink else None,
                    final_states_indices=indices if sink else None,
                )
                self.assertEqual(len(scans), 1)
                self.assertIs(passes[0]["final_states_out"], cache if sink else None)
                self.assertEqual(len(states.gathers), 0 if sink else 1)
                self.assertIs(result, None if sink else states)

    def test_actual_launch_signature_binds_all_parameters(self):
        signature = inspect.signature(
            extract(PASSING, "_state_passing_fwd_kernel", {"tl": TinyTL()})
        )
        captures = []

        class Launch:
            def __getitem__(self, grid):
                def call(**kwargs):
                    signature.bind(**kwargs, BLOCK_SIZE=1024)
                    captures.append(kwargs)

                return call

        class Context:
            def __enter__(self):
                return None

            def __exit__(self, *args):
                return False

        device, torch = NS(index=0), NS(**vars(TORCH))
        torch.empty = lambda shape, **kw: Tensor(shape, dtype=kw["dtype"])
        torch.accelerator = NS(device_index=lambda *a: Context())
        fn = extract(
            PASSING,
            "_state_passing_fwd",
            {
                "torch": torch,
                "triton": NS(cdiv=lambda a, b: (a + b - 1) // b),
                "_state_passing_fwd_kernel": Launch(),
            },
        )
        states, da, ends = Tensor((4, 2, 12)), Tensor((2, 4, 4)), Tensor((1,), np.int32)
        cache, indices = Tensor((5, 2, 4, 3)), Tensor((1,), np.int32, strides=(4,))
        for tensor in (states, da, ends, cache, indices):
            tensor.device = device
        fn(states, da, ends, final_states_out=cache, final_states_indices=indices)
        fn(states, da, ends)
        self.assertEqual([x["STORE_FINAL"] for x in captures], [True, False])
        self.assertEqual(captures[0]["stride_final_indices"], 4)
        self.assertEqual(captures[0]["final_cache_size"], 5)

    def test_mixer_retains_copy_for_fallback_and_prefix(self):
        path = OPS.parent / "mamba_mixer2.py"
        tree = ast.parse(path.read_text())
        assignment = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "direct_state_sink"
                for t in n.targets
            )
        )
        expression = compile(ast.Expression(assignment.value), str(path), "eval")
        for prefix, eligible in ((False, True), (False, False), (True, True)):
            ns = dict(
                is_mamba_cache_all=prefix,
                prefill_state_sink_supported=lambda *a: eligible,
                ssm_state=None,
                state_indices_tensor_p=None,
                num_prefills=1,
                self=NS(num_heads=64, tp_size=1, head_dim=64, ssm_state_size=128),
            )
            self.assertEqual(eval(expression, ns), eligible and not prefix)
        branch = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If) and ast.unparse(n.test) == "not direct_state_sink"
        )
        self.assertTrue(any(isinstance(n, ast.Assign) for n in branch.body))


if __name__ == "__main__":
    unittest.main()

"""CPU math, bounds, dispatch and graph-signature checks; no GPU proof."""

import ast
import copy
import inspect
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

try:
    from . import test_ssd_prefill_fused as sim
except ImportError:
    import test_ssd_prefill_fused as sim

np = sim.np
ROOT = Path(__file__).resolve().parents[2]
ROUTER = ROOT / "vllm/vllm/model_executor/layers/fused_moe/router"
SOURCE = ROUTER / "gb10_fp32_router.py"
GATE = ROUTER / "gate_linear.py"


class TL(sim.TL):
    @staticmethod
    def cdiv(a, b):
        return (a + b - 1) // b

    @staticmethod
    def fma(a, b, c):
        # FP64 intermediate then FP32 rounding models the per-lane FMA.
        return sim.value(
            (
                np.asarray(a, np.float64) * np.asarray(b, np.float64)
                + np.asarray(c, np.float64)
            ).astype(np.float32)
        )

    @staticmethod
    def sum(x, axis):
        # Deterministic balanced reduction, independent of NumPy BLAS.
        values = np.moveaxis(np.asarray(x), axis, -1)
        assert values.shape[-1] & (values.shape[-1] - 1) == 0
        while values.shape[-1] > 1:
            half = values.shape[-1] // 2
            values = (values[..., :half] + values[..., half:]).astype(np.float32)
        return sim.value(values[..., 0])


def run_kernel(x, weight, rows, experts, k, padded=False):
    # Guard words after the output catch writes by masked token/expert lanes.
    output = sim.Tensor(np.full(rows * experts + 31, -917.0, np.float32))
    tl = TL()
    kernel = sim.function(SOURCE, "_small_fp32_router_kernel", dict(tl=tl))
    sim.call(
        kernel,
        tl,
        ((experts + 1) // 2,),
        X=x,
        W=weight,
        Y=output,
        M=rows,
        K=k,
        N=experts,
        XM=x.stride(0),
        XK=1,
        WN=weight.stride(0),
        WK=1,
        BM=4,
        BN=2,
        BK=512,
    )
    assert (output.data[rows * experts :] == -917.0).all()
    return output.data[: rows * experts].reshape(rows, experts)


class MathTests(unittest.TestCase):
    def case(self, rows, experts=128, k=2688, padding=0, bf16=True, exact=False):
        rng = np.random.default_rng(813)
        shape_x, shape_w = (rows, k + padding), (experts, k + padding)
        if exact:
            # Products/sums stay exactly representable, including cancellation.
            x0 = rng.integers(-3, 4, shape_x).astype(np.float32) / 16
            w0 = rng.integers(-3, 4, shape_w).astype(np.float32) / 32
        else:
            x0 = rng.uniform(-1, 1, shape_x).astype(np.float32)
            w0 = rng.uniform(-1, 1, shape_w).astype(np.float32)
        x = sim.Tensor(x0, "bf16" if bf16 else np.float32)
        w = sim.Tensor(w0, np.float32)
        result = run_kernel(x, w, rows, experts, k)
        products = x.data[:, None, :k].astype(np.float64) * w.data[None, :, :k]
        reference = products.sum(axis=-1, dtype=np.float64)
        self.assertTrue(np.isfinite(result).all())
        if exact:
            np.testing.assert_array_equal(result.astype(np.float64), reference)
        else:
            # At most 6 FMA rounds + 9 reduction rounds for K=2688.
            gamma = (6 + 9) * np.finfo(np.float32).eps
            absolute_bound = gamma * np.abs(products).sum(axis=-1) + 1e-7
            self.assertTrue(
                (np.abs(result.astype(np.float64) - reference) <= absolute_bound).all()
            )
        return x, w, result

    def test_all_decode_row_counts_bf16_inputs(self):
        for rows in (1, 2, 3, 4):
            with self.subTest(rows=rows):
                self.case(rows)

    def test_fp32_input_and_padded_outer_strides(self):
        self.case(4, padding=19, bf16=False)

    def test_exact_integer_fraction_cancellation(self):
        self.case(4, exact=True)

    def test_k_tail_and_partial_expert_tile(self):
        for k in (1, 511, 512, 513, 2688):
            with self.subTest(k=k):
                self.case(3, experts=7, k=k, padding=11, exact=True)

    def test_batch_rows_have_identical_reductions(self):
        x, w, four = self.case(4)
        for row in range(4):
            one = sim.Tensor(x.data[row : row + 1], "bf16")
            actual = run_kernel(one, w, 1, 128, 2688)
            np.testing.assert_array_equal(
                actual.view(np.uint32), four[row : row + 1].view(np.uint32)
            )

    def test_zero_input_and_inactive_lanes(self):
        x = sim.Tensor(np.zeros((1, 2688), np.float32), "bf16")
        w = sim.Tensor(np.ones((128, 2688), np.float32))
        np.testing.assert_array_equal(run_kernel(x, w, 1, 128, 2688), 0)


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.torch = sim.torch_double()
        self.x = sim.Tensor(np.zeros((4, 2688), np.float32), "bf16")
        self.w = sim.Tensor(np.zeros((128, 2688), np.float32))
        self.supported = sim.function(
            SOURCE, "small_fp32_router_supported", dict(torch=self.torch)
        )

    def test_eligible_and_shape_fallbacks(self):
        self.assertTrue(self.supported(self.x, self.w))
        for shape in ((0, 2688), (5, 2688), (496, 2688), (4, 2687), (4, 1, 2688)):
            self.assertFalse(
                self.supported(sim.Tensor(np.zeros(shape), "bf16"), self.w)
            )
        self.assertFalse(
            self.supported(self.x, sim.Tensor(np.zeros((127, 2688), np.float32)))
        )

    def test_dtype_device_and_layout_guards(self):
        for t, attr, bad in (
            (self.x, "is_cuda", False),
            (self.w, "device", "cuda:1"),
            (self.w, "dtype", "bf16"),
            (self.x, "dtype", np.float16),
        ):
            old = getattr(t, attr)
            setattr(t, attr, bad)
            self.assertFalse(self.supported(self.x, self.w))
            setattr(t, attr, old)
        self.x.stride = lambda dim: (5376, 2)[dim]
        self.assertFalse(self.supported(self.x, self.w))

    def test_launch_signature_and_fixed_reduction_shape(self):
        calls = []
        signature = inspect.signature(
            sim.function(SOURCE, "_small_fp32_router_kernel", {})
        )

        class Launch:
            def __getitem__(self, grid):
                def run(*args, **kwargs):
                    self_warps = kwargs.pop("num_warps")
                    self_stages = kwargs.pop("num_stages")
                    bound = signature.bind(*args, **kwargs).arguments
                    calls.append((grid, self_warps, self_stages, bound))

                return run

        self.torch.empty = lambda shape, **kw: sim.Tensor(np.empty(shape, np.float32))
        fn = sim.function(
            SOURCE,
            "small_fp32_router",
            dict(
                torch=self.torch,
                small_fp32_router_supported=self.supported,
                _small_fp32_router_kernel=Launch(),
            ),
        )
        for rows in (1, 4):
            x = sim.Tensor(np.zeros((rows, 2688)), "bf16")
            out = fn(x, self.w)
            self.assertEqual(out.shape, (rows, 128))
            grid, warps, stages, args = calls[-1]
            self.assertEqual((grid, warps, stages), ((64,), 4, 2))
            self.assertEqual((args["BM"], args["BN"], args["BK"]), (4, 2, 512))

    def test_custom_op_fallback_keeps_fp32(self):
        calls = []

        class Castable:
            def to(self, dtype):
                calls.append(dtype)
                return self

        x, w, expected = Castable(), Castable(), object()
        self.torch.nn = NS(functional=NS(linear=lambda a, b: expected))
        fn = sim.function(
            SOURCE,
            "small_fp32_router",
            dict(torch=self.torch, small_fp32_router_supported=lambda *a: False),
        )
        self.assertIs(fn(x, w), expected)
        self.assertEqual(calls, [np.dtype("float32"), np.dtype("float32")])

    def test_model_initialization_scope(self):
        tree = ast.parse(GATE.read_text())
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Attribute)
                and t.attr == "allow_gb10_small_fp32_router"
                for t in n.targets
            )
        )
        expression = compile(ast.Expression(node.value), str(GATE), "eval")
        ns = dict(
            force_fp32_compute=True,
            bias=False,
            input_size=2688,
            output_size=128,
            out_dtype=np.dtype("float32"),
            torch=self.torch,
            self=NS(weight=self.w),
            current_platform=NS(
                is_cuda=lambda: True, is_device_capability=lambda cap: cap == 121
            ),
        )
        self.assertTrue(eval(expression, ns))
        for key, bad in (
            ("bias", True),
            ("force_fp32_compute", False),
            ("input_size", 6144),
            ("output_size", 256),
            ("out_dtype", "bf16"),
        ):
            old = ns[key]
            ns[key] = bad
            self.assertFalse(eval(expression, ns))
            ns[key] = old
        ns["current_platform"] = NS(
            is_cuda=lambda: True, is_device_capability=lambda cap: False
        )
        self.assertFalse(eval(expression, ns))

    def test_gate_forward_keeps_large_batch_fallback(self):
        tree = ast.parse(GATE.read_text())
        gate = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "GateLinear"
        )
        method = copy.deepcopy(
            next(
                n
                for n in gate.body
                if isinstance(n, ast.FunctionDef) and n.name == "forward"
            )
        )
        method.returns = None
        for arg in method.args.args:
            arg.annotation = None
        cls = ast.ClassDef(
            name="Gate",
            bases=[ast.Name(id="Base", ctx=ast.Load())],
            keywords=[],
            body=[method],
            decorator_list=[],
        )
        old_result, new_result = object(), object()

        class Base:
            def forward(self, x):
                return old_result, None

        self.torch.ops = NS(vllm=NS(gb10_small_fp32_router=lambda *a: new_result))
        ns = dict(Base=Base, torch=self.torch)
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                str(GATE),
                "exec",
            ),
            ns,
        )
        gate = ns["Gate"]()
        gate.weight, gate.out_dtype = self.w, None
        for name in (
            "allow_ll_bf16_gemm",
            "allow_dsv3_router_gemm",
            "allow_fp32_router_gemm",
            "allow_bf16x3_router_gemm",
            "allow_cublas_router_gemm",
        ):
            setattr(gate, name, False)
        for enabled, rows in (
            (True, 1),
            (True, 4),
            (True, 0),
            (True, 5),
            (True, 496),
            (False, 4),
        ):
            gate.allow_gb10_small_fp32_router = enabled
            x = sim.Tensor(np.zeros((rows, 2688)), "bf16")
            self.assertIs(
                gate.forward(x)[0],
                new_result if enabled and 1 <= rows <= 4 else old_result,
            )

    def test_fake_shape_and_fp32_only_kernel(self):
        captures = []
        x = NS(
            shape=(4, 2688), new_empty=lambda shape, **kw: captures.append((shape, kw))
        )
        fn = sim.function(SOURCE, "small_fp32_router_fake", dict(torch=self.torch))
        fn(x, self.w)
        self.assertEqual(captures, [((4, 128), dict(dtype=np.dtype("float32")))])
        tree = ast.parse(SOURCE.read_text())
        attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        self.assertIn("fma", attributes)
        self.assertNotIn("dot", attributes)
        self.assertFalse({"cpu", "item", "numpy", "synchronize"} & attributes)


if __name__ == "__main__":
    unittest.main()

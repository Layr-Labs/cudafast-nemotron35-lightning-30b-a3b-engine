"""CPU checks of captured-prefill cache-policy dispatch, not GPU timing."""

import ast
from contextlib import nullcontext
import copy
import ctypes
from pathlib import Path
import re
from types import SimpleNamespace as NS
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "vllm/vllm/model_executor/models"
EXPERTS = ROOT / "vllm/vllm/model_executor/layers/fused_moe/experts"
PZ = ROOT / "harness/kernels/pzskip"


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = copy.deepcopy(next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == name
    ))
    node.decorator_list = []
    node.returns = None
    for arg in (*node.args.args, *node.args.kwonlyargs):
        arg.annotation = None
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[name]


class Tensor:
    dtype = "bf16"

    def __init__(self, rows, columns):
        self.shape = (rows, columns)

    def dim(self):
        return 2

    def stride(self, axis):
        return (self.shape[1], 1)[axis]

    def data_ptr(self):
        return 16


class PolicyTests(unittest.TestCase):
    def test_routed_up_selects_stock_only_for_rerouted_prefill(self):
        calls = []
        ns = dict(
            torch=NS(bfloat16="bf16"), os=NS(environ={}),
            prepare=lambda *args, **kwargs: calls.append(kwargs) or "configuration",
            ops=NS(launch_kernel=lambda **kwargs: kwargs),
        )
        run = load_function(EXPERTS / "humming_relu2_epilogue.py", "forward_relu2", ns)
        layer = NS(humming_metas={"w13": NS(
            bias_name="bias", weight_name="weight", weight_scale_name="scales",
            zero_point_name="zp", global_scale_name="gs",
        )}, bias=None, weight=object(), locks=object())
        for repeated, rows, topk, valid, disabled, expected in (
            (False, 496, 6, 2976, False, True),
            (True, 496, 6, 2976, False, False),
            (True, 496, 6, 2976, True, True),
            (False, 5, 6, 30, False, False),
            (True, 496, 3, 1488, False, False),
            (False, 496, 6, 2975, False, False),
        ):
            with self.subTest(repeated=repeated, rows=rows, disabled=disabled):
                layer._prefill_repeated_expert_rows = repeated
                ns["os"].environ["HUMMING_PREFILL_REUSE_CACHE"] = "0" if disabled else "1"
                result = run(layer, Tensor(rows, 2688), Tensor(valid, 1856),
                             "compute", "tuning", "sorted", "experts", "padded",
                             topk, valid)
                self.assertEqual(calls[-1]["evict_first"], expected)
                self.assertIs(result["weight"], layer.weight)
                self.assertEqual(result["valid_shape_m"], valid)
        ns["os"].environ["HUMMING_EVICT_FIRST"] = "0"
        layer._prefill_repeated_expert_rows = False
        run(layer, Tensor(496, 2688), Tensor(2976, 1856),
            "compute", "tuning", "sorted", "experts", "padded", 6, 2976)
        self.assertFalse(calls[-1]["evict_first"])

    def test_pz_dispatch_preserves_layout_and_old_library_fallback(self):
        launches, aligns = [], []
        pz = NS(
            block_rows=lambda config: 32 if 320 <= config < 340 else 64,
            down=lambda *args, **kwargs: launches.append((args, kwargs)),
        )
        ns = dict(torch=NS(bfloat16="bf16", compiler=NS(is_compiling=lambda: False)),
                  os=NS(environ={}), TOP_K=6, INTER=1856, HIDDEN=2688,
                  PZ_CONFIG=333, PZ_BLOCK_M=64, _pzskip=pz,
                  _align=lambda ids, bm: aligns.append((ids, bm)) or ("s", "e", "n"))
        run = load_function(EXPERTS / "humming_zskip.py", "_pz_down", ns)
        layer = NS(zskip_w2=object())
        for repeated, rows, disabled, capability, config, expected in (
            (True, 496, False, 334, 333, 334),
            (False, 496, False, 334, 333, 333),
            (True, 496, True, 334, 333, 333),
            (True, 495, False, 334, 333, 333),
            (True, 496, False, None, 333, 333),
            (True, 496, False, 0, 333, 333),
            (True, 496, False, 334, 322, 322),
        ):
            with self.subTest(repeated=repeated, rows=rows, capability=capability):
                layer._prefill_repeated_expert_rows = repeated
                ns["os"].environ["HUMMING_PREFILL_REUSE_CACHE"] = "0" if disabled else "1"
                ns["PZ_CONFIG"] = config
                if capability is None:
                    if hasattr(pz, "reuse_cache_config"):
                        del pz.reuse_cache_config
                else:
                    pz.reuse_cache_config = lambda: capability
                a, out, ids = Tensor(rows * 6, 1856), Tensor(rows * 6, 2688), object()
                self.assertTrue(run(layer, a, None, out, rows, 6, {}, ids))
                args, kwargs = launches[-1]
                self.assertEqual(kwargs["wpc"], expected)
                self.assertEqual(aligns[-1], (ids, 32))
                self.assertIs(args[0], a)
                self.assertIs(args[4], layer.zskip_w2)
                self.assertIs(args[5], out)
        count = len(launches)
        self.assertFalse(run(layer, a, None, out, 496, 3, {}, ids))
        self.assertEqual(len(launches), count)

    def test_flag_lifetime_and_capture_exception_clear(self):
        setter = load_function(MODELS / "nemotron_h.py", "set_last_moe_row_src", {})
        routed = NS()
        runner = NS(routed_experts=routed, keep_m_row_src=None)
        model = NS(_last_moe_runner=(runner,))
        model.set_last_moe_row_src = lambda row: setter(model, row)
        selected = object()
        self.assertTrue(model.set_last_moe_row_src(selected))
        self.assertIs(runner.keep_m_row_src, selected)
        self.assertTrue(routed._prefill_repeated_expert_rows)
        model.set_last_moe_row_src(None)
        self.assertIsNone(runner.keep_m_row_src)
        self.assertFalse(routed._prefill_repeated_expert_rows)
        self.assertFalse(setter(NS(_last_moe_runner=(None,)), selected))

        def fail_forward(**kwargs):
            self.assertTrue(routed._prefill_repeated_expert_rows)
            raise RuntimeError("deliberate capture failure")

        ns = dict(set_forward_context=lambda *args, **kwargs: nullcontext(),
                  CUDAGraphMode=NS(NONE="none"))
        forward = load_function(ROOT / "vllm/vllm/v1/worker/prefill_graph.py", "_forward", ns)
        capture = NS(runner=NS(get_model=lambda: model, vllm_config=object(),
                               _model_forward=fail_forward),
                     metadata={}, slots={}, logits_indices=selected,
                     input_ids=object(), positions=object())
        with self.assertRaisesRegex(RuntimeError, "deliberate capture failure"):
            forward(capture)
        self.assertIsNone(runner.keep_m_row_src)
        self.assertFalse(routed._prefill_repeated_expert_rows)

    def test_optional_library_marker_and_identical_native_specialization(self):
        class Function:
            def __init__(self, result):
                self.result = result

            def __call__(self):
                return self.result

        for marker, expected in ((None, 0), (334, 334), (333, 0)):
            lib = NS(pzskip_down=Function(0), pzskip_bi_bytes=Function(5120))
            if marker is not None:
                lib.pzskip_reuse_cache_config = Function(marker)
            ns = dict(_lib=None, _load_error=None, _reuse_cache_config=0,
                      LIB_PATH="test-only", ctypes=NS(
                          CDLL=lambda path: lib, c_void_p=ctypes.c_void_p,
                          c_longlong=ctypes.c_longlong, c_int=ctypes.c_int))
            load_function(PZ / "py/__init__.py", "_load", ns)
            feature = load_function(PZ / "py/__init__.py", "reuse_cache_config", ns)
            self.assertEqual(feature(), expected)
            self.assertEqual(len(lib.pzskip_down.argtypes), 21)
        source = (PZ / "csrc/pzskip.cu").read_text()
        new = re.search(r"case 334: return (pzskip::launch_smem4<[^>]+>\(p, st\));", source)
        old = source[source.index("case 333:"):source.index("case 340:")]
        self.assertIsNotNone(new)
        self.assertIn("return " + new.group(1) + ";", old)
        self.assertEqual(new.group(1), "pzskip::launch_smem4<2, 3, 6, 3, 32, 0>(p, st)")


if __name__ == "__main__":
    unittest.main()

"""Small CPU config tests; no claim of Humming compilation or GPU parity."""

import ast
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
EXPERTS = ROOT / "vllm/vllm/model_executor/layers/fused_moe/experts"
spec = importlib.util.spec_from_file_location(
    "prefill_wide_n", EXPERTS / "humming_prefill_wide_n.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def baseline_tables():
    # Extract the real numeric table and helpers, avoiding torch/vLLM imports.
    tree = ast.parse((EXPERTS / "humming_tuned_moe.py").read_text())
    wanted = {"_BIG", "TUNED_CONFIGS"}
    nodes = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in {"_cfg", "_to_humming_list"})
        or (
            isinstance(n, ast.Assign)
            and any(getattr(t, "id", "") in wanted for t in n.targets)
        )
        or (isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") in wanted)
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    ns = {}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *nodes], type_ignores=[])
            ),
            "<baseline-table>",
            "exec",
        ),
        ns,
    )
    table = ns["TUNED_CONFIGS"]
    return (
        ns["_to_humming_list"](table[("w13", 1856, 2688, 128)]),
        ns["_to_humming_list"](table[("w2", 2688, 1856, 128)]),
    )


def select(table, routes):
    return next(cfg for lo, hi, cfg in table if lo < routes <= hi)


class PrefillWideN(unittest.TestCase):
    def setUp(self):
        self.up, self.down = baseline_tables()

    def test_only_496_routes_change_and_up_values_stay_identical(self):
        up, down = module.widen_prefill_w2_n(self.up, self.down)
        for routes in [
            1,
            6,
            12,
            24,
            48,
            384,
            385,
            2970,
            2975,
            2976,
            2977,
            2982,
            6144,
            8192,
        ]:
            self.assertEqual(select(up, routes), select(self.up, routes))
            chosen = select(down, routes)
            if routes == 2976:
                self.assertEqual(chosen["block_shape"], [64, 384, 32])
                self.assertEqual(chosen["warp_shape"], [64, 64, 32])
            else:
                self.assertEqual(chosen, select(self.down, routes))

    def test_k_chain_and_splits_preserved(self):
        _, down = module.widen_prefill_w2_n(self.up, self.down)
        before, after = select(self.down, 2976), select(down, 2976)
        self.assertEqual(
            {k: v for k, v in before.items() if k not in {"block_shape", "warp_shape"}},
            {k: v for k, v in after.items() if k not in {"block_shape", "warp_shape"}},
        )
        self.assertEqual(after["block_shape"][0], 64)
        self.assertEqual(after["block_shape"][2], after["warp_shape"][2])
        self.assertEqual(after["block_shape"][2], 32)

    def test_no_source_or_cross_table_aliasing(self):
        saved = deepcopy((self.up, self.down))
        up, down = module.widen_prefill_w2_n(self.up, self.down)
        select(down, 2976)["block_shape"][0] = 999
        self.assertEqual((self.up, self.down), saved)
        self.assertEqual(select(up, 2976)["block_shape"][0], 64)
        self.assertEqual(select(down, 2975)["block_shape"][0], 64)

    def test_unknown_split_or_geometry_falls_back(self):
        for key, value in [
            ("use_stream_k", True),
            ("use_f16_accum", True),
            ("num_write_splits", 2),
            ("num_sms", 132),
            ("num_stages", 3),
            ("block_shape", [32, 128, 32]),
            ("use_stream_k", 0),
        ]:
            with self.subTest(key=key, value=value):
                changed = deepcopy(self.down)
                changed[-1][2][key] = value
                up, down = module.widen_prefill_w2_n(self.up, changed)
                self.assertIs(up, self.up)
                self.assertIs(down, changed)

    def test_empty_or_range_mismatch_falls_back(self):
        self.assertEqual(module.widen_prefill_w2_n([], []), ([], []))
        changed = deepcopy(self.down)
        changed[-1][0] += 1
        self.assertIs(module.widen_prefill_w2_n(self.up, changed)[1], changed)
        self.assertIs(module.widen_prefill_w2_n(self.up, self.down[:-1])[0], self.up)

    def test_all_columns_covered_once_without_padding(self):
        columns = [
            col for tile in range(7) for col in range(tile * 384, (tile + 1) * 384)
        ]
        self.assertEqual(columns, list(range(2688)))
        self.assertEqual(2688 // 128, 3 * len(range(7)))

    def test_resource_envelope_is_not_a_compilation_claim(self):
        bm, bn, bk, wm, wn, stages, ctas = 64, 384, 32, 64, 64, 4, 2
        stage = sum(
            math.ceil(x / 128) * 128 for x in [bm * bk * 2, bn * bk // 2, bn * 2]
        )
        smem = math.ceil(max(stages * stage, bm * bn * 2) / 1024) * 1024 + bm * 8
        threads = (bm // wm) * (bn // wn) * 32
        self.assertEqual((threads, smem), (192, 49664))
        self.assertLessEqual(smem * ctas, 99 * 1024)
        self.assertLessEqual(threads * ctas, 1536)
        self.assertEqual(bm * bn // threads, 128)  # accumulator registers alone

    def test_real_dispatch_guard_with_cpu_doubles(self):
        tree = ast.parse((EXPERTS / "humming_tuned_moe.py").read_text())
        names = {"_cfg", "_to_humming_list", "tuned_moe_configs"}
        constants = {"_BIG", "TUNED_CONFIGS", "TUNED_LAYER_SHAPES", "TUNED_NUM_EXPERTS"}
        nodes = [
            n
            for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in names)
            or (
                isinstance(n, ast.Assign)
                and any(getattr(t, "id", "") in constants for t in n.targets)
            )
            or (
                isinstance(n, ast.AnnAssign)
                and getattr(n.target, "id", "") in constants
            )
        ]
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        ns = {
            "widen_prefill_w2_n": module.widen_prefill_w2_n,
            "humming_moe_fast_enabled": lambda: True,
            "_meta_matches": lambda *_: True,
            "logger": SimpleNamespace(
                info_once=lambda *_: None, warning_once=lambda *_: None
            ),
        }
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[future, *nodes], type_ignores=[])
                ),
                "<real-dispatch>",
                "exec",
            ),
            ns,
        )
        layer = SimpleNamespace(
            humming_tuned_moe=True, humming_metas={"w13": None, "w2": None}
        )
        for sm, expected in [(121, 384), (120, 128), (90, 128)]:
            ns["current_platform"] = SimpleNamespace(
                is_device_capability=lambda x: x == sm
            )
            _, down = ns["tuned_moe_configs"](layer)
            self.assertEqual(select(down, 2976)["block_shape"][1], expected)
        ns["_meta_matches"] = lambda *_: False
        self.assertIsNone(ns["tuned_moe_configs"](layer))
        layer.humming_tuned_moe = False
        self.assertIsNone(ns["tuned_moe_configs"](layer))


if __name__ == "__main__":
    unittest.main()

"""CPU checks of production ASTs; no Humming/Triton compilation or GPU claims."""

import ast
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPERTS = ROOT / "vllm/vllm/model_executor/layers/fused_moe/experts"
SOURCE = EXPERTS / "humming_prefill_rows.py"


def function(path, name, ns):
    tree = ast.parse(path.read_text())
    if name == "apply":
        tree = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "HummingIndexedExperts"
        )
    node = copy.deepcopy(
        next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
    )
    node.decorator_list, node.returns = [], None
    for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
        arg.annotation = None
    node.body = [
        n for n in node.body if not isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            str(path),
            "exec",
        ),
        ns,
    )
    return ns[name]


class Value(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype).view(Value)


def val(x):
    return np.asarray(x).view(Value)


class Ptr:
    def __init__(self, data, offset=0):
        self.data, self.offset = data.reshape(-1), np.asarray(offset)

    def __add__(self, offset):
        return Ptr(self.data, self.offset + offset)


class TL:
    int32 = np.int32
    block = 0

    def program_id(self, axis):
        assert axis == 0
        return self.block

    def arange(self, start, end):
        return val(np.arange(start, end))

    def sum(self, data, axis):
        return val(np.sum(data, axis=axis, dtype=np.int32))

    def load(self, ptr, mask=True, other=0):
        offsets, mask = np.broadcast_arrays(ptr.offset, np.asarray(mask))
        active = offsets[mask]
        assert np.all(active >= 0) and np.all(active < ptr.data.size), "OOB load"
        result = np.full(offsets.shape, other, dtype=ptr.data.dtype)
        result[mask] = ptr.data[active]
        return val(result)

    def store(self, ptr, data, mask=True):
        offsets, data, mask = np.broadcast_arrays(ptr.offset, data, mask)
        active = offsets[mask]
        assert np.all(active >= 0) and np.all(active < ptr.data.size), "OOB store"
        ptr.data[active] = data[mask]


class Tensor:
    def __init__(self, data, dtype=None, device="cuda:0"):
        self.data = np.asarray(data, dtype=dtype)
        self.device, self.is_cuda = device, device.startswith("cuda")
        self.shape, self.ndim, self.dtype = (
            self.data.shape,
            self.data.ndim,
            self.data.dtype,
        )
        self.contiguous = True

    def numel(self):
        return self.data.size

    def is_contiguous(self):
        return self.contiguous

    def size(self, axis):
        return self.shape[axis]

    def view(self, *shape):
        return Tensor(self.data.reshape(*shape), device=self.device)

    def __getitem__(self, index):
        return Tensor(self.data[index], device=self.device)


TORCH = NS(
    int32=np.dtype("int32"),
    bfloat16=np.dtype("float32"),
    empty=lambda size, dtype, device: Tensor(
        np.full(size, -777, dtype=dtype), device=device
    ),
    empty_like=lambda t: Tensor(np.full_like(t.data, -777), device=t.device),
)


def config_tables():
    tree = ast.parse((EXPERTS / "humming_tuned_moe.py").read_text())
    wanted = {"_BIG", "TUNED_CONFIGS"}
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_cfg", "_to_humming_list")
        or isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in wanted for t in n.targets)
        or isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id in wanted
    ]
    ns = {"Any": object}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<base configs>", "exec"), ns)
    tables = ns["TUNED_CONFIGS"]
    return tuple(
        ns["_to_humming_list"](tables[key])
        for key in (("w13", 1856, 2688, 128), ("w2", 2688, 1856, 128))
    )


def make_environment():
    tl = TL()
    kernel = function(SOURCE, "_partition_prefill_rows", {"tl": tl})
    signature = inspect.signature(kernel)

    class Launch:
        def __getitem__(self, grid):
            def invoke(**kwargs):
                assert kwargs.pop("num_warps") == 4
                signature.bind(**kwargs)
                kw = {
                    k: Ptr(v.data) if isinstance(v, Tensor) else v
                    for k, v in kwargs.items()
                }
                for block in range(grid[0]):
                    tl.block = block
                    kernel(**kw)

            return invoke

    ns = {
        "torch": TORCH,
        "triton": NS(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
        "_partition_prefill_rows": Launch(),
        "json": json,
        "copy": copy,
    }
    make = function(SOURCE, "make_prefill_row_configs", ns)
    split = function(SOURCE, "split_prefill_row_plans", ns)
    guard = function(SOURCE, "prefill_row_split_supported", ns)
    return make, split, guard


def aligned(counts):
    routes = sum(counts)
    rows, experts, reference = [], [], {}
    cursor = 0
    for expert, n in enumerate(counts):
        ids = list(range(cursor, cursor + n))
        reference.update({i: expert for i in ids})
        cursor += n
        padded = ((n + 63) // 64) * 64
        rows.extend(ids + [routes] * (padded - n))
        experts.extend([expert] * (padded // 64))
    active = len(rows)
    capacity = len(experts) + 3
    rows += [-999] * (capacity * 64 - len(rows))
    experts += [-999] * (capacity - len(experts))
    common = dict(
        sorted_ids=Tensor(rows, np.int32),
        expert_ids=Tensor(experts, np.int32),
        num_tokens_padded=Tensor([active], np.int32),
        compute_config="unchanged",
        valid_shape_m=routes,
    )
    return (
        dict(common, top_k=6, tuning_config="large-up"),
        dict(common, top_k=1, tuning_config="large-down"),
        reference,
    )


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.make, self.split, self.guard = make_environment()
        self.narrow = self.make(*config_tables())

    def check_counts(self, counts):
        up, down, reference = aligned(counts)
        old_rows = up["sorted_ids"].data.copy()
        old_blocks = int(up["num_tokens_padded"].data[0]) // 64
        plans = self.split(up, down, sum(counts), self.narrow)
        actual, blocks = {}, 0
        for pair, bm in zip(plans, (32, 64)):
            a, b = pair
            self.assertIs(a["sorted_ids"], b["sorted_ids"])
            self.assertIs(a["num_tokens_padded"], b["num_tokens_padded"])
            self.assertEqual((a["top_k"], b["top_k"]), (6, 1))
            self.assertEqual(a["compute_config"], "unchanged")
            count = int(a["num_tokens_padded"].data[0])
            self.assertEqual(count % bm, 0)
            blocks += count // bm
            for block in range(count // bm):
                expert = int(a["expert_ids"].data[block])
                for row in a["sorted_ids"].data[block * bm : (block + 1) * bm]:
                    if row < sum(counts):
                        self.assertNotIn(int(row), actual)
                        actual[int(row)] = expert
            self.assertTrue((a["expert_ids"].data[count // bm :] == -777).all())
        self.assertEqual(actual, reference)
        self.assertEqual(blocks, old_blocks, "each old weight tile is visited once")
        np.testing.assert_array_equal(up["sorted_ids"].data, old_rows)
        self.assertEqual(up["tuning_config"], "large-up")
        self.assertEqual(plans[1][0]["tuning_config"], "large-up")
        self.assertEqual(plans[0][0]["tuning_config"], self.narrow[0])
        return plans

    def test_empty_metadata(self):
        self.check_counts([])

    def test_all_small_and_empty_large_plan(self):
        p = self.check_counts([1, 16, 32])
        self.assertEqual(int(p[1][0]["num_tokens_padded"].data[0]), 0)

    def test_all_large_and_empty_small_plan(self):
        p = self.check_counts([33, 64, 128])
        self.assertEqual(int(p[0][0]["num_tokens_padded"].data[0]), 0)

    def test_32_33_64_65_96_97_boundaries(self):
        self.check_counts([0, 1, 32, 33, 64, 65, 96, 97, 128])

    def test_496_rows_balanced(self):
        p = self.check_counts([24] * 32 + [23] * 96)
        self.assertEqual(int(p[0][0]["num_tokens_padded"].data[0]), 4096)
        self.assertEqual(int(p[1][0]["num_tokens_padded"].data[0]), 0)

    def test_hot_experts_keep_weight_tile_count(self):
        self.check_counts([496] * 6 + [0] * 122)

    def test_many_distributions(self):
        rng = np.random.default_rng(19)
        for _ in range(20):
            with self.subTest(case=_):
                self.check_counts(
                    rng.multinomial(2976, rng.dirichlet(np.ones(128))).tolist()
                )


class ConfigGuardTests(unittest.TestCase):
    def setUp(self):
        self.make, self.split, self.guard = make_environment()
        self.tables = config_tables()
        self.narrow = self.make(*self.tables)

    def test_config_preserves_k_stages_ctas_and_source(self):
        before = copy.deepcopy(self.tables)
        for narrow, table in zip(self.narrow, self.tables):
            cfg = json.loads(narrow)[0][2]
            old = table[-1][2]
            self.assertEqual(cfg["block_shape"], [32, old["block_shape"][1], 32])
            self.assertEqual(cfg["warp_shape"], [16, old["warp_shape"][1], 32])
            for k in old:
                if k not in ("block_shape", "warp_shape"):
                    self.assertEqual(cfg[k], old[k])
        self.assertEqual(self.tables, before)

    def test_changed_config_falls_back(self):
        self.assertIsNone(self.make([], self.tables[1]))
        self.assertIsNone(self.make(self.tables[0], []))
        for key, item in [
            ("use_stream_k", True),
            ("use_f16_accum", True),
            ("num_write_splits", 2),
            ("block_shape", [32, 64, 32]),
            ("warp_shape", [64, 32, 32]),
        ]:
            table = copy.deepcopy(self.tables)
            table[0][-1][2][key] = item
            self.assertIsNone(self.make(*table))

    def check(self, m=496, **changes):
        ids = Tensor(np.zeros((m, 6), np.int32))
        x = Tensor(np.zeros((m, 2688), np.float32))
        args = dict(
            topk_ids=ids,
            hidden_states=x,
            expert_map=None,
            global_experts=128,
            local_experts=128,
            narrow_configs=self.narrow,
            valid_shape_m=m * 6,
        )
        args.update(changes)
        return self.guard(**args)

    def test_prefill_bounds(self):
        for m in (1, 4, 8, 64, 1025):
            self.assertFalse(self.check(m))
        for m in (65, 496, 1024):
            self.assertTrue(self.check(m))

    def test_parallel_and_disabled_fallback(self):
        for c in (
            dict(expert_map=object()),
            dict(local_experts=64),
            dict(global_experts=64),
            dict(valid_shape_m=2975),
            dict(narrow_configs=None),
        ):
            self.assertFalse(self.check(**c))

    def test_dtype_device_shape_and_stride_fallback(self):
        ids = Tensor(np.zeros((496, 6), np.int32))
        for k, v in [
            ("is_cuda", False),
            ("dtype", np.dtype("int64")),
            ("contiguous", False),
            ("shape", (496, 5)),
            ("ndim", 1),
        ]:
            old = getattr(ids, k)
            setattr(ids, k, v)
            self.assertFalse(self.check(topk_ids=ids))
            setattr(ids, k, old)
        self.assertFalse(
            self.check(hidden_states=Tensor(np.zeros((496, 2688), np.float16)))
        )
        self.assertFalse(
            self.check(
                hidden_states=Tensor(np.zeros((496, 2688), np.float32), device="cuda:1")
            )
        )


class PipelineTests(unittest.TestCase):
    def test_real_apply_order_and_disjoint_output_ownership(self):
        make, split, _ = make_environment()
        narrow = make(*config_tables())
        counts = [1, 32, 33, 64, 65, 96, 97]
        up, down, _ = aligned(counts)
        routes = sum(counts)
        # Route count is divisible by six for the full apply API.
        pad = (-routes) % 6
        counts[-1] += pad
        up, down, ownership = aligned(counts)
        routes = sum(counts)
        m = routes // 6
        ids = np.array([ownership[i] for i in range(routes)], np.int32).reshape(m, 6)
        rng = np.random.default_rng(27)
        x = Tensor(rng.standard_normal((m, 4)).astype(np.float32))
        weights = Tensor(rng.random((m, 6)).astype(np.float32))
        wu = rng.standard_normal((len(counts), 4, 3)).astype(np.float32)
        wd = rng.standard_normal((len(counts), 3, 4)).astype(np.float32)
        outputs = []
        orders = []
        for enabled in (False, True):
            trace = []
            gate = Tensor(np.full((routes, 3), np.nan, np.float32))
            act = Tensor(np.full((routes, 3), np.nan, np.float32))
            out = Tensor(np.full((routes, 4), np.nan, np.float32))
            result = Tensor(np.empty((m, 4), np.float32))

            def gemm(**kw):
                layer = kw["sublayer_name"]
                trace.append(layer)
                bm = 32 if kw["tuning_config"] in narrow else 64
                n = int(kw["num_tokens_padded"].data[0])
                for bi in range(n // bm):
                    expert = int(kw["expert_ids"].data[bi])
                    for row in kw["sorted_ids"].data[bi * bm : (bi + 1) * bm]:
                        if row < routes:
                            kw["outputs"].data[row] = (
                                kw["inputs"].data[row // kw["top_k"]]
                                @ (wu if layer == "w13" else wd)[expert]
                            )

            def activate(**kw):
                trace.append("activation")
                kw["output"].data[:] = np.maximum(kw["input"].data, 0) ** 2

            def reduce(**kw):
                trace.append("reduce")
                acc = np.zeros((m, 4), np.float32)
                for k in range(6):
                    acc += (
                        kw["inputs"].data[:, k, :] * kw["topk_weights"].data[:, k, None]
                    )
                kw["outputs"].data[:] = acc

            ns = dict(
                HummingMethod=NS(
                    may_quant_input=lambda **kw: (kw["inputs"], None),
                    forward_layer=gemm,
                ),
                prefill_row_split_supported=lambda *a: enabled,
                split_prefill_row_plans=split,
                moe_fused_mul_sum=reduce,
            )
            apply = function(EXPERTS / "fused_humming_moe.py", "apply", ns)
            obj = NS(
                layer=None,
                global_num_experts=128,
                num_experts=128,
                _prefill_row_configs=narrow,
                prepare_buffers=lambda *a: dict(
                    gate_up_output=gate, activation_output=act, down_output=out
                ),
                prepare_humming_moe_kwargs=lambda **kw: (up, down),
                apply_activation=activate,
            )
            apply(
                obj,
                result,
                x,
                None,
                None,
                weights,
                Tensor(ids),
                None,
                128,
                None,
                None,
                None,
                None,
                None,
                None,
                False,
            )
            self.assertTrue(
                np.isfinite(gate.data).all() and np.isfinite(out.data).all()
            )
            outputs.append(result.data.copy())
            orders.append(trace)
        np.testing.assert_array_equal(
            outputs[0].view(np.uint32), outputs[1].view(np.uint32)
        )
        self.assertEqual(orders[0], ["w13", "activation", "w2", "reduce"])
        self.assertEqual(orders[1], ["w13", "w13", "activation", "w2", "w2", "reduce"])


if __name__ == "__main__":
    unittest.main()

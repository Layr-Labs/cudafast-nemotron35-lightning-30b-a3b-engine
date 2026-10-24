"""CPU doubles for staging lifetime, not FlashInfer/CUDA execution."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "vllm/vllm/v1/spec_decode/draft_plan_banks.py"
PROPOSER = ROOT / "vllm/vllm/v1/spec_decode/llm_base_proposer.py"


class Event:
    def __init__(self):
        self.complete = False
        self.records = self.queries = self.waits = 0

    def query(self):
        self.queries += 1
        return self.complete

    def synchronize(self):
        self.waits += 1
        self.complete = True

    def record(self):
        self.records += 1
        self.complete = False


class Builder:
    def __init__(self, spec, names, config, device):
        self.kv_cache_spec = spec
        self.names = names
        self.use_dcp = self.is_kvcache_nvfp4 = False
        self.flashinfer_trtllm_api_decode_kernel = None
        self.staging = {}
        self.workspace = object()
        self.calls = []
        self.fail = False

    def _get_workspace_buffer(self):
        return self.workspace

    def set_workspace_buffer(self, workspace):
        self.workspace = workspace

    def build_for_drafting(self, common_attn_metadata, draft_index):
        self.calls.append(draft_index)
        # Model queued H2D use: the returned object retains this exact staging
        # dictionary, which must not be overwritten by a different stage.
        self.staging["length"] = common_attn_metadata.length
        self.staging["index"] = draft_index
        if self.fail:
            raise RuntimeError("plan failed after H2D")
        return self.staging


class PlanBankTests(unittest.TestCase):
    def setUp(self):
        self.stream = object()
        self.events = []

        def event():
            value = Event()
            self.events.append(value)
            return value

        self.torch = NS(cuda=NS(Event=event, current_stream=lambda device: self.stream))
        tree = ast.parse(SOURCE.read_text())
        tree.body = [
            n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))
        ]
        self.namespace = {"torch": self.torch}
        exec(compile(tree, str(SOURCE), "exec"), self.namespace)
        self.klass = self.namespace["DraftPlanBanks"]
        self.builder = Builder("block-spec", ["layer.0", "layer.1"], "config", "cuda:0")
        self.groups = [
            NS(
                layer_names=self.builder.names,
                get_metadata_builder=lambda: self.builder,
            )
        ]
        modules = patch.dict(
            sys.modules,
            {
                "vllm.v1.attention.backends.flashinfer": NS(
                    FlashInferMetadataBuilder=Builder
                )
            },
        )
        modules.start()
        self.addCleanup(modules.stop)

    def create(self, depth=3):
        return self.klass.create(self.groups, "config", "cuda:0", depth)

    def test_distinct_pinned_storage_shared_scratch_and_no_interstage_wait(self):
        banks = self.create()
        results = []
        for index, length in enumerate((128, 126, 127)):
            per_group, per_layer = banks.build(NS(length=length), index)
            results.append(per_group[0])
            self.assertIs(per_layer["layer.0"], per_layer["layer.1"])
        self.assertEqual([r["length"] for r in results], [128, 126, 127])
        self.assertEqual(len({id(r) for r in results}), 3)
        self.assertTrue(
            all(stage[0].workspace is self.builder.workspace for stage in banks._stages)
        )
        self.assertTrue(all(e.waits == 0 and e.records == 1 for e in self.events))
        self.assertEqual(len({id(stage[0].staging) for stage in banks._stages}), 3)

    def test_pending_previous_request_fenced_before_overwrite(self):
        banks = self.create()
        banks.build(NS(length=31), 1)
        event = self.events[0]
        builder = banks._stages[1][0]
        original_build = builder.build_for_drafting

        def guarded(**kwargs):
            self.assertEqual(event.waits, 1)
            return original_build(**kwargs)

        builder.build_for_drafting = guarded
        banks.build(NS(length=400), 1)
        self.assertEqual((event.waits, event.queries, event.records), (1, 1, 2))
        self.assertEqual(builder.staging["length"], 400)

    def test_completed_bank_reuse_queries_without_synchronize(self):
        banks = self.create()
        banks.build(NS(length=32), 2)
        event = self.events[0]
        event.complete = True
        banks.build(NS(length=33), 2)
        self.assertEqual((event.queries, event.waits, event.records), (1, 0, 2))

    def test_plan_exception_records_pending_work_before_propagating(self):
        banks = self.create()
        banks._stages[1][0].fail = True
        with self.assertRaisesRegex(RuntimeError, "after H2D"):
            banks.build(NS(length=129), 1)
        self.assertEqual(self.events[0].records, 1)
        banks._stages[1][0].fail = False
        banks.build(NS(length=130), 1)
        self.assertEqual(self.events[0].waits, 1)

    def test_unexpected_depth_uses_fenced_bank_zero_without_allocating(self):
        banks = self.create(2)
        banks.build(NS(length=127), 0)
        banks.build(NS(length=128), 5)
        self.assertEqual(len(banks._stages), 2)
        self.assertEqual(len(self.events), 1)
        self.assertEqual(self.events[0].waits, 1)
        self.assertEqual(self.builder.calls, [0, 5])

    def test_changed_stream_fails_before_plan_or_buffer_mutation(self):
        banks = self.create()
        banks.build(NS(length=20), 0)
        self.stream = object()
        with self.assertRaisesRegex(RuntimeError, "one owning stream"):
            banks.build(NS(length=90), 0)
        self.assertEqual(self.builder.staging["length"], 20)
        self.assertEqual(self.events[0].records, 1)

    def test_creation_guards_and_every_supported_depth(self):
        for depth in (0, 1, 7):
            self.assertIsNone(self.create(depth))
        for depth in range(2, 7):
            self.assertEqual(len(self.create(depth)._stages), depth)
        for key, value in [
            ("use_dcp", True),
            ("is_kvcache_nvfp4", True),
            ("flashinfer_trtllm_api_decode_kernel", "TRTLLM"),
        ]:
            old = getattr(self.builder, key)
            setattr(self.builder, key, value)
            self.assertIsNone(self.create())
            setattr(self.builder, key, old)
        self.assertIsNone(self.klass.create([], "config", "cuda:0", 3))
        # Attribute names are checked against the vendored builder source.
        source = (ROOT / "vllm/vllm/v1/attention/backends/flashinfer.py").read_text()
        for field in (
            "self.use_dcp",
            "self.is_kvcache_nvfp4",
            "self.flashinfer_trtllm_api_decode_kernel",
        ):
            self.assertIn(field, source)

    def test_proposer_routes_banks_or_original_staging_guard(self):
        tree = ast.parse(PROPOSER.read_text())
        # Non-Nemotron subclasses can override initialize_attn_backend;
        # inherited planning must still have a safe disabled bank default.
        base_class = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "SpecDecodeBaseProposer"
        )
        default = next(
            n
            for n in base_class.body
            if isinstance(n, ast.AnnAssign) and n.target.id == "_draft_plan_banks"
        )
        self.assertIsNone(default.value.value)
        function = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and n.name == "build_per_group_and_layer_attn_metadata"
        )
        ns = dict(self.namespace, CommonAttentionMetadata=object)
        exec(
            compile(
                ast.Module(body=[function], type_ignores=[]), str(PROPOSER), "exec"
            ),
            ns,
        )
        build = ns[function.name]
        events = []
        guard = NS(
            before=lambda: events.append("before"), after=lambda: events.append("after")
        )
        p = NS(
            _draft_plan_banks=self.create(),
            _hostfast_staging_guard=lambda: guard,
            draft_attn_groups=self.groups,
        )
        build(p, NS(length=30), 1)
        self.assertFalse(events)
        p._draft_plan_banks = None
        build(p, NS(length=31), 1)
        self.assertEqual(events, ["before", "after"])
        # The old planner fence is retained only on the shared-builder path.
        propose = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "propose"
        )
        fence = next(
            n
            for n in ast.walk(propose)
            if isinstance(n, ast.If)
            and any(
                isinstance(x, ast.Call)
                and isinstance(x.func, ast.Attribute)
                and x.func.attr == "synchronize"
                and isinstance(x.func.value, ast.Attribute)
                and x.func.value.attr == "_draft_plan_event"
                for statement in n.body
                for x in ast.walk(statement)
            )
            and "_draft_plan_banks" in ast.unparse(n.test)
        )
        condition = compile(ast.Expression(fence.test), "<plan fence>", "eval")
        for active in (None, object()):
            p._draft_plan_banks = active
            self.assertEqual(
                eval(
                    condition, {"self": p, "order_draft_plans": True, "token_index": 1}
                ),
                active is None,
            )


if __name__ == "__main__":
    unittest.main()

"""CPU source-level checks; no vLLM, model, subprocess or GPU is started.

The real resident constructor and in-process client methods are extracted
with AST. Small doubles isolate configuration and synchronous step ordering.
These checks cannot establish CUDA initialization or numerical equivalence.
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def extract(path, class_name, methods, namespace):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == class_name)
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in methods]
    source = "from __future__ import annotations\n" + ast.unparse(cls)
    exec(compile(source, path, "exec"), namespace)
    return namespace[class_name]


def module(name, **attrs):
    result = ModuleType(name)
    result.__dict__.update(attrs)
    return result


class InprocEngineTests(unittest.TestCase):
    def test_constructor_preserves_pins_at_serial_and_every_spec_depth(self):
        for depth in range(7):
            with self.subTest(depth=depth):
                calls = []
                context, config, executor = object(), object(), object()

                class Args:
                    def __init__(self, **kwargs):
                        self.kwargs = kwargs
                        self.disable_log_stats = kwargs["disable_log_stats"]
                        calls.append(("args", kwargs))

                    def create_engine_config(self, usage_context):
                        calls.append(("config", usage_context))
                        return config

                class Engine:
                    def __init__(self, **kwargs):
                        calls.append(("engine", kwargs))
                        self.model_config = NS(hf_config=NS(
                            vocab_size=131072, eos_token_id=[2, 11]))

                    @classmethod
                    def from_engine_args(cls, *args, **kwargs):
                        raise AssertionError("multiprocessing factory was used")

                def get_executor(actual_config):
                    self.assertIs(actual_config, config)
                    calls.append(("executor", actual_config))
                    return executor

                mods = {
                    "vllm": module("vllm", EngineArgs=Args, LLMEngine=Engine,
                                   SamplingParams=object),
                    "vllm.sampling_params": module("vllm.sampling_params",
                        RequestOutputKind=NS(DELTA="delta")),
                    "vllm.usage.usage_lib": module("vllm.usage.usage_lib",
                        UsageContext=NS(ENGINE_CONTEXT=context)),
                    "vllm.v1.executor.abstract": module("vllm.v1.executor.abstract",
                        Executor=NS(get_class=get_executor)),
                    "vllm.v1.worker.hostfast": module("vllm.v1.worker.hostfast",
                        pin_to_fast_cpus=lambda: calls.append(("pin", None)) or []),
                }
                backend_type = extract("harness/vllm_resident/backends.py",
                    "VllmBackend", {"__init__"}, {"TOP_K": 8})
                backend_type._warm_up = lambda self: calls.append(("warmup", None))
                cfg = NS(model_dir="/synthetic/model", depth=depth, ctx_size=8192,
                         max_num_batched_tokens=8192, gpu_memory_utilization=0.45,
                         kv_cache_memory_bytes=4294967296, ident="test")
                with patch.dict(sys.modules, mods):
                    backend = backend_type(cfg)
                self.assertEqual([c[0] for c in calls],
                                 ["args", "config", "executor", "engine", "pin", "warmup"])
                self.assertIs(calls[1][1], context)
                self.assertEqual(calls[3][1], dict(vllm_config=config,
                    executor_class=executor, log_stats=False,
                    usage_context=context, multiprocess_mode=False))
                args = calls[0][1]
                self.assertEqual(args["max_num_seqs"], 1)
                self.assertFalse(args["enable_prefix_caching"])
                self.assertEqual(args["mamba_ssm_cache_dtype"], "float32")
                self.assertEqual(args["kv_cache_memory_bytes"], 4294967296)
                self.assertEqual(args["moe_backend"], "humming")
                self.assertEqual(args["logprobs_mode"], "raw_logits")
                self.assertEqual(args["seed"], 0)
                self.assertEqual(args["max_logprobs"], 8)
                self.assertEqual(args["speculative_config"],
                    {"method": "mtp", "num_speculative_tokens": depth} if depth else None)
                self.assertEqual(backend.depth, depth)
                self.assertEqual(backend.eos_token, 2)

    def test_client_factory_chooses_inproc_without_changing_async_mode(self):
        sentinel = object()
        calls = []

        def inproc(*args):
            calls.append(args)
            return sentinel

        cls = extract("vllm/vllm/v1/engine/core_client.py", "EngineCoreClient",
                      {"make_client"}, {"InprocClient": inproc})
        self.assertIs(cls.make_client(False, False, "config", "executor", False),
                      sentinel)
        self.assertEqual(calls, [("config", "executor", False)])
        with self.assertRaises(NotImplementedError):
            cls.make_client(False, True, "config", "executor", False)

    def test_step_calls_post_step_before_exposing_output(self):
        calls = []
        cls = extract("vllm/vllm/v1/engine/core_client.py", "InprocClient",
                      {"get_output", "_hostfast_idle_if_finished"}, {
                          "EngineCoreOutputs": lambda: "empty",
                          "hostfast": NS(gc_busy=lambda: calls.append("busy"),
                                         gc_idle=lambda: calls.append("idle")),
                      })
        for outputs, executed, expected in [({0: "tokens"}, True, "tokens"),
                                             ({}, False, "empty"),
                                             (None, False, "empty")]:
            calls.clear()

            def step():
                calls.append("step")
                return outputs, executed

            client = cls()
            client.engine_core = NS(step_fn=step, post_step=lambda **kw: calls.append(kw),
                                   scheduler=NS(has_requests=lambda: False), batch_queue=[])
            self.assertEqual(client.get_output(), expected)
            self.assertEqual(calls, ["busy", "step", {"model_executed": executed}, "idle"])

    def test_abort_is_direct_and_empty_abort_does_not_abort_the_core(self):
        calls = []
        cls = extract("vllm/vllm/v1/engine/core_client.py", "InprocClient",
                      {"abort_requests", "add_request", "_hostfast_idle_if_finished"},
                      {"hostfast": NS(gc_idle=lambda: calls.append(("idle",)))})
        client = cls()
        client.engine_core = NS(
            abort_requests=lambda ids: calls.append(("abort", ids)),
            preprocess_add_request=lambda req: (req, 9),
            add_request=lambda req, wave: calls.append(("add", req, wave)),
            scheduler=NS(has_requests=lambda: False,
                         has_unfinished_requests=lambda: False), batch_queue=[],
        )
        client.add_request("r1")
        client.abort_requests([])
        client.abort_requests(["r1"])
        self.assertEqual(calls, [("add", "r1", 9), ("idle",),
                                ("abort", ["r1"]), ("idle",)])


class InprocGcTests(unittest.TestCase):
    def client(self, enabled=True, gc_switch=True):
        events = []
        gc_state = NS(enabled=enabled)

        def toggle(value):
            gc_state.enabled = value
            events.append("enable" if value else "disable")

        # Use the promoted base's actual ownership-aware GC helpers. Only the
        # Python gc module is doubled, so no process-global GC state changes.
        path = ROOT / "vllm/vllm/v1/worker/hostfast.py"
        nodes = [n for n in ast.parse(path.read_text()).body
                 if isinstance(n, ast.FunctionDef)
                 and n.name in {"gc_busy", "gc_idle", "gc_in_shadow"}]
        ns = dict(GC=gc_switch, _gc_paused=False, gc=NS(
            isenabled=lambda: gc_state.enabled,
            disable=lambda: toggle(False), enable=lambda: toggle(True),
            collect=lambda generation: events.append(("collect", generation))))
        exec(compile(ast.unparse(ast.Module(body=nodes, type_ignores=[])),
                     str(path), "exec"), ns)
        hostfast = NS(**{n.name: ns[n.name] for n in nodes})
        cls = extract("vllm/vllm/v1/engine/core_client.py", "InprocClient",
                      {"get_output", "abort_requests", "shutdown",
                       "_hostfast_idle_if_finished"},
                      {"hostfast": hostfast, "EngineCoreOutputs": lambda: "empty"})
        state = NS(active=True)
        client = cls()
        client.engine_core = NS(
            scheduler=NS(has_requests=lambda: state.active,
                         has_unfinished_requests=lambda: state.active), batch_queue=[],
            step_fn=lambda: ({0: "tokens"}, True), post_step=lambda **kw: None,
            abort_requests=lambda ids: setattr(state, "active", False),
            shutdown=lambda: None)
        return client, state, gc_state, events, hostfast

    def test_busy_request_keeps_pause_and_completion_releases_it(self):
        client, state, gc_state, events, hostfast = self.client()
        self.assertEqual(client.get_output(), "tokens")
        self.assertFalse(gc_state.enabled)
        self.assertEqual(events, ["disable"])
        hostfast.gc_in_shadow()
        self.assertEqual(events[-1], ("collect", 1))
        client.get_output()
        self.assertEqual(events.count("disable"), 1)
        state.active = False
        client.get_output()
        self.assertTrue(gc_state.enabled)
        self.assertEqual(events[-1], "enable")

    def test_pending_batch_keeps_pause_even_without_scheduler_requests(self):
        client, state, gc_state, events, _ = self.client()
        state.active = False
        client.engine_core.batch_queue.append(object())
        client.engine_core.step_fn = lambda: ({}, False)
        self.assertEqual(client.get_output(), "empty")
        self.assertFalse(gc_state.enabled)
        client.engine_core.batch_queue.clear()
        client.get_output()
        self.assertTrue(gc_state.enabled)
        self.assertEqual(events, ["disable", "enable"])

    def test_step_and_post_step_errors_or_interrupts_restore_gc(self):
        for site in ("step_fn", "post_step"):
            for exc_type in (RuntimeError, KeyboardInterrupt):
                client, _, gc_state, events, _ = self.client()

                def fail(*args, **kwargs):
                    raise exc_type("synthetic failure")

                setattr(client.engine_core, site, fail)
                with self.assertRaises(exc_type):
                    client.get_output()
                self.assertTrue(gc_state.enabled)
                self.assertEqual(events, ["disable", "enable"])

    def test_abort_drains_pending_batch_before_releasing_gc(self):
        for pending in (False, True):
            client, _, gc_state, events, _ = self.client()
            client.get_output()
            if pending:
                client.engine_core.batch_queue.append(object())

                def drain():
                    self.assertFalse(gc_state.enabled)
                    client.engine_core.batch_queue.pop()
                    events.append("drain")
                    return {}, False

                client.engine_core.step_fn = drain
            client.abort_requests(["r1"])
            self.assertTrue(gc_state.enabled)
            self.assertFalse(client.engine_core.batch_queue)
            self.assertEqual(events, ["disable", "drain", "enable"] if pending
                             else ["disable", "enable"])

    def test_empty_abort_after_completion_drains_queue_and_finished_cleanup(self):
        client, state, gc_state, events, _ = self.client()
        client.get_output()
        state.active = False
        cleanup = [True]
        client.engine_core.scheduler.has_requests = lambda: cleanup[0]
        client.engine_core.batch_queue.extend([object(), object()])

        def drain():
            self.assertFalse(gc_state.enabled)
            if cleanup[0]:
                cleanup[0] = False
                events.append("cleanup")
            else:
                client.engine_core.batch_queue.pop()
                events.append("drain")
            return {}, False

        client.engine_core.step_fn = drain
        client.abort_requests([])
        self.assertEqual(events, ["disable", "cleanup", "drain", "drain", "enable"])
        self.assertFalse(client.engine_core.batch_queue)
        self.assertTrue(gc_state.enabled)

    def test_empty_abort_with_another_live_request_does_not_step_it(self):
        client, _, gc_state, events, _ = self.client()
        client.get_output()
        client.engine_core.batch_queue.append(object())

        def forbidden():
            self.fail("abort advanced another live request")

        client.engine_core.step_fn = forbidden
        client.abort_requests([])
        self.assertEqual(len(client.engine_core.batch_queue), 1)
        self.assertFalse(gc_state.enabled)
        self.assertEqual(events, ["disable"])

    def test_abort_and_shutdown_failures_still_release_gc(self):
        for method in ("abort_requests", "shutdown"):
            client, _, gc_state, events, _ = self.client()
            client.get_output()

            def fail(*args, **kwargs):
                raise RuntimeError("synthetic failure")

            setattr(client.engine_core, method, fail)
            with self.assertRaises(RuntimeError):
                getattr(client, method)(["r1"]) if method == "abort_requests" else client.shutdown()
            self.assertTrue(gc_state.enabled)
            self.assertEqual(events, ["disable", "enable"])

    def test_external_gc_disable_and_kill_switch_are_preserved(self):
        for enabled, gc_switch in ((False, True), (True, False)):
            client, state, gc_state, events, _ = self.client(enabled, gc_switch)
            client.get_output()
            state.active = False
            client.get_output()
            client.shutdown()
            self.assertEqual(gc_state.enabled, enabled)
            self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()

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
                                 ["args", "config", "executor", "engine", "warmup"])
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
        cls = extract("vllm/vllm/v1/engine/core_client.py", "InprocClient",
                      {"get_output"}, {"EngineCoreOutputs": lambda: "empty"})
        for outputs, executed, expected in [({0: "tokens"}, True, "tokens"),
                                             ({}, False, "empty"),
                                             (None, False, "empty")]:
            calls = []

            def step():
                calls.append("step")
                return outputs, executed

            client = cls()
            client.engine_core = NS(step_fn=step, post_step=lambda **kw: calls.append(kw))
            self.assertEqual(client.get_output(), expected)
            self.assertEqual(calls, ["step", {"model_executed": executed}])

    def test_abort_is_direct_and_empty_abort_does_nothing(self):
        cls = extract("vllm/vllm/v1/engine/core_client.py", "InprocClient",
                      {"abort_requests", "add_request"}, {})
        calls = []
        client = cls()
        client.engine_core = NS(
            abort_requests=lambda ids: calls.append(("abort", ids)),
            preprocess_add_request=lambda req: (req, 9),
            add_request=lambda req, wave: calls.append(("add", req, wave)),
        )
        client.add_request("r1")
        client.abort_requests([])
        client.abort_requests(["r1"])
        self.assertEqual(calls, [("add", "r1", 9), ("abort", ["r1"])])


if __name__ == "__main__":
    unittest.main()

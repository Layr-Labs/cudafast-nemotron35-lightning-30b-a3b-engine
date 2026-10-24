"""CPU checks of native MTP graph sizing and dispatch-key construction.

Execute the actual config methods and enums extracted from their source. No
torch/vLLM package import, CUDA capture, weights or timing is involved. The
configuration objects below supply only fields used by those methods.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import enum
from itertools import product
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
CONFIG = "vllm/vllm/config/vllm.py"
COMPILATION = "vllm/vllm/config/compilation.py"


def extract_class(path, name, namespace, methods=None):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    if methods is not None:
        node.decorator_list = []
        node.bases = []
        node.keywords = []
        node.body = [
            n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods
        ]
        assert {n.name for n in node.body} == set(methods)
    source = "from __future__ import annotations\n" + ast.unparse(node)
    exec(compile(source, path, "exec"), namespace)
    return namespace[name]


ENV = {"enum": enum, "Enum": enum.Enum}
GraphMode = extract_class(COMPILATION, "CUDAGraphMode", ENV)
extract_class(COMPILATION, "CompilationMode", ENV)
AttentionSupport = extract_class(
    "vllm/vllm/v1/attention/backend.py", "AttentionCGSupport", ENV
)
ENV.update(
    logger=NS(warning=lambda *a, **kw: None),
    round_up=lambda n, multiple: (n + multiple - 1) // multiple * multiple,
)
Compilation = extract_class(
    COMPILATION,
    "CompilationConfig",
    ENV,
    {"resolve_cudagraph_mode_and_sizes", "adjust_cudagraph_sizes_for_spec_decode"},
)
Config = extract_class(CONFIG, "VllmConfig", ENV, {"_set_cudagraph_sizes"})


def make_config(
    depth=2,
    *,
    max_seqs=1,
    max_tokens=8192,
    max_capture=None,
    sizes=None,
    method="mtp",
    model_type="nemotron_h",
    eager=False,
    mode=None,
    tp=1,
    sp=False,
):
    cfg = Config()
    cfg.model_config = NS(enforce_eager=eager, hf_config=NS(model_type=model_type))
    cfg.scheduler_config = NS(max_num_seqs=max_seqs, max_num_batched_tokens=max_tokens)
    cfg.parallel_config = NS(tensor_parallel_size=tp)
    cfg.speculative_config = NS(method=method) if depth else None
    cfg.num_speculative_tokens = depth
    cfg.performance_mode = "balanced"
    cfg.update_sizes_for_sequence_parallelism = lambda values: [
        n for n in values if n % tp == 0
    ]
    cc = Compilation()
    cc.cudagraph_capture_sizes = sizes
    cc.max_cudagraph_capture_size = max_capture
    cc.cudagraph_mode = GraphMode.FULL_AND_PIECEWISE if mode is None else mode
    cc.mode = ENV["CompilationMode"].VLLM_COMPILE
    cc.pass_config = NS(enable_sp=sp)
    cc.compile_sizes = None
    cc.post_init_cudagraph_sizes = lambda: None
    cc.splitting_ops_contain_attention = lambda: True
    cfg.compilation_config = cc
    return cfg


def resolve(cfg):
    backend = ModuleType("vllm.v1.attention.backend")
    backend.AttentionCGSupport = AttentionSupport
    with patch.dict(sys.modules, {backend.__name__: backend}):
        return cfg.compilation_config.resolve_cudagraph_mode_and_sizes(
            AttentionSupport.UNIFORM_BATCH,
            "FLASHINFER",
            uniform_decode_query_len=1 + cfg.num_speculative_tokens,
            max_num_reqs=cfg.scheduler_config.max_num_seqs,
            tensor_parallel_size=cfg.parallel_config.tensor_parallel_size,
        )


class NativeMtpGraphSizesTests(unittest.TestCase):
    def test_depth_two_preserves_verify_and_single_row_graphs(self):
        cfg = make_config()
        cfg._set_cudagraph_sizes()
        self.assertEqual(cfg.compilation_config.cudagraph_capture_sizes, [1, 2, 3, 4])
        self.assertEqual(resolve(cfg), GraphMode.FULL_AND_PIECEWISE)
        self.assertEqual(cfg.compilation_config.cudagraph_capture_sizes, [1, 2, 3, 4])

    def test_previous_missing_width_rounds_away_small_drafter_sizes(self):
        cfg = make_config(sizes=[1, 2, 4])
        cfg._set_cudagraph_sizes()
        resolve(cfg)
        # Explicit lists retain their old policy, which also demonstrates the
        # old default's depth-two loss of its one-row drafter key.
        self.assertEqual(cfg.compilation_config.cudagraph_capture_sizes, [3])

    def test_all_permitted_depths_keep_exact_verify_width(self):
        for depth in range(1, 7):
            with self.subTest(depth=depth):
                cfg = make_config(depth=depth)
                cfg._set_cudagraph_sizes()
                resolve(cfg)
                sizes = cfg.compilation_config.cudagraph_capture_sizes
                self.assertIn(1, sizes)
                self.assertIn(depth + 1, sizes)
                self.assertEqual(sizes, sorted(set(sizes)))

    def test_record_depth_three_and_serial_sizes_unchanged(self):
        for depth, expected in [(0, [1, 2]), (3, [1, 2, 4, 8])]:
            cfg = make_config(depth=depth)
            cfg._set_cudagraph_sizes()
            resolve(cfg)
            self.assertEqual(cfg.compilation_config.cudagraph_capture_sizes, expected)

    def test_multiple_requests_keep_default_rounding(self):
        cfg = make_config(max_seqs=2)
        cfg._set_cudagraph_sizes()
        self.assertEqual(cfg.compilation_config.cudagraph_capture_sizes, [1, 2, 4, 8])
        resolve(cfg)
        self.assertEqual(cfg.compilation_config.cudagraph_capture_sizes, [3, 6])

    def test_other_model_and_drafter_keep_original_default(self):
        for kwargs in [dict(model_type="other"), dict(method="eagle")]:
            cfg = make_config(**kwargs)
            cfg._set_cudagraph_sizes()
            self.assertEqual(cfg.compilation_config.cudagraph_capture_sizes, [1, 2, 4])

    def test_explicit_sizes_are_never_augmented(self):
        cfg = make_config(sizes=[4, 2, 1, 2])
        cfg._set_cudagraph_sizes()
        self.assertEqual(cfg.compilation_config.cudagraph_capture_sizes, [1, 2, 4])

    def test_explicit_limit_is_respected_including_too_small(self):
        cfg = make_config(max_capture=3)
        cfg._set_cudagraph_sizes()
        self.assertEqual(cfg.compilation_config.cudagraph_capture_sizes, [1, 2, 3])
        resolve(cfg)
        cfg = make_config(max_capture=2)
        cfg._set_cudagraph_sizes()
        self.assertEqual(cfg.compilation_config.cudagraph_capture_sizes, [1, 2])
        with self.assertRaisesRegex(ValueError, "No valid cudagraph sizes"):
            resolve(cfg)

    def test_max_batched_tokens_caps_graph_size(self):
        cfg = make_config(max_tokens=3)
        cfg._set_cudagraph_sizes()
        self.assertEqual(cfg.compilation_config.max_cudagraph_capture_size, 3)
        self.assertEqual(cfg.compilation_config.cudagraph_capture_sizes, [1, 2, 3])

    def test_tp_and_sequence_parallelism_keep_their_policy(self):
        for sp, expected in [(False, [1, 2, 4]), (True, [2, 4])]:
            cfg = make_config(tp=2, sp=sp)
            cfg._set_cudagraph_sizes()
            self.assertEqual(cfg.compilation_config.cudagraph_capture_sizes, expected)
        with self.assertRaisesRegex(ValueError, "Can't determine cudagraph shapes"):
            resolve(cfg)

    def test_eager_none_and_absent_model_do_not_enable_graphs(self):
        cases = [
            make_config(eager=True),
            make_config(mode=GraphMode.NONE),
            make_config(),
        ]
        cases[-1].model_config = None
        for cfg in cases:
            cfg._set_cudagraph_sizes()
            self.assertEqual(cfg.compilation_config.cudagraph_capture_sizes, [])
            self.assertEqual(cfg.compilation_config.max_cudagraph_capture_size, 0)

    def test_inconsistent_explicit_sizes_still_fail(self):
        cfg = make_config(sizes=[1, 2, 3], max_capture=4)
        with self.assertRaisesRegex(ValueError, "should be consistent"):
            cfg._set_cudagraph_sizes()

    def test_actual_dispatcher_keeps_full_verify_and_piecewise_one_row(self):
        @dataclass(frozen=True)
        class Descriptor:
            num_tokens: int
            uniform: bool
            num_reqs: int | None

        namespace = dict(ENV, product=product, replace=replace)
        dispatcher_type = extract_class(
            "vllm/vllm/v1/cudagraph_dispatcher.py",
            "CudagraphDispatcher",
            namespace,
            {"initialize_cudagraph_keys", "_compute_bs_to_padded_graph_size"},
        )
        cfg = make_config()
        cfg._set_cudagraph_sizes()
        resolve(cfg)
        dispatcher = dispatcher_type()
        dispatcher.compilation_config = cfg.compilation_config
        dispatcher.vllm_config = cfg
        dispatcher._get_lora_cases = lambda: [0]
        dispatcher._create_padded_batch_descriptor = lambda n, uniform, *_: Descriptor(
            n, uniform, 1
        )
        keys = []
        dispatcher.add_cudagraph_key = lambda mode, desc: keys.append((mode, desc))
        dispatcher.initialize_cudagraph_keys(GraphMode.FULL_AND_PIECEWISE, 3)
        self.assertEqual(
            [desc.num_tokens for mode, desc in keys if mode == GraphMode.FULL], [3]
        )
        self.assertIn((GraphMode.PIECEWISE, Descriptor(1, False, None)), keys)
        self.assertEqual(dispatcher._bs_to_padded_graph_size[1:4], [1, 2, 3])
        self.assertTrue(dispatcher.keys_initialized)

    def test_manifest_uses_native_pinned_head_and_permitted_depth(self):
        declaration = json.loads((ROOT / "mtp-head.manifest.json").read_text())
        contract = json.loads(
            (ROOT / "fixtures/nemotron35_lightning_30b_a3b_track.json").read_text()
        )
        self.assertEqual(declaration["source"], "pinned")
        self.assertTrue(declaration["spec"]["enabled"])
        self.assertIn(
            declaration["spec"]["num_speculative_tokens"],
            contract["mtp_head"]["permitted_draft_depths"],
        )


if __name__ == "__main__":
    unittest.main()

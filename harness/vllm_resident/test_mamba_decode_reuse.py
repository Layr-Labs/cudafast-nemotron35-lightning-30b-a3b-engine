"""CPU regressions for same-step, cross-group Mamba2 decode metadata reuse.

The vendored vLLM package requires CUDA extensions to import. AST loading below
executes its actual metadata dataclasses, builder methods and helper functions
without importing that package. Only framework construction/configuration is
mocked; tensor operations use real CPU PyTorch. These tests do not simulate
CUDA execution or establish whole-model throughput.
"""

from __future__ import annotations

import abc
import ast
import dataclasses
import importlib.util
import itertools
from pathlib import Path
import random
from types import SimpleNamespace as NS
from typing import ClassVar, TypeVar
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[2]
FAST_PATH = ROOT / "vllm/vllm/v1/worker/prefill_fast.py"
_spec = importlib.util.spec_from_file_location("mamba_reuse_fast_under_test", FAST_PATH)
fast = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fast)


class FrameworkBuilder:
    """Only vLLM's generic base construction; no metadata algorithms here."""

    def __class_getitem__(cls, item):
        return cls

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device


def load_builders():
    namespace = {
        "__name__": __name__, "abc": abc, "torch": torch,
        "dataclass": dataclasses.dataclass, "replace": dataclasses.replace,
        "ClassVar": ClassVar, "M": TypeVar("M"),
        "AttentionMetadataBuilder": FrameworkBuilder,
        "AttentionCGSupport": NS(UNIFORM_BATCH=2), "NULL_BLOCK_ID": -1,
        "cdiv": lambda a, b: (a + b - 1) // b,
    }

    def compile_nodes(path, nodes):
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        module = ast.Module(body=[future] + nodes, type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)

    def load(relative, names):
        path = ROOT / relative
        nodes = ast.parse(path.read_text()).body
        selected = [n for n in nodes if getattr(n, "name", None) in names]
        if {n.name for n in selected} != set(names):
            raise AssertionError(f"actual source definitions missing: {names}")
        compile_nodes(path, selected)

    # Use the actual framework threshold calculation as well: depth affects
    # classification, so a hand-written replacement would weaken this oracle.
    path = ROOT / "vllm/vllm/v1/attention/backend.py"
    tree = ast.parse(path.read_text())
    framework = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                     and n.name == "AttentionMetadataBuilder")
    threshold = next(n for n in framework.body
                     if getattr(n, "name", None) == "_init_reorder_batch_threshold")
    compile_nodes(path, [threshold])
    FrameworkBuilder._init_reorder_batch_threshold = namespace[
        "_init_reorder_batch_threshold"
    ]

    load("vllm/vllm/v1/attention/backends/utils.py",
         {"split_decodes_and_prefills", "mamba_get_block_table_tensor"})
    load("vllm/vllm/v1/attention/backends/mamba_attn.py",
         {"BaseMambaAttentionMetadata", "BaseMambaAttentionMetadataBuilder"})
    load("vllm/vllm/v1/attention/backends/mamba2_attn.py",
         {"Mamba2AttentionMetadata", "Mamba2AttentionMetadataBuilder"})
    return namespace["Mamba2AttentionMetadataBuilder"]


Builder = load_builders()


def make_builder(depth=3, full_graph=True, cap=8):
    spec = NS(block_size=16, num_speculative_blocks=depth)
    cfg = NS(
        speculative_config=NS(num_speculative_tokens=depth, parallel_drafting=False)
        if depth else None,
        num_speculative_tokens=depth,
        compilation_config=NS(
            max_cudagraph_capture_size=cap,
            cudagraph_mode=NS(has_full_cudagraphs=lambda: full_graph),
        ),
        cache_config=NS(mamba_cache_mode="none", use_replayssm=False,
                        replayssm_buffer_len=16),
        scheduler_config=NS(max_num_seqs=cap),
        model_config=NS(max_model_len=8192, get_mamba_chunk_size=lambda: 128),
        parallel_config=NS(decode_context_parallel_size=1),
    )
    return Builder(spec, ["layer"], cfg, torch.device("cpu"))


def common(query_lens, seq_lens, table):
    loc = torch.tensor([0] + list(itertools.accumulate(query_lens)), dtype=torch.int32)
    seq = torch.tensor(seq_lens, dtype=torch.int32)
    return NS(
        num_reqs=len(query_lens), num_actual_tokens=sum(query_lens),
        max_query_len=max(query_lens), query_start_loc_cpu=loc, query_start_loc=loc,
        seq_lens=seq, seq_lens_cpu_upper_bound=seq,
        is_prefilling=torch.zeros(len(query_lens), dtype=torch.bool),
        block_table_tensor=table,
        slot_mapping=torch.zeros(sum(query_lens), dtype=torch.int64),
    )


def build(builder, cm, accepted):
    return builder.build(0, cm, num_accepted_tokens=accepted)


class MambaDecodeReuse(unittest.TestCase):
    def assert_metadata_equal(self, left, right):
        self.assertIs(type(left), type(right))
        for field in dataclasses.fields(left):
            a, b = getattr(left, field.name), getattr(right, field.name)
            if isinstance(a, torch.Tensor):
                self.assertIsInstance(b, torch.Tensor, field.name)
                self.assertEqual(a.dtype, b.dtype, field.name)
                self.assertEqual(a.shape, b.shape, field.name)
                self.assertTrue(torch.equal(a, b), field.name)
            else:
                self.assertEqual(a, b, field.name)

    def test_all_fields_and_private_buffers_match_fresh_build_across_steps(self):
        rng = random.Random(241024)
        checked = 0
        for depth in range(7):
            for full_graph in (False, True):
                for rows in (1, 2, 3, 4, 8):
                    # Three independent builders are essential. A fresh build
                    # on the reused builder would overwrite the same output
                    # storage and could hide a wrong result during comparison.
                    source = make_builder(depth, full_graph)
                    destination = make_builder(depth, full_graph)
                    reference = make_builder(depth, full_graph)
                    for step in range(35):
                        with self.subTest(depth=depth, full_graph=full_graph,
                                          rows=rows, step=step):
                            qlens = [rng.randint(1, depth + 1) for _ in range(rows)]
                            seq = [rng.randint(max(q, 2), 8192) for q in qlens]
                            if rows > 1 and step % 2 == 0:
                                padding = rng.randrange(1, rows)
                                qlens[-padding:] = [0] * padding
                                seq[-padding:] = [0] * padding
                            table_a = torch.tensor(
                                [[rng.randrange(1, 2000) for _ in range(depth + 1)]
                                 for _ in range(rows)], dtype=torch.int32
                            )
                            table_b = table_a + 2000 + step
                            accepted = torch.tensor(
                                [rng.randint(1, depth + 1) for _ in range(rows)],
                                dtype=torch.int32,
                            ) if depth else None
                            cm_a = common(qlens, seq, table_a)
                            cm_b = common(qlens, seq, table_b)
                            expected = build(reference, cm_b, accepted)
                            # Fresh source metadata EACH step; no persistent
                            # cache survives a request/phase/step boundary.
                            cached = build(source, cm_a, accepted)
                            before = cached.state_indices_tensor_d.clone()
                            result = destination.update_block_table(
                                cached, table_b, cm_b.slot_mapping
                            )
                            self.assert_metadata_equal(expected, result)
                            self.assertTrue(torch.equal(
                                cached.state_indices_tensor_d, before
                            ))
                            self.assertTrue(torch.equal(
                                result.state_indices_tensor_d, table_b
                            ))
                            if full_graph:
                                self.assertEqual(
                                    result.state_indices_tensor_d.data_ptr(),
                                    destination.state_indices_tensor_d.data_ptr(),
                                )
                                self.assertNotEqual(
                                    result.state_indices_tensor_d.data_ptr(),
                                    source.state_indices_tensor_d.data_ptr(),
                                )
                                if depth:
                                    self.assertEqual(
                                        result.num_accepted_tokens.data_ptr(),
                                        destination.decode_num_accepted_tokens.data_ptr(),
                                    )
                                    self.assertNotEqual(
                                        result.num_accepted_tokens.data_ptr(),
                                        source.decode_num_accepted_tokens.data_ptr(),
                                    )
                            checked += 1
        self.assertEqual(checked, 2450)

    def fixture(self):
        builder = make_builder()
        cm = common([4], [800], torch.tensor([[3, 4, 5, 6]], dtype=torch.int32))
        metadata = build(builder, cm, torch.tensor([2], dtype=torch.int32))
        return builder, metadata

    def test_supported_decode_is_reusable_and_switch_disables_it(self):
        builder, metadata = self.fixture()
        with patch.object(fast, "MAMBA_DECODE_META_REUSE", True):
            self.assertTrue(fast.mamba_decode_reusable(builder, metadata))
        with patch.object(fast, "MAMBA_DECODE_META_REUSE", False):
            self.assertFalse(fast.mamba_decode_reusable(builder, metadata))

    def test_unsupported_engine_configurations_fall_back(self):
        with patch.object(fast, "MAMBA_DECODE_META_REUSE", True):
            for mode in ("align", "all"):
                with self.subTest(mode=mode):
                    builder, metadata = self.fixture()
                    builder.vllm_config.cache_config.mamba_cache_mode = mode
                    self.assertFalse(fast.mamba_decode_reusable(builder, metadata))
            builder, metadata = self.fixture()
            builder.use_replayssm = True
            self.assertFalse(fast.mamba_decode_reusable(builder, metadata))
            builder, metadata = self.fixture()
            builder.use_spec_decode = False
            self.assertFalse(fast.mamba_decode_reusable(builder, metadata))
            builder, metadata = self.fixture()
            builder.decode_cudagraph_max_bs = 0
            self.assertFalse(fast.mamba_decode_reusable(builder, metadata))

    def test_non_decode_or_incomplete_metadata_falls_back(self):
        invalid = (
            {"num_prefills": 1}, {"num_prefill_tokens": 1},
            {"num_decodes": 0}, {"num_reqs": 2},
            {"num_reqs": 0, "num_decodes": 0},
            {"num_accepted_tokens": None}, {"query_start_loc_d": None},
        )
        with patch.object(fast, "MAMBA_DECODE_META_REUSE", True):
            for changes in invalid:
                with self.subTest(changes=changes):
                    builder, metadata = self.fixture()
                    for name, value in changes.items():
                        setattr(metadata, name, value)
                    self.assertFalse(fast.mamba_decode_reusable(builder, metadata))

    def test_prefill_reuse_remains_controlled_by_its_original_switch(self):
        with patch.object(fast, "MAMBA_META_REUSE", True):
            self.assertTrue(fast.mamba_reusable(NS(num_decodes=0, num_prefills=1)))
            self.assertFalse(fast.mamba_reusable(NS(num_decodes=1, num_prefills=0)))
        with patch.object(fast, "MAMBA_META_REUSE", False):
            self.assertFalse(fast.mamba_reusable(NS(num_decodes=0, num_prefills=1)))

    def test_caller_limits_decode_reuse_to_the_exact_mamba2_builder(self):
        path = ROOT / "vllm/vllm/v1/worker/gpu_model_runner.py"
        tree = ast.parse(path.read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == "_can_reuse_mamba_metadata")
        namespace = {"prefill_fast": fast, "Mamba2AttentionMetadataBuilder": Builder}
        exec(compile(ast.fix_missing_locations(ast.Module(
            body=[method], type_ignores=[]
        )), str(path), "exec"), namespace)
        can_reuse = namespace["_can_reuse_mamba_metadata"]
        builder, metadata = self.fixture()

        class DerivedBuilder(Builder):
            pass

        subclass = DerivedBuilder.__new__(DerivedBuilder)
        subclass.__dict__.update(builder.__dict__)
        with patch.object(fast, "MAMBA_DECODE_META_REUSE", True):
            self.assertTrue(can_reuse(builder, metadata))
            self.assertFalse(can_reuse(subclass, metadata))
            self.assertFalse(can_reuse(object(), metadata))
        # Preserve the prior isinstance-based prefill behavior for subclasses.
        with patch.object(fast, "MAMBA_META_REUSE", True):
            self.assertTrue(can_reuse(subclass, NS(num_decodes=0, num_prefills=1)))

    def test_diagnostic_snapshot_detects_changes_hidden_by_buffer_aliasing(self):
        builder, original = self.fixture()
        snapshot = fast.snapshot_metadata(original)
        self.assert_metadata_equal(original, snapshot)
        for field in dataclasses.fields(original):
            a, b = getattr(original, field.name), getattr(snapshot, field.name)
            if isinstance(a, torch.Tensor) and a.numel():
                self.assertNotEqual(a.data_ptr(), b.data_ptr(), field.name)

        # A fresh diagnostic build on the SAME builder overwrites its private
        # buffers. Comparing original to reference would now falsely succeed.
        cm = common([4], [800], torch.tensor([[103, 104, 105, 106]], dtype=torch.int32))
        reference = build(builder, cm, torch.tensor([3], dtype=torch.int32))
        self.assert_metadata_equal(original, reference)
        with self.assertRaisesRegex(RuntimeError, "state_indices_tensor_d"):
            fast.check_same(snapshot, reference, "mutation-test")


if __name__ == "__main__":
    unittest.main()

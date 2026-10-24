"""CPU checks for the verify-row filter wired into zskip's M=5 down projection.

The device rule lives in zskip.cu (slot_kept / vrow_limit). These tests cover the
host decision, the flattened-slot index the kernel divides by, and that both prep
sites call that helper. No CUDA.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _slot_kept(slot: int, shape_m: int, top_k: int, limit: int) -> bool:
    """Mirror of zskip.cu slot_kept. Padding and a non-positive limit keep the old path."""
    if slot >= shape_m or slot < 0:
        return False
    if top_k <= 0 or limit >= 0x7FFFFFFF:
        return True
    return (slot // top_k) < limit


class SlotIndexTests(unittest.TestCase):
    def test_token_major_layout_matches_moe_align_flatten(self):
        top_k = 6
        shape_m = 5 * top_k
        limit = 3
        for token in range(5):
            for k in range(top_k):
                slot = token * top_k + k
                self.assertEqual(_slot_kept(slot, shape_m, top_k, limit), token < limit)

    def test_null_limit_and_padding_match_the_unchecked_path(self):
        self.assertTrue(_slot_kept(0, 30, 0, 1))
        self.assertTrue(_slot_kept(29, 30, 6, 0x7FFFFFFF))
        self.assertFalse(_slot_kept(30, 30, 6, 1))
        self.assertFalse(_slot_kept(0xFFFFFFFF, 30, 6, 1))

    def test_source_uses_one_helper_at_both_prep_sites(self):
        src = Path(__file__).resolve().parents[1].joinpath("csrc", "zskip.cu").read_text()
        self.assertEqual(src.count("bool slot_kept("), 1)
        self.assertEqual(src.count("slot_kept(id, p.shape_m, p.top_k, limit)"), 3)
        self.assertIn("const int* vrows, int top_k, void* stream", src)
        self.assertIn("id / static_cast<unsigned>(top_k)", src)


def _load_humming_zskip():
    """Load the expert module without executing vLLM's package __init__."""
    logger_mod = ModuleType("vllm.logger")
    logger_mod.init_logger = lambda name: logging.getLogger(name)
    vllm = ModuleType("vllm")
    vllm.logger = logger_mod
    saved = {k: sys.modules.get(k) for k in ("vllm", "vllm.logger")}
    sys.modules["vllm"] = vllm
    sys.modules["vllm.logger"] = logger_mod
    path = (
        Path(__file__).resolve().parents[4]
        / "vllm/vllm/model_executor/layers/fused_moe/experts/humming_zskip.py"
    )
    spec = importlib.util.spec_from_file_location("humming_zskip_vrow_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for key, old in saved.items():
        if old is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = old
    return module


class VRowArgsTests(unittest.TestCase):
    def setUp(self):
        self.hz = _load_humming_zskip()
        self._saved = sys.modules.get("smallk")
        self._flag = self.hz.VROW_SKIP

    def tearDown(self):
        if self._saved is None:
            sys.modules.pop("smallk", None)
        else:
            sys.modules["smallk"] = self._saved
        self.hz.VROW_SKIP = self._flag

    def test_only_the_depth4_verify_passes_the_buffer(self):
        buf = object()
        sys.modules["smallk"] = SimpleNamespace(live_verify_rows=lambda device: buf)
        self.hz.VROW_SKIP = True
        self.assertEqual(self.hz.vrow_args(5, "cuda:0"), (buf, self.hz.TOP_K))
        for n in (1, 4, 6, 8, 496):
            self.assertEqual(self.hz.vrow_args(n, "cuda:0"), (None, 0))

    def test_missing_or_disabled_gate_keeps_every_slot(self):
        sys.modules["smallk"] = SimpleNamespace(live_verify_rows=lambda device: None)
        self.hz.VROW_SKIP = True
        self.assertEqual(self.hz.vrow_args(5, "cuda:0"), (None, 0))
        sys.modules.pop("smallk")
        self.assertEqual(self.hz.vrow_args(5, "cuda:0"), (None, 0))
        sys.modules["smallk"] = SimpleNamespace(live_verify_rows=lambda device: object())
        self.hz.VROW_SKIP = False
        self.assertEqual(self.hz.vrow_args(5, "cuda:0"), (None, 0))


if __name__ == "__main__":
    unittest.main()

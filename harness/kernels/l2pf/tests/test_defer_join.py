"""CPU model of capture fork/join ordering. No torch/GPU dependency."""
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True


def load(enabled=True):
    log = []
    device = SimpleNamespace(index=0)

    class Stream:
        def __init__(self, ident):
            self.device = device
            self.cuda_stream = ident

        def wait_stream(self, other):
            log.append(("wait", self.cuda_stream, other.cuda_stream))

    main, side, foreign = Stream(1), Stream(2), Stream(3)
    torch = ModuleType("torch")
    torch.capturing = True
    torch.current = main
    torch.cuda = SimpleNamespace(
        current_stream=lambda *args: torch.current,
        current_device=lambda: 0,
        is_current_stream_capturing=lambda: torch.capturing,
        Stream=lambda **kw: side)

    class Op:
        def __call__(self, *args):
            log.append(("hint", args[-1]))
            return 0

    lib = SimpleNamespace(l2pf_prefetch=Op())
    spec = importlib.util.spec_from_file_location(
        "l2pf_cpu", Path(__file__).resolve().parents[1] / "py" / "__init__.py")
    mod = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch": torch}), patch("ctypes.CDLL", return_value=lib), \
            patch.dict(os.environ, {"L2PF": "1", "L2PF_DEFER_JOIN": "1" if enabled else "0"}):
        spec.loader.exec_module(mod)
    weight = SimpleNamespace(is_cuda=True, device=device,
                             untyped_storage=lambda: SimpleNamespace(nbytes=lambda: 1024,
                                                                      data_ptr=lambda: 16))
    return mod, torch, weight, log, main, side, foreign


class DeferralTests(unittest.TestCase):
    def test_next_fork_waits_at_later_main_tail_and_capture_end_flushes(self):
        mod, torch, weight, log, *_ = load()
        with mod.capture_scope():
            self.assertTrue(mod.prefetch(weight))
            log.append("arithmetic1")
            mod.join(weight.device)
            log.append("arithmetic2")
            self.assertEqual(log, [("wait", 2, 1), ("hint", 2), "arithmetic1", "arithmetic2"])
            self.assertTrue(mod.prefetch(weight))
            log.append("final_norm")
            mod.join(weight.device)
        self.assertEqual(log, [("wait", 2, 1), ("hint", 2), "arithmetic1", "arithmetic2",
                               ("wait", 1, 2), ("wait", 2, 1), ("hint", 2), "final_norm",
                               ("wait", 1, 2)])
        self.assertIsNone(mod._scope())

    def test_kill_switch_unscoped_eager_and_foreign_keep_immediate_join(self):
        for mode in ("off", "unscoped", "eager", "foreign"):
            mod, torch, weight, log, _, _, foreign = load(mode != "off")
            if mode == "eager":
                torch.capturing = False
            if mode == "unscoped":
                mod.prefetch(weight)
                mod.join(weight.device)
            else:
                with mod.capture_scope():
                    if mode == "foreign":
                        torch.current = foreign
                    mod.prefetch(weight)
                    mod.join(weight.device)
                    self.assertEqual(log[-1][0], "wait", mode)
            ident = 3 if mode == "foreign" else 1
            self.assertEqual(log, [("wait", 2, ident), ("hint", 2), ("wait", ident, 2)], mode)

    def test_no_hint_has_no_wait_exception_flushes_pending_and_resets_scope(self):
        mod, _, weight, log, *_ = load()
        with mod.capture_scope():
            self.assertFalse(mod.prefetch(None))
        self.assertEqual(log, [])
        with self.assertRaisesRegex(ValueError, "body"):
            with mod.capture_scope():
                mod.prefetch(weight)
                raise ValueError("body")
        self.assertEqual(log[-1], ("wait", 1, 2))
        self.assertIsNone(mod._scope())

    def test_foreign_fork_flushes_prior_owner_before_reusing_side(self):
        mod, torch, weight, log, main, _, foreign = load()
        with mod.capture_scope():
            mod.prefetch(weight)
            mod.join(weight.device)
            torch.current = foreign
            mod.prefetch(weight)
            mod.join(weight.device)
            torch.current = main
        self.assertEqual(log, [("wait", 2, 1), ("hint", 2), ("wait", 1, 2),
                               ("wait", 2, 3), ("hint", 2), ("wait", 3, 2)])

    def test_each_capture_closes_its_branch_independently(self):
        mod, _, weight, log, *_ = load()
        for _ in range(3):
            with mod.capture_scope():
                mod.prefetch(weight)
                mod.join(weight.device)
        self.assertEqual(log, [("wait", 2, 1), ("hint", 2), ("wait", 1, 2)] * 3)


if __name__ == "__main__":
    unittest.main()

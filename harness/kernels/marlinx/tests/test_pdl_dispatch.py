"""CPU-only checks for optional ABI, kill switch and eager/capture dispatch."""
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

sys.dont_write_bytecode = True
from unittest.mock import patch


class Op:
    def __init__(self, result=0):
        self.result = result
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        return self.result


class Tensor:
    shape = (5, 2688)
    device = SimpleNamespace(index=0)

    def data_ptr(self):
        return 16

    def stride(self, dim):
        return 2688


def load_module(pdl=True, enabled=True, init_rc=0):
    lib = SimpleNamespace(**{name: Op() for name in (
        "marlinx_init", "marlinx_regs", "marlinx_gemm", "marlinx_gemm_relu2",
        "marlinx_gemm_zs", "marlinx_hog")})
    if pdl:
        lib.marlinx_init_pdl = Op(init_rc)
        for name in ("marlinx_gemm", "marlinx_gemm_relu2", "marlinx_gemm_zs"):
            setattr(lib, name + "_pdl", Op())
    torch = ModuleType("torch")
    torch.capturing = False
    torch.no_grad = lambda: (lambda fn: fn)
    torch.bfloat16 = object()
    torch.empty = lambda *args, **kwargs: Tensor()
    torch.cuda = SimpleNamespace(
        is_current_stream_capturing=lambda: torch.capturing,
        current_stream=lambda *args: SimpleNamespace(cuda_stream=123),
        get_device_properties=lambda *args: SimpleNamespace(multi_processor_count=48))
    source = Path(__file__).resolve().parents[1] / "py" / "__init__.py"
    spec = importlib.util.spec_from_file_location("marlinx_dispatch_test", source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch": torch}), \
            patch.dict(os.environ, {"MARLINX_PDL": "1" if enabled else "0"}), \
            patch("ctypes.CDLL", return_value=lib):
        spec.loader.exec_module(module)
        assert module._init()
    return module, lib, torch


class DispatchTests(unittest.TestCase):
    def run_gemm(self, module, mode):
        kwargs = {"relu2": True} if mode == "relu2" else {"zs": Tensor()} if mode == "zs" else {}
        return module.gemm(Tensor(), Tensor(), Tensor(), Tensor(), 3712, 2688, **kwargs)

    def test_default_and_eager_rejection_disables_all_modes(self):
        for mode in ("gemm", "relu2", "zs"):
            with self.subTest(mode=mode):
                module, lib, _ = load_module()
                self.assertTrue(module._pdl_active)
                op = module._ops_pdl[mode]
                op.result = 801
                self.run_gemm(module, mode)
                self.assertEqual(op.calls, 1)
                self.assertEqual(module._ops[mode].calls, 1)
                self.assertFalse(module._pdl_active)
                self.run_gemm(module, "gemm")
                self.assertEqual(lib.marlinx_gemm_pdl.calls, int(mode == "gemm"))

    def test_capture_rejection_never_retries(self):
        for mode in ("gemm", "relu2", "zs"):
            with self.subTest(mode=mode):
                module, _, torch = load_module()
                torch.capturing = True
                module._ops_pdl[mode].result = 801
                with self.assertRaisesRegex(RuntimeError, "failed: 801"):
                    self.run_gemm(module, mode)
                self.assertEqual(module._ops[mode].calls, 0)
                self.assertTrue(module._pdl_active)

    def test_no_abi_kill_switch_and_init_failure(self):
        for kwargs in ({"pdl": False}, {"enabled": False}, {"init_rc": 801}):
            with self.subTest(kwargs=kwargs):
                module, _, _ = load_module(**kwargs)
                self.assertFalse(module._pdl_active)
                self.run_gemm(module, "gemm")
                self.assertEqual(module._ops["gemm"].calls, 1)

    def test_success_uses_each_pdl_entry(self):
        module, _, _ = load_module()
        for mode in ("gemm", "relu2", "zs"):
            self.run_gemm(module, mode)
            self.assertEqual(module._ops_pdl[mode].calls, 1)
            self.assertEqual(module._ops[mode].calls, 0)

    def test_verify_shared_down_ring_depth(self):
        """M=5 zero-skip shared down passes depth 4; every other call stays at DEPTH."""
        module, _, torch = load_module()
        torch.compiler = SimpleNamespace(is_compiling=lambda: False)
        seen = []

        def capture(*args):
            seen.append(args)
            return 0

        module._ops_pdl["zs"] = capture
        module._ops_pdl["gemm"] = capture

        class Row:
            def __init__(self, m, k):
                self.shape = (m, k)
                self.dtype = torch.bfloat16
                self.device = SimpleNamespace(index=0)

            def reshape(self, *args):
                return self

            def stride(self, dim):
                return 1 if dim == -1 else 16

            def data_ptr(self):
                return 16

        class Buf:
            def data_ptr(self):
                return 32

        class Out:
            def data_ptr(self):
                return 64

            def reshape(self, *args):
                return self

        torch.empty = lambda *args, **kwargs: Out()
        layer = SimpleNamespace(
            _marlinx_ok=True,
            output_size_per_partition=2688,
            weight=Buf(),
            weight_scale=Buf(),
            weight_global_scale=Buf(),
            _marlinx_zs=Buf(),
        )
        self.assertIsNotNone(module.maybe_apply(layer, Row(5, 3712)))
        self.assertEqual(seen[-1][2], 5)
        self.assertEqual(seen[-1][8], 2688)
        self.assertEqual(seen[-1][9], 3712)
        self.assertEqual(seen[-1][11], module.ZS_VERIFY_DEPTH)
        self.assertEqual(len(seen[-1]), 14)

        module.maybe_apply(layer, Row(4, 3712))
        self.assertEqual(seen[-1][2], 4)
        self.assertEqual(seen[-1][11], module.DEPTH)

        layer._marlinx_zs = None
        module.maybe_apply(layer, Row(5, 3712))
        self.assertEqual(seen[-1][11], module.DEPTH)
        self.assertEqual(len(seen[-1]), 13)

        layer._marlinx_zs = Buf()
        module.ZS_VERIFY_DEPTH = module.DEPTH
        module.maybe_apply(layer, Row(5, 3712))
        self.assertEqual(seen[-1][11], module.DEPTH)


if __name__ == "__main__":
    unittest.main()

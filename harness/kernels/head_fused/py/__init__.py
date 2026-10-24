# SPDX-License-Identifier: Apache-2.0
"""Full-K native draft head only; original partial-screen candidate path stays."""
import ctypes
import os
from pathlib import Path
import torch

_lib = None
_failed = False


def available(device):
    global _lib, _failed
    if _failed or torch.cuda.get_device_capability(device) != (12, 1):
        return False
    if _lib is None:
        if torch.cuda.is_current_stream_capturing():
            return False
        try:
            lib = ctypes.CDLL(str(Path(__file__).with_name("libhead_fused.so")))
            lib.head_fused_init.argtypes = []
            lib.head_fused_init.restype = ctypes.c_int
            lib.head_fused_finish.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_float]
                + [ctypes.c_void_p] * 3 + [ctypes.c_float, ctypes.c_void_p])
            lib.head_fused_finish.restype = ctypes.c_int
            if lib.head_fused_init():
                _failed = True
                return False
            _lib = lib
        except (OSError, AttributeError):
            _failed = True
            return False
    return True


class Selector:
    def __init__(self, screen, device):
        if not available(device):
            raise RuntimeError("head_fused unavailable")
        self.wf, self.sf, self.gs = screen.w_full, screen.s_full, screen.gscale
        self.full_mma = os.environ.get("MTP_HEAD_FULL_MMA", "1") != "0"
        assert self.wf.shape == (131072, 1344) and self.sf.shape == (131072, 168)
        assert self.wf.dtype == torch.uint8 and self.sf.dtype == torch.float8_e4m3fn
        assert all(x.is_cuda and x.device == device and x.is_contiguous()
                   for x in (self.wf, self.sf))

    def finish(self, h, candidates, gate=None, theta=0.0):
        assert h.dtype == torch.float32 and h.shape == (2688,) and h.is_cuda and h.is_contiguous()
        assert candidates.shape == (2048,) and candidates.dtype == torch.int64
        assert candidates.device == h.device and candidates.is_contiguous()
        # Eager lifetimes belong to this current stream, captured lifetimes to
        # the graph pool; no global mutable workspace crosses calls or phases.
        partial = torch.empty((4, 2048), device=h.device, dtype=torch.float32)
        out = torch.empty((), device=h.device, dtype=torch.int32)
        if gate is not None:
            assert gate.dtype == torch.int32 and gate.device == h.device and gate.numel() == 3
        gate_ptr = None if gate is None else gate.data_ptr()
        stream = torch.cuda.current_stream(h.device).cuda_stream
        rc = _lib.head_fused_finish(h.data_ptr(), self.wf.data_ptr(), self.sf.data_ptr(),
            candidates.data_ptr(), self.gs, partial.data_ptr(), out.data_ptr(), gate_ptr, theta, stream)
        if rc:
            raise RuntimeError(f"head_fused finish launch: {rc}")
        return out

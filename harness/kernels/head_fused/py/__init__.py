# SPDX-License-Identifier: Apache-2.0
"""Native fused partial-score/selection for the checkpoint's draft head only."""
import ctypes
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
            lib = ctypes.CDLL(str(Path(__file__).with_name('libhead_fused.so')))
            lib.head_fused_init.argtypes = []
            lib.head_fused_init.restype = ctypes.c_int
            common = [ctypes.c_void_p] * 4 + [ctypes.c_float] + [ctypes.c_void_p] * 4
            lib.head_fused_coarse.argtypes = common
            lib.head_fused_coarse.restype = ctypes.c_int
            lib.head_fused_refine.argtypes = common
            lib.head_fused_refine.restype = ctypes.c_int
            if lib.head_fused_init():
                _failed = True
                return False
            _lib = lib
        except (OSError, AttributeError):
            _failed = True
            return False
    return True


class Selector:
    """Immutable model planes; all intermediate data belongs to this call/graph."""
    def __init__(self, screen, device):
        if not available(device):
            raise RuntimeError('head_fused unavailable')
        self.wc, self.sc, self.cc = screen.w_coarse, screen.s_coarse, screen.coarse_cols
        self.wr, self.sr, self.cr = screen.w_scr, screen.s_scr, screen.cols
        self.gs = screen.gscale
        assert self.wc.shape == (131072, 256) and self.wr.shape == (131072, 512)
        assert self.sc.shape == (131072, 32) and self.sr.shape == (131072, 64)
        assert self.wc.dtype == self.wr.dtype == torch.uint8
        assert self.sc.dtype == self.sr.dtype == torch.float8_e4m3fn
        assert self.cc.shape == (512,) and self.cr.shape == (1024,)
        assert self.cc.dtype == self.cr.dtype == torch.int64
        assert all(x.is_cuda and x.device == device and x.is_contiguous()
                   for x in (self.wc,self.sc,self.cc,self.wr,self.sr,self.cr))

    def candidates(self, h, gate=None):
        assert h.dtype == torch.float32 and h.is_cuda and h.is_contiguous() and h.shape == (2688,)
        # The source graph pool owns captured allocations through replay. Eager
        # calls enqueue every producer/consumer on this same current stream.
        partial = torch.empty((256,256), device=h.device, dtype=torch.uint64)
        coarse = torch.empty((64,256), device=h.device, dtype=torch.int64)
        out = torch.empty((64,32), device=h.device, dtype=torch.int64)
        if gate is not None:
            assert gate.dtype == torch.int32 and gate.device == h.device and gate.numel() == 3
        gate_ptr = None if gate is None else gate.data_ptr()
        stream = torch.cuda.current_stream(h.device).cuda_stream
        rc = _lib.head_fused_coarse(h.data_ptr(),self.wc.data_ptr(),self.sc.data_ptr(),
             self.cc.data_ptr(),self.gs,partial.data_ptr(),coarse.data_ptr(),gate_ptr,stream)
        if rc:
            raise RuntimeError(f'head_fused coarse launch: {rc}')
        rc = _lib.head_fused_refine(h.data_ptr(),self.wr.data_ptr(),self.sr.data_ptr(),
             self.cr.data_ptr(),self.gs,coarse.data_ptr(),out.data_ptr(),gate_ptr,stream)
        if rc:
            raise RuntimeError(f'head_fused refine launch: {rc}')
        return out.reshape(-1)

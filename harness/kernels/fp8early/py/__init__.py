# SPDX-License-Identifier: Apache-2.0
"""Optional original-eight-warp FP8 GEMM with last-reader stage release.

No pack or arithmetic changes. Callers must qualify full words and changing
input graphs against the original fp8ws C ABI before dispatching this variant.
"""

import ctypes
from pathlib import Path

import torch

M = 496
SHAPES = ((10304, 2688), (2688, 4096))
_BY_SIZE = {n * k: (n, k) for n, k in SHAPES}
_lib = None
_failed = False
_ready_devices = set()


def prepare(device):
    global _lib, _failed
    if _failed or torch.cuda.is_current_stream_capturing():
        return False
    if torch.cuda.get_device_capability(device) != (12, 1):
        return False
    try:
        if _lib is None:
            lib = ctypes.CDLL(str(Path(__file__).with_name("libfp8early.so")))
            lib.fp8early_init.argtypes = []
            lib.fp8early_init.restype = ctypes.c_int
            lib.fp8early_run.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
            lib.fp8early_run.restype = ctypes.c_int
            _lib = lib
        with torch.cuda.device(device):
            index = torch.cuda.current_device()
            if index not in _ready_devices:
                if _lib.fp8early_init() != 0:
                    return False
                _ready_devices.add(index)
        return True
    except (OSError, AttributeError):
        _failed = True
        return False


def run_out(a, packed, sa, sb, out):
    if _lib is None or a.device.index not in _ready_devices:
        raise RuntimeError("fp8early not prepared")
    n, k = _BY_SIZE[packed.numel()]
    if (a.dtype != torch.float8_e4m3fn or tuple(a.shape) != (M, k)
            or not a.is_contiguous() or out.dtype != torch.bfloat16
            or tuple(out.shape) != (M, n) or not out.is_contiguous()
            or packed.dtype != torch.uint8 or not packed.is_contiguous()
            or not (a.device == packed.device == sa.device == sb.device == out.device)
            or sa.dtype != torch.float32 or sb.dtype != torch.float32
            or sa.numel() != 1 or sb.numel() != 1
            or not sa.is_contiguous() or not sb.is_contiguous()
            or any(t.data_ptr() % 16 for t in (a, packed, out))):
        raise ValueError("fp8early unsupported layout")
    rc = _lib.fp8early_run(a.data_ptr(), packed.data_ptr(), sa.data_ptr(), sb.data_ptr(),
                           out.data_ptr(), M, n, k,
                           torch.cuda.current_stream(a.device).cuda_stream)
    if rc:
        raise RuntimeError("fp8early launch failed: " + str(rc))

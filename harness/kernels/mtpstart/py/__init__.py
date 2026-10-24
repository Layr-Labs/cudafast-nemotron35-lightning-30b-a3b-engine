# SPDX-License-Identifier: Apache-2.0
"""Native MTP start projection with an explicit intermediate BF16 boundary."""
import ctypes
import os
from pathlib import Path

import torch

_ENABLED = os.environ.get("MTP_START_MMA", "1") != "0"
_lib = None
_ready = set()


def prepare(device):
    global _lib
    if not _ENABLED or device.type != "cuda":
        return False
    with torch.cuda.device(device):
        if torch.cuda.get_device_capability(device) != (12, 1):
            return False
        if _lib is None:
            try:
                lib = ctypes.CDLL(str(Path(__file__).with_name("libmtpstart.so")))
            except OSError:
                return False
            lib.mtpstart_init.argtypes = []
            lib.mtpstart_init.restype = ctypes.c_int
            lib.mtpstart_run.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_void_p]
            lib.mtpstart_run.restype = ctypes.c_int
            _lib = lib
        index = torch.cuda.current_device()
        if index not in _ready:
            rc = _lib.mtpstart_init()
            if rc:
                raise RuntimeError(f"mtpstart initialization failed: {rc}")
            _ready.add(index)
    return True


def maybe_apply(x, w, e):
    if (not _ENABLED or _lib is None or not x.is_cuda or x.device.index not in _ready
            or w.device != x.device or e.device != x.device
            or x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16
            or e.dtype != torch.float32 or x.ndim != 2 or not 1 <= x.shape[0] <= 8
            or x.shape[1] != 2688 or tuple(w.shape) != (2688, 2688)
            or tuple(e.shape) != tuple(x.shape)
            or not x.is_contiguous() or not w.is_contiguous() or not e.is_contiguous()
            or x.data_ptr() % 16 or w.data_ptr() % 16 or e.data_ptr() % 4):
        return None
    out = torch.empty_like(x)
    rc = _lib.mtpstart_run(x.data_ptr(), w.data_ptr(), e.data_ptr(), out.data_ptr(),
                           x.shape[0], torch.cuda.current_stream(x.device).cuda_stream)
    if rc:
        raise RuntimeError(f"mtpstart launch failed: {rc}")
    return out

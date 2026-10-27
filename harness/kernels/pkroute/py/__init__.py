# SPDX-License-Identifier: Apache-2.0
"""Adaptive task wrapper; actual-owner admission is in the native MTP model."""

import ctypes
import os
from pathlib import Path

import torch

ENABLED = os.environ.get("MTP_ADAPT_ROUTE", "1") != "0"
H, I = 2688, 1856
_lib = None
_load_failed = False


def _load():
    global _lib, _load_failed
    if _lib is not None or _load_failed:
        return _lib
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name("libpkroute.so")))
        lib.pkroute_run.argtypes = [ctypes.c_void_p] * 15 + [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        lib.pkroute_run.restype = ctypes.c_int
        _lib = lib
    except OSError:
        _load_failed = True
    return _lib


def available() -> bool:
    return ENABLED and _load() is not None


def run(indices, values, counts, routed_planes, shared_planes, ids, part, refine, pdl: bool) -> bool:
    """part[topk + 2, 2688] (fp32) from the compacted slots, as _slots with the (64, 4) tile.
    False (nothing launched) for inputs outside the kernel's contract."""
    topk = ids.numel()
    if (topk != 6 or refine.dtype != torch.int32 or refine.numel() != 1
            or not refine.is_cuda or not refine.is_contiguous()
            or refine.device != part.device or counts.numel() != 8):
        return False
    if (not ENABLED or _load() is None or indices.dtype != torch.int32 or values.dtype != torch.bfloat16
            or counts.dtype != torch.int32 or ids.dtype != torch.int32 or part.dtype != torch.float32
            or tuple(indices.shape) != (topk + 2, I) or tuple(values.shape) != (topk + 2, I)
            or tuple(part.shape) != (topk + 2, H) or not indices.is_contiguous()
            or not values.is_contiguous() or not part.is_contiguous() or not ids.is_contiguous()
            or tuple(ids.shape) != (6,) or tuple(counts.shape) != (8,) or not counts.is_contiguous()):
        return False
    planes = tuple(routed_planes[:5]) + tuple(shared_planes[:5])
    if (len(planes) != 10 or any(int(p.data_ptr()) % 16 for p in planes)
            or any(not t.is_cuda or t.device != part.device or not t.is_contiguous()
                   for t in (indices, values, counts, ids, part, refine, *planes))):
        return False
    rc = _lib.pkroute_run(indices.data_ptr(), values.data_ptr(), counts.data_ptr(),
                          *(p.data_ptr() for p in planes), ids.data_ptr(), part.data_ptr(),
                          topk, refine.data_ptr(), int(bool(pdl)), torch.cuda.current_stream(part.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"pkroute_run failed: {rc}")
    return True

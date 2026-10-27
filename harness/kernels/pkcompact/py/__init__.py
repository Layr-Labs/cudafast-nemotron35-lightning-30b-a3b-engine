# SPDX-License-Identifier: Apache-2.0
"""Official-remote-only CTA-local stable compaction + original64x4 packed down."""
import ctypes
import os
from pathlib import Path
import torch

ENABLED = os.environ.get("MTP_DOWN_LOCAL_COMPACT", "1") != "0"
_lib = None
_failed = False


def _load():
    global _lib, _failed
    if _lib is not None or _failed:
        return _lib
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name("libpkcompact.so")))
        lib.pkcompact_run.argtypes = [ctypes.c_void_p] * 13 + [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.pkcompact_run.restype = ctypes.c_int
        _lib = lib
    except OSError:
        _failed = True
    return _lib


def available():
    return ENABLED and _load() is not None


def run(h, routed_planes, shared_planes, ids, part, pdl):
    topk = ids.numel()
    planes = tuple(routed_planes[:5]) + tuple(shared_planes[:5])
    if (not ENABLED or _load() is None or topk not in (3, 6)
            or h.dtype != torch.bfloat16 or h.shape != (topk + 2, 1856)
            or ids.dtype != torch.int32 or ids.shape != (topk,)
            or part.dtype != torch.float32 or part.shape != (topk + 2, 2688)
            or len(planes) != 10 or any(t.data_ptr() % 16 for t in planes)
            or any(not t.is_cuda or t.device != h.device or not t.is_contiguous()
                   for t in (h, ids, part, *planes))):
        return False  # Nothing launched.
    rc = _lib.pkcompact_run(h.data_ptr(), *(t.data_ptr() for t in planes), ids.data_ptr(),
                            part.data_ptr(), topk, int(bool(pdl)),
                            torch.cuda.current_stream(h.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"pkcompact_run failed: {rc}")
    return True

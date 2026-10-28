"""Optional last-CTA compact-down prototype; owner/graph qualification required."""
import ctypes
import os
from pathlib import Path
import torch

ENABLED = os.environ.get("MTP_DOWN_PKFINISH", "1") != "0"
H, I = 2688, 1856
_lib = None
_load_failed = False


def _load():
    global _lib, _load_failed
    if _lib is not None or _load_failed:
        return _lib
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name("libpkfinish.so")))
        lib.pkfinish_run.argtypes = ([ctypes.c_void_p] * 19
            + [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int,
               ctypes.c_int, ctypes.c_void_p])
        lib.pkfinish_run.restype = ctypes.c_int
        _lib = lib
    except OSError:
        _load_failed = True
    return _lib


def available():
    return ENABLED and _load() is not None


def _range(t):
    return int(t.data_ptr()), int(t.data_ptr()) + t.numel() * t.element_size()


def _disjoint(a, b):
    x, y = _range(a), _range(b)
    return x[1] <= y[0] or y[1] <= x[0]


def run(indices, values, counts, routed_planes, shared_planes, ids, part,
        counters, weights, residual, output, routed_scale, pdl):
    """Scratch/counter must be private, and counters reset by the preceding compact node."""
    topk = ids.numel()
    planes = tuple(routed_planes[:5]) + tuple(shared_planes[:5])
    mutable = (part, counters, output)
    immutable = (indices, values, counts, ids, weights) + planes
    if residual is not None:
        immutable += (residual,)
    if (_load() is None or not output.is_cuda or topk not in (3, 6)
            or indices.dtype != torch.int32 or values.dtype != torch.bfloat16
            or counts.dtype != torch.int32 or ids.dtype != torch.int32
            or part.dtype != torch.float32 or counters.dtype != torch.int32
            or weights.dtype != torch.float32 or output.dtype != torch.bfloat16
            or tuple(indices.shape) != (topk + 2, I)
            or tuple(values.shape) != (topk + 2, I)
            or tuple(counts.shape) != (topk + 2,)
            or tuple(part.shape) != (topk + 2, H)
            or tuple(counters.shape) != (H // 64,)
            or tuple(weights.shape) != (topk,)
            or output.ndim != 2 or output.shape[0] not in (1, 5, 6, 8) or output.shape[1] != H
            or not all(t.is_contiguous() and t.device == output.device for t in mutable + immutable)
            or any(int(p.data_ptr()) % 16 for p in planes)
            or any(not _disjoint(a, b) for i, a in enumerate(mutable) for b in mutable[i+1:] + immutable)
            or (residual is not None and (residual.dtype != torch.bfloat16 or residual.shape != output.shape))):
        return False
    residual_ptr = 0 if residual is None else residual.data_ptr()
    rc = _lib.pkfinish_run(indices.data_ptr(), values.data_ptr(), counts.data_ptr(),
                          *(p.data_ptr() for p in planes), ids.data_ptr(), part.data_ptr(),
                          counters.data_ptr(), weights.data_ptr(), residual_ptr, output.data_ptr(),
                          topk, output.shape[0], routed_scale, int(residual is not None),
                          int(bool(pdl)), torch.cuda.current_stream(output.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"pkfinish_run failed: {rc}")
    return True

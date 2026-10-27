"""Optional exact pkslots producer/consumer warp pipeline, remote-admitted only."""
import ctypes
import os
from pathlib import Path

import torch

ENABLED = os.environ.get("MTP_DOWN_PKSPLIT", "1") != "0"
H, I = 2688, 1856
_lib = None
_load_failed = False


def _load():
    global _lib, _load_failed
    if _lib is not None or _load_failed:
        return _lib
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name("libpksplit.so")))
        lib.pksplit_run.argtypes = [ctypes.c_void_p] * 15 + [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.pksplit_run.restype = ctypes.c_int
        _lib = lib
    except OSError:
        _load_failed = True
    return _lib


def available() -> bool:
    return ENABLED and _load() is not None


def run(indices, values, counts, routed_planes, shared_planes, ids, part, pdl: bool) -> bool:
    """part[topk + 2, 2688] (fp32) from the compacted slots, as _slots with the (64, 4) tile.
    False (nothing launched) for inputs outside the kernel's contract."""
    topk = ids.numel()
    if (not ENABLED or topk not in (3, 6) or counts.shape != (topk + 2,)
            or not counts.is_contiguous() or not ids.is_cuda or not part.is_cuda
            or any(t.device != part.device for t in (indices, values, counts, ids, *routed_planes, *shared_planes))
            or _load() is None or indices.dtype != torch.int32 or values.dtype != torch.bfloat16
            or counts.dtype != torch.int32 or ids.dtype != torch.int32 or part.dtype != torch.float32
            or tuple(indices.shape) != (topk + 2, I) or tuple(values.shape) != (topk + 2, I)
            or tuple(part.shape) != (topk + 2, H) or not indices.is_contiguous()
            or not values.is_contiguous() or not part.is_contiguous() or not ids.is_contiguous()):
        return False
    planes = tuple(routed_planes[:5]) + tuple(shared_planes[:5])
    if any(int(p.data_ptr()) % 16 for p in planes):
        return False
    rc = _lib.pksplit_run(indices.data_ptr(), values.data_ptr(), counts.data_ptr(),
                          *(p.data_ptr() for p in planes), ids.data_ptr(), part.data_ptr(),
                          topk, int(bool(pdl)), torch.cuda.current_stream(part.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"pksplit_run failed: {rc}")
    return True

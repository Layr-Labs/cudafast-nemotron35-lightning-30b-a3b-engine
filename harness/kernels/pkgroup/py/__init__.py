# SPDX-License-Identifier: Apache-2.0
"""pkgroup: the drafter's packed MoE down slot kernel (nemotron_h_mtp_down_packed._slots with
the (64, 4) tile), with subgroup metadata broadcasting (csrc/pkgroup.cu).

The kernel holds _slots' accumulator elements per thread in its Triton layout and performs the
same fp32 multiplies and adds in the same order (the 64-row chunks in order, each thread's two
row reps, the xor butterfly 16, 8, 4 over the row lanes, then (w0 + w2) + (w1 + w3) over the
warps); exactness is checked remotely. The drafter admits it only after a load-time
independent bitwise and timing checks against admitted pkslots. MTP_DOWN_PKGROUP=0 keeps
the previous route.
"""

import ctypes
import os
from pathlib import Path

import torch

ENABLED = os.environ.get("MTP_DOWN_PKGROUP", "1") != "0"
H, I = 2688, 1856
_lib = None
_load_failed = False


def _load():
    global _lib, _load_failed
    if _lib is not None or _load_failed:
        return _lib
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name("libpkgroup.so")))
        lib.pkgroup_run.argtypes = [ctypes.c_void_p] * 15 + [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.pkgroup_run.restype = ctypes.c_int
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
    if (_load() is None or indices.dtype != torch.int32 or values.dtype != torch.bfloat16
            or counts.dtype != torch.int32 or ids.dtype != torch.int32 or part.dtype != torch.float32
            or tuple(indices.shape) != (topk + 2, I) or tuple(values.shape) != (topk + 2, I)
            or tuple(part.shape) != (topk + 2, H) or not indices.is_contiguous()
            or not values.is_contiguous() or not part.is_contiguous() or not ids.is_contiguous()):
        return False
    planes = tuple(routed_planes[:5]) + tuple(shared_planes[:5])
    if any(int(p.data_ptr()) % 16 for p in planes):
        return False
    rc = _lib.pkgroup_run(indices.data_ptr(), values.data_ptr(), counts.data_ptr(),
                          *(p.data_ptr() for p in planes), ids.data_ptr(), part.data_ptr(),
                          topk, int(bool(pdl)), torch.cuda.current_stream(part.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"pkgroup_run failed: {rc}")
    return True

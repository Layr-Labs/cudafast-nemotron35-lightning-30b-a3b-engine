# SPDX-License-Identifier: Apache-2.0
"""Optional exact scale preparation; only official remote startup may execute it.

This module supplies no production admission. Its caller must compare the
actual original653 and new full outputs, guards, statistics and changed graphs.
"""
import ctypes
from pathlib import Path

import torch

_lib = None
_failed = False
_devices = set()
E, G, N, K = 128, 116, 2688, 1856


def prepare(device):
    global _lib, _failed
    if _failed or torch.cuda.is_current_stream_capturing():
        return False
    if torch.cuda.get_device_capability(device) != (12, 1):
        return False
    try:
        if _lib is None:
            lib = ctypes.CDLL(str(Path(__file__).with_name("libscaleready.so")))
            P, LL, I = ctypes.c_void_p, ctypes.c_longlong, ctypes.c_int
            lib.scaleready_init.argtypes = []
            lib.scaleready_init.restype = I
            lib.scaleready_pack.argtypes = [P, P, I, LL, LL, LL, P]
            lib.scaleready_pack.restype = I
            lib.scaleready_down.argtypes = [P, LL, I, P, P, P, I, P, P, P, P,
                                            LL, LL, LL, LL, LL, LL, LL, I, P, P, P, I]
            lib.scaleready_down.restype = I
            _lib = lib
        with torch.cuda.device(device):
            index = torch.cuda.current_device()
            if index not in _devices:
                if _lib.scaleready_init():
                    return False
                _devices.add(index)
        return True
    except (OSError, AttributeError):
        _failed = True
        return False


def pack(original, result=None):
    """Immutable derived scale plane; original checkpoint and repack retained."""
    if (torch.cuda.is_current_stream_capturing() or original.layout != "swap2"
            or original.s.dtype != torch.uint8 or original.w.dtype != torch.uint8
            or original.gs.dtype != torch.float32 or original.s.numel() != E * G * N
            or original.w.numel() != E * K * (N // 2) or original.gs.numel() != E
            or (original.w_expert, original.w_tile, original.w_k) != (K * (N // 2), 64, N // 2)
            or (original.s_expert, original.s_tile, original.s_g) != (G * N, 128, N)
            or not all(x.is_cuda and x.is_contiguous() for x in (original.w, original.s, original.gs))
            or not (original.w.device == original.s.device == original.gs.device)
            or any(x.data_ptr() % 16 for x in (original.w, original.s, original.gs))
            or not prepare(original.w.device)):
        raise ValueError("scaleready unsupported original packed owner")
    if result is None:
        result = torch.empty(2 * original.s.numel(), device=original.s.device, dtype=torch.uint8)
    if (result.dtype != torch.uint8 or result.numel() != 2 * original.s.numel()
            or result.device != original.s.device or not result.is_contiguous()
            or result.data_ptr() % 16):
        raise ValueError("scaleready unsupported derived scale storage")
    rc = _lib.scaleready_pack(original.s.data_ptr(), result.data_ptr(), E,
                              original.s_expert, original.s_tile, original.s_g,
                              torch.cuda.current_stream(original.s.device).cuda_stream)
    if rc:
        raise RuntimeError("scaleready pack failed: " + str(rc))
    return result


def down(a, sorted_ids, expert_ids, ntp, original, scales, out,
         mode=1, max_blocks=None, stats=None, verify_rows=None, pdl=False):
    if _lib is None or a.device.index not in _devices:
        raise RuntimeError("scaleready not prepared")
    rows = a.shape[0]
    if max_blocks is None:
        max_blocks = rows
    tensors = (a, sorted_ids, expert_ids, ntp, original.w, original.s, original.gs, scales, out)
    if (rows < 6 or rows > 48 or rows % 6 or a.dtype != torch.bfloat16
            or a.ndim != 2 or a.shape[1] != K or a.stride(1) != 1 or a.stride(0) % 8
            or out.dtype != torch.bfloat16 or tuple(out.shape) != (rows, N)
            or out.stride(1) != 1 or out.stride(0) % 2
            or sorted_ids.dtype != torch.int32 or sorted_ids.numel() < max_blocks * 16
            or expert_ids.dtype != torch.int32 or expert_ids.numel() < max_blocks
            or ntp.dtype != torch.int32 or ntp.numel() != 1
            or scales.dtype != torch.uint8 or scales.numel() != 2 * E * G * N
            or not all(x.is_contiguous() for x in (sorted_ids, expert_ids, ntp, scales))
            or not all(x.is_cuda and x.device == a.device for x in tensors)
            or a.data_ptr() % 16 or scales.data_ptr() % 16
            or mode not in (0, 1) or not 1 <= max_blocks <= rows):
        raise ValueError("scaleready unsupported input layout")
    if stats is not None and (stats.dtype != torch.int32 or stats.numel() != 4
                              or stats.device != a.device or not stats.is_contiguous()):
        raise ValueError("scaleready unsupported statistics")
    if verify_rows is not None and (rows != 30 or mode != 1
                                    or verify_rows.dtype != torch.int32 or verify_rows.numel() != 1
                                    or verify_rows.device != a.device or not verify_rows.is_contiguous()):
        raise ValueError("scaleready unsupported live-row prefix")
    rc = _lib.scaleready_down(a.data_ptr(), a.stride(0), rows, sorted_ids.data_ptr(),
                              expert_ids.data_ptr(), ntp.data_ptr(), max_blocks,
                              original.w.data_ptr(), scales.data_ptr(), original.gs.data_ptr(),
                              out.data_ptr(), out.stride(0), original.w_expert,
                              original.w_tile, original.w_k, 2 * original.s_expert,
                              2 * original.s_tile, 2 * original.s_g, mode,
                              stats.data_ptr() if stats is not None else None,
                              torch.cuda.current_stream(a.device).cuda_stream,
                              verify_rows.data_ptr() if verify_rows is not None else None, int(pdl))
    if rc:
        # Do not change the original zskip PDL flag on an optional-path failure.
        raise RuntimeError("scaleready down failed: " + str(rc))
    return out

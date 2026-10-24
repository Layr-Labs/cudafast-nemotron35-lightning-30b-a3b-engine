"""Guarded FP8 N32 kernel used by the target Mamba input projection."""

import ctypes
import os

import torch

__all__ = ["maybe_apply", "load_error"]

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libfp8n32.so")
ENABLED = os.environ.get("FP8N32", "1") != "0"
_lib = None
_error = None


def _load():
    global _lib, _error
    if not ENABLED:
        return None
    if _lib is not None or _error is not None:
        return _lib
    try:
        lib = ctypes.CDLL(LIB_PATH)
        lib.fp8n32_run.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
        ]
        lib.fp8n32_run.restype = ctypes.c_int
        _lib = lib
    except OSError as exc:
        _error = str(exc)
    return _lib


_load()


def load_error():
    _load()
    return _error


def maybe_apply(a, b, a_scale, b_scale):
    """BF16 output for one exact supported four-row input projection, else None."""
    if (
        torch.compiler.is_compiling()
        or _load() is None
        or a.shape != (4, 2688)
        or b.shape != (2688, 10304)
        or a.dtype != torch.float8_e4m3fn
        or b.dtype != torch.float8_e4m3fn
        or a_scale.dtype != torch.float32
        or b_scale.dtype != torch.float32
        or a_scale.numel() != 1
        or b_scale.numel() != 1
        or not a.is_cuda
        or not b.is_cuda
        or a.device != b.device
        or a.stride(1) != 1
        or a.stride(0) != 2688
        or b.stride(0) != 1
        or b.stride(1) != 10304
    ):
        return None
    device = a.device
    out = torch.empty((4, 10304), dtype=torch.bfloat16, device=device)
    rc = _lib.fp8n32_run(
        out.data_ptr(), a.data_ptr(), b.data_ptr(),
        a_scale.data_ptr(), b_scale.data_ptr(),
        4, 10304, 2688, b.stride(1),
        device.index if device.index is not None else torch.cuda.current_device(),
        torch.cuda.current_stream(device).cuda_stream,
    )
    if rc != 0:
        raise RuntimeError(f"fp8n32_run failed: {rc}")
    return out

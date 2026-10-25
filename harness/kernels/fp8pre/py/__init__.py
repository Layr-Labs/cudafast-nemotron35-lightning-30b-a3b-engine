# SPDX-License-Identifier: Apache-2.0
"""Exact native FP8 projection kernel for prefill row counts; reads the original weight."""

import ctypes
from pathlib import Path

import torch

SHAPES = {(10304, 2688), (2688, 4096)}
_lib = None
_load_failed = False
_ready_devices = set()


def _load():
    global _lib, _load_failed
    if _lib is not None or _load_failed:
        return _lib
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name("libfp8pre.so")))
        lib.fp8pre_init.argtypes = []
        lib.fp8pre_init.restype = ctypes.c_int
        lib.fp8pre_run.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        lib.fp8pre_run.restype = ctypes.c_int
        _lib = lib
    except OSError:
        _load_failed = True
    return _lib


def supports(weight: torch.Tensor) -> bool:
    """A column-major [K,N] FP8 weight of a supported shape on GB10, with the library ready."""
    if not weight.is_cuda or weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
        return False
    k, n = weight.shape
    if (n, k) not in SHAPES or weight.stride() != (1, k) or weight.data_ptr() % 16:
        return False
    if torch.cuda.get_device_capability(weight.device) != (12, 1) or _load() is None:
        return False
    with torch.cuda.device(weight.device):
        index = torch.cuda.current_device()
        if index not in _ready_devices:
            rc = _lib.fp8pre_init()
            if rc:
                raise RuntimeError(f"fp8pre initialization failed: {rc}")
            _ready_devices.add(index)
    return True


def gemm(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
    """a [M,K] row-major E4M3, b the original column-major [K,N] E4M3 weight."""
    k, n = b.shape
    m = a.shape[0]
    out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    rc = _lib.fp8pre_run(
        a.data_ptr(), b.data_ptr(), sa.data_ptr(), sb.data_ptr(), out.data_ptr(),
        m, n, k, torch.cuda.current_stream(a.device).cuda_stream,
    )
    if rc:
        raise RuntimeError(f"fp8pre launch failed: {rc}, M={m}, N={n}, K={k}")
    return out

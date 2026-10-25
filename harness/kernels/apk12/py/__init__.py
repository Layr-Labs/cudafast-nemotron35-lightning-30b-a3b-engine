# SPDX-License-Identifier: Apache-2.0
"""Exact BF16 small-M attention projections streamed from the lossless 12-bit codec planes."""

import ctypes
import os
from pathlib import Path

import torch

SHAPES = {(4608, 2688), (2688, 4096)}
_RING = int(os.environ.get("APK12_RING", "2"))
_lib = None
_load_failed = False
_ready_devices = set()
# Codec plane [n, k] -> (n/32, 4 warps, 8 columns, k/64, step j, half h, t, pair p) ->
# (tile, group, warp, lane = 4 * column + t, byte 4j + 2h + p): K = 64G + 16j + 8h + 2t + p.
_ORDER = (0, 3, 1, 2, 6, 4, 5, 7)


def _load():
    global _lib, _load_failed
    if _lib is not None or _load_failed:
        return _lib
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name("libapk12.so")))
        lib.apk12_init.argtypes = []
        lib.apk12_init.restype = ctypes.c_int
        lib.apk12_run.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
        lib.apk12_run.restype = ctypes.c_int
        lib.apk12_check.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
        lib.apk12_check.restype = ctypes.c_int
        _lib = lib
    except OSError:
        _load_failed = True
    return _lib


def ready(device) -> bool:
    """Load the library and its kernels on device (before any graph capture)."""
    device = torch.device(device)
    if torch.cuda.get_device_capability(device) != (12, 1) or _load() is None:
        return False
    with torch.cuda.device(device):
        index = torch.cuda.current_device()
        if index not in _ready_devices:
            if _lib.apk12_init():
                return False
            _ready_devices.add(index)
    return True


def pack(planes, n: int, k: int):
    """The codec planes (sm, ex, base, ri, raw) of an [n, k] weight in the kernel's order:
    sign+mantissa bytes and code nibbles per lane, then one base byte per column and
    128-wide block (0 for an escaped block, whose bits stay in raw at ri)."""
    sm, ex, base, ri, raw = planes
    shape = (n // 32, 4, 8, k // 64, 4, 2, 4, 2)
    sm4 = sm.view(shape).permute(_ORDER).contiguous().view(-1)
    codes = torch.stack((ex & 15, ex >> 4), dim=-1).view(shape).permute(_ORDER).reshape(-1, 2)
    ex2 = (codes[:, 0] | (codes[:, 1] << 4)).contiguous()
    meta = torch.where(ri >= 0, 0, base).view(n // 32, 32, k // 128).transpose(1, 2).contiguous()
    return sm4, ex2, meta.view(-1), ri, raw


def _stream(t: torch.Tensor) -> int:
    return torch.cuda.current_stream(t.device).cuda_stream


def check(cp, w: torch.Tensor, n: int, k: int) -> int:
    """Registers of the kernel's decoded fragments that differ from w's bits (0: exact)."""
    bad = torch.zeros(1, dtype=torch.int32, device=w.device)
    rc = _lib.apk12_check(*(t.data_ptr() for t in cp), w.data_ptr(), bad.data_ptr(), n, k, _stream(w))
    if rc:
        raise RuntimeError(f"apk12 check launch failed: {rc}, N={n}, K={k}")
    return int(bad.item())


def gemm(x: torch.Tensor, cp, n: int, k: int, ring: int | None = None) -> torch.Tensor:
    """x @ W.T for the W that pack() laid out in cp; x is [1..8, k] BF16 with unit K stride."""
    if x.stride(0) % 2 or x.data_ptr() % 4:
        x = x.clone(memory_format=torch.contiguous_format)
    if ring is None:
        ring = _RING
    y = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
    rc = _lib.apk12_run(x.data_ptr(), *(t.data_ptr() for t in cp), y.data_ptr(),
                        x.shape[0], n, k, x.stride(0), ring, _stream(x))
    if rc:
        raise RuntimeError(f"apk12 launch failed: {rc}, M={x.shape[0]}, N={n}, K={k}, ring={ring}")
    return y

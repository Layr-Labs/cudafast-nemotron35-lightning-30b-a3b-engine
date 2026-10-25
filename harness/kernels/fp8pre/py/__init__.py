# SPDX-License-Identifier: Apache-2.0
"""Exact native FP8 projection kernel for prefill row counts; reads the original weight."""

import ctypes
from pathlib import Path

import torch

SHAPES = {(10304, 2688), (2688, 4096)}
_lib = None
_load_failed = False
# device index -> {(N, K): [(variant, "BMxBN sS"), ...]} of variants whose init succeeded
_ready: dict[int, dict[tuple[int, int], list[tuple[int, str]]]] = {}


def _load():
    global _lib, _load_failed
    if _lib is not None or _load_failed:
        return _lib
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name("libfp8pre.so")))
        lib.fp8pre_count.argtypes = []
        lib.fp8pre_count.restype = ctypes.c_int
        lib.fp8pre_info.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        lib.fp8pre_info.restype = ctypes.c_int
        lib.fp8pre_init.argtypes = [ctypes.c_int]
        lib.fp8pre_init.restype = ctypes.c_int
        lib.fp8pre_run.argtypes = ([ctypes.c_void_p] * 5 + [ctypes.c_int] * 4
                                   + [ctypes.c_void_p])
        lib.fp8pre_run.restype = ctypes.c_int
        _lib = lib
    except OSError:
        _load_failed = True
    return _lib


def _init_device() -> dict[tuple[int, int], list[tuple[int, str]]]:
    by_shape: dict[tuple[int, int], list[tuple[int, str]]] = {}
    info = (ctypes.c_int * 5)()
    for v in range(_lib.fp8pre_count()):
        if _lib.fp8pre_info(v, info) or _lib.fp8pre_init(v):
            continue
        n, k, bm, bn, stages = info
        by_shape.setdefault((n, k), []).append((v, f"{bm}x{bn} s{stages}"))
    return by_shape


def variants(weight: torch.Tensor) -> list[tuple[int, str]]:
    """(variant, label) pairs ready for this column-major [K,N] FP8 weight on GB10, else []."""
    if not weight.is_cuda or weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
        return []
    k, n = weight.shape
    if (n, k) not in SHAPES or weight.stride() != (1, k) or weight.data_ptr() % 16:
        return []
    if torch.cuda.get_device_capability(weight.device) != (12, 1) or _load() is None:
        return []
    with torch.cuda.device(weight.device):
        index = torch.cuda.current_device()
        if index not in _ready:
            _ready[index] = _init_device()
    return _ready[index].get((n, k), [])


def gemm(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor,
         variant: int) -> torch.Tensor:
    """a [M,K] row-major E4M3, b the original column-major [K,N] E4M3 weight."""
    k, n = b.shape
    m = a.shape[0]
    out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    rc = _lib.fp8pre_run(
        a.data_ptr(), b.data_ptr(), sa.data_ptr(), sb.data_ptr(), out.data_ptr(),
        m, n, k, variant, torch.cuda.current_stream(a.device).cuda_stream,
    )
    if rc:
        raise RuntimeError(f"fp8pre launch failed: {rc}, M={m}, N={n}, K={k}, variant={variant}")
    return out

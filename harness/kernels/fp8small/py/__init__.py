# SPDX-License-Identifier: Apache-2.0
"""Exact native FP8 small-M projection kernels; weights are packed once at load."""

import ctypes
import os
from pathlib import Path

import torch
from vllm.triton_utils import tl, triton

SHAPES = {(10304, 2688), (2688, 4096)}
_lib = None
_load_failed = False
_ready_devices = set()
# Programmatic dependent launch (fp8small_run_pdl): the weight half of the ring
# prologue overlaps the preceding kernel's tail. FP8SMALL_PDL=0 keeps the
# ordinary launch; a library built without it (build.sh fallback) does too.
_PDL_ENABLED = os.environ.get("FP8SMALL_PDL", "1") != "0"
_run = None
_init_pdl = None
_run_pdl = None


def _load():
    global _lib, _load_failed, _run, _init_pdl, _run_pdl
    if _lib is not None or _load_failed:
        return _lib
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name("libfp8small.so")))
        lib.fp8small_init.argtypes = []
        lib.fp8small_init.restype = ctypes.c_int
        lib.fp8small_run.argtypes = (
            [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        )
        lib.fp8small_run.restype = ctypes.c_int
        _lib = lib
        _run = lib.fp8small_run
        if _PDL_ENABLED:
            try:
                init_pdl = lib.fp8small_init_pdl
                run_pdl = lib.fp8small_run_pdl
            except AttributeError:  # built with -DFP8SMALL_NO_PDL
                init_pdl = run_pdl = None
            if run_pdl is not None:
                init_pdl.argtypes = []
                init_pdl.restype = ctypes.c_int
                run_pdl.argtypes = lib.fp8small_run.argtypes
                run_pdl.restype = ctypes.c_int
                _init_pdl, _run_pdl = init_pdl, run_pdl
    except OSError:
        _load_failed = True
    return _lib


@triton.jit
def _repack(src, dst, N: tl.constexpr, K: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    tile = i // ((K // 32) * 256)
    step = (i // 256) % (K // 32)
    warp = (i // 64) % 4
    lane = (i // 2) % 32
    word = i % 2
    col = tile * 32 + warp * 8 + lane // 4
    k4 = step * 8 + lane % 4 + word * 4
    value = tl.load(src + col * (K // 4) + k4, mask=i < N * K // 4, other=0)
    tl.store(dst + i, value, mask=i < N * K // 4)


def prepare(weight: torch.Tensor) -> torch.Tensor | None:
    """Pack a supported column-major [K,N] FP8 weight without changing any byte."""
    global _run
    if not weight.is_cuda or weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
        return None
    k, n = weight.shape
    if (n, k) not in SHAPES or weight.stride() != (1, k) or weight.data_ptr() % 16:
        return None
    if torch.cuda.get_device_capability(weight.device) != (12, 1) or _load() is None:
        return None
    with torch.cuda.device(weight.device):
        index = torch.cuda.current_device()
        if index not in _ready_devices:
            rc = _lib.fp8small_init()
            if rc:
                raise RuntimeError(f"fp8small initialization failed: {rc}")
            if _init_pdl is not None and _init_pdl() == 0:
                _run = _run_pdl
            _ready_devices.add(index)
        packed = torch.empty((n, k), dtype=torch.uint8, device=weight.device)
        _repack[(triton.cdiv(n * k // 4, 256),)](
            weight.t().view(torch.int32), packed.view(torch.int32), n, k
        )
    return packed


def gemm(
    a: torch.Tensor, packed: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor
) -> torch.Tensor:
    """Called only after the runtime guard and load-time preparation."""
    global _run
    n, k = packed.shape
    m = a.shape[0]
    out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    args = (
        a.data_ptr(),
        packed.data_ptr(),
        sa.data_ptr(),
        sb.data_ptr(),
        out.data_ptr(),
        m,
        n,
        k,
        torch.cuda.current_stream(a.device).cuda_stream,
    )
    rc = _run(*args)
    if rc and _run is _run_pdl and not torch.cuda.is_current_stream_capturing():
        # The PDL launch was rejected (the first calls run eagerly, before any
        # capture): use the ordinary launch from now on and redo this call.
        _run = _lib.fp8small_run
        rc = _run(*args)
    if rc:
        raise RuntimeError(f"fp8small launch failed: {rc}, M={m}, N={n}, K={k}")
    return out

# SPDX-License-Identifier: Apache-2.0
"""Prefill (M=496) Mamba in_proj and out_proj FP8 GEMMs with the pinned CUTLASS arithmetic.

The weight is packed once at load (a byte permutation, see README.md); the activation is the
producer's row-major tensor, unchanged. Callers admit a pack with a bitwise comparison against
the CUTLASS operator before using it (vllm .../scaled_mm/fp8ws.py).
"""

import ctypes
from pathlib import Path

import torch

M = 496
# (N, K): the in_proj and the out_proj
SHAPES = ((10304, 2688), (2688, 4096))
_lib = None
_load_failed = False
_ready_devices = set()


def _load():
    global _lib, _load_failed
    if _lib is not None or _load_failed:
        return _lib
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name("libfp8ws.so")))
        lib.fp8ws_init.argtypes = []
        lib.fp8ws_init.restype = ctypes.c_int
        lib.fp8ws_pack.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
        lib.fp8ws_pack.restype = ctypes.c_int
        lib.fp8ws_run.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        lib.fp8ws_run.restype = ctypes.c_int
        _lib = lib
    except OSError:
        _load_failed = True
    return _lib


def _ready(device: torch.device) -> bool:
    with torch.cuda.device(device):
        index = torch.cuda.current_device()
        if index not in _ready_devices:
            if _lib.fp8ws_init() != 0:
                return False
            _ready_devices.add(index)
    return True


def prepare(weight: torch.Tensor) -> torch.Tensor | None:
    """Pack a column-major [K, N] FP8 in_proj or out_proj weight; None if unsupported. No byte
    changes value."""
    if not weight.is_cuda or weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
        return None
    k, n = weight.shape
    if (n, k) not in SHAPES or weight.stride() != (1, k) or weight.data_ptr() % 16:
        return None
    if torch.cuda.get_device_capability(weight.device) != (12, 1) or _load() is None:
        return None
    if not _ready(weight.device):
        return None
    packed = torch.empty((n * k,), dtype=torch.uint8, device=weight.device)
    stream = torch.cuda.current_stream(weight.device).cuda_stream
    if _lib.fp8ws_pack(weight.data_ptr(), packed.data_ptr(), n, k, stream) != 0:
        return None
    return packed


def unpack_reference(packed: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Inverse permutation of the pack as a pure tensor reshape ([n, k] uint8), for proofs."""
    kt, bn, cpr = 128, 112, 8
    p = packed.view(n // bn, k // kt, bn, cpr, 16)
    r = torch.arange(bn, device=packed.device).view(bn, 1)
    c = torch.arange(cpr, device=packed.device).view(1, cpr)
    # pack chunk c' of row r holds logical chunk c' ^ (r & 7); invert by gathering c ^ (r & 7)
    src = (c ^ (r & 7)).view(1, 1, bn, cpr, 1).expand(n // bn, k // kt, bn, cpr, 16)
    logical = torch.gather(p, 3, src)
    return logical.permute(0, 2, 1, 3, 4).reshape(n, k)


def gemm(a: torch.Tensor, packed: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, n: int) -> torch.Tensor:
    """C[496, n] in BF16 (k = a.shape[1]); called only after the caller's shape/dtype guard."""
    out = torch.empty((M, n), device=a.device, dtype=torch.bfloat16)
    rc = _lib.fp8ws_run(a.data_ptr(), packed.data_ptr(), sa.data_ptr(), sb.data_ptr(), out.data_ptr(),
                        M, n, a.shape[1], torch.cuda.current_stream(a.device).cuda_stream)
    if rc:
        raise RuntimeError(f"fp8ws launch failed: {rc}")
    return out

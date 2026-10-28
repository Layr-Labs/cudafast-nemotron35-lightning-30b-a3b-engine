# SPDX-License-Identifier: Apache-2.0
"""Prefill (M=496) attention projection BF16 GEMMs with the cuBLAS arithmetic.

The weight is packed once at load (a byte permutation, see README.md); the activation is the
producer's row-major tensor, unchanged. Callers admit a pack with a bitwise comparison against
F.linear before using it (vllm .../models/nemotron_h_attn_packed.py).
"""

import ctypes
from pathlib import Path

import torch

M = 496
# (N, K) in BF16 values and the tile width BN: the qkv_proj and the o_proj
SHAPES = {(4608, 2688): 96, (2688, 4096): 112}
_lib = None
_load_failed = False
_ready_devices = set()
_twin_ready_devices = set()


def _load():
    global _lib, _load_failed
    if _lib is not None or _load_failed:
        return _lib
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name("libbf16ws.so")))
        lib.bf16ws_init.argtypes = []
        lib.bf16ws_init.restype = ctypes.c_int
        lib.bf16ws_pack.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
        lib.bf16ws_pack.restype = ctypes.c_int
        lib.bf16ws_run.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        lib.bf16ws_run.restype = ctypes.c_int
        _lib = lib
    except OSError:
        _load_failed = True
    return _lib


def _ready(device: torch.device) -> bool:
    with torch.cuda.device(device):
        index = torch.cuda.current_device()
        if index not in _ready_devices:
            if _lib.bf16ws_init() != 0:
                return False
            _ready_devices.add(index)
    return True


def prepare(weight: torch.Tensor) -> torch.Tensor | None:
    """Pack a row-major [N, K] BF16 projection weight; None if unsupported. No byte changes value."""
    if not weight.is_cuda or weight.dtype != torch.bfloat16 or weight.ndim != 2:
        return None
    n, k = weight.shape
    if (n, k) not in SHAPES or not weight.is_contiguous() or weight.data_ptr() % 16:
        return None
    if torch.cuda.get_device_capability(weight.device) != (12, 1) or _load() is None:
        return None
    if not _ready(weight.device):
        return None
    packed = torch.empty((n * k * 2,), dtype=torch.uint8, device=weight.device)
    stream = torch.cuda.current_stream(weight.device).cuda_stream
    if _lib.bf16ws_pack(weight.data_ptr(), packed.data_ptr(), n, k, stream) != 0:
        return None
    return packed


def unpack_reference(packed: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Inverse permutation of the pack as a pure tensor reshape ([n, 2k] uint8), for proofs."""
    kt, cpr = 128, 8
    bn = SHAPES[(n, k)]
    kb = 2 * k
    p = packed.view(n // bn, kb // kt, bn, cpr, 16)
    r = torch.arange(bn, device=packed.device).view(bn, 1)
    c = torch.arange(cpr, device=packed.device).view(1, cpr)
    # pack chunk c' of row r holds logical chunk c' ^ (r & 7); invert by gathering c ^ (r & 7)
    src = (c ^ (r & 7)).view(1, 1, bn, cpr, 1).expand(n // bn, kb // kt, bn, cpr, 16)
    logical = torch.gather(p, 3, src)
    return logical.permute(0, 2, 1, 3, 4).reshape(n, kb)


def gemm(a: torch.Tensor, packed: torch.Tensor, n: int) -> torch.Tensor:
    """C[496, n] in BF16 (k = a.shape[1]); called only after the caller's shape/dtype guard."""
    out = torch.empty((M, n), device=a.device, dtype=torch.bfloat16)
    rc = _lib.bf16ws_run(a.data_ptr(), packed.data_ptr(), out.data_ptr(), M, n, a.shape[1],
                         torch.cuda.current_stream(a.device).cuda_stream)
    if rc:
        raise RuntimeError(f"bf16ws launch failed: {rc}")
    return out


def twin_ready(device: torch.device) -> bool:
    """Try the optional QKV kernel without changing the original admission."""
    if _load() is None or not _ready(device):
        return False
    try:
        init = _lib.bf16ws_twin_init
        run = _lib.bf16ws_twin_run
        init.argtypes, init.restype = [], ctypes.c_int
        run.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        run.restype = ctypes.c_int
        with torch.cuda.device(device):
            index = torch.cuda.current_device()
            if index not in _twin_ready_devices:
                if init() != 0:
                    return False
                _twin_ready_devices.add(index)
        return True
    except AttributeError:
        return False


def gemm_twin(a: torch.Tensor, packed: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """QKV only; callers must first qualify this path and guard all operands."""
    if out is None:
        out = torch.empty((M, 4608), device=a.device, dtype=torch.bfloat16)
    rc = _lib.bf16ws_twin_run(a.data_ptr(), packed.data_ptr(), out.data_ptr(), M, 4608, 2688,
                              torch.cuda.current_stream(a.device).cuda_stream)
    if rc:
        raise RuntimeError(f"bf16ws twin launch failed: {rc}")
    return out

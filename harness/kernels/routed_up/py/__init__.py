"""Small-M NVFP4 routed-up GEMM/ReLU2. Source-only experimental native path.

Load-time byte permutation retains all original nibbles and scale bytes.
The CUDA kernel executes every K group; no zero pruning or K split.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

import torch

N, K, E = 1856, 2688, 128
_lib = None
_load_error = None
_ready = False


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    try:
        lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "librouted_up.so"))
        lib.routed_up_init.argtypes = []
        lib.routed_up_init.restype = ctypes.c_int
        P, LL, I = ctypes.c_void_p, ctypes.c_longlong, ctypes.c_int
        lib.routed_up_run.argtypes = [P, LL, I, P, P, P, I, P, P, P, P, LL, P]
        lib.routed_up_run.restype = ctypes.c_int
        _lib = lib
    except (OSError, AttributeError) as exc:
        _load_error = str(exc)
    return _lib


def available():
    return _load() is not None


def init():
    global _ready
    if _ready:
        return True
    lib = _load()
    if lib is None or torch.cuda.is_current_stream_capturing():
        return False
    rc = lib.routed_up_init()
    if rc:
        raise RuntimeError(f"routed_up_init: {rc}")
    _ready = True
    return True


@dataclass
class Packed:
    w: torch.Tensor
    s: torch.Tensor
    gs: torch.Tensor


@torch.no_grad()
def repack(weight, scales, global_scale):
    assert weight.dtype == torch.uint8 and tuple(weight.shape) == (E, N, K // 2)
    assert scales.dtype == torch.float8_e4m3fn and tuple(scales.shape) == (E, N, K // 16)
    sc = scales.view(torch.uint8)
    gs = global_scale.detach().reshape(-1).to(torch.float32).contiguous()
    assert gs.numel() == E
    wout = torch.empty(E, K, N // 2, dtype=torch.uint8, device=weight.device)
    sout = torch.empty(E, K // 16, N, dtype=torch.uint8, device=weight.device)
    for e0 in range(0, E, 8):
        w = weight[e0:e0 + 8]
        # a1 zskip "swap" layout generalized to the up projection dimensions:
        # n=64T+16c+8h+g -> byte32T+4g+c, nibble h.
        nib = torch.stack((w & 15, w >> 4), dim=-1).reshape(8, N, K)
        x = nib.transpose(1, 2).reshape(8, K, N // 64, 4, 2, 8).permute(0, 1, 2, 5, 3, 4)
        wout[e0:e0 + 8] = (x[..., 0] | (x[..., 1] << 4)).reshape(8, K, N // 2)
        sx = sc[e0:e0 + 8].transpose(1, 2).reshape(8, K // 16, N // 64, 4, 2, 8).permute(0, 1, 2, 5, 3, 4)
        sout[e0:e0 + 8] = sx.reshape(8, K // 16, N)
        del nib, x, sx
    return Packed(wout, sout, gs)


def up(inputs, sorted_ids, expert_ids, ntp, packed, outputs):
    lib = _load()
    assert lib is not None and _ready
    m = inputs.shape[0]
    rc = lib.routed_up_run(
        inputs.data_ptr(), inputs.stride(0), m,
        sorted_ids.data_ptr(), expert_ids.data_ptr(), ntp.data_ptr(),
        min(m * 6, expert_ids.numel()), packed.w.data_ptr(), packed.s.data_ptr(),
        packed.gs.data_ptr(), outputs.data_ptr(), outputs.stride(0),
        torch.cuda.current_stream(inputs.device).cuda_stream,
    )
    if rc:
        raise RuntimeError(f"routed_up_run: {rc}")
    return outputs

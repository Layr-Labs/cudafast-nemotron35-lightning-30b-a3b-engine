"""Humming-exact routed up (w13) for 1..8 token verifies.

Checkpoint NVFP4 w13 (uint8 [E, N, K/2], e4m3 [E, N, K/16], fp32 [E]) is repacked
with zskip's swap layout at N=1856, K=2688. libupskip.so runs the same mma chain
as zskip's down4, then humming's fused relu^2 epilogue. UPSKIP=0 leaves humming.
"""

from __future__ import annotations

import ctypes
import os

import torch

__all__ = ["available", "load_error", "repack", "up", "Packed", "K", "N", "G"]

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libupskip.so")

K = 2688
N = 1856
G = K // 16
NUM_EXPERTS = 128

_lib = None
_load_error: str | None = None


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    try:
        lib = ctypes.CDLL(LIB_PATH)
        P, LL, I = ctypes.c_void_p, ctypes.c_longlong, ctypes.c_int
        lib.upskip_run.argtypes = [
            P, LL, I, P, P, P, I, P, P, P, P, LL, LL, LL, LL, LL, P,
        ]
        lib.upskip_run.restype = ctypes.c_int
        _lib = lib
    except OSError as exc:
        _load_error = str(exc)
    return _lib


def available() -> bool:
    return _load() is not None


def load_error() -> str | None:
    _load()
    return _load_error


class Packed:
    def __init__(self, w: torch.Tensor, s: torch.Tensor, gs: torch.Tensor, strides: tuple):
        self.w, self.s, self.gs = w, s, gs
        self.w_expert, self.w_k, self.s_expert, self.s_g = strides


@torch.no_grad()
def repack(w13_weight: torch.Tensor, w13_weight_scale: torch.Tensor, w13_global_scale: torch.Tensor,
           chunk: int = 16) -> Packed:
    """zskip swap layout at this shape. Input-independent; every checkpoint byte kept."""
    E = w13_weight.shape[0]
    assert w13_weight.dtype == torch.uint8 and tuple(w13_weight.shape) == (E, N, K // 2), w13_weight.shape
    sc = w13_weight_scale.view(torch.uint8)
    assert tuple(sc.shape) == (E, N, G), sc.shape
    if bool(((sc & 0x7F) == 0x7F).any()):
        raise ValueError("upskip: NaN e4m3 scale in w13")
    gs = w13_global_scale.reshape(-1).to(torch.float32).contiguous()
    assert gs.numel() == E
    dev = w13_weight.device
    W = torch.empty(E, K, N // 2, dtype=torch.uint8, device=dev)
    S = torch.empty(E, G, N, dtype=torch.uint8, device=dev)
    for e0 in range(0, E, chunk):
        e1 = min(E, e0 + chunk)
        w = w13_weight[e0:e1]
        nib = torch.stack([w & 0xF, w >> 4], dim=-1).reshape(e1 - e0, N, K)
        # n = 64T + 16c + 8h + g8 -> byte 32T + 4g8 + c, nibble h. Same map as zskip swap.
        x = nib.transpose(1, 2).reshape(e1 - e0, K, N // 64, 4, 2, 8).permute(0, 1, 2, 5, 3, 4)
        W[e0:e1] = (x[..., 0] | (x[..., 1] << 4)).reshape(e1 - e0, K, N // 2)
        sx = sc[e0:e1].transpose(1, 2).reshape(e1 - e0, G, N // 64, 4, 2, 8).permute(0, 1, 2, 5, 3, 4)
        S[e0:e1] = sx.reshape(e1 - e0, G, N)
        del nib, x, sx
    return Packed(W, S, gs, (K * (N // 2), N // 2, G * N, N))


def up(hidden: torch.Tensor, sorted_ids: torch.Tensor, expert_ids: torch.Tensor,
       num_tokens_padded: torch.Tensor, packed: Packed, out: torch.Tensor) -> torch.Tensor:
    """out[r, :] = relu^2(w13(hidden[r // 6])) for the moe_align block-16 layout."""
    lib = _load()
    assert lib is not None, _load_error
    M = hidden.shape[0]
    assert hidden.dtype == torch.bfloat16 and hidden.shape[1] == K and hidden.stride(1) == 1
    assert out.dtype == torch.bfloat16 and out.shape[1] == N and out.stride(1) == 1
    assert out.shape[0] >= M * 6
    assert sorted_ids.dtype == torch.int32 and expert_ids.dtype == torch.int32
    assert num_tokens_padded.dtype == torch.int32 and num_tokens_padded.numel() >= 1
    max_blocks = int(expert_ids.numel())
    rc = lib.upskip_run(
        hidden.data_ptr(), hidden.stride(0), M,
        sorted_ids.data_ptr(), expert_ids.data_ptr(), num_tokens_padded.data_ptr(),
        max_blocks, packed.w.data_ptr(), packed.s.data_ptr(), packed.gs.data_ptr(),
        out.data_ptr(), out.stride(0), packed.w_expert, packed.w_k, packed.s_expert, packed.s_g,
        torch.cuda.current_stream(hidden.device).cuda_stream,
    )
    if rc != 0:
        raise RuntimeError(f"upskip_run failed: {rc}")
    return out

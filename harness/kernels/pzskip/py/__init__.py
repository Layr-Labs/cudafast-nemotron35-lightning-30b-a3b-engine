"""pzskip: humming-exact routed-expert down projection for prefill-sized calls (expert blocks
of up to 64 rows) that does not load the w2 rows whose relu^2 activations are zero in every row
of the block. Reads zskip's "swap" repack (layer.zskip_w2); see README.md.

``PZSKIP=0`` disables it.
"""

from __future__ import annotations

import ctypes
import os

import torch

__all__ = ["LIB_PATH", "available", "down", "BM"]

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libpzskip.so")
ENABLED = os.environ.get("PZSKIP", "1") != "0"
WPC = int(os.environ.get("PZSKIP_WPC", "6"))
BM = 64

_lib = None
_load_error: str | None = None
_scratch: dict = {}


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    try:
        lib = ctypes.CDLL(LIB_PATH)
        P, LL, I = ctypes.c_void_p, ctypes.c_longlong, ctypes.c_int
        lib.pzskip_down.argtypes = [P, LL, I, P, P, P, P, P, P, P, LL, LL, LL, LL, LL, P, I, I, P, I, P]
        lib.pzskip_down.restype = ctypes.c_int
        lib.pzskip_bi_bytes.argtypes = []
        lib.pzskip_bi_bytes.restype = ctypes.c_int
        _lib = lib
    except OSError as exc:
        _load_error = str(exc)
    return _lib


def available() -> bool:
    return ENABLED and _load() is not None


def max_blocks(shape_m: int, num_experts: int) -> int:
    return (shape_m + num_experts * (BM - 1) + BM - 1) // BM


def scratch(device: torch.device, blocks: int) -> torch.Tensor:
    """Per-device block-info workspace; allocate outside CUDA graph capture."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    ws = _scratch.get(idx)
    need = blocks * _load().pzskip_bi_bytes()
    if ws is None or ws.numel() < need:
        ws = torch.zeros(need, dtype=torch.uint8, device=device)
        _scratch[idx] = ws
    return ws


def down(a: torch.Tensor, sorted_ids: torch.Tensor, expert_ids: torch.Tensor,
         num_tokens_padded: torch.Tensor, packed, out: torch.Tensor, num_experts: int = 128,
         dense: bool = False, loadonly: bool = False, computeonly: bool = False, stats: torch.Tensor | None = None, wpc: int = WPC) -> torch.Tensor:
    """out[r] = humming's w2 row r for the moe_align layout with block 64 (rows = token * top_k + slot)."""
    lib = _load()
    assert lib is not None, _load_error
    blocks = min(max_blocks(a.shape[0], num_experts), sorted_ids.numel() // BM)
    ws = scratch(a.device, blocks)
    rc = lib.pzskip_down(
        a.data_ptr(), a.stride(0), a.shape[0], sorted_ids.data_ptr(), expert_ids.data_ptr(),
        num_tokens_padded.data_ptr(), packed.w.data_ptr(), packed.s.data_ptr(), packed.gs.data_ptr(),
        out.data_ptr(), out.stride(0), packed.w_expert, packed.w_k, packed.s_expert, packed.s_g,
        ws.data_ptr(), blocks, 3 if computeonly else (2 if loadonly else (1 if dense else 0)), stats.data_ptr() if stats is not None else None,
        wpc, torch.cuda.current_stream(a.device).cuda_stream,
    )
    if rc != 0:
        raise RuntimeError(f"pzskip_down failed: {rc}")
    return out

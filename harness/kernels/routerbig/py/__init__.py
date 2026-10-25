"""routerbig: the fp32 MoE router GEMM at prefill M with cuBLAS's exact summation tree (ctypes).

router_gemm(x, w) = x . w^T for x [M, 2688] fp32, w [128, 2688] fp32: twelve fma chains of 224
consecutive k (ascending, from 0) added left to right, as the cuBLAS split-K SIMT kernel that
torch.mm runs at M = 496 (see csrc/routerbig.cu). maybe_router_gemm returns None outside the
verified shapes; the caller then runs torch.mm.

ROUTERBIG=0 disables it.
"""

from __future__ import annotations

import ctypes
import os

import torch

__all__ = ["available", "load_error", "router_gemm", "maybe_router_gemm", "ON", "ROUTER_M"]

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "librouterbig.so")
ON = os.environ.get("ROUTERBIG", "1") != "0"
# M values whose cuBLAS kernel was probed (12 x 224 tree) and tested bit-identical.
ROUTER_M = frozenset({480, 496, 512})
N_EXPERTS, K_HIDDEN = 128, 2688

_lib = None
_load_error: str | None = None
_scratch: dict[tuple, torch.Tensor] = {}


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    try:
        lib = ctypes.CDLL(LIB_PATH)
        vp, ci = ctypes.c_void_p, ctypes.c_int
        lib.routerbig_gemm.argtypes = [vp, vp, vp, vp, ci, ci, ci, vp]
        lib.routerbig_gemm.restype = ci
        lib.routerbig_version.restype = ci
        lib.routerbig_partials_only.argtypes = [vp, vp, vp, ci, ci, ci, vp]
        lib.routerbig_partials_only.restype = ci
        lib.routerbig_set_bm.argtypes = [ci]
        lib.routerbig_set_bm.restype = ci
        bm = int(os.environ.get("ROUTERBIG_BM", "0"))
        if bm:
            lib.routerbig_set_bm(bm)
        lib.routerbig_version()
        _lib = lib
    except OSError as e:
        _load_error = repr(e)
    return _lib


def available() -> bool:
    return _load() is not None


def load_error() -> str | None:
    _load()
    return _load_error


def router_gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    lib = _load()
    if lib is None:
        raise RuntimeError(f"routerbig unavailable: {_load_error}")
    m, k = x.shape
    n = w.shape[0]
    key = (x.device, m)
    p = _scratch.get(key)
    if p is None:
        p = _scratch[key] = torch.empty((12, m, n), dtype=torch.float32, device=x.device)
    y = torch.empty((m, n), dtype=torch.float32, device=x.device)
    stream = torch.cuda.current_stream(x.device).cuda_stream
    rc = lib.routerbig_gemm(x.data_ptr(), w.data_ptr(), p.data_ptr(), y.data_ptr(), m, n, k,
                            ctypes.c_void_p(stream))
    if rc != 0:
        raise RuntimeError(f"routerbig_gemm failed: {rc}")
    return y


def set_bm(bm: int) -> None:
    """Rows per CTA of pass 1 (32, 64 or 128); every choice computes the same chains."""
    lib = _load()
    if lib.routerbig_set_bm(int(bm)) != 0:
        raise ValueError(bm)


def partials_only(x: torch.Tensor, w: torch.Tensor) -> None:
    """Timing aid: pass 1 alone into the scratch buffer."""
    lib = _load()
    m, k = x.shape
    n = w.shape[0]
    key = (x.device, m)
    p = _scratch.get(key)
    if p is None:
        p = _scratch[key] = torch.empty((12, m, n), dtype=torch.float32, device=x.device)
    stream = torch.cuda.current_stream(x.device).cuda_stream
    rc = lib.routerbig_partials_only(x.data_ptr(), w.data_ptr(), p.data_ptr(), m, n, k, ctypes.c_void_p(stream))
    if rc != 0:
        raise RuntimeError(f"routerbig_partials_only failed: {rc}")


def maybe_router_gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor | None:
    if not ON or x.dim() != 2 or w.dim() != 2 or x.shape[0] not in ROUTER_M:
        return None
    if (
        x.dtype != torch.float32
        or w.dtype != torch.float32
        or tuple(w.shape) != (N_EXPERTS, K_HIDDEN)
        or x.shape[1] != K_HIDDEN
        or not x.is_contiguous()
        or not w.is_contiguous()
        or not x.is_cuda
        or x.data_ptr() % 16
        or w.data_ptr() % 16
    ):
        return None
    if not available():
        return None
    return router_gemm(x, w)

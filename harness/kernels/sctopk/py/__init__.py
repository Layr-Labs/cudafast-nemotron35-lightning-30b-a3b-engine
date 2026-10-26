"""sctopk: the draft-head screen's per-chunk exact top-k (nemotron_h_mtp_screen._chunk_topk) as one
CUDA kernel per call; csrc/sctopk.cu documents the selection and the output order it reproduces.

``chunk_topk(scores, src, out, n, k, gather, gate)`` launches on the current stream (also under CUDA
graph capture) and returns False when the shape is not supported or the library is missing, so the
caller runs the Triton kernel instead. SCTOPK=0 disables it.
"""

from __future__ import annotations

import ctypes
import os

import torch

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libsctopk.so")
ENABLED = os.environ.get("SCTOPK", "1") != "0"
PDL = os.environ.get("SCTOPK_PDL", "1") != "0"
_lib = None
_err = None


def _load():
    global _lib, _err
    if _lib is not None or _err is not None:
        return _lib
    try:
        lib = ctypes.CDLL(LIB_PATH)
        vp, ci = ctypes.c_void_p, ctypes.c_int
        lib.sct_version.restype = ci
        lib.sct_chunk_topk.argtypes = [vp, vp, vp, ci, ci, ci, ci, vp, ci, vp]
        lib.sct_chunk_topk.restype = ci
        _lib = lib
    except OSError as exc:
        _err = str(exc)
    return _lib


def available() -> bool:
    return ENABLED and _load() is not None


def chunk_topk(scores: torch.Tensor, src: torch.Tensor | None, out: torch.Tensor, n: int, k: int,
               gather: bool, gate: torch.Tensor | None = None, pdl: bool | None = None) -> bool:
    lib = _load() if ENABLED else None
    if lib is None:
        return False
    c = scores.numel() // n
    rc = lib.sct_chunk_topk(
        scores.data_ptr(), src.data_ptr() if (gather and src is not None) else None, out.data_ptr(),
        c, n, k, int(bool(gather)), gate.data_ptr() if gate is not None else None,
        int(PDL if pdl is None else pdl), torch.cuda.current_stream(scores.device).cuda_stream)
    if rc == -1:
        return False
    if rc != 0:
        raise RuntimeError(f"sct_chunk_topk failed: {rc}")
    return True

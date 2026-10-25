"""routerbig: the fp32 MoE router GEMM at prefill M with cuBLAS's exact summation tree (ctypes).

router_gemm(x, w) = x . w^T for x [M, 2688] fp32, w [128, 2688] fp32: twelve fma chains of 224
consecutive k (ascending, from 0) added left to right, as the cuBLAS split-K SIMT kernel that
torch.mm runs at M = 496 (see csrc/routerbig.cu). maybe_router_gemm returns None outside the
verified shapes; the caller then runs torch.mm.

ROUTERBIG=0 disables it.

route_prefill(x, w, bias, ...) (ROUTERBIG_ROUTE=0 disables): pass 1, then one kernel for the
reduce, vLLM's single-group sigmoid top-k (smallk's exact copy) and moe_align_block_size for the
block sizes the align hook saw for this id count; it returns the logits and parks the top-k and
align outputs, which take_topk / take_align hand to the router and the align hook for the same
tensors. A miss anywhere leaves the stock path running on the (always written) logits.
"""

from __future__ import annotations

import ctypes
import os

import torch

__all__ = ["available", "load_error", "router_gemm", "maybe_router_gemm", "ON", "ROUTER_M", "route_prefill",
           "take_topk", "take_align"]

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "librouterbig.so")
ON = os.environ.get("ROUTERBIG", "1") != "0"
ROUTE_ON = ON and os.environ.get("ROUTERBIG_ROUTE", "1") != "0"
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
        cf = ctypes.c_float
        lib.routerbig_route.argtypes = [vp, ci, vp, ci, ci, cf, vp, vp, vp, ci, ci, vp, ci, vp, ci, vp,
                                        ci, vp, ci, vp, ci, vp, vp, vp]
        lib.routerbig_route.restype = ci
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


# ---------------------------------------------------------------------------- fused prefill route
_align_bs: dict[int, list[int]] = {}  # id count -> block sizes the align hook was asked for
# Parked outputs hold the very tensors they belong to (identity checks, no address reuse).
_parked_topk = None  # (logits, owner, values, indices)
_parked_align: dict = {}  # block size -> (topk_ids, (sorted_ids, expert_ids, ntpp))
_counters: dict = {}


def _align_buffers(numel: int, block_size: int, device):
    pad = numel + N_EXPERTS * (block_size - 1)  # as vllm moe_align_block_size sizes them
    if numel < N_EXPERTS:
        pad = min(numel * block_size, pad)
    blocks = (pad + block_size - 1) // block_size
    return (torch.empty((pad,), dtype=torch.int32, device=device),
            torch.empty((blocks,), dtype=torch.int32, device=device),
            torch.empty((1,), dtype=torch.int32, device=device))


def route_prefill(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, topk: int, renormalize: bool,
                  scale: float, owner: int) -> torch.Tensor | None:
    """Router logits for a prefill-sized gate call, with the top-k and align outputs parked; None
    outside the guard (the caller then runs the gate alone)."""
    global _parked_topk
    _parked_topk = None
    _parked_align.clear()
    if not ROUTE_ON or x.dim() != 2 or x.shape[0] not in ROUTER_M or bias is None:
        return None
    m = x.shape[0]
    if (
        x.dtype != torch.float32 or w.dtype != torch.float32 or bias.dtype != torch.float32
        or tuple(w.shape) != (N_EXPERTS, K_HIDDEN) or x.shape[1] != K_HIDDEN or bias.numel() != N_EXPERTS
        or not x.is_contiguous() or not w.is_contiguous() or not bias.is_contiguous()
        or not x.is_cuda or x.data_ptr() % 16 or w.data_ptr() % 16
        or not 1 <= topk <= 8 or m * topk > 4096 or torch.compiler.is_compiling()
    ):
        return None
    lib = _load()
    if lib is None:
        return None
    dev = x.device
    key = (dev, m)
    p = _scratch.get(key)
    if p is None:
        p = _scratch[key] = torch.empty((12, m, N_EXPERTS), dtype=torch.float32, device=dev)
    ctr = _counters.get(dev)
    if ctr is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        ctr = _counters[dev] = torch.zeros((1,), dtype=torch.int32, device=dev)
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    rc = lib.routerbig_partials_only(x.data_ptr(), w.data_ptr(), p.data_ptr(), m, N_EXPERTS, K_HIDDEN, stream)
    if rc != 0:
        raise RuntimeError(f"routerbig_partials_only failed: {rc}")
    y = torch.empty((m, N_EXPERTS), dtype=torch.float32, device=dev)
    vals = torch.empty((m, topk), dtype=torch.float32, device=dev)
    ids = torch.empty((m, topk), dtype=torch.int32, device=dev)
    numel = m * topk
    outs = [(bs, _align_buffers(numel, bs, dev)) for bs in _align_bs.get(numel, [])[:2]]
    args = []
    for i in range(2):
        if i < len(outs):
            bs, (srt, eid, ntp) = outs[i]
            args += [bs, srt.data_ptr(), srt.numel(), eid.data_ptr(), eid.numel(), ntp.data_ptr()]
        else:
            args += [0, None, 0, None, 0, None]
    rc = lib.routerbig_route(p.data_ptr(), m, bias.data_ptr(), topk, int(bool(renormalize)), float(scale),
                             y.data_ptr(), vals.data_ptr(), ids.data_ptr(), len(outs), *args, ctr.data_ptr(), stream)
    if rc != 0:
        raise RuntimeError(f"routerbig_route failed: {rc}")
    _parked_topk = (y, owner, vals, ids)
    for bs, bufs in outs:
        _parked_align[bs] = (ids, bufs)
    return y


def take_topk(logits: torch.Tensor, owner: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The parked (values, indices) of route_prefill for these logits and router, once; else None."""
    global _parked_topk
    parked, _parked_topk = _parked_topk, None
    if parked is None or parked[0] is not logits or parked[1] != owner:
        return None
    return parked[2], parked[3]


def take_align(topk_ids: torch.Tensor, num_experts: int, block_size: int):
    """The parked align outputs for these ids and block size, once; else None (and the block size
    is noted for later route_prefill calls with this id count)."""
    if num_experts != N_EXPERTS or topk_ids.dtype != torch.int32:
        return None
    numel = topk_ids.numel()
    hit = _parked_align.get(block_size)
    if hit is not None and hit[0] is topk_ids:
        del _parked_align[block_size]
        return hit[1]
    seen = _align_bs.setdefault(numel, [])
    if block_size not in seen and len(seen) < 2:
        seen.append(block_size)
    return None

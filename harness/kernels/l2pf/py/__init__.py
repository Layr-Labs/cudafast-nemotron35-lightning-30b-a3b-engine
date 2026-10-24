"""l2pf: L2 prefetch of a weight ahead of the bandwidth-bound GEMM that reads it.

``prefetch(t)`` queues bulk L2 prefetches of t's storage on a side stream forked
from the current stream (eager or under CUDA graph capture); ``join(device)``
makes the current stream wait for them, which a capture needs before it ends.
Only values already in memory move into L2: no kernel output can change.

``L2PF=0`` in the environment disables every prefetch.
"""

from __future__ import annotations

import ctypes
import os

import torch

__all__ = [
    "LIB_PATH", "ENABLED", "MAMBA_BYTES", "ATTN_BYTES", "MOE_BYTES", "DRAFT_BYTES",
    "AHEAD", "AHEAD_MOE", "DRAFT_SCREEN",
    "MOE_LAZY", "MAMBA_LAZY_BYTES", "MOE_EAGER_BYTES", "ROUTER_ON", "AHEAD_ROUTER", "prefetch", "join", "defer", "join_pending",
    "prefetch_router_after",
]

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libl2pf.so")
ENABLED = os.environ.get("L2PF", "1") != "0"
CHUNK = 1 << 16
# Bytes prefetched per window: what DRAM can deliver while the latency-bound
# kernels of that window run (more stalls the rejoin).
MAMBA_BYTES = int(float(os.environ.get("L2PF_MAMBA_MB", "5")) * (1 << 20))
ATTN_BYTES = int(float(os.environ.get("L2PF_ATTN_MB", "3")) * (1 << 20))
# Mamba window with a deferred rejoin (mamba_mixer2, FULL-graph capture only; 0 keeps
# the eager 5 MB window everywhere): the conv + SSM window (~25-30 us, DRAM idle)
# plus whatever spills into out_proj, which reads those bytes anyway.
MAMBA_LAZY_BYTES = int(float(os.environ.get("L2PF_MAMBA_LAZY_MB", "8")) * (1 << 20))
# The MoE window's prefetch rejoins lazily (L2PF_MOE_LAZY=0: right after the routed
# down projection, as before): the MoE op marks it pending (defer) and the next
# fp8small GEMM, the following Mamba layer's in_proj in the same CUDA-graph piece,
# rejoins once its own kernel is queued (join_pending). The prefetch then also runs
# through the latency-bound MoE tail (combine, shared add, residual + norm + FP8
# quant), where DRAM is idle, and never holds the main stream: a window sized past
# what DRAM delivers spills into in_proj, which reads those bytes anyway.
MOE_LAZY = os.environ.get("L2PF_MOE_LAZY", "1") != "0"
# The routed down projection of a small call (zskip) is latency-bound and leaves
# DRAM about half idle: prefetch the head of the next Mamba layer's in_proj then
# (6 MB with the eager rejoin; 8 MB lazily, the extra 2 MB for the MoE tail).
MOE_BYTES = int(float(os.environ.get("L2PF_MOE_MB", "8" if MOE_LAZY else "6")) * (1 << 20))
# A window that rejoins eagerly (no fp8small GEMM after the layer, e.g. the last MoE
# layer's lm_head head) keeps the eager size: an oversized eager window stalls.
MOE_EAGER_BYTES = int(float(os.environ.get("L2PF_MOE_EAGER_MB", "6")) * (1 << 20))
# Drafter window: the latency-bound down projection of the single-row MTP MoE
# prefetches the head of the draft screen's coarse matrix (0 disables).
DRAFT_BYTES = int(float(os.environ.get("L2PF_DRAFT_MB", "3")) * (1 << 20))
# The draft screen's coarse weight matrix (set by the screen at construction).
DRAFT_SCREEN: list[torch.Tensor] = []
# attention layer name -> the module whose .weight follows that attention
# (filled by the model at construction).
AHEAD: dict[str, torch.nn.Module] = {}
# MoE layer name -> the next layer's in_proj module (filled by the model).
AHEAD_MOE: dict[str, torch.nn.Module] = {}
# (Mamba out_proj, next MoE layer's router gate) pairs (filled by the model): the
# out_proj's fp8small op prefetches the fp32 router weight (1.38 MB) once its GEMM
# is done, into the latency-bound gap before the router GEMM (residual add, RMSNorm,
# fp32 cast), and the router op rejoins (gate_linear.smallk_router_gemm_impl). The
# router then reads L2 instead of sharing DRAM with the shared expert's up
# projection in the routing window. L2PF_ROUTER=0 disables.
ROUTER_ON = os.environ.get("L2PF_ROUTER", "1") != "0"
AHEAD_ROUTER: list[tuple[torch.nn.Module, torch.nn.Module]] = []
_router_by_pack: dict[int, torch.Tensor] = {}

_lib = None
_side: dict[int, torch.cuda.Stream] = {}
# devices whose side stream has prefetches not yet rejoined (defer)
_pending: set[int] = set()


def _load():
    global _lib
    try:
        lib = ctypes.CDLL(LIB_PATH)
    except OSError:  # library missing: every call is a no-op
        return None
    lib.l2pf_prefetch.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    lib.l2pf_prefetch.restype = ctypes.c_int
    _lib = lib
    return lib


def _index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    idx = _index(device)
    s = _side.get(idx)
    if s is None:
        s = torch.cuda.Stream(device=idx)
        _side[idx] = s
    return s


def prefetch(t: torch.Tensor | None, max_bytes: int = 0) -> bool:
    """Queue an L2 prefetch of the first max_bytes (0: all) of t's storage on the
    side stream. Returns whether one was queued."""
    if not ENABLED or _lib is None or t is None or not t.is_cuda:
        return False
    st = t.untyped_storage()
    nbytes = st.nbytes()
    if max_bytes:
        nbytes = min(nbytes, max_bytes)
    main = torch.cuda.current_stream(t.device)
    side = _side_stream(t.device)
    side.wait_stream(main)
    rc = _lib.l2pf_prefetch(st.data_ptr(), nbytes, CHUNK, side.cuda_stream)
    if rc != 0:
        raise RuntimeError(f"l2pf_prefetch failed: {rc}")
    return True


def join(device: torch.device) -> None:
    """Make the current stream wait for the side stream's prefetches."""
    torch.cuda.current_stream(device).wait_stream(_side_stream(device))
    _pending.discard(_index(device))


def defer(device: torch.device) -> None:
    """Leave the side stream's prefetches running; a later join or join_pending
    in the same capture rejoins them."""
    _pending.add(_index(device))


def prefetch_router_after(packed: torch.Tensor) -> bool:
    """Called by an fp8small op after queuing its GEMM on `packed`: if that GEMM is a
    Mamba out_proj followed by an MoE layer, prefetch the router weight (side stream,
    after the GEMM) and defer the rejoin to the router op."""
    if not ROUTER_ON or not AHEAD_ROUTER:
        return False
    key = packed.data_ptr()
    if key not in _router_by_pack and len(_router_by_pack) < len(AHEAD_ROUTER):
        for proj, gate in AHEAD_ROUTER:
            p = getattr(proj, "_fp8small_weight", None)
            w = getattr(gate, "weight", None)
            if p is not None and w is not None and w.is_cuda:
                _router_by_pack[p.data_ptr()] = w
    w = _router_by_pack.get(key)
    if w is None or not prefetch(w):
        return False
    defer(w.device)
    return True


def join_pending(device: torch.device) -> bool:
    """join(device) if a deferred prefetch is pending on it."""
    if not _pending or _index(device) not in _pending:
        return False
    join(device)
    return True


if ENABLED:
    _load()

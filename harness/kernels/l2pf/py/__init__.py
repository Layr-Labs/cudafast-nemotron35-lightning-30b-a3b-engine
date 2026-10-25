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
    "prefetch", "join", "prefetch_fp8c_out",
]

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libl2pf.so")
ENABLED = os.environ.get("L2PF", "1") != "0"
CHUNK = 1 << 16
# Bytes prefetched per window: what DRAM can deliver while the latency-bound
# kernels of that window run (more stalls the rejoin).
MAMBA_BYTES = int(float(os.environ.get("L2PF_MAMBA_MB", "5")) * (1 << 20))
ATTN_BYTES = int(float(os.environ.get("L2PF_ATTN_MB", "7")) * (1 << 20))
# The routed down projection of a small call (zskip) is latency-bound and leaves
# DRAM about half idle: prefetch the head of the next Mamba layer's in_proj then.
MOE_BYTES = int(float(os.environ.get("L2PF_MOE_MB", "12")) * (1 << 20))
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
# Previous Mamba/attention op -> the actual following target MoE router.
AHEAD_GATE: dict[str, torch.nn.Module] = {}
_GATE_PREFETCH = os.environ.get("L2PF_GATE", "1") != "0"
_FP8C_TILE_PREFETCH = os.environ.get("L2PF_FP8C_TILES", "1") != "0"

_lib = None
_side: dict[int, torch.cuda.Stream] = {}


def _load():
    global _lib
    try:
        lib = ctypes.CDLL(LIB_PATH)
    except OSError:  # library missing: every call is a no-op
        return None
    lib.l2pf_prefetch.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    lib.l2pf_prefetch.restype = ctypes.c_int
    try:
        lib.l2pf_prefetch3.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                     ctypes.c_void_p, ctypes.c_size_t,
                                     ctypes.c_void_p, ctypes.c_size_t,
                                     ctypes.c_int, ctypes.c_void_p]
        lib.l2pf_prefetch3.restype = ctypes.c_int
    except AttributeError:
        pass  # Older library: retain the original per-range launch ABI.
    tiled = getattr(lib, "l2pf_prefetch_fp8c_out", None)
    if tiled is not None:
        tiled.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
        ]
        tiled.restype = ctypes.c_int
    _lib = lib
    return lib


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    idx = device.index if device.index is not None else torch.cuda.current_device()
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


def gate_weight(layer_name: str, budget: int) -> tuple[torch.Tensor | None, int]:
    """Reserve at most a quarter of the existing window for next-router bytes.

    Prefetches the exact immutable BF16 shadow the existing router selects,
    or its original FP32 weight. Never changes a weight or an arithmetic path.
    The caller subtracts the reservation from its projection prefetch budget.
    """
    if not _GATE_PREFETCH or not ENABLED or _lib is None or budget <= 0:
        return None, 0
    gate = AHEAD_GATE.get(layer_name)
    if gate is None:
        return None, 0
    weight = gate.weight
    try:
        import smallk
        shadow = smallk.gate_w16(weight)
    except (ImportError, AttributeError):
        shadow = None
    if shadow is not None:
        weight = shadow
    reserved = min(weight.untyped_storage().nbytes(), budget // 4)
    if reserved <= 0:
        return None, 0
    return weight, reserved


def _fp8c_out_plan(t: torch.Tensor | None, budget: int):
    """Bounded immutable ranges for each verified fp8c out-projection tile.

    The marker is installed by fp8c.prepare only after its exhaustive native
    byte check. Cover every tile's header, at least six main chunks (all
    supported ring depths), and its entire exception tail, within the same
    total projection-byte budget as the old contiguous-head prefetch.
    """
    if (
        t is None or not isinstance(budget, int) or budget <= 0
        or not t.is_cuda or t.dtype != torch.uint8 or t.ndim != 2
        or t.shape[0] != 84 or not t.is_contiguous()
        or t.data_ptr() % 128
        or getattr(t, "_fp8c_prefetch_layout", None)
        != (1, 2688, 4096, 384, 3584, 32, t.data_ptr())
    ):
        return None
    tiles, stride = t.shape
    tail_offset = 384 + 32 * 3584
    # The producer admits <=93% of 32*4096 raw bytes per tile; both the
    # complete tile and its tail are 128-byte aligned after encoding.
    if stride % 128 or not tail_offset < stride <= 121856:
        return None
    storage = t.untyped_storage()
    if storage.data_ptr() != t.data_ptr() or storage.nbytes() != tiles * stride:
        return None
    tail_bytes = stride - tail_offset
    per_tile = min(budget, tiles * stride) // tiles // 128 * 128
    chunks = min(32, (per_tile - tail_bytes - 384) // 3584)
    if chunks < 6:
        return None
    prefix_bytes = 384 + chunks * 3584
    return tiles, stride, prefix_bytes, tail_offset, tail_bytes


def prefetch_fp8c_out(
    t: torch.Tensor | None, gate: torch.Tensor | None = None,
    gate_bytes: int = 0, max_bytes: int = 0,
) -> bool:
    """Distribute the existing out-projection budget across all 84 CTAs.

    Header/main prefix and the disjoint raw-exception tail are both needed
    by the decoder's initial ring. The router keeps its existing reservation,
    and one side-stream fork, one 128-thread launch and the caller's join are
    preserved. Unsupported metadata/old libraries keep the original policy.
    """
    fallback = ((gate, gate_bytes), (t, max_bytes))
    tiled = getattr(_lib, "l2pf_prefetch_fp8c_out", None) if _lib is not None else None
    if not ENABLED or not _FP8C_TILE_PREFETCH or tiled is None:
        return prefetch_many(fallback)
    plan = _fp8c_out_plan(t, max_bytes)
    if plan is None:
        return prefetch_many(fallback)
    gate_ptr, gate_size = 0, 0
    if gate is not None and gate_bytes > 0:
        if not gate.is_cuda or gate.device != t.device:
            return prefetch_many(fallback)
        storage = gate.untyped_storage()
        gate_ptr = storage.data_ptr()
        gate_size = min(storage.nbytes(), gate_bytes)
        if gate_ptr % 16 or gate_size % 16:
            return prefetch_many(fallback)
    tiles, stride, prefix_bytes, tail_offset, tail_bytes = plan
    main = torch.cuda.current_stream(t.device)
    side = _side_stream(t.device)
    side.wait_stream(main)
    rc = tiled(
        gate_ptr, gate_size, t.data_ptr(), stride, prefix_bytes,
        tail_offset, tail_bytes, tiles, CHUNK, side.cuda_stream,
    )
    if rc != 0:
        raise RuntimeError(f"l2pf_prefetch_fp8c_out failed: {rc}")
    return True


def prefetch_many(ranges) -> bool:
    """Up to three immutable ranges in one CUDA launch and side-stream fork."""
    if not ENABLED or _lib is None:
        return False
    selected = [(t, int(limit)) for t, limit in ranges
                if t is not None and t.is_cuda and limit > 0]
    if not selected:
        return False
    if len(selected) > 3 or any(t.device != selected[0][0].device for t, _ in selected):
        return False
    if len(selected) == 1:
        return prefetch(selected[0][0], max_bytes=selected[0][1])
    fused = getattr(_lib, "l2pf_prefetch3", None)
    if fused is None:
        queued = False
        for tensor, limit in selected:
            queued = prefetch(tensor, max_bytes=limit) or queued
        return queued
    args = []
    for tensor, limit in selected:
        storage = tensor.untyped_storage()
        args.extend((storage.data_ptr(), min(storage.nbytes(), limit)))
    args.extend([0, 0] * (3 - len(selected)))
    device = selected[0][0].device
    main = torch.cuda.current_stream(device)
    side = _side_stream(device)
    side.wait_stream(main)
    rc = fused(*args, CHUNK, side.cuda_stream)
    if rc != 0:
        raise RuntimeError(f"l2pf_prefetch3 failed: {rc}")
    return True


if ENABLED:
    _load()

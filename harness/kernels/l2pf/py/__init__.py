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
    "LIB_PATH", "ENABLED", "LATE_JOIN", "MAMBA_BYTES", "ATTN_BYTES", "MOE_BYTES", "DRAFT_BYTES",
    "AHEAD", "AHEAD_MOE", "DRAFT_SCREEN",
    "prefetch", "join", "join_after_out_proj", "join_capture",
]

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libl2pf.so")
ENABLED = os.environ.get("L2PF", "1") != "0"
LATE_JOIN = os.environ.get("L2PF_LATE_JOIN", "1") != "0"
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
    _lib = lib
    return lib


_capture_pending: set[int] = set()


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    idx = _device_index(device)
    s = _side.get(idx)
    if s is None:
        s = torch.cuda.Stream(device=idx)
        _side[idx] = s
    return s


def _mark_capture(device: torch.device) -> None:
    if torch.cuda.is_current_stream_capturing():
        _capture_pending.add(_device_index(device))


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
    _mark_capture(t.device)
    return True


def join(device: torch.device) -> None:
    """Make the current stream wait for the side stream's prefetches."""
    torch.cuda.current_stream(device).wait_stream(_side_stream(device))
    _capture_pending.discard(_device_index(device))


def join_after_out_proj(x: torch.Tensor, n: int, k: int) -> None:
    if (LATE_JOIN and ENABLED and _lib is not None and MAMBA_BYTES > 0
            and (n, k) == (2688, 4096) and 1 <= x.shape[0] <= 8
            and not torch.cuda.is_current_stream_capturing()):
        join(x.device)


def join_capture(device: torch.device) -> None:
    idx = _device_index(device)
    if LATE_JOIN and idx in _capture_pending:
        join(device)
    else:
        _capture_pending.discard(idx)


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
    _mark_capture(device)
    return True


if ENABLED:
    _load()

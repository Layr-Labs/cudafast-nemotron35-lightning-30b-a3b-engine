"""l2pf: L2 prefetch of a weight ahead of the bandwidth-bound GEMM that reads it.

``prefetch(t)`` queues bulk L2 prefetches of t's storage on a side stream forked
from the current stream (eager or under CUDA graph capture); ``join(device)``
makes the current stream wait for them, which a capture needs before it ends.
Only values already in memory move into L2: no kernel output can change.

``L2PF=0`` disables every prefetch. ``L2PF_DEFER_JOIN=0`` keeps the
original joins. Deferral is enabled only inside an explicit capture_scope;
that scope MUST end before the owning CUDA graph capture ends.
"""

from __future__ import annotations

import ctypes
import os
import threading
from contextlib import contextmanager

import torch

__all__ = [
    "LIB_PATH", "ENABLED", "MAMBA_BYTES", "ATTN_BYTES", "MOE_BYTES", "DRAFT_BYTES",
    "AHEAD", "AHEAD_MOE", "DRAFT_SCREEN",
    "prefetch", "join", "capture_scope", "DEFER_JOIN",
]

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libl2pf.so")
ENABLED = os.environ.get("L2PF", "1") != "0"
DEFER_JOIN = os.environ.get("L2PF_DEFER_JOIN", "1") != "0"
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

_lib = None
_side: dict[int, torch.cuda.Stream] = {}
_scope_state = threading.local()


class _CaptureScope:
    def __init__(self):
        self.main = torch.cuda.current_stream()
        self.pending = False

    def flush(self):
        if self.pending:
            # wait_stream records an event at the side's CURRENT tail. Flush
            # BEFORE the next fork, or the wait would cover the wrong hint.
            self.main.wait_stream(_side_stream(self.main.device))
            self.pending = False


def _scope():
    return getattr(_scope_state, "current", None)


def _owns(scope, main):
    return (scope is not None and scope.main.device == main.device
            and scope.main.cuda_stream == main.cuda_stream)


@contextmanager
def capture_scope():
    """Bound deferral to ONE owning CUDA capture, including piecewise captures.

    Only call inside torch.cuda.graph, and exit before its __exit__. No hint
    branch may escape. Foreign-stream calls and unscoped/eager calls retain
    their original immediate joins. No work is deferred between replays.
    """
    if not DEFER_JOIN or not torch.cuda.is_current_stream_capturing():
        yield
        return
    previous = _scope()
    if previous is not None:
        previous.flush()
    scope = _CaptureScope()
    _scope_state.current = scope
    try:
        yield
    finally:
        try:
            scope.flush()
        finally:
            _scope_state.current = previous


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
    scope = _scope()
    if scope is not None:
        scope.flush()  # last legal point before this same side stream is reused
    side = _side_stream(t.device)
    side.wait_stream(main)
    rc = _lib.l2pf_prefetch(st.data_ptr(), nbytes, CHUNK, side.cuda_stream)
    if rc != 0:
        raise RuntimeError(f"l2pf_prefetch failed: {rc}")
    if _owns(scope, main):
        scope.pending = True
    return True


def join(device: torch.device) -> None:
    """Rejoin now, or at the next fork / owning capture end in a defer scope."""
    main = torch.cuda.current_stream(device)
    scope = _scope()
    if not _owns(scope, main):
        main.wait_stream(_side_stream(device))


if ENABLED:
    _load()

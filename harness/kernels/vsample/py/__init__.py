"""vsample: the greedy spec-decode verify sampler of a one-request batch in one kernel.

libvsample.so (ctypes, static cudart); csrc/vsample.cu documents the op chain it reproduces:
the MTP depth gate's draft masking, logits.argmax(-1) (torch's NaN/tie rules), the greedy
rejection kernel, the verify-row reset and the accepted-token count. Integer work and an argmax
under a strict total order, so outputs are identical to the stock chain for every input.

greedy_verify(...) runs it; maybe_greedy_verify(...) is the guarded entry point the model runner
calls and returns None whenever the stock path must run. VSAMPLE=0 disables it.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

import torch

__all__ = ["LIB_PATH", "ENABLED", "available", "greedy_verify", "maybe_greedy_verify", "VROWS_ALL"]

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libvsample.so")
ENABLED = os.environ.get("VSAMPLE", "1") != "0"
PDL = int(os.environ.get("VSAMPLE_PDL", "1"))
VROWS_ALL = 1 << 30

_lib = None
_load_error: str | None = None
_max_rows = 0
_max_vocab = 0
_scratch: dict = {}  # (device index, stream) -> (pval, pidx, counter)
_REDUX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libvsample_redux.so")
_REDUX_ENABLED = os.environ.get("VSAMPLE_REDUX", "1") != "0"
_redux_choice: dict = {}  # (device, rows, vocab, row stride, pdl) -> library or None
_redux_lock = threading.Lock()
_redux_qualifying = False


def set_redux_qualification(enabled: bool) -> None:
    # Called by resident only in its startup warm-up window. Unseen runtime
    # shapes retain the original library instead of benchmarking in a request.
    global _redux_qualifying
    _redux_qualifying = bool(enabled)


def _load():
    global _lib, _load_error, _max_rows, _max_vocab
    if _lib is not None or _load_error is not None:
        return _lib
    try:
        lib = ctypes.CDLL(LIB_PATH)
        vp, ci, cl = ctypes.c_void_p, ctypes.c_int, ctypes.c_long
        lib.vs_version.restype = ci
        lib.vs_max_rows.restype = ci
        lib.vs_max_vocab.restype = ci
        lib.vs_greedy_verify.argtypes = [vp, cl, ci, ci, vp, vp, vp, vp, vp, vp, ci, vp, vp, vp, vp, ci, vp]
        lib.vs_greedy_verify.restype = ci
        _max_rows = lib.vs_max_rows()
        _max_vocab = lib.vs_max_vocab()
        _lib = lib
    except OSError as exc:  # library missing: callers keep the stock path
        _load_error = str(exc)
    return _lib


def available() -> bool:
    return ENABLED and _load() is not None


def _bufs(device: torch.device):
    # one scratch set (partials + done counter) per (device, stream): launches on one
    # stream are ordered, so a set is never used by two launches at once
    idx = device.index if device.index is not None else torch.cuda.current_device()
    key = (idx, torch.cuda.current_stream(device).cuda_stream)
    b = _scratch.get(key)
    if b is None:
        n = max(_max_rows, 1) * 64
        b = (torch.empty(n, dtype=torch.float32, device=device), torch.empty(n, dtype=torch.int32, device=device),
             torch.zeros(1, dtype=torch.int32, device=device))
        _scratch[key] = b
    return b


def _select_redux(logits, target, bonus, cu, draft, vrows, out_len):
    # Qualify once per exact call shape using private outputs and counters. This
    # runs only outside capture; original remains available on any failed gate.
    if (not _REDUX_ENABLED or not logits.is_cuda or logits.dtype != torch.bfloat16
            or logits.ndim != 2 or logits.stride(1) != 1
            or torch.cuda.is_current_stream_capturing()):
        return None
    device_index = logits.device.index if logits.device.index is not None else torch.cuda.current_device()
    key = (device_index, logits.shape[0], logits.shape[1], logits.stride(0),
           out_len, vrows is not None, PDL)
    if key in _redux_choice:
        return _redux_choice[key]
    if not _redux_qualifying:
        return None
    with _redux_lock:
        if key in _redux_choice:
            return _redux_choice[key]
        if not _redux_qualifying:
            return None
        chosen = None
        try:
            from .redux_qualify_draft import _bind, qualify
            metrics = qualify(LIB_PATH, _REDUX_PATH, logits, target, bonus, cu, draft, vrows, out_len, PDL)
            if metrics is not None:
                chosen = _bind(ctypes.CDLL(_REDUX_PATH))
                print(f"vsample redux: admitted shape={key} qualification={metrics}",
                      file=sys.stderr, flush=True)
        except (ImportError, OSError, AttributeError):
            chosen = None
        _redux_choice[key] = chosen
        return chosen


def greedy_verify(logits: torch.Tensor, target_logits_indices: torch.Tensor, bonus_logits_indices: torch.Tensor,
                  cu_num_draft_tokens: torch.Tensor, draft_token_ids: torch.Tensor, max_spec_len: int,
                  vrows: torch.Tensor | None, num_accepted: torch.Tensor | None) -> torch.Tensor:
    """output_token_ids [1, max_spec_len + 1] int32 of the stock greedy chain; writes num_accepted[0]
    (int32) and resets vrows[0] to VROWS_ALL when given."""
    lib = _load()
    assert lib is not None, _load_error
    alternate = _select_redux(logits, target_logits_indices, bonus_logits_indices,
                              cu_num_draft_tokens, draft_token_ids, vrows, max_spec_len + 1)
    if alternate is not None:
        lib = alternate
    rows, vocab = logits.shape
    out = torch.empty((1, max_spec_len + 1), dtype=torch.int32, device=logits.device)
    pval, pidx, counter = _bufs(logits.device)
    rc = lib.vs_greedy_verify(
        logits.data_ptr(), logits.stride(0), rows, vocab, target_logits_indices.data_ptr(),
        bonus_logits_indices.data_ptr(), cu_num_draft_tokens.data_ptr(), draft_token_ids.data_ptr(),
        None if vrows is None else vrows.data_ptr(), out.data_ptr(), out.shape[1],
        None if num_accepted is None else num_accepted.data_ptr(), pval.data_ptr(), pidx.data_ptr(),
        counter.data_ptr(), PDL, torch.cuda.current_stream(logits.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"vs_greedy_verify failed: {rc} (rows={rows} vocab={vocab})")
    return out


def _i32(t: torch.Tensor | None, n: int, device) -> bool:
    return (t is not None and t.dtype == torch.int32 and t.is_contiguous() and t.device == device
            and t.dim() == 1 and t.numel() >= n)


def maybe_greedy_verify(logits: torch.Tensor | None, metadata, vrows: torch.Tensor | None,
                        num_accepted: torch.Tensor | None) -> torch.Tensor | None:
    """The fused chain for a one-request greedy verify, or None (the caller runs the stock ops).
    The caller has already established that the rejection sampler would take its greedy-argmax path."""
    if not ENABLED or logits is None or metadata is None:
        return None
    if len(metadata.num_draft_tokens) != 1:
        return None
    d = int(metadata.num_draft_tokens[0])
    max_spec_len = int(metadata.max_spec_len)
    if _load() is None or d < 1 or max_spec_len != d:
        return None
    dev = logits.device
    if (
        not logits.is_cuda
        or logits.dtype != torch.bfloat16
        or logits.dim() != 2
        or logits.shape[0] != d + 1
        or logits.shape[0] > _max_rows
        or logits.shape[1] > _max_vocab
        or logits.shape[1] % 8 != 0
        or logits.stride(1) != 1
        or logits.stride(0) % 8 != 0
        or logits.data_ptr() % 16 != 0
        or not _i32(metadata.target_logits_indices, d, dev)
        or not _i32(metadata.bonus_logits_indices, 1, dev)
        or not _i32(metadata.cu_num_draft_tokens, 1, dev)
        or not _i32(metadata.draft_token_ids, d, dev)
        or (vrows is not None and not _i32(vrows, 1, dev))
        or (num_accepted is not None and not _i32(num_accepted, 1, dev))
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    return greedy_verify(logits, metadata.target_logits_indices, metadata.bonus_logits_indices,
                         metadata.cu_num_draft_tokens, metadata.draft_token_ids, max_spec_len, vrows, num_accepted)


_load() if ENABLED else None

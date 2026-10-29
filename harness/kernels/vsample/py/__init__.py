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
        if hasattr(lib, "vs_greedy_verify_partials"):
            lib.vs_greedy_verify_partials.argtypes = lib.vs_greedy_verify.argtypes
            lib.vs_greedy_verify_partials.restype = ci
        _max_rows = lib.vs_max_rows()
        _max_vocab = lib.vs_max_vocab()
        _lib = lib
    except OSError as exc:  # library missing: callers keep the stock path
        _load_error = str(exc)
    return _lib


def available() -> bool:
    return ENABLED and _load() is not None


def partials_available() -> bool:
    lib = _load()
    return ENABLED and lib is not None and hasattr(lib, "vs_greedy_verify_partials")


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


def greedy_verify(logits: torch.Tensor, target_logits_indices: torch.Tensor, bonus_logits_indices: torch.Tensor,
                  cu_num_draft_tokens: torch.Tensor, draft_token_ids: torch.Tensor, max_spec_len: int,
                  vrows: torch.Tensor | None, num_accepted: torch.Tensor | None,
                  partials: torch.Tensor | None = None) -> torch.Tensor:
    """output_token_ids [1, max_spec_len + 1] int32 of the stock greedy chain; writes num_accepted[0]
    (int32) and resets vrows[0] to VROWS_ALL when given."""
    lib = _load()
    assert lib is not None, _load_error
    rows, vocab = logits.shape
    out = torch.empty((1, max_spec_len + 1), dtype=torch.int32, device=logits.device)
    pval, pidx, counter = _bufs(logits.device)
    op = lib.vs_greedy_verify if partials is None else lib.vs_greedy_verify_partials
    source = logits if partials is None else partials
    rc = op(
        source.data_ptr(), source.stride(0), rows, source.shape[1], target_logits_indices.data_ptr(),
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
                        num_accepted: torch.Tensor | None,
                        partials: torch.Tensor | None = None) -> torch.Tensor | None:
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
    if partials is not None and (
        not hasattr(_lib, "vs_greedy_verify_partials") or logits.shape[1] != 131072
        or not 2 <= logits.shape[0] <= 6 or partials.device != dev
        or partials.dtype != torch.int64 or not partials.is_contiguous()
        or tuple(partials.shape) != (logits.shape[0], 2048)
        or partials.data_ptr() % 8
    ):
        partials = None  # Complete original logits were retained by the producer.
    return greedy_verify(logits, metadata.target_logits_indices, metadata.bonus_logits_indices,
                         metadata.cu_num_draft_tokens, metadata.draft_token_ids, max_spec_len,
                         vrows, num_accepted, partials)


_load() if ENABLED else None

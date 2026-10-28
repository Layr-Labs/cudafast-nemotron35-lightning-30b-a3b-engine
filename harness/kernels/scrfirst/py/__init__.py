# SPDX-License-Identifier: Apache-2.0
"""Direct full-hidden gather + existing scrtail STAGE1 dot/cascade.
The original scrtail and token-selection kernel remain unchanged. Remote only.
"""
import ctypes
from pathlib import Path

import torch

_lib = None
_failed = False


def available():
    global _lib, _failed
    if _lib is None and not _failed:
        try:
            lib = ctypes.CDLL(str(Path(__file__).with_name("libscrfirst.so")))
            lib.scrfirst_run.argtypes = ([ctypes.c_void_p] * 21 + [ctypes.c_float]
                                        + [ctypes.c_int] * 4 + [ctypes.c_void_p] * 5)
            lib.scrfirst_run.restype = ctypes.c_int
            _lib = lib
        except OSError:
            _failed = True
    return _lib is not None


def _ptr(t):
    return None if t is None else t.data_ptr()


def run(screen, h, gate, pdl, debug, raw_first=None):
    if not available() or screen.n != 131072:
        return False  # No GPU work.
    n = screen.n
    shapes = ((screen.scores, (n,), torch.float32),
              (screen._hp, (256,), torch.float32), (screen._hm, (256,), torch.float32),
              (screen._hr, (512,), torch.float32), (h, (2688,), torch.float32),
              (screen.pre_cols, (256,), torch.int64),
              (screen.mid_cols, (256,), torch.int64),
              (screen.rest_cols, (512,), torch.int64),
              (screen._cand, (2048,), torch.int64),
              (screen.cand_logits, (2048,), torch.float32),
              (screen._pre_cand, (65536,), torch.int64),
              (screen.mid_scores, (65536,), torch.float32),
              (screen._coarse_cand, (16384,), torch.int64),
              (screen.refine_scores, (16384,), torch.float32))
    for t, shape, dtype in shapes:
        if (not isinstance(t, torch.Tensor) or tuple(t.shape) != shape or t.dtype != dtype
                or not t.is_cuda or t.device != h.device or not t.is_contiguous()):
            return False
    for t, shape in ((screen.w_pre, (n, 128)), (screen.s_pre, (n, 16)),
                     (screen.w_mid, (n, 128)), (screen.s_mid, (n, 16)),
                     (screen.w_rest, (n, 256)), (screen.s_rest, (n, 32)),
                     (screen.w_full, (n, 1344)), (screen.s_full, (n, 168))):
        if (tuple(t.shape) != shape or t.element_size() != 1 or not t.is_cuda
                or t.device != h.device or not t.is_contiguous() or t.data_ptr() % 16):
            return False
    if (gate is None or gate.shape != (4,) or gate.dtype != torch.int32
            or gate.device != h.device or not gate.is_contiguous()):
        return False
    if raw_first is not None and (raw_first.shape != (n,) or raw_first.dtype != torch.float32
                                  or raw_first.device != h.device or not raw_first.is_contiguous()):
        return False
    # scores_out is ALWAYS the original first-score owner, also in production.
    # Debug mid scatter remains conditional on mid_scores, as in original scrtail.
    dbg = (screen._pre_cand, screen.mid_scores, screen.scores,
           screen._coarse_cand, screen.refine_scores) if debug else (None, None, screen.scores, None, None)
    rc = _lib.scrfirst_run(
        _ptr(screen.scores), _ptr(screen._hm), _ptr(screen._hr), _ptr(h),
        _ptr(screen.w_mid), _ptr(screen.s_mid), _ptr(screen.w_rest), _ptr(screen.s_rest),
        _ptr(screen.w_full), _ptr(screen.s_full), _ptr(screen._cand), _ptr(screen.cand_logits),
        *(_ptr(t) for t in dbg), _ptr(gate), _ptr(screen._hp),
        _ptr(screen.w_pre), _ptr(screen.s_pre), float(screen.gscale), 64, 0, int(bool(pdl)), 1,
        _ptr(screen.pre_cols), _ptr(screen.mid_cols), _ptr(screen.rest_cols), _ptr(raw_first),
        torch.cuda.current_stream(h.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"scrfirst_run failed after dispatch: {rc}")
    return True

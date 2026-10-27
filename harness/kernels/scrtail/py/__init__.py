# SPDX-License-Identifier: Apache-2.0
"""scrtail: the draft-head screen's cascade after its first streaming dot as one CUDA kernel
(csrc/scrtail.cu): per 2048-row chunk, the top 1024 of the first scores, the mid dot, the top
256, the rest dot, the top 32 and the full-K dot of the survivors, with the arithmetic and the
selection order of nemotron_h_mtp_screen's kernels. The screen admits it only after a load-time
bitwise comparison with its own cascade (DraftHeadScreen). MTP_SCREEN_TAIL=0 keeps the cascade.
"""

import ctypes
import os
from pathlib import Path

import torch

ENABLED = os.environ.get("MTP_SCREEN_TAIL", "1") != "0"
CHUNK, PRE, CAS, PER = 2048, 1024, 256, 32
_lib = None
_load_failed = False


def _load():
    global _lib, _load_failed
    if _lib is not None or _load_failed:
        return _lib
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name("libscrtail.so")))
        lib.scrtail_run.argtypes = ([ctypes.c_void_p] * 21 + [ctypes.c_float] + [ctypes.c_int] * 4
                                    + [ctypes.c_void_p])
        lib.scrtail_run.restype = ctypes.c_int
        _lib = lib
    except OSError:
        _load_failed = True
    return _lib


def available() -> bool:
    return ENABLED and _load() is not None


def _ptr(t):
    return None if t is None else t.data_ptr()


def run(scores, hm, hr, h, w_mid, s_mid, w_rest, s_rest, w_full, s_full, cand, cand_logits,
        gscale: float, gate=None, win: int = 0, pdl: bool = False, debug=None,
        first=None) -> bool:
    """cand [C*32] int64 and cand_logits [C*32] fp32 from the first scores [C*2048], as the
    screen's cascade writes them. first = (hp, w_pre, s_pre) computes the first dot here too
    (scores then only sizes the chunks). debug = (pre_cand, mid_scores, scores_out, coarse_cand,
    refine_scores) also receives the cascade's intermediate buffers. False (nothing launched)
    for inputs outside the kernel's contract."""
    n = scores.numel()
    if _load() is None or n % CHUNK or not torch.is_tensor(scores):
        return False
    nchunk = n // CHUNK
    shapes = ((hm, (256,)), (hr, (512,)), (h, (2688,)), (w_mid, (n, 128)), (s_mid, (n, 16)),
              (w_rest, (n, 256)), (s_rest, (n, 32)), (w_full, (n, 1344)), (s_full, (n, 168)),
              (cand, (nchunk * PER,)), (cand_logits, (nchunk * PER,)))
    for t, shp in shapes:
        if tuple(t.shape) != shp or not t.is_contiguous() or t.device != scores.device:
            return False
    if (scores.dtype != torch.float32 or hm.dtype != torch.float32 or hr.dtype != torch.float32
            or h.dtype != torch.float32 or cand.dtype != torch.int64
            or cand_logits.dtype != torch.float32):
        return False
    if any(t.element_size() != 1 for t in (w_mid, s_mid, w_rest, s_rest, w_full, s_full)):
        return False
    if any(int(t.data_ptr()) % 16 for t in (w_mid, w_rest, w_full)):
        return False
    if first is not None:
        hp, w_pre, s_pre = first
        if (tuple(hp.shape) != (256,) or hp.dtype != torch.float32 or not hp.is_contiguous()
                or tuple(w_pre.shape) != (n, 128) or tuple(s_pre.shape) != (n, 16)
                or w_pre.element_size() != 1 or s_pre.element_size() != 1
                or not w_pre.is_contiguous() or not s_pre.is_contiguous()
                or int(w_pre.data_ptr()) % 16):
            return False
    else:
        hp = w_pre = s_pre = None
    dbg = debug if debug is not None else (None,) * 5
    if gate is not None and (gate.dtype != torch.int32 or gate.device != scores.device):
        return False
    rc = _lib.scrtail_run(
        _ptr(scores), _ptr(hm), _ptr(hr), _ptr(h), _ptr(w_mid), _ptr(s_mid), _ptr(w_rest),
        _ptr(s_rest), _ptr(w_full), _ptr(s_full), _ptr(cand), _ptr(cand_logits),
        *(_ptr(t) for t in dbg), _ptr(gate), _ptr(hp), _ptr(w_pre), _ptr(s_pre), float(gscale),
        nchunk, int(win), int(bool(pdl)), int(first is not None),
        torch.cuda.current_stream(scores.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"scrtail_run failed: {rc}")
    return True

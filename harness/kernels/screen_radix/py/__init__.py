# SPDX-License-Identifier: Apache-2.0
"""Current-call native radix selection for the existing draft screen."""
import ctypes
from pathlib import Path
import torch

_lib = None
_failed = False
_ready = set()


def available(device):
    global _lib, _failed
    if _failed or torch.cuda.get_device_capability(device) != (12, 1):
        return False
    if _lib is None:
        try:
            lib = ctypes.CDLL(str(Path(__file__).with_name("libscreen_radix.so")))
            lib.screen_radix_init.argtypes = []
            lib.screen_radix_init.restype = ctypes.c_int
            lib.screen_radix_topk.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.c_int, ctypes.c_int,
                                            ctypes.c_int, ctypes.c_int,
                                            ctypes.c_void_p]
            lib.screen_radix_topk.restype = ctypes.c_int
            _lib = lib
        except (OSError, AttributeError):
            _failed = True
            return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _ready:
        if torch.cuda.is_current_stream_capturing():
            return False
        with torch.cuda.device(device):
            if _lib.screen_radix_init() != 0 or not _admit(device):
                _failed = True
                return False
        _ready.add(index)
    return True


def _admit(device):
    """Official remote loading only: check both geometries and difficult keys.

    This guard contains no model input, benchmark prompt, score or prior result.
    A mismatch disables the native selector; the original torch route remains.
    """
    for width, keep in ((2048, 256), (2048, 32), (256, 32)):
        i = torch.arange(width, dtype=torch.int64, device=device)
        values = (((i * 37 + 11) % width).float() - width // 2) / 16
        specials = values.clone()
        specials[:8] = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), 0.0, -0.0,
             float("nan"), 2.0, 2.0], device=device, dtype=torch.float32,
        )
        scores = torch.stack((values, specials))
        reference = scores.sort(dim=-1, descending=True, stable=True).indices[:, :keep]
        for offset in (0, width):
            out = torch.empty((2, keep), dtype=torch.int64, device=device)
            rc = _lib.screen_radix_topk(
                scores.data_ptr(), out.data_ptr(), 2, width, keep, offset,
                torch.cuda.current_stream(device).cuda_stream,
            )
            expected = reference + torch.arange(2, device=device)[:, None] * offset
            if rc or not torch.equal(out, expected):
                return False
    return True


def maybe_topk(scores, width, keep, offset=0):
    if (scores.dtype != torch.float32 or not scores.is_cuda
            or not scores.is_contiguous() or width not in (256, 2048)
            or not 1 <= keep <= min(width, 256)
            or not 0 <= offset <= 2048 or scores.numel() % width
            or torch.compiler.is_compiling() or not available(scores.device)):
        return None
    rows = scores.numel() // width
    if rows < 1:
        return None
    out = torch.empty((rows, keep), dtype=torch.int64, device=scores.device)
    rc = _lib.screen_radix_topk(
        scores.data_ptr(), out.data_ptr(), rows, width, keep, offset,
        torch.cuda.current_stream(scores.device).cuda_stream,
    )
    if rc:
        raise RuntimeError(f"screen_radix launch failed: {rc}")
    return out

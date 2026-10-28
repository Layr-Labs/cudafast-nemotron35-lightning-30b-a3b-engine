# SPDX-License-Identifier: Apache-2.0
"""Unvalidated optional SSD CB sharing. No model dispatch or admission yet."""
import ctypes
from pathlib import Path

import torch

_P = ctypes.c_void_p
_I64 = ctypes.c_int64


class ScanInputs(ctypes.Structure):
    _fields_ = ([(x, _P) for x in ("x", "c", "dt", "da", "states", "initial", "d", "out")]
                + [(x, _I64) for x in (
                    "xs0", "xs1", "xs2", "cs0", "cs1", "cs2", "ts0", "ts1", "ts2",
                    "as0", "as1", "as2", "ss0", "ss1", "ss2", "ss3",
                    "is0", "is1", "is2", "is3", "os0", "os1", "os2", "ds")]
                + [(x, ctypes.c_bool) for x in ("has_initial", "has_d", "d_vector")])


class LaunchInputs(ctypes.Structure):
    _fields_ = [("scan", ScanInputs), ("b", _P), ("cu", _P), ("seq", _P),
                ("bs0", _I64), ("bs1", _I64), ("bs2", _I64),
                ("total_tokens", ctypes.c_int), ("seq_count", ctypes.c_int),
                ("chunks", ctypes.c_int)]


_lib = None
_failed = False
_ready_devices = set()


def prepare(device):
    """Only called outside capture; false retains the original kernel."""
    global _lib, _failed
    if _failed or torch.cuda.is_current_stream_capturing():
        return False
    if torch.cuda.get_device_capability(device) != (12, 1):
        return False
    try:
        if _lib is None:
            lib = ctypes.CDLL(str(Path(__file__).with_name("libssdcb.so")))
            lib.ssdcb_init.argtypes = []
            lib.ssdcb_init.restype = ctypes.c_int
            lib.ssdcb_run.argtypes = [ctypes.POINTER(LaunchInputs), ctypes.c_int,
                                      ctypes.c_int, ctypes.c_int, _P]
            lib.ssdcb_run.restype = ctypes.c_int
            _lib = lib
        with torch.cuda.device(device):
            index = torch.cuda.current_device()
            if index not in _ready_devices:
                if _lib.ssdcb_init():
                    return False
                _ready_devices.add(index)
        return True
    except (OSError, RuntimeError):
        _failed = True
        return False


def supported(x, b, c, dt, da, cu, seq, states, initial, d, z, out, seq_count):
    """Conservative metadata and storage checks; never inspects request data."""
    if (not x.is_cuda or x.ndim != 3 or x.shape[1:] != (64, 64)
            or x.dtype != torch.bfloat16 or not 0 < x.shape[0] <= 2048
            or b.shape != (x.shape[0], 8, 128) or c.shape != b.shape
            or b.dtype != x.dtype or c.dtype != x.dtype or b.stride(-1) != 1
            or c.stride(-1) != 1 or x.stride(-1) != 1 or z is not None
            or dt.ndim != 3 or dt.shape[0] != 64 or dt.shape != da.shape
            or dt.shape[2] not in (128, 256) or not 0 < dt.shape[1] <= 16
            or dt.dtype != torch.float32 or da.dtype != torch.float32
            or states.shape != (dt.shape[1], 64, 64, 128)
            or states.dtype != torch.float32 or not states.is_contiguous()
            or not dt.is_contiguous() or not da.is_contiguous()
            or x.stride(1) != 64 or b.stride(1) != 128 or c.stride(1) != 128
            or x.stride(0) < 4096 or b.stride(0) < 1024 or c.stride(0) < 1024
            or out.shape != x.shape or out.dtype != x.dtype or not out.is_contiguous()
            # Each thread owns a distinct output element. Reject overlapping
            # views even when their innermost dimension has unit stride.
            or out.stride(1) < 64 or out.stride(0) < 64 * out.stride(1)
            or not 0 < seq_count <= dt.shape[1]):
        return False
    if initial is not None and (initial.shape != (seq_count, 64, 64, 128)
                                or initial.dtype != torch.float32
                                or not initial.is_contiguous()):
        return False
    if d is not None and (d.dtype != torch.float32 or d.shape not in ((64,), (64, 64))
                          or not d.is_contiguous()):
        return False
    if (cu.ndim != 1 or seq.ndim != 1 or cu.numel() != dt.shape[1] + 1
            or seq.numel() != dt.shape[1] or not cu.is_contiguous() or not seq.is_contiguous()
            or cu.dtype not in (torch.int32, torch.int64)
            or seq.dtype not in (torch.int32, torch.int64)):
        return False
    inputs = (x, b, c, dt, da, cu, seq, states, initial, d)
    if any(t is not None and t.device != x.device for t in inputs) or out.device != x.device:
        return False
    owner = out.untyped_storage().data_ptr()
    return all(t is None or t.untyped_storage().data_ptr() != owner for t in inputs)


def run(x, b, c, dt, da, cu, seq, states, initial, d, z, out, seq_count):
    """Launch only after remote qualification and a matching baseline complement."""
    if (_lib is None or x.device.index not in _ready_devices
            or not supported(x,b,c,dt,da,cu,seq,states,initial,d,z,out,seq_count)):
        return False
    scan = ScanInputs()
    for name, tensor in (("x",x),("c",c),("dt",dt),("da",da),("states",states),
                         ("initial",initial),("d",d),("out",out)):
        setattr(scan, name, tensor.data_ptr() if tensor is not None else None)
    for prefix, tensor, size in (("xs",x,3),("cs",c,3),("ts",dt,3),("as",da,3),
                                 ("ss",states,4),("is",initial,4),("os",out,3)):
        for i in range(size):
            setattr(scan, prefix + str(i), tensor.stride(i) if tensor is not None else 0)
    scan.ds = d.stride(0) if d is not None else 0
    scan.has_initial, scan.has_d, scan.d_vector = initial is not None, d is not None, d is not None and d.ndim == 2
    p = LaunchInputs(scan, b.data_ptr(), cu.data_ptr(), seq.data_ptr(), *b.stride(),
                     x.shape[0], seq_count, dt.shape[1])
    rc = _lib.ssdcb_run(ctypes.byref(p), dt.shape[2], int(cu.dtype == torch.int64),
                        int(seq.dtype == torch.int64), torch.cuda.current_stream(x.device).cuda_stream)
    if rc:
        raise RuntimeError(f"ssdcb launch failed: {rc}")
    return True

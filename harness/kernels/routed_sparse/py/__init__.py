# SPDX-License-Identifier: Apache-2.0
"""Exact sparse NVFP4 down projection; persistent data contains weights only."""

import ctypes
from pathlib import Path

import torch

_lib = None
_load_failed = False
_ready_devices = set()


def _load():
    global _lib, _load_failed
    if _lib is not None or _load_failed:
        return _lib
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name("librouted_sparse.so")))
        lib.routed_sparse_init.argtypes = []
        lib.routed_sparse_init.restype = ctypes.c_int
        lib.routed_sparse_run.argtypes = [ctypes.c_void_p] * 7 + [
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.routed_sparse_run.restype = ctypes.c_int
        _lib = lib
    except OSError:
        _load_failed = True
    return _lib


def prepare(weight, scales, global_scales):
    """Return byte-exact packed weights for the guarded ModelOpt checkpoint."""
    if (
        not weight.is_cuda
        or weight.dtype != torch.uint8
        or weight.shape != (128, 2688, 928)
        or scales.dtype != torch.float8_e4m3fn
        or scales.shape != (128, 2688, 116)
        or global_scales.dtype != torch.float32
        or global_scales.numel() != 128
        or scales.device != weight.device
        or global_scales.device != weight.device
        or not weight.is_contiguous()
        or not scales.is_contiguous()
        or torch.cuda.get_device_capability(weight.device) != (12, 1)
        or _load() is None
    ):
        return None
    with torch.cuda.device(weight.device):
        device = torch.cuda.current_device()
        if device not in _ready_devices:
            rc = _lib.routed_sparse_init()
            if rc:
                raise RuntimeError(f"routed_sparse initialization failed: {rc}")
            _ready_devices.add(device)
        # E, N16, row-half, row8, K16, K-half, K-pair -> sector order.
        packed = (
            weight.reshape(128, 168, 2, 8, 116, 2, 4)
            .permute(0, 1, 4, 6, 3, 5, 2)
            .contiguous()
        )
        packed_scales = (
            scales.view(torch.uint8)
            .reshape(128, 168, 2, 8, 116)
            .permute(0, 1, 4, 3, 2)
            .contiguous()
        )
        return packed, packed_scales, global_scales.flatten().clone()


def forward(inputs, ids, outputs, packed):
    """Launch on current inputs, or return False for the ordinary Humming path.

    IDs come from the same global top-k router as Humming: six distinct valid
    expert indices per token. The caller excludes expert-parallel remapping.
    Every mask consumed by the GEMM is overwritten by this invocation, on the
    same stream; no activation or mask is reused across calls or requests.
    """
    if packed is None:
        return False
    if (
        inputs.dtype != torch.bfloat16
        or outputs.dtype != torch.bfloat16
        or ids.dtype != torch.int32
        or ids.ndim != 2
        or not 1 <= ids.shape[0] <= 8
        or ids.shape[1] != 6
        or inputs.shape != (ids.numel(), 1856)
        or outputs.shape != (ids.numel(), 2688)
        or not inputs.is_contiguous()
        or not ids.is_contiguous()
        or not outputs.is_contiguous()
        or inputs.device != packed[0].device
        or ids.device != inputs.device
        or outputs.device != inputs.device
    ):
        return False
    masks = torch.empty((ids.numel(), 116), dtype=torch.uint8, device=inputs.device)
    weights, scales, global_scales = packed
    rc = _lib.routed_sparse_run(
        inputs.data_ptr(),
        weights.data_ptr(),
        scales.data_ptr(),
        global_scales.data_ptr(),
        ids.data_ptr(),
        masks.data_ptr(),
        outputs.data_ptr(),
        ids.shape[0],
        torch.cuda.current_stream(inputs.device).cuda_stream,
    )
    if rc:
        raise RuntimeError(f"routed_sparse launch failed: {rc}")
    return True

# SPDX-License-Identifier: Apache-2.0
"""Native FP8 small-M kernels; weight-A is a remotely probed experiment."""

import ctypes
import os
import weakref
from pathlib import Path

import torch
from vllm.triton_utils import tl, triton

SHAPES = {(10304, 2688), (2688, 4096)}
_lib = None
_load_failed = False
_ready_devices = set()
_weight_a_devices = set()
_weight_a_packs = {}
_weight_a_enabled = os.environ.get("FP8SMALL_WEIGHT_A", "1") != "0"


def _load():
    global _lib, _load_failed
    if _lib is not None or _load_failed:
        return _lib
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name("libfp8small.so")))
        lib.fp8small_init.argtypes = []
        lib.fp8small_init.restype = ctypes.c_int
        lib.fp8small_run.argtypes = (
            [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        )
        lib.fp8small_run.restype = ctypes.c_int
        try:
            lib.fp8small_run_weight_a.argtypes = lib.fp8small_run.argtypes
            lib.fp8small_run_weight_a.restype = ctypes.c_int
            lib.fp8small_weight_a_init.argtypes = []
            lib.fp8small_weight_a_init.restype = ctypes.c_int
        except AttributeError:
            pass  # old library keeps the original execution path
        _lib = lib
    except OSError:
        _load_failed = True
    return _lib


@triton.jit
def _repack(src, dst, N: tl.constexpr, K: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    tile = i // ((K // 32) * 256)
    step = (i // 256) % (K // 32)
    warp = (i // 64) % 4
    lane = (i // 2) % 32
    word = i % 2
    col = tile * 32 + warp * 8 + lane // 4
    k4 = step * 8 + lane % 4 + word * 4
    value = tl.load(src + col * (K // 4) + k4, mask=i < N * K // 4, other=0)
    tl.store(dst + i, value, mask=i < N * K // 4)


def prepare(weight: torch.Tensor) -> torch.Tensor | None:
    """Pack a supported column-major [K,N] FP8 weight without changing any byte."""
    if not weight.is_cuda or weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
        return None
    k, n = weight.shape
    if (n, k) not in SHAPES or weight.stride() != (1, k) or weight.data_ptr() % 16:
        return None
    if torch.cuda.get_device_capability(weight.device) != (12, 1) or _load() is None:
        return None
    with torch.cuda.device(weight.device):
        index = torch.cuda.current_device()
        if index not in _ready_devices:
            rc = _lib.fp8small_init()
            if rc:
                raise RuntimeError(f"fp8small initialization failed: {rc}")
            _ready_devices.add(index)
        packed = torch.empty((n, k), dtype=torch.uint8, device=weight.device)
        _repack[(triton.cdiv(n * k // 4, 256),)](
            weight.t().view(torch.int32), packed.view(torch.int32), n, k
        )
        if (_weight_a_enabled and hasattr(_lib, "fp8small_run_weight_a")
                and hasattr(_lib, "fp8small_weight_a_init")):
            if index not in _weight_a_devices:
                rc = _lib.fp8small_weight_a_init()
                if rc:
                    raise RuntimeError(f"fp8small weight-A initialization failed: {rc}")
                _weight_a_devices.add(index)
            enabled = _probe_weight_a(packed)
            if enabled:
                _weight_a_packs[(index, packed.data_ptr())] = weakref.ref(packed)
            print(f"fp8small weight-A N={n} K={k}: {'enabled' if enabled else 'probe mismatch, original fallback'}", flush=True)
    return packed


def _call(fn, a, packed, sa, sb, out):
    n, k = packed.shape
    return fn(
        a.data_ptr(), packed.data_ptr(), sa.data_ptr(), sb.data_ptr(),
        out.data_ptr(), a.shape[0], n, k,
        torch.cuda.current_stream(a.device).cuda_stream,
    )


@torch.no_grad()
def _probe_weight_a(packed):
    """Remote load-time eligibility samples, not a universal exactness proof.

    Use checkpoint weights and current-call-independent finite FP8 inputs;
    no prompt/request outputs are computed or kept here. No local execution.
    Compare BF16 bits, including zero signs, against original native MMA.
    """
    n, k = packed.shape
    scales = torch.ones(1, dtype=torch.float32, device=packed.device)
    cols = torch.arange(8 * k, dtype=torch.int32, device=packed.device)
    for multiplier, modulus, divisor in ((17, 127, 8.0), (13, 31, 64.0)):
        values = (((cols * multiplier + 11) % modulus - modulus // 2).float() / divisor)
        inputs = values.to(torch.float8_e4m3fn).view(8, k)
        for m in (1, 4, 5, 8):
            x = inputs[:m]
            ref = torch.empty((m, n), dtype=torch.bfloat16, device=packed.device)
            got = torch.empty_like(ref)
            for fn, out in ((_lib.fp8small_run, ref), (_lib.fp8small_run_weight_a, got)):
                rc = _call(fn, x, packed, scales, scales, out)
                if rc:
                    raise RuntimeError(f"fp8small weight-A eligibility launch failed: {rc}")
            if not torch.equal(ref.view(torch.int16), got.view(torch.int16)):
                return False
    return True


def gemm(
    a: torch.Tensor, packed: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor
) -> torch.Tensor:
    """Called only after the runtime guard and remote load-time preparation."""
    n, k = packed.shape
    m = a.shape[0]
    out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    key = (a.device.index, packed.data_ptr())
    checked = _weight_a_packs.get(key)
    fn = (_lib.fp8small_run_weight_a
          if _weight_a_enabled and checked is not None and checked() is not None
          else _lib.fp8small_run)
    rc = _call(fn, a, packed, sa, sb, out)
    if rc:
        raise RuntimeError(f"fp8small launch failed: {rc}, M={m}, N={n}, K={k}")
    return out

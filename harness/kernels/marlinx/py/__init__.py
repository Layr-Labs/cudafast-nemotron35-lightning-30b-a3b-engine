"""marlinx: Marlin-exact NVFP4 W4A16 GEMM for 1 <= M <= 8 bf16 rows.

libmarlinx.so computes exactly what vLLM v0.27.1's ``ops.marlin_gemm`` computes
for the NVFP4 W4A16 dense layers of Nemotron-3.5 (shared experts and lm_head)
when M <= 8, from the same Marlin-format tensors (no repack), with a schedule
that streams the weights faster. See README.md for the arithmetic it preserves.

``MARLINX=0`` in the environment disables it; every call outside the guard
(shape allowlist, 1 <= M <= 8, bf16, alignment, no bias, fp32 reduce, not
inside torch.compile tracing) returns None and the caller runs stock Marlin.
"""

from __future__ import annotations

import ctypes
import os

import torch

__all__ = ["LIB_PATH", "available", "prepare", "maybe_apply", "gemm", "hog", "regs", "SHAPES"]

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libmarlinx.so")

# (N, K) of the layers the kernel was verified on: shared expert up, down; lm_head.
SHAPES = frozenset({(3712, 2688), (2688, 3712), (131072, 2688)})
MAX_M = 8
# Register-ring depth in k16 blocks (csrc/marlinx.cu supports 4 and 8; same bits).
DEPTH = int(os.environ.get("MARLINX_DEPTH", "8"))
ENABLED = os.environ.get("MARLINX", "1") != "0"

_lib = None
_load_error: str | None = None
_ready = False
_sms: dict[int, int] = {}


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    try:
        lib = ctypes.CDLL(LIB_PATH)
        lib.marlinx_init.argtypes = []
        lib.marlinx_init.restype = ctypes.c_int
        lib.marlinx_regs.argtypes = [ctypes.c_int]
        lib.marlinx_regs.restype = ctypes.c_int
        lib.marlinx_gemm.argtypes = [
            ctypes.c_void_p, ctypes.c_longlong, ctypes.c_int,  # A lda M
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # B S GS
            ctypes.c_void_p, ctypes.c_longlong,  # C ldc
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # N K sms depth
            ctypes.c_void_p,  # stream
        ]
        lib.marlinx_gemm.restype = ctypes.c_int
        lib.marlinx_hog.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                    ctypes.c_longlong, ctypes.c_void_p, ctypes.c_void_p]
        lib.marlinx_hog.restype = ctypes.c_int
        _lib = lib
    except OSError as exc:  # library missing or unloadable: callers fall back
        _load_error = str(exc)
    return _lib


def available() -> bool:
    return _load() is not None


def _init() -> bool:
    """Loads the kernels into the context (must run outside CUDA graph capture)."""
    global _ready
    if _ready:
        return True
    lib = _load()
    if lib is None or torch.cuda.is_current_stream_capturing():
        return False
    rc = lib.marlinx_init()
    if rc != 0:
        raise RuntimeError(f"marlinx_init failed: {rc}")
    _ready = True
    return True


def regs(depth: int = DEPTH) -> int:
    lib = _load()
    assert lib is not None, _load_error
    return lib.marlinx_regs(depth)


def _sm_count(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    n = _sms.get(idx)
    if n is None:
        n = torch.cuda.get_device_properties(idx).multi_processor_count
        _sms[idx] = n
    return n


def prepare(layer) -> bool:
    """Marks a Marlin NVFP4 linear layer (after prepare_fp4_layer_for_marlin) as eligible."""
    layer._marlinx_ok = False
    if not ENABLED or _load() is None:
        return False
    n = layer.output_size_per_partition
    k = layer.input_size_per_partition
    w, s, gs = layer.weight, layer.weight_scale, getattr(layer, "weight_global_scale", None)
    ok = (
        (n, k) in SHAPES
        and gs is not None
        and gs.dtype == torch.float32
        and gs.numel() == 1
        and w.dtype == torch.int32
        and tuple(w.shape) == (k // 16, 2 * n)
        and w.is_contiguous()
        and s.dtype == torch.float8_e4m3fn
        and tuple(s.shape) == (k // 16, n)
        and s.is_contiguous()
        and w.is_cuda
        and getattr(layer, "bias", None) is None
    )
    if ok:
        ok = _init()
    layer._marlinx_ok = ok
    return ok


def gemm(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
         global_scale: torch.Tensor, n: int, k: int, depth: int = DEPTH) -> torch.Tensor:
    """out[M, n] for a 2-D bf16 x[M, k] (1 <= M <= 8); no guard beyond the C side's."""
    lib = _load()
    assert lib is not None, _load_error
    m = x.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    rc = lib.marlinx_gemm(
        x.data_ptr(), x.stride(0), m,
        weight.data_ptr(), weight_scale.data_ptr(), global_scale.data_ptr(),
        out.data_ptr(), n, n, k, _sm_count(x.device), depth,
        torch.cuda.current_stream(x.device).cuda_stream,
    )
    if rc != 0:
        raise RuntimeError(f"marlinx_gemm failed: {rc} (M={m} N={n} K={k} depth={depth})")
    return out


def maybe_apply(layer, x: torch.Tensor) -> torch.Tensor | None:
    """Marlin-exact output for an eligible layer and input, or None (caller runs Marlin)."""
    if not ENABLED or not getattr(layer, "_marlinx_ok", False):
        return None
    if torch.compiler.is_compiling():
        return None
    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    if (
        m < 1
        or m > MAX_M
        or x2.dtype != torch.bfloat16
        or x2.stride(-1) != 1
        or x2.stride(0) % 8
        or x2.data_ptr() % 16
    ):
        return None
    if not _ready and not _init():
        return None
    n = layer.output_size_per_partition
    out = gemm(x2, layer.weight, layer.weight_scale, layer.weight_global_scale, n, k)
    return out.reshape(*x.shape[:-1], n)


def hog(src: torch.Tensor, blocks: int, threads: int, smem: int, sink: torch.Tensor) -> None:
    """Streams src with a simple read kernel (bandwidth probe / concurrent load)."""
    lib = _load()
    assert lib is not None, _load_error
    rc = lib.marlinx_hog(blocks, threads, smem, src.data_ptr(), src.numel() * src.element_size(),
                         sink.data_ptr(), torch.cuda.current_stream(src.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"marlinx_hog failed: {rc}")

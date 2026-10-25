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

__all__ = ["LIB_PATH", "available", "prepare", "maybe_apply", "gemm", "hog", "regs", "SHAPES", "maybe_apply_relu2"]

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libmarlinx.so")

# (N, K) of the layers the kernel was verified on: shared expert up, down; lm_head.
SHAPES = frozenset({(3712, 2688), (2688, 3712), (131072, 2688)})
MAX_M = 8
# Register-ring depth in k16 blocks (csrc/marlinx.cu supports 4 and 8; same bits).
DEPTH = int(os.environ.get("MARLINX_DEPTH", "8"))
ENABLED = os.environ.get("MARLINX", "1") != "0"
_RDS_RING4 = os.environ.get("RDS_SHARED_RING4", "1") != "0"

_lib = None
_load_error: str | None = None
_ready = False
_sms: dict[int, int] = {}
# Optional ABI: old/no-PDL libraries keep the ordinary launch. A rejected
# eager PDL launch disables all PDL entry points; capture errors never retry.
_PDL_ENABLED = os.environ.get("MARLINX_PDL", "1") != "0"
_pdl_active = False
_init_pdl = None
_ops = {}
_ops_pdl = {}


def _load():
    global _lib, _load_error, _init_pdl, _ops, _ops_pdl
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
        if hasattr(lib, "marlinx_gemm_zs"):
            lib.marlinx_gemm_zs.argtypes = lib.marlinx_gemm.argtypes + [ctypes.c_void_p]
            lib.marlinx_gemm_zs.restype = ctypes.c_int
        if hasattr(lib, "marlinx_gemm_relu2"):
            lib.marlinx_gemm_relu2.argtypes = lib.marlinx_gemm.argtypes
            lib.marlinx_gemm_relu2.restype = ctypes.c_int
        lib.marlinx_hog.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                    ctypes.c_longlong, ctypes.c_void_p, ctypes.c_void_p]
        lib.marlinx_hog.restype = ctypes.c_int
        _ops = {"gemm": lib.marlinx_gemm}
        for mode in ("relu2", "zs"):
            op = getattr(lib, "marlinx_gemm_" + mode, None)
            if op is not None:
                _ops[mode] = op
        if _PDL_ENABLED:
            try:
                init_pdl = lib.marlinx_init_pdl
                ops_pdl = {
                    "gemm": lib.marlinx_gemm_pdl,
                    "relu2": lib.marlinx_gemm_relu2_pdl,
                    "zs": lib.marlinx_gemm_zs_pdl,
                }
            except AttributeError:  # old library or MARLINX_NO_PDL build
                init_pdl = None
            if init_pdl is not None:
                init_pdl.argtypes = []
                init_pdl.restype = ctypes.c_int
                for mode, op in ops_pdl.items():
                    op.argtypes = _ops[mode].argtypes
                    op.restype = ctypes.c_int
                _init_pdl, _ops_pdl = init_pdl, ops_pdl
        _lib = lib
    except OSError as exc:  # library missing or unloadable: callers fall back
        _load_error = str(exc)
    return _lib


def available() -> bool:
    return _load() is not None


def _init() -> bool:
    """Loads the kernels into the context (must run outside CUDA graph capture)."""
    global _ready, _pdl_active
    if _ready:
        return True
    lib = _load()
    if lib is None or torch.cuda.is_current_stream_capturing():
        return False
    rc = lib.marlinx_init()
    if rc != 0:
        raise RuntimeError(f"marlinx_init failed: {rc}")
    if _init_pdl is not None and _init_pdl() == 0:
        _pdl_active = True
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
    layer._marlinx_zs = None
    if ok and ZS_ENABLED and (n, k) == ZS_SHAPE and hasattr(_lib, "marlinx_gemm_zs"):
        layer._marlinx_zs = zs_repack(w, n, k)
    return ok


# Zero-row skip for the shared expert down projection (its input is ReLU^2,
# mostly exact zeros): a k-row-major copy of the Marlin weight lets the kernel
# read only the k rows whose activation is nonzero in some row (see csrc).
ZS_ENABLED = os.environ.get("MARLINX_ZS", "1") != "0"
ZS_SHAPE = (2688, 3712)
# bit offsets inside Marlin's word for k = 2t, 2t+1, 2t+8, 2t+9: (feature g, feature g+8)
_ZS_POS = ((0, 8), (16, 24), (4, 12), (20, 28))


@torch.no_grad()
def zs_repack(w: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """[K/16][16][N/64][8] u32 from Marlin's [K/16][2N] int32 weight; byte j of
    word (kb, kr, half column, g) = feature 16j+g | feature 16j+g+8 << 4 at k row kr."""
    q = w.view(k // 16, n // 128, 2, 8, 4, 4).to(torch.int64) & 0xFFFFFFFF  # kb, col, h, g, t, j
    out = torch.zeros(k // 16, 16, n // 64, 8, dtype=torch.int64, device=w.device)
    shifts = (torch.arange(4, device=w.device) * 8).view(1, 1, 1, 1, 4)
    for s, (pg, pg8) in enumerate(_ZS_POS):
        byte = ((q >> pg) & 15) | (((q >> pg8) & 15) << 4)  # kb, col, h, g, t, j
        word = (byte << shifts).sum(-1)  # kb, col, h, g, t
        for t in range(4):
            kr = (2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9)[s]
            out[:, kr] = word[..., t].reshape(k // 16, n // 64, 8)
    out = out - (out >= (1 << 31)).to(torch.int64) * (1 << 32)
    return out.to(torch.int32).contiguous()


def gemm(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
         global_scale: torch.Tensor, n: int, k: int, depth: int = DEPTH,
         relu2: bool = False, zs: torch.Tensor | None = None) -> torch.Tensor:
    """out[M, n] for a 2-D bf16 x[M, k] (1 <= M <= 8); no guard beyond the C side's."""
    global _pdl_active
    lib = _load()
    assert lib is not None, _load_error
    m = x.shape[0]
    if _RDS_RING4 and zs is not None and (m, n, k, depth) == (5, 2688, 3712, 8):
        depth = 4
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    mode = "zs" if zs is not None else "relu2" if relu2 else "gemm"
    use_pdl = _pdl_active
    op = _ops_pdl[mode] if use_pdl else _ops[mode]
    args = (
        x.data_ptr(), x.stride(0), m,
        weight.data_ptr(), weight_scale.data_ptr(), global_scale.data_ptr(),
        out.data_ptr(), n, n, k, _sm_count(x.device), depth,
        torch.cuda.current_stream(x.device).cuda_stream,
    )
    if zs is not None:
        args += (zs.data_ptr(),)
    rc = op(*args)
    if rc and use_pdl and not torch.cuda.is_current_stream_capturing():
        # Match fp8small: only rejected eager launches retry, never capture.
        _pdl_active = False
        rc = _ops[mode](*args)
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
    out = gemm(x2, layer.weight, layer.weight_scale, layer.weight_global_scale, n, k,
               zs=getattr(layer, "_marlinx_zs", None))
    return out.reshape(*x.shape[:-1], n)


def hog(src: torch.Tensor, blocks: int, threads: int, smem: int, sink: torch.Tensor) -> None:
    """Streams src with a simple read kernel (bandwidth probe / concurrent load)."""
    lib = _load()
    assert lib is not None, _load_error
    rc = lib.marlinx_hog(blocks, threads, smem, src.data_ptr(), src.numel() * src.element_size(),
                         sink.data_ptr(), torch.cuda.current_stream(src.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"marlinx_hog failed: {rc}")


def maybe_apply_relu2(x, weight, weight_scale, global_scale, n, k):
    """Opaque-op runtime path for a prepared single-device shared up projection."""
    if not ENABLED or not _ready or _load() is None:
        return None
    if not hasattr(_lib, "marlinx_gemm_relu2") or (n, k) != (3712, 2688):
        return None
    x2 = x.reshape(-1, k)
    if (
        not 1 <= x2.shape[0] <= MAX_M
        or x2.dtype != torch.bfloat16
        or not x2.is_cuda
        or x.shape[-1] != k
        or x2.stride(-1) != 1
        or x2.stride(0) % 8
        or x2.data_ptr() % 16
        or weight.dtype != torch.int32
        or weight.shape != (k // 16, 2 * n)
        or not weight.is_contiguous()
        or weight_scale.dtype != torch.float8_e4m3fn
        or weight_scale.shape != (k // 16, n)
        or not weight_scale.is_contiguous()
        or global_scale.dtype != torch.float32
        or global_scale.numel() != 1
        or not global_scale.is_contiguous()
        or weight.device != x.device
        or weight_scale.device != x.device
        or global_scale.device != x.device
    ):
        return None
    out = gemm(x2, weight, weight_scale, global_scale, n, k, relu2=True)
    return out.reshape(*x.shape[:-1], n)

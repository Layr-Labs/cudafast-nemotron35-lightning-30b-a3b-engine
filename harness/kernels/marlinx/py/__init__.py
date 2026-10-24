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
import logging
import os

import torch

try:
    from . import kskip_layout
except Exception:  # noqa: BLE001 - loaded outside its package or file missing: no KSkip
    kskip_layout = None

_log = logging.getLogger("marlinx")

__all__ = ["LIB_PATH", "available", "prepare", "maybe_apply", "gemm", "hog", "regs", "SHAPES", "maybe_apply_relu2"]

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libmarlinx.so")

# (N, K) of the layers the kernel was verified on: shared expert up, down; lm_head.
SHAPES = frozenset({(3712, 2688), (2688, 3712), (131072, 2688)})
MAX_M = 8
# Register-ring depth in k16 blocks (csrc/marlinx.cu supports 4 and 8; same bits).
DEPTH = int(os.environ.get("MARLINX_DEPTH", "8"))
ENABLED = os.environ.get("MARLINX", "1") != "0"
# KSkip: the shared expert down projection (N 2688, K 3712) does not load the weight k rows whose
# activations are +-0 in every row of the call, from a load-time k-row repack of the weights
# (kskip_layout.py, every nibble kept). Same output bits, checked per layer at load time against
# the crown kernel; any failure keeps the crown kernel. MARLINX_KSKIP=0 disables it.
KSKIP_ENABLED = os.environ.get("MARLINX_KSKIP", "1") != "0"
KSKIP_SHAPE = (2688, 3712)

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
        if hasattr(lib, "marlinx_gemm_relu2"):
            lib.marlinx_gemm_relu2.argtypes = lib.marlinx_gemm.argtypes
            lib.marlinx_gemm_relu2.restype = ctypes.c_int
        if hasattr(lib, "marlinx_gemm_kskip") and hasattr(lib, "marlinx_kskip_init"):
            lib.marlinx_kskip_init.argtypes = []
            lib.marlinx_kskip_init.restype = ctypes.c_int
            lib.marlinx_gemm_kskip.argtypes = [
                ctypes.c_void_p, ctypes.c_longlong, ctypes.c_int,  # A lda M
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # B BK S GS
                ctypes.c_void_p, ctypes.c_longlong,  # C ldc
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # N K sms depth
                ctypes.c_void_p,  # stream
            ]
            lib.marlinx_gemm_kskip.restype = ctypes.c_int
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
    if ok and KSKIP_ENABLED and (n, k) == KSKIP_SHAPE:
        _kskip_prepare(layer)
    return ok


def gemm(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
         global_scale: torch.Tensor, n: int, k: int, depth: int = DEPTH,
         relu2: bool = False) -> torch.Tensor:
    """out[M, n] for a 2-D bf16 x[M, k] (1 <= M <= 8); no guard beyond the C side's."""
    lib = _load()
    assert lib is not None, _load_error
    m = x.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    op = lib.marlinx_gemm_relu2 if relu2 else lib.marlinx_gemm
    rc = op(
        x.data_ptr(), x.stride(0), m,
        weight.data_ptr(), weight_scale.data_ptr(), global_scale.data_ptr(),
        out.data_ptr(), n, n, k, _sm_count(x.device), depth,
        torch.cuda.current_stream(x.device).cuda_stream,
    )
    if rc != 0:
        raise RuntimeError(f"marlinx_gemm failed: {rc} (M={m} N={n} K={k} depth={depth})")
    return out


def gemm_kskip(x: torch.Tensor, weight: torch.Tensor, bk: torch.Tensor, weight_scale: torch.Tensor,
               global_scale: torch.Tensor, n: int, k: int, depth: int = DEPTH) -> torch.Tensor:
    """gemm() through the KSkip kernel (N 2688, K 3712); bk = _kskip_repack(weight)."""
    lib = _load()
    assert lib is not None, _load_error
    m = x.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    rc = lib.marlinx_gemm_kskip(
        x.data_ptr(), x.stride(0), m,
        weight.data_ptr(), bk.data_ptr(), weight_scale.data_ptr(), global_scale.data_ptr(),
        out.data_ptr(), n, n, k, _sm_count(x.device), depth,
        torch.cuda.current_stream(x.device).cuda_stream,
    )
    if rc != 0:
        raise RuntimeError(f"marlinx_gemm_kskip failed: {rc} (M={m} N={n} K={k} depth={depth})")
    return out


_kskip_ready = False
_kskip_broken = False


@torch.no_grad()
def _kskip_repack(w: torch.Tensor) -> torch.Tensor:
    """Crown int32 [K/16, 2N] -> BK (same shape): the gather of kskip_layout.src_of_dst() per
    (k16 block, 64-feature unit). Input-independent; moves every nibble, changes none."""
    kbs, n2 = w.shape
    units = n2 // kskip_layout.UNIT_WORDS
    src = torch.tensor(kskip_layout.src_of_dst(), dtype=torch.long, device=w.device)
    shifts = torch.arange(0, 32, 4, dtype=torch.int64, device=w.device)
    out = torch.empty_like(w)
    for k0 in range(0, kbs, 32):
        k1 = min(kbs, k0 + 32)
        x = w[k0:k1].reshape(k1 - k0, units, 128).to(torch.int64) & 0xFFFFFFFF
        nib = ((x.unsqueeze(-1) >> shifts) & 0xF).reshape(k1 - k0, units, 1024)
        dst = nib.index_select(2, src).reshape(k1 - k0, units, 128, 8)
        v = (dst << shifts).sum(-1)
        v = torch.where(v >= (1 << 31), v - (1 << 32), v)
        out[k0:k1] = v.to(torch.int32).reshape(k1 - k0, n2)
        del x, nib, dst, v
    return out


@torch.no_grad()
def _kskip_unpack(bk: torch.Tensor) -> torch.Tensor:
    """BK -> crown int32 through the kernel's rebuild expression (csrc kskip_word)."""
    kbs, n2 = bk.shape
    units = n2 // kskip_layout.UNIT_WORDS
    y = bk.reshape(kbs, units, 16, 8).to(torch.int64) & 0xFFFFFFFF  # [kb, unit, k row, g]
    lanes = []
    for t in range(4):
        r0, r1, r8, r9 = y[:, :, 2 * t], y[:, :, 2 * t + 1], y[:, :, 2 * t + 8], y[:, :, 2 * t + 9]
        words = []
        for j in range(4):
            s = 8 * j
            words.append(
                ((r0 >> s) & 0xF) | (((r8 >> s) & 0xF) << 4) | (((r0 >> (s + 4)) & 0xF) << 8)
                | (((r8 >> (s + 4)) & 0xF) << 12) | (((r1 >> s) & 0xF) << 16) | (((r9 >> s) & 0xF) << 20)
                | (((r1 >> (s + 4)) & 0xF) << 24) | (((r9 >> (s + 4)) & 0xF) << 28)
            )
        lanes.append(torch.stack(words, -1))  # [kb, unit, g, j]
    w = torch.stack(lanes, 3)  # [kb, unit, g, t, j]: crown u32 128*unit + 4*(4g + t) + j
    w = torch.where(w >= (1 << 31), w - (1 << 32), w)
    return w.to(torch.int32).reshape(kbs, n2)


def _kskip_test_input(m: int, k: int, seed: int, all_zero: bool) -> torch.Tensor:
    """Deterministic bf16 [m, k] (CPU): relu^2-like values; 75% of the k rows +-0 in every row
    (half -0), and in the other rows some +-0, subnormals, large and negative values."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    x = torch.randn(m, k, generator=gen).relu().square().to(torch.bfloat16).view(torch.int16).clone()
    neg_bit = torch.tensor(-32768, dtype=torch.int16)
    zero = torch.tensor(0, dtype=torch.int16)
    dead = torch.rand(k, generator=gen) < 0.75
    if all_zero:
        dead = torch.ones(k, dtype=torch.bool)
    sign = torch.rand(m, k, generator=gen) < 0.5
    zeros = torch.where(sign, neg_bit, zero)
    r = torch.rand(m, k, generator=gen)
    sub = torch.randint(1, 0x80, (m, k), generator=gen, dtype=torch.int16)
    big = torch.randint(0x4700, 0x4780, (m, k), generator=gen, dtype=torch.int16)
    live = ~dead.unsqueeze(0)
    x = torch.where(live & (r < 0.05), zeros, x)
    x = torch.where(live & (r >= 0.05) & (r < 0.08), sub, x)
    x = torch.where(live & (r >= 0.08) & (r < 0.10), big, x)
    x = torch.where(live & (r >= 0.10) & (r < 0.15), x | neg_bit, x)
    x = torch.where(dead.unsqueeze(0), zeros, x)
    return x.view(torch.bfloat16).contiguous()


@torch.no_grad()
def _kskip_device_check(layer, bk: torch.Tensor) -> bool:
    """KSkip vs the crown kernel on this layer's weights, bitwise, M = 1, 2, 4, 5, 8 x 3 inputs
    (5 = the depth-4 verify)."""
    n, k = KSKIP_SHAPE
    dev = layer.weight.device
    for m in (1, 2, 4, 5, 8):
        for i in range(3):
            x = _kskip_test_input(m, k, 7919 * m + i, all_zero=(m == 4 and i == 2)).to(dev)
            ref = gemm(x, layer.weight, layer.weight_scale, layer.weight_global_scale, n, k)
            out = gemm_kskip(x, layer.weight, bk, layer.weight_scale, layer.weight_global_scale, n, k)
            if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                return False
    return True


def _kskip_prepare(layer) -> bool:
    """Builds layer._marlinx_bk and sets layer._marlinx_kskip when every load-time check passes
    (outside CUDA graph capture); otherwise the layer keeps the crown kernel."""
    global _kskip_ready, _kskip_broken
    layer._marlinx_kskip = False
    if _kskip_broken or kskip_layout is None or not hasattr(_lib, "marlinx_gemm_kskip"):
        return False
    try:
        if torch.cuda.is_current_stream_capturing():
            return False
        if not _kskip_ready:
            rc = _lib.marlinx_kskip_init()
            if rc != 0:
                raise RuntimeError(f"marlinx_kskip_init failed: {rc}")
            _kskip_ready = True
        bk = _kskip_repack(layer.weight)
        if not torch.equal(_kskip_unpack(bk), layer.weight):
            raise RuntimeError("k-row repack does not round-trip")
        if not _kskip_device_check(layer, bk):
            raise RuntimeError("KSkip output differs from the crown kernel")
        layer._marlinx_bk = bk
        layer._marlinx_kskip = True
        return True
    except Exception as exc:  # noqa: BLE001 - any failure keeps the crown kernel
        _kskip_broken = True
        layer._marlinx_kskip = False
        if hasattr(layer, "_marlinx_bk"):
            del layer._marlinx_bk
        _log.warning("marlinx: KSkip disabled (%r); the crown kernel runs", exc)
        return False


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
    if getattr(layer, "_marlinx_kskip", False):
        out = gemm_kskip(x2, layer.weight, layer._marlinx_bk, layer.weight_scale,
                         layer.weight_global_scale, n, k)
        return out.reshape(*x.shape[:-1], n)
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

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
_lean_active = False  # MARLINX_LEAN kernels loaded (see LEAN_LAYERS)


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
        if hasattr(lib, "marlinx_gemm_su"):
            lib.marlinx_gemm_su.argtypes = lib.marlinx_gemm.argtypes + [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]  # Z SU relu2 pdl
            lib.marlinx_gemm_su.restype = ctypes.c_int
        if hasattr(lib, "marlinx_gemm_su6"):
            lib.marlinx_gemm_su6.argtypes = lib.marlinx_gemm.argtypes + [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p]
            lib.marlinx_gemm_su6.restype = ctypes.c_int
        if hasattr(lib, "marlinx_gemm_lean"):
            lib.marlinx_init_lean.argtypes = []
            lib.marlinx_init_lean.restype = ctypes.c_int
            lib.marlinx_gemm_lean.argtypes = lib.marlinx_gemm.argtypes[:11] + [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]  # stream Z2 relu2 pdl
            lib.marlinx_gemm_lean.restype = ctypes.c_int
        if hasattr(lib, "marlinx_gemm_lean_v"):
            lib.marlinx_gemm_lean_v.argtypes = lib.marlinx_gemm.argtypes[:11] + [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p]  # stream Z2 relu2 pdl vrows
            lib.marlinx_gemm_lean_v.restype = ctypes.c_int
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
    global _lean_active
    _lean_active = bool(LEAN_LAYERS) and hasattr(lib, "marlinx_gemm_lean") and lib.marlinx_init_lean() == 0
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
    layer._marlinx_su = None
    layer._marlinx_sut = None
    layer._marlinx_lean = False
    su_on = SU_SHAPES.get((n, k)) in SU_LAYERS
    if ok and _lean_active and not su_on and LEAN_SHAPES.get((n, k)) in LEAN_LAYERS:
        layer._marlinx_lean = True
    if ok and ZS_ENABLED and (n, k) == ZS_SHAPE and hasattr(_lib, "marlinx_gemm_zs"):
        # the lean down reads the Z2 nibble order (same bytes, same size)
        layer._marlinx_zs = z2_repack(w, n, k) if layer._marlinx_lean else zs_repack(w, n, k)
    if ok and SU_SHAPES.get((n, k)) in SU_LAYERS and hasattr(_lib, "marlinx_gemm_su"):
        layer._marlinx_su = su_repack(s, n, k)
        if SU6_ENABLED and SU_SHAPES.get((n, k)) in SU6_LAYERS and hasattr(_lib, "marlinx_gemm_su6"):
            packed = su6_repack(layer._marlinx_su)
            if packed is not None:
                layer._marlinx_su, layer._marlinx_sut = packed
        # the shared up runs through an opaque op that receives the tensors, not the layer
        if layer._marlinx_sut is None:
            _su_by_scale[s.data_ptr()] = (s, layer._marlinx_su)
    return ok


# Unit-major scale copy (MARLINX_SU, comma list of "lm" = lm_head, "down" = shared
# expert down; "0" or "" = none). Marlin's scale tensor is [K/16][N] bytes, so a
# 64-feature half column reads 64 bytes per k16 block at an N-byte stride: for
# lm_head 22 MB of 64-byte requests that run far below the weight stream's rate.
# su_repack stores the same bytes as [N/64][K/16][64]; the kernel stages its half
# column's scales in shared memory with coalesced reads. Values and arithmetic
# are unchanged (a byte permutation of a weight tensor).
SU_SHAPES = {(131072, 2688): "lm", (2688, 3712): "down", (3712, 2688): "up"}
_su_by_scale: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
SU_LAYERS = frozenset(v for v in os.environ.get("MARLINX_SU", "lm").split(",") if v and v != "0")


@torch.no_grad()
def su_repack(s: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """[N/64][K/16][64] bytes from Marlin's [K/16][N] scale bytes (byte permutation)."""
    return s.view(torch.uint8).view(k // 16, n // 64, 64).transpose(0, 1).contiguous()


_SU6_ENV = os.environ.get("MARLINX_SU6", "lm")
SU6_ENABLED = _SU6_ENV not in ("", "0")
SU6_LAYERS = frozenset(v for v in _SU6_ENV.split(",") if v)


@torch.no_grad()
def su6_repack(su: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(codes [N/64][K/16][48], table [64]) from su_repack's bytes, or None when the tensor
    holds more than 64 distinct bytes or the decode check fails. Row layout: bytes 0..31 hold
    code 2i | code 2i+1 << 4 (low nibbles), bytes 32..47 the high 2-bit pairs (code 4i+t at
    bits 2t of byte 32+i)."""
    b = su.view(torch.uint8)
    vals = torch.unique(b)
    if vals.numel() > 64:
        return None
    table = torch.zeros(64, dtype=torch.uint8, device=b.device)
    table[: vals.numel()] = vals
    table[vals.numel():] = vals[-1]
    inv = torch.zeros(256, dtype=torch.int32, device=b.device)
    inv[vals.long()] = torch.arange(vals.numel(), dtype=torch.int32, device=b.device)
    c = inv[b.long()]
    lo = (c[..., 0::2] & 15) | ((c[..., 1::2] & 15) << 4)
    h = (c >> 4) & 3
    hi = h[..., 0::4] | (h[..., 1::4] << 2) | (h[..., 2::4] << 4) | (h[..., 3::4] << 6)
    codes = torch.cat((lo, hi), -1).to(torch.uint8).contiguous()
    lo_b = codes[..., :32].to(torch.int32)
    hi_b = codes[..., 32:].to(torch.int32)
    nib = torch.stack((lo_b & 15, lo_b >> 4), -1).reshape(c.shape)
    two = torch.stack([(hi_b >> (2 * t)) & 3 for t in range(4)], -1).reshape(c.shape)
    if not torch.equal(table[(nib | (two << 4)).long()], b):
        return None
    return codes, table


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


# MARLINX_LEAN (csrc lean_kernel): leaner instruction streams for the shared expert
# projections at M <= 8, same arithmetic and output bits. Comma list: "up", "down"
# ("1" = both, "0" or "" = off; default "down": the lean shared down). The lean down with the zero-row skip reads the Z2
# copy (z2_repack) in place of the zs_repack copy (same size).
_LEAN_ENV = os.environ.get("MARLINX_LEAN", "down")
LEAN_SHAPES = {(3712, 2688): "up", (2688, 3712): "down"}
LEAN_LAYERS = frozenset(("up", "down") if _LEAN_ENV == "1" else
                        (v for v in _LEAN_ENV.split(",") if v in ("up", "down")))
# Z2 nibble order inside each zs_repack word (byte j = feature 16j+g | feature 16j+g+8 << 4,
# i.e. nibble 2j = lo_j, 2j+1 = hi_j): new nibble i <- old nibble _Z2_SRC[rows][i].
# Rows 2t, 2t+1 (k row < 8) feed the even nibbles of Marlin's word, rows 2t+8, 2t+9 the odd.
_Z2_SRC = ((0, 2, 1, 3, 4, 6, 5, 7), (2, 0, 3, 1, 6, 4, 7, 5))


@torch.no_grad()
def z2_repack(w: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """zs_repack(w) with the nibbles of every word permuted to Z2 order (csrc block_mma_z2)."""
    z = zs_repack(w, n, k).to(torch.int64) & 0xFFFFFFFF  # [K/16][16][N/64][8]
    out = torch.zeros_like(z)
    for half in (0, 1):
        rows = slice(0, 8) if half == 0 else slice(8, 16)
        src = z[:, rows]
        acc = torch.zeros_like(src)
        for i, j in enumerate(_Z2_SRC[half]):
            acc |= ((src >> (4 * j)) & 15) << (4 * i)
        out[:, rows] = acc
    out = out - (out >= (1 << 31)).to(torch.int64) * (1 << 32)
    return out.to(torch.int32).contiguous()


def gemm(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
         global_scale: torch.Tensor, n: int, k: int, depth: int = DEPTH,
         relu2: bool = False, zs: torch.Tensor | None = None,
         su: torch.Tensor | None = None, lean: bool = False,
         vrows: torch.Tensor | None = None, sut: torch.Tensor | None = None) -> torch.Tensor:
    """out[M, n] for a 2-D bf16 x[M, k] (1 <= M <= 8); no guard beyond the C side's.
    vrows (lean zero-row-skip down only): live verify rows, rows at or past it read as zero."""
    global _pdl_active
    lib = _load()
    assert lib is not None, _load_error
    m = x.shape[0]
    # Loader lookahead only: both variants keep the same Cursor/MMA/fold order.
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
    if lean and su is None:
        # zs is then the z2_repack copy
        lean_args = args[:11] + (args[12], zs.data_ptr() if zs is not None else None, int(relu2))
        if vrows is not None:
            rc = lib.marlinx_gemm_lean_v(*lean_args, int(use_pdl), vrows.data_ptr())
            if rc and use_pdl and not torch.cuda.is_current_stream_capturing():
                _pdl_active = False
                rc = lib.marlinx_gemm_lean_v(*lean_args, 0, vrows.data_ptr())
            if rc != 0:
                raise RuntimeError(f"marlinx_gemm_lean_v failed: {rc} (M={m} N={n} K={k})")
            return out
        rc = lib.marlinx_gemm_lean(*lean_args, int(use_pdl))
        if rc and use_pdl and not torch.cuda.is_current_stream_capturing():
            _pdl_active = False
            rc = lib.marlinx_gemm_lean(*lean_args, 0)
        if rc != 0:
            raise RuntimeError(f"marlinx_gemm_lean failed: {rc} (M={m} N={n} K={k})")
        return out
    if su is not None and sut is not None:
        su_args = args + (zs.data_ptr() if zs is not None else None, su.data_ptr(), int(relu2))
        rc = lib.marlinx_gemm_su6(*su_args, int(use_pdl), sut.data_ptr())
        if rc and use_pdl and not torch.cuda.is_current_stream_capturing():
            _pdl_active = False
            rc = lib.marlinx_gemm_su6(*su_args, 0, sut.data_ptr())
        if rc != 0:
            raise RuntimeError(f"marlinx_gemm_su6 failed: {rc} (M={m} N={n} K={k} depth={depth})")
        return out
    if su is not None:
        su_args = args + (zs.data_ptr() if zs is not None else None, su.data_ptr(), int(relu2))
        rc = lib.marlinx_gemm_su(*su_args, int(use_pdl))
        if rc and use_pdl and not torch.cuda.is_current_stream_capturing():
            _pdl_active = False
            rc = lib.marlinx_gemm_su(*su_args, 0)
        if rc != 0:
            raise RuntimeError(f"marlinx_gemm_su failed: {rc} (M={m} N={n} K={k} depth={depth})")
        return out
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


# Verify-row prefix for the lean zero-row-skip down (MARLINX_LEAN_VROWS=0 disables): the 5- or
# 6-row verify's shared down reads the spec loop's live-row count (smallk.live_verify_rows), so
# padded rows add no zero-skip weight rows. Admitted per layer only after a bitwise comparison
# of the live rows with the unmasked lean call, every live count, signed and empty-row probes.
LEAN_VROWS = os.environ.get("MARLINX_LEAN_VROWS", "1") != "0"


def _live_rows(layer, x2: torch.Tensor):
    if (not LEAN_VROWS or getattr(layer, "_marlinx_zs", None) is None
            or not getattr(layer, "_marlinx_lean", False)
            or getattr(layer, "_marlinx_su", None) is not None):
        return None
    ok = getattr(layer, "_marlinx_vrows_ok", None)
    if ok is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        ok = _admit_vrows(layer, x2.shape[1])
        layer._marlinx_vrows_ok = ok
    if not ok:
        return None
    import sys as _sys
    smallk = _sys.modules.get("smallk")
    if smallk is None or not hasattr(smallk, "live_verify_rows"):
        return None
    return smallk.live_verify_rows(x2.device)


@torch.no_grad()
def _admit_vrows(layer, k: int) -> bool:
    lib = _load()
    if lib is None or not hasattr(lib, "marlinx_gemm_lean_v"):
        return False
    try:
        n = layer.output_size_per_partition
        dev = layer.weight.device
        gen = torch.Generator(device=dev).manual_seed(5126)
        for m in (5, 6):
            for scale in (0.001, 1.0, 30.0):
                h = torch.randn((m, k), device=dev, generator=gen) * scale
                keep = torch.rand((m, k), device=dev, generator=gen) < 0.4
                x = (h.relu().square() * keep).to(torch.bfloat16).contiguous()
                x.view(torch.int16)[:, :4] = torch.tensor([0, -32768, 1, -32767], dtype=torch.int16,
                                                          device=dev)
                x[1].zero_()
                base = gemm(x, layer.weight, layer.weight_scale, layer.weight_global_scale, n, k,
                            zs=layer._marlinx_zs, lean=True)
                for live in range(1, m + 1):
                    vr = torch.tensor([live], dtype=torch.int32, device=dev)
                    out = gemm(x, layer.weight, layer.weight_scale, layer.weight_global_scale, n, k,
                               zs=layer._marlinx_zs, lean=True, vrows=vr)
                    torch.cuda.synchronize(dev)
                    if not torch.equal(out[:live].view(torch.int16), base[:live].view(torch.int16)):
                        return False
                    if not torch.equal(out[live:].view(torch.int16),
                                       torch.zeros_like(out[live:]).view(torch.int16)):
                        return False
                all_rows = torch.tensor([1 << 30], dtype=torch.int32, device=dev)
                out = gemm(x, layer.weight, layer.weight_scale, layer.weight_global_scale, n, k,
                           zs=layer._marlinx_zs, lean=True, vrows=all_rows)
                torch.cuda.synchronize(dev)
                if not torch.equal(out.view(torch.int16), base.view(torch.int16)):
                    return False
        return True
    except Exception:  # noqa: BLE001 - the unmasked lean call stays
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
    vrows = _live_rows(layer, x2) if m in (5, 6) else None
    out = gemm(x2, layer.weight, layer.weight_scale, layer.weight_global_scale, n, k,
               zs=getattr(layer, "_marlinx_zs", None), su=getattr(layer, "_marlinx_su", None),
               lean=getattr(layer, "_marlinx_lean", False), vrows=vrows,
               sut=getattr(layer, "_marlinx_sut", None))
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
    hit = _su_by_scale.get(weight_scale.data_ptr())
    su = hit[1] if hit is not None and hit[0].shape == weight_scale.shape else None
    lean = su is None and _lean_active and "up" in LEAN_LAYERS
    out = gemm(x2, weight, weight_scale, global_scale, n, k, relu2=True, su=su, lean=lean)
    return out.reshape(*x.shape[:-1], n)

"""zskip: humming-exact routed-expert DOWN projection that skips zero-activation weight rows.

libzskip.so (ctypes, static cudart) computes, for the Nemotron-3.5 routed experts' w2
(NVFP4 W4A16, N=2688, K=1856, 128 experts) at <= 4 tokens, exactly what humming 0.1.10's
batch-invariant indexed GEMM computes (csrc/zskip.cu documents the chain), from a k-row-major
repack of the checkpoint tensors (repack()). In SKIP mode it does not load the k rows whose
activations are zero in every row routed to the expert.

Layouts produced by repack(w2_weight, w2_weight_scale, layout):
  nibbles: per expert and k row, 1344 bytes; 128-column tile t, warp slice w (32 columns), lane
           column c (0..7): bytes [64t + 16w + 2c, +2) = the u16 whose nibble j (bits 4j..4j+3)
           is the e2m1 code of column 128t + 32w + 8j + c.
           layout "rows":  [E][K][1344]         (k row contiguous across all of N)
           layout "tiles": [E][21][K][64]       (a 128-column tile's k rows contiguous)
  scales:  per expert and k16 group g, 2688 bytes; bytes [128t + 32w + 4c + j] = scale of column
           128t + 32w + 8j + c.  "rows": [E][116][2688], "tiles": [E][21][116][128].
"""

from __future__ import annotations

import ctypes
import os

import torch

__all__ = ["LIB_PATH", "available", "load_error", "init", "repack", "down", "probe", "epi", "config_info",
           "DENSE", "SKIP", "LOADONLY", "LOADONLY_DENSE", "K", "N", "G", "NUM_TILES", "CONFIGS", "Packed"]

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libzskip.so")

K = 1856
N = 2688
G = K // 16
NUM_TILES = N // 128
PAIR_CONFIGS = {113: (1, 3), 118: (1, 3), 119: (3, 3), 153: (3, 3),
                154: (2, 3), 155: (2, 3), 156: (1, 2), 157: (1, 4), 158: (2, 4)}
DENSE = 0
SKIP = 1
LOADONLY = 2        # timing probe: SKIP's loads, no math (output garbage)
LOADONLY_DENSE = 3  # timing probe: DENSE's loads, no math
SETUPONLY = 4       # timing probe: masks and lists only
# config id -> (warps, RB rows per row stage, NR row stages, QB groups per group stage, NQ group stages);
# must match ZSKIP_CONFIGS in csrc/zskip.cu
CONFIGS = {0: (4, 32, 4, 8, 6), 1: (4, 32, 3, 8, 4), 2: (2, 32, 4, 8, 6), 3: (2, 32, 3, 8, 4),
           4: (6, 32, 4, 8, 6), 5: (6, 32, 3, 8, 4), 6: (12, 32, 3, 8, 4), 7: (4, 16, 6, 8, 6),
           8: (7, 32, 3, 8, 4), 9: (3, 32, 4, 8, 6)}
# v4 (layout "swap" only): config id -> (warps per CTA, ring depth)
CONFIGS4 = {100: (2, 4, 1, 16), 101: (2, 8, 1, 16), 102: (3, 8, 1, 16), 103: (6, 4, 1, 16), 104: (6, 8, 1, 16), 105: (7, 8, 1, 16), 106: (2, 12, 1, 16), 107: (1, 8, 1, 16), 120: (2, 4, 2, 16), 121: (2, 4, 4, 32), 122: (2, 4, 8, 64), 123: (1, 4, 8, 64), 124: (3, 4, 8, 64), 125: (2, 2, 8, 64), 126: (6, 4, 8, 64), 127: (2, 8, 4, 32), 130: (1, 4, 2, 16), 131: (1, 8, 2, 16), 132: (2, 8, 2, 16), 133: (3, 4, 2, 16), 134: (3, 8, 2, 16), 135: (6, 4, 2, 16), 136: (6, 8, 2, 16), 137: (7, 4, 2, 16), 138: (2, 2, 2, 16), 139: (1, 2, 2, 16), 141: (1, 4, 4, 32), 142: (1, 2, 4, 32), 143: (1, 1, 4, 32), 144: (2, 2, 4, 32), 145: (3, 2, 4, 32), 146: (1, 1, 8, 64), 147: (1, 2, 8, 64), 148: (1, 1, 2, 16), 149: (7, 2, 4, 32), 150: (6, 2, 4, 32), 151: (2, 1, 4, 32), 152: (1, 3, 4, 32)}
CONFIGS4.update({200: (2, 4, 4, 32), 201: (2, 6, 4, 32), 202: (2, 8, 4, 32), 203: (1, 8, 4, 32), 204: (3, 6, 4, 32), 205: (3, 8, 4, 32), 206: (2, 12, 4, 32), 207: (1, 12, 4, 32), 208: (6, 4, 4, 32), 209: (1, 6, 4, 32), 210: (2, 6, 2, 16), 211: (1, 8, 2, 16), 212: (2, 6, 8, 64), 213: (1, 6, 8, 64)})
CONFIGS4.update({300: (2, 2, 4, 32), 301: (1, 2, 4, 32), 302: (2, 4, 4, 32), 303: (3, 2, 4, 32), 304: (1, 4, 4, 32), 305: (2, 1, 4, 32), 306: (6, 2, 4, 32), 307: (2, 2, 2, 16), 308: (1, 2, 8, 64), 309: (2, 3, 4, 32)})
CONFIGS4.update({400: (2, 2, 4, 32), 401: (3, 2, 4, 32), 402: (6, 2, 4, 32), 403: (7, 2, 4, 32), 404: (14, 2, 4, 32), 405: (6, 4, 4, 32), 406: (7, 4, 4, 32), 407: (6, 3, 4, 32), 408: (14, 3, 4, 32), 409: (6, 2, 2, 16), 410: (7, 2, 2, 16), 411: (6, 2, 8, 64), 412: (3, 4, 4, 32), 413: (21, 2, 4, 32)})
CONFIGS4.update({500: (2, 2, 4, 32), 501: (6, 2, 4, 32), 502: (6, 4, 4, 32), 503: (6, 3, 4, 32), 504: (3, 4, 4, 32), 505: (7, 2, 4, 32), 506: (2, 4, 4, 32), 507: (6, 2, 2, 16), 508: (3, 2, 4, 32), 509: (6, 6, 4, 32)})
CONFIGS4[108] = (2, 4, 1, 16)
DEFAULT_CONFIG = int(os.environ.get("ZSKIP_CONFIG", "0"))

_lib = None
_load_error: str | None = None
_ready = False


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    try:
        lib = ctypes.CDLL(LIB_PATH)
        lib.zskip_init.argtypes = []
        lib.zskip_init.restype = ctypes.c_int
        lib.zskip_config.argtypes = [ctypes.c_int, ctypes.c_void_p]
        lib.zskip_config.restype = ctypes.c_int
        P, LL, I = ctypes.c_void_p, ctypes.c_longlong, ctypes.c_int
        lib.zskip_down.argtypes = [P, LL, I, P, P, P, I, P, P, P, P, LL, LL, LL, LL, LL, LL, LL, I, P, I, P, P, I, P, P]
        lib.zskip_down.restype = ctypes.c_int
        prefix_down = getattr(lib, "zskip_down_prefix", None)
        if prefix_down is not None:
            prefix_down.argtypes = lib.zskip_down.argtypes + [P]
            prefix_down.restype = ctypes.c_int
        lib.zskip_probe.argtypes = [ctypes.c_ulonglong, I, I, P, P, P]
        lib.zskip_probe.restype = ctypes.c_int
        lib.zskip_epi.argtypes = [P, I, ctypes.c_float, P, P, P]
        lib.zskip_epi.restype = ctypes.c_int
        _lib = lib
    except OSError as exc:
        _load_error = str(exc)
    return _lib


def prefix_available() -> bool:
    lib = _load()
    return lib is not None and getattr(lib, "zskip_down_prefix", None) is not None


def available() -> bool:
    return _load() is not None


def load_error() -> str | None:
    _load()
    return _load_error


_pdl_on = False  # zskip_set_pdl(1) took effect


def init() -> bool:
    """Loads the kernels and sets their smem attributes (call outside CUDA graph capture)."""
    global _ready, _pdl_on
    if _ready:
        return True
    lib = _load()
    if lib is None or torch.cuda.is_current_stream_capturing():
        return False
    rc = lib.zskip_init()
    if rc != 0:
        raise RuntimeError(f"zskip_init failed: {rc}")
    # Programmatic dependent launch for the v4 pair (prep, down4): both kernels
    # wait for the preceding grid before any memory access. ZSKIP_PDL=0 keeps
    # the ordinary launch; a library built with -DZSKIP_NO_PDL has no switch.
    if os.environ.get("ZSKIP_PDL", "1") != "0":
        try:
            set_pdl = lib.zskip_set_pdl
        except AttributeError:
            set_pdl = None
        if set_pdl is not None:
            set_pdl.argtypes = [ctypes.c_int]
            set_pdl.restype = ctypes.c_int
            _pdl_on = set_pdl(1) == 0
    _ready = True
    return True


def config_info(config: int) -> dict:
    """registers, dynamic smem bytes and warps of a kernel config."""
    lib = _load()
    assert lib is not None, _load_error
    info = (ctypes.c_int * 4)()
    rc = lib.zskip_config(config, info)
    if rc != 0:
        raise ValueError(f"zskip config {config}: {rc}")
    return {"regs": info[0], "smem": info[1], "warps": info[2], "ctas_per_sm": info[3],
            "shape": (*PAIR_CONFIGS[config], 1, 16) if config in PAIR_CONFIGS else CONFIGS.get(config) or CONFIGS4.get(config)}


BI_BYTES = 5120
MAX_BLOCKS = 4096
_scratch: dict = {}


def scratch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-device workspace (block lists) and work-queue counter; allocate outside graph capture."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    ws = _scratch.get(idx)
    if ws is None:
        ws = (torch.zeros(MAX_BLOCKS * BI_BYTES, dtype=torch.uint8, device=device),
              torch.zeros(4, dtype=torch.int32, device=device))
        _scratch[idx] = ws
    return ws


class Packed:
    """Repacked w2 of one MoE layer (every checkpoint byte, k-row-major)."""

    def __init__(self, w: torch.Tensor, s: torch.Tensor, gs: torch.Tensor, layout: str, strides: tuple):
        self.w, self.s, self.gs, self.layout = w, s, gs, layout
        (self.w_expert, self.w_tile, self.w_k, self.s_expert, self.s_tile, self.s_g) = strides

    def nbytes(self) -> int:
        return self.w.numel() + self.s.numel() + self.gs.numel() * 4


@torch.no_grad()
def repack(w2_weight: torch.Tensor, w2_weight_scale: torch.Tensor, w2_global_scale: torch.Tensor,
           layout: str = "rows", chunk: int = 16) -> Packed:
    """ModelOpt NVFP4 w2 (uint8 [E, N, K/2], low nibble = even k; e4m3 [E, N, K/16]; fp32 [E])
    -> Packed. Input-independent and value-preserving; raises on NaN scales (the skip would turn
    0 * NaN = NaN into +0)."""
    E = w2_weight.shape[0]
    assert w2_weight.dtype == torch.uint8 and tuple(w2_weight.shape) == (E, N, K // 2), w2_weight.shape
    sc = w2_weight_scale.view(torch.uint8)
    assert tuple(sc.shape) == (E, N, G), sc.shape
    if bool(((sc & 0x7F) == 0x7F).any()):
        raise ValueError("zskip: NaN e4m3 scale in w2")
    gs = w2_global_scale.reshape(-1).to(torch.float32).contiguous()
    assert gs.numel() == E
    dev = w2_weight.device
    if layout == "rows":
        W = torch.empty(E, K, N // 2, dtype=torch.uint8, device=dev)
        S = torch.empty(E, G, N, dtype=torch.uint8, device=dev)
    elif layout == "tiles":
        W = torch.empty(E, NUM_TILES, K, 64, dtype=torch.uint8, device=dev)
        S = torch.empty(E, NUM_TILES, G, 128, dtype=torch.uint8, device=dev)
    elif layout == "swap":
        W = torch.empty(E, K, N // 2, dtype=torch.uint8, device=dev)
        S = torch.empty(E, G, N, dtype=torch.uint8, device=dev)
    else:
        raise ValueError(layout)
    for e0 in range(0, E, chunk):
        e1 = min(E, e0 + chunk)
        w = w2_weight[e0:e1]
        nib = torch.stack([w & 0xF, w >> 4], dim=-1).reshape(e1 - e0, N, K)  # [e, n, k]
        if layout == "swap":
            # n = 64T + 16c + 8h + g -> byte 32T + 4g + c, nibble h; scales byte 64T + 8g + 2c + h
            x = nib.transpose(1, 2).reshape(e1 - e0, K, N // 64, 4, 2, 8).permute(0, 1, 2, 5, 3, 4)
            W[e0:e1] = (x[..., 0] | (x[..., 1] << 4)).reshape(e1 - e0, K, N // 2)
            sx = sc[e0:e1].transpose(1, 2).reshape(e1 - e0, G, N // 64, 4, 2, 8).permute(0, 1, 2, 5, 3, 4)
            S[e0:e1] = sx.reshape(e1 - e0, G, N)
            del nib, x, sx
            continue
        x = nib.transpose(1, 2).reshape(e1 - e0, K, NUM_TILES, 4, 4, 8)      # n = 128t + 32w + 8j + c
        x = x.permute(0, 1, 2, 3, 5, 4).reshape(e1 - e0, K, NUM_TILES, 4, 8, 2, 2)  # [.., c, jb, jn]
        b = (x[..., 0] | (x[..., 1] << 4)).reshape(e1 - e0, K, NUM_TILES, 64)
        s = sc[e0:e1].transpose(1, 2).reshape(e1 - e0, G, NUM_TILES, 4, 4, 8)
        s = s.permute(0, 1, 2, 3, 5, 4).reshape(e1 - e0, G, NUM_TILES, 128)
        if layout == "rows":
            W[e0:e1] = b.reshape(e1 - e0, K, N // 2)
            S[e0:e1] = s.reshape(e1 - e0, G, N)
        else:
            W[e0:e1] = b.permute(0, 2, 1, 3)
            S[e0:e1] = s.permute(0, 2, 1, 3)
        del nib, x, b, s
    if layout in ("rows", "swap"):
        strides = (K * (N // 2), 64, N // 2, G * N, 128, N)
    else:
        strides = (NUM_TILES * K * 64, K * 64, 64, NUM_TILES * G * 128, G * 128, 128)
    return Packed(W, S, gs, layout, strides)


@torch.no_grad()
def to_swap2(pk: Packed, chunk: int = 16) -> Packed:
    """dadon 12f98e36: interleave paired swap tiles, preserving every weight/scale byte.

    Keep pk alive for pzskip and configs 108/100. This additional input-independent
    buffer costs about 357 MB per layer. Chunk temporaries bound load-time peak use.
    """
    assert pk.layout == "swap"
    E = pk.w.shape[0]
    W, S = torch.empty_like(pk.w), torch.empty_like(pk.s)
    for e0 in range(0, E, chunk):
        e1 = min(E, e0 + chunk)
        W[e0:e1] = pk.w[e0:e1].reshape(e1-e0, K, N//128, 2, 8, 4).permute(
            0, 1, 2, 4, 3, 5).reshape(e1-e0, K, N//2)
        S[e0:e1] = pk.s[e0:e1].reshape(e1-e0, G, N//128, 2, 8, 8).permute(
            0, 1, 2, 4, 3, 5).reshape(e1-e0, G, N)
    return Packed(W, S, pk.gs, "swap2", (K*(N//2), 64, N//2, G*N, 128, N))


@torch.no_grad()
def to_swapc(pk: Packed) -> Packed:
    """swap -> swapc: per expert, 64-column tile and k16 group, the 16 k rows (32 bytes each) and
    the 64 scale bytes contiguous. Same bytes, input-independent."""
    assert pk.layout == "swap"
    E = pk.w.shape[0]
    W = pk.w.view(E, G, 16, N // 64, 32).permute(0, 3, 1, 2, 4).contiguous()
    S = pk.s.view(E, G, N // 64, 64).permute(0, 2, 1, 3).contiguous()
    return Packed(W, S, pk.gs, "swapc", ((N // 64) * G * 512, G * 512, 512, (N // 64) * G * 64, G * 64, 64))


def down(a: torch.Tensor, sorted_ids: torch.Tensor, expert_ids: torch.Tensor, num_tokens_padded: torch.Tensor,
         packed: Packed, out: torch.Tensor, mode: int = SKIP, config: int | None = None,
         max_blocks: int | None = None, stats: torch.Tensor | None = None, ctas_per_sm: int = 0,
         trace: torch.Tensor | None = None,
         verify_rows: torch.Tensor | None = None) -> torch.Tensor:
    """out[r] = humming w2 output row r for the moe_align layout (block 16) of a <= 4-token call.
    a: bf16 [shape_m, K] (relu^2 activations, rows = token * top_k + slot); out: bf16 [shape_m, N]."""
    global _pdl_on
    lib = _load()
    assert lib is not None, _load_error
    if config is None:
        config = 100
    if (config in PAIR_CONFIGS) != (packed.layout == "swap2"):
        raise ValueError("paired configs require swap2; legacy configs require their original layout")
    shape_m = a.shape[0]
    assert a.dtype == torch.bfloat16 and a.shape[1] == K and a.stride(1) == 1
    assert out.dtype == torch.bfloat16 and out.shape[1] == N and out.stride(1) == 1 and out.shape[0] >= shape_m
    if max_blocks is None:
        max_blocks = shape_m  # every block holds >= 1 valid row
    max_blocks = min(max_blocks, expert_ids.numel())
    assert max_blocks <= MAX_BLOCKS
    ws, queue = scratch(a.device)
    args = (
        a.data_ptr(), a.stride(0), shape_m, sorted_ids.data_ptr(), expert_ids.data_ptr(),
        num_tokens_padded.data_ptr(), max_blocks, packed.w.data_ptr(), packed.s.data_ptr(), packed.gs.data_ptr(),
        out.data_ptr(), out.stride(0), packed.w_expert, packed.w_tile, packed.w_k, packed.s_expert,
        packed.s_tile, packed.s_g, mode, stats.data_ptr() if stats is not None else None, config,
        ws.data_ptr(), queue.data_ptr(), ctas_per_sm, trace.data_ptr() if trace is not None else None,
        torch.cuda.current_stream(a.device).cuda_stream,
    )
    launch = lib.zskip_down
    if verify_rows is not None:
        assert shape_m == 30 and config in PAIR_CONFIGS and mode == SKIP
        assert verify_rows.device == a.device and verify_rows.dtype == torch.int32
        assert verify_rows.numel() == 1 and verify_rows.is_contiguous()
        prefix_launch = getattr(lib, "zskip_down_prefix", None)
        if prefix_launch is not None:
            launch = prefix_launch
            args += (verify_rows.data_ptr(),)
    rc = launch(*args)
    if rc != 0 and _pdl_on and not torch.cuda.is_current_stream_capturing():
        # The PDL launch was rejected (the first calls run eagerly, before any capture):
        # use ordinary launches from now on and redo this call.
        lib.zskip_set_pdl(0)
        _pdl_on = False
        rc = launch(*args)
    if rc != 0:
        raise RuntimeError(f"zskip_down failed: {rc} (config={config})")
    return out


def probe(seed: int, blocks: int, iters: int) -> tuple[list[int], list[int]]:
    """Tensor-core semantics probe (see csrc probe_kernel). Returns (counts[7*8], samples)."""
    lib = _load()
    assert lib is not None, _load_error
    counts = torch.zeros(7 * 8, dtype=torch.int64, device="cuda")
    samples = torch.zeros(1 + 64 * 8, dtype=torch.int32, device="cuda")
    rc = lib.zskip_probe(seed, blocks, iters, counts.data_ptr(), samples.data_ptr(),
                         torch.cuda.current_stream().cuda_stream)
    if rc != 0:
        raise RuntimeError(f"zskip_probe failed: {rc}")
    torch.cuda.synchronize()
    return counts.tolist(), [v & 0xFFFFFFFF for v in samples.tolist()]


def epi(x_bits: torch.Tensor, gs: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(bf16 bits, fp32 bits) of the kernel epilogue applied to fp32 bit patterns x_bits (int32)."""
    lib = _load()
    assert lib is not None, _load_error
    n = x_bits.numel()
    ob = torch.zeros(n, dtype=torch.int32, device="cuda")
    om = torch.zeros(n, dtype=torch.int32, device="cuda")
    rc = lib.zskip_epi(x_bits.data_ptr(), n, gs, ob.data_ptr(), om.data_ptr(), torch.cuda.current_stream().cuda_stream)
    if rc != 0:
        raise RuntimeError(f"zskip_epi failed: {rc}")
    return ob, om

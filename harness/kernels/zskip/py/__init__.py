"""zskip: humming-exact routed-expert DOWN projection that skips zero-activation weight rows.

libzskip.so (ctypes, static cudart) computes, for the Nemotron-3.5 routed experts' w2
(NVFP4 W4A16, N=2688, K=1856, 128 experts) at 1..8 tokens, exactly what humming 0.1.10's
batch-invariant indexed GEMM computes (csrc/zskip.cu and README.md document the chain), from a
k-row-major repack of the checkpoint tensors (repack()). In SKIP mode it does not load the k
rows whose activations are zero in every row routed to the expert.

Layout "swap" (repack): per expert and k row, 1344 bytes = 42 tiles of 64 columns x 32 bytes;
byte 32T + 4g + c holds the e2m1 codes of columns 64T + 16c + g (low nibble) and
64T + 16c + g + 8 (high nibble), so one k row of a tile is one 32-byte sector and lane (g, t)
of a warp reads the 4 bytes of A-fragment nibbles it needs. Scales: per expert and k16 group,
2688 bytes, byte 64T + 8g + 2c + h = the scale of column 64T + 16c + g + 8h.
Layout "swap2": the same per 128-column pair P with the two tiles u = 0, 1 interleaved per lane:
byte 64P + 8g + 4u + c, scales byte 128P + 16g + 8u + 2c + h (one 8-byte load per k row and
lane, 64-byte row segments).
"""

from __future__ import annotations

import ctypes
import os

import torch

__all__ = ["LIB_PATH", "available", "load_error", "init", "repack", "down", "probe", "epi", "config_info", "layout_of", "rows_of",
           "DENSE", "SKIP", "LOADONLY", "LOADONLY_DENSE", "K", "N", "G", "NUM_TILES", "CONFIGS", "Packed"]

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libzskip.so")

K = 1856
N = 2688
G = K // 16
NUM_TILES = N // 128
DENSE = 0
SKIP = 1
LOADONLY = 2        # timing probe: SKIP's loads, no math (output garbage)
LOADONLY_DENSE = 3  # timing probe: DENSE's loads, no math
SETUPONLY = 4       # timing probe: masks and lists only
# config id -> kernel shape; must match ZSKIP4_CONFIGS / ZSKIP5_CONFIGS in csrc/zskip.cu
# v4 (prep kernel + register ring): config id -> (64-column tiles per warp, warps per CTA, ring depth);
# tiles per warp 1 reads layout "swap", 2 reads "swap2"
# config id -> (64-column tiles per warp, warps per CTA, ring depth, token n-blocks of 8 per warp,
# half pair); a block of `slots` sorted ids may hold <= 8 * n-blocks rows
CONFIGS = {100: (1, 2, 4, 1, False), 110: (2, 1, 4, 1, False), 113: (2, 1, 3, 1, False),
           212: (2, 1, 2, 2, False), 300: (1, 2, 3, 4, True), 301: (1, 2, 2, 4, True),
           302: (1, 6, 3, 4, True), 303: (1, 2, 4, 4, True)}


def layout_of(config: int) -> str:
    tpw, _, _, _, half = CONFIGS[config]
    return "swap2" if tpw == 2 or half else "swap"


def rows_of(config: int) -> int:
    """Most valid rows a block may hold for this config."""
    return 8 * CONFIGS[config][3]


DEFAULT_CONFIG = int(os.environ.get("ZSKIP_CONFIG", "113"))

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
        lib.zskip_down.argtypes = [P, LL, I, P, P, P, I, P, P, P, P, LL, LL, LL, LL, LL, LL, LL, I, P, I, P, I, I, P]
        lib.zskip_down.restype = ctypes.c_int
        lib.zskip_probe.argtypes = [ctypes.c_ulonglong, I, I, P, P, P]
        lib.zskip_probe.restype = ctypes.c_int
        lib.zskip_epi.argtypes = [P, I, ctypes.c_float, P, P, P]
        lib.zskip_epi.restype = ctypes.c_int
        _lib = lib
    except OSError as exc:
        _load_error = str(exc)
    return _lib


def available() -> bool:
    return _load() is not None


def load_error() -> str | None:
    _load()
    return _load_error


def init() -> bool:
    """Loads the kernels and sets their smem attributes (call outside CUDA graph capture)."""
    global _ready
    if _ready:
        return True
    lib = _load()
    if lib is None or torch.cuda.is_current_stream_capturing():
        return False
    rc = lib.zskip_init()
    if rc != 0:
        raise RuntimeError(f"zskip_init failed: {rc}")
    # programmatic dependent launch of the (prep, down4) pair; ZSKIP_PDL=0 keeps the ordinary launch
    lib.zskip_set_pdl.argtypes = [ctypes.c_int]
    lib.zskip_set_pdl.restype = ctypes.c_int
    lib.zskip_set_pdl(1 if os.environ.get("ZSKIP_PDL", "1") != "0" else 0)
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
            "shape": CONFIGS.get(config), "layout": layout_of(config)}


BI_BYTES = 4864
MAX_BLOCKS = 4096  # >= the block-16 alignment of PREFILL_MAX_TOKENS (4096 tokens: 1656 blocks)
_scratch: dict = {}


def scratch(device: torch.device) -> torch.Tensor:
    """Per-device workspace of the v4 prep kernel; allocate outside graph capture."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    ws = _scratch.get(idx)
    if ws is None:
        ws = torch.zeros(MAX_BLOCKS * BI_BYTES, dtype=torch.uint8, device=device)
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
           layout: str = "swap", chunk: int = 16) -> Packed:
    """ModelOpt NVFP4 w2 (uint8 [E, N, K/2], low nibble = even k; e4m3 [E, N, K/16]; fp32 [E])
    -> Packed in the "swap" layout (module docstring). Input-independent and value-preserving;
    raises on NaN scales (the skip would turn 0 * NaN = NaN into +0)."""
    if layout not in ("swap", "swap2"):
        raise ValueError(layout)
    E = w2_weight.shape[0]
    assert w2_weight.dtype == torch.uint8 and tuple(w2_weight.shape) == (E, N, K // 2), w2_weight.shape
    sc = w2_weight_scale.view(torch.uint8)
    assert tuple(sc.shape) == (E, N, G), sc.shape
    if bool(((sc & 0x7F) == 0x7F).any()):
        raise ValueError("zskip: NaN e4m3 scale in w2")
    gs = w2_global_scale.reshape(-1).to(torch.float32).contiguous()
    assert gs.numel() == E
    dev = w2_weight.device
    W = torch.empty(E, K, N // 2, dtype=torch.uint8, device=dev)
    S = torch.empty(E, G, N, dtype=torch.uint8, device=dev)
    for e0 in range(0, E, chunk):
        e1 = min(E, e0 + chunk)
        w = w2_weight[e0:e1]
        nib = torch.stack([w & 0xF, w >> 4], dim=-1).reshape(e1 - e0, N, K)  # [e, n, k]
        if layout == "swap":
            # n = 64T + 16c + 8h + g -> byte 32T + 4g + c, nibble h; scales byte 64T + 8g + 2c + h
            x = nib.transpose(1, 2).reshape(e1 - e0, K, N // 64, 4, 2, 8).permute(0, 1, 2, 5, 3, 4)
            sx = sc[e0:e1].transpose(1, 2).reshape(e1 - e0, G, N // 64, 4, 2, 8).permute(0, 1, 2, 5, 3, 4)
        else:
            # n = 128P + 64u + 16c + 8h + g -> byte 64P + 8g + 4u + c, nibble h;
            # scales byte 128P + 16g + 8u + 2c + h
            x = nib.transpose(1, 2).reshape(e1 - e0, K, N // 128, 2, 4, 2, 8).permute(0, 1, 2, 6, 3, 4, 5)
            sx = sc[e0:e1].transpose(1, 2).reshape(e1 - e0, G, N // 128, 2, 4, 2, 8).permute(0, 1, 2, 6, 3, 4, 5)
        W[e0:e1] = (x[..., 0] | (x[..., 1] << 4)).reshape(e1 - e0, K, N // 2)
        S[e0:e1] = sx.reshape(e1 - e0, G, N)
        del nib, x, sx
    strides = (K * (N // 2), 64, N // 2, G * N, 128, N)
    return Packed(W, S, gs, layout, strides)


def down(a: torch.Tensor, sorted_ids: torch.Tensor, expert_ids: torch.Tensor, num_tokens_padded: torch.Tensor,
         packed: Packed, out: torch.Tensor, mode: int = SKIP, config: int | None = None,
         max_blocks: int | None = None, stats: torch.Tensor | None = None, slots: int = 16,
         ediv: int = 1) -> torch.Tensor:
    """out[r] = humming w2 output row r. Blocks of `slots` sorted ids (16: a block-16 moe_align layout;
    32 with ediv 2: halves of humming's block-64 layout), each holding <= rows_of(config) valid rows.
    a: bf16 [shape_m, K] (relu^2 activations, rows = token * top_k + slot); out: bf16 [shape_m, N]."""
    lib = _load()
    assert lib is not None, _load_error
    if config is None:
        config = DEFAULT_CONFIG
    shape_m = a.shape[0]
    assert a.dtype == torch.bfloat16 and a.shape[1] == K and a.stride(1) == 1
    assert out.dtype == torch.bfloat16 and out.shape[1] == N and out.stride(1) == 1 and out.shape[0] >= shape_m
    if max_blocks is None:
        max_blocks = shape_m  # every block holds >= 1 valid row
    max_blocks = min(max_blocks, expert_ids.numel() * ediv, sorted_ids.numel() // slots)
    assert max_blocks <= MAX_BLOCKS
    ws = scratch(a.device)
    if packed.layout != layout_of(config):
        raise ValueError(f"zskip config {config} reads layout {layout_of(config)}, not {packed.layout}")
    rc = lib.zskip_down(
        a.data_ptr(), a.stride(0), shape_m, sorted_ids.data_ptr(), expert_ids.data_ptr(),
        num_tokens_padded.data_ptr(), max_blocks, packed.w.data_ptr(), packed.s.data_ptr(), packed.gs.data_ptr(),
        out.data_ptr(), out.stride(0), packed.w_expert, packed.w_tile, packed.w_k, packed.s_expert,
        packed.s_tile, packed.s_g, mode, stats.data_ptr() if stats is not None else None, config,
        ws.data_ptr(), slots, ediv, torch.cuda.current_stream(a.device).cuda_stream,
    )
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

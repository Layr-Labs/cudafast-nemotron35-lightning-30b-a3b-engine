"""smallk: bit-exact replacements for small latency-bound kernels of the Nemotron-3.5 MTP verify.

libsmallk.so (ctypes, static cudart); csrc/smallk.cu documents the arithmetic each kernel
reproduces and README.md the evidence:

- router_gemm(x, w): the fp32 MoE gate y = x . w^T for M = 4 rows, with the summation tree of the
  cuBLAS gemmSN_TN kernel torch picks for (M = 4, N = 128, K = 2688).
  M = 5 is a new unvalidated specialization of the same independent chains;
  equality to the actual M5 cuBLAS selection is not established yet.
  maybe_router_gemm reads the gate weight from a bf16 shadow when every fp32 value of it is exactly
  a bf16 value (checked once per gate outside capture; the checkpoint stores the gate in bf16 and
  vLLM upcasts it on GB10): same fp32 operands, half the bytes. SMALLK_ROUTER_W16=0 turns it off.
  enable_verify_rows(device) (dormant until called, outside graph capture) makes a one-int device
  buffer (default: every row) that target-model gates then read inside the verify graph; rows at
  or past its value take the last valid row's logits bit for bit, so a verify that masks its
  trailing rows adds no distinct experts for them. The buffer never changes any valid row; only
  the spec loop may lower it, and only when its sampler stops acceptance at the same row.
- topk_sigmoid(logits, bias, ...): vLLM's fused grouped top-k for n_group = topk_group = 1,
  sigmoid scoring, fp32 logits and bias (128 experts).
- align(topk_ids, ...): moe_align_block_size + count_and_sort_expert_tokens in one CTA.
- route(...): top-k and align in one CTA. The router hook (maybe_topk_sigmoid) launches it when
  the align block size for this token count is known from an earlier align call, and parks the
  align outputs; the align hook (maybe_align) hands them out for the same topk_ids storage.

Switches (environment, read at import): SMALLK=0 disables everything; SMALLK_ROUTER=0 and
SMALLK_EARLY_ROUTER=0 (both on by default) disable the router GEMM and its schedule change;
SMALLK_TOPK=1 / SMALLK_ALIGN=1 opt into the top-k and align kernels (off by default: no
end-to-end gain), SMALLK_FUSE=0 then keeps them separate.
SMALLK_EARLY_ROUTER=0 keeps the router GEMM in the compiled graph (before the shared-expert stream
forks) instead of inside the MoE custom op. A call outside a kernel's verified guard returns None
and the caller runs the stock op.
"""

from __future__ import annotations

import ctypes
import os

import torch

try:
    import routerw12 as _routerw12
except ImportError:
    _routerw12 = None

__all__ = ["LIB_PATH", "available", "load_error", "init", "router_gemm", "topk_sigmoid", "align", "route",
           "ROUTER_ON", "ROUTER_W16_ON", "TOPK_ON", "ALIGN_ON", "FUSE_ON", "EARLY_ROUTER_ON", "maybe_router_gemm",
           "maybe_topk_sigmoid", "maybe_align", "VROWS_ALL", "enable_verify_rows", "verify_rows", "reset_verify_rows",
           "gate_w16", "request_verify_rows", "live_verify_rows"]

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libsmallk.so")


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) != "0"


ENABLED = _flag("SMALLK")
ROUTER_ON = ENABLED and _flag("SMALLK_ROUTER")
ROUTER_M5_ON = _flag("SMALLK_ROUTER_M5")
ROUTER_W16_ON = ROUTER_ON and _flag("SMALLK_ROUTER_W16")
# Opt-in: exact, but no end-to-end gain at depth 3 (the routing chain overlaps the shared-expert
# GEMMs on the side stream; a shorter chain only moves that overlap into the routed-expert GEMM).
TOPK_ON = ENABLED and _flag("SMALLK_TOPK", "0")
ALIGN_ON = ENABLED and _flag("SMALLK_ALIGN", "0")
FUSE_ON = TOPK_ON and ALIGN_ON and _flag("SMALLK_FUSE")
# models/nemotron_h.py: the gate runs inside the MoE custom op after the shared-expert stream fork
EARLY_ROUTER_ON = ENABLED and _flag("SMALLK_EARLY_ROUTER")
PDL = int(os.environ.get("SMALLK_PDL", "1"))
ROUTER_ROWS = int(os.environ.get("SMALLK_ROUTER_ROWS", "1"))
TOPK_WARPS = int(os.environ.get("SMALLK_TOPK_WARPS", "4"))

# M4 is verified in the carrier; M5 is the new remote-only experiment.
ROUTER_SHAPES = frozenset({(4, 128, 2688), (5, 128, 2688), (6, 128, 2688)})  # M5/M6: same per-row chains
NUM_EXPERTS = 128
MAX_TOPK = 8
# Stock count_and_sort fills an expert's slots in lane order when all ids sit in one warp; with
# more ids the order across warps is a race, so only numel <= 32 is reproduced.
ALIGN_MAX_NUMEL = 32

_lib = None
_load_error: str | None = None
_ready = False
_align_bs: dict[int, int] = {}  # numel -> block size the experts asked for
_parked: tuple | None = None  # (ids_ptr, numel, block_size, sorted_ids, expert_ids, num_tokens_post_pad)
VROWS_ALL = 1 << 30  # verify_rows value meaning "every row is valid"
_vrows: dict = {}  # torch.device -> int32[1] buffer (enable_verify_rows, outside graph capture)
# request_verify_rows(): the first target-gate call outside graph capture creates the buffer, so it
# exists before the verify graph is captured (each capture size is warmed up eagerly first).
_vrows_wanted = False
# A target gate was captured before the buffer existed: that graph ignores the buffer, so nobody
# may lower it (live_verify_rows then returns None).
_vrows_missed = False
_has_vrows_entry = False
_w16: dict = {}  # gate weight data_ptr -> (weight, bf16 shadow or None when not bf16-exact)
_has_w16_entry = False


def _load():
    global _lib, _load_error, _has_vrows_entry, _has_w16_entry
    if _lib is not None or _load_error is not None:
        return _lib
    try:
        lib = ctypes.CDLL(LIB_PATH)
        vp, ci, cl, cf = ctypes.c_void_p, ctypes.c_int, ctypes.c_long, ctypes.c_float
        lib.sk_version.restype = ci
        lib.sk_init.restype = ci
        lib.sk_router_gemm.argtypes = [vp, cl, ci, vp, ci, ci, vp, cl, ci, ci, ci, vp]
        lib.sk_router_gemm.restype = ci
        # The verify-row and bf16 entry points below take these argument lists from
        # library version 5 on (an older bf16 entry has no vrows argument): an older
        # libsmallk.so keeps fp32 reads and no masking, same logits.
        current = lib.sk_version() >= 5
        try:
            lib.sk_router_gemm_v.argtypes = [vp, cl, ci, vp, ci, ci, vp, cl, ci, ci, ci, vp, vp]
            lib.sk_router_gemm_v.restype = ci
            _has_vrows_entry = current
        except AttributeError:
            _has_vrows_entry = False
        try:
            lib.sk_router_gemm_w16.argtypes = [vp, cl, ci, vp, ci, ci, vp, cl, ci, ci, vp, vp]
            lib.sk_router_gemm_w16.restype = ci
            _has_w16_entry = current
        except AttributeError:
            _has_w16_entry = False
        lib.sk_topk_sigmoid.argtypes = [vp, vp, ci, ci, ci, ci, cf, vp, vp, ci, ci, vp]
        lib.sk_topk_sigmoid.restype = ci
        lib.sk_align.argtypes = [vp, ci, ci, ci, vp, ci, vp, ci, vp, ci, vp]
        lib.sk_align.restype = ci
        lib.sk_route.argtypes = [vp, vp, ci, ci, ci, ci, cf, vp, vp, ci, vp, ci, vp, ci, vp, ci, vp]
        lib.sk_route.restype = ci
        _lib = lib
    except OSError as exc:  # library missing or unloadable: callers fall back
        _load_error = str(exc)
    return _lib


def available() -> bool:
    return _load() is not None


def load_error() -> str | None:
    _load()
    return _load_error


def init() -> bool:
    """Loads the kernels into the context. Must run outside CUDA graph capture."""
    global _ready
    if _ready:
        return True
    lib = _load()
    if lib is None or torch.cuda.is_current_stream_capturing():
        return False
    rc = lib.sk_init()
    if rc != 0:
        raise RuntimeError(f"sk_init failed: {rc}")
    _ready = True
    return True


def _stream(t: torch.Tensor) -> int:
    return torch.cuda.current_stream(t.device).cuda_stream


def _pdl(pdl: int | None) -> int:
    return PDL if pdl is None else pdl


def router_gemm(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor | None = None, rows: int | None = None,
                mode: int = 0, pdl: int | None = None, vrows: torch.Tensor | None = None,
                w16: torch.Tensor | None = None, _packet=None) -> torch.Tensor:
    """y[M, N] = x[M, K] . w[N, K]^T in fp32 (x rows unit-stride, w contiguous). mode 1 (mul+add
    chains) exists only to show the fma chain is the one cuBLAS uses. vrows (int32 device scalar):
    rows m >= vrows get row max(vrows, 1) - 1's logits. w16: bf16 copy of w that holds the same
    values (gate_w16), read instead of w (mode 0 only)."""
    lib = _load()
    assert lib is not None, _load_error
    m, k = x.shape
    n = w.shape[0]
    if out is None:
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    rows = ROUTER_ROWS if rows is None else rows
    if (_packet is not None and _routerw12 is not None
            and _routerw12.run(_packet, x, w, w16, out, rows, mode, _pdl(pdl), vrows)):
        return out
    if vrows is not None:
        assert _has_vrows_entry and vrows.dtype == torch.int32 and vrows.device == x.device
    if w16 is not None:
        assert _has_w16_entry and mode == 0 and w16.dtype == torch.bfloat16 and w16.shape == w.shape
        assert w16.is_contiguous() and w16.device == w.device
        rc = lib.sk_router_gemm_w16(x.data_ptr(), x.stride(0), m, w16.data_ptr(), n, k, out.data_ptr(),
                                    out.stride(0), rows, _pdl(pdl), None if vrows is None else vrows.data_ptr(),
                                    _stream(x))
    elif vrows is not None:
        rc = lib.sk_router_gemm_v(x.data_ptr(), x.stride(0), m, w.data_ptr(), n, k, out.data_ptr(),
                                  out.stride(0), rows, mode, _pdl(pdl), vrows.data_ptr(), _stream(x))
    else:
        rc = lib.sk_router_gemm(x.data_ptr(), x.stride(0), m, w.data_ptr(), n, k, out.data_ptr(), out.stride(0),
                                rows, mode, _pdl(pdl), _stream(x))
    if rc != 0:
        raise RuntimeError(f"sk_router_gemm failed: {rc} (M={m} N={n} K={k})")
    return out


def topk_sigmoid(logits: torch.Tensor, bias: torch.Tensor, topk: int, renormalize: bool, scale: float,
                 warps: int | None = None, pdl: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    lib = _load()
    assert lib is not None, _load_error
    m, e = logits.shape
    values = torch.empty((m, topk), dtype=torch.float32, device=logits.device)
    indices = torch.empty((m, topk), dtype=torch.int32, device=logits.device)
    rc = lib.sk_topk_sigmoid(logits.data_ptr(), bias.data_ptr(), m, e, topk, int(renormalize), float(scale),
                             values.data_ptr(), indices.data_ptr(), TOPK_WARPS if warps is None else warps,
                             _pdl(pdl), _stream(logits))
    if rc != 0:
        raise RuntimeError(f"sk_topk_sigmoid failed: {rc} (M={m} E={e} topk={topk})")
    return values, indices


def align_buffers(numel: int, num_experts: int, block_size: int, device) -> tuple[torch.Tensor, ...]:
    """The three outputs, sized exactly as vLLM's moe_align_block_size allocates them."""
    max_num_tokens_padded = numel + num_experts * (block_size - 1)
    if numel < num_experts:
        max_num_tokens_padded = min(numel * block_size, max_num_tokens_padded)
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=device)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=device)
    return sorted_ids, expert_ids, num_tokens_post_pad


def align(topk_ids: torch.Tensor, num_experts: int, block_size: int, sorted_ids: torch.Tensor,
          expert_ids: torch.Tensor, num_tokens_post_pad: torch.Tensor, pdl: int | None = None) -> None:
    lib = _load()
    assert lib is not None, _load_error
    rc = lib.sk_align(topk_ids.data_ptr(), topk_ids.numel(), num_experts, block_size, sorted_ids.data_ptr(),
                      sorted_ids.numel(), expert_ids.data_ptr(), expert_ids.numel(), num_tokens_post_pad.data_ptr(),
                      _pdl(pdl), _stream(topk_ids))
    if rc != 0:
        raise RuntimeError(f"sk_align failed: {rc} (numel={topk_ids.numel()} E={num_experts} bs={block_size})")


def route(logits: torch.Tensor, bias: torch.Tensor, topk: int, renormalize: bool, scale: float, block_size: int,
          pdl: int | None = None) -> tuple[torch.Tensor, ...]:
    """(values, indices, sorted_ids, expert_ids, num_tokens_post_pad) from one kernel."""
    lib = _load()
    assert lib is not None, _load_error
    m, e = logits.shape
    values = torch.empty((m, topk), dtype=torch.float32, device=logits.device)
    indices = torch.empty((m, topk), dtype=torch.int32, device=logits.device)
    sorted_ids, expert_ids, ntpp = align_buffers(m * topk, e, block_size, logits.device)
    rc = lib.sk_route(logits.data_ptr(), bias.data_ptr(), m, e, topk, int(renormalize), float(scale),
                      values.data_ptr(), indices.data_ptr(), block_size, sorted_ids.data_ptr(), sorted_ids.numel(),
                      expert_ids.data_ptr(), expert_ids.numel(), ntpp.data_ptr(), _pdl(pdl), _stream(logits))
    if rc != 0:
        raise RuntimeError(f"sk_route failed: {rc} (M={m} E={e} topk={topk} bs={block_size})")
    return values, indices, sorted_ids, expert_ids, ntpp


# ---------------------------------------------------------------- guarded entry points for vLLM
def _usable() -> bool:
    if _ready:
        return True
    return _load() is not None and init()


def _cuda_device(device) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def enable_verify_rows(device) -> torch.Tensor | None:
    """Turns the verify-row count on for this device and returns its buffer (int32[1], VROWS_ALL
    until the spec loop lowers it), or None if the router kernel or its verify-row entry point is
    unavailable. Call before graph capture: the verify graph captures the buffer's address, so
    write it in place (copy_ / fill_), never rebind it."""
    if not ROUTER_ON or not _usable() or not _has_vrows_entry:
        return None
    dev = _cuda_device(device)
    buf = _vrows.get(dev)
    if buf is None:
        if dev.type != "cuda" or torch.cuda.is_current_stream_capturing():
            return None
        buf = torch.full((1,), VROWS_ALL, dtype=torch.int32, device=dev)
        _vrows[dev] = buf
    return buf


def verify_rows(device) -> torch.Tensor | None:
    """The buffer enable_verify_rows made for this device, else None (gates then pass nullptr)."""
    return _vrows.get(_cuda_device(device)) if _vrows else None


def request_verify_rows() -> None:
    """Asks for the verify-row buffer (see _vrows_wanted). Call at model load."""
    global _vrows_wanted
    _vrows_wanted = True


def live_verify_rows(device) -> torch.Tensor | None:
    """The verify-row buffer if every captured target-gate call reads it, else None."""
    return None if _vrows_missed else verify_rows(device)


def reset_verify_rows() -> None:
    """Every row valid again on every device (stream-ordered fill)."""
    for buf in _vrows.values():
        buf.fill_(VROWS_ALL)


def maybe_router_gemm(x: torch.Tensor, w: torch.Tensor, row_mask: bool = False) -> torch.Tensor | None:
    """M4 router or follow-on M5 experiment; None for unsupported metadata. row_mask (target-model
    gates only): honour verify_rows."""
    if not ROUTER_ON or x.dim() != 2 or x.shape[0] not in (4, 5, 6) or w.dim() != 2:
        return None
    m, k = x.shape
    n = w.shape[0]
    if m == 5 and not ROUTER_M5_ON:
        return None
    if (
        (m, n, k) not in ROUTER_SHAPES
        or x.dtype != torch.float32
        or w.dtype != torch.float32
        or w.shape[1] != k
        or x.stride(1) != 1
        or not w.is_contiguous()
        or not x.is_cuda
    ):
        return None
    if not _usable():
        return None
    vrows = None
    if row_mask:
        global _vrows_missed
        capturing = torch.cuda.is_current_stream_capturing()
        if _vrows_wanted and not _vrows and not capturing:
            enable_verify_rows(x.device)
        vrows = verify_rows(x.device)
        if vrows is None and capturing:
            _vrows_missed = True
    shadow = gate_w16(w)
    packet = None
    if _routerw12 is not None and row_mask:
        packet = _routerw12.lookup(w, shadow, m, row_mask=True)
    return router_gemm(x, w, vrows=vrows, w16=shadow, _packet=packet)


def gate_w16(w: torch.Tensor) -> torch.Tensor | None:
    """bf16 copy of the fp32 gate weight w if bf16(w) widens back to w bit for bit (signed zeros,
    subnormals, infinities and NaN payloads included), else None. Built once per weight, outside
    CUDA graph capture; a call during capture before the first build reads w itself."""
    if not ROUTER_W16_ON or not _has_w16_entry:
        return None
    ptr = w.data_ptr()
    hit = _w16.get(ptr)
    if hit is not None and hit[0].data_ptr() == ptr and hit[0].shape == w.shape:
        return hit[1]
    if torch.cuda.is_current_stream_capturing():
        return None
    w16 = w.detach().to(torch.bfloat16).contiguous()
    back = w16.to(torch.float32)
    # bitwise: equal() treats -0.0 == +0.0, the widening is exact so compare the int32 views
    exact = bool(torch.equal(back.view(torch.int32), w.detach().contiguous().view(torch.int32)))
    _w16[ptr] = (w, w16 if exact else None)  # keeps w alive, so ptr cannot be reused by another tensor
    return w16 if exact else None


def maybe_topk_sigmoid(logits: torch.Tensor, bias: torch.Tensor | None, topk: int, renormalize: bool,
                       scale: float) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Exact grouped top-k (n_group = topk_group = 1, sigmoid, fp32) or None (caller runs vLLM's op).
    With SMALLK_FUSE on and a known align block size, the align outputs are computed too and parked."""
    global _parked
    _parked = None
    if not TOPK_ON or bias is None or logits.dim() != 2 or logits.shape[0] > 64:
        return None
    m = logits.shape[0]
    if (
        logits.shape[1] != NUM_EXPERTS
        or logits.dtype != torch.float32
        or bias.dtype != torch.float32
        or bias.numel() != NUM_EXPERTS
        or not logits.is_contiguous()
        or not bias.is_contiguous()
        or topk < 1
        or topk > MAX_TOPK
        or m < 1
        or m > 64
        or not logits.is_cuda
    ):
        return None
    if not _usable():
        return None
    bs = _align_bs.get(m * topk) if FUSE_ON else None
    if bs is None:
        return topk_sigmoid(logits, bias, topk, renormalize, scale)
    values, indices, sorted_ids, expert_ids, ntpp = route(logits, bias, topk, renormalize, scale, bs)
    _parked = (indices.data_ptr(), indices.numel(), bs, sorted_ids, expert_ids, ntpp)
    return values, indices


def maybe_align(topk_ids: torch.Tensor, num_experts: int, block_size: int,
                expert_map: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """(sorted_ids, expert_ids, num_tokens_post_pad) equal to vLLM's moe_align_block_size with
    ignore_invalid_experts and no padding of sorted ids, or None (caller runs vLLM's op)."""
    global _parked
    parked, _parked = _parked, None
    numel = topk_ids.numel()
    if not ALIGN_ON or numel > ALIGN_MAX_NUMEL or expert_map is not None:
        return None
    if (
        topk_ids.dtype != torch.int32
        or not topk_ids.is_contiguous()
        or numel < 1
        or numel > ALIGN_MAX_NUMEL
        or num_experts != NUM_EXPERTS
        or block_size < 1
        or not topk_ids.is_cuda
    ):
        return None
    if not _usable():
        return None
    _align_bs[numel] = block_size
    if parked is not None and parked[0] == topk_ids.data_ptr() and parked[1] == numel and parked[2] == block_size:
        return parked[3], parked[4], parked[5]
    sorted_ids, expert_ids, ntpp = align_buffers(numel, num_experts, block_size, topk_ids.device)
    align(topk_ids, num_experts, block_size, sorted_ids, expert_ids, ntpp)
    return sorted_ids, expert_ids, ntpp

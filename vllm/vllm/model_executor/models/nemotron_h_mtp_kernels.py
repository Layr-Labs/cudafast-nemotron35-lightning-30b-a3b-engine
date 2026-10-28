# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-batch BF16 kernels for the NemotronH MTP drafter.

The MTP head runs its BF16 work at 1 row (draft steps 2..k) or a handful of
rows (the first pass of a cycle). On GB10 cuBLAS picks a GEMV for one row that
reads the weights at about 160 GB/s, and the FlashInfer CUTLASS grouped GEMM
for the head's MoE overlaps with the shared expert on a second stream. Both are
weight-bandwidth bound, so the kernels here read every weight byte once, at the
device's streaming rate:

* ``mtp_small_linear``: a Triton GEMV for one row and a small-M tensor-core
  GEMM for up to 16 rows; larger M falls back to ``F.linear``.
* ``mtp_moe_single_row``: the MTP MoE (router + 6 routed experts + shared
  expert) for ONE input row, broadcast to every output row. It is used only
  when every MoE input row is the same row, which the single-request proposer
  guarantees (``NemotronHMTP.set_moe_row_src``).

These change the drafter's floating-point summation order only. The target
model never calls them. The target decides every emitted token, so the
emitted stream is unchanged; only the draft acceptance can move.
"""

import os

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# l2pf (harness/kernels/l2pf); absent, nothing changes.
try:
    import l2pf as _l2pf
except ImportError:
    _l2pf = None


def mtp_fast_enabled(kind: str) -> bool:
    """Kill switches (default on): MTP_FAST_KERNELS=0 turns both off,
    MTP_FAST_DENSE=0 / MTP_FAST_MOE=0 one of them."""
    if os.environ.get("MTP_FAST_KERNELS", "1") == "0":
        return False
    return os.environ.get(f"MTP_FAST_{kind}", "1") != "0"


_PDL: bool | None = None


def mtp_pdl() -> bool:
    """Programmatic dependent launch for the drafter's Triton kernels (sm_90+):
    each waits for the preceding grid before any memory access, then lets the
    next kernel launch, so a kernel's launch overlaps the one before it. The
    target never runs these. MTP_PDL=0 disables."""
    global _PDL
    if _PDL is None:
        from vllm.platforms import current_platform

        _PDL = os.environ.get(
            "MTP_PDL", "1"
        ) != "0" and current_platform.is_arch_support_pdl()
    return _PDL


# --------------------------------------------------------------------------- #
# Confidence-gated draft depth
# --------------------------------------------------------------------------- #
DEPTH_GATE = os.environ.get("MTP_DEPTH_GATE", "1") != "0"
DEPTH_THETA = float(os.environ.get("MTP_DEPTH_THETA", "0.5"))
# LOCAL PATCH (cudafast-nemotron35, depth 5): the threshold for drafting past the 4th draft
# (kept count >= 4); at depth 4 the loop ends there and it has no effect. MTP_DEPTH_THETA5 sets it.
DEPTH_THETA5 = float(os.environ.get("MTP_DEPTH_THETA5", "0.6"))
DEPTH_CUMULATIVE = os.environ.get("MTP_DEPTH_CUMULATIVE", "1") != "0"
GATE_PROD_ONE = 0x3F800000
_GATE: torch.Tensor | None = None
_GATE_START: torch.Tensor | None = None
# Adaptive deepest draft (depth >= 5): per request, the device counts the cycles that drafted
# num_speculative_tokens drafts (_ADAPT[1]) and those whose deepest draft was accepted (_ADAPT[2]),
# and keeps drafting past the 4th draft only while (accepted + PRIOR) / (drafted + PRIOR) >= TAU
# (_ADAPT[0] = 1.0 allowed, 0.0 not). Reset at each request's first proposal. Default on (MTP_ADAPT5_MODE below); MTP_ADAPT5=0 disables.
ADAPT5 = os.environ.get("MTP_ADAPT5", "1") != "0"
ADAPT_PRIOR = float(os.environ.get("MTP_ADAPT5_PRIOR", "1.0"))
ADAPT_TAU = float(os.environ.get("MTP_ADAPT5_TAU", "0.25"))
# LOCAL PATCH (cudafast-nemotron35): MTP_ADAPT5_MODE selects the rule: "lowacc" (default, see
# _gate_begin_lowacc_kernel) or "deep" (the deepest-draft average above).
ADAPT_MODE = os.environ.get("MTP_ADAPT5_MODE", "lowacc")
ADAPT_LOWACC_THR = float(os.environ.get("MTP_ADAPT5_LOWACC_THR", "1.8"))
ADAPT_LOWACC_THR_OFF = float(os.environ.get("MTP_ADAPT5_LOWACC_THR_OFF", "2.0"))
ADAPT_MIN_CYCLES = int(os.environ.get("MTP_ADAPT5_MIN_CYCLES", "2"))
ADAPT_BURN = float(os.environ.get("MTP_ADAPT5_BURN", "4"))
ADAPT_CAP = int(os.environ.get("MTP_ADAPT5_CAP", "3"))
_ADAPT: torch.Tensor | None = None


def depth_gate_init(device: torch.device) -> torch.Tensor | None:
    """Allocates the gate (at load, before any graph that reads it is captured)
    and asks smallk for its verify-row buffer."""
    global _GATE, _GATE_START, _ADAPT
    if DEPTH_GATE and _GATE is None and device.type == "cuda":
        _GATE = torch.zeros(4, dtype=torch.int32, device=device)
        _GATE_START = torch.tensor(
            [0, 0, 1, GATE_PROD_ONE], dtype=torch.int32, device=device
        )
        if ADAPT5:
            _ADAPT = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        try:
            import smallk

            smallk.request_verify_rows()
        except ImportError:
            pass
    return _GATE


def _gate_args(t: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """(gate, GATED) for a drafter kernel launched on t's device."""
    g = _GATE
    if g is None or g.device != t.device:
        return t, False
    return g, True


# LOCAL PATCH (drafter3): drafter L2 prefetches that a stopped (depth-gated) pass
# would wait for without reading the bytes are skipped on the device instead.
# MTP_PF_GATED=0 keeps the ungated launches.
_PF_GATED = os.environ.get("MTP_PF_GATED", "1") != "0"
_PF_TABLES: dict = {}


def gated_prefetch(ranges, device: torch.device, name: str = "",
                   max_kept: int = 1 << 30) -> bool | None:
    """Queue an L2 prefetch of ranges [(tensor, bytes)] on l2pf's side stream that the
    kernel skips once the depth gate stopped the loop. None when it does not apply
    (no gate, no gated l2pf launch, or a table that would be built during capture):
    the caller keeps its own launch."""
    if (_l2pf is None or not hasattr(_l2pf, "GatedTable")
            or _GATE is None or _GATE.device != device):
        return None
    key = tuple((t.data_ptr(), int(b)) for t, b in ranges if t is not None)
    table = _PF_TABLES.get(key)
    if table is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        table = _l2pf.GatedTable(ranges, device, name=name, mode=2 if _PF_GATED else 1)
        _PF_TABLES[key] = table
    if not table.nranges:
        return None
    return table.launch(_GATE, max_kept)


@triton.jit
def _gate_begin_adapt_kernel(gate_ptr, adapt_ptr, rej_ptr, HAS_REJ: tl.constexpr,
                             NSPEC: tl.constexpr, PRIOR: tl.constexpr, TAU: tl.constexpr,
                             PROD_ONE: tl.constexpr):
    if HAS_REJ:
        kept_prev = tl.load(gate_ptr + 1)
        rej = tl.load(rej_ptr)
        made = tl.load(adapt_ptr + 1)
        acc = tl.load(adapt_ptr + 2)
        deep = kept_prev >= NSPEC
        made = made + deep.to(tl.float32)
        acc = acc + (deep & (rej == 0)).to(tl.float32)
        tl.store(adapt_ptr + 1, made)
        tl.store(adapt_ptr + 2, acc)
        ok = (acc + PRIOR) >= TAU * (made + PRIOR)
        tl.store(adapt_ptr, ok.to(tl.float32))
    else:
        tl.store(adapt_ptr, 1.0)
        tl.store(adapt_ptr + 1, 0.0)
        tl.store(adapt_ptr + 2, 0.0)
    tl.store(gate_ptr, 0)
    tl.store(gate_ptr + 1, 0)
    tl.store(gate_ptr + 2, 1)
    tl.store(gate_ptr + 3, PROD_ONE)


@triton.jit
def _gate_begin_lowacc_kernel(gate_ptr, adapt_ptr, rej_ptr, HAS_REJ: tl.constexpr,
                              THR: tl.constexpr, THR_OFF: tl.constexpr, MIN_CYCLES: tl.constexpr,
                              PROD_ONE: tl.constexpr, BURN: tl.constexpr = 0.0,
                              CAP: tl.constexpr = 4):
    # LOCAL PATCH (cudafast-nemotron35): MTP_ADAPT5_MODE=lowacc. The previous verify's
    # accepted count among the first 4 drafts is accumulated per request (_ADAPT[1] cycles,
    # _ADAPT[2] sum); drafting past the 4th draft turns on once at least MIN_CYCLES cycles were
    # seen and the running mean is below THR, and turns off again when the mean reaches THR_OFF.
    # Starts off; re-evaluated every cycle.
    if HAS_REJ:
        kept_prev = tl.load(gate_ptr + 1)
        rej = tl.load(rej_ptr)
        # capped at CAP so the per-cycle value reads the same whether or not the cycle drafted
        # past the 4th draft (a 4-draft cycle reports at most 3 here)
        acc4 = tl.minimum(tl.maximum(kept_prev - rej, 0), CAP).to(tl.float32)
        cycles = tl.load(adapt_ptr + 1) + 1.0
        # the first BURN cycles are counted but not accumulated (the decision then uses
        # cycles after them only)
        total = tl.load(adapt_ptr + 2) + tl.where(cycles > BURN, acc4, 0.0)
        tl.store(adapt_ptr + 1, cycles)
        tl.store(adapt_ptr + 2, total)
        was_on = tl.load(adapt_ptr) > 0.5
        seen = tl.maximum(cycles - BURN, 0.0)
        ok = (seen >= MIN_CYCLES) & (total < tl.where(was_on, THR_OFF, THR) * seen)
        tl.store(adapt_ptr, ok.to(tl.float32))
    else:
        tl.store(adapt_ptr, 0.0)
        tl.store(adapt_ptr + 1, 0.0)
        tl.store(adapt_ptr + 2, 0.0)
    tl.store(gate_ptr, 0)
    tl.store(gate_ptr + 1, 0)
    tl.store(gate_ptr + 2, 1)
    tl.store(gate_ptr + 3, PROD_ONE)


def adapt_args():
    """(adapt buffer or the gate, ADAPT, TAU) for the screen's _select_token."""
    a = _ADAPT
    if a is None:
        return _GATE, False, 0.0
    # At depth 4 the check after the 4th draft is moot (the loop ends there) and the flag
    # stays 1.0; only depth >= 5 updates it (depth_gate_begin).
    return a, True, 0.5


def depth_gate_begin(batch_size: int, all_greedy: bool,
                     num_rejected: torch.Tensor | None = None,
                     num_spec: int = 0) -> torch.Tensor | None:
    """Start of a draft loop: clear the stop flag and the kept count, and mark the
    loop active (the screen counts and stops only then). Only for one greedy
    request, the case the screen's captured graph serves. At depth >= 5 the deepest
    draft's acceptance average is first updated from the previous verify."""
    g = _GATE
    if (
        g is None
        or batch_size != 1
        or not all_greedy
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    if _ADAPT is not None and num_spec >= 5:
        rej = num_rejected
        has = (rej is not None and rej.is_cuda and rej.dtype == torch.int32
               and rej.numel() >= 1 and rej.device == g.device)
        if ADAPT_MODE == "lowacc":
            _gate_begin_lowacc_kernel[(1,)](
                g, _ADAPT, rej if has else g, HAS_REJ=has, THR=ADAPT_LOWACC_THR,
                THR_OFF=ADAPT_LOWACC_THR_OFF, MIN_CYCLES=ADAPT_MIN_CYCLES, PROD_ONE=GATE_PROD_ONE,
                BURN=ADAPT_BURN, CAP=ADAPT_CAP,
            )
        else:
            _gate_begin_adapt_kernel[(1,)](
                g, _ADAPT, rej if has else g, HAS_REJ=has, NSPEC=num_spec,
                PRIOR=ADAPT_PRIOR, TAU=ADAPT_TAU, PROD_ONE=GATE_PROD_ONE,
            )
        return g
    g.copy_(_GATE_START)
    return g


@triton.jit(do_not_specialize=["max_rows"])
def _depth_gate_end_kernel(gate_ptr, vrows_ptr, max_rows, VROWS_ALL: tl.constexpr,
                           HAS_VROWS: tl.constexpr):
    kept = tl.load(gate_ptr + 1)
    if HAS_VROWS:
        tl.store(vrows_ptr, tl.minimum(tl.where(kept > 0, kept + 1, VROWS_ALL), max_rows))
    tl.store(gate_ptr, 0)
    tl.store(gate_ptr + 2, 0)


_GATE_END_FUSED = os.environ.get("MTP_GATE_END_FUSED", "1") != "0"
_EH_DEMOTE: dict = {}


def _eh_demote(device: torch.device) -> None:
    """LOCAL PATCH (drafter3): the single-row passes read the packed eh hidden half
    evict-last (nemotron_h_mtp_packed.EH_LAST); after the loop, give its L2 lines back
    the normal priority so the verify's own L2 use is unchanged."""
    from .nemotron_h_mtp_packed import EH_BF16, EH_LAST

    if not EH_LAST or _l2pf is None or not hasattr(_l2pf, "demote"):
        return
    key = device.index
    table = _EH_DEMOTE.get(key)
    if table is None:
        if EH_BF16:
            w = getattr(_l2pf, "DRAFT_NEXT", {}).get("eh_bf16")
            ranges = [] if w is None else [(w[0], w[0].numel() * w[0].element_size())]
        else:
            planes = getattr(_l2pf, "DRAFT_NEXT", {}).get("eh")
            ranges = ([] if planes is None else
                      [(t, t.numel() * t.element_size()) for t in planes[:4]])
        if not ranges or torch.cuda.is_current_stream_capturing():
            return
        table = _l2pf.GatedTable(ranges, device, name="demote", mode=1)
        _EH_DEMOTE[key] = table
    _l2pf.demote(table)


def depth_gate_end(gate: torch.Tensor, max_rows: int | None = None) -> None:
    """End of the draft loop: verify rows = kept drafts + 1 (every row when nothing
    was counted), capped at max_rows when given (LOCAL PATCH, cudafast-nemotron35:
    the free-run emit budget, llm_base_proposer._emit_budget_drafts), then the stop
    flag and the active mark cleared."""
    _eh_demote(gate.device)
    try:
        import smallk
    except ImportError:
        smallk = None
    vrows = smallk.live_verify_rows(gate.device) if smallk is not None else None
    if (
        _GATE_END_FUSED and gate.is_cuda and gate.dtype == torch.int32
        and gate.numel() == 4 and gate.is_contiguous()
        and (vrows is None or (
            vrows.is_cuda and vrows.device == gate.device
            and vrows.dtype == torch.int32 and vrows.numel() == 1
            and vrows.is_contiguous()
        ))
    ):
        _depth_gate_end_kernel[(1,)](
            gate, gate if vrows is None else vrows,
            (1 << 30) if max_rows is None else int(max_rows),
            VROWS_ALL=0 if vrows is None else smallk.VROWS_ALL,
            HAS_VROWS=vrows is not None,
        )
        return
    if vrows is not None:
        kept = gate[1:2]
        rows = torch.where(kept > 0, kept + 1, torch.full_like(kept, smallk.VROWS_ALL))
        if max_rows is not None:
            rows = torch.clamp(rows, max=int(max_rows))
        vrows.copy_(rows)
    gate[0:1].zero_()
    gate[2:3].zero_()


# --------------------------------------------------------------------------- #
# Dense GEMV / small-M GEMM
# --------------------------------------------------------------------------- #

# Measured on GB10 (5 MTP shapes, weights not L2 resident):
#   M=1: BN=8, BK=512, 4 warps, 3 stages -> 225-230 GB/s (cuBLAS gemvx ~160).
#   M<=16: tensor-core tile BN=64, BK=128, 4 stages -> ~226 GB/s (cuBLAS ~221).
_GEMV_BN, _GEMV_BK, _GEMV_WARPS, _GEMV_STAGES = 8, 512, 4, 3
_DOT_BN, _DOT_BK, _DOT_WARPS, _DOT_STAGES = 64, 128, 4, 4
_DOT_MAX_M = 16
# Weight streams are read once per forward: hint L2 to evict them first.
_EVICT_W = os.environ.get("MTP_FAST_EVICT", "1") != "0"


@triton.jit
def _load_w(ptrs, mask, EVICT: tl.constexpr):
    # LOCAL PATCH (drafter3): EVICT 2 loads evict-last (the BF16 eh hidden half every
    # draft pass of a cycle reads); 1 / True evict-first; 0 / False default.
    if EVICT == 2:
        w = tl.load(ptrs, mask=mask, other=0.0, eviction_policy="evict_last")
    elif EVICT:
        w = tl.load(ptrs, mask=mask, other=0.0, eviction_policy="evict_first")
    else:
        w = tl.load(ptrs, mask=mask, other=0.0)
    return w


@triton.jit
def _mtp_gemv_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    N,
    K,
    stride_wn,
    BN: tl.constexpr,
    BK: tl.constexpr,
    EVICT: tl.constexpr,
    gate_ptr=None,
    GATED: tl.constexpr = False,
    launch_pdl: tl.constexpr = False,
):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    # y[n] = sum_k w[n, k] * x[k]   (one row, fp32 accumulation)
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    nmask = offs_n < N
    # weight-load mask: off once the depth gate stopped the draft loop (zeros out)
    wn = nmask
    if GATED:
        wn = nmask & (tl.load(gate_ptr) == 0)
    acc = tl.zeros((BN, BK), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + offs_k
        kmask = kk < K
        w = _load_w(
            w_ptr + offs_n[:, None] * stride_wn + kk[None, :],
            wn[:, None] & kmask[None, :],
            EVICT,
        )
        x = tl.load(x_ptr + kk, mask=kmask, other=0.0)
        acc += w.to(tl.float32) * x[None, :].to(tl.float32)
    y = tl.sum(acc, axis=1)
    tl.store(y_ptr + offs_n, y.to(y_ptr.dtype.element_ty), mask=nmask)


@triton.jit
def _mtp_small_gemm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_ym,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    EVICT: tl.constexpr,
    gate_ptr=None,
    GATED: tl.constexpr = False,
    launch_pdl: tl.constexpr = False,
):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    # y[m, n] = sum_k x[m, k] * w[n, k]   (M <= BM rows, tensor cores)
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    offs_m = tl.arange(0, BM)
    nmask = offs_n < N
    # weight-load mask: off once the depth gate stopped the draft loop (zeros out)
    wn = nmask
    if GATED:
        wn = nmask & (tl.load(gate_ptr) == 0)
    mmask = offs_m < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + offs_k
        kmask = kk < K
        w = _load_w(
            w_ptr + offs_n[:, None] * stride_wn + kk[None, :],
            wn[:, None] & kmask[None, :],
            EVICT,
        )
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + kk[None, :],
            mask=mmask[:, None] & kmask[None, :],
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(w))
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :],
        acc.to(y_ptr.dtype.element_ty),
        mask=mmask[:, None] & nmask[None, :],
    )


def _small_linear_supported(x: torch.Tensor, w: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and x.dim() == 2
        and w.dim() == 2
        and w.stride(1) == 1
        and x.shape[1] == w.shape[1]
    )


def mtp_small_linear_impl(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return _small_linear(x, w, _EVICT_W)


def mtp_small_linear_keep_impl(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """mtp_small_linear with ordinary (not evict-first) weight loads, for a
    weight every draft pass of a cycle reads again: the drafter's other weight
    streams are evict-first, so this one can stay in the 24 MB L2 between
    passes. Same arithmetic as mtp_small_linear. LOCAL PATCH (drafter3): with
    MTP_EH_LAST (default) the loads are evict-last; depth_gate_end demotes."""
    from .nemotron_h_mtp_packed import EH_LAST
    return _small_linear(x, w, 2 if EH_LAST else False)


def _small_linear(x: torch.Tensor, w: torch.Tensor, evict: bool) -> torch.Tensor:
    M = x.shape[0]
    if M == 0 or M > _DOT_MAX_M or not _small_linear_supported(x, w):
        return F.linear(x, w)
    if x.stride(1) != 1:
        x = x.contiguous()
    N, K = w.shape
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    gate, gated = _gate_args(x)
    if M == 1:
        _mtp_gemv_kernel[(triton.cdiv(N, _GEMV_BN),)](
            x,
            w,
            y,
            N,
            K,
            w.stride(0),
            BN=_GEMV_BN,
            EVICT=evict,
            BK=_GEMV_BK,
            gate_ptr=gate,
            GATED=gated,
            num_warps=_GEMV_WARPS,
            num_stages=_GEMV_STAGES,
            launch_pdl=mtp_pdl(),
        )
    else:
        _mtp_small_gemm_kernel[(triton.cdiv(N, _DOT_BN),)](
            x,
            w,
            y,
            M,
            N,
            K,
            x.stride(0),
            w.stride(0),
            y.stride(0),
            BM=16,
            BN=_DOT_BN,
            EVICT=evict,
            BK=_DOT_BK,
            gate_ptr=gate,
            GATED=gated,
            num_warps=_DOT_WARPS,
            num_stages=_DOT_STAGES,
            launch_pdl=mtp_pdl(),
        )
    return y


def _mtp_small_linear_fake(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], w.shape[0]))


direct_register_custom_op(
    op_name="mtp_small_linear",
    op_func=mtp_small_linear_impl,
    fake_impl=_mtp_small_linear_fake,
)

direct_register_custom_op(
    op_name="mtp_small_linear_keep",
    op_func=mtp_small_linear_keep_impl,
    fake_impl=_mtp_small_linear_fake,
)


def install_mtp_small_linear(module: torch.nn.Module) -> int:
    """Route the BF16 unquantized linears under `module` through
    mtp_small_linear. Returns the number of layers switched."""
    from vllm.model_executor.layers.linear import (
        LinearBase,
        UnquantizedLinearMethod,
    )

    class MTPSmallLinearMethod(UnquantizedLinearMethod):
        def apply(self, layer, x, bias=None):
            if hasattr(layer, "_mtp_pk_sm"):
                return torch.ops.vllm.mtp_packed_linear(
                    x, layer._mtp_pk_sm, layer._mtp_pk_ex,
                    layer._mtp_pk_base, layer._mtp_pk_ri, layer._mtp_pk_raw,
                    layer._mtp_pk_n, layer._mtp_pk_k, layer.weight,
                )
            if bias is None and x.dim() == 2:
                return torch.ops.vllm.mtp_small_linear(x, layer.weight)
            return super().apply(layer, x, bias)

    n = 0
    for name, m in module.named_modules():
        if not isinstance(m, LinearBase):
            continue
        if type(getattr(m, "quant_method", None)) is not UnquantizedLinearMethod:
            continue
        w = getattr(m, "weight", None)
        if w is None or w.dtype != torch.bfloat16 or getattr(m, "bias", None) is not None:
            continue
        m.quant_method = MTPSmallLinearMethod()
        n += 1
    logger.info_once("MTP small-M BF16 linear kernels on %d layers", n)
    return n


# --------------------------------------------------------------------------- #
# MoE for one row: 6 routed experts + shared expert (relu^2, non-gated)
# --------------------------------------------------------------------------- #

# Measured on GB10: 8 rows x 1024 columns per step, 8 warps -> ~240 GB/s
# for both kernels (the device's streaming read rate).
_MOE_UP_BN, _MOE_UP_BK, _MOE_UP_WARPS, _MOE_UP_STAGES = 8, 1024, 8, 3
# Packed (12-bit) up projection: 2 rows per program, 2 warps along N and the same
# 4 along K (nemotron_h_mtp_packed.packed_moe_up); each row's arithmetic is unchanged.
_PK_UP_BN, _PK_UP_WN = 2, 2
_MOE_DN_BN, _MOE_DN_BK, _MOE_DN_WARPS, _MOE_DN_STAGES = 8, 1024, 8, 3



@triton.jit
def _mtp_moe_up_kernel(
    x_ptr,  # [K] bf16
    w13_ptr,  # [E, I, K] bf16
    ws_ptr,  # [S*I, K] bf16 (shared up, S = shared_inter / I slices)
    ids_ptr,  # [TOPK] int
    h_ptr,  # [TOPK + S, I] bf16
    I,
    K,
    stride_we,
    stride_wn,
    stride_wsn,
    TOPK: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    EVICT: tl.constexpr,
    gate_ptr=None,
    GATED: tl.constexpr = False,
    launch_pdl: tl.constexpr = False,
):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    # slot j < TOPK: routed expert ids[j]; slot j >= TOPK: shared slice j-TOPK
    pid_n = tl.program_id(0)
    j = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    nmask = offs_n < I
    # weight-load mask: off once the depth gate stopped the draft loop (zeros out)
    wn = nmask
    if GATED:
        wn = nmask & (tl.load(gate_ptr) == 0)
    if j < TOPK:
        e = tl.load(ids_ptr + j).to(tl.int64)
        wbase = w13_ptr + e * stride_we + offs_n[:, None].to(tl.int64) * stride_wn
    else:
        rows = (j - TOPK) * I + offs_n
        wbase = ws_ptr + rows[:, None].to(tl.int64) * stride_wsn
    acc = tl.zeros((BN, BK), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + offs_k
        kmask = kk < K
        w = _load_w(wbase + kk[None, :], wn[:, None] & kmask[None, :], EVICT)
        x = tl.load(x_ptr + kk, mask=kmask, other=0.0)
        acc += w.to(tl.float32) * x[None, :].to(tl.float32)
    h = tl.maximum(tl.sum(acc, axis=1), 0.0)
    h = h * h
    tl.store(h_ptr + j * I + offs_n, h.to(h_ptr.dtype.element_ty), mask=nmask)


@triton.jit
def _mtp_moe_down_kernel(
    h_ptr,  # [TOPK + S, I] bf16
    w2_ptr,  # [E, H, I] bf16
    wsd_ptr,  # [H, S*I] bf16 (shared down)
    ids_ptr,  # [TOPK] int
    tw_ptr,  # [TOPK] fp32 routing weights
    y_ptr,  # [M, H] out
    M,
    H,
    I,
    stride_we,
    stride_wn,
    stride_wsn,
    stride_ym,
    routed_scale,
    TOPK: tl.constexpr,
    S: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    EVICT: tl.constexpr,
    gate_ptr=None,
    GATED: tl.constexpr = False,
    launch_pdl: tl.constexpr = False,
):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    # y[n] = routed_scale * sum_j tw[j] * <w2[ids[j], n, :], h[j]>
    #        + sum_s <wsd[n, s*I:(s+1)*I], h[TOPK + s]>
    pid_n = tl.program_id(0)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    nmask = offs_n < H
    # weight-load mask: off once the depth gate stopped the draft loop (zeros out)
    wn = nmask
    if GATED:
        wn = nmask & (tl.load(gate_ptr) == 0)
    routed = tl.zeros((BN,), dtype=tl.float32)
    for j in range(TOPK):
        e = tl.load(ids_ptr + j).to(tl.int64)
        wbase = w2_ptr + e * stride_we + offs_n[:, None].to(tl.int64) * stride_wn
        acc = tl.zeros((BN, BK), dtype=tl.float32)
        for k0 in range(0, I, BK):
            kk = k0 + offs_k
            kmask = kk < I
            w = _load_w(wbase + kk[None, :], wn[:, None] & kmask[None, :], EVICT)
            h = tl.load(h_ptr + j * I + kk, mask=kmask, other=0.0)
            acc += w.to(tl.float32) * h[None, :].to(tl.float32)
        routed += tl.load(tw_ptr + j).to(tl.float32) * tl.sum(acc, axis=1)
    shared = tl.zeros((BN,), dtype=tl.float32)
    sbase = wsd_ptr + offs_n[:, None].to(tl.int64) * stride_wsn
    for s in range(S):
        acc = tl.zeros((BN, BK), dtype=tl.float32)
        for k0 in range(0, I, BK):
            kk = k0 + offs_k
            kmask = kk < I
            w = _load_w(
                sbase + s * I + kk[None, :],
                wn[:, None] & kmask[None, :],
                EVICT,
            )
            h = tl.load(h_ptr + (TOPK + s) * I + kk, mask=kmask, other=0.0)
            acc += w.to(tl.float32) * h[None, :].to(tl.float32)
        shared += tl.sum(acc, axis=1)
    out = (routed * routed_scale + shared).to(y_ptr.dtype.element_ty)
    for m in range(M):
        tl.store(y_ptr + m * stride_ym + offs_n, out, mask=nmask)


# Exact-zero loads retain the dense 1024-lane accumulator and expert order.
@triton.jit
def _mtp_moe_zero_down_kernel(
    h_ptr,  # [TOPK + S, I] bf16
    w2_ptr,  # [E, H, I] bf16
    wsd_ptr,  # [H, S*I] bf16 (shared down)
    ids_ptr,  # [TOPK] int
    tw_ptr,  # [TOPK] fp32 routing weights
    y_ptr,  # [M, H] out
    M,
    H,
    I,
    stride_we,
    stride_wn,
    stride_wsn,
    stride_wk,
    stride_wsk,
    stride_wss,
    stride_ym,
    routed_scale,
    TOPK: tl.constexpr,
    S: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    EVICT: tl.constexpr,
    launch_pdl: tl.constexpr = False,
):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    # y[n] = routed_scale * sum_j tw[j] * <w2[ids[j], n, :], h[j]>
    #        + sum_s <wsd[n, s*I:(s+1)*I], h[TOPK + s]>
    pid_n = tl.program_id(0)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    nmask = offs_n < H
    routed = tl.zeros((BN,), dtype=tl.float32)
    for j in range(TOPK):
        e = tl.load(ids_ptr + j).to(tl.int64)
        wbase = w2_ptr + e * stride_we + offs_n[:, None].to(tl.int64) * stride_wn
        acc = tl.zeros((BN, BK), dtype=tl.float32)
        for k0 in range(0, I, BK):
            kk = k0 + offs_k
            kmask = kk < I
            h = tl.load(h_ptr + j * I + kk, mask=kmask, other=0.0)
            w = _load_w(wbase + kk[None, :] * stride_wk, nmask[:, None] & kmask[None, :] & (h[None, :] != 0), EVICT)
            acc += w.to(tl.float32) * h[None, :].to(tl.float32)
        routed += tl.load(tw_ptr + j).to(tl.float32) * tl.sum(acc, axis=1)
    shared = tl.zeros((BN,), dtype=tl.float32)
    sbase = wsd_ptr + offs_n[:, None].to(tl.int64) * stride_wsn
    for s in range(S):
        acc = tl.zeros((BN, BK), dtype=tl.float32)
        for k0 in range(0, I, BK):
            kk = k0 + offs_k
            kmask = kk < I
            h = tl.load(h_ptr + (TOPK + s) * I + kk, mask=kmask, other=0.0)
            w = _load_w(
                sbase + s * stride_wss + kk[None, :] * stride_wsk,
                nmask[:, None] & kmask[None, :] & (h[None, :] != 0),
                EVICT,
            )
            acc += w.to(tl.float32) * h[None, :].to(tl.float32)
        shared += tl.sum(acc, axis=1)
    out = (routed * routed_scale + shared).to(y_ptr.dtype.element_ty)
    for m in range(M):
        tl.store(y_ptr + m * stride_ym + offs_n, out, mask=nmask)


# Default on; MTP_RELU2=0 selects the unchanged l2 down kernel at boot.
# A screening process can retain separate CUDA graph dictionaries per mode.
_RELU2_MODE = 0 if os.environ.get("MTP_RELU2", "1") == "0" else 2
_RELU2_BN = 32
_RELU2_WARPS = 8
_RELU2_PACKED = {}


def relu2_prepare(w2, sd, inter, n_shared):
    """Input-independent BF16 transpose. Called before timed work/capture."""
    key = (w2.data_ptr(), sd.data_ptr())
    if key not in _RELU2_PACKED:
        # A zero times an infinite/NaN weight cannot be omitted. Check once
        # after loading, never on the timed path or inside graph capture.
        if not bool(torch.isfinite(w2).all() & torch.isfinite(sd).all()):
            _RELU2_PACKED[key] = False
            return False
        # No arithmetic, quantization, thresholding, or modification of originals.
        wp = w2.transpose(1, 2).contiguous()
        sp = sd.view(sd.shape[0], n_shared, inter).permute(1, 2, 0).contiguous()
        _RELU2_PACKED[key] = (wp, sp)
    return _RELU2_PACKED[key]


def relu2_down(h, w2, sd, ids, tw, y, routed_scale, mode=None, bn=None, warps=None, residual=None):
    mode = _RELU2_MODE if mode is None else mode
    bn = _RELU2_BN if bn is None else bn
    warps = _RELU2_WARPS if warps is None else warps
    topk = ids.numel()
    inter = h.shape[1]
    ns = h.shape[0] - topk
    H = y.shape[1]
    pf = False
    if mode == 2:
        packed = relu2_prepare(w2, sd, inter, ns)
        if packed is False:
            _mtp_moe_down_kernel[(triton.cdiv(H, _MOE_DN_BN),)](
                h, w2, sd, ids, tw, y, y.shape[0], H, inter,
                w2.stride(0), w2.stride(1), sd.stride(0), y.stride(0), routed_scale,
                TOPK=topk, S=ns, BN=_MOE_DN_BN, BK=_MOE_DN_BK,
                EVICT=_EVICT_W, num_warps=_MOE_DN_WARPS,
                num_stages=_MOE_DN_STAGES,
                launch_pdl=mtp_pdl(),
            )
            return False
        wp, sp = packed
        from .nemotron_h_mtp_compact import maybe_down, can_add_residual

        # LOCAL PATCH (cudafast-nemotron35): the down projection over the
        # transposed weights (compact or zero-load) is latency-bound; prefetch
        # the head of the draft screen's coarse matrix, which the drafter
        # streams next, into L2 meanwhile (moves no value). Joined after
        # whichever down kernel runs.
        pf = False
        if _l2pf is not None and _l2pf.DRAFT_BYTES > 0 and bool(_l2pf.DRAFT_SCREEN):
            pf = gated_prefetch([(_l2pf.DRAFT_SCREEN[0], _l2pf.DRAFT_BYTES)], y.device,
                                name="shead")
            if pf is None:
                pf = _l2pf.prefetch(_l2pf.DRAFT_SCREEN[0], max_bytes=_l2pf.DRAFT_BYTES)
        add = can_add_residual(residual, y, routed_scale, ids.numel())
        if maybe_down(h, packed, ids, tw, y, routed_scale, residual=residual if add else None):
            if pf:
                _l2pf.join(y.device)
            return add
        strides = (wp.stride(0), 1, 1, H, H, inter * H)
    else:
        wp, sp = w2, sd
        strides = (w2.stride(0), w2.stride(1), sd.stride(0), 1, 1, inter)
    _mtp_moe_zero_down_kernel[(triton.cdiv(H, bn),)](
        h, wp, sp, ids, tw, y, y.shape[0], H, inter,
        *strides, y.stride(0), routed_scale,
        TOPK=topk, S=ns, BN=bn, BK=1024, EVICT=_EVICT_W,
        num_warps=warps, num_stages=3,
        launch_pdl=mtp_pdl(),
    )
    if pf:
        _l2pf.join(y.device)

    return False

_MOE_LAYERS: dict[str, torch.nn.Module] = {}


def register_mtp_moe_layer(name: str, layer: torch.nn.Module) -> None:
    _MOE_LAYERS[name] = layer


def mtp_moe_weights(layer: torch.nn.Module):
    """(w13 [E,I,H], w2 [E,H,I], shared_up [S*I,H], shared_down [H,S*I], I, S)
    of an NemotronHMTPMoEDecoderLayer, or None when the layout is not the
    plain BF16 one these kernels expect."""
    mixer = layer.mixer
    runner = mixer.experts
    experts = getattr(runner, "routed_experts", None)
    shared = mixer.shared_experts
    if experts is None or shared is None:
        return None
    w13 = getattr(experts, "w13_weight", None)
    w2 = getattr(experts, "w2_weight", None)
    su = getattr(shared.up_proj, "weight", None)
    sd = getattr(shared.down_proj, "weight", None)
    if any(t is None for t in (w13, w2, su, sd)):
        return None
    if any(t.dtype != torch.bfloat16 for t in (w13, w2, su, sd)):
        return None
    if w13.dim() != 3 or w2.dim() != 3:
        return None
    E, I, H = w13.shape
    if tuple(w2.shape) != (E, H, I) or su.shape[1] != H or sd.shape[0] != H:
        return None
    if su.shape[0] % I != 0 or sd.shape[1] != su.shape[0]:
        return None
    if not (
        w13.is_contiguous()
        and w2.is_contiguous()
        and su.stride(1) == 1
        and sd.stride(1) == 1
    ):
        return None
    return w13, w2, su, sd, I, su.shape[0] // I



@torch.no_grad()
def prepare_mtp_relu2(module: torch.nn.Module) -> None:
    """Value-preserving MTP-only down-weight repack, after weight processing."""
    if not _RELU2_MODE:
        return
    for layer in module.modules():
        if not getattr(layer, "_fast_moe", False):
            continue
        weights = mtp_moe_weights(layer)
        if weights is None:
            continue
        layer._fast_moe_weights = weights
        _, w2, _, sd, inter, n_shared = weights
        packed = relu2_prepare(w2, sd, inter, n_shared)
        layer._mtp_fold_residual_ok = False
        if packed is not False and _RELU2_MODE == 2:
            from .nemotron_h_mtp_compact import prepare_residual_add
            layer._mtp_fold_residual_ok = prepare_residual_add(
                w2.device, float(layer.mixer.routed_scaling_factor))
            from .nemotron_h_mtp_down_packed import prepare as prepare_down_packed
            prepare_down_packed(packed)
        logger.info_once(
            "MTP exact-zero down: %s",
            "BF16 transpose ready" if packed is not False else "l2 fallback (nonfinite weights)",
        )


def mtp_moe_row(
    x: torch.Tensor,
    ids: torch.Tensor,
    tw: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    su: torch.Tensor,
    sd: torch.Tensor,
    n_shared: int,
    routed_scale: float,
    M: int,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """routed_scale * sum_j tw[j] * W2[ids[j]] relu2(W13[ids[j]] x)
    + SD relu2(SU x) for one row x [1, H], written to M output rows."""
    topk = ids.numel()
    E, inter, H = w13.shape
    h = torch.empty((topk + n_shared, inter), dtype=x.dtype, device=x.device)
    from .nemotron_h_mtp_packed import PackedBF16, packed_moe_up, word_load_enabled

    gate, gated = _gate_args(x)
    if isinstance(w13, PackedBF16):
        packed_moe_up[(triton.cdiv(inter, _PK_UP_BN), topk + n_shared)](
            x, *w13.planes, *su.planes, ids, h, inter, H,
            inter * H, H, H, gate, TOPK=topk, BN=_PK_UP_BN, BK=_MOE_UP_BK,
            EVICT=_EVICT_W, GATED=gated, num_warps=4 * _PK_UP_WN,
            num_stages=_MOE_UP_STAGES, launch_pdl=mtp_pdl(), WN=_PK_UP_WN,
            WORD=word_load_enabled(x.device),
        )
    else:
        _mtp_moe_up_kernel[(triton.cdiv(inter, _MOE_UP_BN), topk + n_shared)](
            x,
            w13,
            su,
            ids,
            h,
            inter,
            H,
            w13.stride(0),
            w13.stride(1),
            su.stride(0),
            TOPK=topk,
            BN=_MOE_UP_BN,
            EVICT=_EVICT_W,
            BK=_MOE_UP_BK,
            gate_ptr=gate,
            GATED=gated,
            num_warps=_MOE_UP_WARPS,
            num_stages=_MOE_UP_STAGES,
            launch_pdl=mtp_pdl(),
        )
    y = torch.empty((M, H), dtype=x.dtype, device=x.device)
    if _RELU2_MODE:
        added = relu2_down(h, w2, sd, ids, tw, y, routed_scale, residual=residual)
        return y if residual is None or added else y + residual
    _mtp_moe_down_kernel[(triton.cdiv(H, _MOE_DN_BN),)](
        h,
        w2,
        sd,
        ids,
        tw,
        y,
        M,
        H,
        inter,
        w2.stride(0),
        w2.stride(1),
        sd.stride(0),
        y.stride(0),
        routed_scale,
        TOPK=topk,
        S=n_shared,
        BN=_MOE_DN_BN,
        EVICT=_EVICT_W,
        BK=_MOE_DN_BK,
        gate_ptr=gate,
        GATED=gated,
        num_warps=_MOE_DN_WARPS,
        num_stages=_MOE_DN_STAGES,
        launch_pdl=mtp_pdl(),
    )
    return y if residual is None else y + residual


# LOCAL PATCH (cudafast-nemotron35): routed experts the MTP draft evaluates per
# token (the router's top 6 by weight, highest first); MTP_DRAFT_TOPK=0 keeps all.
_DRAFT_TOPK = int(os.environ.get("MTP_DRAFT_TOPK", "3"))
# MTP_TOPK_KEEP_KERNEL=0 keeps torch.sort + slice + index_select for the selection.
_TOPK_KEEP_KERNEL = os.environ.get("MTP_TOPK_KEEP_KERNEL", "1") != "0"


@triton.jit
def _topk_keep_kernel(
    tw_ptr, ids_ptr, otw_ptr, oids_ptr,
    K: tl.constexpr, BK: tl.constexpr, KEEP: tl.constexpr,
    launch_pdl: tl.constexpr = False,
):
    """The KEEP highest of K router weights, highest first, with their expert ids: each
    element's rank is the number of weights above it plus the equal ones at a lower
    slot (a stable descending sort), and ranks below KEEP are scattered to their place."""
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    i = tl.arange(0, BK)
    valid = i < K
    w = tl.load(tw_ptr + i, mask=valid, other=-float("inf"))
    e = tl.load(ids_ptr + i, mask=valid, other=0)
    # NaN ranks as +inf (torch.sort puts NaN first in descending order), so the
    # ranks stay a permutation and every output slot is written.
    key = tl.where(w != w, float("inf"), w)
    wi = key[:, None]
    wj = key[None, :]
    jv = valid[None, :]
    above = jv & ((wj > wi) | ((wj == wi) & (i[None, :] < i[:, None])))
    rank = tl.sum(above.to(tl.int32), axis=1)
    keep = valid & (rank < KEEP)
    tl.store(otw_ptr + rank, w, mask=keep)
    tl.store(oids_ptr + rank, e, mask=keep)



def _mtp_moe_single_row_dispatch(
    hidden_states: torch.Tensor, row_src: torch.Tensor, layer_name: str,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """MoE of the MTP layer for the single row hidden_states[row_src[0]],
    written to every one of the hidden_states.shape[0] output rows."""
    layer = _MOE_LAYERS[layer_name]
    mixer = layer.mixer
    if layer._fast_moe_weights is None:
        layer._fast_moe_weights = mtp_moe_weights(layer) or False
        logger.info_once(
            "MTP single-row MoE kernels: %s",
            "on" if layer._fast_moe_weights else "off (unsupported weight layout)",
        )
    if not layer._fast_moe_weights:
        out = mixer(hidden_states.index_select(0, row_src))
        return out if residual is None else out + residual
    w13, w2, su, sd, inter, n_shared = layer._fast_moe_weights
    M, H = hidden_states.shape
    x = hidden_states if M == 1 else hidden_states.index_select(0, row_src[:1])  # [1, H]
    shared_pf = _shared_prefetch(layer, su, x)
    router_logits, _ = mixer.gate(x)
    topk_weights, topk_ids = mixer.experts.router.select_experts(
        hidden_states=x, router_logits=router_logits
    )
    ids = topk_ids.reshape(-1)
    tw = topk_weights.reshape(-1).to(torch.float32)
    if 0 < _DRAFT_TOPK < ids.numel():
        # LOCAL PATCH (cudafast-nemotron35): the draft keeps the _DRAFT_TOPK
        # highest-weight routed experts of the MTP layer (drafter-only).
        if _TOPK_KEEP_KERNEL and tw.is_cuda and ids.numel() <= 16:
            tw_k = torch.empty(_DRAFT_TOPK, dtype=tw.dtype, device=tw.device)
            ids_k = torch.empty(_DRAFT_TOPK, dtype=ids.dtype, device=ids.device)
            _topk_keep_kernel[(1,)](
                tw.contiguous(), ids.contiguous(), tw_k, ids_k,
                K=ids.numel(), BK=triton.next_power_of_2(ids.numel()), KEEP=_DRAFT_TOPK,
                num_warps=1, launch_pdl=mtp_pdl(),
            )
            tw, ids = tw_k, ids_k
        else:
            tw, order = torch.sort(tw, descending=True)
            tw = tw[:_DRAFT_TOPK].contiguous()
            ids = ids.index_select(0, order[:_DRAFT_TOPK]).contiguous()
    out = mtp_moe_row(
        x,
        ids,
        tw,
        w13,
        w2,
        su,
        sd,
        n_shared,
        float(mixer.routed_scaling_factor),
        M,
        residual=residual,
    )
    if shared_pf:
        _l2pf.join(x.device)
    return out


def _shared_prefetch(layer, su, x) -> bool:
    """LOCAL PATCH (drafter3): the router chain (gate GEMV, top-k, keep) is
    latency-bound; prefetch the head of the packed shared-up planes, which the up
    kernel reads in its last slots, into L2 meanwhile (moves no value). Skipped on
    the device once the depth gate stopped the loop. L2PF_DRAFT_SHARED_MB=0 off."""
    if (_l2pf is None or getattr(_l2pf, "DRAFT_SHARED_BYTES", 0) <= 0
            or not hasattr(_l2pf, "GatedTable") or not x.is_cuda):
        return False
    from .nemotron_h_mtp_packed import PackedBF16

    if not isinstance(su, PackedBF16):
        return False
    table = getattr(layer, "_mtp_shared_pf", None)
    if table is None:
        if torch.cuda.is_current_stream_capturing():
            return False  # the table is built on an eager (warm-up) call
        table = _l2pf.GatedTable(_l2pf.packed_ranges(su.planes, _l2pf.DRAFT_SHARED_BYTES),
                                 x.device, name="shared")
        layer._mtp_shared_pf = table
    gate, gated = _gate_args(x)
    return table.launch(gate if gated else None, 1 << 30)



def mtp_moe_single_row_impl(hidden_states: torch.Tensor, row_src: torch.Tensor,
                            layer_name: str) -> torch.Tensor:
    return _mtp_moe_single_row_dispatch(hidden_states, row_src, layer_name)


def _mtp_moe_single_row_add_impl(hidden_states: torch.Tensor, row_src: torch.Tensor,
                                 residual: torch.Tensor, layer_name: str) -> torch.Tensor:
    return _mtp_moe_single_row_dispatch(hidden_states, row_src, layer_name, residual)


def _mtp_moe_single_row_add_fake(hidden_states: torch.Tensor, row_src: torch.Tensor,
                                 residual: torch.Tensor, layer_name: str) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="mtp_moe_single_row_add_residual", op_func=_mtp_moe_single_row_add_impl,
    fake_impl=_mtp_moe_single_row_add_fake,
)


def _mtp_moe_single_row_fake(
    hidden_states: torch.Tensor, row_src: torch.Tensor, layer_name: str
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="mtp_moe_single_row",
    op_func=mtp_moe_single_row_impl,
    fake_impl=_mtp_moe_single_row_fake,
)

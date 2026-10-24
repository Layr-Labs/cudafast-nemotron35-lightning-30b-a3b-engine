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

* ``draft_nvfp4_logits``: the shared NVFP4 lm_head at one row. The target
  still runs CUTLASS W4A4. The drafter reads a plain (unswizzled) copy of
  the same packed weights and does a bf16×NVFP4 GEMV.

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


def mtp_fast_enabled(kind: str) -> bool:
    """Kill switches (default on): MTP_FAST_KERNELS=0 turns both off,
    MTP_FAST_DENSE=0 / MTP_FAST_MOE=0 one of them."""
    if os.environ.get("MTP_FAST_KERNELS", "1") == "0":
        return False
    return os.environ.get(f"MTP_FAST_{kind}", "1") != "0"

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
    if EVICT:
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
):
    # y[n] = sum_k w[n, k] * x[k]   (one row, fp32 accumulation)
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    nmask = offs_n < N
    acc = tl.zeros((BN, BK), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + offs_k
        kmask = kk < K
        w = _load_w(
            w_ptr + offs_n[:, None] * stride_wn + kk[None, :],
            nmask[:, None] & kmask[None, :],
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
):
    # y[m, n] = sum_k x[m, k] * w[n, k]   (M <= BM rows, tensor cores)
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    offs_m = tl.arange(0, BM)
    nmask = offs_n < N
    mmask = offs_m < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + offs_k
        kmask = kk < K
        w = _load_w(
            w_ptr + offs_n[:, None] * stride_wn + kk[None, :],
            nmask[:, None] & kmask[None, :],
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
    M = x.shape[0]
    if M == 0 or M > _DOT_MAX_M or not _small_linear_supported(x, w):
        return F.linear(x, w)
    if x.stride(1) != 1:
        x = x.contiguous()
    N, K = w.shape
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    if M == 1:
        _mtp_gemv_kernel[(triton.cdiv(N, _GEMV_BN),)](
            x,
            w,
            y,
            N,
            K,
            w.stride(0),
            BN=_GEMV_BN,
            EVICT=_EVICT_W,
            BK=_GEMV_BK,
            num_warps=_GEMV_WARPS,
            num_stages=_GEMV_STAGES,
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
            EVICT=_EVICT_W,
            BK=_DOT_BK,
            num_warps=_DOT_WARPS,
            num_stages=_DOT_STAGES,
        )
    return y


def _mtp_small_linear_fake(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], w.shape[0]))


direct_register_custom_op(
    op_name="mtp_small_linear",
    op_func=mtp_small_linear_impl,
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
):
    # slot j < TOPK: routed expert ids[j]; slot j >= TOPK: shared slice j-TOPK
    pid_n = tl.program_id(0)
    j = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    nmask = offs_n < I
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
        w = _load_w(wbase + kk[None, :], nmask[:, None] & kmask[None, :], EVICT)
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
):
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
            w = _load_w(wbase + kk[None, :], nmask[:, None] & kmask[None, :], EVICT)
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
                nmask[:, None] & kmask[None, :],
                EVICT,
            )
            h = tl.load(h_ptr + (TOPK + s) * I + kk, mask=kmask, other=0.0)
            acc += w.to(tl.float32) * h[None, :].to(tl.float32)
        shared += tl.sum(acc, axis=1)
    out = (routed * routed_scale + shared).to(y_ptr.dtype.element_ty)
    for m in range(M):
        tl.store(y_ptr + m * stride_ym + offs_n, out, mask=nmask)


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
) -> torch.Tensor:
    """routed_scale * sum_j tw[j] * W2[ids[j]] relu2(W13[ids[j]] x)
    + SD relu2(SU x) for one row x [1, H], written to M output rows."""
    topk = ids.numel()
    E, inter, H = w13.shape
    h = torch.empty((topk + n_shared, inter), dtype=x.dtype, device=x.device)
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
        num_warps=_MOE_UP_WARPS,
        num_stages=_MOE_UP_STAGES,
    )
    y = torch.empty((M, H), dtype=x.dtype, device=x.device)
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
        num_warps=_MOE_DN_WARPS,
        num_stages=_MOE_DN_STAGES,
    )
    return y


def mtp_moe_single_row_impl(
    hidden_states: torch.Tensor, row_src: torch.Tensor, layer_name: str
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
        return mixer(hidden_states.index_select(0, row_src))
    w13, w2, su, sd, inter, n_shared = layer._fast_moe_weights
    M, H = hidden_states.shape
    x = hidden_states.index_select(0, row_src[:1])  # [1, H]
    router_logits, _ = mixer.gate(x)
    topk_weights, topk_ids = mixer.experts.router.select_experts(
        hidden_states=x, router_logits=router_logits
    )
    return mtp_moe_row(
        x,
        topk_ids.reshape(-1),
        topk_weights.reshape(-1).to(torch.float32),
        w13,
        w2,
        su,
        sd,
        n_shared,
        float(mixer.routed_scaling_factor),
        M,
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


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

_NVFP4_BN, _NVFP4_BK, _NVFP4_WARPS = 8, 128, 4


@triton.jit
def _nvfp4_e2m1(nibble):
    """E2M1 nibble to fp32. Same bit pattern as nvfp4_emulation_utils."""
    magnitude = nibble & 0x07
    sign = (nibble >> 3) & 1
    fp32_bits = 0x3F000000 + (magnitude.to(tl.int32) << 22)
    val = fp32_bits.to(tl.float32, bitcast=True)
    val = tl.where(magnitude == 0, 0.0, val)
    val = tl.where(magnitude == 1, 0.5, val)
    return tl.where(sign == 1, -val, val)


@triton.jit
def _mtp_nvfp4_gemv_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    g_ptr,
    y_ptr,
    N,
    K,
    stride_w,
    stride_s,
    stride_x,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """y[n] = sum_k x[k] * dequant(w[n, k]).

    w is packed uint8, low nibble first, two values per byte, group 16.
    s is the raw e4m3 group scale (uint8 view). g is the ModelOpt forward
    global scale (amax/2688), applied once, not inverted.
    """
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    nmask = offs_n < N
    offs_b = tl.arange(0, BK // 2)
    offs_k = tl.arange(0, BK)
    offs_g = tl.arange(0, BK // 16)
    gscale = tl.load(g_ptr).to(tl.float32)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k0 in range(0, K, BK):
        raw = tl.load(
            w_ptr + offs_n[:, None] * stride_w + (k0 // 2) + offs_b[None, :],
            mask=nmask[:, None],
            other=0,
            eviction_policy="evict_first",
        )
        decoded = tl.interleave(
            _nvfp4_e2m1(raw & 0x0F), _nvfp4_e2m1((raw >> 4) & 0x0F)
        )
        sf_raw = tl.load(
            s_ptr + offs_n[:, None] * stride_s + (k0 // 16) + offs_g[None, :],
            mask=nmask[:, None],
            other=0,
            eviction_policy="evict_first",
        )
        sf = tl.cast(sf_raw, tl.float8e4nv, bitcast=True).to(tl.float32) * gscale
        decoded = tl.reshape(decoded, (BN, BK // 16, 16)) * sf[:, :, None]
        decoded = tl.reshape(decoded, (BN, BK))
        x = tl.load(x_ptr + (k0 + offs_k) * stride_x, other=0.0)
        acc += tl.sum(decoded * x[None, :].to(tl.float32), axis=1)
    tl.store(y_ptr + offs_n, acc.to(y_ptr.dtype.element_ty), mask=nmask)


def draft_nvfp4_logits(
    lm_head: torch.nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor | None:
    """Draft logits for one row, or None to keep the CUTLASS lm_head.

    Activations stay bf16. The target's W4A4 quant of this same head is not
    used here, so draft ids can differ; committed ids are still the target's.
    """
    if os.environ.get("MTP_DRAFT_NVFP4_GEMV", "1") == "0":
        return None
    if getattr(lm_head, "tp_size", 1) != 1:
        return None
    if getattr(lm_head, "bias", None) is not None:
        return None
    weight = getattr(lm_head, "draft_fp4_weight", None)
    scale = getattr(lm_head, "draft_fp4_scale", None)
    global_scale = getattr(lm_head, "draft_fp4_global_scale", None)
    if weight is None or scale is None or global_scale is None:
        logger.info_once("MTP draft NVFP4 GEMV off (no plain lm_head snapshot)")
        return None
    if (
        hidden_states.dim() != 2
        or hidden_states.shape[0] != 1
        or not hidden_states.is_cuda
        or hidden_states.dtype not in (torch.bfloat16, torch.float16)
    ):
        return None
    if (
        weight.dtype != torch.uint8
        or weight.dim() != 2
        or not weight.is_cuda
        or not weight.is_contiguous()
    ):
        return None
    n_rows, packed_k = weight.shape
    k = packed_k * 2
    if hidden_states.shape[1] != k or k % _NVFP4_BK != 0:
        return None
    if scale.shape != (n_rows, k // 16) or not scale.is_contiguous():
        return None
    scale_u8 = scale.view(torch.uint8)
    y = torch.empty(
        (1, n_rows), dtype=hidden_states.dtype, device=hidden_states.device
    )
    _mtp_nvfp4_gemv_kernel[(triton.cdiv(n_rows, _NVFP4_BN),)](
        hidden_states,
        weight,
        scale_u8,
        global_scale,
        y,
        n_rows,
        k,
        weight.stride(0),
        scale_u8.stride(0),
        hidden_states.stride(1),
        BN=_NVFP4_BN,
        BK=_NVFP4_BK,
        num_warps=_NVFP4_WARPS,
        num_stages=2,
    )
    logger.info_once(
        "MTP draft NVFP4 GEMV on (N=%d, K=%d, plain packed lm_head)", n_rows, k
    )
    return y

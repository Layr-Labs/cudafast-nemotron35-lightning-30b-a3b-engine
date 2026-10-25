"""Lossless 12-bit storage for the target's BF16 attention projections at verify size.

The six attention layers of the target keep qkv_proj and o_proj in BF16 (the frozen
checkpoint format). For 2..16 rows torch's F.linear runs cuBLAS'
cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16 kernels, whose output element is one fp32
chain of m16n8k16 MMAs over k in ascending 16-wide steps from 0, rounded once to BF16.
packed_mm computes the same chain (mma_v2 over BK-wide k chunks, ascending) from a
load-time copy of the weight in the drafter's lossless 12-bit storage
(nemotron_h_mtp_packed.pack_bf16, every value checked bit for bit before use), so it
reads about 21% fewer weight bytes and writes the same BF16 values. Every other shape
(1 row, the prefill) calls F.linear on the original weight, as before.
NEMO_ATTN_PACK12=0 disables it.
"""

import os

import torch
import torch.nn.functional as F
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as tl
from triton.experimental.gluon.language import (
    BlockedLayout,
    DotOperandLayout,
    NVMMADistributedLayout,
    SliceLayout,
)
from triton.experimental.gluon.language.nvidia.ampere import mma_v2

from vllm.utils.torch_utils import direct_register_custom_op

from .nemotron_h_mtp_packed import load_packed, pack_bf16, pdl_prologue

_ENABLED = os.environ.get("NEMO_ATTN_PACK12", "1") != "0"
# Rows for which the cuBLAS kernel F.linear picks has the single ascending MMA chain.
_MIN_M, _MAX_M = 2, 16
# (N, K) -> (BN, warps along N, BK); the k-chunk width does not change the chain.
_CONFIGS = {(4608, 2688): (32, 4, 256), (2688, 4096): (16, 2, 256)}


@gluon.jit
def packed_mm(x, SM, EX, BASE, RI, RAW, y, M, N, K, sxm, sym,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, WN: tl.constexpr,
              launch_pdl: tl.constexpr = False, gate_ptr=None, GATED: tl.constexpr = False):
    """y[m, n] = bf16(sum over k16 steps ascending of mma(x[m, k], W[n, k])) for m < M <= BM.
    GATED (drafter): once the depth gate's stop flag is set, the zeros its masked
    weight loads would give."""
    if launch_pdl:
        pdl_prologue()
    ld: tl.constexpr = BlockedLayout([1, 8], [2, 16], [WN, 1], [1, 0])
    mma: tl.constexpr = NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, WN], instr_shape=[16, 8])
    pid = tl.program_id(0)
    nn = pid * BN + tl.arange(0, BN, layout=SliceLayout(1, ld))
    kk0 = tl.arange(0, BK, layout=SliceLayout(0, ld))
    mm = tl.arange(0, BM, layout=SliceLayout(1, ld))
    acc = tl.full((BM, BN), 0, tl.float32, mma)
    kend = K
    if GATED:
        kend = tl.where(tl.load(gate_ptr) != 0, 0, K)
    for k0 in range(0, kend, BK):
        kk = k0 + kk0
        wmask = (nn[:, None] < N) & (kk[None, :] < K)
        w = load_packed(SM, EX, BASE, RI, RAW, nn[:, None] * K + kk[None, :], wmask)
        a = tl.load(x + mm[:, None] * sxm + kk[None, :],
                    mask=(mm[:, None] < M) & (kk[None, :] < K), other=0.)
        a = tl.convert_layout(a, DotOperandLayout(0, mma, 2))
        b = tl.convert_layout(tl.permute(w, (1, 0)), DotOperandLayout(1, mma, 2))
        acc = mma_v2(a, b, acc)
    out: tl.constexpr = BlockedLayout([1, 4], [4, 8], [WN, 1], [1, 0])
    acc = tl.convert_layout(acc, out)
    om = tl.arange(0, BM, layout=SliceLayout(1, out))
    on = pid * BN + tl.arange(0, BN, layout=SliceLayout(0, out))
    tl.store(y + om[:, None] * sym + on[None, :], acc.to(y.dtype.element_ty),
             mask=(om[:, None] < M) & (on[None, :] < N))


def _packed_linear_impl(
    x: torch.Tensor, sm: torch.Tensor, ex: torch.Tensor, base: torch.Tensor,
    ri: torch.Tensor, raw: torch.Tensor, n: int, k: int, weight: torch.Tensor,
) -> torch.Tensor:
    cfg = _CONFIGS.get((n, k))
    if (
        cfg is None
        or x.dim() != 2
        or not _MIN_M <= x.shape[0] <= _MAX_M
        or x.shape[1] != k
        or x.dtype != torch.bfloat16
        or not x.is_cuda
    ):
        return F.linear(x, weight)
    if x.stride(1) != 1:
        x = x.contiguous()
    from .nemotron_h_mtp_kernels import mtp_pdl

    y = x.new_empty((x.shape[0], n))
    bn, wn, bk = cfg
    packed_mm[(triton.cdiv(n, bn),)](
        x, sm, ex, base, ri, raw, y, x.shape[0], n, k, x.stride(0), y.stride(0),
        BM=16, BN=bn, BK=bk, WN=wn, num_warps=wn, launch_pdl=mtp_pdl(),
    )
    return y


def _packed_linear_fake(
    x: torch.Tensor, sm: torch.Tensor, ex: torch.Tensor, base: torch.Tensor,
    ri: torch.Tensor, raw: torch.Tensor, n: int, k: int, weight: torch.Tensor,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], n))


direct_register_custom_op(
    op_name="nemotron_attn_packed_linear", op_func=_packed_linear_impl,
    fake_impl=_packed_linear_fake,
)

_PLANES = ("sm", "ex", "base", "ri", "raw")


def packed(linear: torch.nn.Module) -> bool:
    return hasattr(linear, "_tgt_pk_sm")


def packed_linear(linear: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """linear(x)[0] for a projection prepared by prepare_attention_packed."""
    return torch.ops.vllm.nemotron_attn_packed_linear(
        x, *(getattr(linear, "_tgt_pk_" + s) for s in _PLANES),
        linear._tgt_pk_n, linear._tgt_pk_k, linear.weight,
    )


def prefetch_planes(linear: torch.nn.Module, nbytes: int) -> list[tuple[torch.Tensor, int]]:
    """(plane, bytes) covering the first rows of a packed projection's sign/mantissa and
    exponent planes, nbytes in all (what its packed GEMM reads first): the target's
    copy (prepare_attention_packed) or the drafter's (nemotron_h_mtp_packed)."""
    pre = "_tgt_pk_" if hasattr(linear, "_tgt_pk_sm") else "_mtp_pk_"
    sm, ex = getattr(linear, pre + "sm"), getattr(linear, pre + "ex")
    a = min(sm.numel(), (2 * nbytes) // 3)
    return [(sm, a), (ex, min(ex.numel(), a // 2))]


@torch.no_grad()
def prepare_attention_packed(model: torch.nn.Module) -> None:
    """Pack qkv_proj / o_proj of every BF16 target attention layer (load time, after the
    weights are final). The BF16 weights stay for the shapes packed_mm does not take."""
    if not _ENABLED:
        return
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    for _, mod in model.named_modules():
        if type(mod).__name__ != "NemotronHAttention":
            continue
        for linear in (getattr(mod, "qkv_proj", None), getattr(mod, "o_proj", None)):
            w = getattr(linear, "weight", None)
            if (
                w is None
                or packed(linear)
                or not isinstance(getattr(linear, "quant_method", None), UnquantizedLinearMethod)
                or getattr(linear, "bias", None) is not None
                or w.dtype != torch.bfloat16
                or w.dim() != 2
                or tuple(w.shape) not in _CONFIGS
                or not w.is_cuda
                or not w.is_contiguous()
            ):
                continue
            p = pack_bf16(w)
            if p is None:
                continue
            for suffix, plane in zip(_PLANES, p.planes):
                linear.register_buffer("_tgt_pk_" + suffix, plane, persistent=False)
            linear._tgt_pk_n, linear._tgt_pk_k = p.shape

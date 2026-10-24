"""Lossless boot-time BF16 storage codec for the native MTP head only.

Arithmetic kernels keep the register and reduction layouts of the BF16 drafter kernels.
Code zero preserves exponent-zero sign/mantissa bits, including subnormals.
Out-of-window blocks retain all 16 original bits in a compact raw side plane.
Compute launches explicitly disable programmatic dependent launch: ordinary
stream/graph dependencies complete producers before these kernels read inputs.
"""
import os
import time
from dataclasses import dataclass

import torch
import triton
import triton.language as pl
from vllm.utils.torch_utils import direct_register_custom_op
from triton.experimental import gluon
from triton.experimental.gluon import language as tl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout


@gluon.jit
def load_packed(SM, EX, BASE, RI, RAW, index, mask):
    bid = index // 128
    sm = tl.load(SM + index, mask=mask, other=0, eviction_policy='evict_first').to(tl.uint16)
    nib = tl.load(EX + index // 2, mask=mask, other=0, eviction_policy='evict_first').to(tl.uint16)
    code = (nib >> ((index % 2)*4)) & 15
    base = tl.load(BASE + bid, mask=mask, other=1, eviction_policy='evict_first').to(tl.uint16)
    exp = tl.where(code == 0, 0, base + code - 1)
    bits = ((sm & 128) << 8) | (exp << 7) | (sm & 127)
    rawid = tl.load(RI + bid, mask=mask, other=-1, eviction_policy='evict_first')
    raw = tl.load(RAW + rawid.to(tl.int64)*128 + index % 128,
                  mask=mask & (rawid >= 0), other=0, eviction_policy='evict_first')
    bits = tl.where(rawid >= 0, raw, bits).to(tl.uint16)
    return bits.to(tl.bfloat16, bitcast=True)


@gluon.jit
def pdl_prologue():
    tl.inline_asm_elementwise("griddepcontrol.wait; // dummy $0", "=r", [],
                              dtype=tl.int32, is_pure=False, pack=1)
    tl.inline_asm_elementwise("griddepcontrol.launch_dependents; // dummy $0", "=r", [],
                              dtype=tl.int32, is_pure=False, pack=1)


@gluon.jit
def packed_gemv(x_ptr, SM, EX, BASE, RI, RAW, y_ptr, N, K, stride_wn,
                BN:tl.constexpr, BK:tl.constexpr, EVICT:tl.constexpr,
                launch_pdl:tl.constexpr=False):
    if launch_pdl:
        pdl_prologue()
    layout:tl.constexpr=BlockedLayout([1,8],[1,32],[2,2],[1,0])
    pid=tl.program_id(0)
    offs_n=pid*BN+tl.arange(0,BN,layout=SliceLayout(1,layout))
    offs_k=tl.arange(0,BK,layout=SliceLayout(0,layout))
    nmask=offs_n<N
    acc=tl.full((BN,BK),0,tl.float32,layout)
    for k0 in range(0,K,BK):
        kk=k0+offs_k
        kmask=kk<K
        w=load_packed(SM,EX,BASE,RI,RAW,offs_n[:,None]*stride_wn+kk[None,:],nmask[:,None]&kmask[None,:])
        x=tl.load(x_ptr+kk,mask=kmask,other=0.0)
        acc+=w.to(tl.float32)*x[None,:].to(tl.float32)
    y=tl.sum(acc,axis=1)
    tl.store(y_ptr+offs_n,y.to(y_ptr.dtype.element_ty),mask=nmask)


@gluon.jit
def packed_moe_up(x_ptr, SM, EX, BASE, RI, RAW, SSM, SEX, SBASE, SRI, SRAW,
                   ids_ptr,h_ptr,I,K,stride_we,stride_wn,stride_wsn,
                   TOPK:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,EVICT:tl.constexpr,
                   launch_pdl:tl.constexpr=False):
    if launch_pdl:
        pdl_prologue()
    layout:tl.constexpr=BlockedLayout([1,8],[1,32],[2,4],[1,0])
    pid_n=tl.program_id(0)
    j=tl.program_id(1)
    offs_n=pid_n*BN+tl.arange(0,BN,layout=SliceLayout(1,layout))
    offs_k=tl.arange(0,BK,layout=SliceLayout(0,layout))
    nmask=offs_n<I
    if j<TOPK:
        e=tl.load(ids_ptr+j).to(tl.int64)
        wbase=e*stride_we+offs_n[:,None].to(tl.int64)*stride_wn
    else:
        rows=(j-TOPK)*I+offs_n
        wbase=rows[:,None].to(tl.int64)*stride_wsn
    acc=tl.full((BN,BK),0,tl.float32,layout)
    for k0 in range(0,K,BK):
        kk=k0+offs_k
        kmask=kk<K
        if j<TOPK:
            w=load_packed(SM,EX,BASE,RI,RAW,wbase+kk[None,:],nmask[:,None]&kmask[None,:])
        else:
            w=load_packed(SSM,SEX,SBASE,SRI,SRAW,wbase+kk[None,:],nmask[:,None]&kmask[None,:])
        x=tl.load(x_ptr+kk,mask=kmask,other=0.0)
        acc+=w.to(tl.float32)*x[None,:].to(tl.float32)
    h=tl.maximum(tl.sum(acc,axis=1),0.0)
    h=h*h
    tl.store(h_ptr+j*I+offs_n,h.to(h_ptr.dtype.element_ty),mask=nmask)


@gluon.jit
def verify_decoded(SM,EX,BASE,RI,RAW,W,COUNTER,N:tl.constexpr,B:tl.constexpr):
    ix=tl.program_id(0)*B+tl.arange(0,B,layout=BlockedLayout([1],[32],[4],[0]))
    actual=load_packed(SM,EX,BASE,RI,RAW,ix,ix<N).to(tl.uint16,bitcast=True)
    expected=tl.load(W+ix,mask=ix<N,other=0)
    bad=tl.sum(((actual!=expected)&(ix<N)).to(tl.int32),axis=0)
    tl.atomic_add(COUNTER,bad,sem='relaxed')


@triton.jit
def _pack_stats(W, BASE, FIT, NB: pl.constexpr, B: pl.constexpr):
    blocks = pl.program_id(0) * B + pl.arange(0, B)
    k = pl.arange(0, 128)
    bits = pl.load(W + blocks[:, None] * 128 + k[None, :],
                   blocks[:, None] < NB, other=0).to(pl.uint16)
    exp = (bits >> 7) & 255
    low = pl.min(pl.where(exp == 0, 256, exp).to(pl.int32), 1)
    high = pl.max(exp.to(pl.int32), 1)
    base = pl.where(high == 0, 1, low)
    fit = (high == 0) | (high - low <= 14)
    pl.store(BASE + blocks, base, blocks < NB)
    pl.store(FIT + blocks, ~fit, blocks < NB)


@gluon.jit
def _decode(SM, EX, BASE, RI, RAW, W, N: tl.constexpr, B: tl.constexpr):
    ix = tl.program_id(0) * B + tl.arange(
        0, B, layout=BlockedLayout([1], [32], [4], [0]))
    bits = load_packed(SM, EX, BASE, RI, RAW, ix, ix < N)
    tl.store(W + ix, bits, ix < N)


@dataclass
class PackedBF16:
    shape: tuple
    planes: tuple

    def decode(self):
        w = torch.empty(self.shape, dtype=torch.bfloat16, device=self.planes[0].device)
        _decode[(triton.cdiv(w.numel(), 4096),)](
            *self.planes, w, w.numel(), 4096, num_warps=4)
        return w

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.planes)


@torch.no_grad()
def pack_bf16(w):
    """Deterministic input-independent GPU pack with exhaustive bit validation.

    Only row-contained 128-value blocks are supported; callers retain the
    original tensor for unsupported shapes. No source BF16 storage is retained.
    """
    if (not w.is_cuda or w.dtype != torch.bfloat16 or not w.is_contiguous()
            or w.dim() < 2 or w.shape[-1] % 128):
        return None
    n = w.numel()
    nb = n // 128
    base = torch.empty(nb, dtype=torch.uint8, device=w.device)
    escape = torch.empty(nb, dtype=torch.int32, device=w.device)
    _pack_stats[(triton.cdiv(nb, 8),)](w.view(torch.uint16), base, escape, nb, 8, num_warps=4)
    prefix = torch.cumsum(escape, 0, dtype=torch.int32)
    count = int(prefix[-1].item())
    ri = torch.where(escape != 0, prefix - 1, -1)
    del escape, prefix
    sm = torch.empty(n, dtype=torch.uint8, device=w.device)
    ex = torch.empty(n // 2, dtype=torch.uint8, device=w.device)
    raw = torch.empty(max(1, count) * 128, dtype=torch.uint16, device=w.device)
    flat = w.view(torch.int16).reshape(-1)
    for first in range(0, nb, 16384):
        last = min(first + 16384, nb)
        bits = flat[first * 128:last * 128].to(torch.int32) & 65535
        exp = (bits >> 7) & 255
        b = base[first:last].to(torch.int32).repeat_interleave(128)
        code = torch.where(exp == 0, 0, exp - b + 1) & 15
        sm[first * 128:last * 128].copy_(((bits >> 8) & 128) | (bits & 127))
        ex[first * 64:last * 64].copy_(code[::2] | (code[1::2] << 4))
        select = torch.nonzero(ri[first:last] >= 0).reshape(-1)
        raw.view(torch.int16).reshape(-1, 128).index_copy_(
            0, ri[first:last].index_select(0, select).to(torch.int64),
            flat[first * 128:last * 128].reshape(-1, 128).index_select(0, select))
    del bits, exp, b, code, select
    packed = PackedBF16(tuple(w.shape), (sm, ex, base, ri, raw))
    counter = torch.zeros(1, dtype=torch.int32, device=w.device)
    verify_decoded[(triton.cdiv(n, 4096),)](
        *packed.planes, w.view(torch.uint16), counter, n, 4096, num_warps=4)
    bad = int(counter.item())
    if bad:
        raise RuntimeError(f"MTP lossless pack failed: {bad}/{n} decoded BF16 bits differ")
    print(f"MTP_PACK12 tensor shape={packed.shape} values={n} mismatches=0 "
          f"raw_bytes={2*n} packed_bytes={packed.nbytes} escapes={count}", flush=True)
    return packed


def _packed_linear_impl(
    x: torch.Tensor, sm: torch.Tensor, ex: torch.Tensor, base: torch.Tensor,
    ri: torch.Tensor, raw: torch.Tensor, n: int, k: int, fallback: torch.Tensor,
) -> torch.Tensor:
    from .nemotron_h_mtp_kernels import mtp_pdl, mtp_small_linear_impl

    p = PackedBF16((n, k), (sm, ex, base, ri, raw))
    if x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape == (1, k):
        if x.stride(1) != 1:
            x = x.contiguous()
        y = x.new_empty((1, n))
        packed_gemv[(triton.cdiv(n, 8),)](
            x, *p.planes, y, n, k, k, BN=8, BK=512, EVICT=True,
            num_warps=4, num_stages=3, launch_pdl=mtp_pdl())
        return y
    if fallback.numel():
        return mtp_small_linear_impl(x, fallback)
    if x.numel() == 0:
        return x.new_empty((*x.shape[:-1], n))
    if (x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2
            and 1 < x.shape[0] <= 16 and x.shape[1] == k):
        if x.stride(1) != 1:
            x = x.contiguous()
        y = x.new_empty((x.shape[0], n))
        packed_small[(triton.cdiv(n, 64),)](
            x, *p.planes, y, x.shape[0], n, k, x.stride(0), y.stride(0),
            num_warps=4, num_stages=4, launch_pdl=mtp_pdl())
        return y
    return mtp_small_linear_impl(x, p.decode())


def _packed_linear_fake(
    x: torch.Tensor, sm: torch.Tensor, ex: torch.Tensor, base: torch.Tensor,
    ri: torch.Tensor, raw: torch.Tensor, n: int, k: int, fallback: torch.Tensor,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], n))


direct_register_custom_op(
    op_name="mtp_packed_linear", op_func=_packed_linear_impl,
    fake_impl=_packed_linear_fake,
)


@torch.no_grad()
def prepare_mtp_packed(module):
    """Pack only the screened shapes after original pinned weights are loaded."""
    if os.environ.get("MTP_PACK12", "1") == "0":
        return
    from .nemotron_h_mtp_kernels import mtp_moe_weights

    start = time.monotonic()
    before = torch.cuda.memory_allocated()
    raw_bytes = packed_bytes = values = retained_bytes = 0
    for name, layer in module.named_modules():
        w = getattr(layer, "weight", None)
        keep_bf16 = name.endswith("eh_proj")
        if (w is not None
                and ((keep_bf16 and tuple(w.shape) == (2688, 5376))
                     or (name.endswith("qkv_proj") and tuple(w.shape) == (4608, 2688)))
                and type(getattr(layer, "quant_method", None)).__name__ == "MTPSmallLinearMethod"):
            p = pack_bf16(w)
            if p is None:
                continue
            raw_bytes += w.numel() * 2
            packed_bytes += p.nbytes
            values += w.numel()
            for suffix, tensor in zip(("sm", "ex", "base", "ri", "raw"), p.planes):
                layer.register_buffer("_mtp_pk_" + suffix, tensor, persistent=False)
            layer._mtp_pk_n, layer._mtp_pk_k = p.shape
            if keep_bf16:
                retained_bytes += w.numel() * w.element_size()
            else:
                w.data = torch.empty(0, dtype=w.dtype, device=w.device)
        if not getattr(layer, "_fast_moe", False):
            continue
        weights = layer._fast_moe_weights or mtp_moe_weights(layer)
        if not weights:
            continue
        up, down, su, sd, inter, ns = weights
        if tuple(up.shape) != (128, 1856, 2688) or tuple(su.shape) != (3712, 2688):
            continue
        pu, ps = pack_bf16(up), pack_bf16(su)
        if pu is None or ps is None:
            continue
        for original, packed in ((up, pu), (su, ps)):
            raw_bytes += original.numel() * 2
            packed_bytes += packed.nbytes
            values += original.numel()
            original.data = torch.empty(0, dtype=original.dtype, device=original.device)
        layer._fast_moe_weights = (pu, down, ps, sd, inter, ns)
    torch.cuda.synchronize()
    print(f"MTP_PACK12 ready values={values} mismatches=0 raw_bytes={raw_bytes} "
          f"packed_bytes={packed_bytes} retained_bf16_bytes={retained_bytes} seconds={time.monotonic()-start:.6f} "
          f"allocated_before={before} allocated_after={torch.cuda.memory_allocated()}", flush=True)


from triton.experimental.gluon.language import NVMMADistributedLayout, DotOperandLayout
from triton.experimental.gluon.language.nvidia.ampere import mma_v2


@gluon.jit
def packed_small(x,SM,EX,BASE,RI,RAW,y,M,N,K,sxm,sym,
                 BM:tl.constexpr=16,BN:tl.constexpr=64,BK:tl.constexpr=128,
                 launch_pdl:tl.constexpr=False):
    if launch_pdl:
        pdl_prologue()
    ld:tl.constexpr=BlockedLayout([1,8],[2,16],[4,1],[1,0])
    mma:tl.constexpr=NVMMADistributedLayout(version=[2,0],warps_per_cta=[1,4],instr_shape=[16,8])
    pid=tl.program_id(0)
    nn=pid*BN+tl.arange(0,BN,layout=SliceLayout(1,ld))
    kk0=tl.arange(0,BK,layout=SliceLayout(0,ld))
    mm=tl.arange(0,BM,layout=SliceLayout(1,ld))
    acc=tl.full((BM,BN),0,tl.float32,mma)
    for k0 in range(0,K,BK):
        kk=k0+kk0
        w=load_packed(SM,EX,BASE,RI,RAW,nn[:,None]*K+kk[None,:],(nn[:,None]<N)&(kk[None,:]<K))
        a=tl.load(x+mm[:,None]*sxm+kk[None,:],mask=(mm[:,None]<M)&(kk[None,:]<K),other=0.)
        a=tl.convert_layout(a,DotOperandLayout(0,mma,2))
        b=tl.convert_layout(tl.permute(w,(1,0)),DotOperandLayout(1,mma,2))
        acc=mma_v2(a,b,acc)
    outlayout:tl.constexpr=BlockedLayout([1,4],[4,8],[4,1],[1,0])
    acc=tl.convert_layout(acc,outlayout)
    om=tl.arange(0,BM,layout=SliceLayout(1,outlayout))
    on=pid*BN+tl.arange(0,BN,layout=SliceLayout(0,outlayout))
    tl.store(y+om[:,None]*sym+on[None,:],acc.to(y.dtype.element_ty),mask=(om[:,None]<M)&(on[None,:]<N))

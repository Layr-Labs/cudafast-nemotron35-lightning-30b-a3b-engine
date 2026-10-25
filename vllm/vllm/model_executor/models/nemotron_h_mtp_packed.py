"""Lossless boot-time BF16 storage codec for the native MTP head only.

Arithmetic kernels keep the register and reduction layouts of the BF16 drafter kernels.
Code zero preserves exponent-zero sign/mantissa bits, including subnormals.
Out-of-window blocks retain all 16 original bits in a compact raw side plane.
Compute launches use the drafter's programmatic-dependent-launch flag, and the
same confidence gate as the BF16 kernels: a set stop flag masks every weight
load, so a stopped draft pass writes zeros and streams no packed weights.
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


_BLOCK128_LINEAR = None
_BLOCK128_UP = None


_WORD_LOAD = os.environ.get("MTP_PACK12_WORDLOAD", "1") != "0"
_WORD_OK: dict[int, bool] = {}
_WORD_KERNEL_OK: set[tuple[int, int]] = set()


def _device_index(device):
    return device.index if device.index is not None else torch.cuda.current_device()


def word_load_enabled(device):
    return _WORD_LOAD and _WORD_OK.get(_device_index(device), False)


def _admit_word_decode(packed, original, counter):
    """Official remote load only: exhaustive real-weight bits and kernel samples.

    Any ordinary compiler/runtime/check failure disables this optional word
    specialization for this device. No original BF16 tensor is newly retained.
    """
    if not _WORD_LOAD:
        return
    dev = _device_index(original.device)
    if _WORD_OK.get(dev) is False:
        return
    try:
        counter.zero_()
        n = original.numel()
        verify_decoded[(triton.cdiv(n, 4096),)](
            *packed.planes, original.view(torch.uint16), counter, n, 4096,
            WORD=True, num_warps=4,
        )
        if int(counter.item()) != 0:
            raise RuntimeError("word decoder differs from original BF16 bits")
        kernel_key = (dev, packed.shape[-1])
        if kernel_key not in _WORD_KERNEL_OK:
            _word_kernel_probe(packed, original)
            _WORD_KERNEL_OK.add(kernel_key)
        _WORD_OK[dev] = True
    except Exception as exc:
        _WORD_OK[dev] = False
        print(f"MTP_PACK12_WORDLOAD unavailable, original byte loader kept: {exc!r}",
              flush=True)


def _word_kernel_probe(packed, original):
    """Same layout/order, private real-weight prefix, bytewise old/new kernels."""
    k = packed.shape[-1]
    n = min(64, original.numel() // k)
    if n < 8:
        raise RuntimeError("insufficient rows for optional word-kernel admission")
    gen = torch.Generator(device=original.device).manual_seed(56103)
    gate = torch.zeros(3, dtype=torch.int32, device=original.device)
    ids = torch.zeros(1, dtype=torch.int32, device=original.device)
    for scale in (0.001, 1.0, 32.0):
        x = (torch.randn((5, k), dtype=torch.float32, generator=gen,
                         device=original.device) * scale).to(torch.bfloat16)
        outputs = []
        for word in (False, True):
            a = x.new_empty((1, n))
            packed_gemv[(triton.cdiv(n, 8),)](
                x, *packed.planes, a, n, k, k, gate, BN=8, BK=512,
                EVICT=True, WORD=word, num_warps=4, num_stages=3,
            )
            b = x.new_empty((5, n))
            packed_small[(triton.cdiv(n, 64),)](
                x, *packed.planes, b, 5, n, k, x.stride(0), b.stride(0), gate,
                WORD=word, num_warps=4, num_stages=4,
            )
            c = x.new_empty((2, 2))
            packed_moe_up[(1, 2)](
                x, *packed.planes, *packed.planes, ids, c, 2, k,
                2 * k, k, k, gate, TOPK=1, BN=2, BK=1024, EVICT=True,
                WN=2, WORD=word, num_warps=8, num_stages=3,
            )
            outputs.append((a, b, c))
        if any(not torch.equal(a.view(torch.int16), b.view(torch.int16))
               for a, b in zip(*outputs)):
            raise RuntimeError("word specialization differs from original kernel")


_GROUP8_LOAD = os.environ.get("MTP_PACK12_GROUP8", "1") != "0"
_GROUP8_OK: dict[tuple[int, int, int], bool] = {}


def group8_enabled(device, n, k):
    return _GROUP8_LOAD and _GROUP8_OK.get((_device_index(device), n, k), False)


def _admit_group8(packed, original):
    if not _GROUP8_LOAD or original.dim() != 2:
        return
    n, k = original.shape
    if (n, k) not in ((4608, 2688), (2688, 4096)):
        return
    key = (_device_index(original.device), n, k)
    if key in _GROUP8_OK:
        return
    try:
        gate = torch.zeros(3, dtype=torch.int32, device=original.device)
        gen = torch.Generator(device=original.device).manual_seed(74113)
        for scale in (0.001, 1.0, 32.0):
            x = (torch.randn((1, k), dtype=torch.float32, generator=gen,
                             device=original.device) * scale).to(torch.bfloat16)
            outputs = []
            for grouped in (False, True):
                y = x.new_empty((1, n))
                packed_gemv[(triton.cdiv(n, 8),)](
                    x, *packed.planes, y, n, k, k, gate,
                    BN=8, BK=512, EVICT=True, GATED=True, launch_pdl=True,
                    WORD=False, GROUP8=grouped, num_warps=4, num_stages=3,
                )
                outputs.append(y)
            if not torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16)):
                raise RuntimeError("GROUP8 differs from original packed_gemv")
        _GROUP8_OK[key] = True
        print(f"MTP_PACK12_GROUP8 exact shape={(n, k)}", flush=True)
    except Exception as exc:
        _GROUP8_OK[key] = False
        print(f"MTP_PACK12_GROUP8 unavailable, original packed_gemv kept: {exc!r}",
              flush=True)


@gluon.jit
def load_packed(SM, EX, BASE, RI, RAW, index, mask, EVICT:tl.constexpr=True,
                WORD:tl.constexpr=False):
    bid = index // 128
    # Escape blocks are supplied entirely by RAW. Read their metadata first,
    # so they do not also stream unused sign/mantissa and exponent planes.
    rawid = tl.load(RI + bid, mask=mask, other=-1, eviction_policy='evict_first' if EVICT else '')
    coded = mask & (rawid < 0)
    sm = tl.load(SM + index, mask=coded, other=0, eviction_policy='evict_first' if EVICT else '').to(tl.uint16)
    if WORD:
        words = EX.to(tl.pointer_type(tl.uint32))
        word = tl.load(words + index // 8, mask=coded, other=0,
                       eviction_policy='evict_first' if EVICT else '')
        code = ((word >> ((index % 8).to(tl.uint32) * 4)) & 15).to(tl.uint16)
    else:
        nib = tl.load(EX + index // 2, mask=coded, other=0,
                      eviction_policy='evict_first' if EVICT else '').to(tl.uint16)
        code = (nib >> ((index % 2)*4)) & 15
    base = tl.load(BASE + bid, mask=coded, other=1, eviction_policy='evict_first' if EVICT else '').to(tl.uint16)
    exp = tl.where(code == 0, 0, base + code - 1)
    bits = ((sm & 128) << 8) | (exp << 7) | (sm & 127)
    raw = tl.load(RAW + rawid.to(tl.int64)*128 + index % 128,
                  mask=mask & (rawid >= 0), other=0, eviction_policy='evict_first' if EVICT else '')
    bits = tl.where(rawid >= 0, raw, bits).to(tl.uint16)
    return bits.to(tl.bfloat16, bitcast=True)


@gluon.jit
def pdl_prologue():
    tl.inline_asm_elementwise("griddepcontrol.wait; // dummy $0", "=r", [],
                              dtype=tl.int32, is_pure=False, pack=1)
    tl.inline_asm_elementwise("griddepcontrol.launch_dependents; // dummy $0", "=r", [],
                              dtype=tl.int32, is_pure=False, pack=1)


@gluon.jit
def stopped_zero(ptrs, mask, lay:tl.constexpr, BN:tl.constexpr):
    """+0 at ptrs: what a stopped (depth-gated) pass's masked weight loads sum to."""
    tl.store(ptrs, tl.full((BN,), 0.0, tl.float32, lay).to(ptrs.dtype.element_ty), mask=mask)


@gluon.jit
def load_packed_group8(SM, EX, BASE, RI, RAW, index, lane, mask, value_mask,
                       EVICT:tl.constexpr=True):
    bid=index//128
    rawid=tl.load(RI+bid,mask=mask,other=-1,eviction_policy='evict_first' if EVICT else '')
    coded=mask&(rawid<0)
    index8=index[:,None]+lane[None,:]
    rawid8,_=tl.broadcast(rawid[:,None],index8)
    sm=tl.load(SM+index8,mask=value_mask&(rawid8<0),other=0,eviction_policy='evict_first' if EVICT else '').to(tl.uint16)
    words=EX.to(tl.pointer_type(tl.uint32))
    word=tl.load(words+index//8,mask=coded,other=0,eviction_policy='evict_first' if EVICT else '')
    code=((word[:,None]>>(lane[None,:].to(tl.uint32)*4))&15).to(tl.uint16)
    base=tl.load(BASE+bid,mask=coded,other=1,eviction_policy='evict_first' if EVICT else '').to(tl.uint16)
    base8,_=tl.broadcast(base[:,None],index8)
    exp=tl.where(code==0,0,base8+code-1)
    bits=((sm&128)<<8)|(exp<<7)|(sm&127)
    raw=tl.load(RAW+rawid8.to(tl.int64)*128+index8%128,
                mask=value_mask&(rawid8>=0),other=0,eviction_policy='evict_first' if EVICT else '')
    bits=tl.where(rawid8>=0,raw,bits)
    bits=tl.where(value_mask,bits,0).to(tl.uint16)
    return bits.to(tl.bfloat16,bitcast=True)


@gluon.jit
def packed_gemv(x_ptr, SM, EX, BASE, RI, RAW, y_ptr, N, K, stride_wn,
                gate_ptr, BN:tl.constexpr, BK:tl.constexpr, EVICT:tl.constexpr,
                GATED:tl.constexpr=False, launch_pdl:tl.constexpr=False,
                WORD:tl.constexpr=False, GROUP8:tl.constexpr=False):
    if launch_pdl:
        pdl_prologue()
    layout:tl.constexpr=BlockedLayout([1,8],[1,32],[2,2],[1,0])
    pid=tl.program_id(0)
    offs_n=pid*BN+tl.arange(0,BN,layout=SliceLayout(1,layout))
    offs_k=tl.arange(0,BK,layout=SliceLayout(0,layout))
    if GROUP8:
        group_layout:tl.constexpr=BlockedLayout([1,8],[32,1],[4,1],[1,0])
        group=tl.arange(0,BN*(BK//8),layout=SliceLayout(1,group_layout))
        lane=tl.arange(0,8,layout=SliceLayout(0,group_layout))
        group_n=pid*BN+group//(BK//8)
        group_k=(group%(BK//8))*8
    nmask=offs_n<N
    if GATED:
        if tl.load(gate_ptr)!=0:
            # Stopped pass: store the zeros the masked loads would sum to, stream nothing.
            stopped_zero(y_ptr+offs_n,nmask,SliceLayout(1,layout),BN)
            return
    wn=nmask
    acc=tl.full((BN,BK),0,tl.float32,layout)
    for k0 in range(0,K,BK):
        kk=k0+offs_k
        kmask=kk<K
        if GROUP8:
            gk=k0+group_k
            index=group_n*stride_wn+gk
            mask=(group_n<N)&(gk<K)
            value_mask=mask[:,None]&(gk[:,None]+lane[None,:]<K)
            w8=load_packed_group8(SM,EX,BASE,RI,RAW,index,lane,mask,value_mask,EVICT=EVICT)
            w=tl.convert_layout(tl.reshape(w8,(BN,BK)),layout,assert_trivial=True)
        else:
            w=load_packed(SM,EX,BASE,RI,RAW,offs_n[:,None]*stride_wn+kk[None,:],wn[:,None]&kmask[None,:],EVICT=EVICT,WORD=WORD)
        x=tl.load(x_ptr+kk,mask=kmask,other=0.0)
        acc+=w.to(tl.float32)*x[None,:].to(tl.float32)
    y=tl.sum(acc,axis=1)
    tl.store(y_ptr+offs_n,y.to(y_ptr.dtype.element_ty),mask=nmask)


@gluon.jit
def packed_moe_up(x_ptr, SM, EX, BASE, RI, RAW, SSM, SEX, SBASE, SRI, SRAW,
                   ids_ptr,h_ptr,I,K,stride_we,stride_wn,stride_wsn,gate_ptr,
                   TOPK:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,EVICT:tl.constexpr,
                   GATED:tl.constexpr=False, launch_pdl:tl.constexpr=False,
                   WN:tl.constexpr=2, WORD:tl.constexpr=False, GROUP8:tl.constexpr=True):
    if launch_pdl:
        pdl_prologue()
    # WN warps along N, 4 along K: the K layout (8 x 32 x 4), and with it every
    # row's accumulation and reduction order, does not depend on WN or BN.
    layout:tl.constexpr=BlockedLayout([1,8],[1,32],[WN,4],[1,0])
    pid_n=tl.program_id(0)
    j=tl.program_id(1)
    offs_n=pid_n*BN+tl.arange(0,BN,layout=SliceLayout(1,layout))
    offs_k=tl.arange(0,BK,layout=SliceLayout(0,layout))
    if GROUP8:
        group_layout:tl.constexpr=BlockedLayout([1,8],[32,1],[WN*4,1],[1,0])
        group=tl.arange(0,BN*(BK//8),layout=SliceLayout(1,group_layout))
        lane=tl.arange(0,8,layout=SliceLayout(0,group_layout))
        group_n=pid_n*BN+group//(BK//8)
        group_k=(group%(BK//8))*8
    nmask=offs_n<I
    if GATED:
        if tl.load(gate_ptr)!=0:
            # Stopped pass: store the zeros the masked loads would give, stream nothing.
            stopped_zero(h_ptr+j*I+offs_n,nmask,SliceLayout(1,layout),BN)
            return
    wn=nmask
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
        if GROUP8:
            gk=k0+group_k
            if j<TOPK:
                index=tl.load(ids_ptr+j).to(tl.int64)*stride_we+group_n.to(tl.int64)*stride_wn+gk
            else:
                index=((j-TOPK)*I+group_n).to(tl.int64)*stride_wsn+gk
            mask=(group_n<I)&(gk<K)
            value_mask=mask[:,None]&(gk[:,None]+lane[None,:]<K)
            if j<TOPK:
                w8=load_packed_group8(SM,EX,BASE,RI,RAW,index,lane,mask,value_mask,EVICT=EVICT)
            else:
                w8=load_packed_group8(SSM,SEX,SBASE,SRI,SRAW,index,lane,mask,value_mask,EVICT=EVICT)
            w=tl.convert_layout(tl.reshape(w8,(BN,BK)),layout)
        else:
            if j<TOPK:
                w=load_packed(SM,EX,BASE,RI,RAW,wbase+kk[None,:],wn[:,None]&kmask[None,:],EVICT=EVICT,WORD=WORD)
            else:
                w=load_packed(SSM,SEX,SBASE,SRI,SRAW,wbase+kk[None,:],wn[:,None]&kmask[None,:],EVICT=EVICT,WORD=WORD)
        x=tl.load(x_ptr+kk,mask=kmask,other=0.0)
        acc+=w.to(tl.float32)*x[None,:].to(tl.float32)
    h=tl.maximum(tl.sum(acc,axis=1),0.0)
    h=h*h
    tl.store(h_ptr+j*I+offs_n,h.to(h_ptr.dtype.element_ty),mask=nmask)


@gluon.jit
def verify_decoded(SM,EX,BASE,RI,RAW,W,COUNTER,N:tl.constexpr,B:tl.constexpr,
                   WORD:tl.constexpr=False):
    ix=tl.program_id(0)*B+tl.arange(0,B,layout=BlockedLayout([1],[32],[4],[0]))
    actual=load_packed(SM,EX,BASE,RI,RAW,ix,ix<N,WORD=WORD).to(tl.uint16,bitcast=True)
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
def pack_bf16(w, check_group8=False):
    """Deterministic input-independent GPU pack with exhaustive bit validation.

    Only row-contained 128-value blocks are supported; callers retain the
    original tensor for unsupported shapes. No source BF16 storage is retained.
    check_group8 additionally compares the grouped native projection with BYTE.
    """
    global _BLOCK128_LINEAR
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
    _admit_word_decode(packed, w, counter)
    if check_group8:
        _admit_group8(packed, w)
        try:
            from .nemotron_h_mtp_block128 import prepare_linear, maybe_linear
            prepare_linear(packed, w)
            _BLOCK128_LINEAR = maybe_linear
        except Exception as exc:
            print(f"MTP_PACK12_BLOCK128 original linear retained: {exc!r}", flush=True)
    print(f"MTP_PACK12 tensor shape={packed.shape} values={n} mismatches=0 "
          f"raw_bytes={2*n} packed_bytes={packed.nbytes} escapes={count}", flush=True)
    return packed


def _packed_linear_impl(
    x: torch.Tensor, sm: torch.Tensor, ex: torch.Tensor, base: torch.Tensor,
    ri: torch.Tensor, raw: torch.Tensor, n: int, k: int, fallback: torch.Tensor,
) -> torch.Tensor:
    from .nemotron_h_mtp_kernels import _gate_args, mtp_pdl, mtp_small_linear_impl

    p = PackedBF16((n, k), (sm, ex, base, ri, raw))
    gate, gated = _gate_args(x)
    if x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape == (1, k):
        if x.stride(1) != 1:
            x = x.contiguous()
        y = x.new_empty((1, n))
        if _BLOCK128_LINEAR is not None and _BLOCK128_LINEAR(
                p, x, y, gate, gated, mtp_pdl(),
                word_load_enabled(x.device), group8_enabled(x.device, n, k)):
            return y
        packed_gemv[(triton.cdiv(n, 8),)](
            x, *p.planes, y, n, k, k, gate, BN=8, BK=512, EVICT=True,
            GATED=gated, num_warps=4, num_stages=3, launch_pdl=mtp_pdl(),
            WORD=word_load_enabled(x.device),
            GROUP8=group8_enabled(x.device, n, k))
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
        from .nemotron_h_mtp_pkmm import maybe_run
        if maybe_run(p, x, y, gate, gated, word_load_enabled(x.device), mtp_pdl()):
            return y
        packed_small[(triton.cdiv(n, 64),)](
            x, *p.planes, y, x.shape[0], n, k, x.stride(0), y.stride(0), gate,
            GATED=gated, num_warps=4, num_stages=4, launch_pdl=mtp_pdl(),
            WORD=word_load_enabled(x.device))
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


def _packed_linear_keep_impl(
    x: torch.Tensor, sm: torch.Tensor, ex: torch.Tensor, base: torch.Tensor,
    ri: torch.Tensor, raw: torch.Tensor, n: int, k: int, fallback: torch.Tensor,
) -> torch.Tensor:
    from .nemotron_h_mtp_kernels import _gate_args, mtp_pdl

    if x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape == (1, k):
        if x.stride(1) != 1:
            x = x.contiguous()
        gate, gated = _gate_args(x)
        y = x.new_empty((1, n))
        packed_gemv[(triton.cdiv(n, 8),)](
            x, sm, ex, base, ri, raw, y, n, k, k, gate,
            BN=8, BK=512, EVICT=False, GATED=gated,
            num_warps=4, num_stages=3, launch_pdl=mtp_pdl(),
            WORD=word_load_enabled(x.device))
        return y
    return torch.ops.vllm.mtp_small_linear_keep(x, fallback)


direct_register_custom_op(
    op_name="mtp_packed_linear_keep", op_func=_packed_linear_keep_impl,
    fake_impl=_packed_linear_fake,
)


@torch.no_grad()
def prepare_eh_hidden_packed(layer):
    """Pack the actual projected-table hidden half; keep BF16 for all fallbacks."""
    if os.environ.get("MTP_PACK12", "1") == "0":
        return
    w = layer.eh_w_h
    if tuple(w.shape) != (2688, 2688):
        return
    packed = pack_bf16(w)
    if packed is None:
        return
    for suffix, tensor in zip(("sm", "ex", "base", "ri", "raw"), packed.planes):
        layer.register_buffer("_mtp_eh_pk_" + suffix, tensor, persistent=False)


_PACK_OPROJ = os.environ.get("MTP_PACK12_OPROJ", "1") != "0"


@torch.no_grad()
def prepare_mtp_packed(module):
    """Pack only the screened shapes after original pinned weights are loaded."""
    global _BLOCK128_UP
    if os.environ.get("MTP_PACK12", "1") == "0":
        return
    from .nemotron_h_mtp_kernels import mtp_moe_weights

    start = time.monotonic()
    before = torch.cuda.memory_allocated()
    raw_bytes = packed_bytes = values = retained_bytes = 0
    for name, layer in module.named_modules():
        w = getattr(layer, "weight", None)
        if (w is not None
                and ((name.endswith("qkv_proj") and tuple(w.shape) == (4608, 2688))
                     or (_PACK_OPROJ and name.endswith("o_proj")
                         and tuple(w.shape) == (2688, 4096)))
                and type(getattr(layer, "quant_method", None)).__name__ == "MTPSmallLinearMethod"):
            p = pack_bf16(w, check_group8=True)
            if p is None:
                continue
            from .nemotron_h_mtp_pkmm import prepare
            prepare(p, word_load_enabled(w.device))
            raw_bytes += w.numel() * 2
            packed_bytes += p.nbytes
            values += w.numel()
            for suffix, tensor in zip(("sm", "ex", "base", "ri", "raw"), p.planes):
                layer.register_buffer("_mtp_pk_" + suffix, tensor, persistent=False)
            layer._mtp_pk_n, layer._mtp_pk_k = p.shape
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
        try:
            from .nemotron_h_mtp_block128 import prepare_up, maybe_up
            prepare_up(pu, ps, up, su)
            _BLOCK128_UP = maybe_up
        except Exception as exc:
            print(f"MTP_PACK12_BLOCK128 original up retained: {exc!r}", flush=True)
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
def packed_small(x,SM,EX,BASE,RI,RAW,y,M,N,K,sxm,sym,gate_ptr,
                 BM:tl.constexpr=16,BN:tl.constexpr=64,BK:tl.constexpr=128,
                 GATED:tl.constexpr=False, launch_pdl:tl.constexpr=False,
                 WORD:tl.constexpr=False):
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
        wmask=(nn[:,None]<N)&(kk[None,:]<K)
        if GATED:
            wmask=wmask & (tl.load(gate_ptr)==0)
        w=load_packed(SM,EX,BASE,RI,RAW,nn[:,None]*K+kk[None,:],wmask,WORD=WORD)
        a=tl.load(x+mm[:,None]*sxm+kk[None,:],mask=(mm[:,None]<M)&(kk[None,:]<K),other=0.)
        a=tl.convert_layout(a,DotOperandLayout(0,mma,2))
        b=tl.convert_layout(tl.permute(w,(1,0)),DotOperandLayout(1,mma,2))
        acc=mma_v2(a,b,acc)
    outlayout:tl.constexpr=BlockedLayout([1,4],[4,8],[4,1],[1,0])
    acc=tl.convert_layout(acc,outlayout)
    om=tl.arange(0,BM,layout=SliceLayout(1,outlayout))
    on=pid*BN+tl.arange(0,BN,layout=SliceLayout(0,outlayout))
    tl.store(y+om[:,None]*sym+on[None,:],acc.to(y.dtype.element_ty),mask=(om[:,None]<M)&(on[None,:]<N))


def maybe_block128_up(p, ps, x, ids, out, gate, gated, pdl, word, evict, bn, bk, wn, stages):
    return (_BLOCK128_UP is not None and _BLOCK128_UP(
        p, ps, x, ids, out, gate, gated, pdl, word, evict, bn, bk, wn, stages))

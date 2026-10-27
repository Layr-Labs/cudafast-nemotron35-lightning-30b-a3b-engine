# SPDX-License-Identifier: Apache-2.0
"""Actual-owner-admitted adaptive native-MTP slot tasks. Official remote only.

Original reduction/math helpers are copied from highest1faceed1. No local
execution, graph, resource, correctness or throughput evidence exists.
"""
import math
import os
import statistics
from types import SimpleNamespace

import torch
import triton
import triton.language as pl
from triton.experimental import gluon
from triton.experimental.gluon import language as tl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout
from .nemotron_h_mtp_packed import load_packed, load_packed_group8, pdl_prologue, stopped_zero


@triton.jit
def route_plan(tw, ids, out_tw, out_ids, refine,
               TAU: pl.constexpr = 0.625, FORCE: pl.constexpr = -1,
               launch_pdl: pl.constexpr = False):
    if launch_pdl:
        pl.extra.cuda.gdc_wait()
        pl.extra.cuda.gdc_launch_dependents()
    i = pl.arange(0, 8)
    valid = i < 6
    w = pl.load(tw + i, mask=valid, other=0.0)
    e = pl.load(ids + i, mask=valid, other=0)
    # Exact original stable keep-3 rank; NaNs sort first, equal slots stable.
    key = pl.where(w != w, float("inf"), w)
    wi, wj = key[:, None], key[None, :]
    above = valid[None, :] & ((wj > wi) | ((wj == wi) & (i[None, :] < i[:, None])))
    rank = pl.sum(above.to(pl.int32), axis=1)
    keep = valid & (rank < 3)
    mass = pl.sum(pl.where(keep, w, 0.0), axis=0)
    total = pl.sum(pl.where(valid, w, 0.0), axis=0)
    active = ((total > 0.0) & (pl.abs(total) < float("inf"))
              & (mass == mass) & (mass < TAU * total))
    if FORCE >= 0:
        active = FORCE != 0
    pl.store(refine, active.to(pl.int32))
    if active:
        # Original six-router order, not a newly sorted six-expert sum.
        pl.store(out_tw + i, w, mask=valid)
        pl.store(out_ids + i, e, mask=valid)
    else:
        pl.store(out_tw + rank, w, mask=keep)
        pl.store(out_ids + rank, e, mask=keep)
        pl.store(out_tw + i, 0.0, mask=(i >= 3) & valid)
        pl.store(out_ids + i, 0, mask=(i >= 3) & valid)
    # Original keep-3 does NOT renormalize. Neither branch renormalizes here.

@gluon.jit
def _up_task(x_ptr, SM, EX, BASE, RI, RAW, SSM, SEX, SBASE, SRI, SRAW,
                   ids_ptr,h_ptr,I,K,stride_we,stride_wn,stride_wsn,gate_ptr,slot,
                   TOPK:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,EVICT:tl.constexpr,
                   GATED:tl.constexpr=False, launch_pdl:tl.constexpr=False,
                   WN:tl.constexpr=2, WORD:tl.constexpr=False, GROUP8:tl.constexpr=True):
    if launch_pdl:
        pdl_prologue()
    # WN warps along N, 4 along K: the K layout (8 x 32 x 4), and with it every
    # row's accumulation and reduction order, does not depend on WN or BN.
    layout:tl.constexpr=BlockedLayout([1,8],[1,32],[WN,4],[1,0])
    pid_n=tl.program_id(0)
    j=slot
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
def adaptive_up(x_ptr, SM, EX, BASE, RI, RAW, SSM, SEX, SBASE, SRI, SRAW,
                ids_ptr, h_ptr, I, K, stride_we, stride_wn, stride_wsn,
                gate_ptr, refine_ptr, BN: tl.constexpr = 2, BK: tl.constexpr = 1024,
                EVICT: tl.constexpr = True, GATED: tl.constexpr = False,
                launch_pdl: tl.constexpr = False, WN: tl.constexpr = 2,
                WORD: tl.constexpr = False):
    if launch_pdl:
        pdl_prologue()
    task = tl.program_id(1)
    slot = tl.where(task < 3, task, task + 3)
    _up_task(x_ptr, SM, EX, BASE, RI, RAW, SSM, SEX, SBASE, SRI, SRAW,
             ids_ptr, h_ptr, I, K, stride_we, stride_wn, stride_wsn, gate_ptr, slot,
             TOPK=6, BN=BN, BK=BK, EVICT=EVICT, GATED=GATED,
             launch_pdl=False, WN=WN, WORD=WORD, GROUP8=True)
    if task < 3 and tl.load(refine_ptr) != 0:
        # Same physical CTA computes an extra routed slot only when requested.
        # No additional up CTA, extra decoded shared tile or extra warps.
        _up_task(x_ptr, SM, EX, BASE, RI, RAW, SSM, SEX, SBASE, SRI, SRAW,
                 ids_ptr, h_ptr, I, K, stride_we, stride_wn, stride_wsn,
                 gate_ptr, task + 3, TOPK=6, BN=BN, BK=BK, EVICT=EVICT,
                 GATED=GATED, launch_pdl=False, WN=WN, WORD=WORD, GROUP8=True)


@triton.jit
def adaptive_compact(h, indices, values, counts, refine,
                     I: pl.constexpr = 1856, BLOCK: pl.constexpr = 2048,
                     launch_pdl: pl.constexpr = False):
    if launch_pdl:
        pl.extra.cuda.gdc_wait()
        pl.extra.cuda.gdc_launch_dependents()
    slot = pl.program_id(0)
    if slot >= 3 and slot < 6 and pl.load(refine) == 0:
        # These up slots were not computed. Do not read their uninitialized h.
        pl.store(counts + slot, 0)
        return
    k = pl.arange(0, BLOCK)
    value = pl.load(h + slot * I + k, mask=k < I, other=0.0)
    live = (k < I) & (value != 0.0)
    position = pl.cumsum(live.to(pl.int32), axis=0) - 1
    pl.store(indices + slot * I + position, k, mask=live)
    pl.store(values + slot * I + position, value, mask=live)
    pl.store(counts + slot, pl.sum(live.to(pl.int32), axis=0))


@triton.jit
def adaptive_combine(part, weights, y, refine,
                     M: pl.constexpr, H: pl.constexpr, ROUTED_SCALE: pl.constexpr,
                     BN: pl.constexpr = 256, launch_pdl: pl.constexpr = False,
                     residual_ptr=None, ADD_RES: pl.constexpr = False):
    if launch_pdl:
        pl.extra.cuda.gdc_wait()
        pl.extra.cuda.gdc_launch_dependents()
    n = pl.program_id(0) * BN + pl.arange(0, BN)
    routed = pl.zeros((BN,), pl.float32)
    for j in range(3):
        routed += pl.load(weights + j).to(pl.float32) * pl.load(
            part + j * H + n, mask=n < H, other=0.0)
    if pl.load(refine) != 0:
        for j in range(3, 6):
            routed += pl.load(weights + j).to(pl.float32) * pl.load(
                part + j * H + n, mask=n < H, other=0.0)
    shared = pl.zeros((BN,), pl.float32)
    for s in range(2):
        shared += pl.load(part + (6 + s) * H + n, mask=n < H, other=0.0)
    out = (routed * ROUTED_SCALE + shared).to(y.dtype.element_ty)
    for row in range(M):
        value = out
        if ADD_RES:
            residual = pl.load(residual_ptr + row * H + n, mask=n < H, other=0.0)
            value = (out.to(pl.float32) + residual.to(pl.float32)).to(y.dtype.element_ty)
        pl.store(y + row * H + n, value, mask=n < H)


_ENABLED = os.environ.get("MTP_ADAPT_ROUTE", "1") != "0"
_ENTRIES = {}


def _signature(t):
    return (id(t), t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)


def _prefetch_signature(k):
    p = k._l2pf
    return None if p is None else (
        id(p), p.DRAFT_BYTES, getattr(p, "DRAFT_SHARED_BYTES", 0),
        k._PF_GATED, tuple(_signature(t) for t in p.DRAFT_SCREEN))


def _owner(layer):
    from . import nemotron_h_mtp_kernels as k
    from . import nemotron_h_mtp_down_packed as d
    from .nemotron_h_mtp_packed import PackedBF16, word_load_enabled
    from .nemotron_h_mtp_compact import _ENABLED as compact, _SLOTS
    w = layer._fast_moe_weights
    if (not _ENABLED or not w or not k.mtp_fast_enabled("MOE")
            or k._DRAFT_TOPK != 3 or not k._TOPK_KEEP_KERNEL or k._RELU2_MODE != 2
            or (k._PK_UP_BN, k._PK_UP_WN, k._MOE_UP_BK, k._MOE_UP_STAGES) != (2, 2, 1024, 3)
            or not compact or not _SLOTS or not d._ENABLED):
        return None
    up, down, shared, sd, inter, ns = w
    if (not isinstance(up, PackedBF16) or not isinstance(shared, PackedBF16)
            or up.shape != (128, 1856, 2688) or shared.shape != (3712, 2688)
            or tuple(down.shape) != (128, 2688, 1856)
            or tuple(sd.shape) != (2688, 3712) or (inter, ns) != (1856, 2)
            or not down.is_cuda or down.dtype != torch.bfloat16
            or torch.cuda.get_device_capability(down.device) != (12, 1)):
        return None
    packed = k._RELU2_PACKED.get((down.data_ptr(), sd.data_ptr()))
    if not packed:
        return None
    key = tuple(t.data_ptr() for t in packed)
    entry = d._PACKS.get(key)
    word = word_load_enabled(down.device)
    if (entry is None or any(a is not b for a, b in zip(entry[:2], packed))
            or entry[2] is None or entry[3] is None
            or any(entry[4].get((n, word)) != (64, 4)
                   or not d._PKSLOTS_OK.get((key, n, word)) for n in (3, 6))):
        # Do not substitute a 64x4 proxy for the owner's actual selected oracle.
        return None
    tensors = (down, sd, *up.planes, *shared.planes,
               *packed, *entry[2].planes, *entry[3].planes)
    if any(not t.is_cuda or t.device != down.device or not t.is_contiguous() for t in tensors):
        return None
    scale = float(layer.mixer.routed_scaling_factor)
    if not math.isfinite(scale):
        return None
    return SimpleNamespace(layer=layer, weights=w, packed=packed, down_entry=entry,
                           tensors=tensors, signatures=tuple(map(_signature, tensors)),
                           scale=scale, device=down.device, word=word, pdl=k.mtp_pdl(),
                           evict=k._EVICT_W, prefetch_signature=_prefetch_signature(k))


def _can(spec, x, ids, tw, m, residual):
    from . import nemotron_h_mtp_kernels as k
    from .nemotron_h_mtp_compact import can_add_residual
    from . import nemotron_h_mtp_compact as c
    from . import nemotron_h_mtp_down_packed as d
    from .nemotron_h_mtp_packed import word_load_enabled
    if (not _ENABLED or spec.layer._fast_moe_weights is not spec.weights
            or spec.scale != float(spec.layer.mixer.routed_scaling_factor)
            or not k.mtp_fast_enabled("MOE") or k._DRAFT_TOPK != 3
            or not k._TOPK_KEEP_KERNEL or k._RELU2_MODE != 2 or k._EVICT_W != spec.evict
            or not c._ENABLED or not c._SLOTS or not d._ENABLED
            or _prefetch_signature(k) != spec.prefetch_signature
            or k._RELU2_PACKED.get((spec.weights[1].data_ptr(), spec.weights[3].data_ptr())) is not spec.packed
            or d._PACKS.get(tuple(t.data_ptr() for t in spec.packed)) is not spec.down_entry
            or any(spec.down_entry[4].get((n, spec.word)) != (64, 4)
                   or not d._PKSLOTS_OK.get((tuple(t.data_ptr() for t in spec.packed), n, spec.word))
                   for n in (3, 6))
            or k.mtp_pdl() != spec.pdl or word_load_enabled(spec.device) != spec.word
            or (k._PK_UP_BN, k._PK_UP_WN, k._MOE_UP_BK, k._MOE_UP_STAGES) != (2, 2, 1024, 3)
            or any(_signature(t) != s for t, s in zip(spec.tensors, spec.signatures))
            or m not in (1, 5, 8) or x.dtype != torch.bfloat16 or x.shape != (1, 2688)
            or not x.is_contiguous() or x.device != spec.device
            or ids.dtype != torch.int32 or ids.numel() != 6 or not ids.is_contiguous()
            or tw.dtype != torch.float32 or tw.numel() != 6 or not tw.is_contiguous()
            or ids.device != x.device or tw.device != x.device):
        return False
    gate, gated = k._gate_args(x)
    if not gated or gate.dtype != torch.int32 or gate.numel() < 1 or not gate.is_contiguous():
        return False
    if residual is not None:
        # Same residual-fold contract as the actual original down path.
        if (residual.shape != (m, 2688) or residual.dtype != x.dtype
                or residual.device != x.device or not residual.is_contiguous()
                or not can_add_residual(residual, residual, spec.scale, 3)
                or not can_add_residual(residual, residual, spec.scale, 6)):
            return False
    return True


def _core(spec, x, ids, tw, m, residual, destination=None, force=-1):
    """All GPU work of the production path; no fallback after this function starts."""
    from . import nemotron_h_mtp_kernels as k
    import pkroute
    up, _, shared, _, _, _ = spec.weights
    selected = torch.empty(6, dtype=torch.int32, device=x.device)
    weights = torch.empty(6, dtype=torch.float32, device=x.device)
    refine = torch.empty(1, dtype=torch.int32, device=x.device)
    h = torch.empty((8, 1856), dtype=x.dtype, device=x.device)
    indices = torch.empty((8, 1856), dtype=torch.int32, device=x.device)
    values = torch.empty_like(h)
    counts = torch.empty(8, dtype=torch.int32, device=x.device)
    part = torch.empty((8, 2688), dtype=torch.float32, device=x.device)
    y = destination if destination is not None else torch.empty((m, 2688), dtype=x.dtype, device=x.device)
    gate, gated = k._gate_args(x)
    route_plan[(1,)](tw, ids, weights, selected, refine, FORCE=force,
                    num_warps=1, launch_pdl=spec.pdl)
    adaptive_up[(928, 5)](
        x, *up.planes, *shared.planes, selected, h, 1856, 2688,
        1856 * 2688, 2688, 2688, gate, refine,
        BN=2, BK=1024, EVICT=k._EVICT_W, GATED=gated, WN=2,
        WORD=spec.word, num_warps=8, num_stages=3, launch_pdl=spec.pdl)
    # Identical head-screen prefetch lifetime to relu2_down; shared-up prefetch
    # is started by the existing dispatcher and joined by it after this returns.
    pf = False
    if k._l2pf is not None and k._l2pf.DRAFT_BYTES > 0 and bool(k._l2pf.DRAFT_SCREEN):
        pf = k.gated_prefetch([(k._l2pf.DRAFT_SCREEN[0], k._l2pf.DRAFT_BYTES)], x.device, name="shead")
        if pf is None:
            pf = k._l2pf.prefetch(k._l2pf.DRAFT_SCREEN[0], max_bytes=k._l2pf.DRAFT_BYTES)
    adaptive_compact[(8,)](h, indices, values, counts, refine,
                           I=1856, BLOCK=2048, num_warps=4, launch_pdl=spec.pdl)
    if not pkroute.run(indices, values, counts, spec.down_entry[2].planes,
                       spec.down_entry[3].planes, selected, part, refine, spec.pdl):
        raise RuntimeError("admitted pkroute contract changed after GPU work began")
    adaptive_combine[(11,)](part, weights, y, refine, M=m, H=2688,
                            ROUTED_SCALE=spec.scale, BN=256, num_warps=4,
                            launch_pdl=spec.pdl, residual_ptr=residual,
                            ADD_RES=residual is not None)
    if pf:
        k._l2pf.join(x.device)
    return y


def _run(spec, x, ids, tw, m, residual):
    if not _can(spec, x, ids, tw, m, residual):
        return None  # No GPU launch occurred in this function.
    return _core(spec, x, ids, tw, m, residual)


def maybe_run(layer, x, ids, tw, m, residual):
    spec = _ENTRIES.get(id(layer))
    return None if spec is None else _run(spec, x, ids, tw, m, residual)


def _words(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _guarded(m, device):
    t = torch.empty(m * 2688 + 128, dtype=torch.bfloat16, device=device)
    t.view(torch.uint8).fill_(0xA5)
    return t, t[64:-64].view(m, 2688)


def _guard(pair):
    return bool(torch.all(pair[0][:64].view(torch.uint8) == 0xA5).item()
                and torch.all(pair[0][-64:].view(torch.uint8) == 0xA5).item())


def _reference(spec, x, ids, tw, m, residual, topk):
    from . import nemotron_h_mtp_kernels as k
    pf = k._shared_prefetch(spec.layer, spec.weights[2], x)
    if topk == 3:
        otw = torch.empty(3, dtype=tw.dtype, device=tw.device)
        oids = torch.empty(3, dtype=ids.dtype, device=ids.device)
        k._topk_keep_kernel[(1,)](tw, ids, otw, oids, K=6, BK=8, KEEP=3,
                                  num_warps=1, launch_pdl=spec.pdl)
        tw, ids = otw, oids
    up, down, shared, sd, _, ns = spec.weights
    y = k.mtp_moe_row(x, ids, tw, up, down, shared, sd, ns, spec.scale, m, residual)
    if pf:
        k._l2pf.join(x.device)
    return y


def _full_new(spec, x, ids, tw, m, residual):
    from . import nemotron_h_mtp_kernels as k
    pf = k._shared_prefetch(spec.layer, spec.weights[2], x)
    y = _run(spec, x, ids, tw, m, residual)
    if y is None:
        raise RuntimeError("private production guard refused actual owner")
    if pf:
        k._l2pf.join(x.device)
    return y


@torch.no_grad()
def _qualify(spec):
    from . import nemotron_h_mtp_kernels as k
    device = spec.device
    gen = torch.Generator(device=device).manual_seed(2026102806)
    x = torch.empty((1, 2688), dtype=torch.bfloat16, device=device)
    ids = torch.empty(6, dtype=torch.int32, device=device)
    tw = torch.empty(6, dtype=torch.float32, device=device)
    residuals = {m: torch.empty((m, 2688), dtype=torch.bfloat16, device=device) for m in (1, 5, 8)}
    gate = k._GATE

    def change(case, topk, first=0):
        scale = (0.001, 1.0, 32.0)[case % 3]
        x.copy_((torch.randn(x.shape, device=device, generator=gen) * scale).to(x.dtype))
        x.view(torch.int16)[0, :4] = torch.tensor([0, -32768, 1, -32767], dtype=torch.int16, device=device)
        ids.copy_((torch.arange(6, device=device, dtype=torch.int32) + first) % 128)
        # Permuted low rule tests ranking; high rule retains native input order.
        tw.copy_(torch.tensor([.03, .48, .01, .32, .02, .14] if topk == 3 else [1/6] * 6,
                              dtype=torch.float32, device=device))
        for r in residuals.values():
            r.copy_((torch.randn(r.shape, device=device, generator=gen) * scale).to(r.dtype))
            r.view(torch.int16)[:, :4] = torch.tensor([0, -32768, 1, -32767], dtype=torch.int16, device=device)
        gate.zero_()
        if case == 3:
            x.zero_()
        if case == 4:
            x[0, :3] = torch.tensor([float("inf"), -float("inf"), float("nan")], device=device)
        if case == 5:
            tw.copy_(torch.tensor([float("nan"), .3, .3, .1, .1, .2], device=device))
        if case == 6:
            gate[0] = 1

    # Complete actual-weight old3/old6 outputs, all experts, residual rounds and
    # changed stop/input values. Private forced mode only for pathological3/6.
    for topk in (3, 6):
        for first in range(0, 128, 3):
            case = first // 3 % 3
            change(case, topk, first)
            m = (1, 5, 8)[first // 3 % 3]
            r = residuals[m] if first % 2 else None
            snap = tuple(t.clone() for t in (x, ids, tw, gate) + (() if r is None else (r,)))
            ref = _reference(spec, x, ids, tw, m, r, topk)
            dst = _guarded(m, device)
            _core(spec, x, ids, tw, m, r, destination=dst[1], force=int(topk == 6))
            if not (_words(ref, dst[1]) and _guard(dst)
                    and all(_words(a, b) for a, b in zip((x, ids, tw, gate) + (() if r is None else (r,)), snap))):
                raise RuntimeError(f"adaptive slot complete words/guards/read-only differ topk={topk} first={first}")
    # Actual native router outputs on private changed hidden inputs, in addition
    # to the deliberately balanced/concentrated policy cases above.
    for case in range(3):
        change(case, 3, case * 43)
        logits, _ = spec.layer.mixer.gate(x)
        actual_tw, actual_ids = spec.layer.mixer.experts.router.select_experts(
            hidden_states=x, router_logits=logits)
        actual_tw = actual_tw.reshape(-1).to(torch.float32)
        actual_ids = actual_ids.reshape(-1)
        if not _can(spec, x, actual_ids, actual_tw, 1, residuals[1]):
            raise RuntimeError("actual native router output contract unsupported")
        plan = torch.empty(1, dtype=torch.int32, device=device)
        planned_tw = torch.empty_like(actual_tw)
        planned_ids = torch.empty_like(actual_ids)
        route_plan[(1,)](actual_tw, actual_ids, planned_tw, planned_ids, plan, num_warps=1,
                        launch_pdl=spec.pdl)
        topk = 6 if int(plan.item()) else 3  # Private load-time only, never serving.
        ref = _reference(spec, x, actual_ids, actual_tw, 1, residuals[1], topk)
        dst = _guarded(1, device)
        _core(spec, x, actual_ids, actual_tw, 1, residuals[1], destination=dst[1])
        if not (_words(ref, dst[1]) and _guard(dst)):
            raise RuntimeError("adaptive actual native router complete output differs")
    metrics = {}
    for m in (1, 5, 8):
        r = residuals[m]
        # Capture the production dynamic FORCE=-1 path into eight DISTINCT outputs.
        # Never time forced-high or forced-low as an equivalent production path.
        change(1, 3)
        for _ in range(2):
            _reference(spec, x, ids, tw, m, r, 3)
            _reference(spec, x, ids, tw, m, r, 6)
            _full_new(spec, x, ids, tw, m, r)
        torch.cuda.synchronize(device)
        graphs, outputs = [], []
        for kind in (3, 6, -1):
            graph, out = torch.cuda.CUDAGraph(), []
            with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
                for _ in range(8):
                    out.append(_reference(spec, x, ids, tw, m, r, kind) if kind != -1
                               else _full_new(spec, x, ids, tw, m, r))
            if len({t.data_ptr() for t in out}) != 8:
                raise RuntimeError("adaptive graph did not retain eight distinct full outputs")
            graphs.append(graph)
            outputs.append(out)
        for case in range(7):
            for topk in (3, 6):
                change(case, topk, case * 17)
                # NaN weights choose exact parent keep3 by the production rule.
                expected = 3 if case == 5 else topk
                saved = tuple(t.clone() for t in (x, ids, tw, r, gate))
                for _ in range(2):
                    graphs[0 if expected == 3 else 1].replay()
                    graphs[2].replay()
                    if not (all(_words(a, b) for a, b in zip(outputs[0 if expected == 3 else 1], outputs[2]))
                            and all(_words(a, b) for a, b in zip((x, ids, tw, r, gate), saved))):
                        raise RuntimeError("adaptive changed-input graph words/read-only differ")
        # Complete equivalent-work warm/64MiB-pressure alternating event medians.
        pressure = torch.empty(64 << 20, dtype=torch.uint8, device=device)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ratios = []
        for topk in (3, 6):
            change(1, topk)
            for stopped in (False, True):
                gate[0] = int(stopped)
                for cold in (False, True):
                    samples = [[], []]
                    for trial in range(5):
                        for arm in ((0, 1) if trial % 2 == 0 else (1, 0)):
                            graph = graphs[2 if arm else (0 if topk == 3 else 1)]
                            elapsed = 0.0
                            for repeat in range(4):
                                if cold:
                                    pressure.fill_(trial + repeat)
                                begin.record(); graph.replay(); end.record(); end.synchronize()
                                elapsed += begin.elapsed_time(end)
                            if not elapsed > 0:
                                raise RuntimeError("adaptive invalid event interval")
                            samples[arm].append(elapsed)
                    a, b = (statistics.median(v) for v in samples)
                    ratios.append(b / a)
        if max(ratios) > 1.01:
            raise RuntimeError(f"adaptive equivalent-work whole cost refused m={m} ratios={ratios}")
        metrics[m] = ratios
    return metrics


@torch.no_grad()
def prepare_all(module):
    """Remote load after original packing AND actual screen prefetch setup."""
    if not _ENABLED or torch.cuda.is_current_stream_capturing():
        return
    from . import nemotron_h_mtp_kernels as k
    try:
        import pkroute
        if not pkroute.available():
            return
    except ImportError:
        return
    for layer in module.modules():
        if not getattr(layer, "_fast_moe", False):
            continue
        spec = _owner(layer)
        if spec is None:
            print("MTP_ADAPT_ROUTE unsupported actual owner/selected down tile; original kept", flush=True)
            continue
        original_gate = k._GATE
        k._GATE = torch.zeros(4, dtype=torch.int32, device=spec.device)
        saved_owners = tuple(t.clone() for t in spec.tensors)
        error = None
        try:
            metrics = _qualify(spec)
        except Exception as exc:
            error = exc
        finally:
            k._GATE = original_gate
        # Even a refused probe must not leave a corrupted original fallback.
        if not all(_words(a, b) for a, b in zip(spec.tensors, saved_owners)):
            raise RuntimeError("adaptive actual read-only owner weights changed")
        if error is not None:
            print(f"MTP_ADAPT_ROUTE refused: {error!r}; original kept", flush=True)
        else:
            _ENTRIES[id(layer)] = spec
            print(f"MTP_ADAPT_ROUTE admitted actual owner3/6 rows1/5/8 ratios={metrics}", flush=True)

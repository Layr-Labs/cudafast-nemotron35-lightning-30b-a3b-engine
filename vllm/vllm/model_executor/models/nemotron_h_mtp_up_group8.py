"""Optional metadata-coalesced lossless native MTP up projections.

D33's eight-value metadata load is applied to original routed/shared up kernels.
Original storage, multiply/accumulate/reduction, ReLU2, gate and PDL are retained.
All admission and graph execution occur only on the official remote worker.
"""
import os
import statistics

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as tl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout
from .nemotron_h_mtp_packed import packed_moe_up, pdl_prologue, stopped_zero

_ENABLED = os.environ.get("MTP_PACK12_UP_GROUP8", "1") != "0"
_ADMITTED = {}

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
def _verify_group8(SM, EX, BASE, RI, RAW, W, COUNTER,
                   N: tl.constexpr, B: tl.constexpr):
    layout: tl.constexpr = BlockedLayout([1, 8], [32, 1], [4, 1], [1, 0])
    group = tl.program_id(0) * B + tl.arange(0, B, layout=SliceLayout(1, layout))
    lane = tl.arange(0, 8, layout=SliceLayout(0, layout))
    index = group * 8
    full_index = index[:, None] + lane[None, :]
    mask = full_index < N
    actual = load_packed_group8(SM, EX, BASE, RI, RAW, index, lane,
                                index < N, mask).to(tl.uint16, bitcast=True)
    expected = tl.load(W + full_index, mask=mask, other=0)
    bad = tl.sum(tl.sum(((actual != expected) & mask).to(tl.int32), axis=1), axis=0)
    tl.atomic_add(COUNTER, bad, sem='relaxed')


@gluon.jit
def _up_group8(x_ptr, SM, EX, BASE, RI, RAW, SSM, SEX, SBASE, SRI, SRAW,
                   ids_ptr,h_ptr,I,K,stride_we,stride_wn,stride_wsn,gate_ptr,
                   TOPK:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,EVICT:tl.constexpr,
                   GATED:tl.constexpr=False, launch_pdl:tl.constexpr=False,
                   WN:tl.constexpr=2, WORD:tl.constexpr=False):
    if launch_pdl:
        pdl_prologue()
    # WN warps along N, 4 along K: the K layout (8 x 32 x 4), and with it every
    # row's accumulation and reduction order, does not depend on WN or BN.
    layout:tl.constexpr=BlockedLayout([1,8],[1,32],[WN,4],[1,0])
    pid_n=tl.program_id(0)
    j=tl.program_id(1)
    offs_n=pid_n*BN+tl.arange(0,BN,layout=SliceLayout(1,layout))
    offs_k=tl.arange(0,BK,layout=SliceLayout(0,layout))
    group_layout:tl.constexpr=BlockedLayout([1,8],[32,1],[4*WN,1],[1,0])
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
        gbase=e*stride_we+group_n.to(tl.int64)*stride_wn
    else:
        grows=(j-TOPK)*I+group_n
        gbase=grows.to(tl.int64)*stride_wsn
    acc=tl.full((BN,BK),0,tl.float32,layout)
    for k0 in range(0,K,BK):
        kk=k0+offs_k
        kmask=kk<K
        gk=k0+group_k
        gmask=(group_n<I)&(gk<K)
        vmask=gmask[:,None]&(gk[:,None]+lane[None,:]<K)
        if j<TOPK:
            w8=load_packed_group8(SM,EX,BASE,RI,RAW,gbase+gk,lane,gmask,vmask,EVICT=EVICT)
        else:
            w8=load_packed_group8(SSM,SEX,SBASE,SRI,SRAW,gbase+gk,lane,gmask,vmask,EVICT=EVICT)
        w=tl.convert_layout(tl.reshape(w8,(BN,BK)),layout,assert_trivial=True)
        x=tl.load(x_ptr+kk,mask=kmask,other=0.0)
        acc+=w.to(tl.float32)*x[None,:].to(tl.float32)
    h=tl.maximum(tl.sum(acc,axis=1),0.0)
    h=h*h
    tl.store(h_ptr+j*I+offs_n,h.to(h_ptr.dtype.element_ty),mask=nmask)


def _key(pu, ps, word, config):
    return (tuple(t.data_ptr() for t in pu.planes),
            tuple(t.data_ptr() for t in ps.planes), bool(word), tuple(config))


def maybe_run(x, pu, ps, ids, h, gate, gated, word, config, pdl):
    if not _ENABLED or _key(pu, ps, word, config) not in _ADMITTED:
        return False
    if (tuple(pu.shape) != (128, 1856, 2688)
            or tuple(ps.shape) != (3712, 2688)
            or x.shape != (1, 2688) or x.dtype != torch.bfloat16
            or not x.is_contiguous() or not ids.is_contiguous()
            or ids.numel() != 3 or ids.dtype not in (torch.int32, torch.int64)
            or h.shape != (5, 1856) or h.dtype != torch.bfloat16
            or not h.is_contiguous() or x.device != h.device
            or ids.device != x.device
            or any(t.device != x.device for t in (*pu.planes, *ps.planes))):
        return False
    bn, bk, wn, stages, evict = config
    _up_group8[(triton.cdiv(1856, bn), 5)](
        x, *pu.planes, *ps.planes, ids, h, 1856, 2688,
        1856 * 2688, 2688, 2688, gate, TOPK=3, BN=bn, BK=bk,
        EVICT=evict, GATED=gated, WN=wn, WORD=word,
        num_warps=4 * wn, num_stages=stages, launch_pdl=pdl,
    )
    return True


@torch.no_grad()
def prepare(pu, ps, original_up, original_shared):
    """Private remote load: all real bits, full up, replay, cold up+consumer.

    The original weights are still available here, before their existing free.
    Only immutable packed weights/config remain; no request values are cached.
    """
    from .nemotron_h_mtp_kernels import (
        _PK_UP_BN, _PK_UP_WN, _MOE_UP_BK, _MOE_UP_STAGES,
        _EVICT_W, mtp_pdl,
    )
    from .nemotron_h_mtp_packed import word_load_enabled
    from .nemotron_h_mtp_compact import _compact
    if (not _ENABLED or tuple(pu.shape) != (128, 1856, 2688)
            or tuple(ps.shape) != (3712, 2688)
            or torch.cuda.is_current_stream_capturing()):
        return False
    config = (_PK_UP_BN, _MOE_UP_BK, _PK_UP_WN, _MOE_UP_STAGES, _EVICT_W)
    if config[:4] != (2, 1024, 2, 3):
        return False
    device = original_up.device
    word = word_load_enabled(device)
    key = _key(pu, ps, word, config)
    if key in _ADMITTED:
        return True
    bn, bk, wn, stages, evict = config
    gate = torch.zeros(3, dtype=torch.int32, device=device)
    pdl = mtp_pdl()

    def launch(grouped, x, ids, h):
        kernel = _up_group8 if grouped else packed_moe_up
        kernel[(triton.cdiv(1856, bn), ids.numel() + 2)](
            x, *pu.planes, *ps.planes, ids, h, 1856, 2688,
            1856 * 2688, 2688, 2688, gate, TOPK=ids.numel(), BN=bn, BK=bk,
            EVICT=evict, GATED=True, WN=wn, WORD=word,
            num_warps=4 * wn, num_stages=stages, launch_pdl=pdl,
        )

    try:
        counter = torch.zeros(1, dtype=torch.int32, device=device)
        for packed, original in ((pu, original_up), (ps, original_shared)):
            counter.zero_()
            _verify_group8[(triton.cdiv(original.numel(), 4096),)](
                *packed.planes, original.view(torch.uint16), counter,
                original.numel(), B=512, num_warps=4,
            )
            if int(counter.item()) != 0:
                raise RuntimeError("UP_GROUP8 differs from original BF16 weight bits")
        gen = torch.Generator(device=device).manual_seed(851934)
        probes = [(torch.randn((1, 2688), device=device, generator=gen) * scale)
                  .to(torch.bfloat16) for scale in (0.001, 1., 32.)]
        edge = torch.zeros((1, 2688), dtype=torch.bfloat16, device=device)
        edge.view(torch.int16)[0, :8] = torch.tensor(
            [0, -32768, 1, -32767, 128, -32640, 16256, -16512],
            dtype=torch.int16, device=device)
        probes.extend((edge, torch.zeros_like(edge)))
        nonfinite = probes[1].clone()
        nonfinite[0, :3] = torch.tensor(
            [float("inf"), -float("inf"), float("nan")],
            dtype=torch.bfloat16, device=device)
        probes.append(nonfinite)
        # Every actual routed/shared row; this is private synthetic routing.
        all_ids = torch.arange(128, dtype=torch.int32, device=device)
        old_all = torch.empty((130, 1856), dtype=torch.bfloat16, device=device)
        new_all = torch.empty_like(old_all)
        for x in probes:
            gate.zero_()
            launch(False, x, all_ids, old_all)
            launch(True, x, all_ids, new_all)
            if not torch.equal(old_all.view(torch.uint8), new_all.view(torch.uint8)):
                raise RuntimeError("UP_GROUP8 complete 128+2 up outputs differ")
        gate[0] = 1
        launch(False, probes[1], all_ids, old_all)
        launch(True, probes[1], all_ids, new_all)
        if not torch.equal(old_all.view(torch.uint8), new_all.view(torch.uint8)):
            raise RuntimeError("UP_GROUP8 stopped up differs")

        xbuf = probes[1].clone()
        idsbuf = torch.tensor([0, 63, 127], dtype=torch.int32, device=device)
        outputs = [torch.empty((5, 1856), dtype=torch.bfloat16, device=device)
                   for _ in range(2)]
        counts = [torch.empty(5, dtype=torch.int32, device=device) for _ in range(2)]
        indices = [torch.empty((5, 1856), dtype=torch.int32, device=device) for _ in range(2)]
        values = [torch.empty_like(outputs[0]) for _ in range(2)]

        def body(grouped):
            i = int(grouped)
            launch(grouped, xbuf, idsbuf, outputs[i])
            _compact[(5,)](
                outputs[i], indices[i], values[i], counts[i], I=1856, BLOCK=2048,
                num_warps=4, launch_pdl=pdl,
            )

        graphs = []
        for grouped in (False, True):
            gate.zero_()
            body(grouped)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
                body(grouped)
            torch.cuda.synchronize(device)
            graphs.append(graph)
        routes = [(0, 63, 127), (31, 64, 126), (42, 7, 42)]
        states = [torch.tensor(g, dtype=torch.int32, device=device)
                  for g in ((0, 0, 0), (0, 2, 1), (1, 2, 1))]
        for x in probes:
            xbuf.copy_(x)
            for route in routes:
                idsbuf.copy_(torch.tensor(route, dtype=torch.int32, device=device))
                for state in states:
                    for graph in graphs:
                        gate.copy_(state)
                        graph.replay()
                    if (not torch.equal(outputs[0].view(torch.uint8), outputs[1].view(torch.uint8))
                            or not torch.equal(counts[0], counts[1])):
                        raise RuntimeError("UP_GROUP8 changed-input/route/gate graph differs")
                    for slot in range(5):
                        nlive = int(counts[0][slot].item())
                        if (not torch.equal(indices[0][slot, :nlive], indices[1][slot, :nlive])
                                or not torch.equal(values[0][slot, :nlive].view(torch.uint8),
                                                   values[1][slot, :nlive].view(torch.uint8))):
                            raise RuntimeError("UP_GROUP8 compact consumer differs")
        xbuf.copy_(probes[1])
        idsbuf.copy_(torch.tensor(routes[0], dtype=torch.int32, device=device))
        flush = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = [[], []]
        for trial in range(7):
            for i in ((0, 1) if trial % 2 == 0 else (1, 0)):
                gate.zero_()
                flush.zero_()
                begin.record()
                graphs[i].replay()
                end.record()
                end.synchronize()
                times[i].append(begin.elapsed_time(end))
        admitted = statistics.median(times[1]) < 0.96 * statistics.median(times[0])
        if admitted:
            _ADMITTED[key] = (pu, ps)  # only the existing immutable packed weights
        print(f"MTP_PACK12_UP_GROUP8 admitted={admitted} "
              "allweight_bits+128experts+shared+changing_graph+consumer+cold4pct", flush=True)
        return admitted
    except Exception as exc:
        print(f"MTP_PACK12_UP_GROUP8 original up retained: {exc!r}", flush=True)
        return False

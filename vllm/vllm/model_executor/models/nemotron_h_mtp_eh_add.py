"""Optional packed EH GEMV plus original projected-embedding Add epilogue.

The GEMV still rounds to BF16 before adding the FP32 projected embedding.
No hnorm, table, head format, proposal rule or original fallback is changed.
"""
import os
import statistics

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as tl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout
from vllm.utils.torch_utils import direct_register_custom_op
from .nemotron_h_mtp_packed import (
    packed_gemv, load_packed, pdl_prologue, word_load_enabled,
)

_ENABLED = os.environ.get("MTP_EH_FUSED_ADD", "1") != "0"
_MODES = {}


@gluon.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;", "=f,f,f", [a, b],
        dtype=tl.float32, is_pure=True, pack=1,
    )


@gluon.jit
def _reference_add(part_ptr, embedding_ptr, out_ptr, N:tl.constexpr, B:tl.constexpr):
    ix=tl.program_id(0)*B+tl.arange(0,B,layout=BlockedLayout([1],[32],[4],[0]))
    part=tl.load(part_ptr+ix,mask=ix<N,other=0.0).to(tl.float32)
    embedding=tl.load(embedding_ptr+ix,mask=ix<N,other=0.0)
    result=_add_rn(part,embedding).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr+ix,result,mask=ix<N)


@gluon.jit
def _gemv_add(x_ptr, SM, EX, BASE, RI, RAW, y_ptr, N, K, stride_wn,
                gate_ptr, BN:tl.constexpr, BK:tl.constexpr, EVICT:tl.constexpr,
                GATED:tl.constexpr=False, launch_pdl:tl.constexpr=False,
                WORD:tl.constexpr=False, embedding_ptr=None):
    if launch_pdl:
        pdl_prologue()
    layout:tl.constexpr=BlockedLayout([1,8],[1,32],[2,2],[1,0])
    pid=tl.program_id(0)
    offs_n=pid*BN+tl.arange(0,BN,layout=SliceLayout(1,layout))
    offs_k=tl.arange(0,BK,layout=SliceLayout(0,layout))
    nmask=offs_n<N
    if GATED:
        if tl.load(gate_ptr)!=0:
            # Stopped pass: store the zeros the masked loads would sum to, stream nothing.
            embedding=tl.load(embedding_ptr+offs_n,mask=nmask,other=0.0)
            value=_add_rn(tl.full((BN,),0.0,tl.float32,SliceLayout(1,layout)),embedding)
            tl.store(y_ptr+offs_n,value.to(y_ptr.dtype.element_ty),mask=nmask)
            return
    wn=nmask
    acc=tl.full((BN,BK),0,tl.float32,layout)
    for k0 in range(0,K,BK):
        kk=k0+offs_k
        kmask=kk<K
        w=load_packed(SM,EX,BASE,RI,RAW,offs_n[:,None]*stride_wn+kk[None,:],wn[:,None]&kmask[None,:],EVICT=EVICT,WORD=WORD)
        x=tl.load(x_ptr+kk,mask=kmask,other=0.0)
        acc+=w.to(tl.float32)*x[None,:].to(tl.float32)
    # Preserve the original GEMV's BF16 rounding before the FP32 embedding Add.
    y=tl.sum(acc,axis=1).to(y_ptr.dtype.element_ty).to(tl.float32)
    embedding=tl.load(embedding_ptr+offs_n,mask=nmask,other=0.0)
    value=_add_rn(y,embedding)
    tl.store(y_ptr+offs_n,value.to(y_ptr.dtype.element_ty),mask=nmask)


def _key(planes, word):
    return (tuple(t.data_ptr() for t in planes), bool(word))


def _impl(x: torch.Tensor, sm: torch.Tensor, ex: torch.Tensor,
          base: torch.Tensor, ri: torch.Tensor, raw: torch.Tensor,
          embedding: torch.Tensor, n: int, k: int,
          fallback: torch.Tensor) -> torch.Tensor:
    from .nemotron_h_mtp_kernels import _gate_args, mtp_pdl
    planes = (sm, ex, base, ri, raw)
    word = word_load_enabled(x.device)
    if (_ENABLED and _key(planes, word) in _MODES and (n, k) == (2688, 2688)
            and x.shape == (1, k) and x.dtype == torch.bfloat16 and x.is_contiguous()
            and embedding.shape == (1, n) and embedding.dtype == torch.float32
            and embedding.is_contiguous() and embedding.device == x.device
            and all(t.device == x.device for t in planes)):
        gate, gated = _gate_args(x)
        out = x.new_empty((1, n))
        _gemv_add[(triton.cdiv(n, 8),)](
            x, *planes, out, n, k, k, gate, BN=8, BK=512,
            EVICT=False, GATED=gated, launch_pdl=mtp_pdl(), WORD=word,
            embedding_ptr=embedding, num_warps=4, num_stages=3,
        )
        return out
    old = torch.ops.vllm.mtp_packed_linear_keep(
        x, *planes, n, k, fallback)
    return (old.float() + embedding).to(old.dtype)


def _fake(x: torch.Tensor, sm: torch.Tensor, ex: torch.Tensor,
          base: torch.Tensor, ri: torch.Tensor, raw: torch.Tensor,
          embedding: torch.Tensor, n: int, k: int,
          fallback: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], n))


direct_register_custom_op(
    op_name="mtp_packed_linear_keep_add", op_func=_impl, fake_impl=_fake,
)


@torch.no_grad()
def prepare(packed, original):
    """Private remote comparison/replay/cold admission before production capture."""
    from .nemotron_h_mtp_kernels import mtp_pdl
    if (not _ENABLED or tuple(packed.shape) != (2688, 2688)
            or torch.cuda.is_current_stream_capturing()):
        return False
    device = original.device
    word = word_load_enabled(device)
    key = _key(packed.planes, word)
    if key in _MODES:
        return True
    gate = torch.zeros(3, dtype=torch.int32, device=device)
    pdl = mtp_pdl()
    n = k = 2688

    def launch(fused, x, embedding, output, verify_torch=False):
        if fused:
            _gemv_add[(triton.cdiv(n, 8),)](
                x, *packed.planes, output, n, k, k, gate,
                BN=8, BK=512, EVICT=False, GATED=True, WORD=word,
                launch_pdl=pdl, embedding_ptr=embedding,
                num_warps=4, num_stages=3,
            )
        else:
            old = torch.empty_like(output)
            packed_gemv[(triton.cdiv(n, 8),)](
                x, *packed.planes, old, n, k, k, gate,
                BN=8, BK=512, EVICT=False, GATED=True, WORD=word,
                launch_pdl=pdl, num_warps=4, num_stages=3,
            )
            # Model a compiled pointwise Add with one consumer launch, not
            # eager float/cast kernels plus an extra output copy.
            _reference_add[(triton.cdiv(n, 1024),)](
                old, embedding, output, N=n, B=1024, num_warps=4,
            )
            if verify_torch:
                expected = (old.float() + embedding).to(old.dtype)
                if not torch.equal(output.view(torch.uint8), expected.view(torch.uint8)):
                    raise RuntimeError("EH reference post kernel differs from torch Add")

    try:
        gen = torch.Generator(device=device).manual_seed(821777)
        probes = []
        for scale in (0.001, 1., 32.):
            x = (torch.randn((1, k), device=device, generator=gen) * scale).to(torch.bfloat16)
            e = torch.randn((1, n), device=device, generator=gen) * scale
            probes.append((x, e))
        xedge = torch.zeros((1, k), dtype=torch.bfloat16, device=device)
        xedge.view(torch.int16)[0, :8] = torch.tensor(
            [0, -32768, 1, -32767, 128, -32640, 16256, -16512],
            dtype=torch.int16, device=device)
        eedge = torch.zeros((1, n), dtype=torch.float32, device=device)
        eedge[0, :8] = torch.tensor(
            [0., -0., 1e-40, -1e-40, 1., -1., float("inf"), float("nan")], device=device)
        probes.extend(((xedge, eedge), (torch.zeros_like(xedge), eedge)))
        states = [torch.tensor(g, dtype=torch.int32, device=device)
                  for g in ((0, 0, 0), (0, 2, 1), (1, 2, 1))]
        outputs = [torch.empty((1, n), dtype=torch.bfloat16, device=device) for _ in range(2)]
        for x, e in probes:
            for state in states:
                for i in range(2):
                    gate.copy_(state)
                    launch(bool(i), x, e, outputs[i], verify_torch=i == 0)
                if not torch.equal(outputs[0].view(torch.uint8), outputs[1].view(torch.uint8)):
                    raise RuntimeError("EH fused Add changes original two-rounding bits")
        xbuf, ebuf = (t.clone() for t in probes[1])
        graphs = []
        for i in range(2):
            gate.zero_()
            launch(bool(i), xbuf, ebuf, outputs[i])
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
                launch(bool(i), xbuf, ebuf, outputs[i])
            torch.cuda.synchronize(device)
            graphs.append(graph)
        for x, e in probes:
            xbuf.copy_(x)
            ebuf.copy_(e)
            for state in states:
                for graph in graphs:
                    gate.copy_(state)
                    graph.replay()
                if not torch.equal(outputs[0].view(torch.uint8), outputs[1].view(torch.uint8)):
                    raise RuntimeError("EH fused Add changed-input/gate graph differs")
        xbuf.copy_(probes[1][0])
        ebuf.copy_(probes[1][1])
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
            _MODES[key] = packed.planes
        print(f"MTP_EH_FUSED_ADD admitted={admitted} "
              "two_rounding_bits+changing_graph+gate+cold4pct", flush=True)
        return admitted
    except Exception as exc:
        print(f"MTP_EH_FUSED_ADD original Add retained: {exc!r}", flush=True)
        return False

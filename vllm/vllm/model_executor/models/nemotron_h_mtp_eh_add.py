# SPDX-License-Identifier: Apache-2.0
"""Optional independent EH epilogue fusion, admitted only on the official GPU.

Only official remote execution may compile or run this source. Original GEMV
and GEMM arithmetic is copied exactly, with the original BF16 output round
followed by the original FP32 projected-embedding add and final BF16 round.
No admission, activation, dispatch-count or speed result is known.
"""
import os
import statistics

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op
from . import nemotron_h_mtp_kernels as original
from .nemotron_h_mtp_kernels import _load_w

_READY = {}
_ENABLED = os.environ.get("MTP_EH_ADD_FUSION", "1") != "0"

@triton.jit
def eh_add_gemv(
    x_ptr,
    w_ptr,
    y_ptr,
    embedding_ptr,
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
    # Preserve the original BF16 intermediate round before FP32 embedding add.
    rounded = y.to(y_ptr.dtype.element_ty).to(tl.float32)
    embedding = tl.load(embedding_ptr + offs_n, mask=nmask, other=0.0)
    result = (rounded + embedding).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs_n, result, mask=nmask)


@triton.jit
def eh_add_small(
    x_ptr,
    w_ptr,
    y_ptr,
    embedding_ptr,
    stride_em,
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
        (acc.to(y_ptr.dtype.element_ty).to(tl.float32) +
         tl.load(embedding_ptr + offs_m[:, None] * stride_em + offs_n[None, :],
                 mask=mmask[:, None] & nmask[None, :], other=0.0)).to(y_ptr.dtype.element_ty),
        mask=mmask[:, None] & nmask[None, :],
    )



def _launch(x, w, embedding, out, gate, gated, pdl, fused):
    from .nemotron_h_mtp_packed import EH_LAST
    m = x.shape[0]
    evict = 2 if EH_LAST else False
    if m == 1:
        kernel = eh_add_gemv if fused else original._mtp_gemv_kernel
        args = (x, w, out, embedding, 2688, 2688, w.stride(0)) if fused else (
            x, w, out, 2688, 2688, w.stride(0))
        kernel[(triton.cdiv(2688, 8),)](
            *args, BN=8, BK=512, EVICT=evict, gate_ptr=gate, GATED=gated,
            launch_pdl=pdl, num_warps=4, num_stages=3)
    else:
        kernel = eh_add_small if fused else original._mtp_small_gemm_kernel
        args = (x, w, out, embedding, embedding.stride(0), m, 2688, 2688,
                x.stride(0), w.stride(0), out.stride(0)) if fused else (
            x, w, out, m, 2688, 2688, x.stride(0), w.stride(0), out.stride(0))
        kernel[(triton.cdiv(2688, 64),)](
            *args, BM=16, BN=64, BK=128, EVICT=evict, gate_ptr=gate, GATED=gated,
            launch_pdl=pdl, num_warps=4, num_stages=4)


def _reference_projection(x: torch.Tensor, w: torch.Tensor, gate: torch.Tensor,
                          gated: bool, pdl: bool) -> torch.Tensor:
    out = x.new_empty((x.shape[0], w.shape[0]))
    _launch(x, w, None, out, gate, gated, pdl, False)
    return out


def _fake_reference(x, w, gate, gated, pdl):
    return x.new_empty((x.shape[0], w.shape[0]))


direct_register_custom_op(
    op_name="mtp_eh_reference_projection", op_func=_reference_projection,
    fake_impl=_fake_reference,
)


def _canonical_reference(x, w, embedding, gate, out, gated, pdl):
    # Compile the original canonical pointwise chain; do not time an eager
    # three-kernel cast/add/cast surrogate against a single fused kernel.
    projection = torch.ops.vllm.mtp_eh_reference_projection(x, w, gate, gated, pdl)
    out.copy_((projection.float() + embedding).to(projection.dtype))
    return out


def _supported(x, w, embedding):
    return (x.is_cuda and x.dtype == torch.bfloat16 and x.ndim == 2
            and 1 <= x.shape[0] <= 8 and x.shape[1] == 2688 and x.stride(1) == 1
            and w.dtype == torch.bfloat16 and tuple(w.shape) == (2688, 2688)
            and w.is_contiguous() and w.device == x.device
            and embedding.dtype == torch.float32 and embedding.device == x.device
            and tuple(embedding.shape) == (x.shape[0], 2688) and embedding.is_contiguous())


def _impl(x: torch.Tensor, w: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
    gate, gated = original._gate_args(x)
    pdl = original.mtp_pdl()
    entry = _READY.get((w.device.index, w.data_ptr()))
    if (not _supported(x, w, embedding) or entry is None or entry[0] is not w
            or not entry[1].get((x.shape[0], gated, pdl), False)):
        projection = torch.ops.vllm.mtp_small_linear_keep(x, w)
        return (projection.float() + embedding).to(projection.dtype)
    out = x.new_empty((x.shape[0], w.shape[0]))
    _launch(x, w, embedding, out, gate, gated, pdl, True)
    return out


def _fake_add(x, w, embedding):
    return x.new_empty((x.shape[0], w.shape[0]))


direct_register_custom_op(
    op_name="mtp_eh_add", op_func=_impl, fake_impl=_fake_add,
)


def _bytes(x):
    return x.contiguous().reshape(-1).view(torch.uint8)


def _same(x, y):
    return torch.equal(_bytes(x), _bytes(y))


def _guarded(shape, dtype, device):
    count = 1
    for value in shape:
        count *= value
    pad = 128
    owner = torch.empty(count + 2 * pad, dtype=dtype, device=device)
    integer = torch.int16 if dtype == torch.bfloat16 else torch.int32
    owner.view(integer).fill_(0x2BAD)
    return owner, owner[pad:-pad].view(shape), pad


def _guards(owner, pad):
    integer = torch.int16 if owner.dtype == torch.bfloat16 else torch.int32
    words = owner.view(integer)
    return bool(torch.all(words[:pad] == 0x2BAD)) and bool(torch.all(words[-pad:] == 0x2BAD))


@torch.no_grad()
def prepare(w, table):
    """Authored official-load-only qualification, never evaluated locally.

    Reference uses the actual original matrix kernels and a compiled canonical
    BF16-round/F32-add/BF16 chain with an explicit guarded destination. Compile
    options are local to this function; the model's compiler settings and depth
    gate are untouched. This is a private latency gate, not a whole speed claim.
    """
    from .nemotron_h_mtp_packed import EH_BF16
    if (not _ENABLED or not EH_BF16 or torch.cuda.is_current_stream_capturing()
            or os.environ.get("MTP_FAST_KERNELS", "1") == "0"
            or os.environ.get("MTP_FAST_DENSE", "1") == "0"
            or w.dtype != torch.bfloat16 or tuple(w.shape) != (2688, 2688)
            or not w.is_cuda or not w.is_contiguous()
            or table.dtype != torch.float32 or table.ndim != 2
            or table.shape[1] != 2688 or table.device != w.device
            or table.shape[0] < 8):
        return ()
    if torch.cuda.get_device_capability(w.device) != (12, 1):
        return ()
    if ((original._GEMV_BN, original._GEMV_BK, original._GEMV_WARPS, original._GEMV_STAGES)
            != (8, 512, 4, 3) or
            (original._DOT_BN, original._DOT_BK, original._DOT_WARPS, original._DOT_STAGES)
            != (64, 128, 4, 4)):
        return ()
    key = (w.device.index, w.data_ptr())
    if key in _READY:
        return tuple(m for m in range(1, 9) if _READY[key][0] is w
                     and all(_READY[key][1].get((m, gated, original.mtp_pdl()), False)
                             for gated in (False, True)))
    flags = {}
    _READY[key] = (w, flags)
    device, pdl = w.device, original.mtp_pdl()
    generator = torch.Generator(device=device).manual_seed(2026102801)
    saved_weight = w.clone()
    pressure = torch.empty(64 << 20, dtype=torch.uint8, device=device)
    # Dynamic M avoids one pointwise compiler variant per row count. Compilation
    # is performed only here, on the official remote GPU before serving capture.
    reference = torch.compile(_canonical_reference, fullgraph=True, dynamic=True,
                              options={"triton.cudagraphs": False})
    original_gate = original._GATE
    original_gate_snapshot = original_gate.clone() if original_gate is not None else None
    try:
        for m in range(1, 9):
            xb, x, xp = _guarded((m, 2688), torch.bfloat16, device)
            eb, embedding, ep = _guarded((m, 2688), torch.float32, device)
            gb, gate, gp = _guarded((3,), torch.int32, device)
            output = [[_guarded((m, 2688), torch.bfloat16, device)
                       for _ in range(8)] for _ in (0, 1)]
            metadata = ((xb, xp), (eb, ep), (gb, gp))
            for gated in (False, True):
                admission = (m, gated, pdl)
                flags[admission] = False
                timings, ratios = [], []

                def case(index):
                    gate.fill_(index % 2 if gated else 0)
                    scale = (0., .001, 1., 32.)[index % 4]
                    x.copy_((torch.randn(x.shape, device=device, generator=generator) * scale).to(x.dtype))
                    ids = ((torch.arange(m, device=device, dtype=torch.int64) * 31 + index * 17)
                           % table.shape[0])
                    embedding.copy_(table.index_select(0, ids))
                    if index in (4, 5, 6, 7):
                        edges = torch.tensor([0, -32768, 1, -32767, 128, -32640,
                                              16256, -16512, 32640, -128, 32704, -64],
                                             dtype=torch.int16, device=device)
                        x.view(torch.int16)[:, :edges.numel()].copy_(edges[None, :].expand(m, -1))
                    if index in (8, 9):
                        projection = _reference_projection(x, w, gate, gated, pdl)
                        embedding.copy_(-projection.float())
                        embedding[:, ::2].add_(.00390625 if index == 8 else -.00390625)
                    if index == 10:
                        x.zero_()
                        x[:, ::2] = -0.
                        embedding.zero_()
                        embedding[:, 1::2] = -0.

                def run(which, destination):
                    if which:
                        _launch(x, w, embedding, destination, gate, gated, pdl, True)
                    else:
                        reference(x, w, embedding, gate, destination, gated, pdl)

                def compare(graphs=None):
                    saved = [owner.clone() for owner, _ in metadata]
                    for outputs in output:
                        for owner, view, pad in outputs:
                            owner.view(torch.int16).fill_(0x2BAD)
                    if graphs:
                        for which in (0, 1):
                            graphs[which].replay()
                    else:
                        for which in (0, 1):
                            for owner, view, pad in output[which]:
                                run(which, view)
                    for old_item, new_item in zip(*output):
                        if (not _same(old_item[0], new_item[0])
                                or not _guards(old_item[0], old_item[2])
                                or not _guards(new_item[0], new_item[2])):
                            raise ValueError("EH add fusion full-output/guard mismatch")
                    if (any(not _same(owner, before) for (owner, _), before in zip(metadata, saved))
                            or not all(_guards(owner, pad) for owner, pad in metadata)):
                        raise ValueError("EH add fusion input/metadata modified")

                try:
                    for index in range(11):
                        case(index)
                        compare()
                    case(2)
                    graphs = []
                    for which in (0, 1):
                        for _ in range(2):
                            run(which, output[which][0][1])
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
                            for owner, destination, pad in output[which]:
                                run(which, destination)
                        graphs.append(graph)
                    for index in (3, 0, 9, 4, 10, 7, 8, 2):
                        case(index)
                        compare(graphs)
                        compare(graphs)
                    for index in (1, 2, 3, 10):
                        case(index)
                        compare(graphs)
                        for cache_pressure in (False, True):
                            samples = [[], []]
                            for repeat in range(5):
                                for which in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                                    if cache_pressure:
                                        pressure.fill_(repeat + 1)
                                    else:
                                        for _ in range(8):
                                            graphs[which].replay()
                                    start = torch.cuda.Event(enable_timing=True)
                                    end = torch.cuda.Event(enable_timing=True)
                                    repetitions = 1 if cache_pressure else 16
                                    start.record()
                                    for _ in range(repetitions):
                                        graphs[which].replay()
                                    end.record()
                                    end.synchronize()
                                    elapsed = start.elapsed_time(end) / (8 * repetitions)
                                    if not elapsed > 0:
                                        raise ValueError("EH add timing invalid")
                                    samples[which].append(elapsed)
                            a, b = (statistics.median(values) for values in samples)
                            timings.append((a, b))
                            ratios.append(b / a)
                            compare(graphs)
                    flags[admission] = (statistics.median(ratios) < .99 and
                                        sum(b for a, b in timings) / sum(a for a, b in timings) < .99
                                        and max(ratios) < 1.01)
                    if not _same(w, saved_weight):
                        raise ValueError("EH add fusion modified immutable weights")
                except Exception as exc:
                    flags[admission] = False
                    print(f"MTP_EH_ADD row={m} gated={gated} refused={exc!r}", flush=True)
        if not _same(w, saved_weight):
            flags.clear()
            raise ValueError("EH add fusion modified immutable weight owner")
        if (original._GATE is not original_gate or
                (original_gate is not None and not _same(original_gate, original_gate_snapshot))):
            flags.clear()
            raise ValueError("EH add fusion changed original depth gate")
        print(f"MTP_EH_ADD admitted={sum(flags.values())}/{len(flags)} "
              "fullwords+guards+8outputs+changedgraphs+compiledoldpipeline+warmpressure", flush=True)
        return tuple(m for m in range(1, 9)
                     if all(flags.get((m, gated, pdl), False) for gated in (False, True)))
    except Exception as exc:
        flags.clear()
        print(f"MTP_EH_ADD original path retained: {exc!r}", flush=True)
        return ()

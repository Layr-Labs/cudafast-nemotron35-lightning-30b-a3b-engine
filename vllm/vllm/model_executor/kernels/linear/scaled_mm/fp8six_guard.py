# SPDX-License-Identifier: Apache-2.0
"""Private remote admission for the optional six-bit FP8 weight encoding."""

import os
import statistics

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

from .fp8ws import maybe_gemm as fp8ws_maybe_gemm
from vllm import _custom_ops as ops
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)
SHAPES = {(10304, 2688), (2688, 4096)}
_synthetic_checked = set()


@triton.jit
def _publish(src, dst, N: tl.constexpr, B: tl.constexpr,
             launch_pdl: tl.constexpr = False):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    ix = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(dst + ix, tl.load(src + ix, mask=ix < N, other=0), mask=ix < N)


def _same(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _check_all_bytes(fp8six, packed):
    # Exercise every raw FP8 bit pattern on the actual codec and device. Spread
    # fixtures over the packed tensor so the bounded exception window can admit
    # the synthetic case; the real tensor is checked independently by prepare.
    key = (packed.device.index, *packed.shape)
    if key in _synthetic_checked:
        return True
    source = torch.zeros_like(packed)
    offsets = torch.arange(256, device=packed.device, dtype=torch.long) * 4096
    source.view(-1)[offsets] = torch.arange(
        256, device=packed.device, dtype=torch.int32).to(torch.uint8)
    encoded = fp8six.encode(source)
    if encoded is None or not torch.equal(fp8six.decode(encoded[0], *packed.shape), source):
        return False
    _synthetic_checked.add(key)
    return True


@torch.no_grad()
def prepare(packed, reference, sa, sb):
    """Return an admitted six-bit buffer or None; all fixtures are private."""
    if (os.environ.get("FP8SIX", "1") == "0" or reference is None
            or sa is None or sb is None or sa.dtype != torch.float32
            or sb.dtype != torch.float32 or sa.numel() != 1 or sb.numel() != 1
            or not sa.is_contiguous() or not sb.is_contiguous()
            or torch.cuda.is_current_stream_capturing()):
        return None
    fp8c = None
    ref_pdl = None
    try:
        import fp8c
        import fp8six

        ref_pdl = fp8c._pdl
        n, k = packed.shape
        candidate = fp8six.prepare(packed)
        if (candidate is None or candidate.numel() >= reference.numel()
                or not _check_all_bytes(fp8six, packed)):
            return None
        candidate_pdl = fp8six._pdl
        # Snapshot source and both immutable encodings; no probe uses request data.
        owners = [(t, t.clone()) for t in (packed, reference, candidate, sa, sb)]
        gen = torch.Generator(device=packed.device).manual_seed(318769)
        pressure = torch.empty(64 << 20, dtype=torch.uint8, device=packed.device)
        timings, ratios = [], []
        for m in (1, 2, 3, 4, 5, 6, 7, 8):
            # Prefix/suffix are part of the owner; offset is 16-byte aligned.
            input_owner = torch.full((32 + m * k + 32,), 91, dtype=torch.uint8,
                                     device=packed.device)
            x = input_owner[32:-32].view(torch.float8_e4m3fn).view(m, k)
            input_bits = (torch.randn((m, k), generator=gen, device=packed.device,
                                      dtype=torch.float32) * .5).to(torch.float8_e4m3fn)
            x.copy_(input_bits)
            saved = input_owner.clone()
            out_owner = torch.full((32 + m * n + 32,), -7., device=packed.device,
                                   dtype=torch.bfloat16)
            out = out_owner[32:-32].view(m, n)
            before_out = out_owner.clone()
            old = fp8c.gemm(x, reference, sa, sb, n, k)
            new = fp8six.gemm(x, candidate, sa, sb, n, k, out=out)
            if (not _same(old, new) or not _same(input_owner, saved)
                    or not _same(out_owner[:32], before_out[:32])
                    or not _same(out_owner[-32:], before_out[-32:])):
                return None

            # Both graphs have an early-signaling producer and consume changing
            # data at fixed addresses, exercising the PDL dependency boundary.
            src = x.view(torch.uint8).clone()
            dst_owner = torch.full((32 + m * k + 32,), 73, dtype=torch.uint8,
                                   device=packed.device)
            dst = dst_owner[32:-32].view(m, k)
            a = dst.view(torch.float8_e4m3fn)
            live_sa, live_sb = sa.clone(), sb.clone()
            graphs, single_graphs, results, single_outputs = [], [], [], []
            graph_guards = [torch.full((32 + m * n + 32,), -11., device=packed.device,
                                       dtype=torch.bfloat16) for _ in range(8)]
            for which in (0, 1):
                lib, comp = (fp8c, reference) if which == 0 else (fp8six, candidate)
                _publish[(triton.cdiv(src.numel(), 1024),)](
                    src, dst, N=src.numel(), B=1024, launch_pdl=True)
                lib.gemm(a, comp, live_sa, live_sb, n, k)
                torch.cuda.synchronize(packed.device)
                graph, outputs = torch.cuda.CUDAGraph(), []
                with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
                    _publish[(triton.cdiv(src.numel(), 1024),)](
                        src, dst, N=src.numel(), B=1024, launch_pdl=True)
                    for index in range(8):
                        if which:
                            guarded = graph_guards[index][32:-32].view(m, n)
                            outputs.append(lib.gemm(a, comp, live_sa, live_sb, n, k,
                                                    out=guarded))
                        else:
                            outputs.append(lib.gemm(a, comp, live_sa, live_sb, n, k))
                graphs.append(graph)
                results.append(outputs)
                single = torch.cuda.CUDAGraph()
                with torch.cuda.graph(single, pool=torch.cuda.graph_pool_handle()):
                    _publish[(triton.cdiv(src.numel(), 1024),)](
                        src, dst, N=src.numel(), B=1024, launch_pdl=True)
                    if which:
                        single_output = lib.gemm(a, comp, live_sa, live_sb, n, k,
                                                 out=graph_guards[0][32:-32].view(m, n))
                    else:
                        single_output = lib.gemm(a, comp, live_sa, live_sb, n, k)
                single_graphs.append(single)
                single_outputs.append(single_output)
            for scale in (.001, 1., 30., .1):
                probe = (torch.randn((m, k), generator=gen, device=packed.device,
                                     dtype=torch.float32) * scale).to(torch.float8_e4m3fn)
                src.copy_(probe.view(torch.uint8))
                live_sa.copy_(sa * scale)
                live_sb.copy_(sb / scale)
                graphs[0].replay()
                graphs[1].replay()
                single_graphs[0].replay()
                single_graphs[1].replay()
                if (not _same(dst, src)
                        or not _same(*single_outputs)
                        or not bool(torch.all(dst_owner[:32] == 73))
                        or not bool(torch.all(dst_owner[-32:] == 73))
                        or any(not bool(torch.all(g[:32] == -11.))
                               or not bool(torch.all(g[-32:] == -11.))
                               for g in graph_guards)
                        or any(not _same(old_y, new_y)
                               for old_y, new_y in zip(*results))):
                    return None
            if m in (1, 5, 6, 8):
                live_sa.copy_(sa)
                live_sb.copy_(sb)
                for cold in (False, True):
                    samples = [[], []]
                    for repeat in range(5):
                        for which in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                            if cold:
                                pressure.fill_(repeat + 1)
                            else:
                                for _ in range(4):
                                    graphs[which].replay()
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            reps = 1 if cold else 12
                            start.record()
                            for _ in range(reps):
                                (single_graphs if cold else graphs)[which].replay()
                            end.record()
                            end.synchronize()
                            samples[which].append(start.elapsed_time(end) / reps)
                    a_time, b_time = (statistics.median(s) for s in samples)
                    if not a_time > 0:
                        return None
                    timings.append((a_time, b_time))
                    ratios.append(b_time / a_time)
        if (fp8c._pdl != ref_pdl or fp8six._pdl != candidate_pdl
                or not all(_same(a, b) for a, b in owners) or not ratios
                or statistics.median(ratios) >= .99 or max(ratios) >= 1.01
                or sum(b for a, b in timings) / sum(a for a, b in timings) >= .99):
            return None
        logger.info("fp8six: admitted %dx%d, encoded %d versus %d bytes", n, k,
                    candidate.numel(), reference.numel())
        return candidate
    except Exception as exc:  # noqa: BLE001 - preserve the checked original
        logger.warning("fp8six: admission failed (%r); original fp8c retained", exc)
        return None
    finally:
        if fp8c is not None and ref_pdl is not None:
            fp8c._pdl = ref_pdl


def _scaled_mm(a, b, comp, sa, sb):
    k, n = b.shape
    if (1 <= a.shape[0] <= 8 and (n, k) in SHAPES
            and a.dtype == b.dtype == torch.float8_e4m3fn and comp.dtype == torch.uint8
            and a.is_contiguous() and a.data_ptr() % 16 == 0
            and a.device == b.device == comp.device == sa.device == sb.device
            and sa.dtype == sb.dtype == torch.float32 and sa.numel() == sb.numel() == 1
            and sa.is_contiguous() and sb.is_contiguous() and a.shape[1] == k):
        import fp8six

        return fp8six.gemm(a, comp, sa, sb, n, k)
    out = fp8ws_maybe_gemm(a, b, sa, sb)
    if out is not None:
        return out
    return ops.cutlass_scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)


def _scaled_mm_fake(a, b, comp, sa, sb):
    return torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.bfloat16)


direct_register_custom_op("fp8six_scaled_mm", _scaled_mm, fake_impl=_scaled_mm_fake)

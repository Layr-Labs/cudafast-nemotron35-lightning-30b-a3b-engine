# SPDX-License-Identifier: Apache-2.0
"""Official-remote-only admission on private data and the actual loaded weights.

These checks are part of engine loading. They have not been run locally.
The timed private chain includes the unchanged prefetch byte budget, side-stream
fork/join, and the complete GEMV. It cannot establish a production overlap,
kernel hit rate, sealed-output correctness, or whole-benchmark improvement.
"""

import ctypes
import logging
import statistics

import torch

LOG = logging.getLogger(__name__)
_checked = set()


def _same(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _reference(fp8c, a, comp, sa, sb, out, n, k, ring, pdl):
    # Deliberately bypass fp8c.gemm's eager PDL retry/global-mode mutation.
    rc = fp8c._lib.fp8c_run(
        a.data_ptr(), comp.data_ptr(), sa.data_ptr(), sb.data_ptr(), out.data_ptr(),
        a.shape[0], n, k, comp.shape[1], int(pdl), ring,
        torch.cuda.current_stream(a.device).cuda_stream)
    if rc:
        raise RuntimeError(f"FP8MIX private original launch refused {rc}")


def _decoded(mx, packet, n, k, keep):
    owner = torch.full((64 + n * k,), 179, dtype=torch.uint8, device=packet.device)
    keep.append(owner)
    out = owner[32:-32].view(n, k)
    rc = mx._lib.fp8mix_decode(packet.data_ptr(), out.data_ptr(), n, k,
                               torch.cuda.current_stream(packet.device).cuda_stream)
    if rc:
        raise RuntimeError(f"FP8MIX private decode failed {rc}")
    if not bool(torch.all(owner[:32] == 179)) or not bool(torch.all(owner[-32:] == 179)):
        raise RuntimeError("FP8MIX private decoder guard changed")
    return out


def _synthetic(mx, fp8c, shape, device, keep):
    key = (device.index, *shape, id(mx._lib))
    if key in _checked:
        return True
    n, k = shape
    t, c = n // 32, k // 128
    r = torch.zeros((t, c, 4, 32, 8, 4), dtype=torch.uint8, device=device)
    keep.append(r)
    # Exact exception counts yield all q0 residues, zero-length records,
    # word/warp-window edges and the final record of a tile. Background zero
    # dominates each column's exponent window; byte0x78 is outside it.
    desired = (0, 1, 1, 1, 1, 7, 31, 32, 63, 124)
    records = r.view(t, c * 4, 1024)
    for i, count in enumerate(desired):
        records[1, i, :count] = 0x78
    records[1, -1, :124] = 0x78
    # All raw E4M3FN bytes including signed zero/NaNs/infinities' code points.
    offsets = torch.arange(256, dtype=torch.long, device=device) * 4096
    r.view(-1)[offsets] = torch.arange(256, dtype=torch.int32, device=device).to(torch.uint8)
    source = r.view(t, c, 4, 32, 4, 2, 4).permute(0, 1, 4, 2, 3, 5, 6).reshape(n, k)
    keep.append(source)
    old = fp8c.encode(source)
    if old is None:
        return False
    keep.append(old[0])
    for mode in (6, 7, "alternating"):
        result = mx.encode(source, old[0], force=mode)
        if result is None:
            return False
        keep.append(result[0])
        if not _same(_decoded(mx, result[0], n, k, keep), source):
            return False
    _checked.add(key)
    return True


def _guarded(m, width, dtype, device, fill, keep):
    owner = torch.full((64 + m * width,), fill, dtype=dtype, device=device)
    keep.append(owner)
    return owner, owner[32:-32].view(m, width)


def _publish(mx, src, sa, sb, dst, da, db, pdl):
    rc = mx._lib.fp8mix_publish_run(
        src.data_ptr(), sa.data_ptr(), sb.data_ptr(), dst.data_ptr(), da.data_ptr(), db.data_ptr(),
        src.numel(), int(pdl), torch.cuda.current_stream(src.device).cuda_stream)
    if rc:
        raise RuntimeError(f"FP8MIX private early producer failed {rc}")


def _prefetch(l2pf, tensor, limit, extra, side):
    current = torch.cuda.current_stream(tensor.device)
    side.wait_stream(current)
    ranges = [(tensor, limit)] if extra is None else [extra, (tensor, limit)]
    with torch.cuda.stream(side):
        if len(ranges) == 1:
            storage = tensor.untyped_storage()
            rc = l2pf._lib.l2pf_prefetch(storage.data_ptr(), min(storage.nbytes(), limit),
                                        l2pf.CHUNK, side.cuda_stream)
        else:
            args = []
            for value, bound in ranges:
                storage = value.untyped_storage()
                args.extend((storage.data_ptr(), min(storage.nbytes(), bound)))
            if hasattr(l2pf._lib, "l2pf_prefetch3"):
                rc = l2pf._lib.l2pf_prefetch3(*args, 0, 0, l2pf.CHUNK, side.cuda_stream)
            else:
                rc = 0
                for value, bound in ranges:
                    storage = value.untyped_storage()
                    status = l2pf._lib.l2pf_prefetch(storage.data_ptr(), min(storage.nbytes(), bound),
                                                   l2pf.CHUNK, side.cuda_stream)
                    if status:
                        rc = status
                        break
    if rc:
        raise RuntimeError(f"FP8MIX private prefetch refused {rc}")
    current.wait_stream(side)


def prefetch_key(l2pf):
    return (l2pf.ENABLED, id(l2pf._lib), l2pf.CHUNK, l2pf.MAMBA_BYTES,
            l2pf.MOE_BYTES, l2pf._GATE_PREFETCH)


def _inner(mx, fp8c, l2pf, comp, b, sa, sb, ring, name, side, keep):
    n, k = b.shape[1], b.shape[0]
    device = comp.device
    if (not l2pf.ENABLED or l2pf._lib is None or fp8c._lib is None
            or not _synthetic(mx, fp8c, (n, k), device, keep)):
        return None
    packed = fp8c.decode(comp, n, k)
    keep.append(packed)
    # Verify the old exact byte pack against the actual unmodified logical B,
    # without cloning B or making another full contiguous transpose.
    original_view = (packed.view(torch.int32).view(n // 32, k // 32, 4, 8, 4, 2)
                     .permute(0, 2, 3, 1, 5, 4))
    if not torch.equal(original_view, b.t().view(torch.int32).view(n // 32, 4, 8, k // 32, 2, 4)):
        return None
    result = mx.encode(packed, comp)
    if result is None:
        return None
    packet, info = result
    keep.append(packet)
    if not _same(_decoded(mx, packet, n, k, keep), packed):
        return None
    modes = (False, True) if bool(fp8c._pdl) else (False,)
    if True in modes and not mx._lib.fp8mix_has_pdl():
        return None
    live_sa, live_sb = sa.clone(), sb.clone()
    keep.extend((live_sa, live_sb))
    # No production selector is published during any probe.
    timings, ratios = [], []
    pressure = torch.empty(64 << 20, dtype=torch.uint8, device=device)
    keep.append(pressure)
    extra = None
    if k == 4096:
        gate, reserved = l2pf.gate_weight(name.rsplit(".out_proj", 1)[0], l2pf.MAMBA_BYTES)
        if gate is not None and reserved > 0:
            keep.append(gate)
            extra = (gate, reserved)
        budgets = (l2pf.MAMBA_BYTES - reserved,)
    else:
        # Same fixed range budgets as next-layer-MoE and first-target windows.
        budgets = (l2pf.MOE_BYTES, 20 << 20)
    if any(bound <= 0 for bound in budgets):
        return None
    for m in range(1, 9):
        a_owner, a_bytes = _guarded(m, k, torch.uint8, device, 91, keep)
        a = a_bytes.view(torch.float8_e4m3fn)
        src = torch.empty_like(a_bytes)
        keep.append(src)
        old_owner, old_out = _guarded(m, n, torch.bfloat16, device, -11., keep)
        new_owner, new_out = _guarded(m, n, torch.bfloat16, device, -11., keep)
        attrs = (ctypes.c_int * 4)()
        for pdl in modes:
            mx._run(a, packet, live_sa, live_sb, new_out, n, k, ring, pdl, attrs=attrs)
            if attrs[1] != 0 or attrs[3] < 1 or attrs[0] > 128:
                return None
            # Raw-byte A includes every FP8 pattern. Every actual M1..8 is
            # checked, not only common M5/6, and float32 special scale values
            # are compared after the exact original scaleB->scaleA epilogue.
            pattern = torch.arange(m * k, dtype=torch.int32, device=device)
            keep.append(pattern)
            for j, scale in enumerate((None, 0., -0., 1., float("inf"), float("nan"),
                                       1e-45, 3.4e38, "all_raw")):
                if scale == "all_raw":
                    bits = (pattern * 17 + 31 * j) & 255
                else:
                    bits = ((pattern * 17 + 31 * j) % 112) + (((pattern // 112) & 1) * 128)
                src.copy_(bits.to(torch.uint8).view(m, k))
                a_bytes.copy_(src)
                if scale is None or scale == "all_raw":
                    live_sa.copy_(sa)
                    live_sb.copy_(sb)
                else:
                    live_sa.fill_(scale)
                    live_sb.fill_(-1. if j & 1 else 1.)
                saved = a_owner.clone()
                keep.append(saved)
                _reference(fp8c, a, comp, live_sa, live_sb, old_out, n, k, ring, pdl)
                mx._run(a, packet, live_sa, live_sb, new_out, n, k, ring, pdl)
                if not _same(old_out, new_out) or not _same(saved, a_owner):
                    return None
            # Independent early-signaling producer: no L2 prefetch join hides
            # the dependency, and A plus both scales change between replays.
            source_sa, source_sb = sa.clone(), sb.clone()
            keep.extend((source_sa, source_sb))
            graphs, graph_inputs = [], []
            for which in (0, 1):
                graph_a_owner, graph_bytes = _guarded(m, k, torch.uint8, device, 91, keep)
                graph_a = graph_bytes.view(torch.float8_e4m3fn)
                graph_sa, graph_sb = sa.clone(), sb.clone()
                keep.extend((graph_sa, graph_sb))
                graph_inputs.append((graph_a_owner, graph_bytes, graph_sa, graph_sb))
                graph = torch.cuda.CUDAGraph()
                keep.append(graph)
                def invoke(which=which, graph_a=graph_a, graph_bytes=graph_bytes,
                           graph_sa=graph_sa, graph_sb=graph_sb):
                    _publish(mx, src, source_sa, source_sb, graph_bytes, graph_sa, graph_sb, pdl)
                    if which:
                        mx._run(graph_a, packet, graph_sa, graph_sb, new_out, n, k, ring, pdl)
                    else:
                        _reference(fp8c, graph_a, comp, graph_sa, graph_sb, old_out, n, k, ring, pdl)
                invoke()
                torch.cuda.current_stream(device).synchronize()
                with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
                    invoke()
                graphs.append(graph)
            for j in range(4):
                # Finite current inputs: an all-NaN expected output would hide
                # a consumer reading a poisoned stale activation or scale.
                bits = ((pattern * (j * 2 + 1) + j * 17) % 112) + (((pattern // 112) & 1) * 128)
                src.copy_(bits.to(torch.uint8).view(m, k))
                source_sa.copy_(sa * (j + 1))
                source_sb.copy_(sb / (j + 1))
                for _, dst_bytes, dst_sa, dst_sb in graph_inputs:
                    dst_bytes.fill_(127)
                    dst_sa.fill_(float("nan"))
                    dst_sb.fill_(float("nan"))
                # Independent owners plus candidate-first order: the reference
                # producer cannot prime the candidate's destination values.
                graphs[1].replay()
                graphs[0].replay()
                if not bool(torch.isfinite(old_out).all()) or not _same(old_out, new_out):
                    return None
                for owner, dst_bytes, dst_sa, dst_sb in graph_inputs:
                    if (not _same(dst_bytes, src) or not _same(source_sa, dst_sa)
                            or not _same(source_sb, dst_sb)
                            or not bool(torch.all(owner[:32] == 91))
                            or not bool(torch.all(owner[-32:] == 91))):
                        return None
        if (not bool(torch.all(a_owner[:32] == 91)) or not bool(torch.all(a_owner[-32:] == 91))
                or any(not bool(torch.all(o[:32] == -11.)) or not bool(torch.all(o[-32:] == -11.))
                       for o in (old_owner, new_owner))):
            return None
        if m not in (1, 5, 6, 8):
            continue
        a_bytes.copy_(((pattern % 65) + 32).to(torch.uint8).view(m, k))
        live_sa.copy_(sa)
        live_sb.copy_(sb)
        for budget in budgets:
            full_graphs = []
            for which in (0, 1):
                graph = torch.cuda.CUDAGraph()
                keep.append(graph)
                def chain(which=which):
                    _prefetch(l2pf, packet if which else comp, budget, extra, side)
                    if which:
                        mx._run(a, packet, live_sa, live_sb, new_out, n, k, ring, bool(fp8c._pdl))
                    else:
                        _reference(fp8c, a, comp, live_sa, live_sb, old_out, n, k, ring, bool(fp8c._pdl))
                chain()
                torch.cuda.current_stream(device).synchronize()
                side.synchronize()
                with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
                    chain()
                full_graphs.append(graph)
            full_graphs[0].replay()
            full_graphs[1].replay()
            if not _same(old_out, new_out):
                return None
            for cold in (False, True):
                samples = [[], []]
                for repeat in range(5):
                    for which in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                        if cold:
                            pressure.fill_(repeat + 1)
                        else:
                            for _ in range(3):
                                full_graphs[which].replay()
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        keep.extend((start, end))
                        start.record()
                        for _ in range(1 if cold else 8):
                            full_graphs[which].replay()
                        end.record()
                        end.synchronize()
                        samples[which].append(start.elapsed_time(end) / (1 if cold else 8))
                old_ms, new_ms = (statistics.median(x) for x in samples)
                if old_ms <= 0:
                    return None
                timings.append((old_ms, new_ms))
                ratios.append(new_ms / old_ms)
    if (not ratios or statistics.median(ratios) >= .99 or max(ratios) >= 1.
            or sum(b for a, b in timings) >= .99 * sum(a for a, b in timings)):
        return None
    # Check both original and selected immutable byte packs once more after
    # every native graph and prefetch probe, without another full B clone.
    after = fp8c.decode(comp, n, k)
    keep.append(after)
    if (not _same(after, packed)
            or not _same(_decoded(mx, packet, n, k, keep), packed)):
        return None
    info.update(private_median_ratio=statistics.median(ratios), private_max_ratio=max(ratios),
                ring=ring, pdl_modes=modes)
    return packet, info, modes


@torch.no_grad()
def prepare(comp, b, sa, sb, ring, name):
    import fp8c
    import l2pf
    import fp8mix as mx

    caller = torch.cuda.current_stream(comp.device)
    private = torch.cuda.Stream(device=comp.device)
    side = torch.cuda.Stream(device=comp.device)
    # Actual owners, probes, graph objects and native-only side-stream sources
    # stay in this frame until both streams have drained, even on exceptions.
    keep = [comp, b, sa, sb, private, side]
    private.wait_stream(caller)
    saved_mode = bool(fp8c._pdl)
    result = None
    with torch.cuda.stream(private):
        try:
            result = _inner(mx, fp8c, l2pf, comp, b, sa, sb, ring, name, side, keep)
        except Exception as exc:
            LOG.warning("FP8MIX private refusal %s (%r); old package retained", name, exc)
        finally:
            # If either drain is unhealthy, propagate it; do not retry a
            # potentially written operation or resume a damaged CUDA context.
            try:
                private.synchronize()
            finally:
                side.synchronize()
    caller.wait_stream(private)
    if bool(fp8c._pdl) != saved_mode:
        return None
    return result

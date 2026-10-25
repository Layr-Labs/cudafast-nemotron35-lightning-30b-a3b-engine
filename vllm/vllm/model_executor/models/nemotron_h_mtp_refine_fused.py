# SPDX-License-Identifier: Apache-2.0
"""Private, load-qualified rest-dot/TopK fusion. No request state is cached."""

import copy
import statistics

import torch

from vllm.triton_utils import tl, triton

from .nemotron_h_mtp_kernels import _gate_args, depth_gate_init, mtp_pdl
from .nemotron_h_mtp_screen import (
    CASCADE_PER, SCREEN_CHUNK, SCREEN_PER, _e2m1_pair, _topk_key,
)


@triton.jit
def _dot16(h, w, s, ids, base, scores, c, tile: tl.constexpr, gscale,
           NBYTES: tl.constexpr):
    # The parent's gathered BR16/BB64 expression and reduction axes, unchanged.
    r = c * 256 + tile * 16 + tl.arange(0, 16)
    rows = tl.load(ids + r).to(tl.int32)
    acc = tl.zeros((16,), tl.float32)
    for b0 in range(0, NBYTES, 64):
        ob = b0 + tl.arange(0, 64)
        wb = tl.load(w + rows[:, None] * NBYTES + ob[None, :],
                     eviction_policy="evict_first").to(tl.int32)
        og = b0 // 8 + tl.arange(0, 8)
        sc = tl.load(s + rows[:, None] * (NBYTES // 8) + og[None, :]).to(tl.float32)
        hl = tl.load(h + 2 * ob)
        hh = tl.load(h + 2 * ob + 1)
        lo, hi = _e2m1_pair(wb)
        part = lo * hl[None, :] + hi * hh[None, :]
        part = tl.reshape(part, (16, 8, 8))
        acc += tl.sum(tl.sum(part, axis=2) * sc, axis=1)
    v = tl.load(base + rows) + acc * gscale
    tl.store(scores + r, v)
    return v


@triton.jit
def _refine_topk(h, w, s, ids, base, scores, out, gscale,
                 NBYTES: tl.constexpr, gate_ptr, GATED: tl.constexpr,
                 launch_pdl: tl.constexpr = False):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    c = tl.program_id(0)
    if GATED:
        if tl.load(gate_ptr) != 0:
            i = tl.arange(0, 256)
            tl.store(scores + c * 256 + i, tl.zeros((256,), tl.float32) * gscale)
            j = tl.arange(0, 32)
            tl.store(out + c * 32 + j, tl.load(ids + c * 256 + j))
            return
    # Each original 16-row tile is kept intact. Concats preserve input order.
    a0 = _dot16(h, w, s, ids, base, scores, c, 0, gscale, NBYTES)
    a1 = _dot16(h, w, s, ids, base, scores, c, 1, gscale, NBYTES)
    a2 = _dot16(h, w, s, ids, base, scores, c, 2, gscale, NBYTES)
    a3 = _dot16(h, w, s, ids, base, scores, c, 3, gscale, NBYTES)
    a4 = _dot16(h, w, s, ids, base, scores, c, 4, gscale, NBYTES)
    a5 = _dot16(h, w, s, ids, base, scores, c, 5, gscale, NBYTES)
    a6 = _dot16(h, w, s, ids, base, scores, c, 6, gscale, NBYTES)
    a7 = _dot16(h, w, s, ids, base, scores, c, 7, gscale, NBYTES)
    a8 = _dot16(h, w, s, ids, base, scores, c, 8, gscale, NBYTES)
    a9 = _dot16(h, w, s, ids, base, scores, c, 9, gscale, NBYTES)
    a10 = _dot16(h, w, s, ids, base, scores, c, 10, gscale, NBYTES)
    a11 = _dot16(h, w, s, ids, base, scores, c, 11, gscale, NBYTES)
    a12 = _dot16(h, w, s, ids, base, scores, c, 12, gscale, NBYTES)
    a13 = _dot16(h, w, s, ids, base, scores, c, 13, gscale, NBYTES)
    a14 = _dot16(h, w, s, ids, base, scores, c, 14, gscale, NBYTES)
    a15 = _dot16(h, w, s, ids, base, scores, c, 15, gscale, NBYTES)
    b0 = tl.cat(a0, a1, can_reorder=False)
    b1 = tl.cat(a2, a3, can_reorder=False)
    b2 = tl.cat(a4, a5, can_reorder=False)
    b3 = tl.cat(a6, a7, can_reorder=False)
    b4 = tl.cat(a8, a9, can_reorder=False)
    b5 = tl.cat(a10, a11, can_reorder=False)
    b6 = tl.cat(a12, a13, can_reorder=False)
    b7 = tl.cat(a14, a15, can_reorder=False)
    d0 = tl.cat(b0, b1, can_reorder=False)
    d1 = tl.cat(b2, b3, can_reorder=False)
    d2 = tl.cat(b4, b5, can_reorder=False)
    d3 = tl.cat(b6, b7, can_reorder=False)
    e0 = tl.cat(d0, d1, can_reorder=False)
    e1 = tl.cat(d2, d3, can_reorder=False)
    v = tl.cat(e0, e1, can_reorder=False)
    key = _topk_key(v)
    low = tl.min(key, axis=0)
    high = tl.max(key, axis=0)
    if low == high:
        t = low
    else:
        active = tl.full((256,), True, tl.int1)
        left = tl.full((), 32, tl.int32)
        bins = tl.arange(0, 16)
        t = tl.full((), 0, tl.int64)
        for shift in tl.static_range(28, -1, -4):
            digit = ((key >> shift) & 15).to(tl.int32)
            hist = tl.histogram(digit, 16, mask=active)
            suffix = tl.sum(hist, axis=0) - tl.cumsum(hist, axis=0) + hist
            pick = tl.max(tl.where(suffix >= left, bins, 0), axis=0)
            left -= tl.sum(tl.where(bins > pick, hist, 0), axis=0)
            t |= pick.to(tl.int64) << shift
            active = active & (digit == pick)
    i = tl.arange(0, 256)
    gt = (key > t).to(tl.int32)
    eq = (key == t).to(tl.int32)
    n_gt = tl.sum(gt, axis=0)
    gt_rank = tl.cumsum(gt, axis=0) - gt
    eq_rank = tl.cumsum(eq, axis=0) - eq
    take = (gt != 0) | ((eq != 0) & (eq_rank < 32 - n_gt))
    pos = tl.where(gt != 0, gt_rank, n_gt + eq_rank)
    val = tl.load(ids + c * 256 + i)
    tl.store(out + c * 32 + pos, val, mask=take)


def run(screen, gated):
    gate, ok = _gate_args(screen.scores)
    _refine_topk[(screen.n // 2048,)](
        screen._hr, screen.w_rest, screen.s_rest, screen._coarse_cand,
        screen.scores, screen.refine_scores, screen._cand, screen.gscale,
        NBYTES=screen.w_rest.shape[1], gate_ptr=gate, GATED=gated and ok,
        launch_pdl=mtp_pdl(), num_warps=4, num_stages=2,
    )


@triton.jit
def _produce(src, dst, gate, init, phase, K: tl.constexpr,
             launch_pdl: tl.constexpr = False):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    i = tl.arange(0, 4096)
    # Changing device data between replays; no host snapshot inside the graph.
    p = tl.load(phase)
    j = (i + p) % K
    tl.store(dst + i, tl.load(src + j, mask=i < K, other=0), mask=i < K)
    g = tl.arange(0, 4)
    tl.store(gate + g, tl.load(init + g))
    tl.store(phase, (p + 37) % K)


@torch.no_grad()
def qualify(screen):
    """All execution is official remote startup, before request/graph binding."""
    if not ((SCREEN_CHUNK, CASCADE_PER, SCREEN_PER) == (2048, 256, 32)
            and screen.cascade and screen.fused_topk and screen.reuse and screen.radix
            and screen.n == 131072 and screen.w_rest.shape[1] == 256
            and tuple(screen.s_rest.shape) == (131072, 32)
            and screen._h.numel() == 2688
            and not torch.cuda.is_current_stream_capturing()):
        return False
    device = screen._h.device
    gate = depth_gate_init(device)
    if gate is None or gate.dtype != torch.int32 or gate.numel() not in (3, 4):
        return False
    saved_gate = gate.clone()
    names = ("_hc", "_hs", "_hr", "scores", "refine_scores", "_coarse_cand",
             "_cand", "cand_logits", "_h", "_out")
    copies, backing = [], []
    try:
        for fused in (False, True):
            obj = copy.copy(screen)
            obj.graph = None
            obj.refine_fused = fused
            guards = {}
            for name in names:
                old = getattr(screen, name)
                buf = torch.full((old.numel() + 64,), -177, dtype=old.dtype, device=device)
                setattr(obj, name, buf[32:-32].view(old.shape))
                guards[name] = buf
            copies.append(obj)
            backing.append(guards)

        def same(a, b):
            return torch.equal(a.view(torch.uint8), b.view(torch.uint8))

        # Fourth slot is also supported by the compatible cumulative gate route.
        init = torch.zeros(4, dtype=torch.int32, device=device)
        init[3] = 0x3F800000
        private_gate = torch.empty_like(init)
        phases = [torch.zeros((), dtype=torch.int32, device=device) for _ in copies]
        idx = torch.arange(2688, dtype=torch.int32, device=device)
        base = ((idx * 1103515245 + 12345) % 65521).float() / 8192 - 4
        cases = [base, base * 0.03125, base * 8, base * 1e-30,
                 torch.where(idx % 2 == 0, 0, -2147483648).to(torch.int32).view(torch.float32),
                 torch.ones_like(base), base.roll(113)]
        special = base.clone()
        special[::137] = float("nan")
        special[1::149] = float("inf")
        special[2::157] = -float("inf")
        cases.append(special)
        # Ordinary eager comparison and changing-device-producer graph comparison.
        for src in cases:
            for gated, stopped, active, kept in (
                    (False, 1, 1, 2), (True, 0, 0, 0), (True, 0, 1, 0),
                    (True, 0, 1, 2), (True, 1, 1, 2)):
                init[0], init[1], init[2] = stopped, kept, active
                init[3] = 0x3F333333 if kept else 0x3F800000
                snapshots = []
                for obj in copies:
                    for name in names:
                        getattr(obj, name).fill_(-177)
                    gate.copy_(init[:gate.numel()])
                    obj._h.copy_(src)
                    obj._greedy_row(obj._h, gated=gated, out=obj._out)
                    snapshots.append(gate.clone())
                if not same(snapshots[0], snapshots[1]):
                    return False
                for name in names:
                    if not same(backing[0][name], backing[1][name]):
                        return False
                for guards in backing:
                    for buf in guards.values():
                        if not (bool((buf[:32] == -177).all()) and bool((buf[-32:] == -177).all())):
                            return False
            # Graph producer supports either gate length without overwriting it.
            graphs = []
            init[0], init[1], init[2] = 0, 0, 1
            init[3] = 0x3F800000
            for obj, phase in zip(copies, phases):
                phase.zero_()
                _produce[(1,)](src, obj._h, private_gate, init, phase, K=2688,
                                launch_pdl=mtp_pdl(), num_warps=4)
                gate.copy_(private_gate[:gate.numel()])
                obj._greedy_row(obj._h, gated=True, out=obj._out)
                torch.cuda.synchronize(device)
                phase.zero_()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
                    _produce[(1,)](src, obj._h, private_gate, init, phase, K=2688,
                                    launch_pdl=mtp_pdl(), num_warps=4)
                    gate.copy_(private_gate[:gate.numel()])
                    obj._greedy_row(obj._h, gated=True, out=obj._out)
                graphs.append(graph)
            for phase in phases:
                phase.zero_()
            for _ in range(3):
                snaps = []
                for graph in graphs:
                    graph.replay()
                    snaps.append(gate.clone())
                if not same(snaps[0], snaps[1]):
                    return False
                for name in names:
                    if not same(backing[0][name], backing[1][name]):
                        return False
                for guards in backing:
                    for buf in guards.values():
                        if not (bool((buf[:32] == -177).all()) and bool((buf[-32:] == -177).all())):
                            return False

        # Full-screen graph budget, including producer, selection and gate writes.
        flush = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
        ratios = []
        for stopped in (0, 1):
            init[0], init[1], init[2] = stopped, 0, 1
            graphs = []
            for obj, phase in zip(copies, phases):
                phase.zero_()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
                    _produce[(1,)](base, obj._h, private_gate, init, phase, K=2688,
                                    launch_pdl=mtp_pdl(), num_warps=4)
                    gate.copy_(private_gate[:gate.numel()])
                    obj._greedy_row(obj._h, gated=True, out=obj._out)
                graphs.append(graph)
            for cold in (False, True):
                times = [[], []]
                for repeat in range(6):
                    for which in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                        phases[which].zero_()
                        if cold:
                            flush.zero_()
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(1 if cold else 16):
                            graphs[which].replay()
                        end.record()
                        end.synchronize()
                        times[which].append(start.elapsed_time(end) / (1 if cold else 16))
                a, b = map(statistics.median, times)
                if stopped:
                    if b > a * 1.02 + 0.001:
                        return False
                else:
                    ratios.append(b / a)
        return statistics.mean(ratios) < 0.97 and max(ratios) < 1.03
    finally:
        torch.cuda.synchronize(device)
        gate.copy_(saved_gate)
        torch.cuda.synchronize(device)

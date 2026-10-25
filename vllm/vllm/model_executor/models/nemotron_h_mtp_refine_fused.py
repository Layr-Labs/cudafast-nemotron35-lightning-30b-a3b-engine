"""Optional exact refine-dot plus chunk radix selection for native MTP.

No screen-policy, threshold, precision or target-logit changes. Original screen
kernels are the private remote reference and unconditional fallback.
"""
import os
import statistics

import torch
from vllm.triton_utils import tl, triton
from .nemotron_h_mtp_screen import (
    _e2m1_pair, _topk_key, CASCADE_PER, SCREEN_PER, screen_graph_enabled,
)
from .nemotron_h_mtp_kernels import depth_gate_init, _gate_args, mtp_pdl

_ENABLED = os.environ.get("MTP_REFINE_FUSED", "1") != "0"


@triton.jit
def _refine_topk(h_ptr, w_ptr, s_ptr, src_ptr, base_ptr,
                 scores_ptr, out_ptr, gscale, gate_ptr,
                 NBYTES: tl.constexpr, BB: tl.constexpr,
                 N: tl.constexpr, K: tl.constexpr,
                 GATED: tl.constexpr, HCOLS: tl.constexpr,
                 hcols_ptr=None, launch_pdl: tl.constexpr = False):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    c = tl.program_id(0)
    i = tl.arange(0, N)
    rows = tl.load(src_ptr + c * N + i).to(tl.int32)
    if GATED:
        if tl.load(gate_ptr) != 0:
            tl.store(scores_ptr + c * N + i,
                     tl.zeros((N,), tl.float32) * gscale)
            # Original stopped chunk selection takes the first K ids.
            j = tl.arange(0, K)
            tl.store(out_ptr + c * K + j, tl.load(src_ptr + c * N + j))
            return
    acc = tl.zeros((N,), tl.float32)
    for b0 in range(0, NBYTES, BB):
        ob = b0 + tl.arange(0, BB)
        wb = tl.load(w_ptr + rows[:, None] * NBYTES + ob[None, :],
                     eviction_policy="evict_first").to(tl.int32)
        og = b0 // 8 + tl.arange(0, BB // 8)
        sc = tl.load(s_ptr + rows[:, None] * (NBYTES // 8) + og[None, :]).to(tl.float32)
        if HCOLS:
            hl = tl.load(h_ptr + tl.load(hcols_ptr + 2 * ob))
            hh = tl.load(h_ptr + tl.load(hcols_ptr + 2 * ob + 1))
        else:
            hl = tl.load(h_ptr + 2 * ob)
            hh = tl.load(h_ptr + 2 * ob + 1)
        lo, hi = _e2m1_pair(wb)
        part = lo * hl[None, :] + hi * hh[None, :]
        part = tl.reshape(part, (N, BB // 8, 8))
        acc += tl.sum(tl.sum(part, axis=2) * sc, axis=1)
    score = tl.load(base_ptr + rows) + acc * gscale
    # Keep stage scores observable for original-stage raw-bit admission.
    tl.store(scores_ptr + c * N + i, score)
    key = _topk_key(score)
    # Original RADIX=True threshold body.
    # Exact 4-bit radix selection: eight histograms replace 32 serial
    # whole-block threshold reductions. Keys include NaNs and signed zeros
    # in the original torch order; survivor scattering below is unchanged.
    low = tl.min(key, axis=0)
    high = tl.max(key, axis=0)
    if low == high:
        # Stopped draft rows normally produce uniform scores. Finish exact
        # tie selection without any histogram, while still writing all K ids.
        t = low
    else:
        active = tl.full((N,), True, tl.int1)
        left = tl.full((), K, tl.int32)
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
    gt = (key > t).to(tl.int32)
    eq = (key == t).to(tl.int32)
    n_gt = tl.sum(gt, axis=0)
    gt_rank = tl.cumsum(gt, axis=0) - gt
    eq_rank = tl.cumsum(eq, axis=0) - eq
    take = (gt != 0) | ((eq != 0) & (eq_rank < K - n_gt))
    pos = tl.where(gt != 0, gt_rank, n_gt + eq_rank)
    val = rows.to(tl.int64)
    tl.store(out_ptr + c * K + pos, val, mask=take)


def maybe_refine(screen, h, gated):
    mode = getattr(screen, "_refine_mode", None)
    if mode is None or not screen.reuse or not screen.radix:
        return False
    warps, direct = mode
    gate, present = _gate_args(screen.refine_scores)
    _refine_topk[(screen.n // 2048,)](
        h if direct else screen._hr, screen.w_rest, screen.s_rest,
        screen._coarse_cand, screen.scores, screen.refine_scores, screen._cand,
        screen.gscale, gate, NBYTES=256, BB=64, N=256, K=32,
        GATED=gated and present, HCOLS=direct,
        hcols_ptr=screen.rest_cols if direct else None,
        launch_pdl=mtp_pdl(), num_warps=warps, num_stages=2,
    )
    return True


@torch.no_grad()
def prepare(screen):
    """Official remote load only: original whole-screen bits and cold graphs.

    No request inputs, outputs or routing values persist. Refused configurations
    retain the complete original head. Only a small immutable mode survives.
    """
    if (not _ENABLED or not screen_graph_enabled() or not screen.cascade or not screen.fused_topk
            or not screen.reuse or not screen.radix or screen.n != 131072
            or tuple(screen.w_rest.shape) != (131072, 256)
            or tuple(screen.s_rest.shape) != (131072, 32)
            or CASCADE_PER != 256 or SCREEN_PER != 32
            or torch.cuda.is_current_stream_capturing()):
        return None
    device = screen._h.device
    gate = depth_gate_init(device)
    saved_gate = None if gate is None else gate.clone()
    saved_h = screen._h.clone()
    screen._refine_mode = None

    def snapshot():
        tensors = [screen._out, screen.scores, screen._coarse_cand,
                   screen.refine_scores, screen._cand, screen.cand_logits]
        if gate is not None:
            tensors.append(gate)
        return [t.clone() for t in tensors]

    def run(mode, h, state):
        screen._refine_mode = mode
        if gate is not None:
            gate.copy_(state)
        screen._greedy_row(h, gated=True, out=screen._out)
        return snapshot()

    try:
        gen = torch.Generator(device=device).manual_seed(3225632)
        probes = [(torch.randn(2688, device=device, generator=gen) * scale)
                  .to(torch.bfloat16).float() for scale in (0.001, 1., 32.)]
        edge = torch.zeros(2688, dtype=torch.bfloat16, device=device)
        edge.view(torch.int16)[:8] = torch.tensor(
            [0, -32768, 1, -32767, 128, -32640, 16256, -16512],
            dtype=torch.int16, device=device)
        probes.extend((edge.float(), torch.zeros(2688, device=device)))
        nonfinite = probes[1].clone()
        nonfinite[:3] = torch.tensor([float("inf"), -float("inf"), float("nan")], device=device)
        probes.append(nonfinite)
        states = [torch.tensor(s, dtype=torch.int32, device=device)
                  for s in ((0, 0, 0), (0, 0, 1), (0, 2, 1), (1, 2, 1))]
        candidates = []
        for mode in ((8, False), (16, False), (8, True), (16, True)):
            try:
                exact = True
                for h in probes:
                    for state in states:
                        old = run(None, h, state)
                        new = run(mode, h, state)
                        if any(not torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                               for a, b in zip(old, new)):
                            exact = False
                            break
                    if not exact:
                        break
                if exact:
                    candidates.append(mode)
            except Exception as exc:
                print(f"MTP_REFINE_FUSED mode={mode} refused: {exc!r}", flush=True)
        if not candidates:
            print("MTP_REFINE_FUSED original screen retained: no private exact mode", flush=True)
            return None

        # Whole original/candidate graph timing avoids a Python-launch bias.
        screen._h.copy_(probes[1])
        start_state = states[1]
        graphs = {}
        for mode in (None, *candidates):
            screen._refine_mode = mode
            if gate is not None:
                gate.copy_(start_state)
            screen._greedy_row(screen._h, gated=True, out=screen._out)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
                screen._greedy_row(screen._h, gated=True, out=screen._out)
            torch.cuda.synchronize(device)
            graphs[mode] = graph
        # Admit the captured executions too, with changing input values and
        # gate generations; the eager check alone is not a replay check.
        replay_exact = []
        for mode in candidates:
            exact = True
            for h in probes[:3]:
                screen._h.copy_(h)
                for state in states:
                    if gate is not None:
                        gate.copy_(state)
                    graphs[None].replay()
                    old = snapshot()
                    if gate is not None:
                        gate.copy_(state)
                    graphs[mode].replay()
                    new = snapshot()
                    if any(not torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                           for a, b in zip(old, new)):
                        exact = False
                        break
                if not exact:
                    break
            if exact:
                replay_exact.append(mode)
        if not replay_exact:
            return None
        candidates = replay_exact
        graphs = {mode: graph for mode, graph in graphs.items()
                  if mode is None or mode in candidates}
        screen._h.copy_(probes[1])
        flush = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = {mode: [] for mode in graphs}
        for trial in range(7):
            order = list(graphs)
            if trial % 2:
                order.reverse()
            for mode in order:
                if gate is not None:
                    gate.copy_(start_state)
                flush.zero_()
                start.record()
                graphs[mode].replay()
                end.record()
                end.synchronize()
                times[mode].append(start.elapsed_time(end))
        baseline = statistics.median(times[None])
        best = min(candidates, key=lambda mode: statistics.median(times[mode]))
        admitted = statistics.median(times[best]) < 0.96 * baseline
        print(f"MTP_REFINE_FUSED mode={best} admitted={admitted} "
              "private_stage_bits+gate+whole_cold_graph4pct", flush=True)
        return best if admitted else None
    except Exception as exc:
        print(f"MTP_REFINE_FUSED original screen retained: {exc!r}", flush=True)
        return None
    finally:
        screen._refine_mode = None
        screen._h.copy_(saved_h)
        if gate is not None:
            gate.copy_(saved_gate)

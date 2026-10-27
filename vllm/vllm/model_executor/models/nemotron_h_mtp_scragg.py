# SPDX-License-Identifier: Apache-2.0
"""Official-remote-only admission for the actual screen's optional radix twin."""
from __future__ import annotations

import statistics

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
_SENTINEL = 0x2BAD


def _bits(t):
    return t.contiguous().reshape(-1).view(torch.uint8)


def _same(a, b):
    return torch.equal(_bits(a), _bits(b))


@torch.no_grad()
def qualify(screen, k, device, tail_api):
    """No request input: private changed-input graphs, actual weights/old tail.

    The existing original-tail versus cascade gate has already passed. Only
    the additional integer histogram/scan variant is selected here.
    """
    from .nemotron_h_mtp_kernels import GATE_PROD_ONE, depth_gate_init

    if (not getattr(tail_api, "aggregate_available", lambda: False)()
            or not screen.tail or not screen.pre or not screen.sctopk or screen._next_pf
            or torch.cuda.is_current_stream_capturing()
            or torch.cuda.get_device_capability(device) != (12, 1)):
        return False
    gate = depth_gate_init(device)
    gate_saved = gate.clone() if gate is not None else None
    # Restore every direct tensor owner, including all scratch, in finally.
    owners = [(n, t, t.clone()) for n, t in vars(screen).items()
              if isinstance(t, torch.Tensor)]
    writable = {"_pre_cand", "mid_scores", "scores", "_coarse_cand", "refine_scores",
                "_cand", "cand_logits", "_h", "_out", "_hp", "_hm", "_hr", "_hc", "_hs"}
    readonly = [(t, before) for n, t, before in owners if n not in writable]
    old_debug = screen._tail_debug
    hback = torch.empty(k + 128, device=device, dtype=torch.float32)
    hback.view(torch.int32).fill_(_SENTINEL)
    h = hback[64:-64]
    oback = [torch.full((129,), _SENTINEL, device=device, dtype=torch.int32)
             for _ in range(2)]
    outs = [b[64:65] for b in oback]
    state = torch.tensor([0, 0, 1, GATE_PROD_ONE], device=device, dtype=torch.int32)
    gen = torch.Generator(device=device).manual_seed(271027)
    names = ("_pre_cand", "mid_scores", "scores", "_coarse_cand", "refine_scores",
             "_cand", "cand_logits")
    pressure = torch.empty(64 << 20, device=device, dtype=torch.uint8)
    timing = []
    ratios = []

    def input_case(trial, stop):
        h.copy_(torch.randn(k, device=device, generator=gen) * (0.25 * 2 ** (trial % 6)))
        if trial == 6:
            h[::3] = 0.0
            h[1::7] = -0.0
        elif trial == 7:
            h.copy_(torch.round(h * 4) / 4)
        elif trial == 8:
            h.mul_(1e-20)
        elif trial == 9:
            h.zero_()  # fully tied screen, no histogram distribution assumption
        state.copy_(torch.tensor([stop, 0, 1, GATE_PROD_ONE], device=device, dtype=torch.int32))

    def capture(debug):
        screen._tail_debug = debug
        graphs = []
        for which in (0, 1):
            screen._tail_aggregate = bool(which)
            for _ in range(2):
                if gate is not None:
                    gate.copy_(state)
                screen._greedy_row(h, gated=gate is not None, out=outs[which])
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
                # This private reset is in BOTH timing graphs. Every repeated
                # replay therefore executes the requested live/stopped case.
                if gate is not None:
                    gate.copy_(state)
                screen._greedy_row(h, gated=gate is not None, out=outs[which])
            graphs.append(graph)
        return graphs

    def guards():
        return (bool(torch.all(hback.view(torch.int32)[:64] == _SENTINEL))
                and bool(torch.all(hback.view(torch.int32)[-64:] == _SENTINEL))
                and all(bool(torch.all(b[:64] == _SENTINEL))
                        and bool(torch.all(b[-64:] == _SENTINEL)) for b in oback))

    def compare(graphs, debug):
        before_h = hback.clone()
        values = []
        for which in (0, 1):
            oback[which].fill_(_SENTINEL)
            if debug:
                for n in names:
                    buf = getattr(screen, n)
                    buf.view(torch.uint8).fill_(0xA5)
            graphs[which].replay()
            selected = names if debug else ("_cand", "cand_logits")
            values.append([oback[which].clone()] + [getattr(screen, n).clone() for n in selected]
                          + ([gate.clone()] if gate is not None else []))
        if (not all(_same(a, b) for a, b in zip(*values))
                or not _same(hback, before_h) or not guards()):
            raise ValueError("aggregated tail changed token/intermediates/gate/input/guards")

    def unchanged():
        if any(not _same(a, b) for a, b in readonly):
            raise ValueError("aggregated tail changed static read-only owner")

    try:
        input_case(0, 0)
        # Prove this exact production-pointer contract reaches the exported
        # AGG entry; a false run result must not silently qualify a cascade
        # fallback as the new kernel. This is a private eager live probe.
        screen._tail_aggregate = False
        screen._tail_debug = False
        if gate is not None:
            gate.copy_(state)
        screen._greedy_row(h, gated=gate is not None, out=outs[0])
        if gate is not None:
            gate.copy_(state)
        from .nemotron_h_mtp_kernels import mtp_pdl

        reached = tail_api.run(
            screen.scores, screen._hm, screen._hr, h, screen.w_mid, screen.s_mid,
            screen.w_rest, screen.s_rest, screen.w_full, screen.s_full,
            screen._cand, screen.cand_logits, screen.gscale,
            gate=gate, win=0, pdl=mtp_pdl(), aggregate=True,
        )
        if not reached:
            return False
        debug_graphs = capture(True)
        for trial in range(10):
            for stop in ((0, 1) if gate is not None else (0,)):
                input_case(trial, stop)
                compare(debug_graphs, True)
                compare(debug_graphs, True)
        unchanged()
        input_case(0, 0)
        graphs = capture(False)  # actual production pointers: debug outputs null
        for trial in (0, 6, 7, 8, 9):
            for stop in ((0, 1) if gate is not None else (0,)):
                input_case(trial, stop)
                compare(graphs, False)
                # Stopped pass has no radix work: correctness still checked,
                # speed admission requires actual live selector work.
                if stop:
                    continue
                for cold in (False, True):
                    samples = [[], []]
                    for repeat in range(5):
                        for which in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                            if cold:
                                pressure.fill_(repeat + 1)
                            else:
                                for _ in range(8):
                                    graphs[which].replay()
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            count = 1 if cold else 16
                            start.record()
                            for _ in range(count):
                                graphs[which].replay()
                            end.record()
                            end.synchronize()
                            value = start.elapsed_time(end) / count
                            if not value > 0:
                                return False
                            samples[which].append(value)
                    old, new = (statistics.median(x) for x in samples)
                    timing.append((old, new))
                    ratios.append(new / old)
                    compare(graphs, False)
        unchanged()
        admitted = (bool(ratios) and statistics.median(ratios) < .995
                    and sum(y for x, y in timing) / sum(x for x, y in timing) < .995
                    and max(ratios) < 1.01)
        logger.info("MTP screen: warp aggregated radix admitted=%s", admitted)
        return admitted
    except Exception as exc:
        logger.warning("MTP screen: warp aggregated radix unavailable (%r); original tail kept", exc)
        return False
    finally:
        screen._tail_aggregate = False
        screen._tail_debug = old_debug
        for _, actual, saved in owners:
            actual.copy_(saved)
        if gate is not None:
            gate.copy_(gate_saved)

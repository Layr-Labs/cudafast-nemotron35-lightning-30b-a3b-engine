# SPDX-License-Identifier: Apache-2.0
"""Optional direct full-hidden gather + existing STAGE1 cascade, remote-only qualification.
Original scrtail mathematical helper and cascade bodies are preserved in scrfirst.
"""
import math
import os
import statistics
from types import SimpleNamespace

import torch
from vllm.triton_utils import triton

try:
    import scrfirst as _scrfirst
except ImportError:
    _scrfirst = None

_ENABLED = os.environ.get("MTP_SCREEN_GATHER_TAIL", "1") != "0"
_ENTRIES = {}
_MUTABLE = ("_h", "_out", "_hp", "_hm", "_hr", "scores", "_pre_cand",
            "mid_scores", "_coarse_cand", "refine_scores", "_cand", "cand_logits")


def _sig(t):
    return id(t), t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device


def _span(t):
    return t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()


def _overlap(a, b):
    al, ah = _span(a)
    bl, bh = _span(b)
    return al < bh and bl < ah


def _owner(screen):
    from . import nemotron_h_mtp_screen as s
    if (not _ENABLED or _scrfirst is None or not _scrfirst.available() or not screen.tail or not screen.cascade or not screen.fused_topk
            or not screen.pre or screen._next_pf or s._scrtail is None
            or not s._scrtail.available() or screen.n != 131072
            or (s.SCREEN_CHUNK, s.PRE_PER, s.CASCADE_PER, s.SCREEN_PER) != (2048, 1024, 256, 32)
            or screen.pre_cols.shape != (256,) or screen.mid_cols.shape != (256,)
            or screen.rest_cols.shape != (512,) or screen.w_pre.shape != (screen.n, 128)
            or screen.s_pre.shape != (screen.n, 16)
            or screen._hp.shape != (256,) or screen._hm.shape != (256,)
            or screen._hr.shape != (512,) or screen.scores.shape != (screen.n,)):
        return None
    tensors = {n: t for n, t in vars(screen).items() if isinstance(t, torch.Tensor)}
    if (not all(n in tensors for n in _MUTABLE) or screen._h.dtype != torch.float32
            or screen._h.shape != (2688,) or not screen._h.is_cuda
            or torch.cuda.get_device_capability(screen._h.device) != (12, 1)
            or any(not t.is_cuda or t.device != screen._h.device or not t.is_contiguous()
                   for t in tensors.values())
            or any(getattr(screen, n).dtype != torch.float32 for n in ("_hp", "_hm", "_hr", "scores"))
            or any(getattr(screen, n).dtype != torch.int64 for n in ("pre_cols", "mid_cols", "rest_cols"))):
        return None
    # New CTA0 stores must never change a different CTA's hidden/weight input.
    outputs = [tensors[n] for n in _MUTABLE if n != "_h"]
    immutable = {n: t for n, t in tensors.items() if n not in _MUTABLE}
    if (any(_overlap(a, b) for i, a in enumerate(outputs) for b in outputs[i + 1:])
            or any(_overlap(a, b) for a in outputs for b in immutable.values())
            or any(_overlap(a, screen._h) for a in outputs)):
        return None
    scale = float(screen.gscale)
    if not math.isfinite(scale):
        return None
    return SimpleNamespace(screen=screen, tensors=tensors,
                           signatures={n: _sig(t) for n, t in tensors.items()},
                           immutable=immutable, scale=scale, n=screen.n,
                           device=screen._h.device, pdl=s.mtp_pdl(),
                           scale_ef=s.SCREEN_SCALE_EF, theta=s.DEPTH_THETA,
                           cumulative=s.DEPTH_CUMULATIVE, tail_module=s._scrtail,
                           fused_module=_scrfirst)


def _can(spec, h, gated):
    from . import nemotron_h_mtp_screen as s
    screen = spec.screen
    if (not _ENABLED or not gated or not screen.tail or not screen.cascade or not screen.fused_topk
            or not screen.pre or screen._next_pf or s._scrtail is not spec.tail_module
            or _scrfirst is not spec.fused_module
            or float(screen.gscale) != spec.scale or screen.n != spec.n
            or s.mtp_pdl() != spec.pdl or s.SCREEN_SCALE_EF != spec.scale_ef
            or s.DEPTH_THETA != spec.theta or s.DEPTH_CUMULATIVE != spec.cumulative
            or any(_sig(getattr(screen, n)) != q for n, q in spec.signatures.items())
            or h.dtype != torch.float32 or h.shape != (2688,)
            or h.device != spec.device or not h.is_contiguous()
            or any(_overlap(h, getattr(screen, n)) for n in _MUTABLE if n != "_h")):
        return False
    gate, ok = s._gate_args(h)
    if gated and (not ok or gate.dtype != torch.int32 or gate.shape != (4,) or not gate.is_contiguous()
                  or _overlap(gate, h)
                  or any(_overlap(gate, getattr(screen, n)) for n in _MUTABLE)):
        return False
    return True


def _valid_out(spec, out):
    return (isinstance(out, torch.Tensor) and out.dtype == torch.int32 and out.numel() == 1
            and out.is_contiguous() and out.device == spec.device
            and not _overlap(out, spec.screen._h)
            and not any(_overlap(out, t) for t in spec.immutable.values())
            and not any(_overlap(out, getattr(spec.screen, n)) for n in _MUTABLE
                        if n != "_out"))


def _run(spec, h, gated, raw_first=None):
    if not _can(spec, h, gated):
        return False  # Nothing launched.
    from . import nemotron_h_mtp_screen as s
    screen = spec.screen
    gate, ok = s._gate_args(h)
    if raw_first is not None and (any(_overlap(raw_first, t) for t in spec.tensors.values())
                                   or _overlap(raw_first, h) or _overlap(raw_first, gate)):
        return False
    return _scrfirst.run(screen, h, gate if gated and ok else None,
                         spec.pdl, screen._tail_debug, raw_first)


def _new_full(spec, h, gated, out):
    from . import nemotron_h_mtp_screen as s
    screen = spec.screen
    gate, ok = s._gate_args(out)
    if (not _valid_out(spec, out) or _overlap(out, h) or _overlap(out, gate)
            or not _run(spec, h, gated)):
        raise RuntimeError("private production gather-tail contract refused")
    s._select_token[(1,)](
        screen.cand_logits, screen._cand, out, N=screen.cand_logits.numel(),
        BLOCK=triton.next_power_of_2(screen.cand_logits.numel()),
        gate_ptr=gate, GATED=gated and ok, THETA=spec.theta,
        CUM=spec.cumulative, launch_pdl=spec.pdl)


def maybe_full(screen, h, gated, out):
    spec = _ENTRIES.get(id(screen))
    if spec is None or not _valid_out(spec, out) or not _can(spec, h, gated):
        return False
    from . import nemotron_h_mtp_screen as s
    gate, _ = s._gate_args(out)
    if _overlap(out, h) or _overlap(out, gate):
        return False
    _new_full(spec, h, gated, out)
    return True


def _words(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _guarded(t):
    owner = torch.empty(t.numel() + 128, dtype=t.dtype, device=t.device)
    owner.view(torch.uint8).fill_(0xA5)
    return owner, owner[64:-64].view(t.shape)


def _guard(pair):
    if pair is None or pair[0] is None:
        return True
    return bool(torch.all(pair[0][:64].view(torch.uint8) == 0xA5).item()
                and torch.all(pair[0][-64:].view(torch.uint8) == 0xA5).item())


def _snapshot(screen):
    return {n: getattr(screen, n).clone() for n in _MUTABLE}


def _restore(screen, saved):
    for n, t in saved.items():
        getattr(screen, n).copy_(t)


def _observe(screen, gate, out):
    return (out.clone(), gate.clone(), *[getattr(screen, n).clone() for n in _MUTABLE])


@torch.no_grad()
def _qualify(spec, guard_pairs, debug, timing):
    from . import nemotron_h_mtp_screen as s
    from . import nemotron_h_mtp_kernels as k
    screen = spec.screen
    screen._tail_debug = debug
    gen = torch.Generator(device=spec.device).manual_seed(2026102841 + int(debug))
    raw_first = _guarded(screen.scores)
    hb = None if timing else _guarded(screen._h)
    h = screen._h if timing else hb[1]
    cont = torch.empty_like(h)
    gate = k._GATE
    gate_start = torch.empty_like(gate)
    oldouts = [_guarded(screen._out) for _ in range(8)]
    newouts = [_guarded(screen._out) for _ in range(8)]
    if timing:
        # Also exercise the actual captured screen's output address. Seven
        # other independently owned outputs remain in each arm.
        oldouts[0] = (None, screen._out)
        newouts[0] = (None, screen._out)
    saved = _snapshot(screen)

    def change(case, stopped):
        h.copy_(torch.randn(h.shape, device=spec.device, generator=gen)
                * (.001, .25, 1., 32.)[case % 4])
        h.view(torch.int32)[:4] = torch.tensor([0, -2147483648, 1, -2147483647],
                                              dtype=torch.int32, device=spec.device)
        if case == 4:
            h.zero_()
        if case == 5:
            h.copy_(torch.round(h * 4) / 4)
        if case == 6:
            h[:3] = torch.tensor([float("inf"), -float("inf"), float("nan")], device=spec.device)
        if case == 7:
            h.mul_(1e-20)
        cont.copy_(torch.randn(h.shape, device=spec.device, generator=gen) * .5)
        gate_start.zero_()
        gate_start[0] = int(stopped)
        gate_start[2] = 1
        gate_start[3] = 0x3F800000 if case % 2 == 0 else 0x3F000000

    def reset():
        for n, t in saved.items():
            if n != "_h" or h is not screen._h:
                getattr(screen, n).copy_(t)
        gate.copy_(gate_start)

    def check_equal(a, b):
        if not all(_words(x, y) for x, y in zip(a, b)):
            raise RuntimeError("gather-tail complete screen state/output differs")
        if not _guard(hb) or not _guard(raw_first) or not all(_guard(p) for p in guard_pairs):
            raise RuntimeError("gather-tail private workspace guard changed")

    # First-dot scores are copied before tail debug can scatter mid scores.
    # The oracle is exactly the original gather + first dot, not old STAGE1.
    for case in range(8):
        for stopped in (False, True):
            change(case, stopped)
            reset()
            s._gather_three[(1,)](h, screen.pre_cols, screen._hp, screen.mid_cols, screen._hm,
                                  screen.rest_cols, screen._hr, NA=256, NB=256, NC=512,
                                  launch_pdl=spec.pdl)
            s._rows_dot(screen._hp, screen.w_pre, screen.s_pre, None,
                        screen.gscale, screen.scores, False, True)
            old = [getattr(screen, n).clone() for n in ("_hp", "_hm", "_hr", "scores")]
            reset()
            if not _run(spec, h, True, raw_first[1]):
                raise RuntimeError("gather-tail actual contract unavailable")
            check_equal(old, [screen._hp, screen._hm, screen._hr, raw_first[1]])
            # Complete screen and an ORIGINAL continuation, from identical state.
            source = h.clone()
            reset()
            screen._greedy_row(h, gated=True, out=oldouts[0][1])
            old_after = _observe(screen, gate, oldouts[0][1])
            screen._greedy_row(cont, gated=True, out=oldouts[1][1])
            old_cont = _observe(screen, gate, oldouts[1][1])
            reset()
            _new_full(spec, h, True, newouts[0][1])
            check_equal(old_after, _observe(screen, gate, newouts[0][1]))
            screen._greedy_row(cont, gated=True, out=newouts[1][1])
            check_equal(old_cont, _observe(screen, gate, newouts[1][1]))
            if not (_words(source, h) and all(_guard(p) for p in oldouts + newouts)):
                raise RuntimeError("gather-tail hidden/output guard or read-only changed")
    # Eight independently owned full outputs, each graph contains ONE complete
    # original/new screen body. Gate resets are OUTSIDE the measured graph, so
    # repeated replays cannot silently time only stopped work or forced policy.
    change(2, False)
    graphs = [[], []]
    for arm, outs in enumerate((oldouts, newouts)):
        for pair in outs:
            reset()
            if arm == 0:
                screen._greedy_row(h, gated=True, out=pair[1])
            else:
                _new_full(spec, h, True, pair[1])
        torch.cuda.synchronize(spec.device)
        for pair in outs:
            graph = torch.cuda.CUDAGraph()
            reset()
            with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
                if arm == 0:
                    screen._greedy_row(h, gated=True, out=pair[1])
                else:
                    _new_full(spec, h, True, pair[1])
            graphs[arm].append(graph)
        if len({pair[1].data_ptr() for pair in outs}) != 8:
            raise RuntimeError("gather-tail eight outputs alias")
    for case in range(8):
        for stopped in (False, True):
            change(case, stopped)
            source = h.clone()
            for i in range(8):
                for _ in range(2):
                    reset(); graphs[0][i].replay()
                    old = _observe(screen, gate, oldouts[i][1])
                    reset(); graphs[1][i].replay()
                    check_equal(old, _observe(screen, gate, newouts[i][1]))
                    if not (_words(source, h) and _guard(oldouts[i]) and _guard(newouts[i])):
                        raise RuntimeError("gather-tail changed-input graph guard/read-only differs")
    if not timing:
        return None
    pressure = torch.empty(64 << 20, dtype=torch.uint8, device=spec.device)
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ratios = []
    for case in (0, 1, 2, 3, 4, 5):
        for stopped in (False, True):
            change(case, stopped)
            for cold in (False, True):
                samples = [[], []]
                for trial in range(5):
                    for arm in ((0, 1) if trial % 2 == 0 else (1, 0)):
                        elapsed = 0.0
                        for i in range(8):
                            # Production overwrites all consumed workspaces;
                            # only the independent private gate input resets.
                            # No bulk snapshot restore biases the timed cache state.
                            gate.copy_(gate_start)
                            if cold:
                                pressure.fill_(case + trial + i)
                            begin.record(); graphs[arm][i].replay(); end.record(); end.synchronize()
                            elapsed += begin.elapsed_time(end)
                        if not elapsed > 0:
                            raise RuntimeError("gather-tail invalid event interval")
                        samples[arm].append(elapsed)
                a, b = (statistics.median(v) for v in samples)
                ratios.append(b / a)
    metrics = statistics.median(ratios), max(ratios)
    if metrics[0] >= .995 or metrics[1] >= 1.01:
        raise RuntimeError(f"gather-tail full-body warm/pressure cost refused {metrics}")
    return metrics


@torch.no_grad()
def prepare(screen):
    from . import nemotron_h_mtp_kernels as k
    if not _ENABLED or torch.cuda.is_current_stream_capturing():
        return False
    spec = _owner(screen)
    if spec is None:
        return False
    original_gate, original_debug = k._GATE, screen._tail_debug
    original_fields = {n: getattr(screen, n) for n in _MUTABLE}
    original_values = {n: t.clone() for n, t in original_fields.items()}
    readonly = {n: t.clone() for n, t in spec.immutable.items()}
    k._GATE = torch.zeros(4, dtype=torch.int32, device=spec.device)
    error, metrics = None, None
    try:
        guarded = {n: _guarded(t) for n, t in original_fields.items()}
        for n, pair in guarded.items():
            pair[1].copy_(original_values[n])
            setattr(screen, n, pair[1])
        private_spec = _owner(screen)
        if private_spec is None:
            raise RuntimeError("private guarded actual screen contract unavailable")
        _qualify(private_spec, tuple(guarded.values()), debug=True, timing=False)
        for n, t in original_fields.items():
            setattr(screen, n, t)
        _restore(screen, original_values)
        metrics = _qualify(spec, (), debug=False, timing=True)
    except Exception as exc:
        error = exc
    finally:
        try:
            torch.cuda.synchronize(spec.device)
            for n, t in original_fields.items():
                setattr(screen, n, t)
            _restore(screen, original_values)
            if not all(_words(getattr(screen, n), t) for n, t in original_values.items()):
                raise RuntimeError("gather-tail mutable owner restoration failed")
        finally:
            screen._tail_debug = original_debug
            k._GATE = original_gate
    if not all(_words(getattr(screen, n), t) for n, t in readonly.items()):
        raise RuntimeError("gather-tail immutable actual screen weight/scale/column modified")
    if error is not None:
        print(f"MTP_SCREEN_GATHER_TAIL refused: {error!r}; original kept", flush=True)
        return False
    _ENTRIES[id(screen)] = spec
    print(f"MTP_SCREEN_GATHER_TAIL admitted full-body ratios={metrics}", flush=True)
    return True

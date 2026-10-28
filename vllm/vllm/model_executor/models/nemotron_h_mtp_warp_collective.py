# SPDX-License-Identifier: Apache-2.0
"""Official-remote-only full-screen admission for integer warp collectives.

Original EB48 gather/first-dot/selector/policy; scragg c4b2 histogram/scan twin.
Qualification derives from 2774620e: actual adaptive owners, changed-output
CUDA graphs and full production-body warm/pressure timing; no local execution.
"""
import math
import os
import statistics
from types import SimpleNamespace

import torch
import triton

_ENABLED = os.environ.get("MTP_SCREEN_TAIL_WARP_AGG", "1") != "0"
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
    if (not _ENABLED or not screen.tail or not screen.cascade or not screen.fused_topk
            or not screen.pre or screen._next_pf or s._scrtail is None
            or not s._scrtail.available()
            or not getattr(s._scrtail, "aggregate_available", lambda: False)()
            or screen.n != 131072
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
    # Mutable screen outputs must never overlap hidden or immutable inputs.
    outputs = [tensors[n] for n in _MUTABLE if n != "_h"]
    immutable = {n: t for n, t in tensors.items() if n not in _MUTABLE}
    if (any(_overlap(a, b) for i, a in enumerate(outputs) for b in outputs[i + 1:])
            or any(_overlap(a, b) for a in outputs for b in immutable.values())
            or any(_overlap(a, screen._h) for a in outputs)):
        return None
    scale = float(screen.gscale)
    if not math.isfinite(scale):
        return None
    spec = SimpleNamespace(screen=screen, tensors=tensors,
                           signatures={n: _sig(t) for n, t in tensors.items()},
                           immutable=immutable, scale=scale, n=screen.n,
                           device=screen._h.device, pdl=s.mtp_pdl(),
                           scale_ef=s.SCREEN_SCALE_EF, theta=s.DEPTH_THETA,
                           cumulative=s.DEPTH_CUMULATIVE, tail_module=s._scrtail)
    adapt, enabled, tau = s._adapt_args()
    spec.theta5 = s.DEPTH_THETA5
    spec.select_kernel = s._select_token
    spec.adapt_getter = s._adapt_args
    spec.adapt_enabled, spec.adapt_tau = bool(enabled), float(tau)
    spec.adapt_tensor = adapt if enabled else None
    if enabled and (not isinstance(adapt, torch.Tensor) or not adapt.is_cuda
                    or adapt.device != spec.device or adapt.dtype != torch.float32
                    or adapt.shape != (4,) or not adapt.is_contiguous()
                    or any(_overlap(adapt, t) for t in tensors.values())):
        return None
    spec.adapt_signature = _sig(adapt) if enabled else None
    return spec



def _can(spec, h, gated):
    from . import nemotron_h_mtp_screen as s
    screen = spec.screen
    if (not _ENABLED or not gated or not screen.tail or not screen.cascade or not screen.fused_topk
            or not screen.pre or screen._next_pf or s._scrtail is not spec.tail_module
            or float(screen.gscale) != spec.scale or screen.n != spec.n
            or s.mtp_pdl() != spec.pdl or s.SCREEN_SCALE_EF != spec.scale_ef
            or s.DEPTH_THETA != spec.theta or s.DEPTH_CUMULATIVE != spec.cumulative
            or any(_sig(getattr(screen, n)) != q for n, q in spec.signatures.items())
            or h.dtype != torch.float32 or h.shape != (2688,)
            or h.device != spec.device or not h.is_contiguous()
            or any(_overlap(h, getattr(screen, n)) for n in _MUTABLE if n != "_h")):
        return False
    adapt, enabled, tau = s._adapt_args()
    if (s._select_token is not spec.select_kernel or s._adapt_args is not spec.adapt_getter
            or s.DEPTH_THETA5 != spec.theta5 or bool(enabled) != spec.adapt_enabled
            or float(tau) != spec.adapt_tau
            or (enabled and (_sig(adapt) != spec.adapt_signature or _overlap(adapt, h)))):
        return False
    gate, ok = s._gate_args(h)
    if enabled and _overlap(adapt, gate):
        return False
    if gated and (not ok or gate.dtype != torch.int32 or gate.shape != (4,) or not gate.is_contiguous()
                  or _overlap(gate, h)
                  or any(_overlap(gate, getattr(screen, n)) for n in _MUTABLE)):
        return False
    return True


def _run(spec, h, gated):
    if not _can(spec, h, gated):
        return False
    from . import nemotron_h_mtp_screen as s
    screen = spec.screen
    # EXACT original two-launch prefix: this reserve changes only integer
    # top-k histogram/scan scheduling in the original tail.
    s._gather_three[(1,)](
        h, screen.pre_cols, screen._hp, screen.mid_cols, screen._hm,
        screen.rest_cols, screen._hr, NA=256, NB=256, NC=512,
        launch_pdl=spec.pdl)
    s._rows_dot(screen._hp, screen.w_pre, screen.s_pre, None,
                screen.gscale, screen.scores, False, gated)
    return True


def maybe_full(screen, h, gated, out):
    spec = _ENTRIES.get(id(screen))
    if spec is None or not _can(spec, h, gated):
        return False
    from . import nemotron_h_mtp_screen as s
    gate, ok = s._gate_args(h)
    adapt, enabled, _ = s._adapt_args()
    if (not ok or out.dtype != torch.int32 or out.device != spec.device
            or out.numel() != 1 or not out.is_contiguous()
            or _overlap(out, h) or _overlap(out, gate)
            or (enabled and _overlap(out, adapt))
            or any(_overlap(out, t) for n, t in spec.tensors.items() if n != "_out")):
        return False
    _new_full(spec, h, gated, out)
    return True


def _new_full(spec, h, gated, out):
    from . import nemotron_h_mtp_screen as s
    screen = spec.screen
    if not _run(spec, h, gated):
        raise RuntimeError("private production warp-collective contract refused")
    gate, ok = s._gate_args(out)
    if not s._scrtail.run(
        screen.scores, screen._hm, screen._hr, h, screen.w_mid, screen.s_mid,
        screen.w_rest, screen.s_rest, screen.w_full, screen.s_full,
        screen._cand, screen.cand_logits, screen.gscale,
        gate=gate if gated and ok else None, win=0, pdl=spec.pdl,
        aggregate=True,
        debug=(screen._pre_cand, screen.mid_scores, screen.scores,
               screen._coarse_cand, screen.refine_scores) if screen._tail_debug else None,
    ):
        raise RuntimeError("actual aggregated scrtail unavailable for private full-body timing")
    _choose(spec, gate, gated and ok, out)



def _choose(spec, gate, gated, out):
    # EXACT original EB48 selector, including fifth-step threshold and the
    # original request-adaptive flag. No selector arithmetic/policy is copied.
    from . import nemotron_h_mtp_screen as s
    screen = spec.screen
    adapt, enabled, tau = s._adapt_args()
    s._select_token[(1,)](
        screen.cand_logits, screen._cand, out, N=screen.cand_logits.numel(),
        BLOCK=triton.next_power_of_2(screen.cand_logits.numel()),
        gate_ptr=gate, GATED=gated, THETA=spec.theta, THETA5=spec.theta5,
        adapt_ptr=adapt, ADAPT=enabled, TAU=tau,
        CUM=spec.cumulative, launch_pdl=spec.pdl)


def _select_probe(spec):
    # Analytical confident-token probes make THETA5 and adaptive stop observable,
    # independently of the checkpoint's random-hidden confidence distribution.
    from . import nemotron_h_mtp_kernels as k
    screen, gate, adapt = spec.screen, k._GATE, k._ADAPT
    saved, saved_gate = _snapshot(screen), gate.clone()
    saved_adapt = None if adapt is None else adapt.clone()
    output = _guarded(screen._out)
    try:
        screen._cand.copy_(torch.arange(screen._cand.numel(), device=spec.device))
        screen.cand_logits.fill_(-1000.)
        screen.cand_logits[0] = 1000.  # exact BF16; denom is exactly one
        for kept in (0, 3, 4):
            for flag in (0., .5, 1.):
                for product in (1., .5):
                    for active in (False, True):
                        for stopped in (False, True):
                            bits = 0x3F800000 if product == 1. else 0x3F000000
                            gate.copy_(torch.tensor([int(stopped), kept, int(active), bits],
                                                    dtype=torch.int32, device=spec.device))
                            if adapt is not None:
                                adapt.copy_(torch.tensor([flag, 4., 9., 11.], dtype=torch.float32, device=spec.device))
                            before_adapt = None if adapt is None else adapt.clone()
                            expected = [int(stopped), kept, int(active), bits]
                            if active and not stopped:
                                expected[1] += 1
                                confidence = product if spec.cumulative else 1.
                                threshold = spec.theta5 if expected[1] >= 4 else spec.theta
                                expected[0] = int(confidence < threshold or
                                                  (spec.adapt_enabled and expected[1] >= 4
                                                   and flag < spec.adapt_tau))
                            _choose(spec, gate, True, output[1])
                            wanted = torch.tensor(expected, dtype=torch.int32, device=spec.device)
                            if (not _words(gate, wanted) or int(output[1].item()) != 0
                                    or not _guard(output)
                                    or (adapt is not None and not _words(adapt, before_adapt))):
                                raise RuntimeError("EB48 selector threshold/adaptive/counter contract differs")
    finally:
        _restore(screen, saved)
        gate.copy_(saved_gate)
        if adapt is not None:
            adapt.copy_(saved_adapt)


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
    from . import nemotron_h_mtp_kernels as k
    adaptive = () if k._ADAPT is None else (k._ADAPT.clone(),)
    return (out.clone(), gate.clone(), *[getattr(screen, n).clone() for n in _MUTABLE], *adaptive)


@torch.no_grad()
def _qualify(spec, guard_pairs, debug, timing):
    from . import nemotron_h_mtp_screen as s
    from . import nemotron_h_mtp_kernels as k
    screen = spec.screen
    screen._tail_debug = debug
    _select_probe(spec)
    adapt = k._ADAPT
    adapt_start = None if adapt is None else torch.empty_like(adapt)
    gen = torch.Generator(device=spec.device).manual_seed(2026102841 + int(debug))
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
        h.copy_(torch.randn(h.shape, device=spec.device, generator=gen, dtype=torch.float32)
                * (.001, .25, 1., 32.)[case % 4])
        h.view(torch.int32)[:4] = torch.tensor([0, -2147483648, 1, -2147483647],
                                              dtype=torch.int32, device=spec.device)
        if case == 4:
            h.zero_()
        if case == 5:
            h.copy_(torch.round(h * 4) / 4)
        if case == 6:
            h[:3] = torch.tensor([float("inf"), -float("inf"), float("nan")], dtype=torch.float32, device=spec.device)
        if case == 7:
            h.mul_(1e-20)
        cont.copy_(torch.randn(h.shape, device=spec.device, generator=gen, dtype=torch.float32) * .5)
        gate_start.zero_()
        gate_start[0] = int(stopped)
        gate_start[1] = (0, 2, 3, 4, 5, 0, 3, 4)[case]
        gate_start[2] = int(case != 7)
        gate_start[3] = 0x3F800000 if case % 2 == 0 else 0x3F000000
        if adapt is not None:
            adapt_start.copy_(torch.tensor([(0., .25, .5, 1.)[case % 4],
                                             float(case + 4), float(case + 9), 11.],
                                            dtype=torch.float32, device=spec.device))
            adapt.copy_(adapt_start)

    def reset():
        for n, t in saved.items():
            if n != "_h" or h is not screen._h:
                getattr(screen, n).copy_(t)
        gate.copy_(gate_start)
        if adapt is not None:
            adapt.copy_(adapt_start)

    def check_equal(a, b):
        if not all(_words(x, y) for x, y in zip(a, b)):
            raise RuntimeError("warp-collective complete screen state/output differs")
        if not _guard(hb) or not all(_guard(p) for p in guard_pairs):
            raise RuntimeError("warp-collective private workspace guard changed")

    # First-dot scores and every aux gather are checked BEFORE tail debug may
    # overwrite scores. The oracle is exactly the original gather + first dot.
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
            if not _run(spec, h, True):
                raise RuntimeError("warp-collective actual contract unavailable")
            check_equal(old, [getattr(screen, n) for n in ("_hp", "_hm", "_hr", "scores")])
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
                raise RuntimeError("warp-collective hidden/output guard or read-only changed")
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
            raise RuntimeError("warp-collective eight outputs alias")
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
                        raise RuntimeError("warp-collective changed-input graph guard/read-only differs")
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
                            raise RuntimeError("warp-collective invalid event interval")
                        samples[arm].append(elapsed)
                a, b = (statistics.median(v) for v in samples)
                ratios.append(b / a)
    metrics = statistics.median(ratios), max(ratios)
    if metrics[0] >= .995 or metrics[1] >= 1.01:
        raise RuntimeError(f"warp-collective full-body warm/pressure cost refused {metrics}")
    return metrics


@torch.no_grad()
def prepare(screen):
    from . import nemotron_h_mtp_kernels as k
    if not _ENABLED or torch.cuda.is_current_stream_capturing():
        return
    spec = _owner(screen)
    if spec is None:
        return
    original_gate, original_debug = k._GATE, screen._tail_debug
    original_adapt = k._ADAPT
    original_adapt_value = None if original_adapt is None else original_adapt.clone()
    adapt_guard = None if original_adapt is None else _guarded(original_adapt)
    original_fields = {n: getattr(screen, n) for n in _MUTABLE}
    original_values = {n: t.clone() for n, t in original_fields.items()}
    readonly = {n: t.clone() for n, t in spec.immutable.items()}
    k._GATE = torch.zeros(4, dtype=torch.int32, device=spec.device)
    if adapt_guard is not None:
        adapt_guard[1].copy_(original_adapt_value)
        k._ADAPT = adapt_guard[1]
    error, metrics = None, None
    try:
        guarded = {n: _guarded(t) for n, t in original_fields.items()}
        for n, pair in guarded.items():
            pair[1].copy_(original_values[n])
            setattr(screen, n, pair[1])
        private_spec = _owner(screen)
        if private_spec is None:
            raise RuntimeError("private guarded actual screen contract unavailable")
        adaptive_guards = () if adapt_guard is None else (adapt_guard,)
        _qualify(private_spec, tuple(guarded.values()) + adaptive_guards, debug=True, timing=False)
        for n, t in original_fields.items():
            setattr(screen, n, t)
        _restore(screen, original_values)
        k._ADAPT = original_adapt
        if original_adapt is not None:
            original_adapt.copy_(original_adapt_value)
        metrics = _qualify(spec, (), debug=False, timing=True)
    except Exception as exc:
        error = exc
    finally:
        try:
            torch.cuda.synchronize(spec.device)
            for n, t in original_fields.items():
                setattr(screen, n, t)
            _restore(screen, original_values)
            k._ADAPT = original_adapt
            if original_adapt is not None:
                original_adapt.copy_(original_adapt_value)
                if not _words(original_adapt, original_adapt_value):
                    raise RuntimeError("EB48 actual adaptive owner restoration failed")
            if not all(_words(getattr(screen, n), t) for n, t in original_values.items()):
                raise RuntimeError("warp-collective mutable owner restoration failed")
        finally:
            screen._tail_debug = original_debug
            k._GATE = original_gate
            k._ADAPT = original_adapt
    if not all(_words(getattr(screen, n), t) for n, t in readonly.items()):
        raise RuntimeError("warp-collective immutable actual screen weight/scale/column modified")
    if error is not None:
        print(f"MTP_SCREEN_TAIL_WARP_AGG refused: {error!r}; original kept", flush=True)
        return
    _ENTRIES[id(screen)] = spec
    print(f"MTP_SCREEN_TAIL_WARP_AGG admitted full-body ratios={metrics}", flush=True)

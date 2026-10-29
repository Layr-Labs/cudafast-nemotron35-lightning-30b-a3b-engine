"""Official-remote-only admission and execution of the A16 router variant.

The parent FP32 kernel, W16 shadow, row mask and stream policy remain the
reference. A refusal occurs before any candidate production launch.
"""
from __future__ import annotations

import sys
import weakref

import torch

_CONVERSION: dict = {}
_ADMITTED: dict = {}


def _sk():
    return sys.modules[__package__]


def _equal(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


def _call(x, w, out, w16, vrows):
    sk = _sk()
    rc = sk._lib.sk_router_gemm_a16(
        x.data_ptr(), x.stride(0), x.shape[0],
        (w if w16 is None else w16).data_ptr(), out.data_ptr(), out.stride(0),
        sk.ROUTER_ROWS, int(w16 is not None), sk.PDL,
        None if vrows is None else vrows.data_ptr(), sk._stream(x),
    )
    if rc != 0:
        raise RuntimeError(f"smallk A16 launch failed: {rc}")
    return out


def _conversion(device):
    sk = _sk()
    if device in _CONVERSION:
        return _CONVERSION[device]
    if sk._lib.sk_router_a16_init() != 0:
        _CONVERSION[device] = False
        return False
    # Exact conversion oracle, including zeros, subnormals, infinities and
    # every NaN payload. No production request buffer is used or modified.
    raw = torch.arange(65536, dtype=torch.int32, device=device).to(torch.uint16)
    bf = raw.view(torch.bfloat16)
    expected = bf.float()
    got = torch.empty_like(expected)
    rc = sk._lib.sk_router_a16_convert(bf.data_ptr(), got.data_ptr(), 65536, sk._stream(bf))
    if rc != 0:
        raise RuntimeError(f"smallk A16 conversion launch failed: {rc}")
    good = _equal(got, expected)
    _CONVERSION[device] = good
    return good


def _qualify(m, w, w16):
    # All private allocations and graph uses belong to one nondefault stream.
    # Persistent weights are ready after the caller wait. No request buffers
    # are handed to that stream; the caller rejoins even on a visible error.
    caller = torch.cuda.current_stream(w.device)
    private = torch.cuda.Stream(device=w.device)
    private.wait_stream(caller)
    try:
        with torch.cuda.stream(private):
            return _qualify_on_stream(m, w, w16)
    finally:
        caller.wait_stream(private)


def _qualify_on_stream(m, w, w16):
    """Private actual-weight eager/changed-input graph checks and pure-kernel timing.

    The original timing input is already FP32: its cast cost is deliberately
    omitted rather than estimating an unobserved Inductor fusion boundary.
    """
    # Keep every local Tensor and CUDAGraph alive until this stream is drained,
    # including exceptions during capture, replay, event recording or comparison.
    stream = torch.cuda.current_stream(w.device)
    try:
        sk = _sk()
        if not _conversion(w.device):
            return False
        k = 2688
        staged = torch.empty((m, k), device=w.device, dtype=torch.bfloat16)
        fp = torch.empty((m, k), device=w.device, dtype=torch.float32)
        vrows = torch.tensor([m], device=w.device, dtype=torch.int32)
        old_guard = torch.full((m * 128 + 32,), 123.0, device=w.device, dtype=torch.float32)
        new_guard = old_guard.clone()
        old = old_guard[16:-16].view(m, 128)
        new = new_guard[16:-16].view(m, 128)
        base = torch.arange(m * k, device=w.device, dtype=torch.int32).view(m, k)
        patterns = [
            ((base % 67).float() - 33).mul_(0.03125).to(torch.bfloat16),
            ((base % 97).float() - 48).mul_(0.00048828125).to(torch.bfloat16),
            ((base % 2) * 32768).to(torch.uint16).view(torch.bfloat16),
        ]
        edges = torch.tensor(
            [0, 32768, 1, 32769, 0x007f, 0x0080, 0x7f7f, 0xff7f,
             0x7f80, 0xff80, 0x7fc1, 0x7f81, 0xffc1, 0xff81],
            device=w.device, dtype=torch.int32,
        )
        patterns.append(edges[base % edges.numel()].to(torch.uint16).view(torch.bfloat16))
        weight_before = w.clone()
        shadow_before = None if w16 is None else w16.clone()
        def original():
            sk.router_gemm(fp, w, out=old, vrows=vrows, w16=w16)
        def candidate():
            _call(staged, w, new, w16, vrows)
        for pattern in patterns:
            staged.copy_(pattern)
            fp.copy_(pattern)
            for count in range(m + 1):
                vrows.fill_(count)
                old_guard.fill_(123.0)
                new_guard.fill_(123.0)
                original()
                candidate()
                if not _equal(old_guard, new_guard):
                    return False
            if not torch.equal(staged.view(torch.uint16), pattern.view(torch.uint16)):
                return False
        if not _equal(w, weight_before) or (w16 is not None and not torch.equal(w16.view(torch.uint16), shadow_before.view(torch.uint16))):
            return False
        # Current-stream graph capture also exercises PDL after changed producers.
        # Copy operations inside both graphs precede their router and stay private.
        source = patterns[0].clone()
        original_graph = torch.cuda.CUDAGraph()
        candidate_graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.current_stream(w.device)
        torch.cuda.synchronize(w.device)
        with torch.cuda.graph(original_graph, stream=stream):
            fp.copy_(source)
            original()
        with torch.cuda.graph(candidate_graph, stream=stream):
            staged.copy_(source)
            candidate()
        for pattern in patterns[:2]:
            source.copy_(pattern)
            for count in (1, m):
                vrows.fill_(count)
                original_graph.replay()
                candidate_graph.replay()
                if not _equal(old_guard, new_guard):
                    return False
        # Separate pure-kernel graphs; no reference conversion/copy launch credit.
        vrows.fill_(m)
        staged.copy_(patterns[0])
        fp.copy_(patterns[0])
        old_time = torch.cuda.CUDAGraph()
        new_time = torch.cuda.CUDAGraph()
        with torch.cuda.graph(old_time, stream=stream):
            for _ in range(8):
                original()
        with torch.cuda.graph(new_time, stream=stream):
            for _ in range(8):
                candidate()
        pressure = torch.empty(64 * 1024 * 1024, device=w.device, dtype=torch.uint8)
        def elapsed(graph, cold):
            if cold:
                pressure.fill_(37)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(stream)
            graph.replay()
            end.record(stream)
            end.synchronize()
            return begin.elapsed_time(end)
        ratios = []
        for cold in (False, True):
            old_time.replay()
            new_time.replay()
            for i in range(5):
                if i % 2:
                    b, a = elapsed(new_time, cold), elapsed(old_time, cold)
                else:
                    a, b = elapsed(old_time, cold), elapsed(new_time, cold)
                ratios.append(b / a if a > 0 else float("inf"))
        ratios.sort()
        return ratios[len(ratios) // 2] < 0.995 and max(ratios) < 1.01

    finally:
        stream.synchronize()


def _supported_weight(w, m):
    sk = _sk()
    if (not sk.ROUTER_A16_ON or m not in (5, 6)
        or w.ndim != 2 or w.dtype != torch.float32
        or w.shape != (128, 2688) or not w.is_cuda or not w.is_contiguous()
        or sk.ROUTER_ROWS not in (1, 2, 4, 8)
        or (m == 5 and not sk.ROUTER_M5_ON)):
        return False
    if w.device.index != torch.cuda.current_device():
        return False
    props = torch.cuda.get_device_properties(w.device)
    return ((props.major, props.minor) == (12, 1)
            and sk._usable() and sk._has_vrows_entry and sk._has_a16_entry)


def _key(w, m, w16):
    sk = _sk()
    return (w.device, m, w.data_ptr(), w._version,
            None if w16 is None else (w16.data_ptr(), w16._version),
            sk.ROUTER_ROWS, sk.PDL)


def _hit(entry, w, w16):
    return (entry is not None and entry[0]() is w
            and (entry[1] is None if w16 is None else entry[1] is not None and entry[1]() is w16)
            and entry[2])


def prepare(w, m):
    """Model post-load only: finish admission before choosing the outer cast.

    A mismatch/refusal returns False before any production work. Native CUDA,
    capture and replay errors propagate after draining private owners; they
    cannot be safely hidden by retrying a production forward.
    """
    if torch.cuda.is_current_stream_capturing() or not _supported_weight(w, m):
        return False
    sk = _sk()
    w16 = sk.gate_w16(w)
    key = _key(w, m, w16)
    entry = _ADMITTED.get(key)
    if entry is None or entry[0]() is not w or (w16 is not None and (entry[1] is None or entry[1]() is not w16)):
        good = _qualify(m, w, w16)
        # Weak owner identities plus exact pointer/version keys reject address
        # reuse without retaining another model's weights or private graph pool.
        entry = (weakref.ref(w), None if w16 is None else weakref.ref(w16), good)
        _ADMITTED[key] = entry
    return _hit(entry, w, w16)


def maybe_router_gemm(x, w, row_mask=False):
    """Production lookup only; no private qualification or capture is started."""
    sk = _sk()
    if (x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_cuda
        or x.shape[0] not in (5, 6) or x.shape[1] != 2688
        or not x.is_contiguous() or w.device != x.device
        or not _supported_weight(w, x.shape[0])):
        return None
    capturing = torch.cuda.is_current_stream_capturing()
    w16 = sk.gate_w16(w)
    if not _hit(_ADMITTED.get(_key(w, x.shape[0], w16)), w, w16):
        return None
    vrows = None
    if row_mask:
        if sk._vrows_wanted and not sk._vrows and not capturing:
            sk.enable_verify_rows(x.device)
        vrows = sk.verify_rows(x.device)
        if vrows is None and capturing:
            sk._vrows_missed = True
    out = torch.empty((x.shape[0], 128), dtype=torch.float32, device=x.device)
    return _call(x, w, out, w16, vrows)

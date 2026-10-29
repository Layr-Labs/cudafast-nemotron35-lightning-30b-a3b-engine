"""Official-remote-only admission for source-dead target router row chains.

Only immutable gate identities and decisions persist; no request/result/graph
is cached. The production input remains the baseline FP32 conversion.
"""
from __future__ import annotations

import sys
import weakref

import torch

_READY: dict = {}
_ADMITTED: dict = {}


def _sk():
    return sys.modules[__package__]


def _equal(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


def _call(x, w, out, w16, vrows):
    sk = _sk()
    rc = sk._lib.sk_router_gemm_suffix(
        x.data_ptr(), x.stride(0), x.shape[0],
        (w if w16 is None else w16).data_ptr(), out.data_ptr(), out.stride(0),
        sk.ROUTER_ROWS, int(w16 is not None), sk.PDL,
        vrows.data_ptr(), sk._stream(x),
    )
    if rc != 0:
        raise RuntimeError(f"smallk suffix launch failed: {rc}")
    return out


def _supported_weight(w, m):
    sk = _sk()
    if (not sk.ROUTER_SUFFIX_ON or m not in (5, 6)
        or w.ndim != 2 or w.dtype != torch.float32
        or w.shape != (128, 2688) or not w.is_cuda or not w.is_contiguous()
        or sk.ROUTER_ROWS not in (1, 2, 4, 8)
        or (m == 5 and not sk.ROUTER_M5_ON)):
        return False
    if w.device.index != torch.cuda.current_device():
        return False
    props = torch.cuda.get_device_properties(w.device)
    return ((props.major, props.minor) == (12, 1)
            and sk._usable() and sk._has_vrows_entry and sk._has_suffix_entry)


def _key(w, m, w16):
    sk = _sk()
    # A baseline shadow first created during an unsupported inference call
    # has no version counter. Refuse before reading _version or admission.
    if w.is_inference() or (w16 is not None and w16.is_inference()):
        return None
    return (w.device, m, w.data_ptr(), w._version,
            None if w16 is None else (w16.data_ptr(), w16._version),
            sk.ROUTER_ROWS, sk.PDL)


def _hit(entry, w, w16):
    return (entry is not None and entry[0]() is w
            and (entry[1] is None if w16 is None else entry[1] is not None and entry[1]() is w16)
            and entry[2])


def _qualify(m, w, w16):
    caller = torch.cuda.current_stream(w.device)
    private = torch.cuda.Stream(device=w.device)
    private.wait_stream(caller)
    try:
        with torch.cuda.stream(private):
            return _qualify_on_stream(m, w, w16)
    finally:
        caller.wait_stream(private)


def _qualify_on_stream(m, w, w16):
    """Keep local owners until finally drains, including any capture exception."""
    stream = torch.cuda.current_stream(w.device)
    try:
        sk = _sk()
        if w.device not in _READY:
            _READY[w.device] = sk._lib.sk_router_suffix_init() == 0
        if not _READY[w.device]:
            return False
        k = 2688
        xo = torch.empty((m, k), dtype=torch.float32, device=w.device)
        xn = torch.empty_like(xo)
        vo = torch.tensor([m], dtype=torch.int32, device=w.device)
        vn = torch.empty_like(vo)
        old_guard = torch.full((m * 128 + 32,), 123.0, dtype=torch.float32, device=w.device)
        new_guard = torch.full_like(old_guard, -237.0)
        old = old_guard[16:-16].view(m, 128)
        new = new_guard[16:-16].view(m, 128)
        old_edges = torch.cat((old_guard[:16], old_guard[-16:])).clone()
        new_edges = torch.cat((new_guard[:16], new_guard[-16:])).clone()
        weight_before = w.clone()
        shadow_before = None if w16 is None else w16.clone()
        base = torch.arange(m * k, dtype=torch.int32, device=w.device).view(m, k)
        edge_words = torch.tensor(
            [0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
             0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
             0x7fc00001, 0x7f800001, 0xffc00001, 0xff800001],
            dtype=torch.int64, device=w.device,
        ).to(torch.int32)
        patterns = [
            ((base % 67).float() - 33).mul_(0.03125),
            ((base % 97).float() - 48).mul_(0.00048828125),
            ((base % 2).to(torch.int64) * 0x80000000).to(torch.int32).view(torch.float32),
            edge_words[base % edge_words.numel()].view(torch.float32),
        ]
        counts = (-2147483648, -1, 0, *range(1, m + 1), m + 1, sk.VROWS_ALL, 2147483647)

        def original():
            sk.router_gemm(xo, w, out=old, vrows=vo, w16=w16)

        def candidate():
            _call(xn, w, new, w16, vn)

        def poisoned():
            old_guard.fill_(123.0)
            new_guard.fill_(-237.0)

        def same():
            return (_equal(old, new)
                    and _equal(torch.cat((old_guard[:16], old_guard[-16:])), old_edges)
                    and _equal(torch.cat((new_guard[:16], new_guard[-16:])), new_edges))

        for pattern in patterns:
            xo.copy_(pattern)
            xn.copy_(pattern)
            for count in counts:
                vo.fill_(count)
                vn.fill_(count)
                expected_count = vo.clone()
                poisoned()
                original()
                candidate()
                if (not same() or not torch.equal(vo, expected_count)
                        or not torch.equal(vn, expected_count)):
                    return False
            if not _equal(xo, pattern) or not _equal(xn, pattern):
                return False

        # Each captured router has its own private current input and count.
        # The copies are recorded before it; replay changes the source values.
        source = patterns[0].clone()
        count_source = vo.clone()
        og = torch.cuda.CUDAGraph()
        ng = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(w.device)
        with torch.cuda.graph(og, stream=stream):
            xo.copy_(source)
            vo.copy_(count_source)
            original()
        with torch.cuda.graph(ng, stream=stream):
            xn.copy_(source)
            vn.copy_(count_source)
            candidate()
        # Alternate counts rather than monotonically extending one prefix.
        replay_counts = (m, 0, m - 1, -1, 1, sk.VROWS_ALL, *reversed(range(m + 1)),
                         2147483647, -2147483648, m + 1)
        for pattern in patterns:
            source.copy_(pattern)
            for count in replay_counts:
                count_source.fill_(count)
                poisoned()
                og.replay()
                ng.replay()
                if (not same() or not torch.equal(vo, count_source)
                        or not torch.equal(vn, count_source)):
                    return False
            if (not _equal(source, pattern) or not _equal(xo, pattern)
                    or not _equal(xn, pattern)):
                return False

        if not _equal(w, weight_before) or (w16 is not None and not torch.equal(
                w16.view(torch.uint16), shadow_before.view(torch.uint16))):
            return False

        # Kernel-only timing does not credit removal of the unchanged cast.
        # The uniform count mixture is an admission rule, not observed traffic.
        xo.copy_(patterns[0])
        xn.copy_(patterns[0])
        old_time = torch.cuda.CUDAGraph()
        new_time = torch.cuda.CUDAGraph()
        with torch.cuda.graph(old_time, stream=stream):
            for _ in range(8):
                original()
        with torch.cuda.graph(new_time, stream=stream):
            for _ in range(8):
                candidate()
        pressure = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=w.device)

        def elapsed(graph, cold):
            if cold:
                pressure.fill_(37)
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record(stream)
            graph.replay()
            end.record(stream)
            end.synchronize()
            return begin.elapsed_time(end)

        ratios = []
        for count in (*range(1, m + 1), sk.VROWS_ALL):
            vo.fill_(count)
            vn.fill_(count)
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
        good = ratios[len(ratios) // 2] < 0.995 and max(ratios) < 1.01
        return (good and same() and _equal(xo, patterns[0])
                and _equal(xn, patterns[0]) and _equal(w, weight_before)
                and torch.equal(vo, vn)
                and (w16 is None or torch.equal(
                    w16.view(torch.uint16), shadow_before.view(torch.uint16))))
    finally:
        # A wait inserted only after this frame unwinds would be too late to
        # own capture/graph tensors through outstanding private GPU work.
        stream.synchronize()


def prepare(w, m):
    if torch.cuda.is_current_stream_capturing() or not _supported_weight(w, m):
        return False
    sk = _sk()
    w16 = sk.gate_w16(w)
    key = _key(w, m, w16)
    if key is None:
        return False
    entry = _ADMITTED.get(key)
    if entry is None or entry[0]() is not w or (w16 is not None and (
            entry[1] is None or entry[1]() is not w16)):
        good = _qualify(m, w, w16)
        entry = (weakref.ref(w), None if w16 is None else weakref.ref(w16), good)
        _ADMITTED[key] = entry
    return _hit(entry, w, w16)


def maybe_router_gemm(x, w, w16, vrows):
    """Production metadata/cache lookup only; never private tests or capture."""
    if (x.ndim != 2 or x.dtype != torch.float32 or not x.is_cuda
        or x.shape[0] not in (5, 6) or x.shape[1] != 2688
        or not x.is_contiguous() or x.device != w.device
        or vrows.dtype != torch.int32 or vrows.shape != (1,)
        or vrows.device != x.device or not vrows.is_contiguous()
        or not _supported_weight(w, x.shape[0])):
        return None
    key = _key(w, x.shape[0], w16)
    if key is None or not _hit(_ADMITTED.get(key), w, w16):
        return None
    out = torch.empty((x.shape[0], 128), dtype=torch.float32, device=x.device)
    return _call(x, w, out, w16, vrows)

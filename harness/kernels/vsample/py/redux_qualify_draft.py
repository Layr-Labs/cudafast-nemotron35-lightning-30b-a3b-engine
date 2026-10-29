"""Remote-only qualification for vsample warp redux.

The candidate runtime may call this outside CUDA graph capture. Run no part of
it locally. It compares the original and alternate CUDA libraries using separate
scratch and the full greedy verify ABI. Until official validation, all behavior
and any performance impact remain unverified.
"""
from __future__ import annotations

import ctypes
import statistics
import torch


def _bind(lib):
    vp, ci, cl = ctypes.c_void_p, ctypes.c_int, ctypes.c_long
    lib.vs_greedy_verify.argtypes = [vp, cl, ci, ci, vp, vp, vp, vp, vp, vp, ci,
                                     vp, vp, vp, vp, ci, vp]
    lib.vs_greedy_verify.restype = ci
    return lib


def _bufs(device, out_len):
    return (torch.empty((out_len,), dtype=torch.int32, device=device),
            torch.empty((1,), dtype=torch.int32, device=device),
            torch.empty((9 * 64,), dtype=torch.float32, device=device),
            torch.empty((9 * 64,), dtype=torch.int32, device=device),
            torch.zeros((1,), dtype=torch.int32, device=device))


def _launch(lib, logits, target, bonus, cu, draft, vrows, out_len, scratch, pdl):
    out, accepted, pval, pidx, counter = scratch
    rc = lib.vs_greedy_verify(
        logits.data_ptr(), logits.stride(0), logits.shape[0], logits.shape[1],
        target.data_ptr(), bonus.data_ptr(), cu.data_ptr(), draft.data_ptr(),
        None if vrows is None else vrows.data_ptr(), out.data_ptr(), out_len,
        accepted.data_ptr(), pval.data_ptr(), pidx.data_ptr(), counter.data_ptr(),
        pdl, torch.cuda.current_stream(logits.device).cuda_stream)
    if rc:
        raise RuntimeError(f"vsample qualification launch failed: {rc}")


def _check_pair(old, new, logits, target, bonus, cu, draft, vrows, out_len, pdl):
    a, b = _bufs(logits.device, out_len), _bufs(logits.device, out_len)
    va = None if vrows is None else vrows.clone()
    vb = None if vrows is None else vrows.clone()
    _launch(old, logits, target, bonus, cu, draft, va, out_len, a, pdl)
    _launch(new, logits, target, bonus, cu, draft, vb, out_len, b, pdl)
    return (torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
            and torch.equal(a[4], b[4])
            and (va is None or torch.equal(va, vb)))


def _synthetic_cases(old, new, device, pdl):
    # Full-ABI cases stress lane, warp and chunk boundaries and altered row strides.
    for rows, vocab, ld in ((2, 8, 8), (5, 4104, 4112),
                            (6, 131072, 131136), (9, 131072, 131072)):
        d = rows - 1
        target = torch.arange(d, dtype=torch.int32, device=device)
        bonus = torch.tensor([d], dtype=torch.int32, device=device)
        cu = torch.tensor([d], dtype=torch.int32, device=device)
        for kind in range(6):
            base = torch.full((rows, ld), -1.0, dtype=torch.bfloat16, device=device)
            logits = base[:, :vocab]
            if kind == 0:
                logits.fill_(float('-inf'))
            elif kind == 1:
                logits[:, 0] = -0.0
                logits[:, min(vocab - 1, 7)] = 0.0
            elif kind == 2:
                logits[:, min(vocab - 1, 7)] = float('inf')
                logits[:, min(vocab - 1, 4096)] = float('inf')
            elif kind == 3:
                logits[:, min(vocab - 1, 31)] = float('nan')
                logits[:, min(vocab - 1, 4096)] = float('nan')
            elif kind == 4:
                logits[:, min(vocab - 1, 31)] = 7.0
                logits[:, min(vocab - 1, 4096)] = 7.0
            else:
                # BF16 subnormal payloads: raw representation avoids host casts.
                raw = base.view(torch.int16)
                raw[:, 0] = 1
                raw[:, min(vocab - 1, 7)] = 2
            draft = logits.argmax(-1)[:d].to(torch.int32)
            for depth in (None, 0, 1, rows, 1 << 30):
                vr = None if depth is None else torch.tensor([depth], dtype=torch.int32, device=device)
                if not _check_pair(old, new, logits, target, bonus, cu, draft, vr, rows, pdl):
                    return False
    return True


def _check_streams(old, new, logits, target, bonus, cu, draft, vrows, out_len, pdl):
    # Two distinct output/counter sets and streams; this does not exercise graph
    # capture, which the production entry point currently explicitly refuses.
    device = logits.device
    parent = torch.cuda.current_stream(device)
    streams = (torch.cuda.Stream(device=device), torch.cuda.Stream(device=device))
    a, b = _bufs(device, out_len), _bufs(device, out_len)
    va = None if vrows is None else vrows.clone()
    vb = None if vrows is None else vrows.clone()
    for stream, lib, scratch, vr in zip(streams, (old, new), (a, b), (va, vb)):
        stream.wait_stream(parent)
        with torch.cuda.stream(stream):
            _launch(lib, logits, target, bonus, cu, draft, vr, out_len, scratch, pdl)
    for stream in streams:
        parent.wait_stream(stream)
    return (torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
            and torch.equal(a[4], b[4])
            and (va is None or torch.equal(va, vb)))


def qualify(old_path, new_path, logits, target, bonus, cu, draft, vrows, out_len, pdl):
    """Return metrics or None; a failure never admits the alternate library.

    Timings compare the full production ABI on the caller's real shape and data.
    Qualification allocates private output/scratch and clones vrows; it must be
    called only once per guarded shape, outside capture, before publication.
    """
    if torch.cuda.is_current_stream_capturing():
        return None
    try:
        old, new = _bind(ctypes.CDLL(old_path)), _bind(ctypes.CDLL(new_path))
        new.vs_redux_pair_probe.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        new.vs_redux_pair_probe.restype = ctypes.c_int
        mismatch = torch.empty((65536 * 16,), dtype=torch.int32, device=logits.device)
        rc = new.vs_redux_pair_probe(mismatch.data_ptr(),
                                     torch.cuda.current_stream(logits.device).cuda_stream)
        if rc or torch.count_nonzero(mismatch).item() != 0:
            return None
        if not _synthetic_cases(old, new, logits.device, pdl):
            return None
        if not _check_pair(old, new, logits, target, bonus, cu, draft, vrows, out_len, pdl):
            return None
        if not _check_streams(old, new, logits, target, bonus, cu, draft, vrows, out_len, pdl):
            return None
        a, b = _bufs(logits.device, out_len), _bufs(logits.device, out_len)
        va = None if vrows is None else vrows.clone()
        vb = None if vrows is None else vrows.clone()
        for _ in range(4):
            _launch(old, logits, target, bonus, cu, draft, va, out_len, a, pdl)
            _launch(new, logits, target, bonus, cu, draft, vb, out_len, b, pdl)
        # Optional pressure buffer rotates access through 64 MiB. The data path
        # is deliberately outside each event interval; it is a cache challenge,
        # not evidence of actual concurrent model traffic or weight residency.
        pressure = torch.zeros((16 * 1024 * 1024,), dtype=torch.int32, device=logits.device)
        results = {}
        for mode in ("warm", "pressure"):
            measurements = [[], []]
            for iteration in range(16):
                for n in ((0, 1) if iteration % 2 == 0 else (1, 0)):
                    if mode == "pressure":
                        pressure.add_(1)
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(8 if mode == "warm" else 1):
                        _launch((old, new)[n], logits, target, bonus, cu, draft,
                                (va, vb)[n], out_len, (a, b)[n], pdl)
                    end.record()
                    end.synchronize()
                    measurements[n].append(start.elapsed_time(end))
            ratios = [b / a for a, b in zip(*measurements) if a > 0]
            if len(ratios) != 16 or statistics.median(ratios) > 0.98 or sum(x < 1 for x in ratios) < 12:
                return None
            results[mode] = {"median_ratio": statistics.median(ratios),
                             "wins": sum(x < 1 for x in ratios)}
        return results
    except Exception:
        # A device-side launch failure may poison the CUDA context. Propagate it
        # rather than presenting a potentially unsafe fallback as success.
        torch.cuda.synchronize(logits.device)
        return None

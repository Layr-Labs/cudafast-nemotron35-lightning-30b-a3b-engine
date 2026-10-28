"""Private qualification helpers for the optional empty-escape decoder.

The main package may select the separate decoder only after these private
checks. All execution and timing belong to the official remote worker.
"""
from __future__ import annotations

import ctypes
import statistics
from pathlib import Path

import torch


def load_optional():
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name("libfp8c_empty.so")))
        lib.fp8c_init.argtypes = []
        lib.fp8c_init.restype = ctypes.c_int
        lib.fp8c_has_pdl.argtypes = []
        lib.fp8c_has_pdl.restype = ctypes.c_int
        lib.fp8c_run.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
        lib.fp8c_run.restype = ctypes.c_int
        lib.fp8c_decode.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.fp8c_decode.restype = ctypes.c_int
        if lib.fp8c_init() != 0:
            return None
        return lib
    except (OSError, AttributeError):
        return None


def _run(lib, a, buf, sa, sb, out, n, k, ring, pdl):
    rc = lib.fp8c_run(a.data_ptr(), buf.data_ptr(), sa.data_ptr(), sb.data_ptr(),
                      out.data_ptr(), a.shape[0], n, k, buf.shape[1], int(pdl), ring,
                      torch.cuda.current_stream(a.device).cuda_stream)
    if rc:
        raise RuntimeError(f"fp8c qualification launch failed: {rc}")


def _measure(graph, repetitions=5, pressure=None):
    first = torch.cuda.Event(enable_timing=True)
    last = torch.cuda.Event(enable_timing=True)
    if pressure is not None:
        elapsed = []
        for _ in range(repetitions):
            # Submit the cache-pressure work before the timed event so the
            # timing covers the complete graph call, not the pressure itself.
            pressure.add_(1)
            first.record()
            graph.replay()
            last.record()
            last.synchronize()
            elapsed.append(first.elapsed_time(last))
        return statistics.median(elapsed)
    first.record()
    for _ in range(repetitions):
        graph.replay()
    last.record()
    last.synchronize()
    return first.elapsed_time(last) / repetitions


@torch.no_grad()
def qualify_synthetic(base, alt, buf, n, k, header_bytes):
    """Compare decoders on well-formed empty, one- and two-escape tiles.

    The remaining tiles are the verified real owner's unmodified bytes.  The
    synthetic tile's offsets, parameters, main planes and exception bytes
    are built consistently, without calling an independent candidate encoder.
    """
    chunks = k // 128
    main_bytes = chunks * 3584
    raw = buf.clone()
    raw[0, :header_bytes + main_bytes + 16].zero_()
    # esc=1, e0=0 in every column; all zero main codes are ordinary.
    raw[0, header_bytes - 63:header_bytes:2] = 1
    reference = torch.empty((n, k), dtype=torch.uint8, device=buf.device)
    candidate = torch.empty_like(reference)
    for case in (0, 1, 2):
        if case == 1:
            # First lane/word/byte has code 1.  It consumes one exception;
            # every subsequent warp/chunk starts and ends at offset 1.
            raw[0, header_bytes] = 1
            offsets = raw[0, :(4 * chunks + 1) * 2].view(torch.int16)
            offsets[1:] = 1
            raw[0, header_bytes + main_bytes] = 0xFF
        elif case == 2:
            # A second escape in warp 1 starts at an unaligned offset 1.
            raw[0, header_bytes + 512] = 1
            offsets[2:] = 2
            raw[0, header_bytes + main_bytes + 1] = 0x80
        for lib, out in ((base, reference), (alt, candidate)):
            rc = lib.fp8c_decode(raw.data_ptr(), raw.shape[1], out.data_ptr(), n, k,
                                 torch.cuda.current_stream(buf.device).cuda_stream)
            if rc:
                return False
        if not torch.equal(reference, candidate):
            return False
    return True


@torch.no_grad()
def qualify(base, alt, packed, buf, sa, sb, pdl, ring, header_bytes):
    """Compare one actual owner, with private outputs and graph replays.

    A false result keeps the original kernel.
    """
    if alt is None or (pdl and alt.fp8c_has_pdl() != 1):
        return False
    n, k = packed.shape
    if (n, k) not in ((10304, 2688), (2688, 4096)) or ring not in (3, 4, 6):
        return False
    saved_buf = buf.clone()
    decoded = torch.empty_like(packed)
    rc = alt.fp8c_decode(buf.data_ptr(), buf.shape[1], decoded.data_ptr(), n, k,
                         torch.cuda.current_stream(buf.device).cuda_stream)
    if rc or not torch.equal(decoded, packed):
        return False
    if not qualify_synthetic(base, alt, buf, n, k, header_bytes):
        return False
    # Private scale owners allow changed-input replay without ever mutating
    # the model's published parameters.
    test_sa, test_sb = sa.clone(), sb.clone()
    pressure = torch.zeros((64 * 1024 * 1024,), device=buf.device, dtype=torch.uint8)
    for m in range(1, 9):
        raw = torch.arange(m * k, device=buf.device, dtype=torch.int32)
        src = ((raw % 71).float() - 35).reshape(m, k).to(torch.float8_e4m3fn)
        a = torch.empty_like(src)
        owner0 = torch.full((m * n + 128,), 0x5A5A, device=buf.device, dtype=torch.int16)
        owner1 = torch.full_like(owner0, 0x5A5A)
        out0 = owner0[64:-64].view(torch.bfloat16).view(m, n)
        out1 = owner1[64:-64].view(torch.bfloat16).view(m, n)
        graphs = []
        for lib, out in ((base, out0), (alt, out1)):
            # Load and warm the exact kernel before capture; a failed launch
            # rejects the optional implementation instead of capturing it.
            a.copy_(src)
            _run(lib, a, buf, test_sa, test_sb, out, n, k, ring, pdl)
            torch.cuda.current_stream(buf.device).synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
                a.copy_(src)
                _run(lib, a, buf, test_sa, test_sb, out, n, k, ring, pdl)
            graphs.append(graph)
        for shift in (0, 1, 2):
            if shift:
                src.copy_(((raw % (43 + shift)).float() - 21).reshape(m, k).to(torch.float8_e4m3fn))
                test_sa.copy_(sa * (0.5 if shift == 1 else 2.0))
                test_sb.copy_(sb * (2.0 if shift == 1 else 0.5))
            graphs[0].replay()
            graphs[1].replay()
            if not torch.equal(out0.view(torch.int16), out1.view(torch.int16)):
                return False
            if not all(bool(torch.all(x == 0x5A5A)) for x in
                       (owner0[:64], owner0[-64:], owner1[:64], owner1[-64:])):
                return False
        if m in (1, 5, 6, 8):
            for pressure_input in (None, pressure):
                samples = [[], []]
                for j in range(6):
                    for idx in ((0, 1) if j % 2 == 0 else (1, 0)):
                        samples[idx].append(_measure(graphs[idx], pressure=pressure_input))
                if statistics.median(samples[1]) >= 0.99 * statistics.median(samples[0]):
                    return False
            graphs[0].replay()
            graphs[1].replay()
            if not torch.equal(out0.view(torch.int16), out1.view(torch.int16)):
                return False
            if not all(bool(torch.all(x == 0x5A5A)) for x in
                       (owner0[:64], owner0[-64:], owner1[:64], owner1[-64:])):
                return False
    # Distinct graphs, streams, input buffers and output owners must coexist.
    # The compressed weight and scales are immutable shared inputs.
    m = 5
    srcs = [(torch.arange(m * k, device=buf.device, dtype=torch.int32)
             .remainder(37 + j).float().reshape(m, k) - 18).to(torch.float8_e4m3fn)
            for j in range(2)]
    ins = [torch.empty_like(x) for x in srcs]
    outs = [torch.empty((m, n), device=buf.device, dtype=torch.bfloat16) for _ in range(2)]
    references = [torch.empty_like(x) for x in outs]
    for j in range(2):
        _run(base, srcs[j], buf, test_sa, test_sb, references[j], n, k, ring, pdl)
    origin = torch.cuda.current_stream(buf.device)
    streams = [torch.cuda.Stream(device=buf.device) for _ in range(2)]
    parallel_graphs = []
    for j, stream in enumerate(streams):
        stream.wait_stream(origin)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(
            graph, pool=torch.cuda.graph_pool_handle(), stream=stream
        ):
            ins[j].copy_(srcs[j])
            _run(alt, ins[j], buf, test_sa, test_sb, outs[j], n, k, ring, pdl)
        parallel_graphs.append(graph)
    for j in range(2):
        with torch.cuda.stream(streams[j]):
            parallel_graphs[j].replay()
    for stream in streams:
        stream.synchronize()
    if not all(torch.equal(outs[j].view(torch.int16), references[j].view(torch.int16))
               for j in range(2)):
        return False
    return torch.equal(buf, saved_buf)

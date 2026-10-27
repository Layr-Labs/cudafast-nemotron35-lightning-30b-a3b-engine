# SPDX-License-Identifier: Apache-2.0
"""fp8c: lossless 7-bit coding of the two Mamba FP8 projection weights for 1..8 rows.

``prepare(packed)`` encodes an fp8small pack (the byte-preserving [N, K] layout
fp8small loads) into one buffer, decodes the whole buffer with the kernel's own
decoder and compares every byte with the pack. It returns None (the caller keeps
fp8small) on any mismatch, unsupported shape or GPU, or an encoding larger than
MAX_RATIO of the original. ``gemm`` then computes exactly what fp8small computes.

FP8C=0 disables it (prepare returns None).
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch

SHAPES = {(10304, 2688), (2688, 4096)}
ENABLED = os.environ.get("FP8C", "1") != "0"
MAX_RATIO = 0.93
MAIN = 3584  # bytes per (tile, chunk of 4 K-steps)
MAX_WARP_CHUNK = 124  # escapes per (tile, chunk, warp): the 32-lane window holds 128 bytes
# Ring depth in chunks of 4 K-steps, per shape (FP8C_RING overrides; 3, 4 or 6).
RING = int(os.environ.get("FP8C_RING", "0"))
if RING not in (0, 3, 4, 6):
    RING = 0
_RING_DEFAULT = {(10304, 2688): 3, (2688, 4096): 4}
_PDL_ENABLED = os.environ.get("FP8C_PDL", "1") != "0"
_lib = None
_load_failed = False
_ready_devices = set()
_pdl = False
stats = {"tensors": 0, "raw_bytes": 0, "bytes": 0, "rejected": 0, "last_reason": "", "fp8e": 0}

# fp8e (csrc/fp8e.cu, its own libfp8e.so): the two-level exponent code. Tried
# after the original fp8c buffer exists; a missing library, window overflow,
# semantic or latency rejection keeps fp8c. Its buffer is [N/32, 1,
# tile_bytes] uint8: the extra dimension tags the format through every alias
# (CMP2 rebinding preserves shape). FP8E=0 disables it.
FP8E_ENABLED = os.environ.get("FP8E", "1") != "0"
E_MAIN = 3072  # bytes per (tile, chunk): SM 2048 + P 1024
E_MAX_REST = 1000  # S codes per (tile, chunk, warp): the 64-word window
E_MAX_EXC = 248  # X nibbles per (tile, chunk, warp): the 32-word window
_elib = None
_eload_failed = False
_eready = set()
_epdl = False


def _load():
    global _lib, _load_failed, _pdl
    if _lib is not None or _load_failed:
        return _lib
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name("libfp8c.so")))
        lib.fp8c_init.argtypes = []
        lib.fp8c_init.restype = ctypes.c_int
        lib.fp8c_has_pdl.argtypes = []
        lib.fp8c_has_pdl.restype = ctypes.c_int
        lib.fp8c_run.argtypes = (
            [ctypes.c_void_p] * 5 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
        )
        lib.fp8c_run.restype = ctypes.c_int
        lib.fp8c_decode.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.fp8c_decode.restype = ctypes.c_int
        lib.fp8c_header_bytes.argtypes = [ctypes.c_int]
        lib.fp8c_header_bytes.restype = ctypes.c_int
        _lib = lib
        _pdl = _PDL_ENABLED and lib.fp8c_has_pdl() == 1
    except OSError:
        _load_failed = True
    return _lib


def header_bytes(k: int) -> int:
    c = k // 128
    return (((4 * c + 1) * 2 + 64) + 127) // 128 * 128


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


@torch.no_grad()
def encode(packed: torch.Tensor):
    """Encode an fp8small pack [N, K] (uint8). Returns (buffer [N/32, tile_bytes] uint8,
    info dict) or None when a warp-chunk holds more escapes than its window."""
    n, k = packed.shape
    t, c = n // 32, k // 128
    dev = packed.device
    # [tile, K/32 step, warp, lane, half, byte] -> [tile, chunk, warp, lane, word=(step,half), byte]
    r = (
        packed.view(t, c, 4, 4, 32, 2, 4)
        .permute(0, 1, 3, 4, 2, 5, 6)
        .reshape(t, c, 4, 32, 8, 4)
    )
    # Per output column (= per lane: column tile*32 + warp*8 + lane//4) parameters.
    rc = r.view(t, c, 4, 8, 4, 8, 4)  # lane -> (lane//4, lane%4)
    exp = (rc >> 3) & 15
    hc = torch.stack([(exp == x).sum((1, 4, 5, 6)) for x in range(16)], -1)  # [t, 4, 8, 16]
    cs = torch.stack([hc[..., x : x + 8].sum(-1) for x in range(9)], -1)  # [t, 4, 8, 9]
    # The 8-exponent window holding most of the column (ties: the higher window).
    e0c = (cs * 16 + torch.arange(9, device=dev)).argmax(-1).to(torch.uint8)  # [t, 4, 8]
    e0b = e0c.view(t, 1, 4, 8, 1, 1, 1)
    win = (exp >= e0b) & (exp <= e0b + 7)
    code = ((rc & 0x80) >> 1) | (((exp - e0b) & 7) << 3) | (rc & 7)
    # Escape code per column: its least frequent window code (ties: the lowest code).
    col = torch.arange(n, device=dev, dtype=torch.int32).view(t, 1, 4, 8, 1, 1, 1)
    idx = torch.where(win, col * 128 + code.to(torch.int32), torch.full_like(col, n * 128))
    counts = torch.bincount(idx.reshape(-1), minlength=n * 128 + 1)[: n * 128].view(t, 4, 8, 128)
    del idx
    escc = counts.argmin(-1).to(torch.uint8)  # [t, 4, 8]
    escb = escc.view(t, 1, 4, 8, 1, 1, 1)
    esc = ~win | (code == escb)
    code = torch.where(esc, escb.expand_as(code), code).reshape(t, c, 4, 32, 8, 4)
    esc = esc.reshape(t, c, 4, 32, 8, 4)
    del exp, win
    # word 7's 7 code bits ride in bit 7 of words 0..6
    shifts = torch.arange(7, device=dev, dtype=torch.uint8).view(7, 1)
    hi = ((code[..., 7:8, :] >> shifts) & 1) << 7
    cw = code[..., :7, :] | hi
    main = torch.cat(
        [
            cw[..., 0:4, :].reshape(t, c, 2048),
            cw[..., 4:6, :].reshape(t, c, 1024),
            cw[..., 6, :].reshape(t, c, 512),
        ],
        dim=2,
    ).reshape(t, c * MAIN)
    del cw, hi, code
    escf = esc.reshape(t, c * 4, 1024)
    cnt = escf.sum(-1, dtype=torch.int32)
    max_wc = int(cnt.max().item())
    if max_wc > MAX_WARP_CHUNK:
        return None
    tot = cnt.sum(-1)
    max_tot = int(tot.max().item())
    if max_tot >= 32768:
        return None
    offs = torch.zeros((t, 4 * c + 1), dtype=torch.int32, device=dev)
    offs[:, 1:] = cnt.cumsum(-1)
    cap = max(_round_up(max_tot, 16), 16)
    vals = r.reshape(t, -1)[esc.reshape(t, -1)]
    tile_of = torch.repeat_interleave(torch.arange(t, device=dev), tot)
    start = tot.cumsum(0) - tot
    pos = torch.arange(vals.numel(), device=dev) - start[tile_of]
    exc = torch.zeros((t, cap), dtype=torch.uint8, device=dev)
    exc[tile_of, pos] = vals
    hdr_bytes = header_bytes(k)
    hdr = torch.zeros((t, hdr_bytes), dtype=torch.uint8, device=dev)
    hdr[:, : (4 * c + 1) * 2] = offs.to(torch.int16).view(torch.uint8)
    params = (e0c.to(torch.int32) | (escc.to(torch.int32) << 8)).view(t, 32)
    hdr[:, hdr_bytes - 64 :] = params.to(torch.int16).view(torch.uint8)
    tile_bytes = _round_up(hdr_bytes + c * MAIN + cap, 128)
    pad = torch.zeros((t, tile_bytes - hdr_bytes - c * MAIN - cap), dtype=torch.uint8, device=dev)
    buf = torch.cat([hdr, main, exc, pad], dim=1).contiguous()
    n_esc = int(tot.sum().item())
    info = dict(
        e0_hist=torch.bincount(e0c.reshape(-1).long(), minlength=9).tolist(), escapes=n_esc,
        escape_rate=n_esc / (n * k), max_warp_chunk=max_wc, cap=cap,
        tile_bytes=tile_bytes, bytes=buf.numel(), ratio=buf.numel() / (n * k),
        bits_per_weight=8.0 * buf.numel() / (n * k),
    )
    return buf, info


def _eload():
    global _elib, _eload_failed, _epdl
    if _elib is not None or _eload_failed:
        return _elib
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name("libfp8e.so")))
        lib.fp8e_init.argtypes = []
        lib.fp8e_init.restype = ctypes.c_int
        lib.fp8e_has_pdl.argtypes = []
        lib.fp8e_has_pdl.restype = ctypes.c_int
        lib.fp8e_run.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
        lib.fp8e_run.restype = ctypes.c_int
        lib.fp8e_decode.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.fp8e_decode.restype = ctypes.c_int
        lib.fp8e_header_bytes.argtypes = [ctypes.c_int]
        lib.fp8e_header_bytes.restype = ctypes.c_int
        _elib = lib
        _epdl = _PDL_ENABLED and lib.fp8e_has_pdl() == 1
    except OSError:
        _eload_failed = True
    return _elib


def header_bytes_e(k: int) -> int:
    c = k // 128
    return _round_up(4 + (4 * c + 1) * 4 + 256, 128)


def _le_bytes(x: torch.Tensor, nbytes: int) -> torch.Tensor:
    """Non-negative int64 values -> little-endian uint8 bytes on a new last dim."""
    return torch.stack([(x >> (8 * i)) & 0xFF for i in range(nbytes)], -1).to(torch.uint8)


def _pack_stream(mask: torch.Tensor, vals: torch.Tensor, bits: int, t: int) -> tuple[torch.Tensor, int]:
    """Per tile, the masked values in order, `bits` each, LSB first in u32 words.
    mask/vals [t, L]. Returns (bytes [t, cap], cap) with cap a multiple of 16."""
    per = 32 // bits
    tot = mask.sum(-1)
    words = int(((tot * bits + 31) // 32).max().item()) if tot.numel() else 0
    cap = max(_round_up(words * 4, 16), 16)
    nw = cap // 4
    pos = mask.to(torch.int64).cumsum(-1) - 1
    tile = torch.arange(t, device=mask.device).view(t, 1).expand_as(mask)
    pos, tile, v = pos[mask], tile[mask], vals[mask].to(torch.int64)
    acc = torch.zeros(t * nw, dtype=torch.int64, device=mask.device)
    acc.scatter_add_(0, tile * nw + pos // per, v << (bits * (pos % per)))
    return _le_bytes(acc.view(t, nw), 4).reshape(t, cap), cap


@torch.no_grad()
def encode_e(packed: torch.Tensor):
    """fp8e encoding of an fp8small pack [N, K] (layout in csrc/fp8e.cu). Returns (buffer
    [N/32, 1, tile_bytes] uint8, info) or None when a window or an offset overflows."""
    n, k = packed.shape
    t, c = n // 32, k // 128
    dev = packed.device
    # [tile, chunk, warp, lane, weight = 4 * word + byte] (word = 2 * step + half, as fp8c)
    r = (
        packed.view(t, c, 4, 4, 32, 2, 4)
        .permute(0, 1, 3, 4, 2, 5, 6)
        .reshape(t, c, 4, 32, 32)
        .to(torch.int32)
    )
    exp = (r >> 3) & 15
    col = exp.view(t, c, 4, 8, 4, 32)  # lane -> (column lane // 4, lane % 4)
    hc = torch.stack([(col == x).sum((1, 4, 5)) for x in range(16)], -1).to(torch.int64)  # [t, 4, 8, 16]
    # Rank the column's exponents by count, ties: the lower exponent first.
    key = hc * 16 + (15 - torch.arange(16, device=dev, dtype=torch.int64))
    order = key.argsort(-1, descending=True)  # exponent at each rank
    rank = order.argsort(-1)  # rank of each exponent
    rk = torch.gather(rank.view(t, 1, 4, 8, 1, 16).expand(t, c, 4, 8, 4, 16), -1, col.long())
    rk = rk.view(t, c, 4, 32, 32)
    del col
    pcode = rk.clamp(max=3)
    rest = rk >= 3
    exc = rk >= 6
    scode = (rk - 3).clamp(0, 3)
    smn = ((r >> 4) & 8) | (r & 7)
    sm_w = (smn.view(t, c, 4, 32, 4, 8).to(torch.int64)
            << (4 * torch.arange(8, device=dev, dtype=torch.int64))).sum(-1)
    p_w = (pcode.view(t, c, 4, 32, 2, 16).to(torch.int64)
           << (2 * torch.arange(16, device=dev, dtype=torch.int64))).sum(-1)
    main = torch.cat(
        [_le_bytes(sm_w, 4).reshape(t, c, 2048), _le_bytes(p_w, 4).reshape(t, c, 1024)], dim=2
    ).reshape(t, c * E_MAIN)
    del sm_w, p_w, smn, pcode
    rc_wc = rest.view(t, c * 4, 1024).sum(-1)
    xc_wc = exc.view(t, c * 4, 1024).sum(-1)
    max_rest, max_exc = int(rc_wc.max().item()), int(xc_wc.max().item())
    if max_rest > E_MAX_REST or max_exc > E_MAX_EXC:
        return None
    soff = torch.zeros((t, 4 * c + 1), dtype=torch.int64, device=dev)
    xoff = torch.zeros_like(soff)
    soff[:, 1:] = rc_wc.cumsum(-1)
    xoff[:, 1:] = xc_wc.cumsum(-1)
    if int(soff[:, -1].max().item()) >= 65536 or int(xoff[:, -1].max().item()) >= 65536:
        return None
    s_bytes, cap_s = _pack_stream(rest.reshape(t, -1), scode.reshape(t, -1), 2, t)
    x_bytes, cap_x = _pack_stream(exc.reshape(t, -1), exp.reshape(t, -1), 4, t)
    hdr_bytes = header_bytes_e(k)
    xbase = hdr_bytes + c * E_MAIN + cap_s
    hdr = torch.zeros((t, hdr_bytes), dtype=torch.uint8, device=dev)
    hdr[:, 0:4] = _le_bytes(torch.full((t,), xbase, dtype=torch.int64, device=dev), 4)
    nb = (4 * c + 1) * 2
    hdr[:, 4 : 4 + nb] = _le_bytes(soff, 2).reshape(t, nb)
    hdr[:, 4 + nb : 4 + 2 * nb] = _le_bytes(xoff, 2).reshape(t, nb)
    lut = torch.zeros((t, 4, 8, 8), dtype=torch.int64, device=dev)
    lut[..., :6] = order[..., :6] << 3
    hdr[:, hdr_bytes - 256 :] = lut.reshape(t, 256).to(torch.uint8)
    tile_bytes = _round_up(xbase + cap_x, 128)
    pad = torch.zeros((t, tile_bytes - xbase - cap_x), dtype=torch.uint8, device=dev)
    buf = torch.cat([hdr, main, s_bytes, x_bytes, pad], dim=1).contiguous().view(t, 1, tile_bytes)
    # Encoded payload footprint, not measured GPU memory traffic. Boundary
    # words may be requested by two adjacent warp-chunks; caches also matter.
    read = t * (hdr_bytes + c * E_MAIN) + 4 * int(
        (((2 * soff[:, -1] + 31) // 32) + ((xoff[:, -1] + 7) // 8)).sum().item())
    info = dict(
        format="fp8e", rest_rate=float(rest.float().mean().item()),
        escape_rate=float(exc.float().mean().item()), max_warp_chunk=max_rest,
        max_warp_chunk_exc=max_exc, tile_bytes=tile_bytes, bytes=buf.numel(),
        ratio=buf.numel() / (n * k), bits_per_weight=8.0 * buf.numel() / (n * k),
        read_bits_per_weight=8.0 * read / (n * k),
    )
    return buf, info


def _eready_dev(device) -> bool:
    if not FP8E_ENABLED or _eload() is None:
        return False
    with torch.cuda.device(device):
        index = torch.cuda.current_device()
        if index not in _eready:
            if _elib.fp8e_init():
                return False
            _eready.add(index)
    return True


def decode_e(buf: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """The fp8e kernel decoder's output, in the fp8small pack layout [N, K] uint8."""
    out = torch.empty((n, k), dtype=torch.uint8, device=buf.device)
    rc = _elib.fp8e_decode(
        buf.data_ptr(), buf.shape[-1], out.data_ptr(), n, k,
        torch.cuda.current_stream(buf.device).cuda_stream,
    )
    if rc:
        raise RuntimeError(f"fp8e decode failed: {rc}")
    return out


def _admit_e_trial(buf: torch.Tensor, refbuf: torch.Tensor, packed: torch.Tensor, n: int, k: int) -> bool:
    """Official-remote boot only; fixed private data, each real weight owner.

    Compare the selected fp8c kernel, fp8small, guarded whole outputs, changed
    graph inputs/scales and warm/cache-pressure latency. No content cache and
    no request data are retained. A library/policy/identity/timing failure
    keeps the original fp8c buffer. Graph timing proves this GEMM only.
    """
    import statistics
    import fp8small

    if torch.cuda.is_current_stream_capturing() or _pdl != _epdl:
        return False
    dev = packed.device
    ring = RING or _RING_DEFAULT[(n, k)]
    stream = torch.cuda.current_stream(dev)
    g = torch.Generator(device=dev).manual_seed(20261027)
    saved = [x.clone() for x in (packed, refbuf, buf)]
    sa = torch.full((1,), 0.0625, device=dev, dtype=torch.float32)
    sb = torch.full((1,), 0.015625, device=dev, dtype=torch.float32)
    pressure = torch.empty(64 * 1024 * 1024, device=dev, dtype=torch.uint8)
    timed = []

    def run(a, out, use_e, pdl):
        lib = _elib if use_e else _lib
        weight = buf if use_e else refbuf
        fn = lib.fp8e_run if use_e else lib.fp8c_run
        rc = fn(a.data_ptr(), weight.data_ptr(), sa.data_ptr(), sb.data_ptr(),
                out.data_ptr(), a.shape[0], n, k, weight.shape[-1],
                int(pdl), ring, stream.cuda_stream)
        if rc:
            raise RuntimeError(f"fp8e admission launch failed: {rc}, new={use_e}")

    def preserved(a, av, backings):
        return (torch.equal(a.view(torch.uint8), av)
                and all(torch.equal(x, y) for x, y in zip((packed, refbuf, buf), saved))
                and all(torch.equal(b[:128], torch.full_like(b[:128], 0x3545))
                        and torch.equal(b[-128:], torch.full_like(b[-128:], 0x3545))
                        for b in backings))

    for m in range(1, 9):
        # Actual contiguous row stride, with front/back allocation guards.
        araw = torch.full((m * k + 512,), 0x5A, device=dev, dtype=torch.uint8)
        a = araw[256:-256].view(torch.float8_e4m3fn).view(m, k)
        b0 = torch.full((m * n + 256,), 0x3545, device=dev, dtype=torch.int16)
        b1 = b0.clone()
        y0 = b0[128:-128].view(torch.bfloat16).view(m, n)
        y1 = b1[128:-128].view(torch.bfloat16).view(m, n)
        if any(x.data_ptr() % 256 for x in (a, y0, y1)):
            return False
        for special in (False, True):
            a.copy_((torch.randn((m, k), generator=g, device=dev) * 4).to(a.dtype))
            if special:
                raw = a.view(torch.uint8).view(-1)
                # Signed zeros, minimum subnormals, largest finite and NaN.
                for j, code in enumerate((0, 128, 1, 129, 126, 254, 127)):
                    raw[j::257] = code
            av = a.view(torch.uint8).clone()
            sv = (sa.clone(), sb.clone())
            want = fp8small.gemm(a, packed, sa, sb).view(torch.int16)
            for pdl in ((False, True) if _pdl else (False,)):
                b0.fill_(0x3545); b1.fill_(0x3545)
                run(a, y0, False, pdl); run(a, y1, True, pdl)
                if not (torch.equal(b0, b1) and torch.equal(y1.view(torch.int16), want)
                        and preserved(a, av, (b0, b1))
                        and torch.equal(sa, sv[0]) and torch.equal(sb, sv[1])):
                    return False
            if not (torch.all(araw[:256] == 0x5A) and torch.all(araw[-256:] == 0x5A)):
                return False
        if m not in (1, 5, 8):
            continue
        # Two captures read the same live input/scales and their own output.
        # No policy fallback is allowed to alter the reference while timing.
        graphs = []
        for use_e, out in ((False, y0), (True, y1)):
            for _ in range(3):
                run(a, out, use_e, _pdl)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(a, out, use_e, _pdl)
            graphs.append(graph)
        for j in range(3):
            a.copy_((torch.randn((m, k), generator=g, device=dev) * (j + 1)).to(a.dtype))
            sa.fill_(0.03125 * (j + 1)); sb.fill_(0.0078125 * (j + 1))
            av = a.view(torch.uint8).clone()
            sv = (sa.clone(), sb.clone())
            b0.fill_(0x3545); b1.fill_(0x3545)
            graphs[0].replay(); graphs[1].replay()
            if not (torch.equal(b0, b1) and preserved(a, av, (b0, b1))
                    and torch.equal(sa, sv[0]) and torch.equal(sb, sv[1])):
                return False
        for cold in (False, True):
            samples = [[], []]
            for graph in graphs:
                for _ in range(16):
                    graph.replay()
            stream.synchronize()
            for sample in range(5):
                for which in ((0, 1) if sample % 2 == 0 else (1, 0)):
                    if cold:
                        pressure.fill_(sample + which)
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record(stream)
                    for _ in range(1 if cold else 32):
                        graphs[which].replay()
                    end.record(stream); end.synchronize()
                    value = begin.elapsed_time(end) / (1 if cold else 32)
                    if not value > 0:
                        return False
                    samples[which].append(value)
            old, new = (statistics.median(x) for x in samples)
            timed.append((old, new))
            if not (torch.equal(b0, b1) and preserved(a, av, (b0, b1))
                    and torch.equal(sa, sv[0]) and torch.equal(sb, sv[1])
                    and torch.all(araw[:256] == 0x5A) and torch.all(araw[-256:] == 0x5A)):
                return False
        sa.fill_(0.0625); sb.fill_(0.015625)
    ratios = [new / old for old, new in timed]
    return (len(ratios) == 6 and statistics.median(ratios) < 0.98
            and sum(new for old, new in timed) / sum(old for old, new in timed) < 0.98
            and max(ratios) < 1.03)


def _admit_e(buf: torch.Tensor, refbuf: torch.Tensor, packed: torch.Tensor, n: int, k: int) -> bool:
    # Trial outputs and statistics are private. Restore original storage even
    # after an unexpected read-only violation, before fallback can publish it.
    old_pack, old_ref = packed.clone(), refbuf.clone()
    try:
        return _admit_e_trial(buf, refbuf, packed, n, k)
    finally:
        packed.copy_(old_pack)
        refbuf.copy_(old_ref)


def _prepare_e(packed: torch.Tensor, n: int, k: int, refbuf: torch.Tensor):
    """The verified fp8e buffer, or None (the caller tries fp8c)."""
    try:
        if not _eready_dev(packed.device) or _elib.fp8e_header_bytes(k) != header_bytes_e(k):
            return None
        enc = encode_e(packed)
        if enc is None:
            return None
        buf, info = enc
        if buf.numel() > 0.9 * n * k or buf.data_ptr() % 256:
            return None
        if not torch.equal(decode_e(buf, n, k), packed):
            stats["last_reason"] = "fp8e decode differs from the pack"
            return None
        if not _admit_e(buf, refbuf, packed, n, k):
            stats["last_reason"] = "fp8e private semantics/graph/latency admission did not pass"
            return None
        return buf, info
    except Exception as exc:  # noqa: BLE001 - fp8e is optional; fp8c follows
        stats["last_reason"] = f"fp8e: {exc!r}"
        return None


def _ready(device) -> bool:
    if _load() is None:
        return False
    with torch.cuda.device(device):
        index = torch.cuda.current_device()
        if index not in _ready_devices:
            rc = _lib.fp8c_init()
            if rc:
                raise RuntimeError(f"fp8c initialization failed: {rc}")
            _ready_devices.add(index)
    return True


def _reject(reason: str):
    stats["rejected"] += 1
    stats["last_reason"] = reason
    return None


def decode(buf: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """The kernel decoder's output, in the fp8small pack layout [N, K] uint8."""
    out = torch.empty((n, k), dtype=torch.uint8, device=buf.device)
    rc = _lib.fp8c_decode(
        buf.data_ptr(), buf.shape[1], out.data_ptr(), n, k,
        torch.cuda.current_stream(buf.device).cuda_stream,
    )
    if rc:
        raise RuntimeError(f"fp8c decode failed: {rc}")
    return out


@torch.no_grad()
def prepare(packed: torch.Tensor, return_info: bool = False):
    """Encode and verify an fp8small pack [N, K]; None keeps fp8small."""
    if not ENABLED or packed is None:
        return None
    if not packed.is_cuda or packed.dtype != torch.uint8 or packed.ndim != 2:
        return None
    n, k = packed.shape
    if (n, k) not in SHAPES or not packed.is_contiguous():
        return None
    if torch.cuda.get_device_capability(packed.device) != (12, 1) or not _ready(packed.device):
        return None
    if _lib.fp8c_header_bytes(k) != header_bytes(k):
        return None
    with torch.cuda.device(packed.device):
        enc = encode(packed)
        if enc is None:
            return _reject("a warp-chunk exceeds the exception window")
        buf, info = enc
        if buf.numel() > MAX_RATIO * n * k:
            return _reject(f"encoded ratio {info['ratio']:.4f} > {MAX_RATIO}")
        if buf.data_ptr() % 256:
            return _reject("buffer misaligned")
        dec = decode(buf, n, k)
        if not torch.equal(dec, packed):
            bad = int((dec != packed).sum().item())
            return _reject(f"decode differs from the pack in {bad} bytes")
        # The original whole's selected fp8c buffer exists before the trial.
        # Keep it on any new-format failure or lack of measured benefit.
        got = _prepare_e(packed, n, k, buf) if FP8E_ENABLED else None
        if got is not None:
            buf, info = got
            stats["fp8e"] += 1
    stats["tensors"] += 1
    stats["raw_bytes"] += n * k
    stats["bytes"] += buf.numel()
    return (buf, info) if return_info else buf


def gemm(
    a: torch.Tensor, buf: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor,
    n: int, k: int, ring: int = 0, pdl: bool | None = None,
) -> torch.Tensor:
    """Called only after the runtime guard and load-time preparation."""
    global _pdl
    m = a.shape[0]
    out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    use_pdl = _pdl if pdl is None else (pdl and _pdl)
    ring = ring or RING or _RING_DEFAULT[(n, k)]
    args = [
        a.data_ptr(), buf.data_ptr(), sa.data_ptr(), sb.data_ptr(), out.data_ptr(),
        m, n, k, buf.shape[1], int(use_pdl), ring,
        torch.cuda.current_stream(a.device).cuda_stream,
    ]
    if buf.dim() == 3:
        return _gemm_e(a, buf, sa, sb, n, k, ring, pdl, out)
    rc = _lib.fp8c_run(*args)
    if rc and use_pdl and not torch.cuda.is_current_stream_capturing():
        # The PDL launch was rejected (eager, before any capture): ordinary
        # launches from now on, and redo this call.
        _pdl = False
        args[9] = 0
        rc = _lib.fp8c_run(*args)
    if rc:
        raise RuntimeError(f"fp8c launch failed: {rc}, M={m}, N={n}, K={k}")
    return out


def _gemm_e(a, buf, sa, sb, n, k, ring, pdl, out):
    """fp8e twin of gemm (buffer [N/32, 1, tile_bytes])."""
    global _epdl
    use_pdl = _epdl if pdl is None else (pdl and _epdl)
    ring = ring or RING or _RING_DEFAULT[(n, k)]
    args = [
        a.data_ptr(), buf.data_ptr(), sa.data_ptr(), sb.data_ptr(), out.data_ptr(),
        a.shape[0], n, k, buf.shape[-1], int(use_pdl), ring,
        torch.cuda.current_stream(a.device).cuda_stream,
    ]
    rc = _elib.fp8e_run(*args)
    if rc and use_pdl and not torch.cuda.is_current_stream_capturing():
        _epdl = False
        args[9] = 0
        rc = _elib.fp8e_run(*args)
    if rc:
        raise RuntimeError(f"fp8e launch failed: {rc}, M={a.shape[0]}, N={n}, K={k}")
    return out

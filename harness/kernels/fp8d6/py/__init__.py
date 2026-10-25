# SPDX-License-Identifier: Apache-2.0
"""Optional exact six-bit per-column dictionary for the Mamba FP8 projections.

All execution is official-remote only. Top63 bytes are dictionary values and
all other bytes are escaped exactly. Existing7-bit fp8c remains the reference
and fallback; admission requires smaller bytes, actual full decode equality,
all supported-M raw-output/changing-graph equality and a cold speed filter.
"""

from __future__ import annotations

import ctypes
import os
import statistics
import subprocess
import sys
from pathlib import Path

import torch

SHAPES = {(10304, 2688), (2688, 4096)}
ENABLED = os.environ.get("FP8D6", "1") != "0"
MAX_RATIO = 0.995  # relative to the already admitted7-bit buffer
MAIN = 3072  # bytes per (tile, chunk of 4 K-steps)
MAX_WARP_CHUNK = 252  # escapes per warp-chunk: two 128-byte windows, plus alignment room
# Ring depth in chunks of 4 K-steps, per shape (FP8D6_RING overrides; 3, 4 or 6).
RING = int(os.environ.get("FP8D6_RING", "0"))
if RING not in (0, 3, 4, 6):
    RING = 0
_RING_DEFAULT = {(10304, 2688): 3, (2688, 4096): 4}
_PDL_ENABLED = os.environ.get("FP8D6_PDL", "1") != "0"
_lib = None
_load_failed = False
_ready_devices = set()
_pdl = False
stats = {"tensors": 0, "raw_bytes": 0, "bytes": 0, "rejected": 0, "last_reason": ""}


def _load():
    global _lib, _load_failed, _pdl
    if _lib is not None or _load_failed:
        return _lib
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name("libfp8d6.so")))
        lib.fp8d6_init.argtypes = []
        lib.fp8d6_init.restype = ctypes.c_int
        lib.fp8d6_has_pdl.argtypes = []
        lib.fp8d6_has_pdl.restype = ctypes.c_int
        lib.fp8d6_run.argtypes = (
            [ctypes.c_void_p] * 5 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
        )
        lib.fp8d6_run.restype = ctypes.c_int
        lib.fp8d6_decode.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.fp8d6_decode.restype = ctypes.c_int
        lib.fp8d6_header_bytes.argtypes = [ctypes.c_int]
        lib.fp8d6_header_bytes.restype = ctypes.c_int
        _lib = lib
        _pdl = _PDL_ENABLED and lib.fp8d6_has_pdl() == 1
    except OSError:
        _load_failed = True
    return _lib


def header_bytes(k: int) -> int:
    c = k // 128
    return (((4 * c + 1) * 2 + 2048) + 127) // 128 * 128


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


@torch.no_grad()
def encode(packed: torch.Tensor):
    """Top-63 byte dictionary per output column; code63 stores a raw escape.

    Six lane words store eight four-byte code words. The six bits of word6
    ride in high-two-bit planes0..2; word7 rides in planes3..5.
    """
    n, k = packed.shape
    t, c = n // 32, k // 128
    dev = packed.device
    r = (packed.view(t, c, 4, 4, 32, 2, 4).permute(0, 1, 3, 4, 2, 5, 6)
         .reshape(t, c, 4, 32, 8, 4))
    rc = r.view(t, c, 4, 8, 4, 8, 4)
    col = torch.arange(n, device=dev, dtype=torch.int64).view(t, 1, 4, 8, 1, 1, 1)
    histogram = torch.bincount((col * 256 + rc.long()).reshape(-1),
                               minlength=n * 256).view(n, 256)
    # Deterministic ties choose the lower original byte. Original bit patterns
    # are dictionary values, not floating-point magnitudes or approximations.
    rank = histogram * 256 - torch.arange(256, device=dev)
    common = rank.topk(63, dim=-1, sorted=True).indices
    lut = torch.zeros((n, 64), dtype=torch.uint8, device=dev)
    lut[:, :63] = common.to(torch.uint8)
    mapping = torch.full((n, 256), 63, dtype=torch.uint8, device=dev)
    mapping.scatter_(1, common, torch.arange(63, device=dev, dtype=torch.uint8)
                     .expand(n, -1))
    code = mapping[col, rc.long()].reshape(t, c, 4, 32, 8, 4)
    esc = code == 63
    words = []
    for i in range(6):
        high = ((code[..., 6 + i // 3, :] >> (2 * (i % 3))) & 3) << 6
        words.append(code[..., i, :] | high)
    cw = torch.stack(words, dim=-2)
    main = torch.cat([cw[..., :4, :].reshape(t, c, 2048),
                      cw[..., 4:6, :].reshape(t, c, 1024)], dim=2).reshape(t, c * MAIN)
    escf = esc.reshape(t, c * 4, 1024)
    cnt = escf.sum(-1, dtype=torch.int32)
    max_wc = int(cnt.max().item())
    if max_wc > MAX_WARP_CHUNK:
        return None
    tot = cnt.sum(-1)
    max_tot = int(tot.max().item())
    if max_tot >= 32768:
        return None
    offsets = torch.zeros((t, 4 * c + 1), dtype=torch.int32, device=dev)
    offsets[:, 1:] = cnt.cumsum(-1)
    cap = max(_round_up(max_tot, 16), 16)
    vals = r.reshape(t, -1)[esc.reshape(t, -1)]
    tile = torch.repeat_interleave(torch.arange(t, device=dev), tot)
    first = tot.cumsum(0) - tot
    at = torch.arange(vals.numel(), device=dev) - first[tile]
    exc = torch.zeros((t, cap), dtype=torch.uint8, device=dev)
    exc[tile, at] = vals
    hdr_bytes = header_bytes(k)
    hdr = torch.zeros((t, hdr_bytes), dtype=torch.uint8, device=dev)
    hdr[:, :(4 * c + 1) * 2] = offsets.to(torch.int16).view(torch.uint8)
    hdr[:, hdr_bytes - 2048:] = lut.view(t, 2048)
    tile_bytes = _round_up(hdr_bytes + c * MAIN + cap, 128)
    pad = torch.zeros((t, tile_bytes - hdr_bytes - c * MAIN - cap),
                      dtype=torch.uint8, device=dev)
    buf = torch.cat([hdr, main, exc, pad], dim=1).contiguous()
    info = dict(escapes=int(tot.sum().item()), max_warp_chunk=max_wc,
                cap=cap, bytes=buf.numel(), tile_bytes=tile_bytes,
                ratio=buf.numel() / (n * k), bits_per_weight=8 * buf.numel() / (n * k))
    return buf, info


def _ready(device) -> bool:
    if _load() is None:
        return False
    with torch.cuda.device(device):
        index = torch.cuda.current_device()
        if index not in _ready_devices:
            rc = _lib.fp8d6_init()
            if rc:
                raise RuntimeError(f"fp8d6 initialization failed: {rc}")
            _ready_devices.add(index)
    return True


def _reject(reason: str):
    stats["rejected"] += 1
    stats["last_reason"] = reason
    return None


def decode(buf: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """The kernel decoder's output, in the fp8small pack layout [N, K] uint8."""
    out = torch.empty((n, k), dtype=torch.uint8, device=buf.device)
    rc = _lib.fp8d6_decode(
        buf.data_ptr(), buf.shape[1], out.data_ptr(), n, k,
        torch.cuda.current_stream(buf.device).cuda_stream,
    )
    if rc:
        raise RuntimeError(f"fp8d6 decode failed: {rc}")
    return out



def _gemm(
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
    rc = _lib.fp8d6_run(*args)
    if rc:
        raise RuntimeError(f"fp8d6 launch failed: {rc}, M={m}, N={n}, K={k}")
    return out


_SELECTED = {}
_PROBED = None
_PROBE = r"""
import torch, fp8c, fp8d6
assert fp8d6._ready(torch.device('cuda'))
for n, k in sorted(fp8d6.SHAPES):
    p = (torch.arange(n*k, device='cuda', dtype=torch.int32) % 61).to(torch.uint8).view(n, k)
    p[:, ::97] = ((p[:, ::97].to(torch.int16) + 128) % 256).to(torch.uint8)
    enc = fp8d6.encode(p)
    assert enc is not None
    buf, info = enc
    assert torch.equal(fp8d6.decode(buf, n, k), p)
    ref = fp8c.prepare(p)
    assert ref is not None
    a = torch.randn(5, k, device='cuda').to(torch.float8_e4m3fn)
    sa = torch.ones(1, device='cuda'); sb = torch.ones(1, device='cuda')
    expected = fp8c.gemm(a, ref, sa, sb, n, k)
    for pdl in (False, True):
        actual = fp8d6._gemm(a, buf, sa, sb, n, k, pdl=pdl)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
torch.cuda.synchronize()
print('fp8d6-probe-ok')
"""


def _probe_once():
    global _PROBED
    if _PROBED is None:
        try:
            child = subprocess.run([sys.executable, '-c', _PROBE],
                capture_output=True, text=True, timeout=180)
            _PROBED = child.returncode == 0 and 'fp8d6-probe-ok' in child.stdout
            if not _PROBED:
                print('fp8d6 optional child rejected: ' + child.stderr[-400:], flush=True)
        except (OSError, subprocess.TimeoutExpired) as exc:
            _PROBED = False
            print(f'fp8d6 optional child unavailable: {exc!r}', flush=True)
    return _PROBED


def _same(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@torch.no_grad()
def _qualify(buf, reference, n, k):
    import fp8c
    gen = torch.Generator(device=buf.device).manual_seed(632560)
    sa = torch.tensor([1.5], device=buf.device, dtype=torch.float32)
    sb = torch.tensor([0.75], device=buf.device, dtype=torch.float32)
    timing_pair = None
    for m in (1, 4, 5, 8):
        inputs = [(torch.randn((m, k), device=buf.device, generator=gen) * scale)
                  .clamp(-448, 448).to(torch.float8_e4m3fn) for scale in (0.001, 1., 32.)]
        inputs.append(torch.zeros((m, k), device=buf.device, dtype=torch.float8_e4m3fn))
        bits = (torch.arange(m*k, device=buf.device) % 256).to(torch.uint8).view(m, k)
        inputs.append(bits.view(torch.float8_e4m3fn))
        a = torch.empty_like(inputs[0])
        for source in inputs:
            a.copy_(source)
            actual = _gemm(a, buf, sa, sb, n, k)
            expected = fp8c.gemm(a, reference, sa, sb, n, k)
            if not _same(actual, expected):
                return False
        a.copy_(inputs[1])
        torch.cuda.synchronize(buf.device)
        old_graph, new_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(old_graph, pool=torch.cuda.graph_pool_handle()):
            old_out = fp8c.gemm(a, reference, sa, sb, n, k)
        with torch.cuda.graph(new_graph, pool=torch.cuda.graph_pool_handle()):
            new_out = _gemm(a, buf, sa, sb, n, k)
        for source in inputs + list(reversed(inputs)):
            a.copy_(source)
            old_graph.replay(); new_graph.replay()
            if not _same(new_out, old_out):
                return False
        if m == 5:
            # Preserve input/output owners with both private graphs until timing
            # completes; no graph or synthetic result survives preparation.
            timing_pair = (a, inputs[1], old_graph, new_graph, old_out, new_out)
    a, source, old_graph, new_graph, old_out, new_out = timing_pair
    a.copy_(source)
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=buf.device)
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = {0: [], 1: []}
    for trial in range(9):
        for kind in ((0, 1) if trial % 2 == 0 else (1, 0)):
            flush.zero_()
            begin.record()
            (old_graph if kind == 0 else new_graph).replay()
            end.record(); end.synchronize()
            times[kind].append(begin.elapsed_time(end))
    return statistics.median(times[1]) < .96 * statistics.median(times[0])


@torch.no_grad()
def prepare(packed, reference):
    if (not ENABLED or reference is None or packed is None
            or torch.cuda.is_current_stream_capturing()):
        return None
    if (not packed.is_cuda or packed.dtype != torch.uint8 or packed.ndim != 2
            or not packed.is_contiguous() or tuple(packed.shape) not in SHAPES):
        return None
    n, k = packed.shape
    if torch.cuda.get_device_capability(packed.device) != (12, 1) or not _probe_once():
        return _reject('optional native probe unavailable')
    if not _ready(packed.device) or _lib.fp8d6_header_bytes(k) != header_bytes(k):
        return _reject('optional native header unavailable')
    enc = encode(packed)
    if enc is None:
        return _reject('exception window overflow')
    buf, info = enc
    if buf.numel() > MAX_RATIO * reference.numel() or buf.data_ptr() % 256:
        return _reject('not smaller than current7-bit reference or misaligned')
    if not torch.equal(decode(buf, n, k), packed):
        return _reject('actual full decoded byte mismatch')
    if not _qualify(buf, reference, n, k):
        return _reject('actual eager/replay bytes or cold4pct filter rejected')
    _SELECTED[(buf.device.index, buf.data_ptr())] = (buf, reference, n, k, info)
    stats['tensors'] += 1; stats['raw_bytes'] += n*k; stats['bytes'] += buf.numel()
    print(f'fp8d6 admitted shape={(n,k)} bytes={buf.numel()} reference7={reference.numel()} '
          f'bits={info["bits_per_weight"]:.3f} escapes={info["escapes"]}', flush=True)
    return buf


def owns(buf):
    entry = _SELECTED.get((buf.device.index, buf.data_ptr()))
    return (entry is not None and entry[0].shape == buf.shape
            and entry[0].stride() == buf.stride() and entry[0].dtype == buf.dtype)


def gemm(a, buf, sa, sb, n, k):
    entry = _SELECTED.get((buf.device.index, buf.data_ptr()))
    if entry is None or (entry[2], entry[3]) != (n, k) or not owns(buf):
        raise RuntimeError('unregistered fp8d6 buffer')
    if not ENABLED:
        import fp8c
        return fp8c.gemm(a, entry[1], sa, sb, n, k)
    return _gemm(a, buf, sa, sb, n, k)

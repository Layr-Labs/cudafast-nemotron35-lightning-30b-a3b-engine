# SPDX-License-Identifier: Apache-2.0
"""Optional lossless tile partition of the already admitted FP8 weight pack.

Only immutable load-time weight statistics choose the partition. Every output
column keeps its original ascending-K FP8 MMA chain and BF16 epilogue. The
actual current six/ seven-bit path remains the reference and fallback.
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
ENABLED = os.environ.get('FP8TILE', '1') != '0'
_lib = None
_failed = False
_pdl = False
_ready_devices = set()
_SELECTED = {}
_PROBED = None
stats = dict(tensors=0, bytes=0, rejected=0, last_reason='')


def _up(x, alignment):
    return (x + alignment - 1) // alignment * alignment


def _load():
    global _lib, _failed, _pdl
    if _lib is not None or _failed:
        return _lib
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name('libfp8tile.so')))
        for name in ('fp8tile_init', 'fp8tile_has_pdl'):
            getattr(lib, name).argtypes = []
            getattr(lib, name).restype = ctypes.c_int
        lib.fp8tile_run.argtypes = ([ctypes.c_void_p] * 8 + [ctypes.c_int] * 10
                                   + [ctypes.c_void_p])
        lib.fp8tile_run.restype = ctypes.c_int
        for name in ('fp8t6_decode', 'fp8t7_decode'):
            fn = getattr(lib, name)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                           ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                           ctypes.c_int, ctypes.c_void_p]
            fn.restype = ctypes.c_int
        _lib = lib
        _pdl = (os.environ.get('FP8TILE_PDL', '1') != '0'
                and lib.fp8tile_has_pdl() == 1)
    except OSError:
        _failed = True
    return _lib


def _ready(device):
    if _load() is None:
        return False
    with torch.cuda.device(device):
        index = torch.cuda.current_device()
        if index not in _ready_devices:
            rc = _lib.fp8tile_init()
            if rc:
                raise RuntimeError(f'fp8tile initialization failed: {rc}')
            _ready_devices.add(index)
    return True


@torch.no_grad()
def _costs6(packed):
    """Same per-column dictionary and exception counts as verified fp8d6."""
    import fp8d6
    n, k = packed.shape
    t, c = n // 32, k // 128
    r = (packed.view(t, c, 4, 4, 32, 2, 4).permute(0, 1, 3, 4, 2, 5, 6)
         .reshape(t, c, 4, 32, 8, 4))
    rc = r.view(t, c, 4, 8, 4, 8, 4)
    col = torch.arange(n, device=packed.device).view(t, 1, 4, 8, 1, 1, 1)
    hist = torch.bincount((col * 256 + rc.long()).reshape(-1),
                          minlength=n * 256).view(n, 256)
    common = (hist * 256 - torch.arange(256, device=packed.device)).topk(
        63, dim=-1, sorted=True).indices
    mapping = torch.full((n, 256), 63, device=packed.device, dtype=torch.uint8)
    mapping.scatter_(1, common, torch.arange(63, device=packed.device,
                    dtype=torch.uint8).expand(n, -1))
    escapes = (mapping[col, rc.long()] == 63).reshape(t, 4 * c, 1024)
    counts = escapes.sum(-1, dtype=torch.int32)
    totals = counts.sum(-1)
    valid = (counts.max(-1).values <= fp8d6.MAX_WARP_CHUNK) & (totals < 32768)
    sizes = ((totals.clamp(min=16) + 15) // 16 * 16
             + fp8d6.header_bytes(k) + c * fp8d6.MAIN + 127) // 128 * 128
    return sizes.cpu().tolist(), valid.cpu().tolist()


def _sizes7(reference7, k):
    c = k // 128
    hdr = _up((4 * c + 1) * 2 + 64, 128)
    # Total escape count is the final unsigned u16 in every tile header.
    totals = (reference7[:, 8*c:8*c+2].contiguous().view(torch.int16)
              .to(torch.int32).reshape(-1) & 65535).cpu().tolist()
    return [_up(hdr + c * 3584 + max(_up(x, 16), 16), 128) for x in totals]


def _partition(packed, reference7, current):
    six, valid = _costs6(packed)
    seven = _sizes7(reference7, packed.shape[1])
    # Exact small metadata search over possible group strides, not an average
    # entropy estimate: each group's worst tile determines its actual stride.
    best, choice = current.numel(), None
    for limit in sorted({s for s, ok in zip(six, valid) if ok}):
        ids = [i for i, (s, ok) in enumerate(zip(six, valid)) if ok and s <= limit]
        chosen = set(ids)
        rest = [i for i in range(len(six)) if i not in chosen]
        b6 = len(ids) * max((six[i] for i in ids), default=0)
        b7 = len(rest) * max((seven[i] for i in rest), default=0)
        size = _up(_up(len(six) * 4, 256) + b6, 256) + b7
        if size < best:
            best, choice = size, ids
    return choice if choice is not None and best < .995 * current.numel() else None


@torch.no_grad()
def _assemble(packed, reference7, current, ids6):
    import fp8d6
    n, k = packed.shape
    t = n // 32
    chosen = set(ids6)
    ids7 = [i for i in range(t) if i not in chosen]
    if len(chosen) != len(ids6) or any(i < 0 or i >= t for i in ids6):
        raise ValueError('invalid tile partition')
    idx6 = torch.tensor(ids6, dtype=torch.int32, device=packed.device)
    idx7 = torch.tensor(ids7, dtype=torch.int32, device=packed.device)
    if ids6:
        sub = packed.view(t, 32*k).index_select(0, idx6.long()).view(-1, k)
        encoded = fp8d6.encode(sub)
        if encoded is None:
            return None
        buf6, _ = encoded
    else:
        buf6 = torch.empty((0, 128), device=packed.device, dtype=torch.uint8)
    if ids7:
        stride7 = max(_sizes7(reference7, k)[i] for i in ids7)
        if stride7 > reference7.shape[1]:
            raise ValueError('reference7 exception bounds exceed its stride')
        buf7 = reference7.index_select(0, idx7.long())[:, :stride7].contiguous()
    else:
        buf7 = torch.empty((0, 128), device=packed.device, dtype=torch.uint8)
    offset6 = _up(t * 4, 256)
    offset7 = _up(offset6 + buf6.numel(), 256)
    anchor = torch.zeros(offset7 + buf7.numel(), device=packed.device, dtype=torch.uint8)
    ids = anchor[:t*4].view(torch.int32)
    ids[:len(ids6)].copy_(idx6); ids[len(ids6):].copy_(idx7)
    payload6 = anchor[offset6:offset6+buf6.numel()].view(buf6.shape)
    payload7 = anchor[offset7:offset7+buf7.numel()].view(buf7.shape)
    payload6.copy_(buf6); payload7.copy_(buf7)
    return dict(anchor=anchor, six=payload6, seven=payload7, ids6=ids[:len(ids6)],
                ids7=ids[len(ids6):], n=n, k=k, reference7=reference7, current=current)


def _decode(entry):
    n, k = entry['n'], entry['k']
    out = torch.empty((n, k), device=entry['anchor'].device, dtype=torch.uint8)
    stream = torch.cuda.current_stream(out.device).cuda_stream
    for key, ids, symbol in [('six', 'ids6', 'fp8t6_decode'),
                              ('seven', 'ids7', 'fp8t7_decode')]:
        if entry[ids].numel():
            buf = entry[key]
            rc = getattr(_lib, symbol)(buf.data_ptr(), buf.shape[1], out.data_ptr(),
                n, k, entry[ids].data_ptr(), entry[ids].numel(), stream)
            if rc:
                raise RuntimeError(f'{symbol} failed: {rc}')
    return out


def _reference(a, entry, sa, sb):
    import fp8c, fp8d6
    ref, n, k = entry['current'], entry['n'], entry['k']
    if fp8d6.owns(ref):
        return fp8d6.gemm(a, ref, sa, sb, n, k)
    return fp8c.gemm(a, ref, sa, sb, n, k)


def _gemm(a, entry, sa, sb, pdl=None):
    import fp8c, fp8d6
    n, k = entry['n'], entry['k']
    out = torch.empty((a.shape[0], n), device=a.device, dtype=torch.bfloat16)
    use_pdl = _pdl if pdl is None else bool(pdl and _pdl)
    ring6 = fp8d6.RING or fp8d6._RING_DEFAULT[(n, k)]
    args = [a.data_ptr(), entry['six'].data_ptr(), entry['seven'].data_ptr(),
            sa.data_ptr(), sb.data_ptr(), out.data_ptr(), entry['ids6'].data_ptr(),
            entry['ids7'].data_ptr(), a.shape[0], n, k, entry['six'].shape[1],
            entry['seven'].shape[1], entry['ids6'].numel(), entry['ids7'].numel(),
            int(use_pdl), ring6, fp8c.RING or (6 if k == 4096 else 3),
            torch.cuda.current_stream(a.device).cuda_stream]
    rc = _lib.fp8tile_run(*args)
    if rc:
        raise RuntimeError(f'fp8tile launch failed: {rc}')
    return out


def _same(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@torch.no_grad()
def _qualify(entry):
    dev, n, k = entry['anchor'].device, entry['n'], entry['k']
    gen = torch.Generator(device=dev).manual_seed(673560)
    sa = torch.tensor([1.5], device=dev); sb = torch.tensor([.75], device=dev)
    for m in (1, 4, 5, 8):
        inputs = [(torch.randn((m, k), device=dev, generator=gen) * s)
                  .clamp(-448, 448).to(torch.float8_e4m3fn) for s in (.001, 1., 32.)]
        inputs += [torch.zeros_like(inputs[0]), (torch.arange(m*k, device=dev)
                   % 256).to(torch.uint8).view(m, k).view(torch.float8_e4m3fn)]
        a = torch.empty_like(inputs[0])
        for pdl in (False, True) if _pdl else (False,):
            for source in inputs:
                a.copy_(source)
                if not _same(_gemm(a, entry, sa, sb, pdl), _reference(a, entry, sa, sb)):
                    return False
            # Captured mutable producers precede the dependent two-grid call.
            # Different graph replays refresh both activation and scale input.
            source = inputs[1].clone(); scale = sa.clone()
            torch.cuda.synchronize(dev)
            old, new = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            with torch.cuda.graph(old, pool=torch.cuda.graph_pool_handle()):
                a.copy_(source); sa.copy_(scale)
                old_out = _reference(a, entry, sa, sb)
            with torch.cuda.graph(new, pool=torch.cuda.graph_pool_handle()):
                a.copy_(source); sa.copy_(scale)
                new_out = _gemm(a, entry, sa, sb, pdl)
            for i, bits in enumerate(inputs + list(reversed(inputs))):
                source.copy_(bits); scale.fill_(1.5 if i % 2 else .625)
                old.replay(); new.replay()
                if not _same(new_out, old_out):
                    return False
            if m == 5 and pdl == _pdl:
                timing = (source, inputs[1], scale, old, new, old_out, new_out, a)
    source, real, scale, old, new, old_out, new_out, a = timing
    source.copy_(real); scale.fill_(1.5)
    flush = torch.empty(64*1024*1024, dtype=torch.uint8, device=dev)
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = {0: [], 1: []}
    for trial in range(9):
        for kind in ((0, 1) if trial % 2 == 0 else (1, 0)):
            flush.zero_(); begin.record()
            (old if kind == 0 else new).replay()
            end.record(); end.synchronize()
            times[kind].append(begin.elapsed_time(end))
    return statistics.median(times[1]) < .96 * statistics.median(times[0])


_PROBE = r'''
import torch, fp8c, fp8tile
assert fp8tile._ready(torch.device('cuda'))
for n,k in sorted(fp8tile.SHAPES):
 p=(torch.arange(n*k,device='cuda',dtype=torch.int32)%61).to(torch.uint8).view(n,k)
 p[:,::97]=((p[:,::97].to(torch.int16)+128)%256).to(torch.uint8)
 ref=fp8c.prepare(p)
 assert ref is not None
 sa=torch.ones(1,device='cuda');sb=torch.ones(1,device='cuda')
 for ids in ([],list(range(n//32)),list(range(0,n//32,2)),[0,n//32-1]):
  e=fp8tile._assemble(p,ref,ref,ids)
  assert e is not None and torch.equal(fp8tile._decode(e),p)
  for m in (1,4,5,8):
   a=torch.randn(m,k,device='cuda').to(torch.float8_e4m3fn)
   for pdl in (False,True):
    assert fp8tile._same(fp8tile._gemm(a,e,sa,sb,pdl),fp8c.gemm(a,ref,sa,sb,n,k))
torch.cuda.synchronize()
print('fp8tile-probe-ok')
'''


def _probe_once():
    global _PROBED
    if _PROBED is None:
        try:
            child = subprocess.run([sys.executable, '-c', _PROBE],
                capture_output=True, text=True, timeout=240)
            _PROBED = child.returncode == 0 and 'fp8tile-probe-ok' in child.stdout
            if not _PROBED:
                print('fp8tile optional child rejected: ' + child.stderr[-400:], flush=True)
        except (OSError, subprocess.TimeoutExpired) as exc:
            _PROBED = False
            print(f'fp8tile optional child unavailable: {exc!r}', flush=True)
    return _PROBED


def _reject(reason):
    stats['rejected'] += 1; stats['last_reason'] = reason
    return None


@torch.no_grad()
def prepare(packed, reference7, current):
    if (not ENABLED or current is None or reference7 is None or packed is None
            or torch.cuda.is_current_stream_capturing()):
        return None
    if (not packed.is_cuda or packed.dtype != torch.uint8 or packed.ndim != 2
            or not packed.is_contiguous() or tuple(packed.shape) not in SHAPES):
        return None
    if torch.cuda.get_device_capability(packed.device) != (12, 1) or not _probe_once():
        return _reject('optional native probe unavailable')
    if not _ready(packed.device):
        return _reject('optional native library unavailable')
    ids = _partition(packed, reference7, current)
    if ids is None:
        return _reject('no smaller tile partition than actual current codec')
    entry = _assemble(packed, reference7, current, ids)
    if entry is None:
        return _reject('six-bit group exception bounds')
    anchor = entry['anchor']
    if anchor.numel() >= .995 * current.numel() or anchor.data_ptr() % 256:
        return _reject('actual total bytes including IDs/alignment or alignment rejected')
    if not torch.equal(_decode(entry), packed):
        return _reject('full actual mixed decode bytes differ')
    if not _qualify(entry):
        return _reject('actual all-M eager/producer-graph bytes or cold4pct filter')
    _SELECTED[(anchor.device.index, anchor.data_ptr())] = entry
    stats['tensors'] += 1; stats['bytes'] += anchor.numel()
    print(f'fp8tile admitted shape={tuple(packed.shape)} tiles6={len(ids)} '
          f'tiles7={entry["ids7"].numel()} bytes={anchor.numel()} '
          f'current={current.numel()} reference7={reference7.numel()}', flush=True)
    return anchor


def owns(buf):
    entry = _SELECTED.get((buf.device.index, buf.data_ptr()))
    return (entry is not None and entry['anchor'].shape == buf.shape
            and entry['anchor'].stride() == buf.stride() and buf.dtype == torch.uint8)


def gemm(a, buf, sa, sb, n, k):
    entry = _SELECTED.get((buf.device.index, buf.data_ptr()))
    if entry is None or not owns(buf) or (n, k) != (entry['n'], entry['k']):
        raise RuntimeError('unregistered mixed FP8 tile buffer')
    return _gemm(a, entry, sa, sb) if ENABLED else _reference(a, entry, sa, sb)

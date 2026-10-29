"""Optional immutable exact router packet; all admission is remote post-load.

Production uses only a prior decision and the original fresh output allocation.
No input, token, output, request state or production CUDA graph is memoized.
"""
from __future__ import annotations

import ctypes
import os
import sys
import weakref

import torch

ENABLED = os.environ.get("SMALLK_ROUTER_W12", "1") != "0"
_lib = None
_attempted = False
_ready: dict = {}
_index: dict = {}
_W16_BYTES = 128 * 2688 * 2


def _modules():
    return sys.modules.get("smallk"), sys.modules.get("l2pf")


def _config():
    sk, lp = _modules()
    if sk is None or lp is None:
        return None
    return (ENABLED, sk.ROUTER_ON, sk.ROUTER_W16_ON, sk.ROUTER_M5_ON,
            sk.EARLY_ROUTER_ON, sk.ROUTER_ROWS, sk.PDL, lp.ENABLED,
            lp._GATE_PREFETCH, lp.MAMBA_BYTES, lp.ATTN_BYTES, lp.CHUNK,
            id(sk._lib), id(lp._lib))


def _load():
    global _lib, _attempted
    if _attempted:
        return _lib
    _attempted = True
    if not ENABLED:
        return None
    try:
        lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "librouterw12.so"))
        vp, ci, cl = ctypes.c_void_p, ctypes.c_int, ctypes.c_long
        lib.rw12_version.restype = ci
        if lib.rw12_version() != 1:
            return None
        lib.rw12_init.restype = ci
        lib.rw12_pack.argtypes = [vp, vp, vp, ci, vp]
        lib.rw12_pack.restype = ci
        lib.rw12_decode.argtypes = [vp, ci, vp, vp]
        lib.rw12_decode.restype = ci
        lib.rw12_router.argtypes = [vp, cl, ci, vp, ci, vp, cl, ci, vp, vp]
        lib.rw12_router.restype = ci
        _lib = lib
    except (OSError, AttributeError):
        pass
    return _lib


def _stream(t):
    return torch.cuda.current_stream(t.device).cuda_stream


def _check(rc, operation):
    if rc:
        # Do not turn a launch/capture error into a retry on production state.
        raise RuntimeError(f"routerw12 {operation} failed: {rc}")


def _equal(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


def _key(w, shadow):
    if shadow is None or w.is_inference() or shadow.is_inference():
        return None
    return (w.device, w.dtype, w.data_ptr(), w._version,
            tuple(w.shape), tuple(w.stride()), w.storage_offset(),
            shadow.device, shadow.dtype, shadow.data_ptr(), shadow._version,
            tuple(shadow.shape), tuple(shadow.stride()), shadow.storage_offset(), _config())


class _Entry:
    def __init__(self, gate, w, shadow, packet, rows, coded_rows):
        self.gate = weakref.ref(gate)
        self.weight = weakref.ref(w)
        self.shadow = weakref.ref(shadow)
        self.key = _key(w, shadow)
        self.packet = packet
        self.packet_version = packet._version
        self.rows = frozenset(rows)
        # Private startup metadata, not a public statistic or request counter.
        self.coded_rows = coded_rows


def _packet_ok(packet):
    return (packet.dtype == torch.uint8 and packet.is_cuda and packet.ndim == 1
            and packet.is_contiguous() and packet.storage_offset() == 0
            and packet.untyped_storage().data_ptr() == packet.data_ptr()
            and packet.untyped_storage().nbytes() == packet.numel()
            and 524800 <= packet.numel() < _W16_BYTES)


def _lookup(w, shadow, m, row_mask):
    if (not ENABLED or not row_mask or m not in (5, 6)
            or shadow is None or _lib is None):
        return None
    ref = _index.get(id(w))
    entry = None if ref is None else ref()
    if (entry is None or entry.weight() is not w or entry.shadow() is not shadow
            or entry.gate() is None or m not in entry.rows
            or entry.key != _key(w, shadow)
            or not _packet_ok(entry.packet) or entry.packet._version != entry.packet_version):
        return None
    return entry


def prefetch_packet(gate, w, shadow, rows, reserved):
    """Same admitted tensor as GEMV, while retaining the original reservation."""
    entry = _lookup(w, shadow, rows, gate.__dict__.get("_smallk_row_mask", False))
    if (entry is None or entry.gate() is not gate or reserved != _W16_BYTES
            or shadow.untyped_storage().nbytes() != _W16_BYTES):
        return None
    return entry.packet


def lookup(w, shadow, rows, row_mask):
    return _lookup(w, shadow, rows, row_mask)


def run(entry, x, w, shadow, out, rows, mode, pdl, vrows):
    """Use the existing caller-owned fresh output; no production allocations."""
    if (rows != 1 or mode != 0
            or x.dtype != torch.float32 or x.ndim != 2 or x.shape[0] not in (5, 6)
            or x.shape[1] != 2688 or x.stride(1) != 1 or x.stride(0) < 2688
            or not x.is_cuda or x.device != w.device or out.device != x.device
            or out.dtype != torch.float32 or out.shape != (x.shape[0], 128)
            or out.stride(1) != 1 or out.stride(0) < 128
            or (vrows is not None and (vrows.device != x.device or vrows.dtype != torch.int32
                or vrows.shape != (1,) or not vrows.is_contiguous()))
            or pdl != _modules()[0].PDL
            or _lookup(w, shadow, x.shape[0], True) is not entry):
        return False
    # The ordinary production output is a fresh allocation. Do not extend the
    # optional path to direct callers with overlapping input/output storage.
    begin = out.data_ptr()
    end = begin + ((out.shape[0] - 1) * out.stride(0) + 128) * out.element_size()
    for read in (x, shadow, entry.packet, *((vrows,) if vrows is not None else ())):
        storage = read.untyped_storage()
        if begin < storage.data_ptr() + storage.nbytes() and storage.data_ptr() < end:
            return False
    _call(x, entry.packet, out, vrows, pdl)
    return True


def _call(x, packet, out, vrows, pdl):
    _check(_lib.rw12_router(x.data_ptr(), x.stride(0), x.shape[0], packet.data_ptr(),
                           packet.numel(), out.data_ptr(), out.stride(0), pdl,
                           None if vrows is None else vrows.data_ptr(), _stream(x)), "router")


def _build_packet(shadow):
    """Actual-owner statistics, no distribution assumption; official remote only."""
    stream = torch.cuda.current_stream(shadow.device)
    try:
        words = shadow.view(torch.int16).to(torch.int32) & 65535
        exponents = ((words >> 7) & 255).view(128, 21, 128)
        low = torch.where(exponents == 0, 256, exponents).amin(dim=2)
        high = exponents.amax(dim=2)
        fit = ((high == 0) | ((high - low) <= 14)).all(dim=1)
        flags = fit.cpu().tolist()  # private startup decision; never emitted
        coded_rows = sum(bool(flag) for flag in flags)
        if not coded_rows:
            return None
        offsets = []
        offset = 512
        for flag in flags:
            offsets.append(offset | int(bool(flag)))
            offset += 4096 if flag else 5376
        if offset >= _W16_BYTES:
            return None
        bases = torch.where(high == 0, 1, low).to(torch.uint8).contiguous()
        # Dedicated owning allocation: l2pf uses its storage base, not a view pointer.
        packet = torch.zeros(offset, dtype=torch.uint8, device=shadow.device)
        header = torch.tensor(offsets, dtype=torch.int32, device=shadow.device)
        packet[:512].view(torch.int32).copy_(header)
        _check(_lib.rw12_pack(shadow.data_ptr(), bases.data_ptr(), packet.data_ptr(),
                             packet.numel(), stream.cuda_stream), "pack")
        decoded_guard = torch.full((128 * 2688 + 32,), 0x1234,
                                   dtype=torch.int16, device=shadow.device)
        decoded = decoded_guard[16:-16].view(128, 2688)
        _check(_lib.rw12_decode(packet.data_ptr(), packet.numel(), decoded.data_ptr(),
                               stream.cuda_stream), "decode")
        if (not torch.equal(decoded, shadow.view(torch.int16))
                or not bool(torch.all(decoded_guard[:16] == 0x1234))
                or not bool(torch.all(decoded_guard[-16:] == 0x1234))
                or not torch.equal(packet[:512].view(torch.int32), header)
                or not _packet_ok(packet)):
            return None
        return packet, coded_rows
    finally:
        # Keep bases/header/packet/decode owners in this frame until native work ends.
        stream.synchronize()


def _qualify(m, w, shadow, packet, projection_ranges):
    """Actual ranges and actual W16 reference, private buffers and graphs only."""
    sk, lp = _modules()
    stream = torch.cuda.current_stream(w.device)
    side = torch.cuda.Stream(device=w.device)
    try:
        xo = torch.empty((m, 2688), dtype=torch.float32, device=w.device)
        xn = torch.empty_like(xo)
        vo = torch.tensor([m], dtype=torch.int32, device=w.device)
        vn = torch.empty_like(vo)
        old_guard = torch.full((m * 128 + 32,), 123.0, dtype=torch.float32, device=w.device)
        new_guard = torch.full_like(old_guard, -237.0)
        old = old_guard[16:-16].view(m, 128)
        new = new_guard[16:-16].view(m, 128)
        old_edges = torch.cat((old_guard[:16], old_guard[-16:])).clone()
        new_edges = torch.cat((new_guard[:16], new_guard[-16:])).clone()
        shadow_before = shadow.view(torch.int16).clone()
        packet_before = packet.clone()
        base = torch.arange(m * 2688, dtype=torch.int32, device=w.device).view(m, 2688)
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

        def prefetch(weight):
            # Exact original projection ranges and reservation; only the gate's
            # owning storage changes. Nothing is published to the global lookup.
            selected = [(weight, _W16_BYTES), *projection_ranges]
            if len(selected) > 3:
                raise RuntimeError("routerw12 private prefetch range overflow")
            args = []
            for tensor, limit in selected:
                storage = tensor.untyped_storage()
                args.extend((storage.data_ptr(), min(storage.nbytes(), int(limit))))
            args.extend([0, 0] * (3 - len(selected)))
            side.wait_stream(stream)
            _check(lp._lib.l2pf_prefetch3(*args, lp.CHUNK, side.cuda_stream), "prefetch")
            stream.wait_stream(side)

        def original(mask=True):
            prefetch(shadow)
            sk.router_gemm(xo, w, out=old, rows=1, mode=0, pdl=sk.PDL,
                           vrows=vo if mask else None, w16=shadow)

        def candidate(mask=True):
            prefetch(packet)
            _call(xn, packet, new, vn if mask else None, sk.PDL)

        def poison():
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
                expected = vo.clone()
                poison()
                original()
                candidate()
                if (not same() or not torch.equal(vo, expected) or not torch.equal(vn, expected)):
                    return False
            poison()
            original(False)
            candidate(False)
            if not same() or not _equal(xo, pattern) or not _equal(xn, pattern):
                return False

        source = patterns[0].clone()
        count_source = vo.clone()
        old_graph, new_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        stream.synchronize()
        side.synchronize()
        with torch.cuda.graph(old_graph, stream=stream):
            xo.copy_(source)
            vo.copy_(count_source)
            original()
        with torch.cuda.graph(new_graph, stream=stream):
            xn.copy_(source)
            vn.copy_(count_source)
            candidate()
        replay_counts = (m, 0, m - 1, -1, 1, sk.VROWS_ALL, *reversed(range(m + 1)),
                         2147483647, -2147483648, m + 1)
        for pattern in patterns:
            source.copy_(pattern)
            for count in replay_counts:
                count_source.fill_(count)
                poison()
                old_graph.replay()
                new_graph.replay()
                if (not same() or not torch.equal(vo, count_source)
                        or not torch.equal(vn, count_source)):
                    return False
            if not _equal(source, pattern) or not _equal(xo, pattern) or not _equal(xn, pattern):
                return False

        xo.copy_(patterns[0])
        xn.copy_(patterns[0])
        old_time, new_time = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
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
        for count in (1, m - 1, m, sk.VROWS_ALL):
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
        return (ratios[len(ratios) // 2] < 0.995 and max(ratios) < 1.01 and same()
                and torch.equal(shadow.view(torch.int16), shadow_before)
                and torch.equal(packet, packet_before)
                and _equal(xo, patterns[0]) and _equal(xn, patterns[0])
                and torch.equal(vo, vn))
    finally:
        # A failure before the join can leave side work using private owners.
        # Attempt both drains before this same frame releases any tensor/graph.
        try:
            side.synchronize()
        finally:
            stream.synchronize()


def _prepare_on_stream(gate, projection_by_rows):
    sk, _ = _modules()
    w = gate.weight
    stream = torch.cuda.current_stream(w.device)
    try:
        shadow = sk.gate_w16(w)
        if (shadow is None or shadow.dtype != torch.bfloat16
                or shadow.shape != (128, 2688) or not shadow.is_contiguous()
                or shadow.storage_offset() != 0
                or shadow.untyped_storage().nbytes() != _W16_BYTES
                or _key(w, shadow) is None):
            return None
        # Existing gate_w16 can return a historical cache entry; re-prove this
        # actual finalized parameter against this actual shadow before coding.
        if not _equal(shadow.to(torch.float32), w):
            return None
        start_key = _key(w, shadow)
        result = _build_packet(shadow)
        if result is None:
            return None
        packet, coded_rows = result
        rows = []
        for m, ranges in projection_by_rows.items():
            if m == 5 and not sk.ROUTER_M5_ON:
                continue
            if _qualify(m, w, shadow, packet, ranges):
                rows.append(m)
        if not rows or start_key != _key(w, shadow):
            return None
        return _Entry(gate, w, shadow, packet, rows, coded_rows)
    finally:
        stream.synchronize()


def prepare(gate, projection_by_rows):
    """One post-load attempt; no optional weight or qualification in forward."""
    if gate.__dict__.get("_routerw12_prepared", False):
        # No hot-reload support: retain old packet owners for captured graphs.
        # A changed identity/version rejects the production lookup.
        return False
    object.__setattr__(gate, "_routerw12_prepared", True)
    sk, lp = _modules()
    if (not ENABLED or sk is None or lp is None or not sk.ROUTER_ON
            or not sk.ROUTER_W16_ON or not sk.EARLY_ROUTER_ON or sk.ROUTER_ROWS != 1
            or not lp.ENABLED or not lp._GATE_PREFETCH or lp._lib is None
            or not hasattr(lp._lib, "l2pf_prefetch3")
            or min(lp.MAMBA_BYTES, lp.ATTN_BYTES) // 4 < _W16_BYTES
            or not gate.__dict__.get("_smallk_row_mask", False)
            or gate.bias is not None or gate.weight.dtype != torch.float32
            or gate.weight.shape != (128, 2688) or not gate.weight.is_cuda
            or not gate.weight.is_contiguous() or gate.weight.is_inference()
            or not projection_by_rows
            or any(m not in (5, 6) for m in projection_by_rows)
            or torch.cuda.is_current_stream_capturing()
            or _load() is None or not sk._usable() or not sk._has_w16_entry
            or not sk._has_vrows_entry):
        return False
    w = gate.weight
    if w.device.index != torch.cuda.current_device():
        return False
    if w.device not in _ready:
        _ready[w.device] = _lib.rw12_init() == 0
    if not _ready[w.device]:
        return False
    for ranges in projection_by_rows.values():
        if (not 1 <= len(ranges) <= 2 or any(
                not tensor.is_cuda or tensor.device != w.device or limit <= 0
                for tensor, limit in ranges)):
            return False
    caller = torch.cuda.current_stream(w.device)
    private = torch.cuda.Stream(device=w.device)
    private.wait_stream(caller)
    try:
        with torch.cuda.stream(private), torch.no_grad(), torch.inference_mode(False):
            entry = _prepare_on_stream(gate, projection_by_rows)
    except torch.OutOfMemoryError:
        # Only allocation refusal after private finally-drain is recoverable.
        entry = None
    finally:
        caller.wait_stream(private)
    if entry is None:
        return False
    # Publication happens once, after all streams and private owners drained.
    # Gate owns entry; the index adds no permanent model/weight/pool owner.
    object.__setattr__(gate, "_routerw12_entry", entry)
    _index[id(w)] = weakref.ref(entry)
    return True

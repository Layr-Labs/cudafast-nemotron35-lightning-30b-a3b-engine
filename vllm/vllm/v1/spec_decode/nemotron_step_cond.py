# SPDX-License-Identifier: Apache-2.0
"""Native Nemotron MTP single-row draft step as one conditional CUDA graph.

Prepared only during remote startup, after the full-step graph and the head
screen graph exist. The graph runs the draft forward and the head screen of a
pass inside an IF node keyed on the depth-gate stop flag; a stopped pass only
publishes the screen's stopped token and a zero hidden row.
"""
import os
import statistics
from types import SimpleNamespace

import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.spec_decode.nemotron_full_step import _inputs, _metadata

logger = init_logger(__name__)
ENABLED = os.environ.get("NEMO_MTP_STEP_COND", "1") != "0"
SPEED = os.environ.get("NEMO_MTP_STEP_COND_SPEED", "1") != "0"


@triton.jit
def _arm(gate, live, out, token):
    tl.store(live, tl.load(gate) == 0)
    tl.store(out, token)


@triton.jit
def _clear(live, hidden, N: tl.constexpr, BLOCK: tl.constexpr):
    if tl.load(live) == 0:
        bits = hidden.to(tl.pointer_type(tl.uint16))
        offs = tl.arange(0, BLOCK)
        for start in range(0, N, BLOCK):
            tl.store(bits + start + offs, tl.zeros((BLOCK,), tl.uint16),
                     mask=start + offs < N)


def _supported() -> bool:
    graph = torch.cuda.CUDAGraph
    return all(hasattr(graph, n) for n in (
        "begin_capture_to_if_node", "end_capture_to_conditional_node",
        "get_currently_capturing_graph"))


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8).clone()


def maybe_run(p, kwargs, metadata, padded, batch_size):
    """(hidden, token) for one live or stopped single-row pass, or None."""
    s = getattr(p, "_nemo_step", None)
    if s is None or batch_size != 1 or padded != s.padded:
        return None
    if _inputs(kwargs) != s.inputs or _metadata(p, metadata) != s.signature:
        return None
    s.graph.replay()
    return s.hidden, s.screen._out.clone()


def prepare(p):
    if (not ENABLED or getattr(p, "_nemo_step_attempted", False)
            or torch.cuda.is_current_stream_capturing()):
        return
    p._nemo_step_attempted = True
    p._nemo_step = None
    saved = []
    restore = []
    try:
        _prepare(p, saved, restore)
    except Exception as exc:
        logger.warning("Native MTP conditional step retained original path: %r", exc)
        p._nemo_step = None
    finally:
        try:
            torch.cuda.synchronize(p.device)
            for t, v in saved:
                t.copy_(v)
            for fn in restore:
                fn()
            torch.cuda.synchronize(p.device)
        except Exception as exc:
            logger.warning("Native MTP conditional step restore: %r", exc)
    s = p._nemo_step
    if s is not None:
        logger.info("Native MTP conditional step admitted: token=%d timing_us=%s",
                    s.token, s.timing)


def _prepare(p, saved, restore):
    full = getattr(p, "_nemo_full", None)
    screen = getattr(p, "_draft_head_screen", None)
    if (full is None or screen is None or getattr(screen, "graph", None) is None
            or not _supported() or p.model_returns_tuple()):
        return
    from vllm.model_executor.models import nemotron_h_mtp_kernels as kernels
    from vllm.v1.spec_decode import nemotron_full_step
    gate = kernels.depth_gate_init(p.device)
    if gate is None:
        return
    _, padded, signature, inputs, (_, _, metadata, zero_row) = full
    kwargs = dict(input_ids=p.input_ids[:padded], positions=p._get_positions(padded),
                  inputs_embeds=None, hidden_states=p.hidden_states[:padded])
    if _inputs(kwargs) != inputs or _metadata(p, metadata) != signature:
        return
    slots = p._get_slot_mapping(padded)
    block = int(p.block_size)
    caches = []
    for n in sorted(p._draft_attn_layer_names):
        c = p.compilation_config.static_forward_context[n].kv_cache
        if (not isinstance(c, torch.Tensor) or not c.is_cuda or c.dim() != 4
                or c.shape[0] < 4 or c.shape[2] != block):
            return
        caches.append(c[:4])
    device = p.device
    values = [gate, p.input_ids[:padded], p.positions[:padded], p.hidden_states[:padded],
              p._slot_mapping_buffer[:padded], screen._h, screen._out]
    values.extend(caches)
    values.extend(t for t in p.model._moe_row_src_buffers())
    torch.cuda.synchronize(device)
    saved.extend((t, t.clone()) for t in values)
    restore.append(lambda: p.model.set_moe_row_src(min(p.max_num_tokens, 5), None))

    p.model.set_moe_row_src(padded, zero_row)
    stop = torch.tensor([1, 1, 1, kernels.GATE_PROD_ONE], dtype=torch.int32, device=device)
    # The stopped screen's token, which must not depend on the row.
    rows = [0.0, 1.0, -2.5, float("nan"), float("inf"), float("-inf")]
    seen = set()
    for i, v in enumerate(rows + [None, None]):
        gate.copy_(stop)
        if v is None:
            screen._h.copy_(torch.randn(screen._h.shape, device=device,
                                        generator=torch.Generator(device).manual_seed(i)))
        else:
            screen._h.fill_(v)
        screen._out.fill_(-7 - i)
        screen.graph.replay()
        torch.cuda.synchronize(device)
        if not torch.equal(gate, stop):
            raise ValueError("stopped screen changed the gate")
        seen.add(int(screen._out[0]))
    if len(seen) != 1:
        raise ValueError("stopped screen token depends on its row")
    token = seen.pop()

    live = torch.ones(1, dtype=torch.bool, device=device)
    start = torch.tensor([0, 0, 1, kernels.GATE_PROD_ONE], dtype=torch.int32, device=device)

    def body():
        with set_forward_context(metadata, p.vllm_config, num_tokens=padded,
                                 cudagraph_runtime_mode=CUDAGraphMode.FULL,
                                 slot_mapping=slots):
            hidden = p.model(**kwargs)
        screen._h.copy_(hidden[0])
        screen._greedy_row(screen._h, gated=True, out=screen._out)
        return hidden

    for _ in range(2):
        gate.copy_(start)
        hidden = body()
        _arm[(1,)](gate, live, screen._out, token)
        _clear[(1,)](live, hidden, N=hidden.numel(), BLOCK=1024)
    torch.cuda.synchronize(device)
    gate.copy_(start)
    plain = torch.cuda.CUDAGraph()
    with torch.cuda.graph(plain, pool=torch.cuda.graph_pool_handle()):
        body()
    torch.cuda.synchronize(device)
    del plain
    if not _probe(device):
        raise ValueError("conditional graph probe failed")
    gate.copy_(start)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    inner = []
    try:
        with torch.cuda.graph(graph, stream=stream, pool=torch.cuda.graph_pool_handle()):
            _arm[(1,)](gate, live, screen._out, token)
            current = torch.cuda.CUDAGraph.get_currently_capturing_graph()
            current.begin_capture_to_if_node(live)
            try:
                inner.append(torch.cuda.current_stream())
                # Keep every stream-tracking helper on the body's capture stream.
                torch.cuda.set_stream(inner[-1])
                hidden = body()
            finally:
                try:
                    current.end_capture_to_conditional_node()
                except Exception:
                    _end_capture(inner[-1] if inner else None)
                    raise
                finally:
                    torch.cuda.set_stream(stream)
            _clear[(1,)](live, hidden, N=hidden.numel(), BLOCK=1024)
    except Exception:
        _end_capture(inner[-1] if inner else None)
        _end_capture(stream)
        raise
    torch.cuda.synchronize(device)

    def original():
        hid = nemotron_full_step.maybe_run(p, kwargs, metadata, padded, 1)
        if hid is None:
            raise ValueError("full-step graph declined private input")
        return hid, screen.greedy(hid[:1])

    def new():
        graph.replay()
        return hidden, screen._out.clone()

    slot = block + 3

    def setup(case, gate_value):
        for t, v in saved:
            t.copy_(v)
        p.model.set_moe_row_src(padded, zero_row)
        gen = torch.Generator(device).manual_seed(1000 + case)
        for c in caches:
            c.view(torch.uint8).copy_(torch.randint(
                0, 120, c.view(torch.uint8).shape, dtype=torch.uint8,
                device=device, generator=gen))
        p.input_ids[:padded].copy_(torch.randint(
            0, 131072, (padded,), device=device, generator=gen).to(p.input_ids.dtype))
        p.hidden_states[:padded].copy_((torch.randn(
            p.hidden_states[:padded].shape, device=device, generator=gen)
            * (0.25 + case)).to(p.hidden_states.dtype))
        p._slot_mapping_buffer[:padded].fill_(-1)
        p._slot_mapping_buffer[0] = slot
        gate.copy_(gate_value)
        screen._out.fill_(-1)

    def observe(hid, tok, stopped):
        out = [_bits(tok), _bits(gate), _bits(screen._out), _bits(p.input_ids[:padded]),
               _bits(p.hidden_states[:padded]), _bits(p.positions[:padded])]
        if not stopped:
            out.append(_bits(hid[:1]))
        out.extend(_bits(c) for c in caches)
        return out

    live_gates = [start, torch.tensor([0, 2, 1, 0x3F000000], dtype=torch.int32, device=device),
                  torch.zeros(4, dtype=torch.int32, device=device)]
    cases = [(g, False) for g in live_gates] + [(stop, True)]
    for case in range(3):
        for g, stopped in cases:
            setup(case, g)
            a = observe(*original(), stopped)
            kept = [c.clone() for c in caches] if stopped else None
            setup(case, g)
            hid, tok = new()
            if stopped:
                torch.cuda.synchronize(device)
                if bool((hid != 0).any()):
                    raise ValueError("stopped step left a hidden row")
                # A stopped pass leaves the drafter KV slot at its own position
                # unwritten; that position is written again before any read.
                for c, o in zip(caches, kept):
                    c[slot // block, :, slot % block] = o[slot // block, :, slot % block]
            b = observe(hid, tok, stopped)
            if len(a) != len(b) or not all(torch.equal(x, y) for x, y in zip(a, b)):
                raise ValueError(f"conditional step differs (case {case}, stopped={stopped})")
    timing = None
    if SPEED:
        timing = _timed(device, original, new, setup, start, stop)
        live_cost = timing[1] - timing[0]
        stop_gain = timing[2] - timing[3]
        if not (stop_gain > 0 and live_cost <= 15.0
                and stop_gain >= 4.0 * max(live_cost, 0.0)):
            raise ValueError(f"conditional step not faster: {timing}")
    p._nemo_step = SimpleNamespace(
        graph=graph, hidden=hidden, screen=screen, padded=padded, signature=signature,
        inputs=inputs, live=live, owners=(kwargs, metadata, zero_row, stream, slots),
        token=token, timing=timing)


_PROBE = r"""
import sys, torch, triton, triton.language as tl
dev = torch.device("cuda", int(sys.argv[1]))
torch.cuda.set_device(dev)
pdl = torch.cuda.get_device_capability(dev)[0] >= 9

@triton.jit
def arm(gate, live, out, token):
    tl.store(live, tl.load(gate) == 0)
    tl.store(out, token)

@triton.jit
def step(x, y, gate, n, BLOCK: tl.constexpr, PDL: tl.constexpr):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if tl.load(gate) != 0:
        tl.store(y + o, tl.zeros((BLOCK,), tl.float32), mask=o < n)
        return
    tl.store(y + o, tl.load(x + o, mask=o < n, other=0.0) * 1.5 + 0.25, mask=o < n)

n = 1 << 14
x = torch.randn(n, device=dev)
w = torch.randn(64, 64, device=dev, dtype=torch.bfloat16)
gate = torch.zeros(4, dtype=torch.int32, device=dev)
live = torch.ones(1, dtype=torch.bool, device=dev)
out = torch.zeros(1, dtype=torch.int32, device=dev)
side = torch.cuda.Stream(device=dev)

def body():
    y = torch.empty(n, device=dev)
    z = torch.empty(n, device=dev)
    for i in range(6):
        kw = dict(launch_pdl=True) if pdl else {}
        step[(n // 1024,)](x if i % 2 == 0 else z, y if i % 2 == 0 else z, gate, n,
                           BLOCK=1024, PDL=pdl, **kw)
        if i % 2 == 0:
            z.copy_(y)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        m = (y[:4096].view(64, 64).to(torch.bfloat16) @ w).float()
    torch.cuda.current_stream().wait_stream(side)
    out.copy_((z[:1] * 100 + m[0, :1]).to(torch.int32))
    return z

body(); arm[(1,)](gate, live, out, 0); torch.cuda.synchronize()
ref = torch.cuda.CUDAGraph()
with torch.cuda.graph(ref):
    zr = body(); tr = out.clone()
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream(device=dev)
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.graph(g, stream=s):
    arm[(1,)](gate, live, out, 4242)
    cur = torch.cuda.CUDAGraph.get_currently_capturing_graph()
    cur.begin_capture_to_if_node(live)
    try:
        torch.cuda.set_stream(torch.cuda.current_stream())
        zc = body()
    finally:
        cur.end_capture_to_conditional_node()
        torch.cuda.set_stream(s)
    tc = out.clone()
torch.cuda.synchronize()
for trial in range(3):
    x.copy_(torch.randn(n, device=dev))
    gate.zero_(); ref.replay(); torch.cuda.synchronize(); a = (tr.clone(), zr.clone())
    zc.fill_(7.0); gate.zero_(); g.replay(); torch.cuda.synchronize()
    assert torch.equal(a[0], tc) and torch.equal(a[1].view(torch.int32), zc.view(torch.int32))
    zc.fill_(7.0); gate[0] = 1; g.replay(); torch.cuda.synchronize()
    assert int(tc[0]) == 4242 and bool((zc == 7.0).all())
print("probe-ok")
"""


def _probe(device) -> bool:
    """Checks the conditional-node construct (PDL kernels, side stream, GEMM,
    allocation inside the body) in a child process."""
    import subprocess
    import sys
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "probe.py")
            with open(path, "w") as f:
                f.write(_PROBE)
            r = subprocess.run([sys.executable, path, str(device.index or 0)],
                               capture_output=True, text=True, timeout=120, cwd=d)
            if r.returncode == 0 and "probe-ok" in r.stdout:
                return True
            logger.warning("conditional graph probe: rc=%s %s", r.returncode, r.stderr[-400:])
            return False
    except Exception as exc:
        logger.warning("conditional graph probe: %r", exc)
        return False


def _end_capture(stream):
    """Ends a stream capture left open by a failed body capture."""
    if stream is None:
        return
    try:
        import ctypes
        cuda = ctypes.CDLL("libcuda.so.1")
        handle = ctypes.c_void_p(stream.cuda_stream)
        status = ctypes.c_int(0)
        if cuda.cuStreamIsCapturing(handle, ctypes.byref(status)) == 0 and status.value != 0:
            graph = ctypes.c_void_p()
            cuda.cuStreamEndCapture(handle, ctypes.byref(graph))
            if graph.value:
                cuda.cuGraphDestroy(graph)
    except Exception:
        pass


def _timed(device, original, new, setup, live_gate, stopped_gate, trials=15):
    """GPU span of each path: a device spin queued ahead hides host launch time."""
    pressure = torch.empty(64 << 20, dtype=torch.uint8, device=device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    result = []
    for g in (live_gate, stopped_gate):
        samples = [[], []]
        for trial in range(trials):
            for which in ((0, 1) if trial % 2 == 0 else (1, 0)):
                setup(trial % 3, g)
                pressure.fill_(trial)
                torch.cuda._sleep(2_000_000)
                start.record()
                (original if which == 0 else new)()
                end.record()
                torch.cuda.synchronize(device)
                samples[which].append(start.elapsed_time(end) * 1000.0)
        result += [statistics.median(samples[0]), statistics.median(samples[1])]
    # (live original, live new, stopped original, stopped new) in microseconds
    return tuple(round(v, 2) for v in result)

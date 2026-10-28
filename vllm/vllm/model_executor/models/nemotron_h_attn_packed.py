"""Lossless 12-bit storage of the target's BF16 attention projections for the 5-row verify.

Each of the six attention layers streams its QKV (4608x2688) and output (2688x4096)
projection from DRAM once per verify: 46.7 MB of BF16 per layer, about 1.25 ms of a
17.8 ms cycle. The weights are repacked once at load with the drafter's exact codec
(nemotron_h_mtp_packed.pack_bf16: sign+mantissa byte, 4-bit exponent code relative to a
per-128-block base, raw side plane for blocks whose exponents do not fit) and the
verify's M = 5 call decodes them in registers into the same BF16 MMA operands
(nemotron_h_attn_packed_kernel.py).

Exactness: the kernel is one tl.dot chain over K tiles of 128 into an fp32 accumulator,
rounded once to BF16. That is the reduction the cuBLAS call produces for M = 2..8 on
these shapes (measured bit-identical on the real weights by an earlier submission,
b9787b2). It is not assumed here: the boot guard compares every packed layer against
F.linear on random inputs and the whole path stays off on a single differing bit.
It also stays off unless a cold-cache timing of the chosen configuration beats
F.linear, and unless a child process has compiled and run every configuration on
synthetic planes first (a compiler abort must not take the engine with it).
Every other row count (prefill, M = 1) calls F.linear on the BF16 weight.

The planes live in a registry the custom op reads, so the compiled graph only sees
(x, weight, n, k, id). ATTN_PACK12=0 disables it.

harness/kernels/apk12 runs the same chain (mma.sync m16n8k16 BF16, ascending K from +0,
one RN rounding) as a CUDA kernel over a fragment-order copy of the planes. It is ranked
with the Triton configurations under the same speed and equality guards, after its own
child-process probe and a load-time check of every decoded weight against the BF16 bits.

LOCAL PATCH (cudafast-nemotron35): the 496-row prefill call of each projection runs
harness/kernels/bf16ws, the same chain as a persistent TMA kernel over a load-time
byte-permuted copy of the BF16 weight, when that projection's copy inverts to the original
bytes and its output equals F.linear bit for bit on synthetic activation sets at load
(_prepare_ws); otherwise F.linear. BF16WS=0 disables it.
"""

import os
import subprocess
import sys

import torch
import torch.nn.functional as F
from vllm.triton_utils import triton, tl

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from .nemotron_h_attn_packed_kernel import CONFIGS, ROWS, run

try:
    import apk12 as _apk12
except ImportError:  # harness/kernels/apk12 not built: the Triton configurations only
    _apk12 = None

logger = init_logger(__name__)

ENABLED = os.environ.get("ATTN_PACK12", "1") != "0"
SHAPES = frozenset({(4608, 2688), (2688, 4096)})
MIN_GAIN = 0.96
KERNEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nemotron_h_attn_packed_kernel.py")
CUDA = ("apk12",)
CUDA_PDL = ("apk12_pdl",)
CUDA_PDL8 = ("apk12_pdl8",)
_cuda_ok = False
_cuda_pdl_ok = False
_cuda_pdl8_ok = False

_chosen: dict[tuple[int, int], tuple] = {}
_word_ok = False
_triton_ok = False
_WORD = os.environ.get("ATTN_PACK12_WORDLOAD", "1") != "0"
# LOCAL PATCH (cudafast-nemotron35): offer the eight-deep-ring PDL variant to
# the existing exactness and timing choice. ATTN_PACK12_PDL8=0 removes it.
_PDL8 = os.environ.get("ATTN_PACK12_PDL8", "1") != "0"
_registry: dict[int, tuple[torch.Tensor, ...]] = {}
# bf16ws: weight storage address -> (weight, pack); the weight reference keeps the address valid
WS_ENABLED = os.environ.get("BF16WS", "1") != "0"
WS_ROWS = 496
_ws: dict[int, tuple[torch.Tensor, torch.Tensor, bool]] = {}
_ws_stats = {"admitted": 0, "rejected": 0, "dispatched": 0}

_PROBE = """
import importlib.util, sys, torch
spec = importlib.util.spec_from_file_location("apk_kernel", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
dev = torch.device("cuda")
for n, k in ((4608, 2688), (2688, 4096)):
    nb = n * k // 128
    sm = torch.randint(0, 256, (n * k,), dtype=torch.uint8, device=dev)
    ex = torch.randint(0, 256, (n * k // 2,), dtype=torch.uint8, device=dev)
    base = torch.randint(60, 120, (nb,), dtype=torch.uint8, device=dev)
    ri = torch.full((nb,), -1, dtype=torch.int32, device=dev)
    ri[0] = 0
    raw = torch.randint(-32768, 32767, (128,), dtype=torch.int16, device=dev)
    x = torch.randn(m.ROWS, k, device=dev, dtype=torch.bfloat16)
    for cfg in m.CONFIGS:
        group = "word" if len(cfg) == 4 else "base"
        if group != sys.argv[2]:
            continue
        y = m.run(x, sm, ex, base, ri, raw, n, k, cfg)
        torch.cuda.synchronize()
        assert y.shape == (m.ROWS, n)
print("probe-ok")
"""

_PROBE_CUDA = """
import importlib.util, sys, torch
spec = importlib.util.spec_from_file_location("apk12", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
dev = torch.device("cuda")
pdl = sys.argv[2] in ("apk12_pdl", "apk12_pdl8")
ring = 8 if sys.argv[2] == "apk12_pdl8" else 4
assert (m.ready_pdl(dev) if pdl else m.ready(dev))
for n, k in ((4608, 2688), (2688, 4096)):
    nb = n * k // 128
    sm = torch.randint(0, 256, (n * k,), dtype=torch.uint8, device=dev)
    ex = torch.randint(0, 256, (n * k // 2,), dtype=torch.uint8, device=dev)
    ex[64:128] = 0
    base = torch.randint(1, 120, (nb,), dtype=torch.uint8, device=dev)
    ri = torch.full((nb,), -1, dtype=torch.int32, device=dev)
    ri[0], ri[nb - 1] = 0, 1
    raw = torch.randint(-32768, 32767, (256,), dtype=torch.int16, device=dev)
    c = torch.stack((ex & 15, ex >> 4), -1).view(-1).int()
    e = torch.where(c == 0, 0, base.int().repeat_interleave(128) + c - 1)
    s = sm.int()
    w = ((s & 128) << 8) | (e << 7) | (s & 127)
    esc = (ri >= 0).repeat_interleave(128)
    w = torch.where(esc, raw.int().view(-1, 128)[ri.clamp(min=0)].view(-1), w).short()
    cp = m.pack((sm, ex, base, ri, raw), n, k)
    assert m.check(cp, w, n, k) == 0
    x = torch.randn(5, k, device=dev, dtype=torch.bfloat16)
    reference = m.gemm(x, cp, n, k)
    y = m.gemm(x, cp, n, k, pdl=pdl, ring=ring)
    torch.cuda.synchronize()
    assert y.shape == (5, n)
    assert torch.equal(y.view(torch.uint8), reference.view(torch.uint8))
print("probe-ok")
"""


def _run(x: torch.Tensor, planes, n: int, k: int, cfg: tuple) -> torch.Tensor:
    if cfg in (CUDA, CUDA_PDL, CUDA_PDL8):
        return _apk12.gemm(x, planes, n, k, pdl=cfg != CUDA, ring=8 if cfg == CUDA_PDL8 else 4)
    return run(x, *planes, n, k, cfg)


def _planes(item, cfg: tuple):
    """An item is (proj, codec planes, apk12's copy of them or None)."""
    return item[2] if cfg in (CUDA, CUDA_PDL, CUDA_PDL8) else item[1]


def _ws_gemm(x: torch.Tensor, w: torch.Tensor, n: int, k: int) -> torch.Tensor | None:
    """bf16ws output for an admitted weight at 496 rows, else None (the caller runs F.linear)."""
    entry = _ws.get(w.data_ptr())
    if entry is None:
        return None
    weight, packed, pipe = entry
    if not (
        w.shape == weight.shape and w.stride() == weight.stride()
        and x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2
        and x.shape[0] == WS_ROWS and x.shape[1] == k and x.is_contiguous()
        and x.data_ptr() % 16 == 0 and x.device == packed.device
    ):
        return None
    import bf16ws

    if not _ws_stats["dispatched"]:
        logger.info("bf16ws: 496-row attention projections dispatched to the packed kernel")
    _ws_stats["dispatched"] += 1
    return bf16ws.gemm(x, packed, n, pipe=pipe)


def _linear_impl(x: torch.Tensor, w: torch.Tensor, n: int, k: int, pid: int) -> torch.Tensor:
    if x.shape[0] == WS_ROWS and _ws:
        out = _ws_gemm(x, w, n, k)
        if out is not None:
            return out
    cfg = _chosen.get((n, k))
    planes = _registry.get(pid)
    if (
        cfg is None
        or planes is None
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.dim() != 2
        or x.shape[0] != ROWS
        or x.shape[1] != k
        or x.stride(1) != 1
    ):
        return F.linear(x, w)
    return _run(x, planes, n, k, cfg)


def _linear_fake(x: torch.Tensor, w: torch.Tensor, n: int, k: int, pid: int) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], n))


direct_register_custom_op(
    op_name="nemo_attn_pk_linear", op_func=_linear_impl, fake_impl=_linear_fake,
)


def apply(proj: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """proj(x)[0] for an attention projection, packed when prepare() enabled it."""
    nk = getattr(proj, "_apk_nk", None)
    if nk is None:
        nk = getattr(proj, "_bws_nk", None)
        if nk is None:
            return proj(x)[0]
        # bf16ws only (no planes): 496 rows run the packed kernel, every other count F.linear
        return torch.ops.vllm.nemo_attn_pk_linear(x, proj.weight, nk[0], nk[1], -1)
    return torch.ops.vllm.nemo_attn_pk_linear(x, proj.weight, nk[0], nk[1], proj._apk_id)


def _probe_group(group: str, script: str = _PROBE, path: str = KERNEL_PATH) -> bool:
    try:
        out = subprocess.run(
            [sys.executable, "-c", script, path, group],
            capture_output=True, text=True, timeout=240,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("attention pack12: probe did not run (%r); off", exc)
        return False
    if out.returncode != 0 or "probe-ok" not in out.stdout:
        logger.warning(
            "attention pack12: probe failed (rc %s): %s", out.returncode, out.stderr[-400:]
        )
        return False
    return True


def _probe() -> bool:
    global _word_ok
    if not _probe_group("base"):
        return False
    _word_ok = _WORD and _probe_group("word")
    return True


def _probe_cuda(dev: torch.device) -> bool:
    global _cuda_ok, _cuda_pdl_ok, _cuda_pdl8_ok
    ordinary = (_apk12 is not None
                and _probe_group(CUDA[0], _PROBE_CUDA, _apk12.__file__)
                and _apk12.ready(dev))
    _cuda_pdl_ok = bool(
        ordinary and hasattr(_apk12, "ready_pdl")
        and _probe_group(CUDA_PDL[0], _PROBE_CUDA, _apk12.__file__)
        and _apk12.ready_pdl(dev)
    )
    _cuda_pdl8_ok = bool(
        _PDL8 and _cuda_pdl_ok and _probe_group(CUDA_PDL8[0], _PROBE_CUDA, _apk12.__file__)
        and _apk12.ready_pdl(dev)
    )
    # ENTIRE fc043 carries the ordinary CUDA path. Retain that qualified
    # baseline choice while independently admitting the new PDL variants.
    _cuda_ok = bool(ordinary)
    return _cuda_ok or _cuda_pdl_ok or _cuda_pdl8_ok


def _cuda_planes(planes, w: torch.Tensor, n: int, k: int):
    """apk12's copy of the planes if its kernel decodes every weight to w's bits."""
    try:
        cp = _apk12.pack(planes, n, k)
        bad = _apk12.check(cp, w, n, k)
    except Exception as exc:  # noqa: BLE001
        logger.warning("attention pack12 %dx%d: apk12 pack failed (%r)", n, k, exc)
        return None
    if bad:
        logger.warning("attention pack12 %dx%d: apk12 decode differs in %d registers", n, k, bad)
        return None
    return cp


@triton.jit
def _probe_producer(src, dst, N: tl.constexpr, B: tl.constexpr,
                    launch_pdl: tl.constexpr = False):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    ix = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(src + ix, mask=ix < N, other=0.0)
    tl.store(dst + ix, value, mask=ix < N)


@torch.no_grad()
def _exact_cuda_graph(items, n: int, k: int, cfg: tuple) -> bool:
    """Private remote replay, including an early-signaling mutable producer.

    All temporary inputs and captured outputs are synthetic and released. This
    checks fresh input visibility after PDL wait, beyond immutable weight bits.
    """
    device = items[0][0].weight.device
    gen = torch.Generator(device=device).manual_seed(643821)
    probes = [(torch.randn((ROWS, k), device=device, generator=gen) * scale)
              .to(torch.bfloat16) for scale in (0.001, 1., 30.)]
    edge = torch.zeros((ROWS, k), dtype=torch.bfloat16, device=device)
    edge.view(torch.int16)[:, :8] = torch.tensor(
        [0, -32768, 1, -32767, 128, -32640, 16256, -16512],
        dtype=torch.int16, device=device).view(1, 8)
    sparse = probes[1].clone()
    sparse[:, ::2] = 0.
    sparse[:, 2::4] = -0.
    probes.extend((edge, sparse, torch.zeros_like(edge)))
    for item in items:
        proj, planes = item[0], _planes(item, cfg)
        src = probes[1].clone()
        xbuf = torch.empty_like(src)
        # Compile and warm the exact producer/consumer before capture.
        _probe_producer[(triton.cdiv(src.numel(), 1024),)](
            src, xbuf, N=src.numel(), B=1024, launch_pdl=True)
        _run(xbuf, planes, n, k, cfg)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
            _probe_producer[(triton.cdiv(src.numel(), 1024),)](
                src, xbuf, N=src.numel(), B=1024, launch_pdl=True)
            output = _run(xbuf, planes, n, k, cfg)
        torch.cuda.synchronize(device)
        # Repeat changing generations in different order; pointers stay fixed.
        for probe in (*probes, *reversed(probes)):
            src.copy_(probe)
            graph.replay()
            expected = F.linear(probe, proj.weight)
            if not torch.equal(output.view(torch.uint8), expected.view(torch.uint8)):
                return False
    return True


def _exact(items, n: int, k: int, cfg: tuple[int, int]) -> bool:
    dev = items[0][0].weight.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(20261025)
    for item in items:
        proj, planes = item[0], _planes(item, cfg)
        for scale in (1.0, 30.0, 1e-3):
            for _ in range(8):
                x = torch.randn(ROWS, k, device=dev, dtype=torch.float32, generator=gen)
                x = (x * scale).to(torch.bfloat16)
                actual, expected = _run(x, planes, n, k, cfg), F.linear(x, proj.weight)
                if cfg in (CUDA, CUDA_PDL, CUDA_PDL8):
                    if not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
                        return False
                elif not torch.equal(actual, expected):
                    return False
    if cfg in (CUDA, CUDA_PDL, CUDA_PDL8):
        return _exact_cuda_graph(items, n, k, cfg)
    return True


def _op_us(fn, flush: torch.Tensor, reps: int = 20) -> float:
    """Median cold-weight GPU time of fn in microseconds. The L2 flush keeps the GPU busy
    long enough for the host to queue the event pair and fn behind it, so the events
    bracket fn alone."""
    fn()
    torch.cuda.synchronize()
    spans = []
    for _ in range(reps):
        flush.zero_()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        spans.append((s, e))
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in spans)
    return ms[len(ms) // 2] * 1000.0


def _graph_us(fns, reps: int = 24, rounds: int = 7) -> float:
    """Median GPU time per call in microseconds: a CUDA graph of `reps` calls that cycle
    through the per-layer callables, so every call reads weights the previous ones pushed
    out of L2 (6 layers of 18-25 MB each) and no host launch time is in the measurement."""
    for fn in fns:
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(reps):
            fns[i % len(fns)]()
    g.replay()
    torch.cuda.synchronize()
    ms = []
    for _ in range(rounds):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ms.append(a.elapsed_time(b))
    return sorted(ms)[rounds // 2] * 1000.0 / reps


def _choose(items, n: int, k: int, flush: torch.Tensor):
    proj = items[0][0]
    x = torch.randn(ROWS, k, device=proj.weight.device, dtype=torch.bfloat16)
    use_graph = True
    try:
        linear_us = _graph_us([(lambda w=it[0].weight: F.linear(x, w)) for it in items])
    except Exception as exc:  # noqa: BLE001
        logger.warning("attention pack12 graph timing unavailable (%r); paired eager timing", exc)
        use_graph = False
        linear_us = _op_us(lambda: F.linear(x, proj.weight), flush)
    choices = []
    cuda = (([CUDA] if _cuda_ok else [])
            + ([CUDA_PDL] if _cuda_pdl_ok else [])
            + ([CUDA_PDL8] if _cuda_pdl8_ok else [])) if all(it[2] is not None for it in items) else []
    for cfg in cuda + (list(CONFIGS) if _triton_ok else []):
        if len(cfg) == 4 and not _word_ok:
            continue
        if cfg not in (CUDA, CUDA_PDL, CUDA_PDL8) and n % cfg[0]:
            continue
        try:
            if use_graph:
                us = _graph_us([(lambda pl=_planes(it, cfg): _run(x, pl, n, k, cfg)) for it in items])
            else:
                us = _op_us(lambda: _run(x, _planes(items[0], cfg), n, k, cfg), flush)
        except Exception as exc:  # noqa: BLE001  (a configuration this Triton cannot build)
            logger.warning("attention pack12 %dx%d: config %s failed (%r)", n, k, cfg, exc)
            continue
        choices.append((us, cfg))
    choices.sort(key=lambda item: item[0])
    logger.info("attention pack12 %dx%d: linear %.1f us, packed %s", n, k, linear_us, choices)
    for us, cfg in choices:
        if us > MIN_GAIN * linear_us:
            break
        try:
            if _exact(items, n, k, cfg):
                return cfg
        except Exception as exc:  # noqa: BLE001
            logger.warning("attention pack12 %dx%d: equality failed for %s (%r)", n, k, cfg, exc)
        logger.warning("attention pack12 %dx%d: rejecting configuration %s", n, k, cfg)
    return None


@torch.no_grad()
def _ws_activations(device: torch.device, k: int) -> list[torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(20261027)
    sets = []
    for scale in (1.0, 0.01, 30.0, 0.001):
        # explicit dtypes: model loading runs under a BF16 default dtype
        x = torch.randn((WS_ROWS, k), device=device, generator=g, dtype=torch.float32) * scale
        sets.append(x.to(torch.bfloat16))
    # every BF16 bit pattern (infinities and NaNs included), shifted per row
    bits = (torch.arange(WS_ROWS * k, device=device, dtype=torch.int64) * 7
            + torch.arange(WS_ROWS, device=device, dtype=torch.int64).repeat_interleave(k) * 131)
    bits = bits % 65536
    bits = torch.where(bits >= 32768, bits - 65536, bits).to(torch.int16)
    sets.append(bits.view(torch.bfloat16).reshape(WS_ROWS, k))
    return sets


@torch.no_grad()
def _ws_pipe_qualify(bf16ws, packed, n, activations) -> bool:
    """Check the optional pipeline on private owners and time complete captured calls."""
    if os.environ.get("BF16WS_STAGE_PIPE", "1") == "0":
        return False
    device = packed.device
    try:
        if torch.cuda.is_current_stream_capturing() or not bf16ws.pipe_ready(device):
            return False
        # Compare full output words for the same finite, signed-zero and
        # non-finite BF16 inputs already used by the reference admission.
        for a in activations:
            ref = bf16ws.gemm(a, packed, n)
            alt = bf16ws.gemm(a, packed, n, pipe=True)
            if not torch.equal(ref.view(torch.int16), alt.view(torch.int16)):
                return False
        torch.cuda.synchronize(device)
        staged = activations[0].clone()
        out0 = torch.empty((WS_ROWS, n), device=device, dtype=torch.bfloat16)
        out1 = torch.empty_like(out0)
        for _ in range(2):
            bf16ws.gemm(staged, packed, n, out=out0)
            bf16ws.gemm(staged, packed, n, pipe=True, out=out1)
        torch.cuda.synchronize(device)
        graphs = []
        for is_pipe, out in ((False, out0), (True, out1)):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
                for _ in range(3):
                    bf16ws.gemm(staged, packed, n, pipe=is_pipe, out=out)
            graphs.append(graph)
        ratios = []
        for sample, a in enumerate(activations[:2]):
            staged.copy_(a)
            graphs[0].replay()
            graphs[1].replay()
            if not torch.equal(out0.view(torch.int16), out1.view(torch.int16)):
                return False
            for trial in range(5):
                measured = [0.0, 0.0]
                for which in ((0, 1) if (sample + trial) % 2 == 0 else (1, 0)):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[which].replay()
                    end.record()
                    end.synchronize()
                    measured[which] = start.elapsed_time(end)
                if measured[0] <= 0 or measured[1] <= 0:
                    return False
                ratios.append(measured[1] / measured[0])
        ratios.sort()
        return ratios[len(ratios) // 2] < 0.99 and ratios[-1] < 1.01
    except Exception as exc:  # noqa: BLE001 - keep the existing admitted kernel
        logger.warning("bf16ws: stage pipeline admission failed (%r); original kept", exc)
        return False


@torch.no_grad()
def _ws_faster_than_linear(bf16ws, w, packed, n, activations, pipe: bool) -> bool:
    """Admit the selected BF16WS path only if it beats the actual base F.linear."""
    device = packed.device
    try:
        if torch.cuda.is_current_stream_capturing():
            return False
        staged = activations[0].clone()
        out = torch.empty((WS_ROWS, n), device=device, dtype=torch.bfloat16)
        for _ in range(2):
            F.linear(staged, w)
            bf16ws.gemm(staged, packed, n, pipe=pipe, out=out)
        torch.cuda.synchronize(device)
        linear_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(linear_graph, pool=torch.cuda.graph_pool_handle()):
            for _ in range(3):
                linear_out = F.linear(staged, w)
        packed_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(packed_graph, pool=torch.cuda.graph_pool_handle()):
            for _ in range(3):
                bf16ws.gemm(staged, packed, n, pipe=pipe, out=out)
        graphs = (linear_graph, packed_graph)
        ratios = []
        for sample, a in enumerate(activations[:2]):
            staged.copy_(a)
            linear_graph.replay()
            packed_graph.replay()
            if not torch.equal(linear_out.view(torch.int16), out.view(torch.int16)):
                return False
            for trial in range(5):
                measured = [0.0, 0.0]
                for which in ((0, 1) if (sample + trial) % 2 == 0 else (1, 0)):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[which].replay()
                    end.record()
                    end.synchronize()
                    measured[which] = start.elapsed_time(end)
                if measured[0] <= 0 or measured[1] <= 0:
                    return False
                ratios.append(measured[1] / measured[0])
        ratios.sort()
        return ratios[len(ratios) // 2] < 0.99 and ratios[-1] < 1.01
    except Exception as exc:  # noqa: BLE001 - keep the highest-base F.linear
        logger.warning("bf16ws: F.linear comparison failed (%r); original kept", exc)
        return False


def _prepare_ws(projs) -> int:
    """Pack and admit the 496-row bf16ws copy of each projection: the pack must invert to the
    weight's bytes and the kernel must equal F.linear bit for bit on every synthetic set."""
    if not WS_ENABLED:
        return 0
    try:
        import bf16ws
    except ImportError:
        return 0
    for proj in projs:
        w = proj.weight
        n, k = w.shape
        ok = False
        try:
            packed = bf16ws.prepare(w)
            if packed is not None:
                ok = torch.equal(bf16ws.unpack_reference(packed, n, k), w.view(torch.uint8))
                activations = _ws_activations(w.device, k)
                for a in activations:
                    if not ok:
                        break
                    ok = torch.equal(F.linear(a, w).view(torch.int16),
                                     bf16ws.gemm(a, packed, n).view(torch.int16))
                torch.cuda.synchronize(w.device)
        except Exception as exc:  # noqa: BLE001  (any failure keeps F.linear)
            logger.warning("bf16ws: admission failed (%r); this projection keeps F.linear", exc)
            ok = False
        if not ok:
            _ws_stats["rejected"] += 1
            continue
        pipe = _ws_pipe_qualify(bf16ws, packed, n, activations)
        if not _ws_faster_than_linear(bf16ws, w, packed, n, activations, pipe):
            _ws_stats["rejected"] += 1
            continue
        _ws[w.data_ptr()] = (w, packed, pipe)
        proj._bws_nk = (n, k)
        _ws_stats["admitted"] += 1
    logger.info("bf16ws: %d attention projections admitted for the 496-row prefill (%d rejected); "
                "BF16WS=0 disables", _ws_stats["admitted"], _ws_stats["rejected"])
    return _ws_stats["admitted"]


def prepare(model: torch.nn.Module) -> int:
    """Pack the target's attention projections and enable the shapes that pass the
    guards. Returns the number of projections enabled."""
    if not torch.cuda.is_available():
        return 0
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    projs = []
    for module in model.modules():
        if type(module).__name__ != "NemotronHAttention":
            continue
        for proj in (module.qkv_proj, module.o_proj):
            w = getattr(proj, "weight", None)
            if (
                w is not None
                and w.dtype == torch.bfloat16
                and w.is_cuda
                and w.is_contiguous()
                and tuple(w.shape) in SHAPES
                and isinstance(getattr(proj, "quant_method", None), UnquantizedLinearMethod)
            ):
                projs.append(proj)
    if not projs:
        return 0
    _prepare_ws(projs)
    if not ENABLED:
        return 0
    global _triton_ok
    _triton_ok = _probe()
    cuda_ok = _probe_cuda(projs[0].weight.device)
    if not (_triton_ok or cuda_ok):
        return 0

    from .nemotron_h_mtp_packed import pack_bf16

    groups: dict[tuple[int, int], list] = {}
    for proj in projs:
        packed = pack_bf16(proj.weight)
        if packed is None:
            continue
        sm, ex, base, ri, raw = packed.planes
        planes = (sm, ex, base, ri, raw.view(torch.int16))
        n, k = proj.weight.shape
        cplanes = _cuda_planes(planes, proj.weight, n, k) if cuda_ok else None
        groups.setdefault((n, k), []).append((proj, planes, cplanes))
    if not groups:
        return 0
    dev = projs[0].weight.device
    flush = torch.empty(96 << 20, dtype=torch.uint8, device=dev)
    enabled = 0
    for (n, k), items in groups.items():
        cfg = None
        try:
            cfg = _choose(items, n, k, flush)
        except Exception as exc:  # noqa: BLE001
            logger.warning("attention pack12 %dx%d: guard failed (%r); off", n, k, exc)
            cfg = None
        if cfg is None:
            continue
        for item in items:
            proj, planes = item[0], _planes(item, cfg)
            pid = len(_registry)
            _registry[pid] = planes
            proj._apk_planes = planes
            proj._apk_nk = (n, k)
            proj._apk_id = pid
        _chosen[(n, k)] = cfg
        enabled += len(items)
    return enabled

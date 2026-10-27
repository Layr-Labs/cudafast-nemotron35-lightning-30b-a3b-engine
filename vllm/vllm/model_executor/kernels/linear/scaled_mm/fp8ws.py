# SPDX-License-Identifier: Apache-2.0
"""LOCAL PATCH (cudafast-nemotron35): the prefill (M=496) Mamba in_proj FP8 GEMM read from a
load-time packed copy of the weight (harness/kernels/fp8ws).

Same arithmetic as the pinned CUTLASS kernel: one ascending-K e4m3 m16n8k32 chain per output from
+0, then fp32 x scale_b, x scale_a, one RN to BF16. The pack is a byte permutation of the weight;
the activation is the producer's tensor, unchanged. Each layer's pack is admitted at load only if
its output is bitwise equal to ops.cutlass_scaled_mm on several synthetic activation sets and
scale pairs at the real shape; otherwise the layer keeps CUTLASS. The original weight stays (it
serves every other M through CUTLASS).

FP8WS=0 disables it at load. This composition retains the original prefill graph;
there is no filesystem A/B switch. After the inherited byte and output checks,
five alternating fixed-pointer graph timings must improve the actual M=496
projection with warm and 64MiB pressure inputs, or this layer keeps CUTLASS.
"""

import os
import statistics

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger

logger = init_logger(__name__)
M, N, K = 496, 10304, 2688
ENABLED = os.environ.get("FP8WS", "1") != "0"
active = True
# weight storage address -> (weight, pack); the weight reference keeps the address valid
_PACKS: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
_stats = {"admitted": 0, "rejected": 0, "dispatched": 0}


def has_packs() -> bool:
    return bool(_PACKS)


def _activations(device: torch.device) -> list[torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(20261027)
    sets = []
    for scale in (20.0, 0.02, 150.0, 1.0):
        # explicit dtypes: model loading runs under a BF16 default dtype
        x = torch.randn((M, K), device=device, generator=g, dtype=torch.float32) * scale
        sets.append(x.clamp(-448, 448).to(torch.float8_e4m3fn))
    # every E4M3 bit pattern (NaNs included), shifted per row
    bits = (torch.arange(M * K, device=device, dtype=torch.int64) * 7
            + torch.arange(M, device=device, dtype=torch.int64).repeat_interleave(K))
    sets.append((bits % 256).to(torch.uint8).view(torch.float8_e4m3fn).reshape(M, K))
    return sets


def _projection_faster(fp8ws, weight, packed, sa, sb) -> bool:
    """Official-runner-only private whole-projection graph gate, not a score."""
    dev = weight.device
    if torch.cuda.is_current_stream_capturing():
        return False
    patterns = _activations(dev)[:2]
    a = torch.empty_like(patterns[0])
    graphs, outputs, streams = [], [], []
    for which in (0, 1):
        a.copy_(patterns[0])
        call = (lambda: ops.cutlass_scaled_mm(
            a, weight, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)) if which == 0 else (
                lambda: fp8ws.gemm(a, packed, sa, sb))
        for _ in range(2):
            call()
        torch.cuda.synchronize(dev)
        stream = torch.cuda.Stream(device=dev)
        stream.wait_stream(torch.cuda.current_stream(dev))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream, pool=torch.cuda.graph_pool_handle()):
            output = call()
        torch.cuda.synchronize(dev)
        graphs.append(graph)
        outputs.append(output)
        streams.append(stream)
    pressure = torch.empty(64 << 20, device=dev, dtype=torch.uint8)
    ratios, old_times, new_times = [], [], []
    for value in patterns:
        a.copy_(value)
        for graph in graphs:
            graph.replay()
        torch.cuda.synchronize(dev)
        if not torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16)):
            return False
        for cold in (False, True):
            samples = [[], []]
            for trial in range(5):
                for which in ((0, 1) if trial % 2 == 0 else (1, 0)):
                    a.copy_(value)
                    if cold:
                        pressure.fill_(trial)
                    torch.cuda.synchronize(dev)
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[which].replay()
                    end.record()
                    end.synchronize()
                    elapsed = start.elapsed_time(end)
                    if not elapsed > 0:
                        return False
                    samples[which].append(elapsed)
            old, new = [statistics.median(v) for v in samples]
            old_times.append(old)
            new_times.append(new)
            ratios.append(new / old)
    logger.info("fp8ws: private complete-projection timing ratios=%s", tuple(ratios))
    return (statistics.median(ratios) < 0.995
            and sum(new_times) / sum(old_times) < 0.995 and max(ratios) < 1.01)


def register(weight: torch.Tensor, weight_scale: torch.Tensor | None,
             input_scale: torch.Tensor | None) -> bool:
    """Pack and admit one [K, N] column-major in_proj weight. True if its prefill uses fp8ws."""
    if not ENABLED or tuple(weight.shape) != (K, N):
        return False
    try:
        import fp8ws
    except ImportError:
        return False
    try:
        packed = fp8ws.prepare(weight)
        if packed is None:
            return False
        dev = weight.device
        f32 = torch.float32
        scales = [(torch.tensor([0.0123], device=dev, dtype=f32), torch.tensor([1.7], device=dev, dtype=f32)),
                  (torch.tensor([3.0], device=dev, dtype=f32), torch.tensor([0.00071], device=dev, dtype=f32))]
        if (weight_scale is not None and input_scale is not None
                and weight_scale.numel() == 1 and input_scale.numel() == 1):
            scales.insert(0, (input_scale.detach().float().reshape(1).contiguous(),
                              weight_scale.detach().float().reshape(1).contiguous()))
        ok = torch.equal(fp8ws.unpack_reference(packed), weight.t().view(torch.uint8))
        for a in _activations(dev):
            for sa, sb in scales:
                if not ok:
                    break
                ref = ops.cutlass_scaled_mm(a, weight, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
                got = fp8ws.gemm(a, packed, sa, sb)
                ok = torch.equal(ref.view(torch.int16), got.view(torch.int16))
        torch.cuda.synchronize(dev)
        if ok:
            ok = _projection_faster(fp8ws, weight, packed, *scales[0])
    except Exception as exc:  # noqa: BLE001  (any failure keeps CUTLASS)
        logger.warning("fp8ws: admission failed (%r); this layer keeps CUTLASS", exc)
        ok = False
    if not ok:
        _stats["rejected"] += 1
        logger.warning("fp8ws: byte/output/graph/timing admission refused; this layer keeps CUTLASS")
        return False
    _PACKS[weight.data_ptr()] = (weight, packed)
    _stats["admitted"] += 1
    if _stats["admitted"] in (1, 23):
        logger.info("fp8ws: %d Mamba in_proj weights admitted for the M=496 prefill (%d rejected); "
                    "FP8WS=0 disables", _stats["admitted"], _stats["rejected"])
    return True


def maybe_gemm(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor,
               sb: torch.Tensor) -> torch.Tensor | None:
    """fp8ws output for an admitted weight at M=496, else None (the caller runs CUTLASS)."""
    if not active or a.shape[0] != M:
        return None
    entry = _PACKS.get(b.data_ptr())
    if entry is None:
        return None
    weight, packed = entry
    if not (
        b.shape == weight.shape and b.stride() == weight.stride()
        and a.dtype == torch.float8_e4m3fn and a.shape[1] == K and a.is_contiguous()
        and a.data_ptr() % 16 == 0 and a.device == packed.device
        and sa.dtype == sb.dtype == torch.float32 and sa.numel() == sb.numel() == 1
        and sa.is_contiguous() and sb.is_contiguous()
        and sa.device == sb.device == a.device
    ):
        return None
    import fp8ws

    if not _stats["dispatched"]:
        logger.info("fp8ws: M=496 in_proj dispatched to the packed kernel")
    _stats["dispatched"] += 1
    return fp8ws.gemm(a, packed, sa, sb)

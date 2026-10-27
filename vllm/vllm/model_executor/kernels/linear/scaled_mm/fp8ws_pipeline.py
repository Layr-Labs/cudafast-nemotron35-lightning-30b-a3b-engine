# SPDX-License-Identifier: Apache-2.0
"""Optional actual original-fp8ws/new-pipeline gate on the complete4ae source.

All execution is official-remote-only. Original fp8ws registration must already
have passed. Refusal retains that complete parent's original fp8ws, not CUTLASS.
Private contents, outputs and graphs never persist across requests.
"""
import os
import statistics

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
ENABLED = os.environ.get("FP8WS_PIPELINE", "1") != "0"
_ADMITTED = {}
_GUARD8, _GUARD16 = 0xA5, 0x5A5A


def qualify(backend, weight, packed, scales, activation_factory):
    if (not ENABLED or torch.cuda.is_current_stream_capturing()
            or not getattr(backend, "pipeline_available", lambda d: False)(weight.device)):
        return False
    device = weight.device
    try:
        patterns = activation_factory(device)
        m, k = patterns[0].shape
        n = weight.shape[1]
        a_store = torch.empty(m * k + 32, dtype=torch.uint8, device=device)
        a = a_store[16:-16].view(torch.float8_e4m3fn).view(m, k)
        out_store = torch.empty(m * n + 32, dtype=torch.bfloat16, device=device)
        out = out_store[16:-16].view(m, n)
        sa, sb = [x.detach().clone().float().reshape(1) for x in scales[0]]
        packed_before = packed.clone()

        def setup(value, pair):
            a_store.fill_(_GUARD8)
            a.view(torch.uint8).copy_(value.view(torch.uint8))
            out_store.view(torch.int16).fill_(_GUARD16)
            out.view(torch.int16).fill_(0x7FC0)
            sa.copy_(pair[0])
            sb.copy_(pair[1])

        def check(value, pair):
            if (not bool(torch.all(a_store[:16] == _GUARD8))
                    or not bool(torch.all(a_store[-16:] == _GUARD8))
                    or not bool(torch.all(out_store[:16].view(torch.int16) == _GUARD16))
                    or not bool(torch.all(out_store[-16:].view(torch.int16) == _GUARD16))):
                raise ValueError("fp8ws pipeline private guard changed")
            if (not torch.equal(a.view(torch.uint8), value.view(torch.uint8))
                    or not torch.equal(sa.view(torch.int32), pair[0].view(torch.int32))
                    or not torch.equal(sb.view(torch.int32), pair[1].view(torch.int32))):
                raise ValueError("fp8ws pipeline changed private input/scales")

        # Direct comparison to the admitted original backend on actual weights.
        for value in patterns:
            for pair in scales:
                setup(value, pair)
                ref = backend.gemm(a, packed, sa, sb)
                got = backend.gemm_pipeline(a, packed, sa, sb, out=out)
                torch.cuda.synchronize(device)
                check(value, pair)
                if not torch.equal(ref.view(torch.int16), got.view(torch.int16)):
                    raise ValueError("fp8ws pipeline original/new output words differ")

        setup(patterns[0], scales[0])
        graphs, outputs, streams = [], [], []
        for which in (0, 1):
            call = (lambda: backend.gemm(a, packed, sa, sb)) if which == 0 else (
                lambda: backend.gemm_pipeline(a, packed, sa, sb, out=out))
            for _ in range(2):
                call()
            torch.cuda.synchronize(device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream, pool=torch.cuda.graph_pool_handle()):
                output = call()
            torch.cuda.synchronize(device)
            graphs.append(graph)
            outputs.append(output)
            streams.append(stream)
        for value in reversed(patterns):
            for pair in reversed(scales):
                setup(value, pair)
                for graph in graphs:
                    graph.replay()
                torch.cuda.synchronize(device)
                check(value, pair)
                if not torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16)):
                    raise ValueError("fp8ws pipeline changed-content graph words differ")

        # Full projection graph latency, five alternating samples per case.
        # Reset/copy and 64MiB pressure are outside each event interval.
        pressure = torch.empty(64 << 20, device=device, dtype=torch.uint8)
        ratios, old_times, new_times = [], [], []
        for value in patterns[:2]:
            for cold in (False, True):
                samples = [[], []]
                for trial in range(5):
                    for which in ((0, 1) if trial % 2 == 0 else (1, 0)):
                        setup(value, scales[0])
                        if cold:
                            pressure.fill_(trial)
                        torch.cuda.synchronize(device)
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        graphs[which].replay()
                        end.record()
                        end.synchronize()
                        elapsed = start.elapsed_time(end)
                        if not elapsed > 0:
                            raise ValueError("invalid fp8ws pipeline private timing")
                        samples[which].append(elapsed)
                        check(value, scales[0])
                old, new = [statistics.median(s) for s in samples]
                ratios.append(new / old)
                old_times.append(old)
                new_times.append(new)
        if not torch.equal(packed, packed_before):
            raise ValueError("fp8ws pipeline modified immutable pack")
        if (statistics.median(ratios) >= 0.995
                or sum(new_times) / sum(old_times) >= 0.995 or max(ratios) >= 1.01):
            raise ValueError(f"fp8ws pipeline did not improve original projection: {ratios}")
        _ADMITTED[weight.data_ptr()] = (weight, packed, backend, tuple(ratios))
        logger.info("fp8ws pipeline admitted over original: ratios=%s", tuple(ratios))
        return True
    except Exception as exc:
        logger.warning("fp8ws pipeline kept original fp8ws: %r", exc)
        return False


def maybe_gemm(a, weight, packed, sa, sb):
    entry = _ADMITTED.get(weight.data_ptr())
    if entry is None or entry[0] is not weight or entry[1] is not packed:
        return None
    return entry[2].gemm_pipeline(a, packed, sa, sb)

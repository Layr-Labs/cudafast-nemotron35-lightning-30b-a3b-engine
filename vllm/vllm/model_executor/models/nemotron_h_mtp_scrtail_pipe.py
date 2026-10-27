# SPDX-License-Identifier: Apache-2.0
"""Optional scrtail warp-role qualification, executed only by the official runner.

No request values or private graphs survive this load-time qualification. The
original scrtail admission must already have passed; a refusal keeps that path.
"""
import statistics

import torch

from vllm.logger import init_logger

from .nemotron_h_mtp_kernels import GATE_PROD_ONE, depth_gate_init

logger = init_logger(__name__)
_GUARD = 0x5A5AA55A
_NAN = 0x7FC01234


def qualify_rest_pipeline(screen, k: int, device, backend) -> bool:
    if (not screen.tail or screen._next_pf
            or torch.cuda.is_current_stream_capturing()
            or torch.cuda.get_device_capability(device) != (12, 1)
            or not getattr(backend, "rest_pipeline_available", lambda: False)()):
        return False
    gate = depth_gate_init(device)
    if gate is None:
        return False
    names = ("_h", "_out", "_hp", "_hm", "_hr", "_pre_cand", "mid_scores",
             "scores", "_coarse_cand", "refine_scores", "_cand", "cand_logits")
    originals, guarded = {}, {}
    old_flags = screen.tail, screen._tail_debug, screen._tail_rest_pipeline
    saved_gate = gate.clone()
    try:
        # Rebind only mutable screen scratch to guarded private owners. Actual
        # W4A16 weights/scales stay the original read-only kernel arguments.
        for name in names:
            original = getattr(screen, name)
            if (not isinstance(original, torch.Tensor) or not original.is_cuda
                    or not original.is_contiguous()
                    or original.dtype not in (torch.float32, torch.int32, torch.int64)):
                return False
            originals[name] = original
            backing = torch.empty(original.numel() + 32, dtype=original.dtype, device=device)
            body = backing[16:16 + original.numel()].view(original.shape)
            guarded[name] = backing, body
            setattr(screen, name, body)
        h_store = torch.empty(k + 32, dtype=torch.float32, device=device)
        h = h_store[16:16 + k]
        out_store = torch.empty(33, dtype=torch.int32, device=device)
        out = out_store[16:17].view(())
        gen = torch.Generator(device=device)
        gen.manual_seed(20261028)
        patterns = []
        for trial in range(12):
            value = torch.randn(k, generator=gen, device=device) * (0.25 * 2 ** (trial % 6))
            if trial == 6:
                value[::3] = 0.0
                value[1::7] = -0.0
            elif trial == 7:
                value = torch.round(value * 4) / 4
            elif trial == 8:
                value *= 1e-20
            elif trial == 9:
                value.zero_()
            elif trial == 10:
                value[0], value[3] = float("inf"), -float("inf")
            elif trial == 11:
                value[0] = float("nan")
            patterns.append(value.float().contiguous())

        def setup(value, stop):
            for backing, body in guarded.values():
                backing.view(torch.int32).fill_(_GUARD)
                if body.dtype == torch.float32:
                    body.view(torch.int32).fill_(_NAN)
                else:
                    body.fill_(-7)
            h_store.view(torch.int32).fill_(_GUARD)
            h.copy_(value)
            out_store.fill_(_GUARD)
            out.fill_(-1)
            gate.copy_(torch.tensor([stop, 1, 1, GATE_PROD_ONE],
                                    dtype=torch.int32, device=device))

        def check_guards(value):
            for backing in [v[0] for v in guarded.values()] + [h_store, out_store]:
                if (not bool(torch.all(backing[:16].view(torch.int32) == _GUARD))
                        or not bool(torch.all(backing[-16:].view(torch.int32) == _GUARD))):
                    raise ValueError("scrtail rest-pipeline private guard changed")
            if not torch.equal(h.view(torch.int32), value.view(torch.int32)):
                raise ValueError("scrtail rest-pipeline modified private hidden input")

        def observation():
            return [out.reshape(1).view(torch.uint8).clone(), gate.view(torch.uint8).clone()] + [
                body.reshape(-1).view(torch.uint8).clone() for _, body in guarded.values()]

        # Both production writes and the complete intermediate-buffer debug
        # writes compare. NaNs compare as words, never floating-point equality.
        screen.tail = True
        for debug in (True, False):
            screen._tail_debug = debug
            for value in patterns:
                for stop in (0, 1):
                    observed = []
                    for pipeline in (False, True):
                        setup(value, stop)
                        screen._tail_rest_pipeline = pipeline
                        screen._greedy_row(h, gated=True, out=out)
                        torch.cuda.synchronize(device)
                        check_guards(value)
                        observed.append(observation())
                    if not all(torch.equal(a, b) for a, b in zip(*observed)):
                        raise ValueError("scrtail rest-pipeline full private state differs")

        # Capture the complete actual screen call on fixed private pointers.
        # Changed scores/row choices/gates are regenerated on every replay.
        screen._tail_debug = False
        graphs, streams = {}, []
        for pipeline in (False, True):
            screen._tail_rest_pipeline = pipeline
            for _ in range(2):
                setup(patterns[0], 0)
                screen._greedy_row(h, gated=True, out=out)
            setup(patterns[0], 0)
            torch.cuda.synchronize(device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream, pool=torch.cuda.graph_pool_handle()):
                screen._greedy_row(h, gated=True, out=out)
            torch.cuda.synchronize(device)
            graphs[pipeline] = graph
            streams.append(stream)
        for value in reversed(patterns):
            for stop in (1, 0):
                observed = []
                for pipeline in (False, True):
                    setup(value, stop)
                    graphs[pipeline].replay()
                    torch.cuda.synchronize(device)
                    check_guards(value)
                    observed.append(observation())
                if not all(torch.equal(a, b) for a, b in zip(*observed)):
                    raise ValueError("scrtail rest-pipeline changed-pointer-content graph differs")

        # CUDA events surround complete production-mode screen graph calls,
        # excluding resets and 64MiB pressure. Five alternating measurements
        # include stopped and active gates, not just isolated rest-dot time.
        pressure = torch.empty(64 << 20, dtype=torch.uint8, device=device)
        ratios, old_medians, new_medians = [], [], []
        for value in (patterns[0], patterns[6], patterns[7], patterns[8], patterns[9]):
            for stop in (0, 1):
                for cold in (False, True):
                    samples = [[], []]
                    for trial in range(5):
                        for which in ((0, 1) if trial % 2 == 0 else (1, 0)):
                            setup(value, stop)
                            if cold:
                                pressure.fill_(trial + stop)
                            torch.cuda.synchronize(device)
                            begin = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            begin.record()
                            graphs[bool(which)].replay()
                            end.record()
                            end.synchronize()
                            elapsed = begin.elapsed_time(end)
                            if not elapsed > 0:
                                raise ValueError("invalid scrtail rest-pipeline private timing")
                            samples[which].append(elapsed)
                            check_guards(value)
                    old, new = [statistics.median(v) for v in samples]
                    old_medians.append(old)
                    new_medians.append(new)
                    ratios.append(new / old)
        if (not ratios or statistics.median(ratios) >= 0.995
                or sum(new_medians) / sum(old_medians) >= 0.995 or max(ratios) >= 1.01):
            raise ValueError("scrtail rest-pipeline did not improve complete screen")
        screen._tail_rest_pipeline_ratios = tuple(ratios)
        logger.info("MTP scrtail rest-pipeline admitted: ratios=%s", tuple(ratios))
        return True
    except Exception as exc:
        logger.warning("MTP scrtail rest-pipeline retained original tail: %r", exc)
        return False
    finally:
        torch.cuda.synchronize(device)
        for name, original in originals.items():
            setattr(screen, name, original)
        screen.tail, screen._tail_debug, screen._tail_rest_pipeline = old_flags
        gate.copy_(saved_gate)
        torch.cuda.synchronize(device)

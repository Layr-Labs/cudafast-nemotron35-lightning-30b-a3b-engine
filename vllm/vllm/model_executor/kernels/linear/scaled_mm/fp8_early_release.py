# SPDX-License-Identifier: Apache-2.0
"""Optional eight-warp FP8 stage release at the last shared-memory reader.

Original fp8ws pack, MMA order and epilogue remain the reference and fallback.
All qualification executes only in the official remote resident. Decisions
retain immutable model weights/packs only, never private inputs or graphs.
"""

import os
import statistics

import torch

from vllm.logger import init_logger

_logger = init_logger(__name__)
_ENABLED = os.environ.get("FP8WS_EARLY_RELEASE", "1") != "0"
_ADMITTED = {}
_GUARD = 128
_CANARY = 11.25
_CALLS = 8


def _same_words(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _input_owner(device, k):
    owner = torch.full((496 * k + 2 * _GUARD,), 90,
                       dtype=torch.uint8, device=device)
    view = owner[_GUARD:-_GUARD].view(torch.float8_e4m3fn).view(496, k)
    return owner, view


def _output_owner(device, n):
    owner = torch.full((496 * n + 2 * _GUARD,), _CANARY,
                       dtype=torch.bfloat16, device=device)
    return owner, owner[_GUARD:-_GUARD].view(496, n)


def _scale_owner(source):
    owner = torch.full((1 + 2 * _GUARD,), _CANARY,
                       dtype=torch.float32, device=source.device)
    view = owner[_GUARD:_GUARD + 1]
    view.copy_(source.reshape(1))
    return owner, view


@torch.no_grad()
def register(weight, packed, scales, activations):
    """Called only after the original pack inversion/CUTLASS word admission."""
    if not _ENABLED or torch.cuda.is_current_stream_capturing():
        return False
    key = (weight.data_ptr(), packed.data_ptr())
    cached = _ADMITTED.get(key)
    if cached is not None and cached[0] is weight and cached[1] is packed:
        return cached[2]
    _ADMITTED[key] = (weight, packed, False)
    try:
        import fp8early
        import fp8ws

        if not fp8early.prepare(weight.device):
            _logger.info("FP8 early release unavailable; original fp8ws retained")
            return False
        k, n = weight.shape
        if ((n, k) not in fp8early.SHAPES or weight.stride() != (1, k)
                or not scales or len(activations) != 5):
            return False
        original = fp8ws._load()
        if original is None:
            return False
        ao, a = _input_owner(weight.device, k)
        sao, sa = _scale_owner(scales[0][0])
        sbo, sb = _scale_owner(scales[0][1])
        old = [_output_owner(weight.device, n) for _ in range(_CALLS)]
        new = [_output_owner(weight.device, n) for _ in range(_CALLS)]
        packed_before = packed.clone()
        weight_before = weight.t().view(torch.uint8).clone()

        def reset_outputs():
            for owner, _ in (*old, *new):
                owner.fill_(_CANARY)

        def words_equal():
            return all(_same_words(x[0], y[0]) for x, y in zip(old, new))

        def output_guards():
            return all(bool(torch.all(owner[:_GUARD] == _CANARY))
                       and bool(torch.all(owner[-_GUARD:] == _CANARY))
                       for owner, _ in (*old, *new))

        def input_snapshot():
            return ao.clone(), sao.clone(), sbo.clone()

        def readonly(before):
            return all(_same_words(x, y) for x, y in zip((ao, sao, sbo), before))

        def call(early, dest):
            if early:
                fp8early.run_out(a, packed, sa, sb, dest)
            else:
                rc = original.fp8ws_run(
                    a.data_ptr(), packed.data_ptr(), sa.data_ptr(), sb.data_ptr(),
                    dest.data_ptr(), 496, n, k,
                    torch.cuda.current_stream(a.device).cuda_stream,
                )
                if rc:
                    raise RuntimeError("original fp8ws probe launch failed: " + str(rc))

        def prepare(case, scale_pair):
            ao.fill_(90)
            a.copy_(activations[case])
            sao.fill_(_CANARY)
            sbo.fill_(_CANARY)
            sa.copy_(scale_pair[0])
            sb.copy_(scale_pair[1])
            reset_outputs()

        # Full real-shape words, padding and readonly inputs on finite
        # magnitudes and every FP8 bit pattern, with model/synthetic scales.
        for case in range(len(activations)):
            for scale_pair in scales:
                prepare(case, scale_pair)
                before = input_snapshot()
                call(False, old[0][1])
                call(True, new[0][1])
                if not words_equal() or not output_guards() or not readonly(before):
                    _logger.info("FP8 early release word/owner refusal: shape=%s case=%s",
                                 (n, k), case)
                    return False

        prepare(0, scales[0])
        old_graph, new_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(old_graph):
            for _, dest in old:
                call(False, dest)
        with torch.cuda.graph(new_graph):
            for _, dest in new:
                call(True, dest)
        # Same pointers, changed input/scalar contents, both orderings and
        # eight distinct full outputs per graph test each launch destination.
        for case in (4, 3, 2, 1, 0):
            for scale_pair in reversed(scales):
                prepare(case, scale_pair)
                before = input_snapshot()
                if case % 2:
                    new_graph.replay()
                    old_graph.replay()
                else:
                    old_graph.replay()
                    new_graph.replay()
                if not words_equal() or not output_guards() or not readonly(before):
                    _logger.info("FP8 early release changed-graph refusal: shape=%s case=%s",
                                 (n, k), case)
                    return False

        pressure = torch.empty(64 * 1024 * 1024, dtype=torch.uint8,
                               device=weight.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        medians = []
        for case in (0, 2):
            prepare(case, scales[0])
            before = input_snapshot()
            for cold in (False, True):
                times = [[], []]
                for sample in range(5):
                    for which in ((0, 1) if sample % 2 == 0 else (1, 0)):
                        for owner, _ in (old if which == 0 else new):
                            owner.fill_(_CANARY)
                        if cold:
                            pressure.fill_(sample)
                        start.record()
                        (old_graph if which == 0 else new_graph).replay()
                        end.record()
                        end.synchronize()
                        elapsed = start.elapsed_time(end)
                        if elapsed <= 0:
                            return False
                        times[which].append(elapsed)
                medians.append(tuple(statistics.median(x) for x in times))
                if not words_equal() or not output_guards() or not readonly(before):
                    return False
        if (not _same_words(packed, packed_before)
                or not _same_words(weight.t().view(torch.uint8), weight_before)):
            return False
        ratios = tuple(new_ms / old_ms for old_ms, new_ms in medians)
        total = sum(x[1] for x in medians) / sum(x[0] for x in medians)
        if max(ratios) >= 1.01 or statistics.median(ratios) >= 0.99 or total >= 0.99:
            _logger.info("FP8 early release private speed refusal: shape=%s ratios=%s",
                         (n, k), ratios)
            return False
        _ADMITTED[key] = (weight, packed, True)
        _logger.info("FP8 early release admitted: shape=%s ratios=%s total=%s",
                     (n, k), ratios, total)
        return True
    except Exception as exc:
        _logger.warning("FP8 early release optional qualification refused: %s",
                        type(exc).__name__)
        return False


def maybe_gemm(a, weight, packed, sa, sb):
    key = (weight.data_ptr(), packed.data_ptr())
    entry = _ADMITTED.get(key)
    if (not _ENABLED or entry is None or entry[0] is not weight
            or entry[1] is not packed or not entry[2]):
        return None
    import fp8early

    n, k = weight.shape[1], weight.shape[0]
    if (a.dtype != torch.float8_e4m3fn or tuple(a.shape) != (496, k)
            or not a.is_contiguous() or a.data_ptr() % 16
            or not (a.device == weight.device == packed.device == sa.device == sb.device)
            or sa.dtype != torch.float32 or sb.dtype != torch.float32
            or sa.numel() != 1 or sb.numel() != 1
            or not sa.is_contiguous() or not sb.is_contiguous()):
        return None
    out = torch.empty((496, n), dtype=torch.bfloat16, device=a.device)
    fp8early.run_out(a, packed, sa, sb, out)
    return out

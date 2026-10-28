# SPDX-License-Identifier: Apache-2.0
"""Private remote-only qualification for the proposed conv-to-SSM sink.

Inputs and state are private synthetic buffers; loaded layer parameters are
read only. Admission requires paired direct checks, changed-input CUDA graph
replays and a measured improvement of the full captured producer/consumer.
"""
import statistics
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _same(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _private_view(shape, strides, dtype, device):
    # Include before/after canaries and slot padding in the comparison.
    end = sum((n - 1) * s for n, s in zip(shape, strides)) + 1
    backing = torch.full((end + 32,), 0.03125, dtype=dtype, device=device)
    return backing, torch.as_strided(backing, shape, strides, 16)


def _byte_span(tensor):
    start = tensor.data_ptr()
    extent = 1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))
    return start, start + extent * tensor.element_size()


@torch.no_grad()
def check_real_layers(state, conv_state, weight, bias, A, D, dt_bias,
                      dtype, ring_len, is_blackwell, lazy, pipeline, pdl, row_stride):
    """Compare paired direct calls on private buffers; never enables the sink.

    Returns False on an unsupported layout, exception or mismatch. This is
    deliberately only one component of qualification, not an admission test.
    """
    from .causal_conv1d import causal_conv1d_update
    from . import mamba_ssm_spec_replay as rp

    try:
        dev = state.device
        if (not state.is_cuda or state.dtype != torch.float32
                or state.ndim != 4 or tuple(state.shape[1:]) != (64, 64, 128)
                or state.stride()[1:] != (8192, 128, 1)
                or conv_state.ndim != 3 or conv_state.shape[1] != 6144
                or conv_state.dtype != dtype or dtype not in (torch.float16, torch.bfloat16)
                or tuple(weight.shape) != (6144, 4)
                or row_stride < 6208
                or any(t.device != dev for t in (conv_state, weight, A, D, dt_bias))
                or (bias is not None and bias.device != dev)
                or torch.cuda.is_current_stream_capturing()
                or (lazy and (rp.prefill_tag_view(state) is None or rp.LAZY_PERIOD <= 0))
                or ring_len * (4096 + 1024 + 64) > 2 * 524288):
            return False

        slot_count = 8
        state_shape = (slot_count, 64, 64, 128)
        state_stride = (state.stride(0), 8192, 128, 1)
        conv_shape = (slot_count, *conv_state.shape[1:])
        conv_stride = conv_state.stride()
        for nrows in (5, 6):
            state_pairs = [_private_view(state_shape, state_stride, torch.float32, dev)
                           for _ in range(2)]
            conv_pairs = [_private_view(conv_shape, conv_stride, dtype, dev)
                          for _ in range(2)]
            states = [p[1] for p in state_pairs]
            convs = [p[1] for p in conv_pairs]
            if lazy:
                for item in states:
                    tags = rp.prefill_tag_view(item)
                    if tags is None:
                        return False
                    tags[1] = 37
                    tags[5] = 37
            table = torch.tensor([[1, 2, 3, 4, 5, 6][:nrows]],
                                 dtype=torch.int32, device=dev)
            accepted = torch.ones((1,), dtype=torch.int32, device=dev)
            cu = torch.tensor([0, nrows], dtype=torch.int32, device=dev)
            seq = torch.empty((1,), dtype=torch.int32, device=dev)
            valid = torch.empty((1,), dtype=torch.int32, device=dev)
            outputs = [torch.empty((nrows + 2, 64, 64), dtype=dtype, device=dev)
                       for _ in range(2)]
            positions = (37, 38, 52, 53, 54, 69, 70, 71)
            for case, position in enumerate(positions):
                # Changed inputs, short/full prefixes, wrap and zero acceptance.
                indices = torch.arange(nrows * 6144, device=dev, dtype=torch.int32)
                values = (((indices + case * 37) % 257).float() / 128 - 1).to(dtype)
                values = values.view(nrows, 6144)
                if case == 5:
                    tiny = 2.0 ** (-133 if dtype == torch.bfloat16 else -24)
                    edges = torch.tensor([0., -0., tiny, -tiny, 1., -1.,
                                          float("inf"), float("-inf"), float("nan")],
                                         dtype=dtype, device=dev)
                    for offset in (0, 4096, 5120):
                        values[0, offset:offset + edges.numel()] = edges
                projections = [_private_view((nrows, row_stride), (row_stride, 1),
                                             dtype, dev)[1] for _ in range(2)]
                inputs = [item[:, row_stride - 6208:row_stride - 64]
                          for item in projections]
                raw_dts = [item[:, row_stride - 64:] for item in projections]
                dts = [item[:, :, None].expand(-1, -1, 64) for item in raw_dts]
                for item, raw_dt in zip(inputs, raw_dts):
                    item.copy_(values)
                    raw_dt.fill_((case - 3) / 4)
                prefix = nrows if case == 0 else 1 + case % nrows
                valid.fill_(prefix)
                accepted.fill_(1 if case == 0 else case % (nrows + 1))
                seq.fill_(position + nrows)
                # Last two cases deliberately disable the producer by alias or
                # null slot; both sides must then execute the ordinary path.
                if case == 4:
                    table.copy_(torch.tensor([[5, 3, 1, 2, 4, 6][:nrows]],
                                             dtype=torch.int32, device=dev))
                elif case == len(positions) - 2:
                    table[0, 1] = 1
                elif case == len(positions) - 1:
                    table.zero_()
                else:
                    table[0, 0] = 1
                    table[0, 1] = 2
                    table[0, 2:] = torch.arange(3, nrows + 1,
                                                dtype=torch.int32, device=dev)
                for output in outputs:
                    output.fill_(0.0625)
                for which in range(2):
                    item = states[which]
                    sink = None
                    if which:
                        sink = {
                            "state": item, "ring": item.view(dtype), "seq_lens": seq,
                            "verify_rows": valid, "slot_elems": 524288,
                            "ring_len": ring_len, "max_rows": nrows, "lazy": lazy,
                            "lazy_period": rp.LAZY_PERIOD, "x_features": 4096,
                            "b_features": 1024,
                        }
                    result = causal_conv1d_update(
                        inputs[which], convs[which], weight, bias, "silu",
                        conv_state_indices=table, num_accepted_tokens=accepted,
                        query_start_loc=cu, max_query_len=nrows,
                        _private_ring_sink=sink,
                    )
                    x = result[:, :4096].view(nrows, 64, 64)
                    B = result[:, 4096:5120].view(nrows, 8, 128)
                    C = result[:, 5120:6144].view(nrows, 8, 128)
                    rp.selective_state_update_spec_replay(
                        item, x, dts[which], A, B, C, D, dt_bias, dt_softplus=True,
                        state_batch_indices=table, out=outputs[which][1:-1],
                        num_accepted_tokens=accepted, cu_seqlens=cu, seq_lens=seq,
                        ring_len=ring_len, is_blackwell=is_blackwell, lazy=lazy,
                        _pipeline=pipeline, _private_verify_rows=valid,
                        _private_ring_producer=bool(which),
                        _private_launch_pdl=pdl,
                    )
                torch.cuda.synchronize(dev)
                if not (_same(outputs[0], outputs[1])
                        and _same(state_pairs[0][0], state_pairs[1][0])
                        and _same(conv_pairs[0][0], conv_pairs[1][0])
                        and _same(raw_dts[0], raw_dts[1])
                        and _same(inputs[0][:, 5120:], inputs[1][:, 5120:])):
                    return False
                if case == 0 and _same(inputs[0][:, :5120], inputs[1][:, :5120]):
                    # No producer activation is not a successful experiment.
                    return False
                if prefix < nrows and not _same(inputs[0][prefix:, :5120],
                                                inputs[1][prefix:, :5120]):
                    return False
                if case >= len(positions) - 2 and not _same(inputs[0][:, :5120],
                                                            inputs[1][:, :5120]):
                    return False
        # Passing this direct check alone must never activate the path.
        return True
    except Exception as exc:
        logger.warning("Private conv-ring direct check failed: %r", exc)
        return False


@torch.no_grad()
def check_changed_graph_replay(state, conv_state, weight, bias, A, D, dt_bias,
                               dtype, ring_len, is_blackwell, lazy, pipeline, pdl,
                               row_stride):
    """Check changed-input replays and time the complete paired captured chain."""
    from .causal_conv1d import causal_conv1d_update
    from . import mamba_ssm_spec_replay as rp

    try:
        dev = state.device
        nrows = 6
        state_shape = (8, 64, 64, 128)
        state_stride = (state.stride(0), 8192, 128, 1)
        state_pairs = [_private_view(state_shape, state_stride, torch.float32, dev)
                       for _ in range(2)]
        conv_pairs = [_private_view((8, *conv_state.shape[1:]), conv_state.stride(),
                                    dtype, dev) for _ in range(2)]
        states = [pair[1] for pair in state_pairs]
        convs = [pair[1] for pair in conv_pairs]
        if lazy:
            for item in states:
                tag = rp.prefill_tag_view(item)
                if tag is None:
                    return False
                tag[1] = 37
                tag[5] = 37
        table = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.int32, device=dev)
        accepted = torch.ones((1,), dtype=torch.int32, device=dev)
        cu = torch.tensor([0, nrows], dtype=torch.int32, device=dev)
        seq = torch.full((1,), 43, dtype=torch.int32, device=dev)
        valid = torch.full((1,), nrows, dtype=torch.int32, device=dev)
        projections = [_private_view((nrows, row_stride), (row_stride, 1),
                                     dtype, dev)[1] for _ in range(2)]
        inputs = [item[:, row_stride - 6208:row_stride - 64] for item in projections]
        raw_dts = [item[:, row_stride - 64:] for item in projections]
        dts = [item[:, :, None].expand(-1, -1, 64) for item in raw_dts]
        outputs = [torch.full((nrows + 2, 64, 64), .0625, dtype=dtype, device=dev)
                   for _ in range(2)]
        sinks = [{
            "state": item, "ring": item.view(dtype), "seq_lens": seq,
            "verify_rows": valid, "slot_elems": 524288, "ring_len": ring_len,
            "max_rows": nrows, "lazy": lazy, "lazy_period": rp.LAZY_PERIOD,
            "x_features": 4096, "b_features": 1024,
        } for item in states]

        def call(which):
            result = causal_conv1d_update(
                inputs[which], convs[which], weight, bias, "silu",
                conv_state_indices=table, num_accepted_tokens=accepted,
                query_start_loc=cu, max_query_len=nrows,
                _private_ring_sink=sinks[which] if which else None,
            )
            rp.selective_state_update_spec_replay(
                states[which], result[:, :4096].view(nrows, 64, 64), dts[which], A,
                result[:, 4096:5120].view(nrows, 8, 128),
                result[:, 5120:6144].view(nrows, 8, 128), D, dt_bias,
                dt_softplus=True, state_batch_indices=table, out=outputs[which][1:-1],
                num_accepted_tokens=accepted, cu_seqlens=cu, seq_lens=seq,
                ring_len=ring_len, is_blackwell=is_blackwell, lazy=lazy,
                _pipeline=pipeline, _private_verify_rows=valid,
                _private_ring_producer=bool(which), _private_launch_pdl=pdl,
            )

        stream = torch.cuda.Stream(device=dev)
        stream.wait_stream(torch.cuda.current_stream(dev))
        graphs = []
        for which in range(2):
            with torch.cuda.stream(stream):
                call(which)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream, pool=torch.cuda.graph_pool_handle()):
                call(which)
            graphs.append(graph)
        torch.cuda.current_stream(dev).wait_stream(stream)
        for case, pos in enumerate((37, 38, 53, 68, 69, 85, 101, 102)):
            indices = torch.arange(nrows * 6144, dtype=torch.int32, device=dev)
            values = (((indices + case * 83) % 257).float() / 128 - 1).to(dtype)
            values = values.view(nrows, 6144)
            for item, raw_dt in zip(inputs, raw_dts):
                item.copy_(values)
                raw_dt.fill_((case - 3) / 4)
            valid.fill_(1 + case % nrows)
            accepted.fill_(1 + case % nrows)
            seq.fill_(pos + nrows)
            if case == 4:
                table.copy_(torch.tensor([[5, 3, 1, 2, 4, 6]],
                                         dtype=torch.int32, device=dev))
            elif case == 6:
                table[0, 1] = 1
            elif case == 7:
                table.zero_()
            else:
                table.copy_(torch.tensor([[1, 2, 3, 4, 5, 6]],
                                         dtype=torch.int32, device=dev))
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize(dev)
            if not (_same(outputs[0], outputs[1])
                    and _same(state_pairs[0][0], state_pairs[1][0])
                    and _same(conv_pairs[0][0], conv_pairs[1][0])
                    and _same(raw_dts[0], raw_dts[1])
                    and _same(inputs[0][:, 5120:], inputs[1][:, 5120:])):
                return False
            if case == 0 and _same(inputs[0][:, :5120], inputs[1][:, :5120]):
                return False
            prefix = 1 + case % nrows
            if prefix < nrows and not _same(inputs[0][prefix:, :5120],
                                            inputs[1][prefix:, :5120]):
                return False
            if case >= 6 and not _same(inputs[0][:, :5120], inputs[1][:, :5120]):
                return False
        # Restore private state after the alias/null cases. Capture addresses
        # stay fixed. Each replay receives the same changed source input, with
        # the reset included in both timings.
        for backing, item in state_pairs:
            backing.fill_(0.03125)
            if lazy:
                rp.prefill_tag_view(item)[1] = 37
        for backing, _ in conv_pairs:
            backing.fill_(0.03125)
        table.copy_(torch.tensor([[1, 2, 3, 4, 5, 6]],
                                 dtype=torch.int32, device=dev))
        accepted.fill_(1)
        ratios = []
        for pos, prefix in ((37, 1), (53, 3), (69, 6)):
            seq.fill_(pos + nrows)
            valid.fill_(prefix)
            samples = [[], []]
            for round_index in range(5):
                order = (0, 1) if round_index % 2 == 0 else (1, 0)
                for which in order:
                    inputs[which].copy_(values)
                    graphs[which].replay()
                    torch.cuda.synchronize(dev)
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(12):
                        inputs[which].copy_(values)
                        graphs[which].replay()
                    end.record()
                    end.synchronize()
                    samples[which].append(start.elapsed_time(end) / 12)
            base, candidate = (statistics.median(item) for item in samples)
            if base <= 0 or candidate <= 0:
                return False
            ratios.append(candidate / base)
        admitted = statistics.median(ratios) < 0.98 and max(ratios) < 1.03
        logger.info("Private conv-ring full-chain graph ratios=%s admitted=%s",
                    ratios, admitted)
        return admitted
    except Exception as exc:
        logger.warning("Private conv-ring graph check failed: %r", exc)
        return False


@torch.no_grad()
def prepare(state, conv_state, weight, bias, A, D, dt_bias, dtype, ring_len,
            is_blackwell, lazy, pipeline, row_stride):
    """Conservative per-layer admission after remote boot qualification."""
    if os.environ.get("NEMO_CONV_RING_FUSION", "1") == "0":
        return False
    try:
        from . import mamba_ssm_spec_replay as rp

        if (not lazy or pipeline or not state.is_cuda
                or torch.cuda.is_current_stream_capturing()
                or torch.cuda.get_device_capability(state.device) != (12, 1)
                or A.shape != (64, 64, 128) or A.dtype != torch.float32
                or A.stride()[1:] != (0, 0)
                or D.shape != (64, 64) or dt_bias.shape != (64, 64)
                or dt_bias.stride(1) != 0 or state.shape[0] < 8
                or conv_state.shape[0] < 8 or rp.prefill_tag_view(state) is None
                or ring_len < rp.LAZY_PERIOD + 6):
            return False
        if (any(stride < 0 for stride in state.stride())
                or any(stride < 0 for stride in conv_state.stride())):
            return False
        state_lo, state_hi = _byte_span(state)
        conv_lo, conv_hi = _byte_span(conv_state)
        if state_lo < conv_hi and conv_lo < state_hi:
            return False
        # The producer uses platform PDL; test the consumer with and without
        # its wait when supported, because capture configuration can vary.
        modes = (False, True) if rp._pdl_enabled() else (False,)
        for pdl in modes:
            args = (state, conv_state, weight, bias, A, D, dt_bias, dtype,
                    ring_len, is_blackwell, lazy, pipeline, pdl, row_stride)
            if not check_real_layers(*args) or not check_changed_graph_replay(*args):
                return False
        return True
    except Exception as exc:
        logger.warning("Private conv-ring qualification retained original path: %r", exc)
        return False

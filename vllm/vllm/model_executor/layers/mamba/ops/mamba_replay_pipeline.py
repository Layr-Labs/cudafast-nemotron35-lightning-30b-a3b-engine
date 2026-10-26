"""Remote-boot qualification of a raw-input replay software pipeline.

No additional cache is stored. Full guarded backing, both checkpoint slots,
raw history, output and optional pre-verify state must remain byte identical.
"""
import os
import statistics

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
_TIMING = {}
_SEEN = set()


def _same(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _layout(state, dtype, ring_len):
    if (not state.is_cuda or state.dtype != torch.float32 or state.ndim != 4
            or dtype not in (torch.bfloat16, torch.float16)
            or tuple(state.shape[1:]) != (64, 64, 128) or state.shape[0] < 6
            or state.stride()[1:] != (8192, 128, 1)
            or state.stride(0) < 524289 or state.data_ptr() % 16):
        return None
    end = state.storage_offset() + (state.shape[0] - 1) * state.stride(0) + 524289
    if end * 4 > state.untyped_storage().nbytes():
        return None
    from . import mamba_ssm_spec_replay as rp
    if (rp.LAZY_PERIOD not in (16, 20) or ring_len < rp.LAZY_PERIOD + 5
            or ring_len * (4096 + 1024 + 64) * 2 > 524288 * 4):
        return None
    return (state.device, dtype, state.stride(0), ring_len, rp.LAZY_PERIOD,
            rp._pdl_enabled(), rp._const_div_enabled())


def _timing(rp, states, x, dt, A, B, C, D, bias, table, accepted, cu, seq,
            rows, outputs, ring_len, is_blackwell):
    """Compare executable structure on generic private lengths, not prompts."""
    dev = x.device
    stream = torch.cuda.Stream(device=dev)
    stream.wait_stream(torch.cuda.current_stream())
    graphs = []
    for which in range(2):
        def call():
            rp.selective_state_update_spec_replay(
                states[which], x, dt, A, B, C, D, bias,
                dt_softplus=True, state_batch_indices=table, out=outputs[which][1:6],
                num_accepted_tokens=accepted, cu_seqlens=cu, seq_lens=seq,
                ring_len=ring_len, is_blackwell=is_blackwell, lazy=True,
                _pipeline=bool(which), _private_verify_rows=rows,
                _private_launch_pdl=rp._pdl_enabled(),
            )
        with torch.cuda.stream(stream):
            call()
            call()
        torch.cuda.synchronize(dev)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream, pool=torch.cuda.graph_pool_handle()):
            call()
        graphs.append(graph)
    torch.cuda.synchronize(dev)
    ratios, totals = [], [[], []]
    for replay, prefix in ((0, 1), (1, 5), (4, 3), (9, 5), (14, 1), (19, 5)):
        seq.fill_(37 + replay + 5)
        rows.fill_(prefix)
        samples = [[], []]
        for round_index in range(5):
            for which in ((0, 1) if round_index % 2 == 0 else (1, 0)):
                graphs[which].replay()
                torch.cuda.synchronize(dev)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):
                    graphs[which].replay()
                end.record()
                end.synchronize()
                samples[which].append(start.elapsed_time(end) / 20)
        a, b = (statistics.median(v) for v in samples)
        if min(a, b) <= 0:
            return False
        ratios.append(b / a)
        totals[0].append(a)
        totals[1].append(b)
    aggregate = sum(totals[1]) / sum(totals[0])
    admitted = (aggregate < 0.98 and statistics.median(ratios) < 0.98
                and max(ratios) < 1.03)
    logger.info("Mamba raw replay pipeline private ratios=%s aggregate=%.5f admitted=%s",
                ratios, aggregate, admitted)
    return admitted


@torch.no_grad()
def prepare(state, dtype, A, bias, D, ring_len, is_blackwell):
    """Fixed per-layer decision after binding, before compile/capture/requests."""
    if os.environ.get("NEMO_MAMBA_REPLAY_PIPELINE", "1") == "0":
        return False
    try:
        key = _layout(state, dtype, ring_len)
        if (key is None or torch.cuda.is_current_stream_capturing()
                or torch.cuda.get_device_capability(state.device) != (12, 1)
                or A.dtype != torch.float32 or A.shape != (64, 64, 128)
                or A.stride()[1:] != (0, 0) or bias.shape != (64, 64)
                or bias.stride(1) != 0 or D.shape != (64, 64)
                or any(t.device != state.device for t in (A, bias, D))):
            return False
        if _TIMING.get(key) is False:
            return False
        from . import mamba_ssm_spec_replay as rp
        stride, dev = state.stride(0), state.device
        length = 7 * stride + 16
        shape, strides = (7, 64, 64, 128), (stride, 8192, 128, 1)
        advances = (1, 1, 3, 0, 2, 5, 1, 4, 2, 0, 5, 3)
        cycles = 24 if key in _SEEN else 48
        for input_dtype in (torch.bfloat16, torch.float16):
            for pdl in ((False, True) if rp._pdl_enabled() else (False,)):
                for observe in (False, True):
                    backing = [torch.full((length,), 0.03125, device=dev,
                                          dtype=torch.float32) for _ in range(2)]
                    states = [torch.as_strided(t, shape, strides) for t in backing]
                    table = torch.tensor([[1, 2, 3, 4, 5]], device=dev, dtype=torch.int32)
                    accepted = torch.ones(1, device=dev, dtype=torch.int32)
                    cu = torch.tensor([0, 5], device=dev, dtype=torch.int32)
                    seq = torch.empty(1, device=dev, dtype=torch.int32)
                    rows = torch.empty(1, device=dev, dtype=torch.int32)
                    outputs = [torch.empty((7, 64, 64), device=dev, dtype=input_dtype)
                               for _ in range(2)]
                    replayed = [torch.empty((3, 64, 64, 128), device=dev,
                                           dtype=torch.float32) for _ in range(2)]
                    prompt, pos, previous_advance = 37, 37, 1
                    for st in states:
                        rp.prefill_tag_view(st)[1] = prompt
                    for cycle in range(cycles):
                        prefix = 5 if cycle % 6 == 0 else 1 + cycle % 5
                        projection = (((torch.arange(5 * 6208, device=dev,
                                                     dtype=torch.int32)
                                        + cycle * 19) % 257).float() / 128 - 1)
                        projection = projection.to(input_dtype).view(5, -1)
                        x = projection[:, :4096].view(5, 64, 64)
                        B = projection[:, 4096:5120].view(5, 8, 128)
                        C = projection[:, 5120:6144].view(5, 8, 128)
                        raw_dt = projection[:, 6144:]
                        regimes = torch.tensor([-40., -25., -8., -1., -0., 0.,
                                                .001, 1., 8., 20., 25., 40.],
                                               device=dev, dtype=input_dtype)
                        index = (torch.arange(320, device=dev).view(5, 64)
                                 + cycle * 3) % regimes.numel()
                        raw_dt.copy_(regimes[index])
                        dt = raw_dt[:, :, None].expand(-1, -1, 64)
                        rows.fill_(prefix)
                        seq.fill_(pos + 5)
                        accepted.fill_(previous_advance)
                        for t in outputs:
                            t.fill_(0.0625)
                        for t in replayed:
                            t.fill_(0.09375)
                        for which in range(2):
                            rp.selective_state_update_spec_replay(
                                states[which], x, dt, A, B, C, D, bias,
                                dt_softplus=True, state_batch_indices=table,
                                out=outputs[which][1:6], num_accepted_tokens=accepted,
                                cu_seqlens=cu, seq_lens=seq, ring_len=ring_len,
                                is_blackwell=is_blackwell, lazy=True,
                                replayed_out=replayed[which][1:2] if observe else None,
                                _pipeline=bool(which), _private_verify_rows=rows,
                                _private_launch_pdl=pdl,
                                _private_no_prefix=(cycle % 6 == 0),
                            )
                        torch.cuda.synchronize(dev)
                        if not (_same(outputs[0], outputs[1])
                                and _same(replayed[0], replayed[1])
                                and _same(backing[0], backing[1])):
                            return False
                        advance = min(prefix, advances[cycle % len(advances)])
                        pos += advance
                        previous_advance = advance
                    for empty in (True, False):
                        cu[1] = 0 if empty else 5
                        table.zero_()
                        accepted.zero_()
                        for which in range(2):
                            outputs[which].fill_(0.0625)
                            replayed[which].fill_(0.09375)
                            rp.selective_state_update_spec_replay(
                                states[which], x, dt, A, B, C, D, bias,
                                dt_softplus=True, state_batch_indices=table,
                                out=outputs[which][1:6], num_accepted_tokens=accepted,
                                cu_seqlens=cu, seq_lens=seq, ring_len=ring_len,
                                is_blackwell=is_blackwell, lazy=True,
                                replayed_out=replayed[which][1:2] if observe else None,
                                _pipeline=bool(which), _private_verify_rows=rows,
                                _private_launch_pdl=pdl,
                                _private_no_prefix=(cycle % 6 == 0),
                            )
                        torch.cuda.synchronize(dev)
                        if not (_same(outputs[0], outputs[1])
                                and _same(replayed[0], replayed[1])
                                and _same(backing[0], backing[1])):
                            return False
                    if key not in _TIMING and input_dtype == dtype and not observe:
                        cu[1] = 5
                        table.copy_(torch.tensor([[1, 2, 3, 4, 5]], device=dev,
                                                  dtype=torch.int32))
                        _TIMING[key] = _timing(
                            rp, states, x, dt, A, B, C, D, bias, table, accepted,
                            cu, seq, rows, outputs, ring_len, is_blackwell)
                        if not _TIMING[key]:
                            return False
        _SEEN.add(key)
        return bool(_TIMING.get(key, False))
    except Exception as exc:
        logger.warning("Mamba raw replay pipeline retained original path: %r", exc)
        return False

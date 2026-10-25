# SPDX-License-Identifier: Apache-2.0
"""Private boot admission for partial conv-X/SSM fusion.

Only immutable shape choice survives. No request, recurrent history or model
output participates in admission. Real loaded parameters are checked per layer.
All execution happens during official remote startup, before compile/capture.
"""
import os
import statistics

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
_CHOICES = {}


def _same(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _layout(conv, state, dtype, weight, conv_bias, A, bias, D, ring_len):
    if (not state.is_cuda or dtype != torch.bfloat16 or state.dtype != torch.float32
            or state.ndim != 4 or tuple(state.shape[1:]) != (64, 64, 128)
            or state.shape[0] < 6 or state.stride()[1:] != (8192, 128, 1)
            or state.stride(0) < 524289 or state.stride(0) % 4
            or state.data_ptr() % 16 or conv.ndim != 3 or conv.shape[0] < 6
            or conv.shape[1] != 6144 or conv.dtype != torch.bfloat16
            or conv.device != state.device or any(s <= 0 for s in conv.stride())
            or weight.ndim != 2 or weight.shape[0] != 6144
            or weight.shape[1] != 4 or weight.stride(1) != 1
            or conv.shape[2] < weight.shape[1] + 3
            or A.shape != (64,) or A.dtype != torch.float32 or not A.is_contiguous()
            or bias.shape != (64,) or not bias.is_contiguous()
            or bias.dtype not in (torch.float32, torch.bfloat16, torch.float16)
            or D.shape != (64,) or not D.is_contiguous()
            or any(t.device != state.device for t in (weight, A, bias, D))
            or (conv_bias is not None and (conv_bias.shape != (6144,)
                or conv_bias.device != state.device or not conv_bias.is_contiguous()))):
        return None
    ss_end = state.storage_offset() + (state.shape[0] - 1) * state.stride(0) + 524289
    conv_span = 1 + 6143 * conv.stride(1) + (conv.shape[2] - 1) * conv.stride(2)
    cv_end = conv.storage_offset() + (conv.shape[0] - 1) * conv.stride(0) + conv_span
    if (ss_end * 4 > state.untyped_storage().nbytes()
            or cv_end * conv.element_size() > conv.untyped_storage().nbytes()
            or conv.stride(0) < conv_span):
        return None
    # The supported real layouts have one contiguous inner axis and no overlap.
    if not ((conv.stride(1) == 1 and conv.stride(2) >= 6144)
            or (conv.stride(2) == 1 and conv.stride(1) >= conv.shape[2])):
        return None
    from . import mamba_ssm_spec_replay as rp
    if not 5 <= rp.LAZY_PERIOD <= 24 or ring_len < rp.LAZY_PERIOD + 5 or ring_len * 5184 * 2 > 524288 * 4:
        return None
    return (state.device, state.stride(0), tuple(conv.shape[1:]), conv.stride(),
            weight.dtype, weight.stride(), bias.dtype, D.dtype,
            conv_bias is not None, ring_len, rp.LAZY_PERIOD, rp._pdl_enabled(), rp._const_div_enabled())


def _buffers(conv, state, dtype):
    dev, stride = state.device, state.stride(0)
    s = torch.full((7 * stride + 16,), .03125, device=dev, dtype=torch.float32)
    ss = torch.as_strided(s, (7, 64, 64, 128), (stride, 8192, 128, 1), 8)
    c = torch.full((7 * conv.stride(0) + 16,), .03125, device=dev, dtype=dtype)
    cv = torch.as_strided(c, (7, *conv.shape[1:]), conv.stride(), 8)
    projection = torch.full((5 * 10304 + 16,), .0625, device=dev, dtype=dtype)
    out = torch.full((7, 64, 64), .0625, device=dev, dtype=dtype)
    replay = torch.full((3, 64, 64, 128), .09375, device=dev, dtype=torch.float32)
    return {"s": s, "ss": ss, "c": c, "cv": cv, "p": projection,
            "o": out, "r": replay}


def _chain(b, weight, conv_bias, A, bias, D, table, accepted, cu, seq, rows,
           ring_len, is_blackwell, fused, observe=False, no_prefix=False):
    from .causal_conv1d import causal_conv1d_update
    from . import mamba_ssm_spec_replay as rp
    p = b["p"][8:-8].view(5, 10304)
    raw_dt = p[:, 10240:]
    conv_result = causal_conv1d_update(
        p[:, 4096:10240], b["cv"], weight, conv_bias, activation="silu",
        conv_state_indices=table, num_accepted_tokens=accepted,
        query_start_loc=cu, max_query_len=5,
        _partial_x=bool(fused),
    )
    if fused:
        values, produced = conv_result
        if not produced:
            raise RuntimeError("partial conv-X private producer did not engage")
    else:
        values = conv_result
    x = values[:, :4096].view(5, 64, 64)
    B = values[:, 4096:5120].view(5, 8, 128)
    C = values[:, 5120:6144].view(5, 8, 128)
    rp.selective_state_update_spec_replay(
        b["ss"], x, raw_dt[:, :, None].expand(-1, -1, 64),
        A[:, None, None].expand(-1, 64, 128), B, C,
        D[:, None].expand(-1, 64), bias[:, None].expand(-1, 64),
        dt_softplus=True, state_batch_indices=table, out=b["o"][1:6],
        num_accepted_tokens=accepted, cu_seqlens=cu, seq_lens=seq,
        ring_len=ring_len, is_blackwell=is_blackwell, lazy=True,
        replayed_out=b["r"][1:2] if observe else None,
        _conv_x=(b["cv"], weight, conv_bias) if fused else None,
        _private_verify_rows=rows, _private_no_prefix=no_prefix,
    )


def _check(conv, state, dtype, weight, conv_bias, A, bias, D, ring_len,
           is_blackwell, cycles):
    from . import mamba_ssm_spec_replay as rp
    dev = state.device
    advances = (1, 1, 3, 1, 2, 5, 1, 4, 2, 1, 5, 3)
    guard = torch.full((8,), .0625, device=dev, dtype=dtype)
    for observe in (False, True):
        bs = [_buffers(conv, state, dtype) for _ in range(2)]
        table = torch.tensor([[1, 2, 3, 4, 5]], device=dev, dtype=torch.int32)
        accepted = torch.ones(1, device=dev, dtype=torch.int32)
        cu = torch.tensor([0, 5], device=dev, dtype=torch.int32)
        seq = torch.empty(1, device=dev, dtype=torch.int32)
        rows = torch.empty(1, device=dev, dtype=torch.int32)
        regimes = torch.tensor([-40., -25., -8., -1., -0., 0., .001,
                                1., 8., 20., 25., 40.], device=dev, dtype=dtype)
        for b in bs:
            rp.prefill_tag_view(b["ss"])[1] = 37
        pos, previous_advance = 37, 1
        for cycle in range(cycles):
            prefix = 5 if cycle % 6 == 0 else 1 + cycle % 5
            raw = torch.arange(5 * 10304, device=dev, dtype=torch.int32) + cycle * 19
            projection = ((raw % 257).float() / 128 - 1)
            projection *= (.03125, .5, 1., 4., 16.)[cycle % 5]
            projection[(raw % 17) == 0] = -0.0
            projection = projection.to(dtype).view(5, -1)
            index = (torch.arange(320, device=dev).view(5, 64) + cycle * 3) % regimes.numel()
            projection[:, 10240:].copy_(regimes[index])
            rows.fill_(prefix)
            seq.fill_(pos + 5)
            accepted.fill_(previous_advance)
            for fused, b in enumerate(bs):
                b["p"][8:-8].copy_(projection.flatten())
                b["o"].fill_(.0625)
                b["r"].fill_(.09375)
                _chain(b, weight, conv_bias, A, bias, D, table, accepted, cu, seq,
                       rows, ring_len, is_blackwell, bool(fused), observe,
                       no_prefix=cycle % 6 == 0)
            torch.cuda.synchronize(dev)
            if not (all(_same(bs[0][k], bs[1][k]) for k in ("s", "c", "p", "o", "r"))
                    and all(_same(b["p"][:8], guard) and _same(b["p"][-8:], guard) for b in bs)):
                return False
            linear = 2 * rp.LAZY_PERIOD if cycles >= 2 * rp.LAZY_PERIOD else rp.LAZY_PERIOD
            advance = 1 if cycle < linear else min(prefix, advances[cycle % 12])
            pos += advance
            previous_advance = advance
        # Each edge starts on independent complete physical private backing.
        # NULL leading conv slot, ring slot and alternate checkpoint differ.
        for slot_values, query, replay, previous in (
            ([1, 0, 3, 4, 5], (0, 5), rp.LAZY_PERIOD - 1, 5),
            ([0, 2, 3, 4, 5], (0, 5), rp.LAZY_PERIOD - 1, 0),
            ([1, 2, 0, 4, 5], (0, 5), rp.LAZY_PERIOD + 1, 3),
            ([1, 2, 3, 4, 5], (0, 1), rp.LAZY_PERIOD - 1, 5),
            ([1, 2, 3, 4, 5], (0, 3), 1, 3),
            ([1, 2, 3, 4, 5], (1, 5), rp.LAZY_PERIOD - 1, 5),
            ([1, 2, 3, 4, 5], (0, 0), 1, 0),
            ([0, 0, 0, 0, 0], (0, 5), 1, 0),
        ):
            edge = [_buffers(conv, state, dtype) for _ in range(2)]
            table.copy_(torch.tensor([slot_values], device=dev, dtype=torch.int32))
            cu.copy_(torch.tensor(query, device=dev, dtype=torch.int32))
            seq.fill_(37 + replay + query[1] - query[0])
            accepted.fill_(previous)
            for fused, b in enumerate(edge):
                rp.prefill_tag_view(b["ss"])[1] = 37
                b["p"][8:-8].copy_(projection.flatten())
                _chain(b, weight, conv_bias, A, bias, D, table, accepted, cu, seq,
                       rows, ring_len, is_blackwell, bool(fused), observe, no_prefix=True)
            torch.cuda.synchronize(dev)
            if not (all(_same(edge[0][k], edge[1][k]) for k in ("s", "c", "p", "o", "r"))
                    and all(_same(b["p"][:8], guard) and _same(b["p"][-8:], guard) for b in edge)):
                return False
    return True


def _time(conv, state, dtype, weight, conv_bias, A, bias, D, ring_len, is_blackwell):
    from . import mamba_ssm_spec_replay as rp
    dev = state.device
    bs = [_buffers(conv, state, dtype) for _ in range(2)]
    projection = (((torch.arange(5 * 10304, device=dev, dtype=torch.int32) + 13)
                   % 257).float() / 128 - 1).to(dtype).view(5, -1)
    for b in bs:
        rp.prefill_tag_view(b["ss"])[1] = 37
        b["p"][8:-8].copy_(projection.flatten())
    table = torch.tensor([[1, 2, 3, 4, 5]], device=dev, dtype=torch.int32)
    accepted = torch.ones(1, device=dev, dtype=torch.int32)
    cu = torch.tensor([0, 5], device=dev, dtype=torch.int32)
    seq = torch.tensor([42], device=dev, dtype=torch.int32)
    rows = torch.full((1,), 5, device=dev, dtype=torch.int32)
    flush = torch.empty(64 << 20, device=dev, dtype=torch.uint8)
    stream = torch.cuda.Stream(device=dev)
    stream.wait_stream(torch.cuda.current_stream())
    graphs = {}

    def call(which):
        _chain(bs[which], weight, conv_bias, A, bias, D, table, accepted, cu, seq,
               rows, ring_len, is_blackwell, bool(which))

    for which in (0, 1):
        with torch.cuda.stream(stream):
            call(which)
            call(which)
        torch.cuda.synchronize(dev)
        for mode, repeats in (("cold", 1), ("warm", 20)):
            graph = torch.cuda.CUDAGraph()
            pool = torch.cuda.graph_pool_handle()
            captured = False
            outer = torch.cuda.current_stream()
            try:
                with torch.cuda.graph(graph, stream=stream, pool=pool):
                    for _ in range(repeats):
                        call(which)
                captured = True
                graphs[(which, mode)] = graph
            finally:
                torch.cuda.set_stream(outer)
                if not captured:
                    try:
                        torch._C._cuda_endAllocateToPool(dev.index, pool)
                    except Exception:
                        pass
    torch.cuda.synchronize(dev)
    records = {which: {"cold": [], "warm": []} for which in (0, 1)}
    for replay, prefix in ((0, 1), (1, 5), (min(4, rp.LAZY_PERIOD - 1), 3),
                           (min(9, rp.LAZY_PERIOD - 1), 5),
                           (max(rp.LAZY_PERIOD - 6, 0), 1), (rp.LAZY_PERIOD - 1, 5)):
        seq.fill_(37 + replay + 5)
        rows.fill_(prefix)
        samples = {(which, mode): [] for which in (0, 1) for mode in ("cold", "warm")}
        for trial in range(5):
            for which in ((0, 1) if trial % 2 == 0 else (1, 0)):
                for mode, repeats in (("cold", 1), ("warm", 20)):
                    # Restore private projection; conv mutates its BC slice.
                    bs[which]["p"][8:-8].copy_(projection.flatten())
                    if mode == "cold":
                        flush.zero_()
                    else:
                        graphs[(which, mode)].replay()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[(which, mode)].replay()
                    end.record()
                    end.synchronize()
                    samples[(which, mode)].append(start.elapsed_time(end) / repeats)
        for which in (0, 1):
            for mode in ("cold", "warm"):
                value = statistics.median(samples[(which, mode)])
                if value <= 0:
                    return False
                records[which][mode].append(value)
    aggregates, ratios = [], []
    for mode in ("cold", "warm"):
        ref, cand = records[0][mode], records[1][mode]
        aggregates.append(sum(cand) / sum(ref))
        ratios.extend(b / a for a, b in zip(ref, cand))
    logger.info("Mamba conv X wholechain cold/warm=%s percase=%s", aggregates, ratios)
    return max(aggregates) < .98 and statistics.median(ratios) < .98 and max(ratios) < 1.03


@torch.no_grad()
def prepare(conv, state, dtype, weight, conv_bias, A, bias, D, ring_len, is_blackwell):
    """Install only after full conv/SSM backing bytes and wholechain timing pass."""
    if os.environ.get("NEMO_MAMBA_CONV_X_FUSION", "1") == "0":
        return False
    try:
        key = _layout(conv, state, dtype, weight, conv_bias, A, bias, D, ring_len)
        if (key is None or torch.cuda.is_current_stream_capturing()
                or torch.cuda.get_device_capability(state.device) != (12, 1)):
            return False
        from . import mamba_ssm_spec_replay as rp
        if key in _CHOICES:
            admitted = _CHOICES[key] and _check(conv, state, dtype, weight, conv_bias,
                                               A, bias, D, ring_len, is_blackwell, rp.LAZY_PERIOD + 4)
        else:
            _CHOICES[key] = False
            admitted = _check(conv, state, dtype, weight, conv_bias, A, bias, D,
                              ring_len, is_blackwell, 2 * rp.LAZY_PERIOD + 8) and _time(
                                  conv, state, dtype, weight, conv_bias, A, bias, D,
                                  ring_len, is_blackwell)
            _CHOICES[key] = admitted
        logger.info("Mamba conv X fixed private admission=%s", admitted)
        return bool(admitted)
    except Exception as exc:
        logger.warning("Mamba conv X retained original path: %r", exc)
        return False

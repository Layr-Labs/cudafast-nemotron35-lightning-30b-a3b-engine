"""Remote-boot-selected exact SSM replay tile/warp and raw-load pipeline.

No model output or recurrent state is cached. Every layer compares the entire
private backing and outputs against the original planner-selected launch.
"""
import os
import statistics

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
_VARIANTS = ((4, 1), (8, 1), (8, 2), (16, 2))
_CHOICES = {}


def _same(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _layout(state, dtype, ring_len, is_blackwell):
    if (not state.is_cuda or state.dtype != torch.float32 or state.ndim != 4
            or dtype != torch.bfloat16
            or tuple(state.shape[1:]) != (64, 64, 128) or state.shape[0] < 6
            or state.stride()[1:] != (8192, 128, 1)
            or state.stride(0) < 524289 or state.stride(0) % 4
            or state.data_ptr() % 16):
        return None
    end = state.storage_offset() + (state.shape[0] - 1) * state.stride(0) + 524289
    if end * 4 > state.untyped_storage().nbytes():
        return None
    from . import mamba_ssm_spec_replay as rp
    if (rp.LAZY_PERIOD != 20 or ring_len < rp.LAZY_PERIOD + 5
            or ring_len * 5184 * 2 > 524288 * 4):
        return None
    original = rp.try_get_optimal_ssm_config(64, 128, 1, 64, "float32", is_blackwell)
    return (state.device, dtype, state.stride(0), ring_len, rp.LAZY_PERIOD,
            rp._pdl_enabled(), rp._const_div_enabled(), tuple(original))


def _check(rp, state, dtype, A, bias, D, ring_len, is_blackwell, variant, cycles):
    stride, dev = state.stride(0), state.device
    length = 7 * stride + 16
    shape, strides = (7, 64, 64, 128), (stride, 8192, 128, 1)
    advances = (1, 1, 3, 0, 2, 5, 1, 4, 2, 0, 5, 3)
    for input_dtype in (dtype,):
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
                            _pipeline=bool(which), _pipeline_config=variant if which else None,
                            _private_verify_rows=rows,
                            _private_launch_pdl=pdl,
                            _private_no_prefix=(cycle % 6 == 0),
                        )
                    torch.cuda.synchronize(dev)
                    if not (_same(outputs[0], outputs[1])
                            and _same(replayed[0], replayed[1])
                            and _same(backing[0], backing[1])):
                        return False
                    linear_cycles = 40 if cycles >= 40 else 20
                    advance = (1 if cycle < linear_cycles else
                               min(prefix, advances[cycle % len(advances)]))
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
                            _pipeline=bool(which), _pipeline_config=variant if which else None,
                            _private_verify_rows=rows,
                            _private_launch_pdl=pdl,
                            _private_no_prefix=(cycle % 6 == 0),
                        )
                    torch.cuda.synchronize(dev)
                    if not (_same(outputs[0], outputs[1])
                            and _same(replayed[0], replayed[1])
                            and _same(backing[0], backing[1])):
                        return False
    return True


def _time_variants(rp, state, dtype, A, bias, D, ring_len, is_blackwell, variants):
    """Generic private cold and warm state timing; excludes setup and cache flush.

    A warm graph contains twenty real update nodes, avoiding a Python launch gap
    between short kernels. A separate single-update graph measures a cold state
    after a private 64 MiB buffer write outside its CUDA event interval.
    """
    dev, stride = state.device, state.stride(0)
    backing = torch.full((7 * stride + 16,), 0.03125, device=dev, dtype=torch.float32)
    st = torch.as_strided(backing, (7, 64, 64, 128), (stride, 8192, 128, 1))
    rp.prefill_tag_view(st)[1] = 37
    projection = (((torch.arange(5 * 6208, device=dev, dtype=torch.int32) + 13)
                   % 257).float() / 128 - 1).to(dtype).view(5, -1)
    x = projection[:, :4096].view(5, 64, 64)
    B = projection[:, 4096:5120].view(5, 8, 128)
    C = projection[:, 5120:6144].view(5, 8, 128)
    dt = projection[:, 6144:, None].expand(-1, -1, 64)
    table = torch.tensor([[1, 2, 3, 4, 5]], device=dev, dtype=torch.int32)
    accepted = torch.ones(1, device=dev, dtype=torch.int32)
    cu = torch.tensor([0, 5], device=dev, dtype=torch.int32)
    seq = torch.tensor([42], device=dev, dtype=torch.int32)
    rows = torch.full((1,), 5, device=dev, dtype=torch.int32)
    out = torch.empty((5, 64, 64), device=dev, dtype=dtype)
    flush = torch.empty(64 << 20, device=dev, dtype=torch.uint8)
    stream = torch.cuda.Stream(device=dev)
    stream.wait_stream(torch.cuda.current_stream())
    graphs = {}

    def call(config):
        rp.selective_state_update_spec_replay(
            st, x, dt, A, B, C, D, bias, dt_softplus=True,
            state_batch_indices=table, out=out, num_accepted_tokens=accepted,
            cu_seqlens=cu, seq_lens=seq, ring_len=ring_len,
            is_blackwell=is_blackwell, lazy=True,
            _pipeline=config is not None, _pipeline_config=config,
            _private_verify_rows=rows, _private_launch_pdl=rp._pdl_enabled(),
        )

    configs = [None, *variants]
    for config in configs:
        with torch.cuda.stream(stream):
            call(config)
            call(config)
        torch.cuda.synchronize(dev)
        for mode, repeats in (("cold", 1), ("warm", 20)):
            graph = torch.cuda.CUDAGraph()
            pool = torch.cuda.graph_pool_handle()
            captured = False
            outer = torch.cuda.current_stream()
            try:
                with torch.cuda.graph(graph, stream=stream, pool=pool):
                    for _ in range(repeats):
                        call(config)
                captured = True
                graphs[(config, mode)] = graph
            finally:
                torch.cuda.set_stream(outer)
                if not captured:
                    try:
                        torch._C._cuda_endAllocateToPool(dev.index, pool)
                    except Exception:
                        pass
    torch.cuda.synchronize(dev)
    records = {config: {"cold": [], "warm": []} for config in configs}
    for replay, prefix in ((0, 1), (1, 5), (4, 3), (9, 5), (14, 1), (19, 5)):
        seq.fill_(37 + replay + 5)
        rows.fill_(prefix)
        samples = {(config, mode): [] for config in configs for mode in ("cold", "warm")}
        for trial in range(5):
            order = configs if trial % 2 == 0 else list(reversed(configs))
            for config in order:
                for mode, repeats in (("cold", 1), ("warm", 20)):
                    if mode == "cold":
                        flush.zero_()
                    else:
                        graphs[(config, mode)].replay()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[(config, mode)].replay()
                    end.record()
                    end.synchronize()
                    samples[(config, mode)].append(start.elapsed_time(end) / repeats)
        for config in configs:
            for mode in ("cold", "warm"):
                value = statistics.median(samples[(config, mode)])
                if value <= 0:
                    return None
                records[config][mode].append(value)
    admitted = {}
    for config in variants:
        aggregates, ratios = [], []
        for mode in ("cold", "warm"):
            reference, candidate = records[None][mode], records[config][mode]
            aggregates.append(sum(candidate) / sum(reference))
            ratios.extend(b / a for a, b in zip(reference, candidate))
        logger.info("Mamba SSM replay config=%s cold/warm ratios=%s percase=%s",
                    config, aggregates, ratios)
        if (max(aggregates) < 0.98 and statistics.median(ratios) < 0.98
                and max(ratios) < 1.03):
            admitted[config] = max(aggregates)
    return min(admitted, key=admitted.get) if admitted else None


@torch.no_grad()
def prepare(state, dtype, A, bias, D, ring_len, is_blackwell):
    """Return fixed admitted (M tile, warps), else preserve the original kernel.

    The first immutable layout checks every variant against the original using
    actual loaded parameters, then chooses from survivors by cold/warm timing.
    Later layers independently recheck the chosen kernel on their own parameters.
    Choice metadata persists; no history, request data or model result persists.
    """
    if os.environ.get("NEMO_MAMBA_REPLAY_PIPELINE", "1") == "0":
        return None
    try:
        key = _layout(state, dtype, ring_len, is_blackwell)
        if (key is None or torch.cuda.is_current_stream_capturing()
                or torch.cuda.get_device_capability(state.device) != (12, 1)
                or A.dtype != torch.float32 or A.shape != (64, 64, 128)
                or A.stride()[1:] != (0, 0) or bias.shape != (64, 64)
                or bias.stride(1) != 0 or D.shape != (64, 64)
                or any(t.device != state.device for t in (A, bias, D))):
            return None
        from . import mamba_ssm_spec_replay as rp
        if key in _CHOICES:
            chosen = _CHOICES[key]
            return chosen if chosen is not None and _check(
                rp, state, dtype, A, bias, D, ring_len, is_blackwell,
                chosen, cycles=24) else None
        _CHOICES[key] = None
        survivors = [config for config in _VARIANTS if _check(
            rp, state, dtype, A, bias, D, ring_len, is_blackwell, config, cycles=48)]
        if survivors:
            _CHOICES[key] = _time_variants(
                rp, state, dtype, A, bias, D, ring_len, is_blackwell, survivors)
        logger.info("Mamba SSM replay fixed layout choice=%s", _CHOICES[key])
        return _CHOICES[key]
    except Exception as exc:
        logger.warning("Mamba SSM replay tuning retained original path: %r", exc)
        return None

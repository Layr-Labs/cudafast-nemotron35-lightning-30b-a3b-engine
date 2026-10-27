# SPDX-License-Identifier: Apache-2.0
"""Private qualification for the optional eight-token convolution packet.

Only immutable weight owners and an admission decision survive this function.
No request activation, recurrent state, output, graph or closure is retained.
All execution of this module belongs to the official remote runner.
"""

import os
import statistics

import torch

from vllm.logger import init_logger

_logger = init_logger(__name__)
_ENABLED = os.environ.get("VLLM_CONV_TOKEN_PACKET", "1") != "0"
_DECISIONS = {}
_GUARD = 128
_CANARY = 13.25


def _guarded_like(source):
    # Preserve holes as well as logical strides; projected-state views need
    # not be compact, and the old launcher retains their actual strides.
    if any(s <= 0 for s in source.stride()):
        raise ValueError("unsupported convolution view")
    span = 1 + sum((n - 1) * s for n, s in zip(source.shape, source.stride()))
    owner = torch.full(
        (span + 2 * _GUARD,), _CANARY, dtype=source.dtype, device=source.device,
    )
    view = torch.as_strided(owner, source.shape, source.stride(), _GUARD)
    return owner, view


def _words_equal(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def _signature(tensor):
    if tensor is None:
        return None
    try:
        version = tensor._version
    except RuntimeError:
        # An inference-mode immutable weight can have no version counter.
        version = None
    return (
        tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
        tensor.dtype, tensor.device.index, version,
    )


def admit_token_packet(launch, x, states, initial, out, weight, bias, apc):
    """Compare against the retained scalar-token kernel on private storage.

    A refusal, optional compilation failure, or unknown layout keeps the old
    path. The caller catches optional failures before mutating actual state.
    """
    if not _ENABLED or x.numel() == 0:
        return False
    key = (
        _signature(weight), _signature(bias), tuple(x.shape), tuple(x.stride()),
        tuple(states.shape), tuple(states.stride()), tuple(out.stride()),
        tuple(initial.shape), initial.dtype, bool(apc),
    )
    cached = _DECISIONS.get(key)
    if cached is not None and cached[0] is weight and cached[1] is bias:
        return cached[2]
    if torch.cuda.is_current_stream_capturing():
        return False
    if torch.cuda.get_device_capability(x.device) != (12, 1):
        return False
    # Cache a refusal first. A failed optional probe is not retried for the
    # same immutable weight/shape, including after an exception.
    _DECISIONS[key] = (weight, bias, False, None)
    if (
        x.dtype != torch.bfloat16 or states.dtype != torch.bfloat16
        or out.dtype != torch.bfloat16 or initial.dtype != torch.bool
        or x.ndim != 2 or states.ndim != 3
        or weight.ndim != 2 or weight.shape[1] not in (2, 3, 4)
        or initial.device != x.device
        or any(n <= 0 for n in (*x.shape, *states.shape))
    ):
        return False

    x_owner, private_x = _guarded_like(x)
    old_state_owner, old_states = _guarded_like(states)
    new_state_owner, new_states = _guarded_like(states)
    old_out_owner, old_out = _guarded_like(out)
    new_out_owner, new_out = _guarded_like(out)
    private_initial = torch.empty_like(initial)
    state_seed = torch.empty_like(states)
    weight_before = weight.detach().clone()
    bias_before = bias.detach().clone() if bias is not None else None
    generator = torch.Generator(device=x.device)
    generator.manual_seed(274143)

    def refuse(reason):
        _logger.info(
            "Conv token packet private qualification refused: shape=%s reason=%s",
            tuple(x.shape), reason,
        )
        return False

    def random_values(shape, scale):
        return torch.randn(
            shape, dtype=torch.float32, device=x.device, generator=generator,
        ).mul_(scale).to(torch.bfloat16)

    def prepare(case):
        x_owner.fill_(_CANARY)
        if case == 0:
            private_x.copy_(x)
            state_seed.copy_(states)
            private_initial.copy_(initial)
        else:
            scale = (0.25, 8.0, 1.0)[case - 1]
            private_x.copy_(random_values(x.shape, scale))
            state_seed.copy_(random_values(states.shape, scale))
            private_initial.fill_(case != 2)
            if case == 3:
                # A repeating signed/zero/subnormal packet changes contents
                # at fixed pointers without using or retaining request data.
                edge = torch.tensor(
                    [0.0, -0.0, 2.0**-133, -2.0**-133,
                     0.5, -0.5, 64.0, -64.0],
                    dtype=torch.bfloat16, device=x.device,
                )
                private_x.copy_(edge[torch.arange(x.shape[1], device=x.device) % 8])
        old_state_owner.fill_(_CANARY)
        new_state_owner.fill_(_CANARY)
        old_states.copy_(state_seed)
        new_states.copy_(state_seed)
        old_out_owner.fill_(_CANARY)
        new_out_owner.fill_(_CANARY)

    def equal_private_results():
        # Whole owners include untouched holes/prefix/suffix canaries and
        # every logical state element, including unreferenced cache lines.
        return (
            _words_equal(old_out_owner, new_out_owner)
            and _words_equal(old_state_owner, new_state_owner)
        )

    # The exact pre-existing metadata/cache mapping is shared read-only.
    # Probe actual inputs and independent values with both initial-state
    # branches; no production cache/output is used as a destination.
    for case in range(4):
        prepare(case)
        immutable_x = x_owner.clone()
        immutable_initial = private_initial.clone()
        launch(False, private_x, old_states, private_initial, old_out)
        launch(True, private_x, new_states, private_initial, new_out)
        if (
            not equal_private_results()
            or not _words_equal(x_owner, immutable_x)
            or not torch.equal(private_initial, immutable_initial)
        ):
            return refuse("word/readonly check case " + str(case))

    # Capture identical fixed-pointer work for both paths. Refill all inputs
    # outside replay, and compare outputs AND recurrent-cache side effects.
    prepare(1)
    old_graph, new_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(old_graph):
        for _ in range(16):
            launch(False, private_x, old_states, private_initial, old_out)
    with torch.cuda.graph(new_graph):
        for _ in range(16):
            launch(True, private_x, new_states, private_initial, new_out)
    for case in (3, 2, 0, 1):
        prepare(case)
        immutable_x = x_owner.clone()
        old_graph.replay()
        new_graph.replay()
        if not equal_private_results() or not _words_equal(x_owner, immutable_x):
            return refuse("changed-content graph check case " + str(case))

    pressure = torch.empty(64 * 1024 * 1024, device=x.device, dtype=torch.uint8)
    medians = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for case in (1, 2):
        prepare(case)
        for cold in (False, True):
            times = [[], []]
            for sample in range(5):
                for which in ((0, 1) if sample % 2 == 0 else (1, 0)):
                    # Reset state and initialize output outside the events.
                    state = old_states if which == 0 else new_states
                    output_owner = old_out_owner if which == 0 else new_out_owner
                    state.copy_(state_seed)
                    output_owner.fill_(_CANARY)
                    if cold:
                        pressure.fill_(sample)
                    start.record()
                    (old_graph if which == 0 else new_graph).replay()
                    end.record()
                    end.synchronize()
                    elapsed = start.elapsed_time(end)
                    if elapsed <= 0:
                        return refuse("nonpositive private event duration")
                    times[which].append(elapsed)
            old_ms, new_ms = (statistics.median(v) for v in times)
            medians.append((old_ms, new_ms))
            if not equal_private_results():
                return refuse("post-timing cache/output words")
    if not torch.equal(weight, weight_before):
        return refuse("weight owner changed or contains NaN")
    if bias is not None and not torch.equal(bias, bias_before):
        return refuse("bias owner changed or contains NaN")
    ratios = tuple(new / old for old, new in medians)
    total_ratio = sum(new for _, new in medians) / sum(old for old, _ in medians)
    if max(ratios) >= 1.01 or statistics.median(ratios) >= 0.99 or total_ratio >= 0.99:
        return refuse("private speed margin ratios=" + str(ratios))
    _DECISIONS[key] = (weight, bias, True, (ratios, total_ratio))
    _logger.info(
        "Conv token packet private qualification admitted: shape=%s ratios=%s total=%.6f",
        tuple(x.shape), ratios, total_ratio,
    )
    return True

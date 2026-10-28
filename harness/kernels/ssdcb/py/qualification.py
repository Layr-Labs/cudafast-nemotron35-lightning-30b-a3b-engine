# SPDX-License-Identifier: Apache-2.0
"""Remote-only synthetic fixture for the optional model warmup admission.

These private data are synthetic and have no request-derived values. A pass
is evidence for these cases only, not a proof of equivalence for all inputs.
"""
import statistics

import torch


class _RequireLaunch:
    def __init__(self, module):
        self.module = module
        self.launched = False

    def run(self, *args):
        result = self.module.run(*args)
        self.launched |= bool(result)
        return result


def _bits_equal(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@torch.no_grad()
def probe(module, device, chunk, use_initial, d_vector, *, multi=False,
          index64=False, use_cache=False, padded=False, edges=False):
    """Compare state and scan together in eager and changed-input replay.

    Failure or a missing launch rejects the optional path at model warmup.
    """
    from vllm.model_executor.layers.mamba.ops.ssd_prefill_fused import fused_prefill

    if (chunk not in (128, 256) or (multi and use_cache)
            or torch.cuda.is_current_stream_capturing()):
        return False, None
    if not module.prepare(device):
        return False, None
    gen = torch.Generator(device=device).manual_seed(6317 + chunk)
    total, chunks = chunk + 37, 2
    def random(shape, dtype, padding=0):
        if not padding:
            return torch.randn(shape, device=device, dtype=dtype, generator=gen) * 0.125
        # Padded leading stride exercises views without importing request data.
        stride = [0] * len(shape)
        stride[-1] = 1
        for i in range(len(shape) - 2, -1, -1):
            stride[i] = stride[i + 1] * shape[i + 1]
        stride[0] += padding
        tensor = torch.empty_strided(shape, stride, device=device, dtype=dtype)
        return tensor.normal_(generator=gen).mul_(0.125)

    pad = 128 if padded else 0
    x = random((total, 64, 64), torch.bfloat16, pad)
    b = random((total, 8, 128), torch.bfloat16, pad)
    c = random((total, 8, 128), torch.bfloat16, pad)
    dt = random((64, chunks, chunk), torch.float32).abs_()
    da = random((64, chunks, chunk), torch.float32).abs_().neg_()
    initial = random((2 if multi else 1, 64, 64, 128), torch.float32) if use_initial else None
    d = random((64, 64) if d_vector else (64,), torch.float32)
    if edges:
        # Test signed zero and special-value propagation independently of data
        # from the model; exact byte comparison can reject different NaN bits.
        x[0, 0, :4] = torch.tensor([0.0, -0.0, float("inf"), float("nan")],
                                    device=device, dtype=x.dtype)
        b[0, 0, :4] = torch.tensor([0.0, -0.0, 1.0, -1.0], device=device, dtype=b.dtype)
        c[0, 0, :4] = torch.tensor([-0.0, 0.0, 1.0, -1.0], device=device, dtype=c.dtype)
    index_dtype = torch.int64 if index64 else torch.int32
    cu = torch.tensor([0, chunk, total], device=device, dtype=index_dtype)
    last = torch.tensor([0, 1] if multi else [1], device=device, dtype=index_dtype)
    seq = torch.tensor([0, 1] if multi else [0, 0], device=device, dtype=index_dtype)
    inputs = [x, b, c, dt, da, cu, last, seq, d]
    if initial is not None:
        inputs.append(initial)
    guard = 64
    size = total * 64 * 64
    owners = [torch.full((size + 2 * guard,), 17, device=device, dtype=x.dtype)
              for _ in range(2)]
    outs = [owner[guard:guard + size].view(total, 64, 64) for owner in owners]
    final_index = torch.tensor([1], device=device, dtype=index_dtype) if use_cache else None
    caches = ([torch.full((3, 64, 64, 128), 23, device=device, dtype=torch.float32)
               for _ in range(2)] if use_cache else [None, None])
    if final_index is not None:
        inputs.append(final_index)
    proxy = _RequireLaunch(module)

    def call(which):
        return fused_prefill(x, b, c, dt, da, cu, last, seq, outs[which],
                             D=d, initial=initial,
                             final_cache=caches[which], final_index=final_index,
                             _private_shared_cb=proxy if which else None)

    def compare(states):
        torch.cuda.synchronize(device)
        state_ok = ((states[0] is None and states[1] is None
                     and _bits_equal(caches[0], caches[1])
                     and bool((caches[0][0] == 23).all())
                     and bool((caches[0][2] == 23).all())) if use_cache else
                    (states[0] is not None and states[1] is not None
                     and _bits_equal(states[0], states[1])))
        return (proxy.launched and state_ok
                and _bits_equal(owners[0], owners[1])
                and bool((owners[0][:guard] == 17).all())
                and bool((owners[0][-guard:] == 17).all()))

    before = [t.clone() for t in inputs]
    states = [call(0), call(1)]
    if not compare(states) or any(not _bits_equal(a, t) for a, t in zip(before, inputs)):
        return False, None

    graphs, graph_states = [], []
    for which in range(2):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = call(which)
        graphs.append(graph)
        graph_states.append(result)
    # Replay different, still valid chunk boundaries and different token data.
    x.neg_()
    b.mul_(-0.5)
    c.mul_(1.25)
    dt.mul_(0.5)
    da.mul_(0.75)
    if initial is not None:
        initial.neg_()
    cu.copy_(torch.tensor([0, chunk - 1, total], device=device, dtype=cu.dtype))
    for owner in owners:
        owner.fill_(17)
    before = [t.clone() for t in inputs]
    for graph in graphs:
        graph.replay()
    if not compare(graph_states) or any(not _bits_equal(a, t) for a, t in zip(before, inputs)):
        return False, None

    # This metadata causes the candidate to decline only the second chunk;
    # the original scan's complement must still produce the full output.
    # With no initial-state tensor the out-of-range sequence ID is only used
    # for equality, so the original kernel has no out-of-range INIT access.
    if not use_initial and not multi:
        seq.copy_(torch.tensor([0, 7], device=device, dtype=seq.dtype))
        for owner in owners:
            owner.fill_(17)
        for graph in graphs:
            graph.replay()
        if not compare(graph_states):
            return False, None
        seq.zero_()

    # Time the entire state-pass/scan chain, in alternating order, both warm
    # and after the same separate memory-pressure operation.  The pressure
    # kernel is outside the timed interval and is not a claim about cache state.
    pressure = torch.zeros((64 * 1024 * 1024,), device=device, dtype=torch.uint8)
    timings = []
    for pressured in (False, True):
        samples = [[], []]
        for turn in range(5):
            for which in ((0, 1) if turn % 2 == 0 else (1, 0)):
                if pressured:
                    pressure.add_(1)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                # The pressured case times a single replay so the first-call
                # cost cannot be hidden by seven subsequent warm replays.
                count = 1 if pressured else 8
                for _ in range(count):
                    graphs[which].replay()
                end.record()
                end.synchronize()
                samples[which].append(start.elapsed_time(end) / count)
        old, new = (statistics.median(v) for v in samples)
        timings.append((old, new))
    return all(old > 0 and new < old * 0.98 for old, new in timings), timings


@torch.no_grad()
def qualify(module, device, chunk):
    """Conservative warmup admission, independent of request values.

    Run before graph capture on an official remote worker. This covers several
    layouts and boundary situations, but cannot prove every possible input.
    Callers must independently enforce the supported metadata/layout contract.
    """
    cases = (
        dict(use_initial=False, d_vector=False),
        dict(use_initial=True, d_vector=True, padded=True, index64=True),
        dict(use_initial=True, d_vector=False, multi=True),
        dict(use_initial=False, d_vector=True, use_cache=True, padded=True),
        dict(use_initial=False, d_vector=False, edges=True),
    )
    ratios = []
    for case in cases:
        passed, timing = probe(module, device, chunk, **case)
        if not passed or timing is None:
            return False, ratios
        if not case.get("edges"):
            ratios.extend(new / old for old, new in timing)
    return (statistics.median(ratios) < 0.98 and max(ratios) < 1.0), ratios

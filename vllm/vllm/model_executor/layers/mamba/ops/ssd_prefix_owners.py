# SPDX-License-Identifier: Apache-2.0
"""Optional single-sequence, per-prefix ownership for the original SSD state pass.

The original ascending dot and FP32 recurrence are retained; earlier prefixes
are recomputed instead of reassociated. There is no cross-CTA synchronization.
Qualification is authored for official remote execution only. No request
input, boundary, recurrent cache, output, graph or closure is retained.
"""

import os
import statistics

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

from .triton_helpers import fast_exp

_ENABLED = os.environ.get("VLLM_SSD_PREFIX_OWNERS", "1") != "0"
_DECISIONS = {}
_logger = init_logger(__name__)
_GUARD = 128
_CANARY = 11.25


@triton.jit
def _state_prefix_owner_kernel(
    X,
    B,
    DT,
    DA,
    CU,
    LAST,
    INIT,
    OUT,
    FINAL_CACHE,
    FINAL_INDEX,
    H: tl.constexpr,
    N: tl.constexpr,
    RATIO: tl.constexpr,
    CHUNK: tl.constexpr,
    XS: tl.constexpr,
    BS: tl.constexpr,
    TS: tl.constexpr,
    AS: tl.constexpr,
    IS: tl.constexpr,
    OS: tl.constexpr,
    HAS_INIT: tl.constexpr,
    WRITE_FINAL: tl.constexpr,
    FS: tl.constexpr,
    CACHE_SLOTS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # One independent CTA owns exactly one requested prefix boundary.
    target = tl.program_id(0)
    batch = 0
    head = tl.program_id(1)
    start = tl.load(LAST + batch - 1, mask=batch > 0, other=-1) + 1
    end = tl.load(LAST + batch) + 1
    # The empty-range case has the same initial final-cache value as the
    # original; exactly target zero owns that store, and no boundary is stored.
    if target >= end and not (target == 0 and end <= start):
        return
    m = tl.arange(0, BM)
    n = tl.program_id(2) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    state_offsets = m[:, None] * OS[2] + n[None, :] * OS[3]
    state_mask = (m[:, None] < H) & (n[None, :] < N)
    if HAS_INIT:
        states = tl.load(
            INIT
            + batch * IS[0]
            + head * IS[1]
            + m[:, None] * IS[2]
            + n[None, :] * IS[3],
            mask=state_mask,
            other=0.0,
        ).to(tl.float32)
    else:
        states = tl.zeros((BM, BN), tl.float32)
    for chunk in range(start, tl.minimum(end, target + 1)):
        begin = tl.load(CU + chunk)
        limit = tl.load(CU + chunk + 1) - begin
        xp = X + begin * XS[0] + head * XS[1]
        bp = B + begin * BS[0] + (head // RATIO) * BS[1]
        tp = DT + head * TS[0] + chunk * TS[1]
        ap = DA + head * AS[0] + chunk * AS[1]
        dA_cs = tl.load(ap + (CHUNK - 1) * AS[2]).to(tl.float32)
        acc = tl.zeros((BM, BN), tl.float32)
        for k in range(0, limit, BK):
            x = tl.load(
                xp + m[:, None] * XS[2] + (k + kk[None, :]) * XS[0],
                mask=(m[:, None] < H) & (kk[None, :] < limit - k),
                other=0.0,
            )
            b = tl.load(
                bp + (k + kk[:, None]) * BS[0] + n[None, :] * BS[2],
                mask=(kk[:, None] < limit - k) & (n[None, :] < N),
                other=0.0,
            ).to(tl.float32)
            dA_cs_k = tl.load(ap + (k + kk) * AS[2], mask=kk < limit - k, other=0.0).to(
                tl.float32
            )
            dt_k = tl.load(tp + (k + kk) * TS[2], mask=kk < limit - k, other=0.0).to(
                tl.float32
            )
            scale = fast_exp(tl.minimum(dA_cs - dA_cs_k, 0.0)) * dt_k
            b *= scale[:, None]
            b = b.to(X.dtype.element_ty)
            acc += tl.dot(x, b)
        # Same FP32 recurrence as _state_passing_fwd_kernel, in chunk order.
        new_states = acc.to(tl.float32)
        states = fast_exp(dA_cs) * states + new_states
        # Earlier prefixes are recomputed privately. Their real owner alone
        # publishes each boundary; no identical-value multi-CTA write race.
        if chunk == target:
            tl.store(
                OUT + chunk * OS[0] + head * OS[1] + state_offsets,
                states, mask=state_mask,
            )

    # Same terminal FP32 registers as OUT[end - 1]. No new reduction or rounding.
    # The host guard restricts this sink to one nonempty prefill sequence.
    if WRITE_FINAL and (target == end - 1 or (target == 0 and end <= start)):
        cache_slot = tl.load(FINAL_INDEX).to(tl.int64)
        cache_slot = tl.where(cache_slot < 0, cache_slot + CACHE_SLOTS, cache_slot)
        tl.store(
            FINAL_CACHE + cache_slot * FS[0] + head * FS[1]
            + m[:, None] * FS[2] + n[None, :] * FS[3],
            states,
            mask=state_mask,
        )


def _guarded(source):
    if source is None:
        return None, None
    if any(n <= 0 for n in source.shape) or any(s <= 0 for s in source.stride()):
        raise ValueError("unsupported SSD owner layout")
    span = 1 + sum((n - 1) * s for n, s in zip(source.shape, source.stride()))
    if (span + 2 * _GUARD) * source.element_size() > 1024 * 1024 * 1024:
        raise ValueError("private SSD owner exceeds qualification budget")
    owner = torch.full(
        (span + 2 * _GUARD,), _CANARY, dtype=source.dtype, device=source.device,
    )
    view = torch.as_strided(owner, source.shape, source.stride(), _GUARD)
    return owner, view


def _same_words(a, b):
    return a is b if a is None or b is None else torch.equal(
        a.view(torch.uint8), b.view(torch.uint8),
    )


def _layout(tensor):
    return None if tensor is None else (
        tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype,
    )


def admit_prefix_owners(
    launch, x, B, C, dt, dA, cu, last, seq, initial, states, out,
    final_cache, final_index, D,
):
    """Qualify complete state+scan execution, not just the changed state pass."""
    if not _ENABLED or last.numel() != 1 or not 1 < dt.shape[1] <= 4:
        return False
    if x.shape[0] < dt.shape[1] or torch.cuda.get_device_capability(x.device) != (12, 1):
        return False
    # D is an immutable model weight. Everything else retained in this key
    # is shape/type/stride metadata, never tensor contents or request owners.
    try:
        d_version = None if D is None else D._version
    except RuntimeError:
        d_version = None
    key = (
        x.device.index, tuple(_layout(t) for t in (x, B, C, dt, dA, cu, last, seq,
                                                   initial, states, out, final_cache,
                                                   final_index)),
        None if D is None else (D.data_ptr(), _layout(D), d_version),
    )
    cached = _DECISIONS.get(key)
    if cached is not None and cached[0] is D:
        return cached[1]
    if torch.cuda.is_current_stream_capturing():
        return False
    _DECISIONS[key] = (D, False, None)
    if (
        x.dtype != torch.bfloat16 or B.dtype != x.dtype or C.dtype != x.dtype
        or dt.dtype != torch.float32 or dA.dtype != dt.dtype
        or states.dtype != torch.float32 or out.dtype != torch.bfloat16
        or x.shape[1:] != (64, 64) or B.shape != C.shape
        or B.shape[1:] != (8, 128) or dt.shape != dA.shape
        or dt.shape[0] != 64 or dt.shape[2] not in (128, 256)
    ):
        return False

    sources = [x, B, C, dt, dA, cu, last, seq, initial, final_index]
    inputs = [_guarded(t) for t in sources]
    old_outputs = [_guarded(t) for t in (states, out, final_cache)]
    new_outputs = [_guarded(t) for t in (states, out, final_cache)]
    generator = torch.Generator(device=x.device)
    generator.manual_seed(102743)
    d_before = None if D is None else D.detach().clone().contiguous()
    chunks, chunk_size = dt.shape[1:]

    def refuse(reason):
        _logger.info(
            "SSD prefix ownership private qualification refused: chunks=%s reason=%s",
            chunks, reason,
        )
        return False

    def random_values(shape, dtype, scale):
        return torch.randn(
            shape, device=x.device, dtype=torch.float32, generator=generator,
        ).mul_(scale).to(dtype)

    def prepare(case):
        for source, (owner, view) in zip(sources, inputs):
            if source is not None:
                owner.fill_(_CANARY)
                view.copy_(source)
        if case:
            for i in (0, 1, 2):
                inputs[i][1].copy_(random_values(sources[i].shape, sources[i].dtype,
                                                (0.125, 1.0, 4.0)[case - 1]))
            positive = torch.rand(
                dt.shape, device=x.device, dtype=torch.float32, generator=generator,
            ).mul_(0.05).add_(0.01)
            inputs[3][1].copy_(positive)
            inputs[4][1].copy_(-torch.cumsum(positive, dim=-1))
            if initial is not None:
                inputs[8][1].copy_(random_values(initial.shape, initial.dtype, 0.125))
            if case == 1:
                inputs[6][1].fill_(max(chunks - 2, 0))
            elif case == 2:
                # Balanced short chunks change the same CU pointer contents.
                step = max(1, min(chunk_size, x.shape[0] // chunks))
                inputs[5][1].copy_(torch.arange(
                    chunks + 1, device=x.device, dtype=cu.dtype,
                ) * step)
                inputs[6][1].fill_(0)
            else:
                # The original's empty state-prefix behavior is preserved.
                inputs[6][1].fill_(-1)
            if final_index is not None:
                inputs[9][1].fill_(-1 if case % 2 else 0)
        for owner, _ in (*old_outputs, *new_outputs):
            if owner is not None:
                owner.fill_(_CANARY)

    def call(use_prefix):
        px, pb, pc, pdt, pda, pcu, plast, pseq, pinit, pindex = [v for _, v in inputs]
        ps, po, pf = [v for _, v in (new_outputs if use_prefix else old_outputs)]
        launch(use_prefix, px, pb, pc, pdt, pda, pcu, plast, pseq,
               pinit, ps, po, pf, pindex)

    def result_equal():
        return all(_same_words(a[0], b[0]) for a, b in zip(old_outputs, new_outputs))

    def snapshot_inputs():
        return [None if owner is None else owner.clone() for owner, _ in inputs]

    def inputs_equal(snapshot):
        return all(_same_words(owner, before)
                   for (owner, _), before in zip(inputs, snapshot))

    # Compare every intermediate FP32 boundary, final-cache owner and BF16
    # scan output, with all padding/holes/canaries and read-only input owners.
    for case in range(4):
        prepare(case)
        before = snapshot_inputs()
        call(False)
        call(True)
        if not result_equal() or not inputs_equal(before):
            return refuse("word/owner check case " + str(case))

    prepare(0)
    old_graph, new_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(old_graph):
        for _ in range(8):
            call(False)
    with torch.cuda.graph(new_graph):
        for _ in range(8):
            call(True)
    for case in (3, 2, 1, 0):
        prepare(case)
        before = snapshot_inputs()
        old_graph.replay()
        new_graph.replay()
        if not result_equal() or not inputs_equal(before):
            return refuse("changed-metadata/content graph case " + str(case))

    pressure = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=x.device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    medians = []
    for case in (0, 1):
        prepare(case)
        for cold in (False, True):
            times = [[], []]
            for sample in range(5):
                for which in ((0, 1) if sample % 2 == 0 else (1, 0)):
                    # Reinitialize complete output owners outside events;
                    # input/initial/cache indices never change in timing.
                    for owner, _ in (old_outputs if which == 0 else new_outputs):
                        if owner is not None:
                            owner.fill_(_CANARY)
                    if cold:
                        pressure.fill_(sample)
                    start.record()
                    (old_graph if which == 0 else new_graph).replay()
                    end.record()
                    end.synchronize()
                    ms = start.elapsed_time(end)
                    if ms <= 0:
                        return refuse("nonpositive event duration")
                    times[which].append(ms)
            old_ms, new_ms = (statistics.median(v) for v in times)
            medians.append((old_ms, new_ms))
            if not result_equal():
                return refuse("post-timing boundary/cache/output words")
    if D is not None and not _same_words(D.contiguous(), d_before):
        return refuse("immutable D weight changed")
    ratios = tuple(new / old for old, new in medians)
    total_ratio = sum(new for _, new in medians) / sum(old for old, _ in medians)
    if max(ratios) >= 1.01 or statistics.median(ratios) >= 0.99 or total_ratio >= 0.99:
        return refuse("complete state+scan private speed margin " + str(ratios))
    _DECISIONS[key] = (D, True, (ratios, total_ratio))
    _logger.info(
        "SSD prefix ownership private qualification admitted: chunks=%s ratios=%s total=%.6f",
        chunks, ratios, total_ratio,
    )
    return True


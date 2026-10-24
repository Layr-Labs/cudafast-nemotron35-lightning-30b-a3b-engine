"""Draft-only BF16 exact-zero compaction; original target kernels are untouched.

Keep every nonzero ReLU2 activation, including NaNs, in original K order.
The compact reduction has a different FP32 tree from the existing drafter.
Proposal quality, exact-token acceptance and throughput need remote validation.
"""

import os

import torch

from vllm.triton_utils import tl, triton

_ENABLED = os.environ.get("MTP_RELU2_COMPACT", "1") != "0"


@triton.jit
def _compact(h, indices, values, counts, I: tl.constexpr, BLOCK: tl.constexpr):
    slot = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    value = tl.load(h + slot * I + k, mask=k < I, other=0.0)
    live = (k < I) & (value != 0.0)
    position = tl.cumsum(live.to(tl.int32), axis=0) - 1
    tl.store(indices + slot * I + position, k, mask=live)
    tl.store(values + slot * I + position, value, mask=live)
    tl.store(counts + slot, tl.sum(live.to(tl.int32), axis=0))


@triton.jit
def _down(
    indices, values, counts, w2, sd, ids, weights, y,
    M: tl.constexpr, H: tl.constexpr, I: tl.constexpr,
    ROUTED_SCALE: tl.constexpr,
    TOPK: tl.constexpr, S: tl.constexpr,
    BN: tl.constexpr, BK: tl.constexpr,
):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    lane = tl.arange(0, BK)
    routed = tl.zeros((BN,), tl.float32)
    for j in range(TOPK):
        expert = tl.load(ids + j).to(tl.int64)
        count = tl.load(counts + j)
        acc = tl.zeros((BK, BN), tl.float32)
        for start in range(0, count, BK):
            pos = start + lane
            k = tl.load(indices + j * I + pos, mask=pos < count, other=0)
            value = tl.load(values + j * I + pos, mask=pos < count, other=0.0)
            weight = tl.load(
                w2 + expert * I * H + k[:, None].to(tl.int64) * H + n[None, :],
                mask=(pos[:, None] < count) & (n[None, :] < H),
                other=0.0, eviction_policy="evict_first",
            )
            acc += weight.to(tl.float32) * value[:, None].to(tl.float32)
        routed += tl.load(weights + j).to(tl.float32) * tl.sum(acc, axis=0)
    shared = tl.zeros((BN,), tl.float32)
    for s in range(S):
        slot = TOPK + s
        count = tl.load(counts + slot)
        acc = tl.zeros((BK, BN), tl.float32)
        for start in range(0, count, BK):
            pos = start + lane
            k = tl.load(indices + slot * I + pos, mask=pos < count, other=0)
            value = tl.load(values + slot * I + pos, mask=pos < count, other=0.0)
            weight = tl.load(
                sd + s * I * H + k[:, None].to(tl.int64) * H + n[None, :],
                mask=(pos[:, None] < count) & (n[None, :] < H),
                other=0.0, eviction_policy="evict_first",
            )
            acc += weight.to(tl.float32) * value[:, None].to(tl.float32)
        shared += tl.sum(acc, axis=0)
    out = (routed * ROUTED_SCALE + shared).to(y.dtype.element_ty)
    for row in range(M):
        tl.store(y + row * H + n, out, mask=n < H)


def maybe_down(h, packed, ids, weights, y, routed_scale):
    """Reuse the existing finite, exact BF16 transpose prepared before capture."""
    if not _ENABLED:
        return False
    w2, sd = packed
    # Preserve the compact fast path for a smaller native draft expert set.
    # Counts and ordered K values remain current-call data; weights stay BF16.
    topk = ids.numel()
    if not 1 <= topk <= 6:
        return False
    slots = topk + 2
    if (
        h.dtype != torch.bfloat16 or h.shape != (slots, 1856)
        or w2.dtype != torch.bfloat16 or w2.shape != (128, 1856, 2688)
        or sd.dtype != torch.bfloat16 or sd.shape != (2, 1856, 2688)
        or weights.dtype != torch.float32 or weights.numel() != topk
        or ids.dtype not in (torch.int32, torch.int64)
        or y.dtype != torch.bfloat16 or y.ndim != 2
        or not 1 <= y.shape[0] <= 8 or y.shape[1] != 2688
        or not all(t.is_contiguous() for t in (h, w2, sd, ids, weights, y))
        or not h.is_cuda
        or not all(t.device == h.device for t in (w2, sd, ids, weights, y))
    ):
        return False
    indices = torch.empty(h.shape, dtype=torch.int32, device=h.device)
    values = torch.empty_like(h)
    counts = torch.empty(slots, dtype=torch.int32, device=h.device)
    _compact[(slots,)](h, indices, values, counts, I=1856, BLOCK=2048, num_warps=4)
    _down[(triton.cdiv(2688, 32),)](
        indices, values, counts, w2, sd, ids, weights, y,
        M=y.shape[0], H=2688, I=1856, ROUTED_SCALE=routed_scale,
        TOPK=topk, S=2, BN=32, BK=64, num_warps=4, num_stages=2,
    )
    return True

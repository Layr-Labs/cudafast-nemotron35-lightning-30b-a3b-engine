"""Draft-only BF16 exact-zero compaction; original target kernels are untouched.

Keep every nonzero ReLU2 activation, including NaNs, in original K order.
The compact reduction has a different FP32 tree from the existing drafter.
Expert slots run independently, then combine in original expert order.
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


@triton.jit
def _down_partial(
    indices, values, counts, w2, sd, ids, partial,
    H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
    BN: tl.constexpr, BK: tl.constexpr,
):
    # Independent expert slots; full stable compact K sequence in every slot.
    slot = tl.program_id(1)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    lane = tl.arange(0, BK)
    count = tl.load(counts + slot)
    if slot < TOPK:
        expert = tl.load(ids + slot).to(tl.int64)
        base = w2 + expert * I * H
    else:
        base = sd + (slot - TOPK).to(tl.int64) * I * H
    acc = tl.zeros((BK, BN), tl.float32)
    for start in range(0, count, BK):
        pos = start + lane
        k = tl.load(indices + slot * I + pos, mask=pos < count, other=0)
        value = tl.load(values + slot * I + pos, mask=pos < count, other=0.0)
        weight = tl.load(
            base + k[:, None].to(tl.int64) * H + n[None, :],
            mask=(pos[:, None] < count) & (n[None, :] < H),
            other=0.0, eviction_policy="evict_first",
        )
        acc += weight.to(tl.float32) * value[:, None].to(tl.float32)
    tl.store(partial + slot * H + n, tl.sum(acc, axis=0), mask=n < H)


@triton.jit
def _down_combine(partial, weights, y, H: tl.constexpr, M: tl.constexpr,
                  TOPK: tl.constexpr, S: tl.constexpr,
                  ROUTED_SCALE: tl.constexpr, BN: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    routed = tl.zeros((BN,), tl.float32)
    for j in range(TOPK):
        part = tl.load(partial + j * H + n, mask=n < H, other=0.0)
        routed += tl.load(weights + j).to(tl.float32) * part
    shared = tl.zeros((BN,), tl.float32)
    for s in range(S):
        shared += tl.load(partial + (TOPK + s) * H + n, mask=n < H, other=0.0)
    out = (routed * ROUTED_SCALE + shared).to(y.dtype.element_ty)
    for row in range(M):
        tl.store(y + row * H + n, out, mask=n < H)


def maybe_down(h, packed, ids, weights, y, routed_scale):
    """Reuse the existing finite, exact BF16 transpose prepared before capture."""
    if not _ENABLED:
        return False
    w2, sd = packed
    if (
        h.dtype != torch.bfloat16 or h.shape != (8, 1856)
        or w2.dtype != torch.bfloat16 or w2.shape != (128, 1856, 2688)
        or sd.dtype != torch.bfloat16 or sd.shape != (2, 1856, 2688)
        or weights.dtype != torch.float32 or weights.numel() != 6
        or ids.dtype not in (torch.int32, torch.int64) or ids.numel() != 6
        or y.dtype != torch.bfloat16 or y.ndim != 2
        or not 1 <= y.shape[0] <= 8 or y.shape[1] != 2688
        or not all(t.is_contiguous() for t in (h, w2, sd, ids, weights, y))
        or not h.is_cuda
        or not all(t.device == h.device for t in (w2, sd, ids, weights, y))
    ):
        return False
    indices = torch.empty(h.shape, dtype=torch.int32, device=h.device)
    values = torch.empty_like(h)
    counts = torch.empty(8, dtype=torch.int32, device=h.device)
    _compact[(8,)](h, indices, values, counts, I=1856, BLOCK=2048, num_warps=4)
    partial = torch.empty((8, 2688), dtype=torch.float32, device=h.device)
    _down_partial[(triton.cdiv(2688, 32), 8)](
        indices, values, counts, w2, sd, ids, partial,
        H=2688, I=1856, TOPK=6, BN=32, BK=64,
        num_warps=4, num_stages=2,
    )
    _down_combine[(triton.cdiv(2688, 128),)](
        partial, weights, y, H=2688, M=y.shape[0], TOPK=6, S=2,
        ROUTED_SCALE=routed_scale, BN=128, num_warps=4,
    )
    return True

"""Draft-only BF16 exact-zero compaction; original target kernels are untouched.

Keep every nonzero ReLU2 activation, including NaNs, in original K order.
The compact reduction has a different FP32 tree from the existing drafter.
Supports one to six routed experts plus the original two shared experts;
this includes the current top-three draft policy.
Proposal quality, exact-token acceptance and throughput need remote validation.
"""

import os

import torch

from vllm.triton_utils import tl, triton
from vllm.logger import init_logger

logger = init_logger(__name__)

from .nemotron_h_mtp_kernels import mtp_pdl

_ENABLED = os.environ.get("MTP_RELU2_COMPACT", "1") != "0"


@triton.jit
def _compact(
    h, indices, values, counts, I: tl.constexpr, BLOCK: tl.constexpr,
    launch_pdl: tl.constexpr = False,
):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
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
    launch_pdl: tl.constexpr = False,
):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
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
def _down_slots(
    indices, values, counts, w2, sd, ids, part,
    H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
    BN: tl.constexpr, BK: tl.constexpr,
    launch_pdl: tl.constexpr = False,
):
    """_down's per-slot sums, one program per (column block, slot): part[slot, n] =
    tl.sum(acc, axis=0) with the same tile, chunk order and reduction as _down."""
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    slot = tl.program_id(1)
    lane = tl.arange(0, BK)
    count = tl.load(counts + slot)
    if slot < TOPK:
        base = w2 + tl.load(ids + slot).to(tl.int64) * I * H
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
    tl.store(part + slot * H + n, tl.sum(acc, axis=0), mask=n < H)


@triton.jit
def _down_combine(
    part, weights, y,
    M: tl.constexpr, H: tl.constexpr, ROUTED_SCALE: tl.constexpr,
    TOPK: tl.constexpr, S: tl.constexpr, BN: tl.constexpr,
    launch_pdl: tl.constexpr = False,
    residual_ptr=None, ADD_RES: tl.constexpr = False,
):
    """_down's epilogue over the per-slot sums, in _down's order."""
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    routed = tl.zeros((BN,), tl.float32)
    for j in range(TOPK):
        routed += tl.load(weights + j).to(tl.float32) * tl.load(part + j * H + n, mask=n < H, other=0.0)
    shared = tl.zeros((BN,), tl.float32)
    for s in range(S):
        shared += tl.load(part + (TOPK + s) * H + n, mask=n < H, other=0.0)
    out = (routed * ROUTED_SCALE + shared).to(y.dtype.element_ty)
    for row in range(M):
        value = out
        if ADD_RES:
            residual = tl.load(residual_ptr + row * H + n, mask=n < H, other=0.0)
            # Retain the original MoE BF16 rounding before the separate Add's
            # FP32 sum and BF16 result. The following RMS is still original.
            value = (out.to(tl.float32) + residual.to(tl.float32)).to(y.dtype.element_ty)
        tl.store(y + row * H + n, value, mask=n < H)


# MTP_DOWN_SLOTS=0 runs _down (one program per column block, slots in a loop).
_SLOTS = os.environ.get("MTP_DOWN_SLOTS", "1") != "0"


_ADD_RES = os.environ.get("MTP_DOWN_ADD_RESIDUAL", "1") != "0"
_ADD_OK: dict[tuple[int, float], bool] = {}


@torch.no_grad()
def prepare_residual_add(device, routed_scale):
    """Official remote startup only: compare the folded result with old+BF16 Add."""
    if not (_ADD_RES and _SLOTS and _ENABLED):
        return False
    key = (device.index, float(routed_scale))
    if key in _ADD_OK:
        return _ADD_OK[key]
    _ADD_OK[key] = False
    try:
        gen = torch.Generator(device=device)
        gen.manual_seed(20261025)
        for m in (1, 5, 8):
            for topk in (3, 6):
                for case, scale in enumerate((0.0, 1e-3, 1.0, 32.0)):
                    part = torch.randn(topk + 2, 2688, device=device, generator=gen, dtype=torch.float32) * scale
                    weights = torch.rand(topk, device=device, generator=gen, dtype=torch.float32)
                    weights /= weights.sum()
                    residual = (torch.randn(m, 2688, device=device, generator=gen, dtype=torch.float32) * scale).to(torch.bfloat16)
                    if case == 0:
                        residual[:, ::2] = -0.0
                    old = torch.empty_like(residual)
                    new = torch.empty_like(residual)
                    args = dict(M=m, H=2688, ROUTED_SCALE=routed_scale,
                                TOPK=topk, S=2, BN=256, num_warps=4, launch_pdl=mtp_pdl())
                    _down_combine[(triton.cdiv(2688, 256),)](part, weights, old, **args)
                    reference = old + residual
                    _down_combine[(triton.cdiv(2688, 256),)](
                        part, weights, new, **args, residual_ptr=residual, ADD_RES=True)
                    if not torch.equal(reference.view(torch.uint8), new.view(torch.uint8)):
                        logger.warning("MTP folded residual differs; original Add retained")
                        return False
        _ADD_OK[key] = True
        logger.info("MTP ordered fold plus BF16 residual Add admitted")
    except Exception as exc:  # noqa: BLE001
        logger.warning("MTP folded residual unavailable (%r); original Add retained", exc)
    return _ADD_OK[key]


def can_add_residual(residual, y, routed_scale, topk):
    return (residual is not None and _ADD_OK.get((y.device.index, float(routed_scale)), False)
            and _SLOTS and topk in (3, 6)
            and y.ndim == 2 and y.shape[0] in (1, 5, 8) and y.shape[1] == 2688
            and residual.dtype == y.dtype == torch.bfloat16
            and residual.device == y.device and residual.shape == y.shape
            and residual.is_contiguous() and y.is_contiguous())


def maybe_down(h, packed, ids, weights, y, routed_scale, residual=None):
    """Reuse the existing finite, exact BF16 transpose prepared before capture."""
    if not _ENABLED:
        return False
    w2, sd = packed
    topk = ids.numel()
    if residual is not None and not can_add_residual(residual, y, routed_scale, topk):
        return False
    slots = topk + 2
    if (
        not 1 <= topk <= 6
        or h.dtype != torch.bfloat16 or h.shape != (slots, 1856)
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
    if _SLOTS:
        from .nemotron_h_mtp_inline_compact import maybe_down as maybe_inline_down

        if maybe_inline_down(h, packed, ids, weights, y, routed_scale, residual):
            return True
    indices = torch.empty(h.shape, dtype=torch.int32, device=h.device)
    values = torch.empty_like(h)
    counts = torch.empty(slots, dtype=torch.int32, device=h.device)
    _compact[(slots,)](
        h, indices, values, counts, I=1856, BLOCK=2048, num_warps=4,
        launch_pdl=mtp_pdl(),
    )
    if _SLOTS:
        # The slots' sums in parallel programs, then _down's epilogue: the same
        # values, with five times the programs streaming the gathered rows.
        part = torch.empty((slots, 2688), dtype=torch.float32, device=h.device)
        from .nemotron_h_mtp_down_packed import maybe_slots
        if not maybe_slots(packed, indices, values, counts, ids, part, mtp_pdl()):
            _down_slots[(triton.cdiv(2688, 32), slots)](
                indices, values, counts, w2, sd, ids, part,
                H=2688, I=1856, TOPK=topk, BN=32, BK=64, num_warps=4, num_stages=3,
                launch_pdl=mtp_pdl(),
            )
        _down_combine[(triton.cdiv(2688, 256),)](
            part, weights, y, M=y.shape[0], H=2688, ROUTED_SCALE=routed_scale,
            TOPK=topk, S=2, BN=256, num_warps=4, launch_pdl=mtp_pdl(),
            residual_ptr=residual, ADD_RES=residual is not None,
        )
        return True
    _down[(triton.cdiv(2688, 32),)](
        indices, values, counts, w2, sd, ids, weights, y,
        M=y.shape[0], H=2688, I=1856, ROUTED_SCALE=routed_scale,
        TOPK=topk, S=2, BN=32, BK=64, num_warps=4, num_stages=2,
        launch_pdl=mtp_pdl(),
    )
    return True

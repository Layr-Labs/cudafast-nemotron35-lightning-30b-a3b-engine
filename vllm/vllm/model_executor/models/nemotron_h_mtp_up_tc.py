# SPDX-License-Identifier: Apache-2.0
"""Draft-only native BF16 tensorcore routed/shared up experiment.

All native BF16 weight bytes remain. The FP32 dot reduction changes from
the original SIMT GEMV; proposals/acceptance may change. No target calls here.
No local compilation, syntax execution, import or benchmark was performed.
"""
import os

import torch

from vllm.triton_utils import tl, triton

_ENABLED = os.environ.get("MTP_MOE_TENSORCORE_UP", "1") != "0"


@triton.jit
def _up(
    X, W, SHARED, IDS, OUT,
    SW_E: tl.constexpr, SW_N: tl.constexpr, SS_N: tl.constexpr,
    H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
    BN: tl.constexpr, BK: tl.constexpr,
):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    slot = tl.program_id(1)
    k = tl.arange(0, BK)
    cols = tl.arange(0, 16)
    if slot < TOPK:
        expert = tl.load(IDS + slot).to(tl.int64)
        base = W + expert * SW_E + n[:, None].to(tl.int64) * SW_N
    else:
        sn = (slot - TOPK) * I + n
        base = SHARED + sn[:, None].to(tl.int64) * SS_N
    acc = tl.zeros((BN, 16), dtype=tl.float32)
    # Replicate the same input vector across the MMA's token columns. One
    # output column is retained. This trades extra tensorcore arithmetic for
    # coalesced dense BF16 loads and a much smaller FP32 accumulator tile.
    for k0 in range(0, H, BK):
        kk = k0 + k
        w = tl.load(base + kk[None, :],
                    mask=(n[:, None] < I) & (kk[None, :] < H), other=0.0,
                    eviction_policy="evict_first")
        x = tl.load(X + kk, mask=kk < H, other=0.0)
        xr = tl.broadcast_to(x[:, None], (BK, 16))
        acc = tl.dot(w, xr, acc, allow_tf32=False, out_dtype=tl.float32)
    # Select column0 rather than combining independently computed columns.
    value = tl.sum(tl.where(cols[None, :] == 0, acc, 0.0), axis=1)
    value = tl.maximum(value, 0.0)
    value = value * value
    tl.store(OUT + slot * I + n, value.to(OUT.dtype.element_ty), mask=n < I)


def maybe_up(x, w, shared, ids, h):
    if not _ENABLED:
        return False
    if (x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16
            or shared.dtype != torch.bfloat16 or h.dtype != torch.bfloat16
            or not x.is_cuda or w.device != x.device or shared.device != x.device
            or ids.device != x.device or h.device != x.device
            or tuple(x.shape) != (1, 2688) or x.stride(-1) != 1
            or tuple(w.shape) != (128, 1856, 2688) or not w.is_contiguous()
            or tuple(shared.shape) != (3712, 2688) or not shared.is_contiguous()
            or ids.numel() != 6 or ids.dtype not in (torch.int32, torch.int64)
            or not ids.is_contiguous()
            or tuple(h.shape) != (8, 1856) or not h.is_contiguous()):
        return False
    _up[(triton.cdiv(1856, 32), 8)](
        x, w, shared, ids, h,
        SW_E=w.stride(0), SW_N=w.stride(1), SS_N=shared.stride(0),
        H=2688, I=1856, TOPK=6, BN=32, BK=128,
        num_warps=4, num_stages=3,
    )
    return True

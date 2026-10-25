"""Gluon small-M GEMM for the lossless 12-bit attention projections."""

import torch
import triton

from vllm.model_executor.models.nemotron_h_mtp_packed import packed_small

ROWS = 5
CONFIGS = ((64, 4),)


def run(x, sm, ex, base, ri, raw, n: int, k: int, cfg: tuple[int, int]) -> torch.Tensor:
    """y = x @ W.T for W stored in the planes (raw is the int16 view of the uint16 plane)."""
    bn, warps = cfg
    y = x.new_empty((x.shape[0], n))
    gate = torch.zeros(1, dtype=torch.int32, device=x.device)
    packed_small[(triton.cdiv(n, bn),)](
        x, sm, ex, base, ri, raw, y,
        x.shape[0], n, k, x.stride(0), y.stride(0), gate,
        BM=16, BN=bn, BK=128, GATED=False,
        num_warps=warps, num_stages=4, launch_pdl=False,
    )
    return y

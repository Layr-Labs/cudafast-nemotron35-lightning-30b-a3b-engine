# SPDX-License-Identifier: Apache-2.0
"""Small-M FP32 router for GB10, without reduced-precision dot products.

Each CTA streams two expert rows once and reuses them for all four token
lanes. FP32 FMA chains and a fixed reduction replace the generic GEMM.
The summation order differs from cuBLAS: exact-token validation is required.
No weights are repacked, rounded, quantized, or retained between requests.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _small_fp32_router_kernel(
    X,
    W,
    Y,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    XM: tl.constexpr,
    XK: tl.constexpr,
    WN: tl.constexpr,
    WK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    experts = tl.program_id(0) * BN + tl.arange(0, BN)
    rows = tl.arange(0, BM)
    lanes = tl.arange(0, BK)
    acc = tl.zeros((BM, BN, BK), dtype=tl.float32)
    for start in range(tl.cdiv(K, BK)):
        cols = start * BK + lanes
        # This load has no token-row axis: four token lanes reuse one stream.
        w = tl.load(
            W + experts[:, None] * WN + cols[None, :] * WK,
            mask=(experts[:, None] < N) & (cols[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        x = tl.load(
            X + rows[:, None] * XM + cols[None, :] * XK,
            mask=(rows[:, None] < M) & (cols[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        acc = tl.fma(x[:, None, :], w[None, :, :], acc)
    result = tl.sum(acc, axis=2)
    tl.store(
        Y + rows[:, None] * N + experts[None, :],
        result,
        mask=(rows[:, None] < M) & (experts[None, :] < N),
    )


def small_fp32_router_supported(x, weight):
    return (
        x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.ndim == weight.ndim == 2
        and 1 <= x.shape[0] <= 4
        and x.shape[1] == 2688
        and weight.shape == (128, 2688)
        and weight.dtype == torch.float32
        and x.dtype in (torch.bfloat16, torch.float32)
        and x.stride(1) == weight.stride(1) == 1
        and x.stride(0) >= 2688
        and weight.stride(0) >= 2688
    )


def small_fp32_router(x, weight):
    # Device-family selection belongs to GateLinear.__init__; this shape
    # fallback keeps the custom-op contract valid for larger/fake batches.
    if not small_fp32_router_supported(x, weight):
        return torch.nn.functional.linear(x.to(torch.float32), weight.to(torch.float32))
    out = torch.empty((x.shape[0], 128), device=x.device, dtype=torch.float32)
    _small_fp32_router_kernel[(64,)](
        x,
        weight,
        out,
        M=x.shape[0],
        K=2688,
        N=128,
        XM=x.stride(0),
        XK=x.stride(1),
        WN=weight.stride(0),
        WK=weight.stride(1),
        BM=4,
        BN=2,
        BK=512,
        num_warps=4,
        num_stages=2,
    )
    return out


def small_fp32_router_fake(x, weight):
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.float32)


direct_register_custom_op(
    op_name="gb10_small_fp32_router",
    op_func=small_fp32_router,
    fake_impl=small_fp32_router_fake,
)

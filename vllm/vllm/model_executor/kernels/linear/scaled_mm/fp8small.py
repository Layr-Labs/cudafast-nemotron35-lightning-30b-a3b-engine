# SPDX-License-Identifier: Apache-2.0
"""Preserve a runtime small-M dispatch through torch.compile and CUDA graphs."""

import torch

from vllm import _custom_ops as ops
from vllm.utils.torch_utils import direct_register_custom_op


def prepare_weight(weight: torch.Tensor) -> torch.Tensor | None:
    try:
        import fp8small
    except ImportError:
        return None
    return fp8small.prepare(weight)


def _scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    packed: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
) -> torch.Tensor:
    # This guard executes on real tensor metadata inside the custom op, so a
    # symbolic compilation range can contain both prefill and small decode M.
    if (
        1 <= a.shape[0] <= 8
        and a.dtype == b.dtype == torch.float8_e4m3fn
        and a.is_contiguous()
        and a.data_ptr() % 16 == 0
        and a.device == b.device == packed.device == sa.device == sb.device
        and sa.dtype == sb.dtype == torch.float32
        and sa.numel() == sb.numel() == 1
        and sa.is_contiguous()
        and sb.is_contiguous()
        and a.shape[1] == packed.shape[1]
    ):
        import fp8small

        return fp8small.gemm(a, packed, sa, sb)
    return ops.cutlass_scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)


def _scaled_mm_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    packed: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.bfloat16)


direct_register_custom_op("fp8small_scaled_mm", _scaled_mm, fake_impl=_scaled_mm_fake)

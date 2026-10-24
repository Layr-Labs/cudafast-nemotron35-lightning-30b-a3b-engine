# SPDX-License-Identifier: Apache-2.0
"""Preserve a runtime small-M dispatch through torch.compile and CUDA graphs."""

import torch

from vllm import _custom_ops as ops
from vllm.utils.torch_utils import direct_register_custom_op

try:
    import l2pf as _l2pf
except ImportError:
    _l2pf = None


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

        out = fp8small.gemm(a, packed, sa, sb)
    else:
        out = ops.cutlass_scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    # LOCAL PATCH (cudafast-nemotron35): rejoin an MoE-window L2 prefetch the
    # previous MoE layer deferred (l2pf.MOE_LAZY, fused_humming_moe.py) once this
    # GEMM (the next Mamba layer's in_proj) is queued. Moves no value.
    if _l2pf is not None and getattr(_l2pf, "_pending", None):
        _l2pf.join_pending(a.device)
    # LOCAL PATCH (cudafast-nemotron35): a Mamba out_proj followed by an MoE layer
    # prefetches that layer's router weight now; the router op rejoins (l2pf.AHEAD_ROUTER).
    if _l2pf is not None and a.shape[0] <= 8 and getattr(_l2pf, "AHEAD_ROUTER", None):
        _l2pf.prefetch_router_after(packed)
    return out


def _scaled_mm_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    packed: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.bfloat16)


direct_register_custom_op("fp8small_scaled_mm", _scaled_mm, fake_impl=_scaled_mm_fake)

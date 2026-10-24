# SPDX-License-Identifier: Apache-2.0
"""Shared ReLU2 up epilogue generates the same-call sparse-down mask."""
import os

import torch

from vllm.utils.torch_utils import direct_register_custom_op

try:
    import marlinx as _marlinx
except ImportError:
    _marlinx = None

_ENABLED = (os.environ.get("MARLINX_FUSED_MASK", "1") != "0"
            and os.environ.get("MARLINX_SHARED_RELU2", "1") != "0")


def _shared_mlp_mask(
    x: torch.Tensor,
    up_weight: torch.Tensor, up_scale: torch.Tensor, up_global: torch.Tensor,
    down_weight: torch.Tensor, down_zs: torch.Tensor,
    down_scale: torch.Tensor, down_global: torch.Tensor,
) -> torch.Tensor:
    # Runtime metadata/ctypes stay within this op; no workspace/input mutation.
    current = x.contiguous()
    if current.data_ptr() % 16:
        current = current.clone()
    hidden, mask = _marlinx.gemm_relu2_mask(current, up_weight, up_scale, up_global)
    return _marlinx.gemm_zs_mask(hidden, down_weight, down_zs,
                                down_scale, down_global, mask)


def _shared_mlp_mask_fake(
    x: torch.Tensor,
    up_weight: torch.Tensor, up_scale: torch.Tensor, up_global: torch.Tensor,
    down_weight: torch.Tensor, down_zs: torch.Tensor,
    down_scale: torch.Tensor, down_global: torch.Tensor,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], 2688))


direct_register_custom_op(
    op_name="nemotron_shared_mlp_mask",
    op_func=_shared_mlp_mask,
    fake_impl=_shared_mlp_mask_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def maybe_shared_mlp_mask(up, down, x):
    if (not _ENABLED or _marlinx is None
            or not getattr(up, "_marlinx_mask_ok", False)
            or not getattr(down, "_marlinx_mask_ok", False)
            or getattr(down, "_marlinx_zs", None) is None
            or getattr(up, "tp_size", 1) != 1 or getattr(down, "tp_size", 1) != 1
            or getattr(up, "bias", None) is not None or getattr(down, "bias", None) is not None
            or up.output_size_per_partition != 3712 or up.input_size_per_partition != 2688
            or down.output_size_per_partition != 2688 or down.input_size_per_partition != 3712
            or x.ndim != 2 or not 1 <= x.shape[0] <= 8 or x.shape[1] != 2688
            or x.dtype != torch.bfloat16 or not x.is_cuda
            or x.device != up.weight.device or x.device != down.weight.device):
        return None
    return torch.ops.vllm.nemotron_shared_mlp_mask(
        x, up.weight, up.weight_scale, up.weight_global_scale,
        down.weight, down._marlinx_zs, down.weight_scale, down.weight_global_scale,
    )

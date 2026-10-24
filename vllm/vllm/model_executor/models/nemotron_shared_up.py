# SPDX-License-Identifier: Apache-2.0
"""Marlin-exact shared up projection with original BF16 rounding and ReLU2."""

import os

import torch

from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
)
from vllm.utils.torch_utils import direct_register_custom_op

try:
    import marlinx as _marlinx
except ImportError:
    _marlinx = None

_ENABLED = os.environ.get("MARLINX_SHARED_RELU2", "1") != "0"


def _shared_up_relu2(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    global_scale: torch.Tensor,
    workspace: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    if _marlinx is not None and hasattr(_marlinx, "maybe_apply_relu2"):
        out = _marlinx.maybe_apply_relu2(
            x, weight, weight_scale, global_scale, n, k
        )
        if out is not None:
            return out
    # Complete semantic fallback inside the opaque op: never skip activation
    # merely because the optional native library or runtime guard declined.
    out = apply_fp4_marlin_linear(
        input=x,
        weight=weight,
        weight_scale=weight_scale,
        weight_global_scale=global_scale,
        workspace=workspace,
        size_n=n,
        size_k=k,
        bias=None,
    )
    return torch.square(torch.relu(out))


def _shared_up_relu2_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    global_scale: torch.Tensor,
    workspace: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], n))


direct_register_custom_op(
    op_name="nemotron_shared_up_relu2",
    op_func=_shared_up_relu2,
    mutates_args=["workspace"],
    fake_impl=_shared_up_relu2_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def maybe_shared_up_relu2(layer, x: torch.Tensor) -> torch.Tensor | None:
    """Eligibility is static metadata; ctypes/dispatch stay inside the op."""
    if not _ENABLED or not getattr(layer, "_marlinx_ok", False):
        return None
    if (
        getattr(layer, "tp_size", 1) != 1
        or getattr(layer, "bias", None) is not None
        or layer.output_size_per_partition != 3712
        or layer.input_size_per_partition != 2688
        or x.ndim != 2
        or x.shape[1] != 2688
        or not 1 <= x.shape[0] <= 8
        or x.dtype != torch.bfloat16
        or not x.is_cuda
    ):
        return None
    return torch.ops.vllm.nemotron_shared_up_relu2(
        x, layer.weight, layer.weight_scale, layer.weight_global_scale,
        layer.workspace, 3712, 2688,
    )

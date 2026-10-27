# SPDX-License-Identifier: Apache-2.0
"""Eight-group native-order gated RMSNorm to static E4M3 input activation.

This is a guarded producer for the existing QuantizedActivation interface.
It preserves both native BF16 rounding boundaries; runtime equivalence and
performance are established only by official remote validation.
"""

import os
import sys

import torch

from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8StaticTensorSym
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_ENABLED = os.environ.get("VLLM_MAMBA_GATED_NORM_FP8", "1") != "0"
_PDL = None


def _pdl_enabled():
    global _PDL
    if _PDL is None:
        from vllm.platforms import current_platform

        _PDL = (os.environ.get("VLLM_MAMBA_GATED_NORM_PDL", "1") != "0"
                and current_platform.is_arch_support_pdl())
    return _PDL


@triton.jit
def _gated_norm_fp8_kernel(
    X, GATE, WEIGHT, SCALE, OUT,
    SX: tl.constexpr, SG: tl.constexpr, SO: tl.constexpr,
    EPS: tl.constexpr, GROUP: tl.constexpr,
    verify_rows_ptr=None, PREFIX: tl.constexpr = False,
    launch_pdl: tl.constexpr = False,
):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)
    group = tl.program_id(1)
    columns = group * GROUP + tl.arange(0, GROUP)
    if PREFIX:
        limit = tl.maximum(tl.load(verify_rows_ptr), 1)
        if row >= limit:
            # Invalid causal suffix, explicitly initialized without reading
            # gate/SSM/norm-weight or doing the grouped reduction.
            zero = tl.full((GROUP,), 0.0, tl.float32)
            tl.store(OUT + row * SO + columns, zero.to(OUT.dtype.element_ty))
            return
    x = tl.load(X + row * SX + columns).to(tl.float32)
    gate = tl.load(GATE + row * SG + columns).to(tl.float32)
    # Native source: x * silu(gate.float()), grouped mean-square, rsqrt.
    silu = gate * tl.sigmoid(gate)
    value = x * silu
    variance = tl.sum(value * value, axis=0) / GROUP
    value = value * tl.rsqrt(variance + EPS)
    # Native source rounds BEFORE the norm-weight multiplication and again
    # when that BF16 multiplication returns its BF16 tensor.
    rounded = value.to(tl.bfloat16).to(tl.float32)
    weight = tl.load(WEIGHT + columns).to(tl.float32)
    normed = (rounded * weight).to(tl.bfloat16).to(tl.float32)
    scale = tl.load(SCALE).to(tl.float32)
    quantized = normed * tl.div_rn(1.0, scale)
    # Ordered comparisons retain NaNs rather than replacing them with a bound.
    quantized = tl.where(quantized > 448.0, 448.0, quantized)
    quantized = tl.where(quantized < -448.0, -448.0, quantized)
    tl.store(OUT + row * SO + columns, quantized.to(OUT.dtype.element_ty))


def _gated_norm_fp8(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    verify_rows = None
    if x.shape[0] in (5, 6) and os.environ.get("VLLM_MAMBA_GATED_NORM_PREFIX", "1") != "0":
        smallk = sys.modules.get("smallk")
        if smallk is not None and hasattr(smallk, "live_verify_rows"):
            verify_rows = smallk.live_verify_rows(x.device)
    _gated_norm_fp8_kernel[(x.shape[0], 8)](
        x, gate, weight, scale, out,
        SX=x.stride(0), SG=gate.stride(0), SO=out.stride(0),
        EPS=eps, GROUP=512, num_warps=4, enable_fp_fusion=False,
        verify_rows_ptr=verify_rows, PREFIX=verify_rows is not None,
        launch_pdl=_pdl_enabled(),
    )
    return out


def _gated_norm_fp8_fake(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)


direct_register_custom_op(
    op_name="mamba_gated_norm_static_fp8",
    op_func=_gated_norm_fp8,
    mutates_args=[],
    fake_impl=_gated_norm_fp8_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def maybe_gated_norm_fp8(norm, projection, x: torch.Tensor, gate: torch.Tensor):
    """Return a compatible prequantized activation, else the original call runs."""
    if not _ENABLED:
        return None
    kernel = getattr(getattr(projection, "quant_method", None), "fp8_linear", None)
    if kernel is None or type(kernel).__name__ != "CutlassFP8ScaledMMLinearKernel":
        return None
    if kernel.input_quant_key() != kFp8StaticTensorSym:
        return None
    if (
        norm.tp_size != 1
        or projection.tp_size != 1
        or not projection.input_is_parallel
        or getattr(projection, "bias", None) is not None
        or not norm.use_rms_norm
        or norm.n_groups != 8
        or norm.group_size != 512
        or kernel.quant_fp8.num_token_padding is not None
        or kernel.fp8_dtype != torch.float8_e4m3fn
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or gate.dtype != x.dtype
        or gate.device != x.device
        or x.ndim != 2
        or x.shape[1] != 4096
        or not 1 <= x.shape[0] <= 8
        or gate.shape != x.shape
        or x.stride(-1) != 1
        or gate.stride(-1) != 1
        or norm.weight.dtype != torch.bfloat16
        or norm.weight.device != x.device
        or norm.weight.shape != (4096,)
        or not norm.weight.is_contiguous()
    ):
        return None
    names = kernel.layer_param_names
    scale = getattr(projection, names[2], None)
    upper = getattr(projection, names[3], None)
    if (
        scale is None
        or upper is not None
        or scale.dtype != torch.float32
        or scale.numel() != 1
        or not scale.is_contiguous()
        or scale.device != x.device
    ):
        return None
    data = torch.ops.vllm.mamba_gated_norm_static_fp8(
        x, gate, norm.weight, scale, norm.variance_epsilon
    )
    return QuantizedActivation(
        data=data, scale=scale, orig_dtype=x.dtype, orig_shape=x.shape,
        quant_key=kFp8StaticTensorSym,
    )

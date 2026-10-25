# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    is_fp4_marlin_supported,
    prepare_fp4_layer_for_marlin,
)

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)

# marlinx (harness/kernels/marlinx): a Marlin-exact NVFP4 GEMM for M <= 8 on the
# shared-expert and lm_head shapes. MARLINX=0 disables it; when the package is
# not built, or a call is outside its guard, stock Marlin runs.
try:
    import marlinx as _marlinx
except ImportError:
    _marlinx = None

# shco (harness/kernels/shco): shx, the Marlin-exact shared expert at the prefill
# M (496 rows) with a footprint that shares SMs with the routed up projection.
# prepare() stages tables; the model post-load hook admits the whole cohort.
try:
    import shco as _shco
except ImportError:
    _shco = None


class MarlinNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 weight-only GEMM via Marlin (W4A16)."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if is_fp4_marlin_supported():
            return True, None
        return False, "Marlin FP4 not available"

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        logger.warning_once(
            "Your GPU does not have native support for FP4 computation but "
            "FP4 quantization is being used. Weight-only FP4 compression "
            "will be used leveraging the Marlin kernel. This may degrade "
            "performance for compute-heavy workloads."
        )
        prepare_fp4_layer_for_marlin(layer)
        if _marlinx is not None and _marlinx.prepare(layer):
            logger.info_once("marlinx: Marlin-exact NVFP4 GEMM enabled for M <= 8")
        if _shco is not None and _shco.prepare(layer):
            logger.info_once("shx: shared-expert tables prepared for remote cohort admission")

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is None and _marlinx is not None:
            out = _marlinx.maybe_apply(layer, x)
            if out is not None:
                return out
        return apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )

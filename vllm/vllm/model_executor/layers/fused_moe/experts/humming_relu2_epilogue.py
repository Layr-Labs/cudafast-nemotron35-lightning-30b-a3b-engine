# SPDX-License-Identifier: Apache-2.0
"""LOCAL PATCH: fuse expert-up ReLU squared after the original BF16 rounding.

No installed Humming file is edited. A derived kernel adds only an epilogue
wrapper before the pinned header is included. The wrapper calls the original
epilogue first, so the intermediate BF16 value is retained exactly.
"""

import json
import torch
from humming import dtypes, ops
from humming.kernel.humming import HummingKernel

_EPILOGUE = r"""
#include <cuda_bf16.h>
#define EpilogueArithmetic UnfusedEpilogueArithmetic
#include <humming/arith/epilogue_arith.cuh>
#undef EpilogueArithmetic

template <class MmaOpClass, class BlockShape, class WarpShape,
          class ElementA, class ElementB, class ElementC, class ElementBS,
          class LayerConfig, class TuningConfig>
class EpilogueArithmetic : public UnfusedEpilogueArithmetic<
    MmaOpClass, BlockShape, WarpShape, ElementA, ElementB, ElementC, ElementBS,
    LayerConfig, TuningConfig> {
  using Base = UnfusedEpilogueArithmetic<
      MmaOpClass, BlockShape, WarpShape, ElementA, ElementB, ElementC, ElementBS,
      LayerConfig, TuningConfig>;
 public:
  static_assert(ElementC::kBits == 16 && MmaOpClass::kCTypeBits == 32);
  static_assert(!TuningConfig::kUseStreamK);
  CUDA_INLINE void may_apply_on_smem_write(uint32_t &regs, uint32_t row, uint32_t col) {
    Base::may_apply_on_smem_write(regs, row, col);
    // regs already contains the original BF16 output. Use explicit non-FTZ
    // operations even if Humming enables fast math for unrelated arithmetic.
    uint32_t squared;
    asm volatile(
        "{ .reg .b16 lo, hi; .reg .f32 x, y;\n"
        "mov.b32 {lo, hi}, %1;\n"
        "cvt.f32.bf16 x, lo; cvt.f32.bf16 y, hi;\n"
        "max.NaN.f32 x, x, 0f00000000; max.NaN.f32 y, y, 0f00000000;\n"
        "mul.rn.f32 x, x, x; mul.rn.f32 y, y, y;\n"
        "cvt.rn.bf16x2.f32 %0, y, x; }\n"
        : "=r"(squared) : "r"(regs));
    regs = squared;
  }
};
"""


class Relu2HummingKernel(HummingKernel):
    def prepare(self):
        marker = "#if "
        at = self.code.index(marker)
        self.code = self.code[:at] + _EPILOGUE + self.code[at:]
        super().prepare()


_CACHE = {}


def prepare(meta, compute_config, tuning_config):
    key = (meta.to_str(), compute_config, tuning_config)
    if key in _CACHE:
        return _CACHE[key]
    assert meta.a_dtype == dtypes.bfloat16 and meta.c_dtype == dtypes.bfloat16
    assert meta.b_dtype.num_bits == 4
    lc = json.loads(meta.to_str())
    lc.pop("sublayer_name", None)
    cc = json.loads(compute_config)
    assert cc["gemm_type"] == "indexed" and cc["use_batch_invariant"]
    assert not cc.get("use_f16_accum", False)
    tc = json.loads(tuning_config)
    if isinstance(tc, dict):
        tc = [[0, 1 << 30, tc]]
    result = []
    for lo, hi, config in tc:
        assert config["block_shape"][2] == config["warp_shape"][2] == 32
        assert not config.get("use_stream_k", False)
        assert not config.get("use_f16_accum", False)
        params = lc | cc | config
        num_sms = params.pop("num_sms", 0)
        kernel = Relu2HummingKernel(**params)
        kernel.load_cubin()
        result.extend([lo, hi, kernel.kernel_id, num_sms])
    configs = torch.tensor(result, dtype=torch.int64, device="cpu")
    _CACHE[key] = configs
    return configs


def forward_relu2(
    layer,
    inputs,
    outputs,
    compute_config,
    tuning_config,
    sorted_ids,
    expert_ids,
    num_tokens_padded,
    top_k,
    valid_shape_m,
    input_scale=None,
):
    meta = layer.humming_metas["w13"]
    assert inputs.dtype == outputs.dtype == torch.bfloat16 and input_scale is None
    assert getattr(layer, meta.bias_name, None) is None
    configs = prepare(meta, compute_config, tuning_config)
    return ops.launch_kernel(
        configs=configs,
        inputs=inputs,
        weight=getattr(layer, meta.weight_name),
        outputs=outputs,
        input_scale=None,
        weight_scale=getattr(layer, meta.weight_scale_name, None),
        zero_point=getattr(layer, meta.zero_point_name, None),
        bias=None,
        global_scale=getattr(layer, meta.global_scale_name, None),
        sorted_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_padded=num_tokens_padded,
        locks=layer.locks,
        top_k=top_k,
        valid_shape_m=valid_shape_m,
    )

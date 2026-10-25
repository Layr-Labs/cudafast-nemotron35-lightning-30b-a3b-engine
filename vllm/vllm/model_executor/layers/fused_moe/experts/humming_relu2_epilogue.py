# SPDX-License-Identifier: Apache-2.0
"""LOCAL PATCH: fuse expert-up ReLU squared after the original BF16 rounding.

No installed Humming file is edited. A derived kernel adds only an epilogue
wrapper before the pinned header is included. The wrapper calls the original
epilogue first, so the intermediate BF16 value is retained exactly.
"""

import json
import os
from pathlib import Path

import humming
import torch
from humming import dtypes, ops
from humming.kernel.humming import HummingKernel
from vllm.logger import init_logger

logger = init_logger(__name__)

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

# Separate class AND config cache: KernelRuntime interns by class/config.
_EVICT_CACHE = {}
_EVICT_FAILED = False
# The evict-first hint serves humming's prefill-sized bucket: the tuning-config range(s) whose
# block_m is 64 (above the block_m-16 decode and small-call ranges; humming_tuned_moe.py's last
# w13 range, valid_shape_m > 384), the same calls pzskip serves for w2. Keyed on the config.
_EVICT_BLOCK_M = 64
_EVICT_RANGES = {}


def _tuning_list(tuning_config):
    tc = json.loads(tuning_config)
    return [[0, 1 << 30, tc]] if isinstance(tc, dict) else tc


def _evict_bucket(tuning_config, valid_shape_m):
    """True when valid_shape_m dispatches (lo < m <= hi) to a block_m-64 range."""
    ranges = _EVICT_RANGES.get(tuning_config)
    if ranges is None:
        ranges = tuple((lo, hi) for lo, hi, config in _tuning_list(tuning_config)
                       if config["block_shape"][0] == _EVICT_BLOCK_M)
        _EVICT_RANGES[tuning_config] = ranges
    return any(lo < valid_shape_m <= hi for lo, hi in ranges)
_EVICT_PREFIX = r"""
#include <humming/utils/all.cuh>
template<bool C, typename T>
CUDA_INLINE void evf_load(const T *g, T *s, bool pred = true) {
  if constexpr (!C || sizeof(T) != 16) {
    if (pred) legacy_load<C>(g, s);
  } else {
    uint32_t sm = cast_smem_ptr_to_uint(s);
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;"
                 : "=l"(policy));
    asm volatile("{.reg .pred p; setp.ne.s32 p,%0,0; "
                 "@p cp.async.cg.shared.global.L2::cache_hint [%1],[%2],16,%3;}"
                 :: "r"((uint32_t)pred), "r"(sm), "l"(g), "l"(policy)
                 : "memory");
  }
}
"""


class EvictFirstRelu2HummingKernel(Relu2HummingKernel):
    def prepare(self):
        assert self.use_cp_async and not self.use_tma_b and not self.use_tma_bs
        # Keep the pinned 2D loader's loop, addresses and predicates verbatim.
        # Include only B/BS under the replacement; indexed A remains stock.
        header = (Path(humming.__file__).parent / "include/humming/utils/ptx/legacy_load.cuh").read_text()
        start = header.index("template <bool use_cp_async, uint32_t num_int4s, uint32_t threads, uint32_t gmem_stride")
        end = header.index("CUDA_INLINE void cp_async_commit_group", start)
        body = header[start:end].replace("legacy_load_2d", "evf_load_2d")
        body = body.replace("legacy_load_pred<use_cp_async>", "evf_load<use_cp_async>")
        body = body.replace("legacy_load<use_cp_async>", "evf_load<use_cp_async>")
        assert "legacy_load<" not in body and "legacy_load_pred<" not in body
        code = _EVICT_PREFIX + body + r"""
#define legacy_load_2d evf_load_2d
#include <humming/memory/g2s_loader/loader_b.cuh>
#include <humming/memory/g2s_loader/loader_bs.cuh>
#undef legacy_load_2d
"""
        at = self.code.index("#if ")
        self.code = self.code[:at] + code + self.code[at:]
        super().prepare()


def _prepare_stock(meta, compute_config, tuning_config):
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


def prepare(meta, compute_config, tuning_config, evict_first=False):
    global _EVICT_FAILED
    stock = _prepare_stock(meta, compute_config, tuning_config)
    if not evict_first or _EVICT_FAILED:
        return stock
    key = (meta.to_str(), compute_config, tuning_config)
    if key in _EVICT_CACHE:
        return _EVICT_CACHE[key]
    try:
        values = stock.tolist()
        tc = json.loads(tuning_config)
        if isinstance(tc, dict):
            tc = [[0, 1 << 30, tc]]
        lc = json.loads(meta.to_str())
        lc.pop("sublayer_name", None)
        changed = 0
        for i, (lo, hi, config) in enumerate(tc):
            if config["block_shape"][0] != _EVICT_BLOCK_M:
                continue
            original = HummingKernel._id2kernel[values[4 * i + 2]]
            assert tuple(original.problem_shape[1:]) == (1856, 2688)
            assert config["block_shape"] == [64, 64, 32]
            params = lc | json.loads(compute_config) | config
            params.pop("num_sms", None)
            kernel = EvictFirstRelu2HummingKernel(**params)
            kernel.load_cubin()
            # Exclude the stock-cubin singleton no-op before admitting the hint.
            assert type(kernel) is EvictFirstRelu2HummingKernel
            assert "createpolicy.fractional.L2::evict_first" in kernel.code
            assert "cp.async.cg.shared.global.L2::cache_hint" in kernel.code
            assert kernel.kernel_filename != original.kernel_filename
            assert kernel.kernel_id != original.kernel_id
            values[4 * i + 2] = kernel.kernel_id
            changed += 1
        assert changed == 1
        configs = torch.tensor(values, dtype=torch.int64, device="cpu")
        _EVICT_CACHE[key] = configs
        return configs
    except Exception as exc:
        _EVICT_FAILED = True
        logger.warning_once("humming evict-first unavailable; using stock: %s", exc)
        return stock


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
    configs = prepare(
        meta, compute_config, tuning_config,
        evict_first=(top_k == 6
                     and _evict_bucket(tuning_config, valid_shape_m)
                     and os.environ.get("HUMMING_EVICT_FIRST", "1") != "0"),
    )
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

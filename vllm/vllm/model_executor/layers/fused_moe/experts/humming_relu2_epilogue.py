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


# LOCAL PATCH (cudafast-nemotron35, bw-humming): address-local claim-when-free
# tile scheduler for the w13 kernels. Upstream humming hands CTA c the output
# tiles c, c + G, c + 2G, ... of the persistent grid G (a static schedule: a CTA
# that starts late, e.g. beside the shared expert, or that drew more tiles,
# leaves a tail). Here every CTA claims its next tile from one atomic counter
# when it becomes free, in the same global order (m block major, i.e. expert by
# expert in ascending weight address, n block minor, so the CTAs in flight
# stream the same K rows of a few adjacent experts). Only WHICH CTA computes a
# tile and WHEN changes: every tile is still computed by exactly one CTA over
# all K blocks in ascending order with the same epilogue (no stream-K), so each
# output element's fp32 chain and its rounding are unchanged.
# The counters are module globals: the last CTA to finish claiming resets them,
# so every launch (eager or graph replay) starts from zero.
_DYN_SCHEDULER = r"""
#define Scheduler HummingStaticScheduler
#include <humming/scheduler.cuh>
#undef Scheduler

__device__ uint32_t hm_dyn_claim = 0;
__device__ uint32_t hm_dyn_done = 0;

template <
    class SharedStorage,
    class ProblemShape, class BlockShape,
    class LayerConfig, class ComputeConfig, class TuningConfig>
class Scheduler : public HummingStaticScheduler<
    SharedStorage, ProblemShape, BlockShape, LayerConfig, ComputeConfig, TuningConfig> {
  using Base = HummingStaticScheduler<
      SharedStorage, ProblemShape, BlockShape, LayerConfig, ComputeConfig, TuningConfig>;
  static_assert(!TuningConfig::kUseStreamK);
  static_assert(!TuningConfig::kUseWarpSpec);
  static_assert(TuningConfig::kMultiCastSizeA * TuningConfig::kMultiCastSizeB == 1);
  static_assert(ComputeConfig::kGemmType == GemmType::INDEXED);
  static constexpr uint32_t N_BLOCKS = ProblemShape::N / BlockShape::N;
  static constexpr uint32_t K_BLOCKS = ProblemShape::K / BlockShape::K;
  uint32_t total_tiles;

 public:
  CUDA_INLINE
  Scheduler(
      SharedStorage &smem, const void *output_ptr, CUtensorMap *tensor_map_buffer,
      uint32_t shape_m, uint32_t top_k, const uint32_t *row_index_blocks, const uint32_t *expert_ids,
      const uint32_t *num_tokens_padded_ptr, const uint32_t *expert_layout_ptr, bool use_int64_expert_layout)
      : Base(smem, output_ptr, tensor_map_buffer, shape_m, top_k, row_index_blocks, expert_ids,
             num_tokens_padded_ptr, expert_layout_ptr, use_int64_expert_layout) {
    total_tiles = CEIL_DIV(num_tokens_padded_ptr[0], BlockShape::M) * N_BLOCKS;
  }

  CUDA_INLINE
  bool get_next_block() {
    // The claim is broadcast through the row-index staging slot that
    // fetch_moe_index_block refills right after (no static shared memory: it
    // would shift the dynamic shared memory base off its 128-byte alignment).
    uint32_t *slot = this->smem.wr_row_index;
    if (threadIdx.x == 0) slot[0] = atomicAdd(&hm_dyn_claim, 1u);
    __syncthreads();
    const uint32_t tile = slot[0];
    __syncthreads();
    if (tile >= total_tiles) {
      if (threadIdx.x == 0) {
        if (atomicAdd(&hm_dyn_done, 1u) == gridDim.x - 1) {
          atomicExch(&hm_dyn_claim, 0u);
          atomicExch(&hm_dyn_done, 0u);
        }
      }
      return false;
    }
    this->m_block_id = tile / N_BLOCKS;
    this->n_block_id = tile % N_BLOCKS;
    this->k_block_id = 0;
    this->slice_iters = K_BLOCKS;
    this->fetch_moe_index_block();
    return true;
  }
};
"""


def _dyn_enabled() -> bool:
    return os.environ.get("HUMMING_W13_DYN", "1") != "0"


def _dyn_num_sms(lo: int, hi: int, num_sms: int) -> int:
    """Persistent grid (SMs x CTAs per SM) of the claim-when-free kernels: the
    prefill entry may take every SM's slots, since a CTA that only becomes
    resident once the shared expert's CTA leaves just claims fewer tiles."""
    if hi >= (1 << 29):
        return int(os.environ.get("HUMMING_W13_DYN_PF_SMS", "48"))
    return int(os.environ.get("HUMMING_W13_DYN_DEC_SMS", "44"))


class Relu2HummingKernel(HummingKernel):
    dyn_scheduler = False

    def prepare(self):
        marker = "#if "
        at = self.code.index(marker)
        extra = _DYN_SCHEDULER if self.dyn_scheduler else ""
        self.code = self.code[:at] + extra + _EPILOGUE + self.code[at:]
        super().prepare()


class DynRelu2HummingKernel(Relu2HummingKernel):
    dyn_scheduler = True


_CACHE = {}

# Separate class AND config cache: KernelRuntime interns by class/config.
_EVICT_CACHE = {}
# Scheduler variants (_variant_key) whose evict-first kernel failed: stock retained.
_EVICT_FAILED = set()
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


# Both the claim-when-free scheduler and the evict-first B/BS loader: the
# evict-first loader text is inserted first, then Relu2HummingKernel.prepare
# adds the scheduler and epilogue right before the kernel header (each piece
# is the same text as in its single-feature class). Its own class name keeps
# KernelRuntime's interning (class name + config) apart from the other three.
class DynEvictFirstRelu2HummingKernel(EvictFirstRelu2HummingKernel):
    dyn_scheduler = True


def _w13_k128_enabled():
    return os.environ.get("HUMMING_W13_K128", "1") != "0"


def _w13_k128_params(lo, hi, params):
    # Only the five-to-eight-token verify bucket. The K16 MMA chain stays
    # serial and ascending; four adjacent K32 payloads share one pipeline wait.
    if (_w13_k128_enabled() and (lo, hi) == (24, 48)
            and params["block_shape"] == [16, 64, 32]
            and params["warp_shape"] == [16, 32, 32]
            and params["num_stages"] == 8):
        return dict(params, block_shape=[16, 64, 128],
                    warp_shape=[16, 32, 128], num_stages=2)
    return params


def _variant_key(dyn):
    """Cache-key part of the scheduler variant: the claim-when-free kernels'
    persistent grids come from the environment, so they are part of it."""
    if not dyn:
        return (False, _w13_k128_enabled())
    return (True, _dyn_num_sms(0, 1 << 30, 0), _dyn_num_sms(0, 1, 0),
            _w13_k128_enabled())


def _prepare_stock(meta, compute_config, tuning_config, dyn):
    key = (meta.to_str(), compute_config, tuning_config) + _variant_key(dyn)
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
        params = _w13_k128_params(lo, hi, lc | cc | config)
        num_sms = params.pop("num_sms", 0)
        if dyn:
            num_sms = _dyn_num_sms(lo, hi, num_sms)
        kernel = (DynRelu2HummingKernel if dyn else Relu2HummingKernel)(**params)
        kernel.load_cubin()
        result.extend([lo, hi, kernel.kernel_id, num_sms])
    configs = torch.tensor(result, dtype=torch.int64, device="cpu")
    _CACHE[key] = configs
    return configs


def prepare(meta, compute_config, tuning_config, evict_first=False, dyn=None):
    dyn = _dyn_enabled() if dyn is None else bool(dyn)
    # The evict-first entry keeps the stock entry's num_sms (with dyn: the
    # claim-when-free prefill grid) and replaces only its kernel id.
    stock = _prepare_stock(meta, compute_config, tuning_config, dyn)
    variant = _variant_key(dyn)
    if not evict_first or variant in _EVICT_FAILED:
        return stock
    key = (meta.to_str(), compute_config, tuning_config) + variant
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
            if not lo < 2976 <= hi:
                continue
            original = HummingKernel._id2kernel[values[4 * i + 2]]
            assert tuple(original.problem_shape[1:]) == (1856, 2688)
            assert config["block_shape"] == [64, 64, 32]
            params = lc | json.loads(compute_config) | config
            params.pop("num_sms", None)
            kernel_cls = (DynEvictFirstRelu2HummingKernel if dyn
                          else EvictFirstRelu2HummingKernel)
            kernel = kernel_cls(**params)
            kernel.load_cubin()
            # Exclude the stock-cubin singleton no-op before admitting the hint.
            assert type(kernel) is kernel_cls
            assert type(original) is (DynRelu2HummingKernel if dyn else Relu2HummingKernel)
            assert ("hm_dyn_claim" in kernel.code) == dyn
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
        _EVICT_FAILED.add(variant)
        logger.warning_once("humming evict-first unavailable (dyn=%s); using stock: %s", dyn, exc)
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
    nvsc=None,
    dyn=None,
):
    meta = layer.humming_metas["w13"]
    assert inputs.dtype == outputs.dtype == torch.bfloat16 and input_scale is None
    assert getattr(layer, meta.bias_name, None) is None
    # LOCAL PATCH (cudafast-nemotron35, nvsc): <= 8 tokens on a repacked layer
    # read 6-bit block scales (humming_nvsc.py); nvsc=False forces stock.
    if nvsc is not False and inputs.shape[0] <= 8:
        from vllm.model_executor.layers.fused_moe.experts import humming_nvsc

        if humming_nvsc.launch(
            layer, inputs, outputs, compute_config, tuning_config, sorted_ids,
            expert_ids, num_tokens_padded, top_k, valid_shape_m, dyn=dyn,
        ):
            return outputs
    configs = prepare(
        meta, compute_config, tuning_config,
        evict_first=(inputs.shape[0] == 496 and top_k == 6
                     and valid_shape_m == 2976
                     and os.environ.get("HUMMING_EVICT_FIRST", "1") != "0"),
        dyn=dyn,
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

# SPDX-License-Identifier: Apache-2.0
"""Experimental prefill w13: two N64 tiles share one indexed A stage.

Only the complete c41 NVSC/PFUP M496 path is eligible. Problem N, weight
packing, K32 warp arithmetic, output row order and BF16 ReLU2 remain fixed.
Compilation/resource checks happen before launch, outside graph capture.
They are not numerical or performance admission; official validation is
required. A launch error is visible and never retried on production output.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
ENABLED = os.environ.get("HUMMING_W13_APAIR", "1") != "0"
_CONFIGS: dict = {}
_FAILED: set = set()
_CLASS = None

# PyPI humming-kernels 0.1.10 publishing commit 4351af3a8fcdce1a8dee50104ba49566af2427fb.
# Check actual installed bytes remotely; the container's header identity was
# not inferred from the release version alone. Mismatch leaves original NVSC.
_PINS = {
    "kernel/humming.cuh": "60f9de86f22082a76628c042d96a4bb2cf67fea4",
    "memory/g2s_loader/loader_a.cuh": "af5a34acd58ed278b75048ae7499295bc31aeb99",
    "memory/g2s_loader/loader_b.cuh": "dd6fa2dc1eed92c9001249b7a465d1e6363de846",
    "memory/s2r_loader/loader_a.cuh": "c84e0b9456069a279f2c8f72034d8d62f765c323",
    "memory/s2r_loader/loader_b.cuh": "495ffe35998f2984f2a86a108d528d48bb79c9cd",
    "memory/g2s_pipeline.cuh": "bb6775f3bd0c2a45f825638c26c3c10886d60f95",
    "memory/s2r_pipeline.cuh": "3e8cd7a0ec157652edac6ab42a9c474ddc453c0d",
    "utils/storage.cuh": "05a741aff64754941d184a6d111705ac446f38eb",
    "utils/ptx/legacy_load.cuh": "bed5ca6b51a07387189b063b1e269923f3c607ad",
    "mma/wmma.cuh": "50f206cbf4721bcecee69bbb563551cc45b3044d",
    "arith/mainloop_arith.cuh": "00704bc9d8bf8c11a5c2e9451ab1f347b61e76be",
    "arith/epilogue_arith.cuh": "77500544cf45c5c4369fee6a4090ef4d798836c9",
    "utils/ptx/barrier.cuh": "0c717df8ea95c3c05ab7b388062d704030a7cf6d",
    "datatype/dequant.cuh": "ad67ecbac1a447be400fc13459d6037a57969e0d",
    "epilogue/gmem_writer.cuh": "37a543a5f90d567b46bf8e4e229071c7dbb6b6b1",
    "epilogue/smem_writer.cuh": "58daa6054902948e8b3676694f70438e7ee11969",
    "epilogue/pipeline.cuh": "df708fba56aa1528baf8efeb99cae17d7f8f9f74",
    "scheduler.cuh": "4ca098d86624452941c1d51680c7d3fc55737d64",
}
_PY_PINS = {
    "config/config.py": "772b0d47ff947f21bdf9332a610e2d72ad70e9a1",
    "kernel/humming.py": "23d625f32778f9d86fd11bb8a6fbfe501306be4e",
    "jit/runtime.py": "a8adc8529ee92cd41b5b68d542d50766b865cd13",
}

_B_AND_OUTPUT = r"""
#include <humming/utils/all.cuh>
CUDA_INLINE void apair_evf_copy(const int4 *g, int4 *s, bool valid) {
  const uint32_t sm = cast_smem_ptr_to_uint(s);
  const uint32_t bytes = valid ? 16u : 0u;
  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0],[%1],16,%2,%3;"
               :: "r"(sm), "l"(g), "r"(bytes), "l"(policy) : "memory");
}

#define G2SMemoryLoaderB ApairStockG2SMemoryLoaderB
#include <humming/memory/g2s_loader/loader_b.cuh>
#undef G2SMemoryLoaderB
template <class ProblemShape, class BlockShape, class ElementA, class ElementB,
          class ComputeConfig, class TuningConfig>
class G2SMemoryLoaderB : public ApairStockG2SMemoryLoaderB<
    ProblemShape, BlockShape, ElementA, ElementB, ComputeConfig, TuningConfig> {
  using Base = ApairStockG2SMemoryLoaderB<
      ProblemShape, BlockShape, ElementA, ElementB, ComputeConfig, TuningConfig>;
  static_assert(ProblemShape::N == 1856 && ProblemShape::K == 2688);
  static_assert(BlockShape::N == 128 && BlockShape::K == 32);
  static_assert(ElementA::kBits == 16 && ElementB::kBits == 4);
  static_assert(TuningConfig::kNumThreads == 256 && TuningConfig::kNumLoadThreads == 256);
  static_assert(TuningConfig::kUseCpAsync && !TuningConfig::kUseTmaB && !TuningConfig::kUseWarpSpec);
  bool tail = false;
 public:
  using Base::Base;
  CUDA_INLINE void seek(uint32_t expert_id, uint32_t n, uint32_t k) {
    Base::seek(expert_id, n, k);
    tail = n == 14;
  }
  template <bool kShouldAdvance = true>
  CUDA_INLINE void load(int4 *s, void *mbar_ptr) {
    // Packed B has two K16 rows, each 64 int4s for this N128 tile.
    // The real problem stride stays 928 int4s. Never form a dummy N29
    // payload address: src-size zero uses the current, legal N28 base.
    const uint32_t t = threadIdx.x;
    if (t < 128) {
      const uint32_t row = t / 64, col = t % 64;
      const bool valid = !tail || col < 32;
      const int4 *g = this->gmem_ptr;
      if (valid) g += row * 928 + col;
      apair_evf_copy(g, s + t, valid);
    }
    if constexpr (kShouldAdvance) Base::advance();
  }
};

#define EpilogueGmemWriter ApairStockEpilogueGmemWriter
#include <humming/epilogue/gmem_writer.cuh>
#undef EpilogueGmemWriter
template <class SharedStorage, class ArithClass, class ProblemShape,
          class BlockShape, class PadShape, class ElementC,
          class ComputeConfig, class TuningConfig>
class EpilogueGmemWriter : public ApairStockEpilogueGmemWriter<
    SharedStorage, ArithClass, ProblemShape, BlockShape, PadShape, ElementC,
    ComputeConfig, TuningConfig> {
  using Base = ApairStockEpilogueGmemWriter<SharedStorage, ArithClass,
      ProblemShape, BlockShape, PadShape, ElementC, ComputeConfig, TuningConfig>;
  static_assert(ProblemShape::N == 1856 && BlockShape::M == 64 && BlockShape::N == 128);
  static_assert(PadShape::N == 0 && ElementC::kBits == 16);
  static_assert(ComputeConfig::kGemmType == GemmType::INDEXED);
  static_assert(!TuningConfig::kUseTmaC && !TuningConfig::kUseStreamK);
  static_assert(TuningConfig::kNumWriteSplits == 1 && TuningConfig::kNumMathThreads == 256);
 public:
  using Base::Base;
  CUDA_INLINE void write(uint32_t slice_id, uint32_t slice_count, uint32_t split_idx) {
    // Same pinned legacy writer/swizzle and indexed row lookup; always mask
    // by the real N even though PadShape::N is zero. No tail output exists.
    constexpr uint32_t count = 64 * 128 * 2 / 16;
    constexpr uint32_t iters = count / 256;
    const uint32_t base = offsetof(SharedStorage, reduce) / 128 % 8;
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < iters; i++) {
      const uint32_t offset = threadIdx.x + 256 * i;
      const uint32_t row = offset / 8, col = offset % 8;
      const uint32_t swizzled = row * 8 + (col ^ ((row + base) % 8));
      const uint32_t out_row = this->smem.wr_row_index[row % 64];
      const uint32_t out_col = row / 64 * 8 + col;
      if (out_row < this->output_shape_m && this->col_offset + out_col * 8 < 1856) {
        this->gmem_ptr[out_row * (1856 / 8) + out_col] = this->smem_ptr[swizzled];
      }
    }
  }
};
"""


def _replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def _loader_text(nvsc) -> str:
    # Retain the original exact 6-bit byte decode, including j interleaving
    # and the per-thread second K-group word. Only the two-record addressing
    # and zero-filled tail transport differ.
    text = nvsc._LOADERS
    text = _replace_once(text,
        "BlockShape::N == 64 && (BlockShape::K == 32 || BlockShape::K == 128)",
        "BlockShape::N == 128 && BlockShape::K == 32")
    text = _replace_once(text, "BlockShape::N == 64 && WarpShape::N == 32",
                         "BlockShape::N == 128 && WarpShape::N == 32")
    text = _replace_once(text, "const int4 *gmem_ptr;",
                         "const int4 *gmem_ptr;\n  const int4 *gmem_ptr_second;\n  bool has_second;")
    text = _replace_once(text, nvsc._STAGE_LOAD, r"""
    if (t < 2 * kStageInt4s) {
      const bool second_half = t >= kStageInt4s;
      const bool valid = !second_half || has_second;
      const int4 *g = gmem_ptr;
      if (valid) g = (second_half ? gmem_ptr_second : gmem_ptr) + t % kStageInt4s;
      apair_evf_copy(g, smem_ptr + t, valid);
    }""")
    text = _replace_once(text, "void advance() { gmem_ptr += kStageInt4s; }",
        "void advance() { gmem_ptr += kStageInt4s; gmem_ptr_second += kStageInt4s; }")
    text = _replace_once(text, r"""    const uint32_t row =
        (expert_id * kNumTiles + n_block_id) * kProblemNumGroups + k_block_id * kNumGroups;
    gmem_ptr = gmem_ptr_raw + kHeaderInt4s + row * kRowInt4s;""", r"""
    const uint32_t tile0 = n_block_id * 2;
    has_second = tile0 + 1 < kNumTiles;
    const uint32_t tile1 = has_second ? tile0 + 1 : tile0;
    const uint32_t row0 = (expert_id * kNumTiles + tile0) * kProblemNumGroups + k_block_id * kNumGroups;
    const uint32_t row1 = (expert_id * kNumTiles + tile1) * kProblemNumGroups + k_block_id * kNumGroups;
    gmem_ptr = gmem_ptr_raw + kHeaderInt4s + row0 * kRowInt4s;
    gmem_ptr_second = gmem_ptr_raw + kHeaderInt4s + row1 * kRowInt4s;""")
    text = _replace_once(text, "+ 96u * (uint32_t(iter_id) / 2u);",
        "+ 192u * (uint32_t(iter_id) / 2u) + 96u * ((threadIdx.x / 32 % 4) / 2);")
    return text


def _scheduler_text(pfup) -> str:
    text = pfup.sched_text(True, True)
    text = _replace_once(text,
        "(BlockShape::M == 64 || BlockShape::M == 32) && BlockShape::N == 64",
        "BlockShape::M == 64 && BlockShape::N == 128")
    text = _replace_once(text, "N_BLOCKS = ProblemShape::N / BlockShape::N;",
                         "N_BLOCKS = CEIL_DIV(ProblemShape::N, BlockShape::N);")
    text = _replace_once(text, "hm_pf_smem[61] = BlockShape::M;",
        "hm_pf_smem[61] = BlockShape::M;\n      hm_pf_smem[63] = n == 14;")
    # Word 63 held the second row-validity ballot until thread 0 combined
    # it. It is now safe to publish the tail flag with the existing barrier.
    text = _replace_once(text, "kNWarps = 64 / WarpShape::N;",
                         "kNWarps = 128 / WarpShape::N;")
    text = _replace_once(text,
        "pf_act = (groups >> g0) & ((1u << (WarpShape::M / 16)) - 1u);",
        "pf_act = (groups >> g0) & ((1u << (WarpShape::M / 16)) - 1u);\n"
        "    if (hm_pf_smem[63] && (threadIdx.x / 32 % kNWarps) >= 2) pf_act = 0;")
    text = text.replace("Block N = 64 (asserted by the scheduler), so N_WARPS = 64",
                        "Block N = 128 (asserted by the scheduler), so N_WARPS = 128")
    return text


def _check_headers():
    import humming
    from importlib.metadata import version

    assert version("humming-kernels") == "0.1.10"
    package = Path(humming.__file__).parent
    for root, pins in ((package / "include/humming", _PINS), (package, _PY_PINS)):
        for name, want in pins.items():
            raw = (root / name).read_bytes()
            got = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            assert got == want, (name, got)


def _kernel_class():
    global _CLASS
    if _CLASS is not None:
        return _CLASS
    from humming import dtypes
    from vllm.model_executor.layers.fused_moe.experts import humming_nvsc as nvsc
    from vllm.model_executor.layers.fused_moe.experts import humming_pfup as pfup
    from vllm.model_executor.layers.fused_moe.experts.humming_relu2_epilogue import Relu2HummingKernel

    loaders = _B_AND_OUTPUT + _loader_text(nvsc)
    schedule = _scheduler_text(pfup)

    class ApairNvscPrefillHummingKernel(Relu2HummingKernel):
        dyn_scheduler = True

        def check_shape(self):
            # All original check_shape constraints hold for these exact values
            # except N % blockN. N-tail is handled by B/BS transport and writer.
            assert list(self.problem_shape)[1:] == [1856, 2688]
            assert list(self.pad_shape) == [0, 0, 0]
            assert list(self.block_shape) == [64, 128, 32]
            assert list(self.warp_shape) == [32, 32, 32]
            assert self.a_dtype == self.c_dtype == dtypes.bfloat16

        def scheduler_code(self):
            return schedule

        def prepare(self):
            assert self.num_threads == self.num_math_threads == self.num_load_threads == 256
            assert self.num_stages == 4 and self.num_ctas_per_sm == 2 and self.num_write_splits == 1
            assert self.num_experts == 128 and self.is_indexed_gemm
            assert self.weight_scale_group_size == 16 and self.bs_dtype == dtypes.float8e4m3
            assert self.b_dtype == dtypes.float4e2m1 and not self.has_zero_point and not self.has_bias
            assert self.is_group_weight_scale and self.is_tensor_weight_scale
            assert self.use_batch_invariant and not self.use_fused_e8m0_scale and not self.use_int_weight_scale
            assert not self.has_input_scale and not self.use_f16_accum
            assert self.use_cp_async and not self.use_warp_spec and not self.use_mbarrier
            assert not self.use_stream_k and not self.use_tma
            assert not self.reduce_overlap_last_stage_only
            assert self.multi_cast_size_a == self.multi_cast_size_b == 1
            assert self.sm_version == 121
            at = self.code.index("#if ")
            self.code = self.code[:at] + loaders + self.code[at:]
            super().prepare()

    _CLASS = ApairNvscPrefillHummingKernel
    return _CLASS


def _resource_check(kernel, props):
    # Humming's register_kernel owns its own CUmodule. Inspect a temporary
    # module before publishing configs, then unload it without ever launching.
    # HummingKernel overrides KernelRuntime.load_cubin and has no self.func.
    import cuda.bindings.driver as drv

    def checked(result):
        assert int(result[0]) == 0, result[0]
        return result[1] if len(result) > 1 else None

    module = checked(drv.cuModuleLoad(kernel.kernel_filename.encode()))
    try:
        function = checked(drv.cuModuleGetFunction(module, kernel.kernel_name.encode()))
        def attr(name):
            return checked(drv.cuFuncGetAttribute(getattr(drv.CUfunction_attribute, name), function))
        assert attr("CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK") >= 256
        assert 0 < attr("CU_FUNC_ATTRIBUTE_NUM_REGS") <= 128
        assert attr("CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES") == 0
        static = attr("CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES")
        dynamic = kernel.get_cubin_symbol_value("SMEM_SIZE")
        assert kernel.get_cubin_symbol_value("NUM_THREADS") == 256
        assert 0 < dynamic and static + dynamic <= props.shared_memory_per_block
        assert 2 * (static + dynamic) <= props.shared_memory_per_multiprocessor
    finally:
        checked(drv.cuModuleUnload(module))


def _configs(meta, compute_config, tuning_config, props):
    from humming.kernel.humming import HummingKernel
    from vllm.model_executor.layers.fused_moe.experts import humming_nvsc as nvsc

    _check_headers()
    # Prepare the original exact variant first. This also validates all
    # existing NVSC/PFUP assumptions before deriving a separate configuration.
    baseline = nvsc._prefill_configs(meta, compute_config, tuning_config, True).tolist()
    assert len(baseline) == 4
    original = HummingKernel._id2kernel[baseline[2]]
    assert list(original.block_shape) == [64, 64, 32]
    assert list(original.warp_shape) == [32, 32, 32]
    assert original.num_stages == 4 and original.num_ctas_per_sm == 4
    assert "hm_pf_smem" in original.code and "nvsc_table" in original.code
    lc = json.loads(meta.to_str())
    lc.pop("sublayer_name", None)
    tc = json.loads(tuning_config)
    if isinstance(tc, dict):
        tc = [[0, 1 << 30, tc]]
    selected = [config for lo, hi, config in tc if lo < 2976 <= hi]
    assert len(selected) == 1
    params = lc | json.loads(compute_config) | selected[0]
    params.pop("num_sms", None)
    # Same total grid warp budget: original 4 CTAs x 4 warps -> 2 x 8.
    params = params | {"block_shape": [64, 128, 32], "num_ctas_per_sm": 2}
    cls = _kernel_class()
    kernel = cls(**params)
    kernel.load_cubin()
    assert type(kernel) is cls and kernel.kernel_filename != original.kernel_filename
    _resource_check(kernel, props)
    return torch.tensor([0, 1 << 30, kernel.kernel_id, baseline[3]], dtype=torch.int64, device="cpu")


def _disjoint(tensors):
    # Reject overlaps of physical storage, including distinct views. All
    # request tensors are read/written only by the original ordered stream.
    ranges = []
    for tensor in tensors:
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
            return False
        storage = tensor.untyped_storage()
        lo, n = storage.data_ptr(), storage.nbytes()
        if n and any(lo < hi0 and lo0 < lo + n for lo0, hi0 in ranges):
            return False
        ranges.append((lo, lo + n))
    return True


def launch_prefill(layer, inputs, outputs, compute_config, tuning_config,
                   sorted_ids, expert_ids, num_tokens_padded, top_k, valid_shape_m, dyn):
    """True only after the one candidate launch; False means nothing launched."""
    if not ENABLED or not dyn or top_k != 6 or valid_shape_m != 2976:
        return False
    from vllm.model_executor.layers.fused_moe.experts import humming_pfup as pfup
    if pfup.flags() != (True, True):
        return False
    if inputs.shape != (496, 2688) or inputs.dtype != torch.bfloat16:
        return False
    if outputs.dtype != torch.bfloat16 or outputs.numel() != 2976 * 1856:
        return False
    if not inputs.is_cuda or not outputs.is_cuda or not inputs.is_contiguous() or not outputs.is_contiguous():
        return False
    if inputs.device != outputs.device or inputs.device.index != torch.cuda.current_device():
        return False
    props = torch.cuda.get_device_properties(inputs.device)
    if (props.major, props.minor, props.multi_processor_count) != (12, 1, 48):
        return False
    meta = layer.humming_metas["w13"]
    bs = getattr(layer, "nvsc_w13_bs", None)
    weight = getattr(layer, meta.weight_name)
    gs = getattr(layer, meta.global_scale_name, None)
    if bs is None or getattr(layer, meta.zero_point_name, None) is not None:
        return False
    if not _disjoint((inputs, outputs, weight, bs, gs, sorted_ids, expert_ids,
                      num_tokens_padded, layer.locks)):
        return False
    key = (inputs.device.index, meta.to_str(), compute_config, tuning_config, pfup.flags())
    if key in _FAILED:
        return False
    configs = _CONFIGS.get(key)
    if configs is None:
        # Never compile, allocate configs, query resources or read headers
        # inside a CUDA capture. A cold capture uses the original kernel.
        if torch.cuda.is_current_stream_capturing():
            return False
        try:
            configs = _configs(meta, compute_config, tuning_config, props)
        except Exception as exc:  # noqa: BLE001
            _FAILED.add(key)
            logger.warning("w13 A-pair unavailable before launch; keeping NVSC: %s", exc)
            return False
        _CONFIGS[key] = configs
        logger.info_once("w13 A-pair prefill prepared; resource checks are not numerical/performance admission")
    from humming import ops
    # Same arguments, route buffers, global scales, locks and current CUDA
    # stream as the original NVSC call. No retry after any execution error.
    ops.launch_kernel(
        configs=configs, inputs=inputs, weight=weight, outputs=outputs,
        input_scale=None, weight_scale=bs, zero_point=None, bias=None,
        global_scale=gs, sorted_ids=sorted_ids, expert_ids=expert_ids,
        num_tokens_padded=num_tokens_padded, locks=layer.locks,
        top_k=top_k, valid_shape_m=valid_shape_m,
    )
    return True

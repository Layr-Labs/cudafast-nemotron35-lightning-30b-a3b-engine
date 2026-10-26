# SPDX-License-Identifier: Apache-2.0
"""Private per-call K16 masks emitted by the actual routed-up ReLU2 writer.

The numerical producer is the exact selected NVSC/stock Humming kernel. Its
already-prepared source is extended before a single new compilation. No
installed header, weight, runtime global pointer or request cache is changed.
"""

from __future__ import annotations

import dataclasses
import json
import os
import statistics

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
ENABLED = os.environ.get("HUMMING_PRODUCED_MASKS", "1") != "0"
_CLASSES = {}
_CONFIGS = {}

# Pinned Humming 0.1.10, public source4351af3a8fcdce1a8dee50104ba49566af2427fb.
# This reuses the ignored non-streamK locks ABI for a *private call-local*
# integer output. The original layer.locks is never modified or repurposed.
# check_tensor_locks() in the pinned launcher returns immediately without
# inspecting this argument when use_stream_k is false. Strong owners are the
# caller's local tensor and then its ordinary captured allocation lifetime.
_WRITER = r"""
template <class MmaOpClass, class BlockShape, class WarpShape,
          class ElementA, class ElementB, class ElementC, class ElementBS,
          class LayerConfig, class TuningConfig>
class EpilogueArithmetic : public ProducedMaskArithmeticBase<
    MmaOpClass, BlockShape, WarpShape, ElementA, ElementB, ElementC, ElementBS,
    LayerConfig, TuningConfig> {
 public:
  uint16_t *produced_mask = nullptr;
};

#define EpilogueGmemWriter ProducedMaskBaseGmemWriter
#include <humming/epilogue/gmem_writer.cuh>
#undef EpilogueGmemWriter

template <class SharedStorage, class ArithClass, class ProblemShape,
          class BlockShape, class PadShape, class ElementC,
          class ComputeConfig, class TuningConfig>
class EpilogueGmemWriter : public ProducedMaskBaseGmemWriter<
    SharedStorage, ArithClass, ProblemShape, BlockShape, PadShape, ElementC,
    ComputeConfig, TuningConfig> {
  using Base = ProducedMaskBaseGmemWriter<
      SharedStorage, ArithClass, ProblemShape, BlockShape, PadShape, ElementC,
      ComputeConfig, TuningConfig>;
 public:
  using Base::Base;
  static_assert(!TuningConfig::kUseStreamK && !TuningConfig::kUseTmaC);
  static_assert(ComputeConfig::kGemmType == GemmType::INDEXED);
  static_assert(ElementC::kBits == 16 && TuningConfig::kNumWriteSplits == 1);
  static_assert(BlockShape::N == 64 && ProblemShape::N - PadShape::N == 1856);
  static_assert(TuningConfig::kNumMathThreads % 32 == 0);

  CUDA_INLINE
  void write(uint32_t slice_id, uint32_t slice_count, uint32_t split_idx) {
    // The indexing, swizzle, gather, guards and original output store below
    // are the pinned write_legacy loop. Only the integer mask side output is
    // added; it observes the same already-rounded int4 used by the store.
    constexpr uint32_t total = BlockShape::M * BlockShape::N * 2 / 16;
    constexpr uint32_t nt = TuningConfig::kNumMathThreads;
    constexpr bool full = total % nt == 0;
    constexpr uint32_t iters = CEIL_DIV(total, nt);
    const uint32_t smem_base = offsetof(SharedStorage, reduce) / 128 % 8;
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < iters; ++i) {
      const uint32_t smem_offset = threadIdx.x + nt * i;
      if (full || i != iters - 1 || smem_offset < total) {
        const uint32_t smem_row = smem_offset / 8;
        const uint32_t smem_col = smem_offset % 8;
        const uint32_t swizzled = smem_row * 8 + (smem_col ^ ((smem_row + smem_base) % 8));
        const uint32_t local_row = smem_row % BlockShape::M;
        const uint32_t row = this->smem.wr_row_index[local_row];
        const uint32_t col = smem_row / BlockShape::M * 8 + smem_col;
        const bool valid_row = row < this->output_shape_m;
        const bool valid_col = PadShape::N == 0 ||
            this->col_offset + col * 8 < ProblemShape::N - PadShape::N;
        if (!valid_row || !valid_col) continue;
        const int4 val = this->smem_ptr[swizzled];
        const uint32_t dst = row * ((ProblemShape::N - PadShape::N) / 8) + col;
        this->gmem_ptr[dst] = val;  // original non-streamK output store
        const uint32_t words[4] = {static_cast<uint32_t>(val.x), static_cast<uint32_t>(val.y),
                                   static_cast<uint32_t>(val.z), static_cast<uint32_t>(val.w)};
        uint32_t mask8 = 0;
        PRAGMA_UNROLL
        for (int j = 0; j < 4; ++j) {
          mask8 |= static_cast<uint32_t>((words[j] & 0x7FFFu) != 0) << (2 * j);
          mask8 |= static_cast<uint32_t>((words[j] & 0x7FFF0000u) != 0) << (2 * j + 1);
        }
        // Adjacent lanes own the two contiguous 8-feature halves of a K16.
        // N/16 and the 64-column tile are integral, so each live pair has the
        // same row/column guard and exactly one even-half writer.
        const uint32_t peer = __shfl_xor_sync(__activemask(), mask8, 1);
        if ((col & 1u) == 0) {
          const uint32_t group = (this->col_offset + col * 8) / 16;
          this->arith.produced_mask[row * 116 + group] =
              static_cast<uint16_t>(mask8 | (peer << 8));
        }
      }
    }
  }
};

#define EpiloguePipeline ProducedMaskBasePipeline
#include <humming/epilogue/pipeline.cuh>
#undef EpiloguePipeline

template <class MmaOpClass, class SharedStorage, class ArithClass,
          class ProblemShape, class BlockShape, class WarpShape, class PadShape,
          class ElementA, class ElementC, class LayerConfig, class ComputeConfig,
          class TuningConfig>
class EpiloguePipeline : public ProducedMaskBasePipeline<
    MmaOpClass, SharedStorage, ArithClass, ProblemShape, BlockShape, WarpShape,
    PadShape, ElementA, ElementC, LayerConfig, ComputeConfig, TuningConfig> {
  using Base = ProducedMaskBasePipeline<
      MmaOpClass, SharedStorage, ArithClass, ProblemShape, BlockShape, WarpShape,
      PadShape, ElementA, ElementC, LayerConfig, ComputeConfig, TuningConfig>;
  using Output = std::conditional_t<TuningConfig::kUseTmaC, const void *, void *>;
 public:
  static_assert(!TuningConfig::kUseStreamK);
  CUDA_INLINE
  EpiloguePipeline(SharedStorage &smem, Output out, CUtensorMap *maps, ArithClass &arith,
                   const uint32_t *GS, int32_t *call_masks, uint32_t rows, uint32_t top_k)
      : Base(smem, out, maps, arith, GS, nullptr, rows, top_k) {
    arith.produced_mask = reinterpret_cast<uint16_t *>(call_masks);
  }
};
"""


def _extended_kernel(original):
    """Clone the exact selected producer, extending its fully prepared source.

    KernelRuntime.prepare compiles immediately. The override therefore uses
    the original's immutable prepared source and calls KernelRuntime exactly
    once, after insertion. It never calls original.prepare then edits too late.
    """
    from humming.jit.runtime import KernelRuntime

    cls = type(original)
    key = (cls, original.kernel_id)
    new_cls = _CLASSES.get(key)
    if new_cls is None:
        source = original.code
        marker = "class EpilogueArithmetic : public UnfusedEpilogueArithmetic<"
        if source.count(marker) != 1 or "ProducedMaskArithmeticBase" in source:
            raise ValueError("unsupported routed-up arithmetic source")
        source = source.replace(marker, "class ProducedMaskArithmeticBase : public UnfusedEpilogueArithmetic<", 1)
        at = source.index("#if ")
        extended = source[:at] + _WRITER + source[at:]

        def prepare(self):
            if (self.problem_shape != original.problem_shape or self.block_shape != original.block_shape
                    or self.warp_shape != original.warp_shape or self.num_stages != original.num_stages
                    or self.use_stream_k or self.use_tma_c or self.use_warp_spec):
                raise ValueError("producer clone configuration differs")
            self.code = extended
            KernelRuntime.prepare(self)

        # The original class id differentiates NVSC/stock/dyn/evict variants.
        new_cls = type("ProducedMasks" + cls.__name__ + str(original.kernel_id),
                       (cls,), {"prepare": prepare, "__module__": __name__})
        _CLASSES[key] = new_cls
    params = {f.name: getattr(original, f.name) for f in dataclasses.fields(original) if f.init}
    kernel = new_cls(**params)
    kernel.load_cubin()
    if (type(kernel) is not new_cls or kernel.kernel_id == original.kernel_id
            or kernel.kernel_filename == original.kernel_filename
            or "ProducedMaskArithmeticBase" not in kernel.code):
        raise ValueError("masked producer did not load a distinct extended kernel")
    return kernel


def _producer_configs(layer, compute_config, tuning_config):
    """Use the actual selected decode source, including NVSC/evict/grid overrides."""
    from humming.kernel.humming import HummingKernel

    from vllm.model_executor.layers.fused_moe.experts import (
        humming_nvsc,
        humming_relu2_epilogue as fused,
    )

    meta = layer.humming_metas["w13"]
    compressed = getattr(layer, "nvsc_w13_bs", None)
    dyn = fused._dyn_enabled()
    use_nvsc = (compressed is not None and humming_nvsc.ENABLED
                and fused._variant_key(dyn) not in humming_nvsc._FAILED)
    if use_nvsc:
        original = humming_nvsc._configs(meta, compute_config, tuning_config, dyn)
        scale = compressed
    else:
        original = fused.prepare(meta, compute_config, tuning_config, dyn=dyn)
        scale = getattr(layer, meta.weight_scale_name, None)
    values = original.tolist()
    matches = [i for i in range(0, len(values), 4) if values[i] < 30 <= values[i + 1]]
    if len(matches) != 1:
        raise ValueError("missing actual five-token routed-up configuration")
    i = matches[0]
    source = HummingKernel._id2kernel[values[i + 2]]
    if (source.use_stream_k or source.use_tma_c or source.use_warp_spec
            or source.num_write_splits != 1 or tuple(source.problem_shape[1:]) != (1856, 2688)
            or source.block_shape[1] != 64 or source.pad_shape[1] != 0):
        raise ValueError("unsupported actual routed-up writer")
    key = (source.kernel_id, tuple(values))
    new = _CONFIGS.get(key)
    if new is None:
        kernel = _extended_kernel(source)
        rewritten = list(values)
        rewritten[i + 2] = kernel.kernel_id
        new = torch.tensor(rewritten, dtype=torch.int64, device="cpu")
        _CONFIGS[key] = new
    return original, new, scale, source.kernel_id


def _launch(layer, inputs, outputs, masks, configs, scale, kwargs):
    from humming import ops

    meta = layer.humming_metas["w13"]
    return ops.launch_kernel(
        configs=configs, inputs=inputs, outputs=outputs,
        weight=getattr(layer, meta.weight_name), input_scale=None,
        weight_scale=scale, zero_point=getattr(layer, meta.zero_point_name, None), bias=None,
        global_scale=getattr(layer, meta.global_scale_name, None),
        sorted_ids=kwargs["sorted_ids"], expert_ids=kwargs["expert_ids"],
        num_tokens_padded=kwargs["num_tokens_padded"],
        locks=masks.view(torch.int32) if masks is not None else layer.locks,
        top_k=kwargs["top_k"], valid_shape_m=kwargs["valid_shape_m"],
    )


def reference_masks(activation):
    """Independent tensor reference used only by the remote private qualifier."""
    bits = activation.view(torch.int16).to(torch.int32) & 0x7FFF
    live = (bits.reshape(30, 116, 16) != 0).to(torch.int32)
    weights = 1 << torch.arange(16, dtype=torch.int32, device=activation.device)
    return (live * weights).sum(dim=-1).to(torch.int16)


def _guarded(shape, dtype, device):
    # Both views are contiguous/aligned; surrounding storage detects overruns.
    count = 1
    for size in shape:
        count *= size
    storage = torch.full((count + 128,), 15371, dtype=torch.int16, device=device)
    tensor = storage[64:-64].view(dtype).view(shape)
    return tensor, storage


def _guards_ok(storage):
    return bool(torch.all(storage[:64] == 15371) and torch.all(storage[-64:] == 15371))


def _qualify(layer, original, extended, scale, compute_config, tuning_config):
    """Remote startup only: actual production up + checked 653 down, not an
    independently instantiated stock producer. All inputs are private probes.
    Changing graph inputs and prefixes exercise same-call mask ownership.
    """
    import zskip

    from vllm.model_executor.layers.fused_moe.experts.humming_relu2_epilogue import forward_relu2
    from vllm.model_executor.layers.fused_moe.experts.humming_zskip import _align

    dev = layer.w2_global_scale.device
    gen = torch.Generator(device=dev).manual_seed(972673)
    x, xg = _guarded((5, 2688), torch.bfloat16, dev)
    x.copy_(torch.randn((5, 2688), generator=gen, dtype=torch.bfloat16, device=dev))
    a0, ag0 = _guarded((30, 1856), torch.bfloat16, dev)
    a1, ag1 = _guarded((30, 1856), torch.bfloat16, dev)
    y0, yg0 = _guarded((30, 2688), torch.bfloat16, dev)
    y1, yg1 = _guarded((30, 2688), torch.bfloat16, dev)
    masks, mg = _guarded((30, 116), torch.int16, dev)
    counts = torch.tensor([5], dtype=torch.int32, device=dev)
    # Use one padded alignment allocation for each routing, reused in its graph.
    cases = []
    for crowded in (False, True):
        ids = torch.tensor([[(29 * j + (0 if crowded else 17 * t)) % 128
                             for j in range(6)] for t in range(5)], dtype=torch.int32, device=dev)
        sid, eid, ntp = _align(ids, 16)
        kw = dict(sorted_ids=sid, expert_ids=eid, num_tokens_padded=ntp, top_k=6,
                  valid_shape_m=30, compute_config=compute_config, tuning_config=tuning_config)

        def old_pair(kw=kw):
            # Exact public production entry, including its NVSC dispatch.
            forward_relu2(layer, x, a0, **kw)
            zskip.down(a0, kw["sorted_ids"], kw["expert_ids"], kw["num_tokens_padded"],
                       layer.zskip_pair_w2, y0, config=653, max_blocks=30, verify_rows=counts)

        def new_pair(kw=kw):
            _launch(layer, x, a1, masks, extended, scale, kw)
            zskip.down(a1, kw["sorted_ids"], kw["expert_ids"], kw["num_tokens_padded"],
                       layer.zskip_pair_w2, y1, config=673, max_blocks=30,
                       verify_rows=counts, row_masks=masks)

        # Original low-level entry must really be the same actual producer.
        forward_relu2(layer, x, a0, **kw)
        _launch(layer, x, a1, None, original, scale, kw)
        if not torch.equal(a0.view(torch.int16), a1.view(torch.int16)):
            return False
        old_pair()
        new_pair()
        torch.cuda.synchronize(dev)
        old_graph, new_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(old_graph):
            old_pair()
        with torch.cuda.graph(new_graph):
            new_pair()
        # All live rows, zero input, signed zero and tiny magnitudes; poisoned
        # outputs make missed producer writes visible on every eager invocation.
        for magnitude in (1.0, 0.0, 2.0**-40, -0.25):
            x.copy_((torch.randn((5, 2688), generator=gen, device=dev) * magnitude).to(torch.bfloat16))
            for prefix in (5, 1, 4, 2, 3, 5):
                counts.fill_(prefix)
                masks.fill_(-23131)
                a1.fill_(float("nan"))
                y1.fill_(float("nan"))
                expected_x = x.clone()
                old_pair()
                new_pair()
                if not torch.equal(x.view(torch.int16), expected_x.view(torch.int16)):
                    return False
                if not torch.equal(a0.view(torch.int16), a1.view(torch.int16)):
                    return False
                if not torch.equal(masks, reference_masks(a1)):
                    return False
                if not torch.equal(y0.view(torch.int16), y1.view(torch.int16)):
                    return False
                # Same captured graph, changed private activation input/prefix.
                x.neg_()
                counts.fill_(6 - prefix)
                masks.fill_(-12345)
                expected_x = x.clone()
                old_graph.replay()
                new_graph.replay()
                torch.cuda.synchronize(dev)
                if (not torch.equal(x.view(torch.int16), expected_x.view(torch.int16))
                        or not torch.equal(a0.view(torch.int16), a1.view(torch.int16))
                        or not torch.equal(masks, reference_masks(a1))
                        or not torch.equal(y0.view(torch.int16), y1.view(torch.int16))):
                    return False
                if not all(_guards_ok(s) for s in (xg, ag0, ag1, yg0, yg1, mg)):
                    return False
        cases.append((old_graph, new_graph))

    # Both full pairs have the same launches. Include mask producer cost, the
    # original selected producer, real scales/weights and prefix loads. Keep
    # setup/private probes out of requests; no request work is precomputed.
    pressure = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=dev)
    x.copy_(torch.randn((5, 2688), generator=gen, dtype=torch.bfloat16, device=dev))
    ratios = []
    for old_graph, new_graph in cases:
        for prefix in (1, 5):
            counts.fill_(prefix)
            for cold in (False, True):
                pairs = [[], []]
                for repeat in range(5):
                    for which in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                        graph = (old_graph, new_graph)[which]
                        graph.replay()
                        if cold:
                            pressure.fill_(repeat)
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record()
                        for _ in range(1 if cold else 4):
                            graph.replay()
                        end.record()
                        end.synchronize()
                        pairs[which].append(begin.elapsed_time(end))
                ratios.append(statistics.median(pairs[1]) / statistics.median(pairs[0]))
    if not all(_guards_ok(s) for s in (xg, ag0, ag1, yg0, yg1, mg)):
        return False
    median, worst = statistics.median(ratios), max(ratios)
    logger.info("UP_MASKS_PAIR layer=%s median_ratio=%.6f worst_ratio=%.6f",
                getattr(layer, "layer_name", "?"), median, worst)
    return median < 0.99 and worst < 1.01


def prepare_layer(layer):
    """After zskip AND NVSC preparation; refuse without changing either path."""
    layer.produced_masks_state = None
    if not ENABLED or getattr(layer, "zskip_pair_config", None) != 653:
        return
    try:
        import zskip

        from vllm import envs
        from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import _humming_batch_invariant
        from vllm.model_executor.layers.fused_moe.experts.humming_relu2_epilogue import _dyn_enabled, _variant_key
        from vllm.model_executor.layers.fused_moe.experts.humming_tuned_moe import tuned_moe_configs

        if not getattr(layer, "zskip_pair_prefix_ok", False) or not zskip.masks_available():
            return
        configs = tuned_moe_configs(layer)
        if configs is None:
            return
        compute = json.dumps(dict(use_batch_invariant=_humming_batch_invariant(),
                                  use_f16_accum=envs.VLLM_HUMMING_USE_F16_ACCUM,
                                  gemm_type="indexed"))
        tuning = json.dumps(configs[0])
        meta = layer.humming_metas["w13"]
        if getattr(layer, meta.bias_name, None) is not None:
            return
        zskip.config_info(673)
        original, extended, scale, original_id = _producer_configs(layer, compute, tuning)
        if not _qualify(layer, original, extended, scale, compute, tuning):
            logger.info("UP_MASKS_DECLINED layer=%s", getattr(layer, "layer_name", "?"))
            return
        # Immutable kernel/weight metadata only. Never persist a request's mask.
        layer.produced_masks_state = dict(compute=compute, tuning=tuning, configs=extended,
                                         scale=scale, source_id=original_id,
                                         variant=_variant_key(_dyn_enabled()),
                                         compressed=scale is getattr(layer, "nvsc_w13_bs", None),
                                         weight=getattr(layer, meta.weight_name),
                                         stock_scale=getattr(layer, meta.weight_scale_name, None))
        logger.info("UP_MASKS_ADMITTED layer=%s source=%s", getattr(layer, "layer_name", "?"), original_id)
    except Exception as exc:  # optional exact path; actual f0 selections retained
        logger.warning_once("UP_MASKS_UNAVAILABLE %r; original producer/653 retained", exc)


def try_launch(layer, inputs, outputs, kwargs):
    """Return a private same-call mask, or None before launching any work."""
    state = getattr(layer, "produced_masks_state", None)
    if state is None or not ENABLED or torch.compiler.is_compiling():
        return None
    if (tuple(inputs.shape) != (5, 2688) or tuple(outputs.shape) != (30, 1856)
            or inputs.dtype != torch.bfloat16 or outputs.dtype != torch.bfloat16
            or not inputs.is_contiguous() or not outputs.is_contiguous()
            or inputs.device != outputs.device or inputs.data_ptr() % 16 or outputs.data_ptr() % 16
            or kwargs["valid_shape_m"] != 30 or kwargs["top_k"] != 6
            or kwargs["compute_config"] != state["compute"] or kwargs["tuning_config"] != state["tuning"]
            or getattr(layer, "zskip_pair_config", None) != 653
            or not getattr(layer, "zskip_pair_prefix_ok", False)):
        return None
    from vllm.model_executor.layers.fused_moe.experts import humming_nvsc
    from vllm.model_executor.layers.fused_moe.experts.humming_relu2_epilogue import _dyn_enabled, _variant_key

    meta = layer.humming_metas["w13"]
    if (getattr(layer, meta.weight_name) is not state["weight"]
            or getattr(layer, meta.weight_scale_name, None) is not state["stock_scale"]
            or _variant_key(_dyn_enabled()) != state["variant"]
            or (state["compressed"] and (getattr(layer, "nvsc_w13_bs", None) is not state["scale"]
                                         or not humming_nvsc.ENABLED or state["variant"] in humming_nvsc._FAILED))):
        return None
    masks = torch.empty((30, 116), dtype=torch.int16, device=inputs.device)
    # No catch after this launch: runtime errors propagate normally. A failed
    # partial producer must not be silently re-executed during graph capture.
    _launch(layer, inputs, outputs, masks, state["configs"], state["scale"], kwargs)
    return masks

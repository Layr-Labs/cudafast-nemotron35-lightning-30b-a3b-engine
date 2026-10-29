# SPDX-License-Identifier: Apache-2.0
"""LOCAL PATCH (cudafast-nemotron35, vllm/VENDOR.json "local_patches"): nvsc,
lossless 6-bit block scales for the routed expert-up (w13) humming kernel at
decode (<= 8 tokens, valid_shape_m <= 48).

Format. A w13 layer whose FP8 E4M3 block-scale bytes take at most 64 distinct
values (21 of the 23 routed layers of the pinned checkpoint; layers 6 and 8
have 106 and 99) gets at load a 64-entry byte table (the sorted distinct
values) and a 6-bit index per scale, stored tile-major after a 128-byte
header: for expert e, 64-column tile t and 32-wide K block (one pipeline
stage: K groups 2kb and 2kb+1, codes c0 and c1) one 96-byte record = 16
little-endian u32 whose byte i holds the low nibbles of c0 | c1 << 4, then 16
u16 whose nibbles (order 0, 2, 1, 3) hold the high two bits of c0 | c1 << 2.
Word j, byte i is humming's scale byte 4j + i of that 64-byte tile row, so the
kernel's own byte permutation is kept; one stage reads 96 contiguous bytes
instead of 2 x 64 strided ones. The repack is input-independent; the load
check decodes every byte back and compares it with the original, and a
mismatch or more than 64 values keeps that layer on the stock kernel.

Kernel. A derived humming kernel (no installed file is edited) defines the two
B-scale loaders before the pinned headers are included (their include guards
then skip the stock ones). The global->shared loader issues 6 x 16-byte
cp.async per stage in the stock pipeline slots and commit groups and, with
the first tile, 4 more for the table into a static shared array (completed by
the first stage wait). At the first of a stage's two K16 iterations the
shared->register loader reads the thread's 48 bits, looks both groups' four
codes up, hands the mainloop the same uint32 of four original E4M3 bytes the
stock loader reads and keeps the second group's for the next iteration. Dequantization,
the K16 MMA chain, the epilogue and ReLU^2 are the stock code, so the output
is bit-identical by construction (harness/tests/nvsc_exact.py checks it
against the stock kernel and times both).

Prefill (LOCAL PATCH nvsc-prefill, default ON, NVSC_PREFILL=0 disables: bit-identical
on all 23 layers x 4 real 496-token prompts; ~12 us/layer = ~0.29 ms per prefill in
isolation; benchd-leg ABAB 8 pairs: prefill -0.32% +- 0.26). The 496-token
w13 call (valid_shape_m 2976, the claim-when-free + evict-first kernel of
humming_relu2_epilogue.py) runs block [64, 64, 32], warp [32, 32, 32]: the
same 64-column x 32-K tile and the same two warps per 64-column mini block
(the stock shared->register index uses warp_id % 2 only; the M warps read
the same scales), so the decode stream serves it as is: no second repack and
no extra memory. A derived class (NvscDynEvictFirstRelu2HummingKernel, own
name and cache key) takes the same two loaders, with the 96-byte stage loads
under the same L2 evict_first policy as the stock B/BS loads of that kernel;
the B loader, scheduler, epilogue and pipeline (4 stages) are that kernel's
own text. Other prefill shapes and more than 8 tokens keep the stock kernel
and the stock scale tensor, which stays in place as a view of the same
allocation (header and stream first, then the stock bytes; the stream view
handed to the launcher has the stock shape). NVSC=0 restores the current
behaviour; NVSC_PREFILL=0 keeps the stock prefill kernel.
"""

from __future__ import annotations

import json
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("NVSC", "1") != "0"
PREFILL_ENABLED = os.environ.get("NVSC_PREFILL", "1") != "0"
PREFILL_VALID_M = 2976

DECODE_MAX_VALID_M = 48  # 8 tokens x top-6
HEADER = 128  # 64-byte table + pad; keeps the stream 128-byte aligned
STAGE = 96  # 2 K groups x 64 scales x 6 bits
TILE_N = 64
NUM_CODES = 64

_STATS = {"packed": [], "stock": {}}
_CONFIGS: dict = {}
# Scheduler variants (humming_relu2_epilogue._variant_key) whose derived kernel
# failed to build: those calls keep the same variant's stock kernel.
_FAILED: set = set()

_LOADERS = r"""
#include <humming/utils/all.cuh>
#define G2SMemoryLoaderBS NvscStockG2SMemoryLoaderBS
#include <humming/memory/g2s_loader/loader_bs.cuh>
#undef G2SMemoryLoaderBS
#define S2RMemoryLoaderBS NvscStockS2RMemoryLoaderBS
#include <humming/memory/s2r_loader/loader_bs.cuh>
#undef S2RMemoryLoaderBS

// nvsc: this layer's 64-entry E4M3 byte table, loaded once per CTA.
__shared__ __align__(128) uint8_t nvsc_table[128];

template <class ProblemShape, class BlockShape, class ElementBS,
          class LayerConfig, class TuningConfig>
class G2SMemoryLoaderBS {
 private:
  static_assert(ElementBS::kBits == 8);
  static_assert(LayerConfig::kIsGroupWeightScale && !LayerConfig::kIsBlockWeightScale);
  static_assert(!LayerConfig::kIsChannelWeightScale);
  static_assert(LayerConfig::kWeightScaleGroupSize == 16);
  static_assert(BlockShape::N == 64 && (BlockShape::K == 32 || BlockShape::K == 128));
  static_assert(ProblemShape::N % 64 == 0 && ProblemShape::K % 32 == 0);
  static_assert(!TuningConfig::kUseTmaBS && !TuningConfig::kUseWarpSpec);
  static_assert(TuningConfig::kUseCpAsync && !TuningConfig::kUseMBarrier);
  static constexpr uint32_t kHeaderInt4s = 8;
  static constexpr uint32_t kRowInt4s = 3;  // per K group: a stage (2 groups) is 96 bytes
  static constexpr uint32_t kNumGroups = BlockShape::K / 16;
  static constexpr uint32_t kStageInt4s = kRowInt4s * kNumGroups;
  static constexpr uint32_t kProblemNumGroups = ProblemShape::K / 16;
  static constexpr uint32_t kNumTiles = ProblemShape::N / 64;
  static constexpr uint32_t kLoadThreadOffset =
      TuningConfig::kNumThreads - TuningConfig::kNumLoadThreads;
  static_assert(TuningConfig::kNumLoadThreads >= kStageInt4s);

 public:
  const int4 *gmem_ptr_raw;
  const int4 *gmem_ptr;
  bool table_pending = true;

  CUDA_INLINE
  G2SMemoryLoaderBS(const void *ptr) {
    gmem_ptr_raw = reinterpret_cast<const int4 *>(ptr);
  }

  template <bool kShouldAdvance = true>
  CUDA_INLINE void load(int4 *smem_ptr, void *mbar_ptr) {
    const uint32_t t = threadIdx.x - kLoadThreadOffset;
    if (t < kStageInt4s) legacy_load<true>(gmem_ptr + t, smem_ptr + t);
    if constexpr (kShouldAdvance) advance();
  }

  CUDA_INLINE
  void advance() { gmem_ptr += kStageInt4s; }

  CUDA_INLINE
  void seek(uint32_t expert_id, uint32_t n_block_id, uint32_t k_block_id) {
    if (table_pending) {
      // Joins the first tile's stage-0 commit group.
      table_pending = false;
      const uint32_t t = threadIdx.x - kLoadThreadOffset;
      if (t < 4) legacy_load<true>(gmem_ptr_raw + t, reinterpret_cast<int4 *>(&nvsc_table[0]) + t);
    }
    const uint32_t row =
        (expert_id * kNumTiles + n_block_id) * kProblemNumGroups + k_block_id * kNumGroups;
    gmem_ptr = gmem_ptr_raw + kHeaderInt4s + row * kRowInt4s;
  }
};

template <class MmaOpClass, class BlockShape, class WarpShape,
          class ElementA, class ElementBS, class LayerConfig, class TuningConfig>
class S2RMemoryLoaderBS {
 private:
  static_assert(MmaOpClass::kMmaType != MmaType::WGMMA);
  static_assert(ElementA::kBits == 16 && ElementBS::kBits == 8);
  static_assert(LayerConfig::kIsGroupWeightScale && !LayerConfig::kIsBlockWeightScale);
  static_assert(!LayerConfig::kIsChannelWeightScale && !LayerConfig::kUseFusedE8m0Scale);
  static_assert(LayerConfig::kWeightScaleGroupSize == 16);
  static_assert(BlockShape::N == 64 && WarpShape::N == 32);
  static_assert(BlockShape::K == WarpShape::K &&
                (BlockShape::K == 32 || BlockShape::K == 128) && 256 / ElementA::kBits == 16);

  CUDA_INLINE static uint32_t lookup(uint32_t c) {
    const uint32_t b0 = nvsc_table[c & 0xFFu];
    const uint32_t b1 = nvsc_table[(c >> 8) & 0xFFu];
    const uint32_t b2 = nvsc_table[(c >> 16) & 0xFFu];
    const uint32_t b3 = nvsc_table[c >> 24];
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
  }

 public:
  uint32_t second = 0;

  // Called for (stage, K group 0) and then (same stage, K group 1); group 0
  // decodes both and keeps group 1's word.
  CUDA_INLINE
  void load(const int4 *smem_ptr, uint32_t *regs_ptr, int32_t iter_id) {
    if (iter_id % 2 == 1) {
      regs_ptr[0] = second;
      return;
    }
    // The stock loader's uint32 index in the 64-byte row (two warps per
    // 64-column mini block).
    const uint32_t j = (threadIdx.x % 32) / 4 * 2 + (threadIdx.x / 32) % 2;
    const uint8_t *stage = reinterpret_cast<const uint8_t *>(smem_ptr)
        + 96u * (uint32_t(iter_id) / 2u);
    const uint32_t lo = reinterpret_cast<const uint32_t *>(stage)[j];
    const uint32_t hi = reinterpret_cast<const uint16_t *>(stage + 64)[j];
    const uint32_t s = (hi | (hi << 12)) & 0x0F0F0F0Fu;
    const uint32_t c0 = (lo & 0x0F0F0F0Fu) | ((s << 4) & 0x30303030u);
    const uint32_t c1 = ((lo >> 4) & 0x0F0F0F0Fu) | ((s << 2) & 0x30303030u);
    regs_ptr[0] = lookup(c0);
    second = lookup(c1);
  }
};
"""


def _kernel_classes():
    """The derived kernel classes {False: stock schedule, True: dyn}. dyn adds the
    claim-when-free w13 tile scheduler (humming_relu2_epilogue.py,
    HUMMING_W13_DYN): Relu2HummingKernel.prepare inserts it, with the epilogue,
    after these loaders. Each variant has its own
    class name, since KernelRuntime interns kernels by class name + config. The
    scheduler adds no static shared memory (its counters are __device__ globals
    and the claim travels through the row-index staging slot), so the static
    shared footprint stays nvsc_table's 128 bytes at 128-byte alignment."""
    from vllm.model_executor.layers.fused_moe.experts.humming_relu2_epilogue import (
        Relu2HummingKernel,
    )

    class NvscRelu2HummingKernel(Relu2HummingKernel):
        def prepare(self):
            assert self.use_cp_async and not self.use_tma_bs and not self.use_warp_spec
            assert "#if " not in _LOADERS
            at = self.code.index("#if ")
            self.code = self.code[:at] + _LOADERS + self.code[at:]
            super().prepare()

    class NvscDynRelu2HummingKernel(NvscRelu2HummingKernel):
        dyn_scheduler = True

    return {False: NvscRelu2HummingKernel, True: NvscDynRelu2HummingKernel}


_KERNEL_CLASSES: dict = {}

_EVF_STAGE = r"""
CUDA_INLINE void nvsc_evf_load(const int4 *g, int4 *s) {
  uint32_t sm = cast_smem_ptr_to_uint(s);
  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0],[%1],16,%2;"
               :: "r"(sm), "l"(g), "l"(policy) : "memory");
}
"""
_STAGE_LOAD = "if (t < kStageInt4s) legacy_load<true>(gmem_ptr + t, smem_ptr + t);"
assert _LOADERS.count(_STAGE_LOAD) == 1
_LOADERS_PF = _LOADERS.replace(
    "// nvsc: this layer's", _EVF_STAGE + "\n// nvsc: this layer's", 1
).replace(_STAGE_LOAD, "if (t < kStageInt4s) nvsc_evf_load(gmem_ptr + t, smem_ptr + t);")


def _prefill_kernel_classes():
    """{dyn: class} of the nvsc prefill kernels: the nvsc loaders inserted
    first, then EvictFirstRelu2HummingKernel.prepare adds the evict-first B
    loader (its loader_bs include is then skipped by the include guard) and
    Relu2HummingKernel.prepare the scheduler and epilogue."""
    from vllm.model_executor.layers.fused_moe.experts.humming_relu2_epilogue import (
        EvictFirstRelu2HummingKernel,
    )

    class NvscEvictFirstRelu2HummingKernel(EvictFirstRelu2HummingKernel):
        def prepare(self):
            assert self.use_cp_async and not self.use_tma_bs and not self.use_warp_spec
            assert "#if " not in _LOADERS_PF
            at = self.code.index("#if ")
            self.code = self.code[:at] + _LOADERS_PF + self.code[at:]
            super().prepare()

    class NvscDynEvictFirstRelu2HummingKernel(NvscEvictFirstRelu2HummingKernel):
        dyn_scheduler = True

    return {False: NvscEvictFirstRelu2HummingKernel, True: NvscDynEvictFirstRelu2HummingKernel}


_PF_KERNEL_CLASSES: dict = {}
_PF_CONFIGS: dict = {}
_PF_FAILED: set = set()


# ---------------------------------------------------------------------------
# load-time repack


def encode(ws: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """[E, G, N] E4M3 bytes -> (table[64], stream[E * N/64 * G/2 * 96]) or
    None when the tensor holds more than 64 distinct bytes."""
    e, g, n = ws.shape
    assert n % TILE_N == 0 and g % 2 == 0
    b = ws.view(torch.uint8)
    vals = torch.unique(b)
    if vals.numel() > NUM_CODES:
        return None
    table = torch.empty(NUM_CODES, dtype=torch.uint8, device=b.device)
    table[: vals.numel()] = vals
    table[vals.numel():] = vals[-1]
    inv = torch.zeros(256, dtype=torch.int32, device=b.device)
    inv[vals.long()] = torch.arange(vals.numel(), dtype=torch.int32, device=b.device)
    nt, kb = n // TILE_N, g // 2
    stream = torch.empty(e, nt, kb, STAGE, dtype=torch.uint8, device=b.device)
    step = 16
    for e0 in range(0, e, step):
        c = inv[b[e0 : e0 + step].long()].view(-1, kb, 2, nt, 16, 4)
        c0, c1 = c[:, :, 0], c[:, :, 1]  # [e', KB, NT, 16, 4]
        lo = (c0 & 15) | ((c1 & 15) << 4)  # byte i of word j
        nib = (c0 >> 4) | ((c1 >> 4) << 2)
        hi = torch.stack(
            (nib[..., 0] | (nib[..., 2] << 4), nib[..., 1] | (nib[..., 3] << 4)), -1
        )  # u16 j, nibble order 0, 2, 1, 3
        rec = torch.cat((lo.flatten(-2), hi.flatten(-2)), -1).to(torch.uint8)
        stream[e0 : e0 + step] = rec.permute(0, 2, 1, 3)
    return table, stream.view(-1)


def decode(table: torch.Tensor, stream: torch.Tensor, shape) -> torch.Tensor:
    """The kernel's decode in torch: stream -> [E, G, N] bytes."""
    e, g, n = shape
    nt, kb = n // TILE_N, g // 2
    out = torch.empty(e, g, n, dtype=torch.uint8, device=stream.device)
    tab = table.long()
    s_all = stream.view(e, nt, kb, STAGE)
    step = 16
    for e0 in range(0, e, step):
        r = s_all[e0 : e0 + step].permute(0, 2, 1, 3).long()  # [e', KB, NT, 96]
        lo = r[..., 0:64:4] | (r[..., 1:64:4] << 8) | (r[..., 2:64:4] << 16) | (r[..., 3:64:4] << 24)
        hi = r[..., 64:96:2] | (r[..., 65:96:2] << 8)
        s = (hi | (hi << 12)) & 0x0F0F0F0F
        c0 = (lo & 0x0F0F0F0F) | ((s << 4) & 0x30303030)
        c1 = ((lo >> 4) & 0x0F0F0F0F) | ((s << 2) & 0x30303030)
        grp = torch.stack([torch.stack([(c >> (8 * i)) & 255 for i in range(4)], -1)
                           for c in (c0, c1)], 2)  # [e', KB, 2, NT, 16, 4]
        out[e0 : e0 + step] = tab[grp].to(torch.uint8).view(-1, g, n)
    return out


def prepare_layer(layer: torch.nn.Module) -> bool:
    """Attach layer.nvsc_w13_bs (the compressed stream, launcher-shaped) and
    re-point the stock w13 scale parameter into the same allocation. Any
    failure leaves the layer unchanged on the stock kernel."""
    name = str(getattr(layer, "layer_name", "?"))
    layer.nvsc_w13_bs = None
    if not ENABLED:
        return False
    try:
        meta = layer.humming_metas["w13"]
        param = getattr(layer, meta.weight_scale_name)
        ws = param.data
        if ws.dtype != torch.float8_e4m3fn or ws.dim() != 3 or not ws.is_contiguous():
            _STATS["stock"][name] = "layout"
            return False
        e, g, n = ws.shape
        if (meta.shape_n, meta.shape_k) != (n, g * 16) or n % TILE_N:
            _STATS["stock"][name] = "shape"
            return False
        enc = encode(ws)
        if enc is None:
            _STATS["stock"][name] = f"{int(torch.unique(ws.view(torch.uint8)).numel())} values"
            logger.info("nvsc: %s keeps stock w13 scales (more than 64 values)", name)
            return False
        table, stream = enc
        size = ws.numel()
        assert HEADER + stream.numel() <= size
        buf = torch.zeros(HEADER + stream.numel() + size, dtype=torch.uint8, device=ws.device)
        buf[:NUM_CODES] = table
        buf[HEADER : HEADER + stream.numel()] = stream
        buf[HEADER + stream.numel():] = ws.view(torch.uint8).flatten()
        stock = buf[HEADER + stream.numel():].view(torch.float8_e4m3fn).view(e, g, n)
        got = decode(buf[:NUM_CODES], buf[HEADER : HEADER + stream.numel()], (e, g, n))
        if not torch.equal(got, ws.view(torch.uint8)) or not torch.equal(
            stock.view(torch.uint8), ws.view(torch.uint8)
        ):
            _STATS["stock"][name] = "verify"
            logger.warning("nvsc: %s decode mismatch; keeping stock w13 scales", name)
            return False
        param.data = stock
        layer.nvsc_w13_bs = buf[:size].view(torch.float8_e4m3fn).view(e, g, n)
        _STATS["packed"].append(name)
        return True
    except Exception as exc:  # noqa: BLE001
        _STATS["stock"][name] = f"error {exc}"
        logger.warning("nvsc: %s repack failed; keeping stock w13 scales: %s", name, exc)
        layer.nvsc_w13_bs = None
        return False


# ---------------------------------------------------------------------------
# dispatch


def _configs(meta, compute_config: str, tuning_config: str, dyn: bool = False) -> torch.Tensor:
    from vllm.model_executor.layers.fused_moe.experts.humming_relu2_epilogue import (
        _dyn_num_sms,
        _variant_key,
        _w13_k128_params,
    )

    key = (meta.to_str(), compute_config, tuning_config) + _variant_key(dyn)
    if key in _CONFIGS:
        return _CONFIGS[key]
    if not _KERNEL_CLASSES:  # one class object per name
        _KERNEL_CLASSES.update(_kernel_classes())
    kernel_cls = _KERNEL_CLASSES[dyn]
    lc = json.loads(meta.to_str())
    lc.pop("sublayer_name", None)
    cc = json.loads(compute_config)
    assert cc["gemm_type"] == "indexed" and cc["use_batch_invariant"]
    tc = json.loads(tuning_config)
    if isinstance(tc, dict):
        tc = [[0, 1 << 30, tc]]
    result = []
    covered = 0
    for lo, hi, config in tc:
        if lo >= DECODE_MAX_VALID_M:
            continue
        assert lo == covered and hi <= DECODE_MAX_VALID_M, (lo, hi)
        assert config["block_shape"][1:] == [64, 32], config
        assert config["warp_shape"][1:] == [32, 32], config
        assert not config.get("use_stream_k", False)
        params = _w13_k128_params(lo, hi, lc | cc | config)
        num_sms = params.pop("num_sms", 0)
        if dyn:  # the claim-when-free decode grid, as for the stock dyn kernel
            num_sms = _dyn_num_sms(lo, hi, num_sms)
        bucket_cls = kernel_cls
        from vllm.model_executor.layers.fused_moe.experts.humming_relu2_epilogue import _VSW_EVICT
        if (lo, hi) == (24, 48) and _VSW_EVICT:
            if not _PF_KERNEL_CLASSES:
                _PF_KERNEL_CLASSES.update(_prefill_kernel_classes())
            bucket_cls = _PF_KERNEL_CLASSES[dyn]
        kernel = bucket_cls(**params)
        kernel.load_cubin()
        assert type(kernel) is bucket_cls
        if (lo, hi) == (24, 48):
            logger.info_once("VSW_VERIFY_NVSC stages=%d sms=%d evict=%s",
                             params["num_stages"], num_sms,
                             type(kernel) is _PF_KERNEL_CLASSES.get(dyn))
        assert "nvsc_table" in kernel.code
        assert ("hm_dyn_claim" in kernel.code) == dyn
        result.extend([lo, hi, kernel.kernel_id, num_sms])
        covered = hi
    assert covered == DECODE_MAX_VALID_M, covered
    configs = torch.tensor(result, dtype=torch.int64, device="cpu")
    _CONFIGS[key] = configs
    return configs


def _prefill_configs(meta, compute_config: str, tuning_config: str, dyn: bool) -> torch.Tensor:
    """One-entry config list: the record's evict-first prefill entry (same
    tuning config and persistent grid) with the nvsc kernel."""
    from humming.kernel.humming import HummingKernel

    from vllm.model_executor.layers.fused_moe.experts import humming_relu2_epilogue as fused

    key = (meta.to_str(), compute_config, tuning_config) + fused._variant_key(dyn)
    if key in _PF_CONFIGS:
        return _PF_CONFIGS[key]
    if not _PF_KERNEL_CLASSES:
        _PF_KERNEL_CLASSES.update(_prefill_kernel_classes())
    kernel_cls = _PF_KERNEL_CLASSES[dyn]
    evict = fused.prepare(meta, compute_config, tuning_config, evict_first=True, dyn=dyn).tolist()
    tc = json.loads(tuning_config)
    if isinstance(tc, dict):
        tc = [[0, 1 << 30, tc]]
    lc = json.loads(meta.to_str())
    lc.pop("sublayer_name", None)
    result = []
    for i, (lo, hi, config) in enumerate(tc):
        if not lo < PREFILL_VALID_M <= hi:
            continue
        original = HummingKernel._id2kernel[evict[4 * i + 2]]
        want = (fused.DynEvictFirstRelu2HummingKernel if dyn
                else fused.EvictFirstRelu2HummingKernel)
        # The record entry may itself be the pfup-derived class (humming_pfup.build).
        assert isinstance(original, want), type(original)
        assert config["block_shape"] == [64, 64, 32], config
        assert config["warp_shape"][1:] == [32, 32], config
        assert not config.get("use_stream_k", False)
        params = lc | json.loads(compute_config) | config
        params.pop("num_sms", None)
        if dyn:
            # LOCAL PATCH (cudafast-nemotron35, pfup): prefill claim order and
            # padded-row MMA skip (humming_pfup.py).
            from vllm.model_executor.layers.fused_moe.experts import humming_pfup

            kernel = humming_pfup.build(kernel_cls, params)
        else:
            kernel = kernel_cls(**params)
            kernel.load_cubin()
        assert isinstance(kernel, kernel_cls)
        assert "nvsc_table" in kernel.code and "nvsc_evf_load" in kernel.code
        assert ("hm_dyn_claim" in kernel.code) == dyn
        assert "cp.async.cg.shared.global.L2::cache_hint" in kernel.code
        assert kernel.kernel_filename != original.kernel_filename
        result.extend([0, 1 << 30, kernel.kernel_id, evict[4 * i + 3]])
    assert len(result) == 4, result
    configs = torch.tensor(result, dtype=torch.int64, device="cpu")
    _PF_CONFIGS[key] = configs
    return configs


def launch_prefill(layer, inputs, outputs, compute_config, tuning_config, sorted_ids,
                   expert_ids, num_tokens_padded, top_k, valid_shape_m, dyn=None) -> bool:
    """The 496-token routed up GEMM (+ReLU^2) with compressed scales in the
    evict-first prefill kernel. False (nothing launched) when not covered."""
    from vllm.model_executor.layers.fused_moe.experts.humming_relu2_epilogue import (
        _dyn_enabled,
        _variant_key,
    )

    bs = getattr(layer, "nvsc_w13_bs", None)
    if bs is None or not ENABLED or not PREFILL_ENABLED:
        return False
    if valid_shape_m != PREFILL_VALID_M or top_k != 6:
        return False
    dyn = _dyn_enabled() if dyn is None else bool(dyn)
    variant = _variant_key(dyn)
    if variant in _PF_FAILED:
        return False
    # Experimental adjacent-N reuse is a strict prefill subset. A refusal
    # launches nothing and keeps this complete original NVSC implementation.
    from vllm.model_executor.layers.fused_moe.experts import humming_apair

    if humming_apair.launch_prefill(
        layer, inputs, outputs, compute_config, tuning_config, sorted_ids,
        expert_ids, num_tokens_padded, top_k, valid_shape_m, dyn,
    ):
        return True
    from humming import ops

    meta = layer.humming_metas["w13"]
    try:
        configs = _prefill_configs(meta, compute_config, tuning_config, dyn)
    except Exception as exc:  # noqa: BLE001
        _PF_FAILED.add(variant)
        logger.warning("nvsc: prefill kernel unavailable (dyn=%s); using stock: %s", dyn, exc)
        return False
    logger.info_once("nvsc: 6-bit w13 block scales in the prefill kernel on %d routed layers",
                     len(_STATS["packed"]))
    ops.launch_kernel(
        configs=configs,
        inputs=inputs,
        weight=getattr(layer, meta.weight_name),
        outputs=outputs,
        input_scale=None,
        weight_scale=bs,
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
    return True


def launch(layer, inputs, outputs, compute_config, tuning_config, sorted_ids,
           expert_ids, num_tokens_padded, top_k, valid_shape_m, dyn=None) -> bool:
    """Run the routed up GEMM (+ReLU^2) with compressed scales. False (and
    nothing launched) when the layer or call shape is not covered. dyn: the
    claim-when-free scheduler (None: HUMMING_W13_DYN, default on)."""
    from vllm.model_executor.layers.fused_moe.experts.humming_relu2_epilogue import (
        _dyn_enabled,
        _variant_key,
    )

    bs = getattr(layer, "nvsc_w13_bs", None)
    if bs is None or not ENABLED:
        return False
    dyn = _dyn_enabled() if dyn is None else bool(dyn)
    variant = _variant_key(dyn)
    if variant in _FAILED:
        return False
    if not (0 < valid_shape_m <= DECODE_MAX_VALID_M) or inputs.shape[0] > 8:
        return False
    from humming import ops

    meta = layer.humming_metas["w13"]
    try:
        configs = _configs(meta, compute_config, tuning_config, dyn)
    except Exception as exc:  # noqa: BLE001
        _FAILED.add(variant)
        logger.warning("nvsc: derived kernel unavailable (dyn=%s); using stock: %s", dyn, exc)
        return False
    logger.info_once(
        "nvsc: 6-bit w13 block scales at decode on %d routed layers; stock on %s",
        len(_STATS["packed"]),
        ", ".join(f"{k} ({v})" for k, v in sorted(_STATS["stock"].items())) or "none",
    )
    ops.launch_kernel(
        configs=configs,
        inputs=inputs,
        weight=getattr(layer, meta.weight_name),
        outputs=outputs,
        input_scale=None,
        weight_scale=bs,
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
    return True

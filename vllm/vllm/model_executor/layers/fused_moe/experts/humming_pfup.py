# SPDX-License-Identifier: Apache-2.0
"""LOCAL PATCH (cudafast-nemotron35, pfup): the 496-token routed-up (w13 +
ReLU^2) humming kernel with two exact changes to its claim-when-free kernel
(humming_relu2_epilogue._DYN_SCHEDULER). Only the prefill entry uses it.

PFUP_REGROUP (default on): claim order. The record claims tiles m block
major (tile t -> m = t / 29, n = t % 29), so an expert with more than 64
routed rows reads its 2.8 MB of weights once per m block, 29 claims apart.
Here the m blocks of one expert (a run of equal expert_ids, which
moe_align_block_size makes contiguous) are claimed n block major inside
the run: the run [g0, g0 + s) covers the same tiles [g0 * 29, (g0 + s) * 29)
as before, tile t -> n = (t - 29 g0) / s, m = g0 + (t - 29 g0) % s, so the
s tiles that read the same weight slice are claimed together and the later
ones hit L2. A bijection on the same tile set: each tile is still computed
whole by one CTA over all K blocks in ascending order (no stream-K).

PFUP_MSKIP (default on): a 64-row expert block holds the expert's rows and
then padding (sorted id = M * top_k, row index = M). The A loader already
skips the padding rows' loads and the epilogue skips their stores; their
accumulators are never read. A warp's m16 group whose 16 rows are all
padding skips its mma.sync, and a warp whose 32 rows are all padding also
skips the B dequantisation that only feeds those MMAs. Every MMA of a row
that is stored still runs, in the same order, with the same operands.

The per-block expert ids (u8) and the tile's row-validity masks live in a
256-byte static shared array (128-byte aligned, a multiple of 128 bytes,
like nvsc_table), so humming's dynamic shared memory base stays aligned.
Grids above 240 m blocks fall back to the record order.
"""

from __future__ import annotations

import os

REGROUP = os.environ.get("PFUP_REGROUP", "1") != "0"
MSKIP = os.environ.get("PFUP_MSKIP", "1") != "0"

_SCHED = r"""
// LOCAL PATCH (cudafast-nemotron35, pfup): claim-when-free prefill w13 scheduler.
#define PFUP_REGROUP {regroup}
#define PFUP_MSKIP {mskip}
#define Scheduler HummingStaticScheduler
#include <humming/scheduler.cuh>
#undef Scheduler

__device__ uint32_t hm_dyn_claim = 0;
__device__ uint32_t hm_dyn_done = 0;
// bytes 0..239: expert id of m block i (u8); words 60-63: row-group validity (see get_next_block).
__shared__ __align__(128) uint32_t hm_pf_smem[64];
#define HM_PF_MAX_MBLOCKS 240u

template <
    class SharedStorage,
    class ProblemShape, class BlockShape,
    class LayerConfig, class ComputeConfig, class TuningConfig>
class Scheduler : public HummingStaticScheduler<
    SharedStorage, ProblemShape, BlockShape, LayerConfig, ComputeConfig, TuningConfig> {{
  using Base = HummingStaticScheduler<
      SharedStorage, ProblemShape, BlockShape, LayerConfig, ComputeConfig, TuningConfig>;
  static_assert(!TuningConfig::kUseStreamK);
  static_assert(!TuningConfig::kUseWarpSpec);
  static_assert(TuningConfig::kMultiCastSizeA * TuningConfig::kMultiCastSizeB == 1);
  static_assert(ComputeConfig::kGemmType == GemmType::INDEXED);
  static_assert((BlockShape::M == 64 || BlockShape::M == 32) && BlockShape::N == 64);
  static_assert(TuningConfig::kNumThreads >= 64);
  static constexpr uint32_t N_BLOCKS = ProblemShape::N / BlockShape::N;
  static constexpr uint32_t K_BLOCKS = ProblemShape::K / BlockShape::K;
  uint32_t total_tiles;
  uint32_t pf_m_blocks;

 public:
  CUDA_INLINE
  Scheduler(
      SharedStorage &smem, const void *output_ptr, CUtensorMap *tensor_map_buffer,
      uint32_t shape_m, uint32_t top_k, const uint32_t *row_index_blocks, const uint32_t *expert_ids,
      const uint32_t *num_tokens_padded_ptr, const uint32_t *expert_layout_ptr, bool use_int64_expert_layout)
      : Base(smem, output_ptr, tensor_map_buffer, shape_m, top_k, row_index_blocks, expert_ids,
             num_tokens_padded_ptr, expert_layout_ptr, use_int64_expert_layout) {{
    pf_m_blocks = CEIL_DIV(num_tokens_padded_ptr[0], BlockShape::M);
    total_tiles = pf_m_blocks * N_BLOCKS;
#if PFUP_REGROUP
    if (pf_m_blocks <= HM_PF_MAX_MBLOCKS) {{
      uint8_t *eid = reinterpret_cast<uint8_t *>(hm_pf_smem);
      for (uint32_t i = threadIdx.x; i < pf_m_blocks; i += TuningConfig::kNumThreads)
        eid[i] = static_cast<uint8_t>(expert_ids[i]);
    }}
    __syncthreads();
#endif
  }}

  CUDA_INLINE
  bool get_next_block() {{
    // The claim travels through the row-index staging slot that
    // fetch_moe_index_block refills right after.
    uint32_t *slot = this->smem.wr_row_index;
    if (threadIdx.x == 0) {{
      const uint32_t tile = atomicAdd(&hm_dyn_claim, 1u);
      uint32_t m = tile / N_BLOCKS;
      uint32_t n = tile % N_BLOCKS;
#if PFUP_REGROUP
      if (tile < total_tiles && pf_m_blocks <= HM_PF_MAX_MBLOCKS) {{
        const uint8_t *eid = reinterpret_cast<const uint8_t *>(hm_pf_smem);
        const uint8_t e = eid[m];
        uint32_t g0 = m;
        uint32_t g1 = m + 1;
        while (g0 > 0 && eid[g0 - 1] == e) g0--;
        while (g1 < pf_m_blocks && eid[g1] == e) g1++;
        const uint32_t s = g1 - g0;
        const uint32_t local = tile - g0 * N_BLOCKS;
        m = g0 + local % s;
        n = local / s;
      }}
#endif
      slot[0] = tile;
      slot[1] = m;
      slot[2] = n;
    }}
    __syncthreads();
    const uint32_t tile = slot[0];
    const uint32_t m = slot[1];
    const uint32_t n = slot[2];
    __syncthreads();
    if (tile >= total_tiles) {{
      if (threadIdx.x == 0) {{
        if (atomicAdd(&hm_dyn_done, 1u) == gridDim.x - 1) {{
          atomicExch(&hm_dyn_claim, 0u);
          atomicExch(&hm_dyn_done, 0u);
        }}
      }}
      return false;
    }}
    this->m_block_id = m;
    this->n_block_id = n;
    this->k_block_id = 0;
    this->slice_iters = K_BLOCKS;
    this->fetch_moe_index_block();
#if PFUP_MSKIP
    // word 60: bit g = rows 16 g .. 16 g + 15 of the block hold a stored row; word 61: block rows.
    if (threadIdx.x < BlockShape::M) {{
      const bool valid = this->smem.rd_row_index[threadIdx.x] < this->shape_m;
      const uint32_t bits = __ballot_sync(0xFFFFFFFFu, valid);
      if (threadIdx.x % 32 == 0) {{
        const uint32_t g = ((bits & 0xFFFFu) != 0u ? 1u : 0u) | ((bits >> 16) != 0u ? 2u : 0u);
        hm_pf_smem[62 + threadIdx.x / 32] = g;
      }}
    }}
    __syncthreads();
    if (threadIdx.x == 0) {{
      uint32_t g = hm_pf_smem[62];
      if (BlockShape::M == 64) g |= hm_pf_smem[63] << 2;
      hm_pf_smem[60] = g;
      hm_pf_smem[61] = BlockShape::M;
    }}
    __syncthreads();
#endif
    return true;
  }}
}};
"""

_WMMA = r"""
// LOCAL PATCH (cudafast-nemotron35, pfup): skip the MMAs of all-padding m16 groups.
#include <humming/datatype/dequant.cuh>
#define WMMA HmStockWMMA
#include <humming/mma/wmma.cuh>
#undef WMMA

template <
    class MmaOpClass_, class SharedStorage, class ArithClass,
    class WarpShape,
    class ElementA, class ElementB,
    class LayerConfig>
struct WMMA : public HmStockWMMA<
    MmaOpClass_, SharedStorage, ArithClass, WarpShape, ElementA, ElementB, LayerConfig> {
  using Base = HmStockWMMA<
      MmaOpClass_, SharedStorage, ArithClass, WarpShape, ElementA, ElementB, LayerConfig>;
  using MmaOpClass = MmaOpClass_;
  using MmaShape = typename MmaOpClass::MmaShape;
  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  // Block N = 64 (asserted by the scheduler), so N_WARPS = 64 / WarpShape::N; warp w covers rows
  // WarpShape::M * (w / N_WARPS % M_WARPS) + [0, WarpShape::M) (S2RMemoryLoaderA: m_iter_id =
  // warp_id / N_WARPS % M_WARPS), its m16 group m rows 16 m further.
  static_assert((WarpShape::M == 32 || WarpShape::M == 16) && WarpShape::N == 32 && MmaShape::M == 16);
  static constexpr uint32_t kNWarps = 64 / WarpShape::N;
  static_assert(!LayerConfig::kUseFusedE8m0Scale);
  uint32_t pf_act = 3;

  CUDA_INLINE
  WMMA(SharedStorage &smem, ArithClass &arith) : Base(smem, arith) {}

  CUDA_INLINE
  void zero_accum() {
    Base::zero_accum();
    const uint32_t groups = hm_pf_smem[60];
    const uint32_t m_warps = hm_pf_smem[61] / WarpShape::M;
    const uint32_t g0 = (threadIdx.x / 32 / kNWarps % m_warps) * (WarpShape::M / 16);
    pf_act = (groups >> g0) & ((1u << (WarpShape::M / 16)) - 1u);
  }

  CUDA_INLINE
  void transform_b(uint32_t buffer_id) {
    if (pf_act) Base::transform_b(buffer_id);
  }

  CUDA_INLINE
  void run(uint32_t stage_id, uint32_t iter_id) {
    uint32_t buffer_id = iter_id % 2;
    PRAGMA_UNROLL
    for (uint32_t k = 0; k < kPartMmaShapeK / MmaShape::K; k++) {
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < WarpShape::N / MmaShape::N; j++) {
        PRAGMA_UNROLL
        for (uint32_t m = 0; m < WarpShape::M / MmaShape::M; m++) {
          if (pf_act & (1u << m)) {
            MmaOpClass::fma(this->regs_a[buffer_id][m][k], this->regs_b[buffer_id][j][k],
                            this->regs_c[0][m][j], this->regs_c[0][m][j]);
            this->arith.may_apply_as_and_bs_on_mma_c(this->regs_c_as_ptr(), m, j, k, iter_id);
          }
        }
      }
    }
  }
};
"""

_CLASSES: dict = {}
_FAILED: set = set()


def flags() -> tuple[bool, bool]:
    return REGROUP, MSKIP


def sched_text(regroup: bool, mskip: bool) -> str:
    text = _SCHED.format(regroup=int(regroup), mskip=int(mskip))
    return text + (_WMMA if mskip else "")


def make_class(base_cls, regroup: bool, mskip: bool):
    """A subclass of a claim-when-free (dyn) w13 kernel class with the pfup
    scheduler text; one class object per (base, flags), named apart so that
    KernelRuntime (interning by class name + config) keeps them distinct."""
    assert getattr(base_cls, "dyn_scheduler", False), base_cls
    key = (base_cls, bool(regroup), bool(mskip))
    if key not in _CLASSES:
        text = sched_text(regroup, mskip)

        def scheduler_code(self, _text=text):
            return _text

        name = f"Pfup{int(regroup)}{int(mskip)}0{base_cls.__name__}"
        _CLASSES[key] = type(name, (base_cls,), {"scheduler_code": scheduler_code})
    return _CLASSES[key]


def build(base_cls, params: dict):
    """The prefill kernel for base_cls with the env flags; on any build failure
    (logged once per base class) the base kernel itself."""
    from vllm.logger import init_logger

    if (REGROUP or MSKIP) and base_cls not in _FAILED:
        try:
            cls = make_class(base_cls, REGROUP, MSKIP)
            kernel = cls(**params)
            kernel.load_cubin()
            assert type(kernel) is cls
            assert "hm_pf_smem" in kernel.code
            return kernel
        except Exception as exc:  # noqa: BLE001
            _FAILED.add(base_cls)
            init_logger(__name__).warning(
                "pfup: prefill w13 variant unavailable for %s; using it unchanged: %s",
                base_cls.__name__, exc)
    kernel = base_cls(**params)
    kernel.load_cubin()
    return kernel

// zskip: humming-exact routed-expert DOWN projection (NVFP4 W4A16) for the
// Nemotron-3.5 MTP verify (<= 4 tokens), that does not load the weight rows
// whose activations are exactly zero.
//
// What stock humming 0.1.10 computes for this call (indexed gemm, batch-
// invariant tuned config block 16x128x32, warp 16x32x32, no stream-K, see
// ../README.md): for every output element (block position p, column n)
//   acc = +0
//   for g = 0 .. 115 (k16 groups, ascending):
//     acc = mma.sync.m16n8k16.row.col.f32.bf16.bf16.f32(A[p, 16g:16g+16],
//                                                      B[16g:16g+16, n], acc)
//   out = cvt.rn.bf16(mul.ftz(acc, mul.ftz(gs[e], 4)))
// with A = the bf16 relu^2 activations in natural k order (MMA row = block
// position p, k-slot = k % 16) and
//   B[k, n] = hmul2(hmul2(fp4bits(k, n), 2^124), hmul2(fp8bits(g, n), 2^120))
// where fp4bits/fp8bits place the e2m1 nibble / e4m3 byte into a bf16 by bit
// placement (value x 2^-126 / x 2^-120): B = e2m1 * e4m3 / 4 exactly.
//
// This kernel performs the same operations on the same operand positions in
// the same order. Only the weight layout (a load-time repack, every byte
// kept: k-major rows, see py/__init__.py) and the loads differ:
//   mode DENSE: every row is loaded (humming's arithmetic, element for element);
//   mode SKIP:  for each expert block and k16 group, the 16-bit mask of k with
//     A != 0 (|bits| != 0) in ANY valid row of the block is computed from A.
//     Rows with mask bit 0 are not loaded; their B fragment halves are set to
//     +0 (bits 0x0000); the scales of a group with mask 0 are not loaded and
//     its MMAs are not issued. Exactness argument (tested on the device by
//     zskip_probe): with acc != -0, mma(acc, a, b) == mma(acc, a, b') when b'
//     differs from b only at k-slots where a == 0 and b' = +0, and
//     mma(acc, 0, b) == acc for acc not -0 / NaN (NaN is canonicalised by the
//     MMA, the epilogue multiply canonicalises it again). If a -0 accumulator
//     ever appears in a warp, that warp recomputes its tile in DENSE mode.
//
// Sources: this header holds the kernels, the config lists and the declarations of the per-config
// entry points; zskip_launch.cuh defines the entry points (kernel launches and attribute queries)
// and each zskip_<part>.cu instantiates them, with their kernels, for its part of the config lists;
// zskip.cu holds the C API, the kernels that several configs launch (prep_kernel,
// c3lpt_order_kernel) and the probes. Each .cu is compiled on its own and all are linked into
// libzskip.so (build.sh).

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace zskip {

#ifndef ZSKIP_NO_PDL
// Programmatic dependent launch (sm_90+). Both v4 kernels call these first, before
// any global memory access; launched without the PDL attribute they are no-ops.
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_launch_dependents() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
#endif

constexpr int K = 1856;           // intermediate size (down K)
constexpr int N = 2688;           // hidden size (down N)
constexpr int G = K / 16;         // 116 k16 groups
constexpr int RMAX = 64;          // valid rows per expert block (block_m 16 decode .. 64 prefill)
constexpr int WARP_COLS = 32;
constexpr int NWARPS_TOTAL = N / WARP_COLS;  // 84

enum Mode { DENSE = 0, SKIP = 1, LOADONLY = 2, LOADONLY_DENSE = 3, SETUPONLY = 4 };  // 2-4: timing probes

struct Params {
  const __nv_bfloat16* A;   // [shape_m, lda] relu^2 activations (rows = token*top_k + slot)
  long long lda;            // elements
  int shape_m;              // rows of A and C
  const int* sorted_ids;    // moe_align_block_size (block 16)
  const int* expert_ids;
  const int* num_tokens_padded;
  const uint8_t* W;         // repacked e2m1 nibbles
  const uint8_t* S;         // repacked e4m3 scales
  const float* GS;          // [E] fp32 global scales (weight_scale_2)
  __nv_bfloat16* C;         // [shape_m, ldc]
  long long ldc;            // elements
  long long w_expert, w_tile, w_k;  // byte strides: expert, 128-column tile, k row
  long long s_expert, s_tile, s_g;  // byte strides: expert, 128-column tile, k16 group
  int mode;
  int* stats;               // optional [4]: 0 dense redo items, 1 blocks with > RMAX rows
  uint8_t* ws;              // [max_blocks * BI_BYTES] per-block lists (prep_kernel -> down3_kernel)
  int* queue;               // work-item counter (reset by prep_kernel)
  int max_blocks;           // expert blocks covered (grid of prep_kernel)
  unsigned long long* trace;  // optional timing trace: 8 words per item (see down3_kernel)
  const int* verify_rows;  // optional confidence prefix, paired M5 only
  int c3lpt;  // shape30/config113 only: read same-call prep permutation
};

__device__ __forceinline__ int prefix_row_limit(const Params& p) {
  if (p.verify_rows == nullptr) return p.shape_m;
  const int rows = *p.verify_rows;
  // Avoid multiplying the ALL_ROWS sentinel. Eligible calls have 5 or 6 tokens x6.
  const int ntok = p.shape_m / 6;
  return rows >= ntok ? p.shape_m : (rows < 1 ? 1 : rows) * 6;
}

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ unsigned smid() {
  unsigned r;
  asm volatile("mov.u32 %0, %smid;" : "=r"(r));
  return r;
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N_>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N_) : "memory"); }

__device__ __forceinline__ uint32_t lds_u32(const void* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}
__device__ __forceinline__ uint32_t lds_u16(const void* p) {
  return *reinterpret_cast<const uint16_t*>(p);
}

__device__ __forceinline__ uint32_t hmul2_u32(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}

// humming fp_to_fp<Float4E2M1, BFloat16>: nibbles at bits 12..15 and 28..31.
__device__ __forceinline__ uint32_t fp4_to_bf16x2(uint32_t v) {
  return (v & 0x80008000u) | ((v & 0x70007000u) >> 6);
}
// humming fp_to_fp<Float8E4M3, BFloat16>: bytes at bits 8..15 and 24..31.
__device__ __forceinline__ uint32_t fp8_to_bf16x2(uint32_t v) {
  return (v & 0x80008000u) | ((v & 0x7F007F00u) >> 4);
}

// prepare_exp_scale_factor<nv_bfloat162, E>: (E * 0x00010001 + 0x007F007F) << 7
constexpr uint32_t kExp124 = (124u * 0x00010001u + 0x007F007Fu) << 7;  // 2^124 in both halves
constexpr uint32_t kExp120 = (120u * 0x00010001u + 0x007F007Fu) << 7;  // 2^120

__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float mul_ftz(float a, float b) {
  float r;
  asm("mul.rn.ftz.f32 %0, %1, %2;\n" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ uint32_t to_bf16x2(float lo, float hi) {
  __nv_bfloat162 r = __float22bfloat162_rn(make_float2(lo, hi));
  return *reinterpret_cast<uint32_t*>(&r);
}

// bit 2i / 2i+1: bf16 element 2i / 2i+1 of the 16-byte chunk has nonzero magnitude
__device__ __forceinline__ uint32_t nz_mask8(uint4 v) {
  uint32_t m = 0;
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    m |= ((w[i] & 0x7FFFu) != 0u ? 1u : 0u) << (2 * i);
    m |= ((w[i] & 0x7FFF0000u) != 0u ? 1u : 0u) << (2 * i + 1);
  }
  return m;
}

template <int N_MAX>
__device__ __forceinline__ void cp_async_wait_dyn(int n) {
  // waits until at most n of this thread's commit groups are pending (n >= 15 waits for 15)
  switch (n < 0 ? 0 : n) {
    case 0: cp_async_wait<0>(); break;
    case 1: cp_async_wait<1>(); break;
    case 2: cp_async_wait<2>(); break;
    case 3: cp_async_wait<3>(); break;
    case 4: cp_async_wait<4>(); break;
    case 5: cp_async_wait<5>(); break;
    case 6: cp_async_wait<6>(); break;
    case 7: cp_async_wait<7>(); break;
    case 8: cp_async_wait<8>(); break;
    case 9: cp_async_wait<9>(); break;
    case 10: cp_async_wait<10>(); break;
    case 11: cp_async_wait<11>(); break;
    case 12: cp_async_wait<12>(); break;
    case 13: cp_async_wait<13>(); break;
    case 14: cp_async_wait<14>(); break;
    default: cp_async_wait<15>(); break;
  }
}

// ---------------------------------------------------------------------------
// Per expert block info, built once per call by prep_kernel into p.ws (BI_BYTES per block):
//   int nrows, nact, nag, expert; int rows[4]; int8 pos_row[16] (block position -> row index or
//   RMAX); u16 mask[128] (bit b of group g: k = 16g + b nonzero in some valid row); u16 rbase[128]
//   (active rows before group g); u16 gbase[128] (active groups before g); u16 act_g[128] (active
//   groups in order); u16 act_k[K] (active k in order).
constexpr int BI_HDR = 320;
constexpr int BI_MASK = BI_HDR;
constexpr int BI_RBASE = BI_MASK + 256;
constexpr int BI_GBASE = BI_RBASE + 256;
constexpr int BI_ACTG = BI_GBASE + 256;
constexpr int BI_ACTK = BI_ACTG + 256;
constexpr int BI_USED = BI_ACTK + K * 2;  // 4800
constexpr int BI_BYTES = 5120;
static_assert(BI_USED <= BI_BYTES, "block info");

struct BlockHdr {
  int nrows, nact, nag, expert;
  int rows[RMAX];
  int8_t pos_row[16];
};
static_assert(sizeof(BlockHdr) <= BI_HDR, "block header");

// Builds the lists for one expert block into dst (shared or global memory). mask_of(g) returns the
// group's 16-bit mask. Needs all NT threads; ends with a barrier.
template <int NT>
__device__ __forceinline__ void build_lists(uint8_t* dst, int nrows_valid, const Params& p, const int* rows,
                                            bool dense) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  uint16_t* mask = reinterpret_cast<uint16_t*>(dst + BI_MASK);
  uint16_t* rbase = reinterpret_cast<uint16_t*>(dst + BI_RBASE);
  uint16_t* gbase = reinterpret_cast<uint16_t*>(dst + BI_GBASE);
  uint16_t* act_g = reinterpret_cast<uint16_t*>(dst + BI_ACTG);
  uint16_t* act_k = reinterpret_cast<uint16_t*>(dst + BI_ACTK);
  __shared__ int s_tot[2];
  const int row_limit = prefix_row_limit(p);
  for (int g = tid; g < 128; g += NT) {
    uint32_t m = 0;
    if (g < G) {
      if (dense) {
        m = 0xFFFFu;
      } else {
        for (int r = 0; r < nrows_valid; ++r) {
          if (rows[r] >= row_limit) continue;
          const uint4* q = reinterpret_cast<const uint4*>(p.A + (long long)rows[r] * p.lda + 16 * g);
          m |= nz_mask8(__ldg(q)) | (nz_mask8(__ldg(q + 1)) << 8);
        }
      }
    }
    mask[g] = static_cast<uint16_t>(m);
  }
  __syncthreads();
  if (warp == 0) {
    uint32_t cr[4], cg[4], sr = 0, sg = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t m = mask[4 * lane + i];
      cr[i] = __popc(m);
      cg[i] = m ? 1u : 0u;
      sr += cr[i];
      sg += cg[i];
    }
    uint32_t ir = sr, ig = sg;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const uint32_t vr = __shfl_up_sync(0xFFFFFFFFu, ir, d), vg = __shfl_up_sync(0xFFFFFFFFu, ig, d);
      if (lane >= d) {
        ir += vr;
        ig += vg;
      }
    }
    uint32_t er = ir - sr, eg = ig - sg;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      rbase[4 * lane + i] = static_cast<uint16_t>(er);
      gbase[4 * lane + i] = static_cast<uint16_t>(eg);
      er += cr[i];
      eg += cg[i];
    }
    if (lane == 31) {
      s_tot[0] = static_cast<int>(ir);
      s_tot[1] = static_cast<int>(ig);
    }
  }
  __syncthreads();
  for (int g = tid; g < G; g += NT) {
    uint32_t m = mask[g];
    if (m) act_g[gbase[g]] = static_cast<uint16_t>(g);
    int i = rbase[g];
    while (m) {
      const int b = __ffs(m) - 1;
      act_k[i++] = static_cast<uint16_t>(16 * g + b);
      m &= m - 1u;
    }
  }
  if (tid == 0) {
    BlockHdr* h = reinterpret_cast<BlockHdr*>(dst);
    h->nact = s_tot[0];
    h->nag = s_tot[1];
  }
  __syncthreads();
}

// C3LPT block order of the paired decode (defined in zskip.cu)
__global__ void __launch_bounds__(32) c3lpt_order_kernel(const Params p);

// one CTA per expert block (grid = max_blocks): rows, masks and lists into p.ws; block 0 also
// resets the work queue of the main kernel (launched next on the same stream)
template <int BM, int CAP>
__global__ void __launch_bounds__(128) prep_kernel(const Params p) {
#ifndef ZSKIP_NO_PDL
  pdl_wait();
  pdl_launch_dependents();
#endif
  const int tid = threadIdx.x, lane = tid & 31;
  const int mb = blockIdx.x;
  uint8_t* bi = p.ws + (long long)mb * BI_BYTES;
  BlockHdr* h = reinterpret_cast<BlockHdr*>(bi);
  __shared__ int s_rows[RMAX];
  __shared__ int s_nrows;
  if (mb == 0 && tid == 0) *p.queue = 0;
  if (mb * BM >= *p.num_tokens_padded) {
    if (tid == 0) {
      h->nrows = 0;
      h->nact = 0;
      h->nag = 0;
    }
    return;
  }
  if (tid < 32) {
    int base = 0;
#pragma unroll
    for (int c = 0; c < (BM + 31) / 32; ++c) {
      const int pos = 32 * c + lane;
      const unsigned id = pos < BM ? static_cast<unsigned>(p.sorted_ids[mb * BM + pos]) : 0xFFFFFFFFu;
      const bool valid = pos < BM && id < static_cast<unsigned>(p.shape_m);
      const unsigned ball = __ballot_sync(0xFFFFFFFFu, valid);
      const int rank = base + __popc(ball & ((1u << lane) - 1u));
      const bool keep = valid && rank < CAP;
      if (pos < 16) h->pos_row[pos] = static_cast<int8_t>(keep ? rank : RMAX);
      if (keep) {
        s_rows[rank] = static_cast<int>(id);
        h->rows[rank] = static_cast<int>(id);
      }
      base += __popc(ball);
    }
    if (lane == 0) {
      const int n = base;
      s_nrows = n < CAP ? n : CAP;
      h->nrows = s_nrows;
      h->expert = p.expert_ids[mb];
      if (n > CAP && p.stats) atomicAdd(p.stats + 1, 1);
    }
  }
  __syncthreads();
  const bool dense = p.mode == DENSE || p.mode == LOADONLY_DENSE;
  build_lists<128>(bi, s_nrows, p, s_rows, dense);
}
#ifndef ZSKIP_NO_FUSED

template <int NT, int BM, int CAP>
__device__ __forceinline__ void fused_prep(const Params& p, int mb, uint8_t* bi) {
  static_assert(NT >= 32 && NT % 32 == 0, "fused prep needs whole warps");
  const int tid = threadIdx.x, lane = tid & 31;
  BlockHdr* h = reinterpret_cast<BlockHdr*>(bi);
  __shared__ int s_rows[RMAX];
  __shared__ int s_nrows;
  if (tid < 32) {
    int base = 0;
#pragma unroll
    for (int c = 0; c < (BM + 31) / 32; ++c) {
      const int pos = 32 * c + lane;
      const unsigned id = pos < BM ? static_cast<unsigned>(p.sorted_ids[mb * BM + pos]) : 0xFFFFFFFFu;
      const bool valid = pos < BM && id < static_cast<unsigned>(p.shape_m);
      const unsigned ball = __ballot_sync(0xFFFFFFFFu, valid);
      const int rank = base + __popc(ball & ((1u << lane) - 1u));
      const bool keep = valid && rank < CAP;
      if (pos < 16) h->pos_row[pos] = static_cast<int8_t>(keep ? rank : RMAX);
      if (keep) {
        s_rows[rank] = static_cast<int>(id);
        h->rows[rank] = static_cast<int>(id);
      }
      base += __popc(ball);
    }
    if (lane == 0) {
      const int n = base;
      s_nrows = n < CAP ? n : CAP;
      h->nrows = s_nrows;
      h->expert = p.expert_ids[mb];
      if (n > CAP && p.stats && blockIdx.x == 0) atomicAdd(p.stats + 1, 1);
    }
  }
  __syncthreads();
  const bool dense = p.mode == DENSE || p.mode == LOADONLY_DENSE;
  build_lists<NT>(bi, s_nrows, p, s_rows, dense);
}

#endif

// CTA = W warps = 32W output columns (one column slice) of one expert block per work item.
// Shared memory (dynamic): rows ring NR*RB row segments of 16W bytes (active row i of the block's
// list -> slot i % (NR*RB)); group ring NQ*QB group slots of 32W scale bytes + 4 x 32 B of A
// (active group j -> slot j % (NQ*QB)); one 16 x 32 bf16 B tile per warp; the block info.
template <int W, int RB, int NR, int QB, int NQ>
struct Layout3 {
  static constexpr int SEG = 16 * W;
  static constexpr int GSLOT = 32 * W + 32 * RMAX;
  static constexpr int ROWS = NR * RB * SEG;
  static constexpr int GROUPS = NQ * QB * GSLOT;
  static constexpr int TILE_STRIDE = 80;  // bytes per k row of a warp's 16 x 32 bf16 tile (conflict-free ldmatrix)
  static constexpr int TILE = 16 * TILE_STRIDE;
  static constexpr int TILES = W * TILE;
  static constexpr int BYTES = ROWS + GROUPS + TILES + BI_BYTES;
  static constexpr int SLICES = NWARPS_TOTAL / W;
};

// groups with at most this many active rows build their B fragments through the warp's bf16 tile
// (dequantise only the active rows, ldmatrix.trans); denser groups dequantise fragments directly
#ifndef ZSKIP_TILE_MAX_ROWS
#define ZSKIP_TILE_MAX_ROWS 6
#endif

__device__ __forceinline__ void ldsm_x4_trans(uint32_t (&d)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(static_cast<unsigned>(__cvta_generic_to_shared(p))));
}

template <int W, int RB, int NR, int QB, int NQ>
__global__ void __launch_bounds__(W * 32) down3_kernel(const Params p) {
  using L = Layout3<W, RB, NR, QB, NQ>;
  constexpr int NT = W * 32;
  static_assert(RB >= 16, "a group's rows must span at most two row stages");
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* rring = smem;
  uint8_t* gring = smem + L::ROWS;
  uint8_t* tiles = smem + L::ROWS + L::GROUPS;
  uint8_t* bi = smem + L::ROWS + L::GROUPS + L::TILES;
  const BlockHdr* hdr = reinterpret_cast<const BlockHdr*>(bi);
  const uint16_t* s_mask = reinterpret_cast<const uint16_t*>(bi + BI_MASK);
  const uint16_t* rbase = reinterpret_cast<const uint16_t*>(bi + BI_RBASE);
  const uint16_t* gbase = reinterpret_cast<const uint16_t*>(bi + BI_GBASE);
  const uint16_t* act_g = reinterpret_cast<const uint16_t*>(bi + BI_ACTG);
  const uint16_t* act_k = reinterpret_cast<const uint16_t*>(bi + BI_ACTK);
  __shared__ int s_item;

  if (p.mode == SETUPONLY) return;  // timing probe: prep kernel and launch only
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int t = lane & 3, c = lane >> 2;
  const bool math = p.mode == DENSE || p.mode == SKIP;
  uint8_t* tile = tiles + warp * L::TILE;
  for (int i = lane; i < L::TILE / 4; i += 32) reinterpret_cast<uint32_t*>(tile)[i] = 0u;
  __syncwarp();
  // tile path, per lane: column n = lane of this warp's 32 = 8j + c' with c' = lane & 7, j = lane >> 3
  const int nib_byte = warp * 16 + 2 * (lane & 7) + (lane >> 4);
  const int nib_shift = 4 * ((lane >> 3) & 1);
  const int sc_byte = warp * 32 + 4 * (lane & 7) + (lane >> 3);
  const uint8_t* ldsm_addr0 = tile + ((lane & 7) + ((lane >> 3) & 1) * 8) * L::TILE_STRIDE + (lane >> 4) * 16;
  const uint8_t* ldsm_addr1 = ldsm_addr0 + 32;
  const int nitems = p.max_blocks * L::SLICES;

  for (;;) {
    if (tid == 0) s_item = atomicAdd(p.queue, 1);
    __syncthreads();
    const int item = s_item;
    if (item >= nitems) break;
    const int mb = item / L::SLICES, slice = item - mb * L::SLICES;
    unsigned long long tr[4] = {0, 0, 0, 0};
    if (p.trace && tid == 0) tr[0] = gtimer();
    {
      const uint8_t* src = p.ws + (long long)mb * BI_BYTES;
      for (int ch = tid; ch < BI_USED / 16; ch += NT) cp_async16(bi + ch * 16, src + ch * 16);
      cp_async_commit();
      cp_async_wait<0>();
      __syncthreads();
    }
    const int nrows = hdr->nrows;
    if (p.trace && tid == 0) tr[1] = gtimer();
    if (nrows == 0) {
      __syncthreads();
      continue;
    }
    const int e = hdr->expert;
    const int wg0 = slice * W;
    const uint8_t* wexp = p.W + e * p.w_expert;
    const uint8_t* sexp = p.S + e * p.s_expert;
    const int ra = hdr->pos_row[c], rb = hdr->pos_row[c + 8];
    const bool hi_any = __any_sync(0xFFFFFFFFu, hdr->pos_row[8 + (lane & 7)] < RMAX);
    bool dense = p.mode == DENSE || p.mode == LOADONLY_DENSE;
    float acc[4][4];

    for (int pass = 0; pass < 2; ++pass) {
      const int nact = hdr->nact, nag = hdr->nag;
      const int nsr = (nact + RB - 1) / RB, nsq = (nag + QB - 1) / QB;
      // row stages (RB active rows) and group stages (QB active groups) share this thread's
      // cp.async FIFO; cid_* = commit index of the stage held by each ring slot
      auto issue = [&](int sr, int sq) {
        int cr = 0, i0 = 0, cq = 0, j0 = 0;
        if (sr >= 0) {
          i0 = sr * RB;
          cr = min(RB, nact - i0);
        }
        if (sq >= 0) {
          j0 = sq * QB;
          cq = min(QB, nag - j0);
        }
        const int nr_ch = cr * W, ns_ch = cq * 2 * W, per_a = nrows * 2, na_ch = cq * per_a;
        const int total = nr_ch + ns_ch + na_ch;
        for (int ch = tid; ch < total; ch += NT) {
          const void* src;
          uint8_t* dst;
          if (ch < nr_ch) {
            const int i = i0 + ch / W, q = ch % W;
            const int k = act_k[i], wq = wg0 + q;
            src = wexp + (long long)k * p.w_k + (wq >> 2) * p.w_tile + (wq & 3) * 16;
            dst = rring + (i % (NR * RB)) * L::SEG + q * 16;
          } else if (ch < nr_ch + ns_ch) {
            const int c2 = ch - nr_ch, jj = j0 + c2 / (2 * W), q = c2 % (2 * W);
            const int gg = act_g[jj], wq = wg0 + (q >> 1);
            src = sexp + (long long)gg * p.s_g + (wq >> 2) * p.s_tile + (wq & 3) * 32 + (q & 1) * 16;
            dst = gring + (jj % (NQ * QB)) * L::GSLOT + q * 16;
          } else {
            const int c3 = ch - nr_ch - ns_ch, jj = j0 + c3 / per_a, rem = c3 % per_a;
            const int r = rem >> 1, h = rem & 1, gg = act_g[jj];
            src = p.A + (long long)hdr->rows[r] * p.lda + 16 * gg + 8 * h;
            dst = gring + (jj % (NQ * QB)) * L::GSLOT + 32 * W + r * 32 + h * 16;
          }
          cp_async16(dst, src);
        }
        cp_async_commit();
      };

#pragma unroll
      for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[j][i] = 0.0f;
      }
      int cid_r[NR], cid_q[NQ];
#pragma unroll
      for (int i = 0; i < NR; ++i) cid_r[i] = 0;
#pragma unroll
      for (int i = 0; i < NQ; ++i) cid_q[i] = 0;
      int pr = 0, pq = 0, ncommit = 0, done_r = 0, done_q = 0;
      bool neg0 = false;
      for (int g = 0; g < G; ++g) {
        const uint32_t m = s_mask[g];
        if (m == 0u) continue;  // all valid rows zero: no loads, no MMA (acc unchanged unless -0)
        const int j = gbase[g];
        const int need_r = (rbase[g + 1] - 1) / RB, need_q = j / QB;
        if (need_r >= done_r || need_q >= done_q) {
          __syncthreads();  // every thread is done with the slots of the groups before g
          const int lim_r = rbase[g] / RB + NR, lim_q = j / QB + NQ;
          for (;;) {
            const bool ir = pr < nsr && pr < lim_r, iq = pq < nsq && pq < lim_q;
            if (!ir && !iq) break;
            issue(ir ? pr : -1, iq ? pq : -1);
            if (ir) {
              cid_r[pr % NR] = ncommit;
              ++pr;
            }
            if (iq) {
              cid_q[pq % NQ] = ncommit;
              ++pq;
            }
            ++ncommit;
          }
          const int c_need = max(cid_r[need_r % NR], cid_q[need_q % NQ]);
          cp_async_wait_dyn<15>(ncommit - c_need - 1);
          __syncthreads();
          if (p.trace && tid == 0 && tr[2] == 0) tr[2] = gtimer();
          done_r = max(done_r, need_r + 1);
          done_q = max(done_q, need_q + 1);
        }
        if (!math) continue;
        const int rb0 = rbase[g];
        const uint8_t* gsl = gring + (j % (NQ * QB)) * L::GSLOT;
        const uint8_t* ga = gsl + 32 * W;
        const uint32_t a0 = ra < RMAX ? lds_u32(ga + ra * 32 + 4 * t) : 0u;
        const uint32_t a2 = ra < RMAX ? lds_u32(ga + ra * 32 + 16 + 4 * t) : 0u;
        uint32_t a1 = 0u, a3 = 0u;
        if (hi_any) {
          a1 = rb < RMAX ? lds_u32(ga + rb * 32 + 4 * t) : 0u;
          a3 = rb < RMAX ? lds_u32(ga + rb * 32 + 16 + 4 * t) : 0u;
        }
        uint32_t bfr[4][2];
        if (__popc(m) <= ZSKIP_TILE_MAX_ROWS) {
          // dequantise only the active rows into the zeroed tile, then ldmatrix.trans
          const uint32_t sv = hmul2_u32(fp8_to_bf16x2(static_cast<uint32_t>(gsl[sc_byte]) << 8), kExp120);
          const uint32_t sdup = __byte_perm(sv, 0, 0x1010);
          uint32_t mm = m;
          int i = rb0;
          while (mm) {
            const int r = __ffs(mm) - 1;
            const uint32_t byte = rring[(i % (NR * RB)) * L::SEG + nib_byte];
            uint32_t b = fp4_to_bf16x2(((byte >> nib_shift) & 0xFu) << 12);
            b = hmul2_u32(hmul2_u32(b, kExp124), sdup);
            *reinterpret_cast<uint16_t*>(tile + r * L::TILE_STRIDE + 2 * lane) = static_cast<uint16_t>(b);
            mm &= mm - 1u;
            ++i;
          }
          __syncwarp();
          uint32_t d[4];
          ldsm_x4_trans(d, ldsm_addr0);
          bfr[0][0] = d[0];
          bfr[0][1] = d[1];
          bfr[1][0] = d[2];
          bfr[1][1] = d[3];
          ldsm_x4_trans(d, ldsm_addr1);
          bfr[2][0] = d[0];
          bfr[2][1] = d[1];
          bfr[3][0] = d[2];
          bfr[3][1] = d[3];
          __syncwarp();
          mm = m;
          while (mm) {
            const int r = __ffs(mm) - 1;
            *reinterpret_cast<uint16_t*>(tile + r * L::TILE_STRIDE + 2 * lane) = 0;
            mm &= mm - 1u;
          }
        } else {
          uint32_t x[4];
#pragma unroll
          for (int h = 0; h < 4; ++h) {
            const int r = 2 * t + (h & 1) + (h >> 1) * 8;
            if ((m >> r) & 1u) {
              const int i = rb0 + __popc(m & ((1u << r) - 1u));
              x[h] = lds_u16(rring + (i % (NR * RB)) * L::SEG + warp * 16 + 2 * c);
            } else {
              x[h] = 0u;
            }
          }
          const uint32_t sc = lds_u32(gsl + warp * 32 + 4 * c);
          const uint32_t q01 = __byte_perm(x[0], x[1], 0x5410);
          const uint32_t q23 = __byte_perm(x[2], x[3], 0x5410);
          const uint32_t m01 = (m >> (2 * t)) & 3u;
          const uint32_t m23 = (m >> (2 * t + 8)) & 3u;
          const uint32_t keep01 = ((m01 & 1u) ? 0x0000FFFFu : 0u) | ((m01 & 2u) ? 0xFFFF0000u : 0u);
          const uint32_t keep23 = ((m23 & 1u) ? 0x0000FFFFu : 0u) | ((m23 & 2u) ? 0xFFFF0000u : 0u);
          const uint32_t s01 =
              hmul2_u32(fp8_to_bf16x2(((sc << 8) & 0x0000FF00u) | ((sc << 16) & 0xFF000000u)), kExp120);
          const uint32_t s23 = hmul2_u32(fp8_to_bf16x2(((sc >> 8) & 0x0000FF00u) | (sc & 0xFF000000u)), kExp120);
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) {
            const uint32_t sj = (jj < 2) ? __byte_perm(s01, 0, jj == 0 ? 0x1010 : 0x3232)
                                         : __byte_perm(s23, 0, jj == 2 ? 0x1010 : 0x3232);
            const uint32_t b0 = fp4_to_bf16x2(q01 << (12 - 4 * jj));
            const uint32_t b1 = fp4_to_bf16x2(q23 << (12 - 4 * jj));
            bfr[jj][0] = hmul2_u32(hmul2_u32(b0, kExp124), sj) & keep01;
            bfr[jj][1] = hmul2_u32(hmul2_u32(b1, kExp124), sj) & keep23;
          }
        }
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) mma_bf16(acc[jj], a0, a1, a2, a3, bfr[jj][0], bfr[jj][1]);
        if (!dense) {
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) {
            neg0 |= (__float_as_uint(acc[jj][0]) == 0x80000000u) | (__float_as_uint(acc[jj][1]) == 0x80000000u);
            if (hi_any)
              neg0 |= (__float_as_uint(acc[jj][2]) == 0x80000000u) | (__float_as_uint(acc[jj][3]) == 0x80000000u);
          }
        }
      }
      cp_async_wait<0>();
      if (!__syncthreads_or(neg0 ? 1 : 0) || dense || !math) break;
      // a -0 accumulator appeared: the skip may differ from humming in the sign of a zero, so
      // recompute this item with every row loaded (humming's arithmetic, no substitution)
      if (tid == 0 && p.stats) atomicAdd(p.stats + 0, 1);
      build_lists<NT>(bi, 0, p, nullptr, true);
      dense = true;
    }

    const float gs4 = mul_ftz(p.GS[e], 4.0f);
    const int wg = wg0 + warp;
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int col = wg * WARP_COLS + 8 * jj + 2 * t;
      if (ra < RMAX) {
        *reinterpret_cast<uint32_t*>(p.C + (long long)hdr->rows[ra] * p.ldc + col) =
            to_bf16x2(mul_ftz(acc[jj][0], gs4), mul_ftz(acc[jj][1], gs4));
      }
      if (rb < RMAX) {
        *reinterpret_cast<uint32_t*>(p.C + (long long)hdr->rows[rb] * p.ldc + col) =
            to_bf16x2(mul_ftz(acc[jj][2], gs4), mul_ftz(acc[jj][3], gs4));
      }
    }
    if (p.trace && tid == 0) {
      tr[3] = gtimer();
      unsigned long long* o = p.trace + 8ull * item;
      o[0] = tr[0];
      o[1] = tr[1];
      o[2] = tr[2];
      o[3] = tr[3];
      o[4] = smid();
      o[5] = blockIdx.x;
      o[6] = static_cast<unsigned long long>(hdr->nact) | (static_cast<unsigned long long>(hdr->nag) << 32);
      o[7] = static_cast<unsigned long long>(nrows);
    }
    __syncthreads();  // the next item overwrites the block info and the rings
  }
}

// ---------------------------------------------------------------------------
// v4: one warp per (expert block, 64-column tile), weights as the MMA A operand (16 output
// columns x 16 k), activations as B (16 k x 8 token slots), one fp32 chain per output element
// over the k16 groups in ascending order (the same products in the same k-slots as humming; the
// operand roles of the two bf16 inputs are swapped). Layout "swap" (repack): per expert and k row
// 1344 bytes = 42 tiles x 32 bytes; byte 4g + c of a tile = nibbles of columns 64T + 16c + g (low)
// and 64T + 16c + g + 8 (high); scales per k16 group 2688 bytes = 42 tiles x 64 bytes, byte
// 8g + 2c + h = scale of column 64T + 16c + g + 8h. Lane (g = lane/4, t = lane%4) loads the
// 4-byte words of k rows 2t, 2t+1, 2t+8, 2t+9 of each active group (inactive rows are not
// loaded: one k row of a tile is one 32-byte sector), its 8 scale bytes and its two activation
// words, through a register ring of R active groups.

__device__ __forceinline__ uint32_t ldg_u32(const void* p) {
  uint32_t v;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ uint2 ldg_u64(const void* p) {
  uint2 v;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t ldg_u32_l1(const void* p) {  // activations: L2 resident, reused
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}

template <int WPC, int R, int NH, int BM, bool Fused = false>
__global__ void __launch_bounds__(WPC * 32) down4_kernel(const Params p) {
#ifndef ZSKIP_NO_PDL
  pdl_wait();
  pdl_launch_dependents();
#endif
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g8 = lane >> 2, t = lane & 3;
  const int mb = blockIdx.y;
  const int T = blockIdx.x * WPC + warp;  // 64-column tile, 0..41
  if (mb * BM >= *p.num_tokens_padded) return;
  const uint8_t* bi = p.ws + (long long)mb * BI_BYTES;
#ifndef ZSKIP_NO_FUSED
  extern __shared__ __align__(16) uint8_t smem4[];
  if constexpr (Fused) {
    fused_prep<WPC * 32, BM, 8 * NH>(p, mb, smem4);
    bi = smem4;
  }
#endif
  const BlockHdr* hdr = reinterpret_cast<const BlockHdr*>(bi);
  const int nrows = hdr->nrows;
  if (nrows == 0 || p.mode == SETUPONLY) return;
  const int e = hdr->expert;
  const uint16_t* gmask = reinterpret_cast<const uint16_t*>(bi + BI_MASK);
  const uint16_t* gact = reinterpret_cast<const uint16_t*>(bi + BI_ACTG);
  const bool math = p.mode == DENSE || p.mode == SKIP;
  bool dense = p.mode == DENSE || p.mode == LOADONLY_DENSE;
  const __nv_bfloat16* xrow[NH];
#pragma unroll
  for (int h = 0; h < NH; ++h) xrow[h] = g8 + 8 * h < nrows ? p.A + (long long)hdr->rows[g8 + 8 * h] * p.lda : nullptr;
  const uint8_t* wbase = p.W + e * p.w_expert + T * 32 + g8 * 4;
  const uint8_t* sbase = p.S + e * p.s_expert + T * 64 + g8 * 8;
  // the k-slot pair of this lane in a group: rows 2t, 2t+1 (low k) and 2t+8, 2t+9 (high k)
  const int kr[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};

  float acc[NH][4][4];
  for (int pass = 0; pass < 2; ++pass) {
    const int nsteps = dense ? G : hdr->nag;
    uint32_t wq[R][4], xb[R][NH][2], mk[R];
    uint2 sq[R];
    auto load = [&](int slot, int j) {
      if (j < nsteps) {
        const int grp = dense ? j : gact[j];
        const uint32_t m = dense ? 0xFFFFu : gmask[grp];
        const int kb = 16 * grp;
#pragma unroll
        for (int h = 0; h < 4; ++h) {
          wq[slot][h] = ((m >> kr[h]) & 1u) ? ldg_u32(wbase + (long long)(kb + kr[h]) * p.w_k) : 0u;
        }
        sq[slot] = ldg_u64(sbase + (long long)grp * p.s_g);
#pragma unroll
        for (int h = 0; h < NH; ++h) {
          xb[slot][h][0] = xrow[h] ? ldg_u32_l1(xrow[h] + kb + 2 * t) : 0u;
          xb[slot][h][1] = xrow[h] ? ldg_u32_l1(xrow[h] + kb + 8 + 2 * t) : 0u;
        }
        mk[slot] = m;
      } else {
        mk[slot] = 0u;
      }
    };
#pragma unroll
    for (int c = 0; c < 4; ++c) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int h = 0; h < NH; ++h) acc[h][c][i] = 0.0f;
      }
    }
#pragma unroll
    for (int i = 0; i < R; ++i) load(i, i);
    bool neg0 = false;
    for (int j0 = 0; j0 < nsteps; j0 += R) {
#pragma unroll
      for (int i = 0; i < R; ++i) {
        if (j0 + i < nsteps) {
          const uint32_t m = mk[i];
          const uint32_t w0 = wq[i][0], w1 = wq[i][1], w2 = wq[i][2], w3 = wq[i][3];
          const uint2 sc = sq[i];
          uint32_t bx[NH][2];
#pragma unroll
          for (int h = 0; h < NH; ++h) { bx[h][0] = xb[i][h][0]; bx[h][1] = xb[i][h][1]; }
          load(i, j0 + i + R);
          if (math) {
            const uint32_t m01 = (m >> (2 * t)) & 3u;
            const uint32_t m23 = (m >> (2 * t + 8)) & 3u;
            const uint32_t keep01 = ((m01 & 1u) ? 0x0000FFFFu : 0u) | ((m01 & 2u) ? 0xFFFF0000u : 0u);
            const uint32_t keep23 = ((m23 & 1u) ? 0x0000FFFFu : 0u) | ((m23 & 2u) ? 0xFFFF0000u : 0u);
            // scales: byte 2c + h of sc -> column 16c + g8 + 8h; humming's fp8 bit placement, x 2^120
            const uint32_t s01 = hmul2_u32(fp8_to_bf16x2(((sc.x << 8) & 0x0000FF00u) | ((sc.x << 16) & 0xFF000000u)), kExp120);
            const uint32_t s23 = hmul2_u32(fp8_to_bf16x2(((sc.x >> 8) & 0x0000FF00u) | (sc.x & 0xFF000000u)), kExp120);
            const uint32_t s45 = hmul2_u32(fp8_to_bf16x2(((sc.y << 8) & 0x0000FF00u) | ((sc.y << 16) & 0xFF000000u)), kExp120);
            const uint32_t s67 = hmul2_u32(fp8_to_bf16x2(((sc.y >> 8) & 0x0000FF00u) | (sc.y & 0xFF000000u)), kExp120);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const uint32_t sp = c == 0 ? s01 : (c == 1 ? s23 : (c == 2 ? s45 : s67));
              const uint32_t slo = __byte_perm(sp, 0, 0x1010);  // column g8 (h = 0)
              const uint32_t shi = __byte_perm(sp, 0, 0x3232);  // column g8 + 8 (h = 1)
              const uint32_t sel = c | (c << 4) | ((c + 4) << 8) | ((c + 4) << 12);
              const uint32_t x01 = __byte_perm(w0, w1, sel);  // bytes: w0[c], w0[c], w1[c], w1[c]
              const uint32_t x23 = __byte_perm(w2, w3, sel);
              const uint32_t a0 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x01 << 12), kExp124), slo) & keep01;
              const uint32_t a1 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x01 << 8), kExp124), shi) & keep01;
              const uint32_t a2 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x23 << 12), kExp124), slo) & keep23;
              const uint32_t a3 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x23 << 8), kExp124), shi) & keep23;
#pragma unroll
              for (int h = 0; h < NH; ++h)
                if (h == 0 || nrows > 8 * h) mma_bf16(acc[h][c], a0, a1, a2, a3, bx[h][0], bx[h][1]);
            }
            if (!dense) {
#pragma unroll
              for (int c = 0; c < 4; ++c) {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
#pragma unroll
                  for (int h = 0; h < NH; ++h)
                    if (h == 0 || nrows > 8 * h) neg0 |= __float_as_uint(acc[h][c][q]) == 0x80000000u;
                }
              }
            }
          }
        }
      }
    }
    if (!__any_sync(0xFFFFFFFFu, neg0) || dense || !math) break;
    // a -0 accumulator appeared: recompute this warp's tile with every row loaded
    if (lane == 0 && p.stats) atomicAdd(p.stats + 0, 1);
    dense = true;
  }
  if (!math) return;

  const float gs4 = mul_ftz(p.GS[e], 4.0f);
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int col = T * 64 + 16 * c + g8;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int cc = col + ((q >> 1) ? 8 : 0);
#pragma unroll
      for (int h = 0; h < NH; ++h) {
        const int tok = 8 * h + 2 * t + (q & 1);
        if (tok < nrows) {
          const uint32_t v = to_bf16x2(mul_ftz(acc[h][c][q], gs4), 0.0f);
          reinterpret_cast<uint16_t*>(p.C)[(long long)hdr->rows[tok] * p.ldc + cc] = static_cast<uint16_t>(v);
        }
      }
    }
  }
}


#include "zskip_pair.cuh"
#include "zskip_lean.cuh"

// v6: down4 with the -0 guard moved to the end. With skipped rows read as +0, every step keeps
// the accumulator equal to stock's or both zero (the two differ at most in the sign of a zero), so
// a nonzero final accumulator equals stock's. A warp whose valid outputs include an exact zero
// (either sign) recomputes its tile in DENSE mode, as down4 does for a -0 seen at any step.
template <int WPC, int R, int NH, int BM>
__global__ void __launch_bounds__(WPC * 32) down6_kernel(const Params p) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g8 = lane >> 2, t = lane & 3;
  const int mb = blockIdx.y;
  const int T = blockIdx.x * WPC + warp;  // 64-column tile, 0..41
  if (mb * BM >= *p.num_tokens_padded) return;
  const uint8_t* bi = p.ws + (long long)mb * BI_BYTES;
  const BlockHdr* hdr = reinterpret_cast<const BlockHdr*>(bi);
  const int nrows = hdr->nrows;
  if (nrows == 0 || p.mode == SETUPONLY) return;
  const int e = hdr->expert;
  const uint16_t* gmask = reinterpret_cast<const uint16_t*>(bi + BI_MASK);
  const uint16_t* gact = reinterpret_cast<const uint16_t*>(bi + BI_ACTG);
  const bool math = p.mode == DENSE || p.mode == SKIP;
  bool dense = p.mode == DENSE || p.mode == LOADONLY_DENSE;
  const __nv_bfloat16* xrow[NH];
#pragma unroll
  for (int h = 0; h < NH; ++h) xrow[h] = g8 + 8 * h < nrows ? p.A + (long long)hdr->rows[g8 + 8 * h] * p.lda : nullptr;
  const uint8_t* wbase = p.W + e * p.w_expert + T * 32 + g8 * 4;
  const uint8_t* sbase = p.S + e * p.s_expert + T * 64 + g8 * 8;
  // the k-slot pair of this lane in a group: rows 2t, 2t+1 (low k) and 2t+8, 2t+9 (high k)
  const int kr[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};

  float acc[NH][4][4];
  for (int pass = 0; pass < 2; ++pass) {
    const int nsteps = dense ? G : hdr->nag;
    uint32_t wq[R][4], xb[R][NH][2], mk[R];
    uint2 sq[R];
    auto load = [&](int slot, int j) {
      if (j < nsteps) {
        const int grp = dense ? j : gact[j];
        const uint32_t m = dense ? 0xFFFFu : gmask[grp];
        const int kb = 16 * grp;
#pragma unroll
        for (int h = 0; h < 4; ++h) {
          wq[slot][h] = ((m >> kr[h]) & 1u) ? ldg_u32(wbase + (long long)(kb + kr[h]) * p.w_k) : 0u;
        }
        sq[slot] = ldg_u64(sbase + (long long)grp * p.s_g);
#pragma unroll
        for (int h = 0; h < NH; ++h) {
          xb[slot][h][0] = xrow[h] ? ldg_u32_l1(xrow[h] + kb + 2 * t) : 0u;
          xb[slot][h][1] = xrow[h] ? ldg_u32_l1(xrow[h] + kb + 8 + 2 * t) : 0u;
        }
        mk[slot] = m;
      } else {
        mk[slot] = 0u;
      }
    };
#pragma unroll
    for (int c = 0; c < 4; ++c) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int h = 0; h < NH; ++h) acc[h][c][i] = 0.0f;
      }
    }
#pragma unroll
    for (int i = 0; i < R; ++i) load(i, i);
    bool neg0 = false;
    for (int j0 = 0; j0 < nsteps; j0 += R) {
#pragma unroll
      for (int i = 0; i < R; ++i) {
        if (j0 + i < nsteps) {
          const uint32_t m = mk[i];
          const uint32_t w0 = wq[i][0], w1 = wq[i][1], w2 = wq[i][2], w3 = wq[i][3];
          const uint2 sc = sq[i];
          uint32_t bx[NH][2];
#pragma unroll
          for (int h = 0; h < NH; ++h) { bx[h][0] = xb[i][h][0]; bx[h][1] = xb[i][h][1]; }
          load(i, j0 + i + R);
          if (math) {
            const uint32_t m01 = (m >> (2 * t)) & 3u;
            const uint32_t m23 = (m >> (2 * t + 8)) & 3u;
            const uint32_t keep01 = ((m01 & 1u) ? 0x0000FFFFu : 0u) | ((m01 & 2u) ? 0xFFFF0000u : 0u);
            const uint32_t keep23 = ((m23 & 1u) ? 0x0000FFFFu : 0u) | ((m23 & 2u) ? 0xFFFF0000u : 0u);
            // scales: byte 2c + h of sc -> column 16c + g8 + 8h; humming's fp8 bit placement, x 2^120
            const uint32_t s01 = hmul2_u32(fp8_to_bf16x2(((sc.x << 8) & 0x0000FF00u) | ((sc.x << 16) & 0xFF000000u)), kExp120);
            const uint32_t s23 = hmul2_u32(fp8_to_bf16x2(((sc.x >> 8) & 0x0000FF00u) | (sc.x & 0xFF000000u)), kExp120);
            const uint32_t s45 = hmul2_u32(fp8_to_bf16x2(((sc.y << 8) & 0x0000FF00u) | ((sc.y << 16) & 0xFF000000u)), kExp120);
            const uint32_t s67 = hmul2_u32(fp8_to_bf16x2(((sc.y >> 8) & 0x0000FF00u) | (sc.y & 0xFF000000u)), kExp120);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const uint32_t sp = c == 0 ? s01 : (c == 1 ? s23 : (c == 2 ? s45 : s67));
              const uint32_t slo = __byte_perm(sp, 0, 0x1010);  // column g8 (h = 0)
              const uint32_t shi = __byte_perm(sp, 0, 0x3232);  // column g8 + 8 (h = 1)
              const uint32_t sel = c | (c << 4) | ((c + 4) << 8) | ((c + 4) << 12);
              const uint32_t x01 = __byte_perm(w0, w1, sel);  // bytes: w0[c], w0[c], w1[c], w1[c]
              const uint32_t x23 = __byte_perm(w2, w3, sel);
              const uint32_t a0 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x01 << 12), kExp124), slo) & keep01;
              const uint32_t a1 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x01 << 8), kExp124), shi) & keep01;
              const uint32_t a2 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x23 << 12), kExp124), slo) & keep23;
              const uint32_t a3 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x23 << 8), kExp124), shi) & keep23;
#pragma unroll
              for (int h = 0; h < NH; ++h)
                if (h == 0 || nrows > 8 * h) mma_bf16(acc[h][c], a0, a1, a2, a3, bx[h][0], bx[h][1]);
            }
          }
        }
      }
    }
    if (math && !dense) {
#pragma unroll
      for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
          for (int h = 0; h < NH; ++h)
            if (8 * h + 2 * t + (q & 1) < nrows) neg0 |= (__float_as_uint(acc[h][c][q]) & 0x7FFFFFFFu) == 0u;
    }
    if (!__any_sync(0xFFFFFFFFu, neg0) || dense || !math) break;
    // a -0 accumulator appeared: recompute this warp's tile with every row loaded
    if (lane == 0 && p.stats) atomicAdd(p.stats + 0, 1);
    dense = true;
  }
  if (!math) return;

  const float gs4 = mul_ftz(p.GS[e], 4.0f);
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int col = T * 64 + 16 * c + g8;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int cc = col + ((q >> 1) ? 8 : 0);
#pragma unroll
      for (int h = 0; h < NH; ++h) {
        const int tok = 8 * h + 2 * t + (q & 1);
        if (tok < nrows) {
          const uint32_t v = to_bf16x2(mul_ftz(acc[h][c][q], gs4), 0.0f);
          reinterpret_cast<uint16_t*>(p.C)[(long long)hdr->rows[tok] * p.ldc + cc] = static_cast<uint16_t>(v);
        }
      }
    }
  }
}


// ---------------------------------------------------------------------------
// v5 (prefill): down4's arithmetic, element for element (same fragments, same mma order, same
// -0 dense redo), but each warp stages its operands through a shared-memory ring of S active
// groups with cp.async: the active k rows of its 64-column tile (32 bytes each, inactive rows are
// not copied and read as 0), the tile's 64 scale bytes and the 32-byte k16 slices of the block's
// activation rows (rows past nrows are zero-filled). S groups of loads are in flight per warp
// without holding them in registers.
__device__ __forceinline__ void cp16(void* smem, const void* gmem, bool valid) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N_>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N_)); }

template <int NH>
struct Stage5 {
  static constexpr int W_BYTES = 512, S_BYTES = 64, A_BYTES = 8 * NH * 32;
  static constexpr int BYTES = W_BYTES + S_BYTES + A_BYTES;
};

template <int WPC, int S, int NH, int BM>
__global__ void __launch_bounds__(WPC * 32) down5_kernel(const Params p) {
  extern __shared__ __align__(16) uint8_t smem5[];
  using ST = Stage5<NH>;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g8 = lane >> 2, t = lane & 3;
  const int mb = blockIdx.y;
  const int T = blockIdx.x * WPC + warp;  // 64-column tile, 0..41
  if (mb * BM >= *p.num_tokens_padded) return;
  const uint8_t* bi = p.ws + (long long)mb * BI_BYTES;
  const BlockHdr* hdr = reinterpret_cast<const BlockHdr*>(bi);
  const int nrows = hdr->nrows;
  if (nrows == 0 || p.mode == SETUPONLY) return;
  const int e = hdr->expert;
  const uint16_t* gmask = reinterpret_cast<const uint16_t*>(bi + BI_MASK);
  const uint16_t* gact = reinterpret_cast<const uint16_t*>(bi + BI_ACTG);
  const bool math = p.mode == DENSE || p.mode == SKIP;
  bool dense = p.mode == DENSE || p.mode == LOADONLY_DENSE;
  uint8_t* ring = smem5 + warp * S * ST::BYTES;
  const uint8_t* wtile = p.W + e * p.w_expert + T * 32;
  const uint8_t* stile = p.S + e * p.s_expert + T * 64;
  // copy roles: weights lane -> (row l/2, half l%2); scales lanes 0..3; activations 2*NH chunks... per lane
  const int crow = lane >> 1, chalf = lane & 1;
  const __nv_bfloat16* arow[(16 * NH + 31) / 32];
  bool aval[(16 * NH + 31) / 32];
#pragma unroll
  for (int i = 0; i < (16 * NH + 31) / 32; ++i) {
    const int c = lane + 32 * i, ts = c >> 1;
    aval[i] = c < 16 * NH && ts < nrows;
    arow[i] = aval[i] ? p.A + (long long)hdr->rows[ts] * p.lda + 8 * (c & 1) : p.A;
  }
  const int kr[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};

  float acc[NH][4][4];
  for (int pass = 0; pass < 2; ++pass) {
    const int nsteps = dense ? G : hdr->nag;
    auto issue = [&](int j) {
      if (j < nsteps) {
        uint8_t* st = ring + (j % S) * ST::BYTES;
        const int grp = dense ? j : gact[j];
        const uint32_t m = dense ? 0xFFFFu : gmask[grp];
        const int kb = 16 * grp;
        if ((m >> crow) & 1u) cp16(st + crow * 32 + chalf * 16, wtile + (long long)(kb + crow) * p.w_k + chalf * 16, true);
        if (lane < 4) cp16(st + ST::W_BYTES + lane * 16, stile + (long long)grp * p.s_g + lane * 16, true);
#pragma unroll
        for (int i = 0; i < (16 * NH + 31) / 32; ++i) {
          const int c = lane + 32 * i;
          if (c < 16 * NH) cp16(st + ST::W_BYTES + ST::S_BYTES + c * 16, arow[i] + kb, aval[i]);
        }
      }
      cp_commit();
    };
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int h = 0; h < NH; ++h) acc[h][c][i] = 0.0f;
    __syncwarp();
#pragma unroll
    for (int i = 0; i < S - 1; ++i) issue(i);
    bool neg0 = false;
    for (int j = 0; j < nsteps; ++j) {
      cp_wait<S - 2>();
      __syncwarp();
      const uint8_t* st = ring + (j % S) * ST::BYTES;
      const int grp = dense ? j : gact[j];
      const uint32_t m = dense ? 0xFFFFu : gmask[grp];
      uint32_t w[4];
#pragma unroll
      for (int h = 0; h < 4; ++h)
        w[h] = ((m >> kr[h]) & 1u) ? *reinterpret_cast<const uint32_t*>(st + kr[h] * 32 + g8 * 4) : 0u;
      const uint2 sc = *reinterpret_cast<const uint2*>(st + ST::W_BYTES + g8 * 8);
      uint32_t bx[NH][2];
#pragma unroll
      for (int h = 0; h < NH; ++h) {
        const uint8_t* ar = st + ST::W_BYTES + ST::S_BYTES + (g8 + 8 * h) * 32;
        bx[h][0] = *reinterpret_cast<const uint32_t*>(ar + 4 * t);
        bx[h][1] = *reinterpret_cast<const uint32_t*>(ar + 16 + 4 * t);
      }
      __syncwarp();
      issue(j + S - 1);
      if (math) {
        const uint32_t w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
        const uint32_t m01 = (m >> (2 * t)) & 3u;
        const uint32_t m23 = (m >> (2 * t + 8)) & 3u;
        const uint32_t keep01 = ((m01 & 1u) ? 0x0000FFFFu : 0u) | ((m01 & 2u) ? 0xFFFF0000u : 0u);
        const uint32_t keep23 = ((m23 & 1u) ? 0x0000FFFFu : 0u) | ((m23 & 2u) ? 0xFFFF0000u : 0u);
        const uint32_t s01 = hmul2_u32(fp8_to_bf16x2(((sc.x << 8) & 0x0000FF00u) | ((sc.x << 16) & 0xFF000000u)), kExp120);
        const uint32_t s23 = hmul2_u32(fp8_to_bf16x2(((sc.x >> 8) & 0x0000FF00u) | (sc.x & 0xFF000000u)), kExp120);
        const uint32_t s45 = hmul2_u32(fp8_to_bf16x2(((sc.y << 8) & 0x0000FF00u) | ((sc.y << 16) & 0xFF000000u)), kExp120);
        const uint32_t s67 = hmul2_u32(fp8_to_bf16x2(((sc.y >> 8) & 0x0000FF00u) | (sc.y & 0xFF000000u)), kExp120);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const uint32_t sp = c == 0 ? s01 : (c == 1 ? s23 : (c == 2 ? s45 : s67));
          const uint32_t slo = __byte_perm(sp, 0, 0x1010);
          const uint32_t shi = __byte_perm(sp, 0, 0x3232);
          const uint32_t sel = c | (c << 4) | ((c + 4) << 8) | ((c + 4) << 12);
          const uint32_t x01 = __byte_perm(w0, w1, sel);
          const uint32_t x23 = __byte_perm(w2, w3, sel);
          const uint32_t a0 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x01 << 12), kExp124), slo) & keep01;
          const uint32_t a1 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x01 << 8), kExp124), shi) & keep01;
          const uint32_t a2 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x23 << 12), kExp124), slo) & keep23;
          const uint32_t a3 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x23 << 8), kExp124), shi) & keep23;
#pragma unroll
          for (int h = 0; h < NH; ++h)
            if (h == 0 || nrows > 8 * h) mma_bf16(acc[h][c], a0, a1, a2, a3, bx[h][0], bx[h][1]);
        }
        if (!dense) {
#pragma unroll
          for (int c = 0; c < 4; ++c)
#pragma unroll
            for (int q = 0; q < 4; ++q)
#pragma unroll
              for (int h = 0; h < NH; ++h)
                if (h == 0 || nrows > 8 * h) neg0 |= __float_as_uint(acc[h][c][q]) == 0x80000000u;
        }
      }
    }
    cp_wait<0>();
    __syncwarp();
    if (!__any_sync(0xFFFFFFFFu, neg0) || dense || !math) break;
    if (lane == 0 && p.stats) atomicAdd(p.stats + 0, 1);
    dense = true;
  }
  if (!math) return;
  const float gs4 = mul_ftz(p.GS[e], 4.0f);
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int col = T * 64 + 16 * c + g8;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int cc = col + ((q >> 1) ? 8 : 0);
#pragma unroll
      for (int h = 0; h < NH; ++h) {
        const int tok = 8 * h + 2 * t + (q & 1);
        if (tok < nrows) {
          const uint32_t v = to_bf16x2(mul_ftz(acc[h][c][q], gs4), 0.0f);
          reinterpret_cast<uint16_t*>(p.C)[(long long)hdr->rows[tok] * p.ldc + cc] = static_cast<uint16_t>(v);
        }
      }
    }
  }
}

// v7: down6 with the block's activation slice of each active group staged once per CTA in shared
// memory (cp.async, ring of R+1 groups, one barrier per group) and read by every warp of the CTA,
// instead of each 64-column warp re-reading it from L2. Same fragments and order as down4/down6.
template <int WPC, int R, int NH, int BM>
__global__ void __launch_bounds__(WPC * 32) down7_kernel(const Params p) {
  __shared__ __align__(16) uint8_t abuf[R + 1][8 * NH * 32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g8 = lane >> 2, t = lane & 3;
  const int mb = blockIdx.y;
  const int T = blockIdx.x * WPC + warp;  // 64-column tile, 0..41
  if (mb * BM >= *p.num_tokens_padded) return;
  const uint8_t* bi = p.ws + (long long)mb * BI_BYTES;
  const BlockHdr* hdr = reinterpret_cast<const BlockHdr*>(bi);
  const int nrows = hdr->nrows;
  if (nrows == 0 || p.mode == SETUPONLY) return;
  const int e = hdr->expert;
  const uint16_t* gmask = reinterpret_cast<const uint16_t*>(bi + BI_MASK);
  const uint16_t* gact = reinterpret_cast<const uint16_t*>(bi + BI_ACTG);
  const bool math = p.mode == DENSE || p.mode == SKIP;
  bool dense = p.mode == DENSE || p.mode == LOADONLY_DENSE;
  // activation copy role: chunk c = threadIdx.x (16 bytes: token slot c / 2, k half c % 2)
  const int ac = threadIdx.x;
  const bool a_on = ac < 16 * NH;
  const bool a_val = a_on && (ac >> 1) < nrows;
  const __nv_bfloat16* a_src = a_val ? p.A + (long long)hdr->rows[ac >> 1] * p.lda + 8 * (ac & 1) : p.A;
  const uint8_t* wbase = p.W + e * p.w_expert + T * 32 + g8 * 4;
  const uint8_t* sbase = p.S + e * p.s_expert + T * 64 + g8 * 8;
  // the k-slot pair of this lane in a group: rows 2t, 2t+1 (low k) and 2t+8, 2t+9 (high k)
  const int kr[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};

  float acc[NH][4][4];
  for (int pass = 0; pass < 2; ++pass) {
    const int nsteps = dense ? G : hdr->nag;
    uint32_t wq[R][4], mk[R];
    uint2 sq[R];
    auto load = [&](int slot, int j) {
      if (j < nsteps) {
        const int grp = dense ? j : gact[j];
        const uint32_t m = dense ? 0xFFFFu : gmask[grp];
        const int kb = 16 * grp;
#pragma unroll
        for (int h = 0; h < 4; ++h) {
          wq[slot][h] = ((m >> kr[h]) & 1u) ? ldg_u32(wbase + (long long)(kb + kr[h]) * p.w_k) : 0u;
        }
        sq[slot] = ldg_u64(sbase + (long long)grp * p.s_g);
        mk[slot] = m;
      } else {
        mk[slot] = 0u;
      }
    };
#pragma unroll
    for (int c = 0; c < 4; ++c) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int h = 0; h < NH; ++h) acc[h][c][i] = 0.0f;
      }
    }
    auto issue_a = [&](int j) {
      if (j < nsteps && a_on) {
        const int grp = dense ? j : gact[j];
        cp16(&abuf[j % (R + 1)][ac * 16], a_src + 16 * grp, a_val);
      }
      cp_commit();
    };
    __syncthreads();
#pragma unroll
    for (int i = 0; i < R; ++i) { load(i, i); issue_a(i); }
    bool neg0 = false;
    for (int j0 = 0; j0 < nsteps; j0 += R) {
#pragma unroll
      for (int i = 0; i < R; ++i) {
        if (j0 + i < nsteps) {
          const uint32_t m = mk[i];
          const uint32_t w0 = wq[i][0], w1 = wq[i][1], w2 = wq[i][2], w3 = wq[i][3];
          const uint2 sc = sq[i];
          cp_wait<R - 1>();
          __syncthreads();
          uint32_t bx[NH][2];
          const uint8_t* ab = abuf[(j0 + i) % (R + 1)];
#pragma unroll
          for (int h = 0; h < NH; ++h) {
            bx[h][0] = *reinterpret_cast<const uint32_t*>(ab + (g8 + 8 * h) * 32 + 4 * t);
            bx[h][1] = *reinterpret_cast<const uint32_t*>(ab + (g8 + 8 * h) * 32 + 16 + 4 * t);
          }
          issue_a(j0 + i + R);
          load(i, j0 + i + R);
          if (math) {
            const uint32_t m01 = (m >> (2 * t)) & 3u;
            const uint32_t m23 = (m >> (2 * t + 8)) & 3u;
            const uint32_t keep01 = ((m01 & 1u) ? 0x0000FFFFu : 0u) | ((m01 & 2u) ? 0xFFFF0000u : 0u);
            const uint32_t keep23 = ((m23 & 1u) ? 0x0000FFFFu : 0u) | ((m23 & 2u) ? 0xFFFF0000u : 0u);
            // scales: byte 2c + h of sc -> column 16c + g8 + 8h; humming's fp8 bit placement, x 2^120
            const uint32_t s01 = hmul2_u32(fp8_to_bf16x2(((sc.x << 8) & 0x0000FF00u) | ((sc.x << 16) & 0xFF000000u)), kExp120);
            const uint32_t s23 = hmul2_u32(fp8_to_bf16x2(((sc.x >> 8) & 0x0000FF00u) | (sc.x & 0xFF000000u)), kExp120);
            const uint32_t s45 = hmul2_u32(fp8_to_bf16x2(((sc.y << 8) & 0x0000FF00u) | ((sc.y << 16) & 0xFF000000u)), kExp120);
            const uint32_t s67 = hmul2_u32(fp8_to_bf16x2(((sc.y >> 8) & 0x0000FF00u) | (sc.y & 0xFF000000u)), kExp120);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const uint32_t sp = c == 0 ? s01 : (c == 1 ? s23 : (c == 2 ? s45 : s67));
              const uint32_t slo = __byte_perm(sp, 0, 0x1010);  // column g8 (h = 0)
              const uint32_t shi = __byte_perm(sp, 0, 0x3232);  // column g8 + 8 (h = 1)
              const uint32_t sel = c | (c << 4) | ((c + 4) << 8) | ((c + 4) << 12);
              const uint32_t x01 = __byte_perm(w0, w1, sel);  // bytes: w0[c], w0[c], w1[c], w1[c]
              const uint32_t x23 = __byte_perm(w2, w3, sel);
              const uint32_t a0 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x01 << 12), kExp124), slo) & keep01;
              const uint32_t a1 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x01 << 8), kExp124), shi) & keep01;
              const uint32_t a2 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x23 << 12), kExp124), slo) & keep23;
              const uint32_t a3 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x23 << 8), kExp124), shi) & keep23;
#pragma unroll
              for (int h = 0; h < NH; ++h)
                if (h == 0 || nrows > 8 * h) mma_bf16(acc[h][c], a0, a1, a2, a3, bx[h][0], bx[h][1]);
            }
          }
        }
      }
    }
    if (math && !dense) {
#pragma unroll
      for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
          for (int h = 0; h < NH; ++h)
            if (8 * h + 2 * t + (q & 1) < nrows) neg0 |= (__float_as_uint(acc[h][c][q]) & 0x7FFFFFFFu) == 0u;
    }
    cp_wait<0>();
    if (!__syncthreads_or(neg0 ? 1 : 0) || dense || !math) break;
    // a -0 accumulator appeared: recompute this warp's tile with every row loaded
    if (lane == 0 && p.stats) atomicAdd(p.stats + 0, 1);
    dense = true;
  }
  if (!math) return;

  const float gs4 = mul_ftz(p.GS[e], 4.0f);
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int col = T * 64 + 16 * c + g8;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int cc = col + ((q >> 1) ? 8 : 0);
#pragma unroll
      for (int h = 0; h < NH; ++h) {
        const int tok = 8 * h + 2 * t + (q & 1);
        if (tok < nrows) {
          const uint32_t v = to_bf16x2(mul_ftz(acc[h][c][q], gs4), 0.0f);
          reinterpret_cast<uint16_t*>(p.C)[(long long)hdr->rows[tok] * p.ldc + cc] = static_cast<uint16_t>(v);
        }
      }
    }
  }
}



// v8: down7 on the "swapc" layout: per expert, 64-column tile and k16 group, the 16 k rows (32 bytes
// each, the same bytes as "swap") are one contiguous 512-byte block and the 64 scale bytes another.
// Each lane loads 16 bytes (row lane/2) with one vector load when that row is active and the warp
// redistributes them through shared memory into down4's per-lane words. Same fragments and order.
__device__ __forceinline__ uint4 ldg_v4(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
template <int WPC, int R, int NH, int BM>
__global__ void __launch_bounds__(WPC * 32) down8_kernel(const Params p) {
  __shared__ __align__(16) uint8_t abuf[R + 1][8 * NH * 32];
  __shared__ __align__(16) uint8_t tbuf[WPC][576];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g8 = lane >> 2, t = lane & 3;
  const int mb = blockIdx.y;
  const int T = blockIdx.x * WPC + warp;  // 64-column tile, 0..41
  if (mb * BM >= *p.num_tokens_padded) return;
  const uint8_t* bi = p.ws + (long long)mb * BI_BYTES;
  const BlockHdr* hdr = reinterpret_cast<const BlockHdr*>(bi);
  const int nrows = hdr->nrows;
  if (nrows == 0 || p.mode == SETUPONLY) return;
  const int e = hdr->expert;
  const uint16_t* gmask = reinterpret_cast<const uint16_t*>(bi + BI_MASK);
  const uint16_t* gact = reinterpret_cast<const uint16_t*>(bi + BI_ACTG);
  const bool math = p.mode == DENSE || p.mode == SKIP;
  bool dense = p.mode == DENSE || p.mode == LOADONLY_DENSE;
  // activation copy role: chunk c = threadIdx.x (16 bytes: token slot c / 2, k half c % 2)
  const int ac = threadIdx.x;
  const bool a_on = ac < 16 * NH;
  const bool a_val = a_on && (ac >> 1) < nrows;
  const __nv_bfloat16* a_src = a_val ? p.A + (long long)hdr->rows[ac >> 1] * p.lda + 8 * (ac & 1) : p.A;
  const uint8_t* wbase = p.W + e * p.w_expert + T * p.w_tile + lane * 16;
  const uint8_t* sbase = p.S + e * p.s_expert + T * p.s_tile + (lane & 3) * 16;
  uint8_t* tb = tbuf[warp];
  // the k-slot pair of this lane in a group: rows 2t, 2t+1 (low k) and 2t+8, 2t+9 (high k)
  const int kr[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};

  float acc[NH][4][4];
  for (int pass = 0; pass < 2; ++pass) {
    const int nsteps = dense ? G : hdr->nag;
    uint4 wv[R], sv[R];
    uint32_t mk[R];
    auto load = [&](int slot, int j) {
      if (j < nsteps) {
        const int grp = dense ? j : gact[j];
        const uint32_t m = dense ? 0xFFFFu : gmask[grp];
        const int kb = 16 * grp;
        wv[slot] = ((m >> (lane >> 1)) & 1u) ? ldg_v4(wbase + (long long)grp * 512) : make_uint4(0u, 0u, 0u, 0u);
        if (lane < 4) sv[slot] = ldg_v4(sbase + (long long)grp * p.s_g);
        mk[slot] = m;
      } else {
        mk[slot] = 0u;
      }
    };
#pragma unroll
    for (int c = 0; c < 4; ++c) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int h = 0; h < NH; ++h) acc[h][c][i] = 0.0f;
      }
    }
    auto issue_a = [&](int j) {
      if (j < nsteps && a_on) {
        const int grp = dense ? j : gact[j];
        cp16(&abuf[j % (R + 1)][ac * 16], a_src + 16 * grp, a_val);
      }
      cp_commit();
    };
    __syncthreads();
#pragma unroll
    for (int i = 0; i < R; ++i) { load(i, i); issue_a(i); }
    bool neg0 = false;
    for (int j0 = 0; j0 < nsteps; j0 += R) {
#pragma unroll
      for (int i = 0; i < R; ++i) {
        if (j0 + i < nsteps) {
          const uint32_t m = mk[i];
          *reinterpret_cast<uint4*>(tb + lane * 16) = wv[i];
          if (lane < 4) *reinterpret_cast<uint4*>(tb + 512 + lane * 16) = sv[i];
          __syncwarp();
          const uint32_t w0 = ((m >> kr[0]) & 1u) ? *reinterpret_cast<const uint32_t*>(tb + kr[0] * 32 + g8 * 4) : 0u;
          const uint32_t w1 = ((m >> kr[1]) & 1u) ? *reinterpret_cast<const uint32_t*>(tb + kr[1] * 32 + g8 * 4) : 0u;
          const uint32_t w2 = ((m >> kr[2]) & 1u) ? *reinterpret_cast<const uint32_t*>(tb + kr[2] * 32 + g8 * 4) : 0u;
          const uint32_t w3 = ((m >> kr[3]) & 1u) ? *reinterpret_cast<const uint32_t*>(tb + kr[3] * 32 + g8 * 4) : 0u;
          const uint2 sc = *reinterpret_cast<const uint2*>(tb + 512 + g8 * 8);
          __syncwarp();
          cp_wait<R - 1>();
          __syncthreads();
          uint32_t bx[NH][2];
          const uint8_t* ab = abuf[(j0 + i) % (R + 1)];
#pragma unroll
          for (int h = 0; h < NH; ++h) {
            bx[h][0] = *reinterpret_cast<const uint32_t*>(ab + (g8 + 8 * h) * 32 + 4 * t);
            bx[h][1] = *reinterpret_cast<const uint32_t*>(ab + (g8 + 8 * h) * 32 + 16 + 4 * t);
          }
          issue_a(j0 + i + R);
          load(i, j0 + i + R);
          if (math) {
            const uint32_t m01 = (m >> (2 * t)) & 3u;
            const uint32_t m23 = (m >> (2 * t + 8)) & 3u;
            const uint32_t keep01 = ((m01 & 1u) ? 0x0000FFFFu : 0u) | ((m01 & 2u) ? 0xFFFF0000u : 0u);
            const uint32_t keep23 = ((m23 & 1u) ? 0x0000FFFFu : 0u) | ((m23 & 2u) ? 0xFFFF0000u : 0u);
            // scales: byte 2c + h of sc -> column 16c + g8 + 8h; humming's fp8 bit placement, x 2^120
            const uint32_t s01 = hmul2_u32(fp8_to_bf16x2(((sc.x << 8) & 0x0000FF00u) | ((sc.x << 16) & 0xFF000000u)), kExp120);
            const uint32_t s23 = hmul2_u32(fp8_to_bf16x2(((sc.x >> 8) & 0x0000FF00u) | (sc.x & 0xFF000000u)), kExp120);
            const uint32_t s45 = hmul2_u32(fp8_to_bf16x2(((sc.y << 8) & 0x0000FF00u) | ((sc.y << 16) & 0xFF000000u)), kExp120);
            const uint32_t s67 = hmul2_u32(fp8_to_bf16x2(((sc.y >> 8) & 0x0000FF00u) | (sc.y & 0xFF000000u)), kExp120);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const uint32_t sp = c == 0 ? s01 : (c == 1 ? s23 : (c == 2 ? s45 : s67));
              const uint32_t slo = __byte_perm(sp, 0, 0x1010);  // column g8 (h = 0)
              const uint32_t shi = __byte_perm(sp, 0, 0x3232);  // column g8 + 8 (h = 1)
              const uint32_t sel = c | (c << 4) | ((c + 4) << 8) | ((c + 4) << 12);
              const uint32_t x01 = __byte_perm(w0, w1, sel);  // bytes: w0[c], w0[c], w1[c], w1[c]
              const uint32_t x23 = __byte_perm(w2, w3, sel);
              const uint32_t a0 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x01 << 12), kExp124), slo) & keep01;
              const uint32_t a1 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x01 << 8), kExp124), shi) & keep01;
              const uint32_t a2 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x23 << 12), kExp124), slo) & keep23;
              const uint32_t a3 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x23 << 8), kExp124), shi) & keep23;
#pragma unroll
              for (int h = 0; h < NH; ++h)
                if (h == 0 || nrows > 8 * h) mma_bf16(acc[h][c], a0, a1, a2, a3, bx[h][0], bx[h][1]);
            }
          }
        }
      }
    }
    if (math && !dense) {
#pragma unroll
      for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
          for (int h = 0; h < NH; ++h)
            if (8 * h + 2 * t + (q & 1) < nrows) neg0 |= (__float_as_uint(acc[h][c][q]) & 0x7FFFFFFFu) == 0u;
    }
    cp_wait<0>();
    if (!__syncthreads_or(neg0 ? 1 : 0) || dense || !math) break;
    // a -0 accumulator appeared: recompute this warp's tile with every row loaded
    if (lane == 0 && p.stats) atomicAdd(p.stats + 0, 1);
    dense = true;
  }
  if (!math) return;

  const float gs4 = mul_ftz(p.GS[e], 4.0f);
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int col = T * 64 + 16 * c + g8;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int cc = col + ((q >> 1) ? 8 : 0);
#pragma unroll
      for (int h = 0; h < NH; ++h) {
        const int tok = 8 * h + 2 * t + (q & 1);
        if (tok < nrows) {
          const uint32_t v = to_bf16x2(mul_ftz(acc[h][c][q], gs4), 0.0f);
          reinterpret_cast<uint16_t*>(p.C)[(long long)hdr->rows[tok] * p.ldc + cc] = static_cast<uint16_t>(v);
        }
      }
    }
  }
}

}  // namespace zskip

using namespace zskip;

#define ZSKIP_HIDDEN __attribute__((visibility("hidden")))

// set through the C API (zskip.cu), read by the launchers (zskip_launch.cuh)
extern ZSKIP_HIDDEN int g_sms;
extern ZSKIP_HIDDEN int g_rds_ring2;  // old Python/new library remains stock until explicit init
extern ZSKIP_HIDDEN int g_pdl;  // zskip_set_pdl: launch the v4 pair with programmatic dependent launch

#ifndef ZSKIP_NO_PDL
template <typename Kern>
static cudaError_t launch_pdl(Kern kernel, dim3 grid, dim3 block, cudaStream_t stream, const Params& p,
                              int dyn_smem = 0) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = dyn_smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, kernel, p);
}
#endif

// Config lists. Each list (ZSKIP4_CONFIGS: each of its parts A..E, which it concatenates in order) is
// instantiated by one translation unit: zskip_v3.cu: ZSKIP_CONFIGS and ZSKIP4F_CONFIGS;
// zskip_v4a.cu .. zskip_v4e.cu: ZSKIP4A_CONFIGS .. ZSKIP4E_CONFIGS; zskip_v5.cu .. zskip_v8.cu:
// ZSKIP5_CONFIGS .. ZSKIP8_CONFIGS; zskip_pair.cu: configs 113, 114, 118, 119; zskip_lean.cu:
// ZSKIP_LEAN_CONFIGS. NVVM keeps a device function's __shared__ variable (build_lists' s_tot,
// fused_prep's s_rows and s_nrows) at module scope when two or more kernels of the unit use it and
// moves it into the kernel otherwise, and the kernel's shared memory layout follows from that. So a
// unit holding one kernel that uses such a variable also holds another one that uses it:
// build_lists<128>: the prep_kernel instances (zskip.cu), the 4-warp down3 configs; build_lists<64>:
// the 2-warp down3 configs and config 108; build_lists<192>: the 6-warp down3 configs;
// build_lists<32> and fused_prep<32, 16, 8>: the fused pair kernels. cicc's code for a kernel that
// uses an extern __shared__ array also depends on the array's other users in the unit, so each such
// array's users are compiled in one unit (smem: down3_kernel, smem4: config 108, smem5: down5_kernel,
// smem_pair: the fused pair kernels).

// config id, warps, RB, NR, QB, NQ
#define ZSKIP_CONFIGS(X)                                                                           \
  X(0, 4, 32, 4, 8, 6) X(1, 4, 32, 3, 8, 4) X(2, 2, 32, 4, 8, 6) X(3, 2, 32, 3, 8, 4)               \
  X(4, 6, 32, 4, 8, 6) X(5, 6, 32, 3, 8, 4) X(6, 12, 32, 3, 8, 4) X(7, 4, 16, 6, 8, 6)               \
  X(8, 7, 32, 3, 8, 4) X(9, 3, 32, 4, 8, 6)

// v4 config id, warps per CTA, ring depth (active groups in flight per warp), 8-row MMA halves per
// block (rows per block cap = 8 NH), moe_align block size. 100..107: decode (<= 8 tokens, block 16);
// 120..: prefill (block_m 16 / 32 / 64, every valid row of the block)
#define ZSKIP4A_CONFIGS(X) X(100, 2, 4, 1, 16) X(101, 2, 8, 1, 16) X(102, 3, 8, 1, 16) X(103, 6, 4, 1, 16) \
  X(104, 6, 8, 1, 16) X(105, 7, 8, 1, 16) X(106, 2, 12, 1, 16) X(107, 1, 8, 1, 16)
#define ZSKIP4B_CONFIGS(X) X(120, 2, 4, 2, 16) X(121, 2, 4, 4, 32) X(122, 2, 4, 8, 64) X(123, 1, 4, 8, 64)
#define ZSKIP4C_CONFIGS(X) X(124, 3, 4, 8, 64) X(125, 2, 2, 8, 64) X(126, 6, 4, 8, 64) X(127, 2, 8, 4, 32)
#define ZSKIP4D_CONFIGS(X) X(130, 1, 4, 2, 16) X(131, 1, 8, 2, 16) X(132, 2, 8, 2, 16) X(133, 3, 4, 2, 16) \
  X(134, 3, 8, 2, 16) X(135, 6, 4, 2, 16) X(136, 6, 8, 2, 16) X(137, 7, 4, 2, 16) X(138, 2, 2, 2, 16)      \
  X(139, 1, 2, 2, 16)
#define ZSKIP4E_CONFIGS(X) X(141, 1, 4, 4, 32) X(142, 1, 2, 4, 32) X(143, 1, 1, 4, 32) X(144, 2, 2, 4, 32) \
  X(145, 3, 2, 4, 32) X(146, 1, 1, 8, 64) X(147, 1, 2, 8, 64) X(148, 1, 1, 2, 16) X(149, 7, 2, 4, 32)      \
  X(150, 6, 2, 4, 32) X(151, 2, 1, 4, 32) X(152, 1, 3, 4, 32)
#define ZSKIP4_CONFIGS(X) \
  ZSKIP4A_CONFIGS(X) ZSKIP4B_CONFIGS(X) ZSKIP4C_CONFIGS(X) ZSKIP4D_CONFIGS(X) ZSKIP4E_CONFIGS(X)
#ifndef ZSKIP_NO_FUSED
#define ZSKIP4F_CONFIGS(X) X(108, 2, 4, 1, 16)
#endif

// v5 config id, warps per CTA, smem ring stages, 8-row halves, block size
#define ZSKIP5_CONFIGS(X) X(200, 2, 4, 4, 32) X(201, 2, 6, 4, 32) X(202, 2, 8, 4, 32) X(203, 1, 8, 4, 32) \
  X(204, 3, 6, 4, 32) X(205, 3, 8, 4, 32) X(206, 2, 12, 4, 32) X(207, 1, 12, 4, 32) X(208, 6, 4, 4, 32)   \
  X(209, 1, 6, 4, 32) X(210, 2, 6, 2, 16) X(211, 1, 8, 2, 16) X(212, 2, 6, 8, 64) X(213, 1, 6, 8, 64)
#define ZSKIP6_CONFIGS(X) X(300, 2, 2, 4, 32) X(301, 1, 2, 4, 32) X(302, 2, 4, 4, 32) X(303, 3, 2, 4, 32) \
  X(304, 1, 4, 4, 32) X(305, 2, 1, 4, 32) X(306, 6, 2, 4, 32) X(307, 2, 2, 2, 16) X(308, 1, 2, 8, 64) X(309, 2, 3, 4, 32)
#define ZSKIP7_CONFIGS(X) X(400, 2, 2, 4, 32) X(401, 3, 2, 4, 32) X(402, 6, 2, 4, 32) X(403, 7, 2, 4, 32) \
  X(404, 14, 2, 4, 32) X(405, 6, 4, 4, 32) X(406, 7, 4, 4, 32) X(407, 6, 3, 4, 32) X(408, 14, 3, 4, 32)   \
  X(409, 6, 2, 2, 16) X(410, 7, 2, 2, 16) X(411, 6, 2, 8, 64) X(412, 3, 4, 4, 32) X(413, 21, 2, 4, 32)
#define ZSKIP8_CONFIGS(X) X(500, 2, 2, 4, 32) X(501, 6, 2, 4, 32) X(502, 6, 4, 4, 32) X(503, 6, 3, 4, 32) \
  X(504, 3, 4, 4, 32) X(505, 7, 2, 4, 32) X(506, 2, 4, 4, 32) X(507, 6, 2, 2, 16) X(508, 3, 2, 4, 32) X(509, 6, 6, 4, 32)

#ifndef ZSKIP_NO_PAIR
// lean paired decode (zskip_lean.cuh): config id, warps per CTA, ring depth R, flags (F_ZHALF 1,
// F_NOSPLIT 2, F_BPF 4, F_RPF 8, F_WPF 16, F_S6 32, prefetch distance code << 8); 653 is the default;
// 663 is 653 reading the 6-bit scale plane (to_s6)
#define ZSKIP_LEAN_CONFIGS(X) X(604, 1, 2, 260) X(613, 1, 2, 2820) X(616, 1, 2, 2824) X(630, 1, 2, 0) \
  X(640, 1, 3, 272) X(645, 1, 4, 272) X(653, 1, 3, 274) X(656, 1, 4, 274) X(658, 1, 2, 274) \
  X(663, 1, 3, 306)
#endif

// Per-config entry points, defined in zskip_launch.cuh and instantiated (with the kernels they use)
// by the unit of the config's list; the C API in zskip.cu calls them: launch* for zskip_down, info*
// for zskip_config, load* for zskip_init.
template <int W, int RB, int NR, int QB, int NQ>
ZSKIP_HIDDEN int launch3(const Params& p, cudaStream_t stream, int ctas_per_sm);
template <int W, int RB, int NR, int QB, int NQ>
ZSKIP_HIDDEN int info3(int* info);
template <int WPC, int R, int NH, int BM>
ZSKIP_HIDDEN int launch4(const Params& p, cudaStream_t stream);
template <int WPC, int R, int NH, int BM>
ZSKIP_HIDDEN int info4(int* info);
template <int WPC, int R, int NH, int BM>
ZSKIP_HIDDEN cudaError_t load4();
#ifndef ZSKIP_NO_FUSED
template <int WPC, int R, int NH, int BM>
ZSKIP_HIDDEN int launch4f(const Params& p, cudaStream_t stream);
template <int WPC, int R, int NH, int BM>
ZSKIP_HIDDEN int info4f(int* info);
#endif
template <int WPC, int S, int NH, int BM>
ZSKIP_HIDDEN int launch5(const Params& p, cudaStream_t stream);
template <int WPC, int R, int NH, int BM>
ZSKIP_HIDDEN int launch6(const Params& p, cudaStream_t stream);
template <int WPC, int R, int NH, int BM>
ZSKIP_HIDDEN int launch7(const Params& p, cudaStream_t stream);
template <int WPC, int R, int NH, int BM>
ZSKIP_HIDDEN int launch8(const Params& p, cudaStream_t stream);
#ifndef ZSKIP_NO_PAIR
template <bool Fused, bool HalfSkip = false>
ZSKIP_HIDDEN int launch_pair(const Params& p, cudaStream_t stream);
template <bool Fused, bool HalfSkip>
ZSKIP_HIDDEN int info_pair(int* info);
template <bool Fused, bool HalfSkip>
ZSKIP_HIDDEN cudaError_t load_pair();
template <int WPC, int R, int F>
ZSKIP_HIDDEN int launch_lean(const Params& p, cudaStream_t stream);
template <int WPC, int R, int F>
ZSKIP_HIDDEN int info_lean(int* info);
#endif
// prep_kernel<BM, CAP> launches for the other units; the prep_kernel instances are compiled in zskip.cu
template <int BM, int CAP>
ZSKIP_HIDDEN void launch_prep(const Params& p, cudaStream_t stream);
#ifndef ZSKIP_NO_PDL
template <int BM, int CAP>
ZSKIP_HIDDEN cudaError_t launch_prep_pdl(const Params& p, cudaStream_t stream);
#endif

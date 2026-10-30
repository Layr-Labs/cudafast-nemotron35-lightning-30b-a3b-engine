// pzskip: humming-exact routed-expert DOWN projection for PREFILL-sized calls
// (expert blocks of up to 64 rows, humming's block_m for the large-M bucket),
// that does not load the w2 rows whose relu^2 activations are zero in every row
// of the block.
//
// Arithmetic: zskip's down4_kernel (harness/kernels/zskip), which is humming
// 0.1.10's batch-invariant indexed GEMM element for element: per output element
//   acc = +0; for each k16 group g ascending: acc = mma.m16n8k16.bf16(...);
//   out = cvt.rn.bf16(mul.ftz(acc, mul.ftz(gs[e], 4)))
// with the weights dequantised by humming's ops (e2m1 * e4m3 / 4 exactly) and
// used as the MMA A operand, the activations as B (token slots as columns).
// Only the block size changes: a warp covers 64 output columns of one expert
// block for up to 32 of its rows (4 token tiles of 8), so each loaded weight
// fragment feeds up to 4 MMAs; a second warp covers rows 32..63.
// Skip semantics as zskip: masks are the union over the block's valid rows; a
// row with mask bit 0 is not loaded (its weight halves are +0); a group with
// mask 0 is skipped entirely. A -0 accumulator anywhere in a warp makes the
// warp recompute its tile with every row loaded.
//
// Weight layout: zskip's "swap" repack (layer.zskip_w2), read as down4 reads it.
//
// Sources: this header holds the kernels and their launchers; each pzskip_<part>.cu dispatches its
// configs (compiling their kernels) and pzskip.cu holds the C API and the prep_kernel instances.
// Each .cu is compiled on its own and all are linked into libpzskip.so (build.sh). The kernels that
// use the same extern __shared__ array are compiled in one unit (smem_raw: pzskip_smem.cu).

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdlib>
#include <cstdio>
#include <climits>

namespace pzskip {

constexpr int K = 1856;
constexpr int N = 2688;
constexpr int G = K / 16;  // 116
constexpr int BM = 64;     // rows per expert block (humming's block_m for prefill)
constexpr int HALF = 32;   // rows per warp
constexpr int TT = HALF / 8;
constexpr int NT64 = N / 64;  // 42 column tiles

// per-block info written by prep_kernel: [0..63] rows (int32, -1 invalid), then
// nrows, expert, nag, then u16 mask[128], u16 act_g[128]
constexpr int BI_ROWS = 0;
constexpr int BI_NROWS = BM * 4;
constexpr int BI_EXPERT = BI_NROWS + 4;
constexpr int BI_NAG = BI_EXPERT + 4;
constexpr int BI_MASK = 512;
constexpr int BI_ACTG = BI_MASK + 256;
constexpr int BI_BYTES = BI_ACTG + 256;  // 1024

struct Params {
  const __nv_bfloat16* A;  // [shape_m, lda] relu^2 activations
  long long lda;
  int shape_m;
  const int* sorted_ids;   // moe_align (block 64)
  const int* expert_ids;
  const int* num_tokens_padded;
  const uint8_t* W;        // zskip "swap" nibbles
  const uint8_t* S;        // zskip "swap" scales
  const float* GS;         // [E]
  __nv_bfloat16* C;        // [shape_m, ldc]
  long long ldc;
  long long w_expert, w_k, s_expert, s_g;
  uint8_t* ws;             // [max_blocks * BI_BYTES]
  int max_blocks;
  int dense;               // 1: load every row (humming's computation, for checks)
  int* stats;              // optional: [0] dense redo warps
  int half64;              // smem5 prep: 32-row halves of a 64-row moe_align layout (config 369)
};

__device__ __forceinline__ uint32_t hmul2_u32(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t fp4_to_bf16x2(uint32_t v) {
  return (v & 0x80008000u) | ((v & 0x70007000u) >> 6);
}
__device__ __forceinline__ uint32_t fp8_to_bf16x2(uint32_t v) {
  return (v & 0x80008000u) | ((v & 0x7F007F00u) >> 4);
}
constexpr uint32_t kExp124 = (124u * 0x00010001u + 0x007F007Fu) << 7;
constexpr uint32_t kExp120 = (120u * 0x00010001u + 0x007F007Fu) << 7;

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
__device__ __forceinline__ uint32_t ldg_u32_l1(const void* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}

// one CTA per expert block: rows, masks (union over valid rows) and the active group list
template <int BMT = BM>
__global__ void __launch_bounds__(128) prep_kernel(const Params p) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int mb = blockIdx.x;
  uint8_t* bi = p.ws + (long long)mb * BI_BYTES;
  int* rows = reinterpret_cast<int*>(bi + BI_ROWS);
  uint16_t* mask = reinterpret_cast<uint16_t*>(bi + BI_MASK);
  uint16_t* actg = reinterpret_cast<uint16_t*>(bi + BI_ACTG);
  __shared__ int s_rows[BMT];
  __shared__ int s_n;
  __shared__ uint16_t s_mask[128];
  const bool live = mb * BMT < *p.num_tokens_padded;
  if (tid < BMT) {
    const unsigned id = live ? static_cast<unsigned>(p.sorted_ids[mb * BMT + tid]) : 0xFFFFFFFFu;
    const bool valid = id < static_cast<unsigned>(p.shape_m);
    rows[tid] = valid ? static_cast<int>(id) : -1;
    s_rows[tid] = valid ? static_cast<int>(id) : -1;
  }
  __syncthreads();
  if (tid == 0) {
    int n = 0;
    for (int r = 0; r < BMT; ++r) n += s_rows[r] >= 0 ? 1 : 0;
    s_n = n;
    *reinterpret_cast<int*>(bi + BI_NROWS) = n;
    *reinterpret_cast<int*>(bi + BI_EXPERT) = live ? p.expert_ids[mb] : -1;
  }
  __syncthreads();
  for (int g = tid; g < 128; g += 128) {
    uint32_t m = 0;
    if (g < G && s_n > 0) {
      if (p.dense == 1) {
        m = 0xFFFFu;
      } else {
        for (int r = 0; r < BMT; ++r) {
          const int row = s_rows[r];
          if (row < 0) continue;
          const uint4* q = reinterpret_cast<const uint4*>(p.A + (long long)row * p.lda + 16 * g);
          m |= nz_mask8(__ldg(q)) | (nz_mask8(__ldg(q + 1)) << 8);
          if (m == 0xFFFFu) break;
        }
      }
    }
    s_mask[g] = static_cast<uint16_t>(m);
    mask[g] = static_cast<uint16_t>(m);
  }
  __syncthreads();
  if (warp == 0) {  // ordered list of active groups
    int base = 0;
    for (int c = 0; c < 4; ++c) {
      const int g = 32 * c + lane;
      const bool a = g < G && s_mask[g] != 0;
      const unsigned ball = __ballot_sync(0xFFFFFFFFu, a);
      if (a) actg[base + __popc(ball & ((1u << lane) - 1u))] = static_cast<uint16_t>(g);
      base += __popc(ball);
    }
    if (lane == 0) *reinterpret_cast<int*>(bi + BI_NAG) = base;
  }
}

// grid (NT64 / WPC, max_blocks, 2): warp = (block, 64-column tile, 32-row half)
template <int WPC, int R, int HALF_ = HALF>
__global__ void __launch_bounds__(WPC * 32) down_kernel(const Params p) {
  constexpr int HALF = HALF_;
  constexpr int TT = HALF / 8;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g8 = lane >> 2, t = lane & 3;
  const int mb = blockIdx.y;
  const int half = blockIdx.z;
  const int T = blockIdx.x * WPC + warp;
  if (T >= NT64 || mb * BM >= *p.num_tokens_padded) return;
  const uint8_t* bi = p.ws + (long long)mb * BI_BYTES;
  const int nrows = *reinterpret_cast<const int*>(bi + BI_NROWS);
  const int e = *reinterpret_cast<const int*>(bi + BI_EXPERT);
  if (e < 0 || nrows <= half * HALF) return;
  const int* rows = reinterpret_cast<const int*>(bi + BI_ROWS) + half * HALF;
  const uint16_t* gmask = reinterpret_cast<const uint16_t*>(bi + BI_MASK);
  const uint16_t* gact = reinterpret_cast<const uint16_t*>(bi + BI_ACTG);
  const int nag = *reinterpret_cast<const int*>(bi + BI_NAG);
  // rows are packed first in the sorted block (moe_align fills valid ids before padding)
  int myrows = nrows - half * HALF;
  if (myrows > HALF) myrows = HALF;
  const int ntt = (myrows + 7) >> 3;
  const __nv_bfloat16* xrow[TT];
#pragma unroll
  for (int tt = 0; tt < TT; ++tt) {
    const int r = 8 * tt + g8;
    xrow[tt] = (r < myrows && rows[r] >= 0) ? p.A + (long long)rows[r] * p.lda : nullptr;
  }
  const uint8_t* wbase = p.W + e * p.w_expert + T * 32 + g8 * 4;
  const uint8_t* sbase = p.S + e * p.s_expert + T * 64 + g8 * 8;
  const int kr[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};

  float acc[TT][4][4];
  bool dense = p.dense != 0;
  for (int pass = 0; pass < 2; ++pass) {
    const int nsteps = dense ? G : nag;
    uint32_t wq[R][4], mk[R], xq[R][TT][2];
    uint2 sq[R];
    auto load = [&](int slot, int j) {
      if (j < nsteps) {
        const int grp = dense ? j : gact[j];
        const uint32_t m = dense ? 0xFFFFu : gmask[grp];
        const int kb = 16 * grp;
#pragma unroll
        for (int h = 0; h < 4; ++h)
          wq[slot][h] = ((m >> kr[h]) & 1u) ? ldg_u32(wbase + (long long)(kb + kr[h]) * p.w_k) : 0u;
        sq[slot] = ldg_u64(sbase + (long long)grp * p.s_g);
#pragma unroll
        for (int tt = 0; tt < TT; ++tt) {
          xq[slot][tt][0] = (tt < ntt && xrow[tt]) ? ldg_u32_l1(xrow[tt] + kb + 2 * t) : 0u;
          xq[slot][tt][1] = (tt < ntt && xrow[tt]) ? ldg_u32_l1(xrow[tt] + kb + 8 + 2 * t) : 0u;
        }
        mk[slot] = m;
      } else {
        mk[slot] = 0u;
      }
    };
#pragma unroll
    for (int tt = 0; tt < TT; ++tt)
#pragma unroll
      for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[tt][c][i] = 0.0f;
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
          uint32_t xb[TT][2];
#pragma unroll
          for (int tt = 0; tt < TT; ++tt) {
            xb[tt][0] = xq[i][tt][0];
            xb[tt][1] = xq[i][tt][1];
          }
          load(i, j0 + i + R);
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
            for (int tt = 0; tt < TT; ++tt)
              if (tt < ntt) mma_bf16(acc[tt][c], a0, a1, a2, a3, xb[tt][0], xb[tt][1]);
          }
          if (!dense) {
#pragma unroll
            for (int tt = 0; tt < TT; ++tt)
#pragma unroll
              for (int c = 0; c < 4; ++c)
#pragma unroll
                for (int q = 0; q < 4; ++q) neg0 |= __float_as_uint(acc[tt][c][q]) == 0x80000000u;
          }
        }
      }
    }
    if (!__any_sync(0xFFFFFFFFu, neg0) || dense) break;
    if (lane == 0 && p.stats) atomicAdd(p.stats + 0, 1);
    dense = true;
  }

  const float gs4 = mul_ftz(p.GS[e], 4.0f);
#pragma unroll
  for (int tt = 0; tt < TT; ++tt) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int col = T * 64 + 16 * c + g8;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int r = 8 * tt + 2 * t + (q & 1);
        const int cc = col + ((q >> 1) ? 8 : 0);
        if (r < myrows && rows[r] >= 0) {
          const uint32_t v = to_bf16x2(mul_ftz(acc[tt][c][q], gs4), 0.0f);
          reinterpret_cast<uint16_t*>(p.C)[(long long)rows[r] * p.ldc + cc] = static_cast<uint16_t>(v);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// smem-staged variant: CTA = (expert block, 128 output columns = 64-column tiles 2P, 2P+1), 4
// warps; warp w owns tile 2P + (w >> 1), 16-column groups c in {2 (w & 1), 2 (w & 1) + 1}, and
// every row of the block (8 token tiles). Per stage the CTA copies with cp.async (16-byte pieces)
// only the active k rows of the 128 columns' weights, their scales and the block's activations
// for KG active k16 groups; skipped rows stay stale in shared memory and are masked to +0 exactly
// as in down_kernel. Arithmetic per output element is down_kernel's.
__device__ __forceinline__ void cp16(void* smem, const void* gmem) {
  const unsigned sa = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(gmem) : "memory");
}

// Same cp.async copy and bytes; only immutable B/BS L2 priority changes.
template<int POLICY>
__device__ __forceinline__ void cp16_weight(void* smem, const void* gmem) {
  if constexpr (POLICY == 0) cp16(smem, gmem);
  else {
    const unsigned sa = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;"
                 :: "r"(sa), "l"(gmem), "l"(policy) : "memory");
  }
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N_>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N_) : "memory"); }

constexpr int XS = 48;  // activation row stride in shared memory (bytes): conflict-free reads

template <int KG, int NS, int NTL, int BMT = BM>
struct SmemStage {
  uint8_t w[KG][16][32 * NTL];
  uint8_t s[KG][64 * NTL];
  uint8_t x[KG][BMT][XS];
};

template <int KG, int NS, int NTL, int MINB = 1>
__global__ void __launch_bounds__(64 * NTL, MINB) smem_kernel(const Params p) {
  constexpr int NTH = 64 * NTL;
  extern __shared__ __align__(16) uint8_t smem_raw[];
  using Stage = SmemStage<KG, NS, NTL>;
  Stage* stg = reinterpret_cast<Stage*>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g8 = lane >> 2, t = lane & 3;
  const int P = blockIdx.x;  // tile of 64 * NTL columns
  const int mb = blockIdx.y;
  if (mb * BM >= *p.num_tokens_padded) return;
  const uint8_t* bi = p.ws + (long long)mb * BI_BYTES;
  const int nrows = *reinterpret_cast<const int*>(bi + BI_NROWS);
  const int e = *reinterpret_cast<const int*>(bi + BI_EXPERT);
  if (e < 0 || nrows == 0) return;
  // the block's lists in shared memory: no dependent global loads in the issue loop
  __shared__ int rows[BM];
  __shared__ uint16_t gmask[128];
  __shared__ uint16_t gact[128];
  for (int q = tid; q < 128; q += NTH) {
    if (q < BM) rows[q] = reinterpret_cast<const int*>(bi + BI_ROWS)[q];
    gmask[q] = reinterpret_cast<const uint16_t*>(bi + BI_MASK)[q];
    gact[q] = reinterpret_cast<const uint16_t*>(bi + BI_ACTG)[q];
  }
  const int nag = *reinterpret_cast<const int*>(bi + BI_NAG);
  __syncthreads();
  const int ntt = (nrows + 7) >> 3;
  const int Tl = warp >> 1;          // which 64-column tile of the CTA
  const int c0 = 2 * (warp & 1);     // first of this warp's two 16-column groups
  const uint8_t* wexp = p.W + e * p.w_expert + (long long)(NTL * P) * 32;
  const uint8_t* sexp = p.S + e * p.s_expert + (long long)(NTL * P) * 64;
  const int kr[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};

  float acc[8][2][4];
  bool dense = p.dense == 1;
  const bool loadonly = p.dense == 2;
  const bool computeonly = p.dense == 3;
  for (int pass = 0; pass < 2; ++pass) {
    const int nsteps = dense ? G : nag;
    const int nst = (nsteps + KG - 1) / KG;
    auto issue = [&](int st) {
      if (computeonly) return;
      Stage& S = stg[st % NS];
      for (int gi = 0; gi < KG; ++gi) {
        const int j = st * KG + gi;
        if (j >= nsteps) break;
        const int grp = dense ? j : gact[j];
        const uint32_t m = dense ? 0xFFFFu : gmask[grp];
        // weights: 16 rows x 64 bytes (4 pieces); scales: 128 bytes (8 pieces);
        // activations: nrows x 32 bytes (2 pieces)
        constexpr int WP = 2 * NTL;   // 16-byte pieces per weight row
        constexpr int SP = 4 * NTL;   // 16-byte pieces of scales
        for (int q = tid; q < 16 * WP + SP + 2 * nrows; q += NTH) {
          if (q < 16 * WP) {
            const int r = q / WP, piece = q % WP;
            if ((m >> r) & 1u)
              cp16(&S.w[gi][r][16 * piece], wexp + (long long)(16 * grp + r) * p.w_k + 16 * piece);
          } else if (q < 16 * WP + SP) {
            const int piece = q - 16 * WP;
            cp16(&S.s[gi][16 * piece], sexp + (long long)grp * p.s_g + 16 * piece);
          } else {
            const int r = (q - 16 * WP - SP) >> 1, piece = (q - 16 * WP - SP) & 1;
            cp16(&S.x[gi][r][16 * piece], p.A + (long long)rows[r] * p.lda + 16 * grp + 8 * piece);
          }
        }
      }
    };
#pragma unroll
    for (int tt = 0; tt < 8; ++tt)
#pragma unroll
      for (int c = 0; c < 2; ++c)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[tt][c][i] = 0.0f;
#pragma unroll
    for (int st = 0; st < NS - 1; ++st) {
      if (st < nst) issue(st);
      cp_commit();
    }
    bool neg0 = false;
    for (int st = 0; st < nst; ++st) {
      cp_wait<NS - 2>();
      __syncthreads();
      if (st + NS - 1 < nst) issue(st + NS - 1);
      cp_commit();
      const Stage& S = stg[st % NS];
      if (loadonly) continue;
      for (int gi = 0; gi < KG; ++gi) {
        const int j = st * KG + gi;
        if (j >= nsteps) break;
        const int grp = dense ? j : gact[j];
        const uint32_t m = dense ? 0xFFFFu : gmask[grp];
        const uint32_t w0 = *reinterpret_cast<const uint32_t*>(&S.w[gi][kr[0]][Tl * 32 + g8 * 4]);
        const uint32_t w1 = *reinterpret_cast<const uint32_t*>(&S.w[gi][kr[1]][Tl * 32 + g8 * 4]);
        const uint32_t w2 = *reinterpret_cast<const uint32_t*>(&S.w[gi][kr[2]][Tl * 32 + g8 * 4]);
        const uint32_t w3 = *reinterpret_cast<const uint32_t*>(&S.w[gi][kr[3]][Tl * 32 + g8 * 4]);
        const uint2 sc = *reinterpret_cast<const uint2*>(&S.s[gi][Tl * 64 + g8 * 8]);
        const uint32_t m01 = (m >> (2 * t)) & 3u;
        const uint32_t m23 = (m >> (2 * t + 8)) & 3u;
        const uint32_t keep01 = ((m01 & 1u) ? 0x0000FFFFu : 0u) | ((m01 & 2u) ? 0xFFFF0000u : 0u);
        const uint32_t keep23 = ((m23 & 1u) ? 0x0000FFFFu : 0u) | ((m23 & 2u) ? 0xFFFF0000u : 0u);
        uint32_t xb[8][2];
#pragma unroll
        for (int tt = 0; tt < 8; ++tt) {
          if (tt < ntt) {
            const uint8_t* xr = &S.x[gi][8 * tt + g8][0];
            xb[tt][0] = *reinterpret_cast<const uint32_t*>(xr + 4 * t);
            xb[tt][1] = *reinterpret_cast<const uint32_t*>(xr + 16 + 4 * t);
          } else {
            xb[tt][0] = xb[tt][1] = 0u;
          }
        }
#pragma unroll
        for (int ci = 0; ci < 2; ++ci) {
          const int c = c0 + ci;
          const uint32_t sx = (c < 2) ? sc.x : sc.y;
          const uint32_t sp = hmul2_u32(
              fp8_to_bf16x2(((c & 1) == 0) ? (((sx << 8) & 0x0000FF00u) | ((sx << 16) & 0xFF000000u))
                                           : (((sx >> 8) & 0x0000FF00u) | (sx & 0xFF000000u))),
              kExp120);
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
          for (int tt = 0; tt < 8; ++tt)
            if (tt < ntt) mma_bf16(acc[tt][ci], a0, a1, a2, a3, xb[tt][0], xb[tt][1]);
        }
      }
    }
    // A chain starts at +0 and round-to-nearest never yields -0 from it, so a -0 can only be a
    // final value; the insurance check (dense recompute) runs once on the final accumulators.
    if (!dense) {
#pragma unroll
      for (int tt = 0; tt < 8; ++tt)
#pragma unroll
        for (int ci = 0; ci < 2; ++ci)
#pragma unroll
          for (int q = 0; q < 4; ++q) neg0 |= __float_as_uint(acc[tt][ci][q]) == 0x80000000u;
    }
    cp_wait<0>();
    const bool redo = __syncthreads_or(neg0 ? 1 : 0) != 0;
    if (!redo || dense) break;
    if (tid == 0 && p.stats) atomicAdd(p.stats + 0, 1);
    dense = true;
  }
  const float gs4 = mul_ftz(p.GS[e], 4.0f);
#pragma unroll
  for (int tt = 0; tt < 8; ++tt) {
#pragma unroll
    for (int ci = 0; ci < 2; ++ci) {
      const int col = (NTL * P + Tl) * 64 + 16 * (c0 + ci) + g8;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int r = 8 * tt + 2 * t + (q & 1);
        const int cc = col + ((q >> 1) ? 8 : 0);
        if (r < nrows) {
          const uint32_t v = to_bf16x2(mul_ftz(acc[tt][ci][q], gs4), 0.0f);
          reinterpret_cast<uint16_t*>(p.C)[(long long)rows[r] * p.ldc + cc] = static_cast<uint16_t>(v);
        }
      }
    }
  }
}

// BMT = rows per expert block: 64 -> two 32-row halves (2 warps per 64-column tile); 32 or 16 ->
// one warp per 64-column tile covering the whole block (4 or 2 token tiles). Smaller blocks have
// sparser union masks (fewer weight rows loaded per block) at the cost of more blocks.
template <int KG, int NS, int NTL, int MINB = 1, int BMT = BM, int POLICY = 0>
__global__ void __launch_bounds__(32 * NTL * (BMT == 64 ? 2 : 1), MINB) smem4_kernel(const Params p) {
  constexpr int HALVES = BMT == 64 ? 2 : 1;
  constexpr int TTW = BMT == 64 ? 4 : BMT / 8;  // token tiles per warp
  constexpr int NTH = 32 * NTL * HALVES;
  extern __shared__ __align__(16) uint8_t smem_raw[];
  using Stage = SmemStage<KG, NS, NTL, BMT>;
  Stage* stg = reinterpret_cast<Stage*>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g8 = lane >> 2, t = lane & 3;
  const int P = blockIdx.x;  // tile of 64 * NTL columns
  const int mb = blockIdx.y;
  if (mb * BMT >= *p.num_tokens_padded) return;
  const uint8_t* bi = p.ws + (long long)mb * BI_BYTES;
  const int nrows = *reinterpret_cast<const int*>(bi + BI_NROWS);
  const int e = *reinterpret_cast<const int*>(bi + BI_EXPERT);
  if (e < 0 || nrows == 0) return;
  // the block's lists in shared memory: no dependent global loads in the issue loop
  __shared__ int rows[BMT];
  __shared__ uint16_t gmask[128];
  __shared__ uint16_t gact[128];
  for (int q = tid; q < 128; q += NTH) {
    if (q < BMT) rows[q] = reinterpret_cast<const int*>(bi + BI_ROWS)[q];
    gmask[q] = reinterpret_cast<const uint16_t*>(bi + BI_MASK)[q];
    gact[q] = reinterpret_cast<const uint16_t*>(bi + BI_ACTG)[q];
  }
  const int nag = *reinterpret_cast<const int*>(bi + BI_NAG);
  __syncthreads();
  const int ntt = (nrows + 7) >> 3;
  const int Tl = warp / HALVES;      // which 64-column tile of the CTA
  const int rh = warp % HALVES;      // which 32-row half of the block (BMT 64 only)
  const int c0 = 0;
  const int tt0 = TTW * rh;
  const bool active = 32 * rh < nrows;
  const uint8_t* wexp = p.W + e * p.w_expert + (long long)(NTL * P) * 32;
  const uint8_t* sexp = p.S + e * p.s_expert + (long long)(NTL * P) * 64;
  const int kr[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};

  float acc[TTW][4][4];
  bool dense = p.dense == 1;
  const bool loadonly = p.dense == 2;
  const bool computeonly = p.dense == 3;
  for (int pass = 0; pass < 2; ++pass) {
    const int nsteps = dense ? G : nag;
    const int nst = (nsteps + KG - 1) / KG;
    auto issue = [&](int st) {
      if (computeonly) return;
      Stage& S = stg[st % NS];
      for (int gi = 0; gi < KG; ++gi) {
        const int j = st * KG + gi;
        if (j >= nsteps) break;
        const int grp = dense ? j : gact[j];
        const uint32_t m = dense ? 0xFFFFu : gmask[grp];
        // weights: 16 rows x 64 bytes (4 pieces); scales: 128 bytes (8 pieces);
        // activations: nrows x 32 bytes (2 pieces)
        constexpr int WP = 2 * NTL;   // 16-byte pieces per weight row
        constexpr int SP = 4 * NTL;   // 16-byte pieces of scales
        for (int q = tid; q < 16 * WP + SP + 2 * nrows; q += NTH) {
          if (q < 16 * WP) {
            const int r = q / WP, piece = q % WP;
            if ((m >> r) & 1u)
              cp16_weight<POLICY>(&S.w[gi][r][16 * piece], wexp + (long long)(16 * grp + r) * p.w_k + 16 * piece);
          } else if (q < 16 * WP + SP) {
            const int piece = q - 16 * WP;
            cp16_weight<POLICY>(&S.s[gi][16 * piece], sexp + (long long)grp * p.s_g + 16 * piece);
          } else {
            const int r = (q - 16 * WP - SP) >> 1, piece = (q - 16 * WP - SP) & 1;
            cp16(&S.x[gi][r][16 * piece], p.A + (long long)rows[r] * p.lda + 16 * grp + 8 * piece);
          }
        }
      }
    };
#pragma unroll
    for (int tt = 0; tt < TTW; ++tt)
#pragma unroll
      for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[tt][c][i] = 0.0f;
#pragma unroll
    for (int st = 0; st < NS - 1; ++st) {
      if (st < nst) issue(st);
      cp_commit();
    }
    bool neg0 = false;
    for (int st = 0; st < nst; ++st) {
      cp_wait<NS - 2>();
      __syncthreads();
      if (st + NS - 1 < nst) issue(st + NS - 1);
      cp_commit();
      const Stage& S = stg[st % NS];
      if (loadonly || !active) continue;
      for (int gi = 0; gi < KG; ++gi) {
        const int j = st * KG + gi;
        if (j >= nsteps) break;
        const int grp = dense ? j : gact[j];
        const uint32_t m = dense ? 0xFFFFu : gmask[grp];
        const uint32_t w0 = *reinterpret_cast<const uint32_t*>(&S.w[gi][kr[0]][Tl * 32 + g8 * 4]);
        const uint32_t w1 = *reinterpret_cast<const uint32_t*>(&S.w[gi][kr[1]][Tl * 32 + g8 * 4]);
        const uint32_t w2 = *reinterpret_cast<const uint32_t*>(&S.w[gi][kr[2]][Tl * 32 + g8 * 4]);
        const uint32_t w3 = *reinterpret_cast<const uint32_t*>(&S.w[gi][kr[3]][Tl * 32 + g8 * 4]);
        const uint2 sc = *reinterpret_cast<const uint2*>(&S.s[gi][Tl * 64 + g8 * 8]);
        const uint32_t m01 = (m >> (2 * t)) & 3u;
        const uint32_t m23 = (m >> (2 * t + 8)) & 3u;
        const uint32_t keep01 = ((m01 & 1u) ? 0x0000FFFFu : 0u) | ((m01 & 2u) ? 0xFFFF0000u : 0u);
        const uint32_t keep23 = ((m23 & 1u) ? 0x0000FFFFu : 0u) | ((m23 & 2u) ? 0xFFFF0000u : 0u);
        uint32_t xb[TTW][2];
#pragma unroll
        for (int tt = 0; tt < TTW; ++tt) {
          if (tt0 + tt < ntt) {
            const uint8_t* xr = &S.x[gi][8 * (tt0 + tt) + g8][0];
            xb[tt][0] = *reinterpret_cast<const uint32_t*>(xr + 4 * t);
            xb[tt][1] = *reinterpret_cast<const uint32_t*>(xr + 16 + 4 * t);
          } else {
            xb[tt][0] = xb[tt][1] = 0u;
          }
        }
#pragma unroll
        for (int ci = 0; ci < 4; ++ci) {
          const int c = c0 + ci;
          const uint32_t sx = (c < 2) ? sc.x : sc.y;
          const uint32_t sp = hmul2_u32(
              fp8_to_bf16x2(((c & 1) == 0) ? (((sx << 8) & 0x0000FF00u) | ((sx << 16) & 0xFF000000u))
                                           : (((sx >> 8) & 0x0000FF00u) | (sx & 0xFF000000u))),
              kExp120);
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
          for (int tt = 0; tt < TTW; ++tt)
            if (tt0 + tt < ntt) mma_bf16(acc[tt][ci], a0, a1, a2, a3, xb[tt][0], xb[tt][1]);
        }
      }
    }
    // A chain starts at +0 and round-to-nearest never yields -0 from it, so a -0 can only be a
    // final value; the insurance check (dense recompute) runs once on the final accumulators.
    if (!dense) {
#pragma unroll
      for (int tt = 0; tt < TTW; ++tt)
#pragma unroll
        for (int ci = 0; ci < 4; ++ci)
#pragma unroll
          for (int q = 0; q < 4; ++q) neg0 |= __float_as_uint(acc[tt][ci][q]) == 0x80000000u;
    }
    cp_wait<0>();
    const bool redo = __syncthreads_or(neg0 ? 1 : 0) != 0;
    if (!redo || dense) break;
    if (tid == 0 && p.stats) atomicAdd(p.stats + 0, 1);
    dense = true;
  }
  const float gs4 = mul_ftz(p.GS[e], 4.0f);
#pragma unroll
  for (int tt = 0; tt < TTW; ++tt) {
#pragma unroll
    for (int ci = 0; ci < 4; ++ci) {
      const int col = (NTL * P + Tl) * 64 + 16 * (c0 + ci) + g8;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int r = 8 * (tt0 + tt) + 2 * t + (q & 1);
        const int cc = col + ((q >> 1) ? 8 : 0);
        if (r < nrows) {
          const uint32_t v = to_bf16x2(mul_ftz(acc[tt][ci][q], gs4), 0.0f);
          reinterpret_cast<uint16_t*>(p.C)[(long long)rows[r] * p.ldc + cc] = static_cast<uint16_t>(v);
        }
      }
    }
  }
}

// prep_kernel<BMT> launches; the prep_kernel instances are compiled in pzskip.cu
template <int BMT>
__attribute__((visibility("hidden"))) void launch_prep(const Params& p, cudaStream_t st);

template <int KG, int NS, int NTL = 2, int MINB = 1, int BMT = BM, int POLICY = 0>
static int launch_smem4(const Params& p, cudaStream_t st) {
  static_assert(N % (64 * NTL) == 0, "the column tile must divide N");
  static_assert(BMT == 16 || BMT == 32 || BMT == 64, "block rows");
  launch_prep<BMT>(p, st);
  cudaError_t e0 = cudaGetLastError();
  if (e0 != cudaSuccess) return (int)e0;
  const int sm = (int)sizeof(SmemStage<KG, NS, NTL, BMT>) * NS;
  static int attr = 0;
  if (sm > attr) {
    cudaError_t e = cudaFuncSetAttribute(smem4_kernel<KG, NS, NTL, MINB, BMT, POLICY>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    if (e != cudaSuccess) return (int)e;
    attr = sm;
  }
  dim3 grid(N / (64 * NTL), p.max_blocks);
  smem4_kernel<KG, NS, NTL, MINB, BMT, POLICY><<<grid, 32 * NTL * (BMT == 64 ? 2 : 1), sm, st>>>(p);
  return (int)cudaGetLastError();
}

template <int KG, int NS, int NTL = 2, int MINB = 1>
static int launch_smem(const Params& p, cudaStream_t st) {
  static_assert(N % (64 * NTL) == 0, "the column tile must divide N");
  const int sm = (int)sizeof(SmemStage<KG, NS, NTL>) * NS;
  static int attr = 0;
  if (sm > attr) {
    cudaError_t e = cudaFuncSetAttribute(smem_kernel<KG, NS, NTL, MINB>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    if (e != cudaSuccess) return (int)e;
    attr = sm;
  }
  dim3 grid(N / (64 * NTL), p.max_blocks);
  smem_kernel<KG, NS, NTL, MINB><<<grid, 64 * NTL, sm, st>>>(p);
  return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Configs 356/357: config 333's CTA and arithmetic (32-row block, 384 columns, 6 warps, warp =
// 64-column tile x the block's rows; per output element the same MMA chain over the block's
// active k16 groups and the same epilogue) with far fewer instructions per group. 333 is bound by
// instruction issue (its loads alone run at ~90% of its time, its compute alone at ~75%):
//  - each thread's cp.async roles are fixed per CTA (one weight piece; at most one scale or
//    activation piece), with 32-bit offsets instead of a divide-heavy piece loop;
//  - the token-tile count is a template parameter (no predicated MMAs / activation reads);
//  - weight rows the block does not load are not zeroed in registers. Exact: such a row holds a
//    stale but finite weight (the fp4 and fp8 bit decodes never yield Inf/NaN, the scales are
//    always loaded) and every valid row of the block has a +-0 activation there, so its products
//    are +-0 and leave the fp32 chain unchanged (the chain starts at +0 and is never -0 before its
//    end; the -0 insurance recompute of 333 is kept).
//  - prep2: prep's outputs, byte for byte, with the mask loads of 8 rows in flight per step.
// The grid is 333's one-shot grid (column tile fastest): the resident CTAs walk the weights in
// address order, a few consecutive experts at a time, all K-ascending in step.

// defined in pzskip_smem5.cu
__global__ void __launch_bounds__(128) prep2_kernel(const Params p);

// S6 (config 379): the swap scale plane as 6-bit offsets from S6_BASE (zskip py to_s6_swap):
// per expert, k16 group and 64-column tile 48 bytes, lane group g8's 8 low nibbles as a u32 at
// 4 g8 (scale j in bits 4j..4j+3) and their high 2-bit parts as a u16 at 32 + 2 g8 (bits 2j..2j+1).
// s6_pair rebuilds the 8 bytes the 8-bit plane's uint2 at 8 g8 holds.
constexpr uint32_t PZ_S6_BASE = 62u;
__device__ __forceinline__ uint2 s6_pair(const uint8_t* s, int Tl, int g8) {
  const uint32_t nib = *reinterpret_cast<const uint32_t*>(s + Tl * 48 + g8 * 4);
  const uint32_t two = *reinterpret_cast<const uint16_t*>(s + Tl * 48 + 32 + g8 * 2);
  uint32_t r[2];
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    const uint32_t n4 = nib >> (16 * q), t4 = two >> (8 * q);
    const uint32_t lo = (n4 & 0xFu) | ((n4 & 0xF0u) << 4) | ((n4 & 0xF00u) << 8) | ((n4 & 0xF000u) << 12);
    const uint32_t hi = (t4 & 0x3u) | ((t4 & 0xCu) << 6) | ((t4 & 0x30u) << 12) | ((t4 & 0xC0u) << 18);
    r[q] = (lo | (hi << 4)) + PZ_S6_BASE * 0x01010101u;
  }
  return make_uint2(r[0], r[1]);
}

// one k16 group of a warp's 64-column tile for NT token tiles of 8 rows
template <int NT, bool S6 = false>
__device__ __forceinline__ void s5_group(float (&acc)[4][4][4], const uint8_t (*w)[192], const uint8_t* s,
                                         const uint8_t (*x)[XS], int lane, int Tl) {
  const int g8 = lane >> 2, t = lane & 3;
  const uint32_t w0 = *reinterpret_cast<const uint32_t*>(&w[2 * t][Tl * 32 + g8 * 4]);
  const uint32_t w1 = *reinterpret_cast<const uint32_t*>(&w[2 * t + 1][Tl * 32 + g8 * 4]);
  const uint32_t w2 = *reinterpret_cast<const uint32_t*>(&w[2 * t + 8][Tl * 32 + g8 * 4]);
  const uint32_t w3 = *reinterpret_cast<const uint32_t*>(&w[2 * t + 9][Tl * 32 + g8 * 4]);
  const uint2 sc = S6 ? s6_pair(s, Tl, g8) : *reinterpret_cast<const uint2*>(&s[Tl * 64 + g8 * 8]);
  uint32_t xb[NT][2];
#pragma unroll
  for (int tt = 0; tt < NT; ++tt) {
    const uint8_t* xr = &x[8 * tt + g8][0];
    xb[tt][0] = *reinterpret_cast<const uint32_t*>(xr + 4 * t);
    xb[tt][1] = *reinterpret_cast<const uint32_t*>(xr + 16 + 4 * t);
  }
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const uint32_t sx = (c < 2) ? sc.x : sc.y;
    // the two scale bytes of column group c into the high bytes of the two halves (as 333)
    const uint32_t sp = hmul2_u32(fp8_to_bf16x2(__byte_perm(sx, 0u, (c & 1) == 0 ? 0x1404 : 0x3424)), kExp120);
    const uint32_t slo = __byte_perm(sp, 0, 0x1010);
    const uint32_t shi = __byte_perm(sp, 0, 0x3232);
    const uint32_t sel = c | (c << 4) | ((c + 4) << 8) | ((c + 4) << 12);
    const uint32_t x01 = __byte_perm(w0, w1, sel);
    const uint32_t x23 = __byte_perm(w2, w3, sel);
    const uint32_t a0 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x01 << 12), kExp124), slo);
    const uint32_t a1 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x01 << 8), kExp124), shi);
    const uint32_t a2 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x23 << 12), kExp124), slo);
    const uint32_t a3 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x23 << 8), kExp124), shi);
#pragma unroll
    for (int tt = 0; tt < NT; ++tt) mma_bf16(acc[tt][c], a0, a1, a2, a3, xb[tt][0], xb[tt][1]);
  }
}

template <int KG, int NS, int MINB, int POLICY = 0, bool S6 = false>
__global__ void __launch_bounds__(192, MINB) smem5_kernel(const Params p) {
  constexpr int BMT = 32, NTL = 6;
  extern __shared__ __align__(128) uint8_t s5_smem[];
  using Stage = SmemStage<KG, NS, NTL, BMT>;
  static_assert(sizeof(Stage) % 128 == 0, "stages stay 128-byte aligned");
  Stage* stg = reinterpret_cast<Stage*>(s5_smem);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g8 = lane >> 2, t = lane & 3;
  const int P = blockIdx.x;  // tile of 384 columns
  const int mb = blockIdx.y;
  if (mb * BMT >= *p.num_tokens_padded) return;
  const uint8_t* bi = p.ws + (long long)mb * BI_BYTES;
  const int nrows = *reinterpret_cast<const int*>(bi + BI_NROWS);
  const int e = *reinterpret_cast<const int*>(bi + BI_EXPERT);
  if (e < 0 || nrows == 0) return;
  struct alignas(128) Lists {  // a multiple of 128 bytes: the dynamic stages start 128-byte aligned
    int rows[BMT];
    uint16_t gmask[128];
    uint16_t gact[128];
  };
  __shared__ Lists L;
  for (int q = tid; q < 128; q += 192) {
    if (q < BMT) L.rows[q] = reinterpret_cast<const int*>(bi + BI_ROWS)[q];
    L.gmask[q] = reinterpret_cast<const uint16_t*>(bi + BI_MASK)[q];
    L.gact[q] = reinterpret_cast<const uint16_t*>(bi + BI_ACTG)[q];
  }
  const int nag = *reinterpret_cast<const int*>(bi + BI_NAG);
  __syncthreads();
  const int ntt = (nrows + 7) >> 3;
  const uint8_t* wexp = p.W + e * p.w_expert + (long long)(NTL * P) * 32;
  const uint8_t* sexp = p.S + e * p.s_expert + (long long)(NTL * P) * (S6 ? 48 : 64);
  // fixed cp.async roles: weight piece (tid / 12, tid % 12) of each group; an extra piece: scales
  // (tid < 24) or activation row (tid - 24) / 2, half (tid - 24) % 2 (tid - 24 < 2 nrows)
  const unsigned woff = static_cast<unsigned>((tid / 12) * p.w_k + 16 * (tid % 12));
  const unsigned xoff = (tid >= 24 && tid - 24 < 2 * nrows)
                            ? static_cast<unsigned>((long long)L.rows[(tid - 24) >> 1] * p.lda * 2 + 16 * ((tid - 24) & 1))
                            : 0u;
  float acc[4][4][4];
  bool dense = p.dense == 1;
  const bool loadonly = p.dense == 2;
  const bool computeonly = p.dense == 3;
  for (int pass = 0; pass < 2; ++pass) {
    const int nsteps = dense ? G : nag;
    const int nst = (nsteps + KG - 1) / KG;
    auto issue = [&](int st) {
      if (computeonly) return;
      Stage& S = stg[st % NS];
#pragma unroll
      for (int gi = 0; gi < KG; ++gi) {
        const int j = st * KG + gi;
        if (j < nsteps) {
          const int grp = dense ? j : L.gact[j];
          const uint32_t m = dense ? 0xFFFFu : L.gmask[grp];
          const int wr = tid / 12, wpc = tid - 12 * wr;
          if ((m >> wr) & 1u)
            cp16_weight<POLICY>(&S.w[gi][wr][16 * wpc], wexp + (woff + grp * (unsigned)(16 * p.w_k)));
          if (tid < 24) {
            if (!S6 || tid < 18)
              cp16_weight<POLICY>(&S.s[gi][16 * tid], sexp + (unsigned)(grp * p.s_g + 16 * tid));
          } else if (tid - 24 < 2 * nrows)
            cp16(&S.x[gi][(tid - 24) >> 1][16 * ((tid - 24) & 1)], reinterpret_cast<const uint8_t*>(p.A) + (xoff + 32u * grp));
        }
      }
    };
#pragma unroll
    for (int tt = 0; tt < 4; ++tt)
#pragma unroll
      for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[tt][c][i] = 0.0f;
    if constexpr (POLICY != 0) {
      // Not unrolled with the L2 hint: ptxas (CUDA 13.0, sm_121a) folds the unrolled stages'
      // constant shared-memory offsets into a uniform-register address on some hinted LDGSTS
      // and emits it with an unset descriptor register (illegal instruction at run time).
      // Same copies in the same order; the stock stream (POLICY 0) is unchanged.
#pragma unroll 1
      for (int st = 0; st < NS - 1; ++st) {
        if (st < nst) issue(st);
        cp_commit();
      }
    } else {
#pragma unroll
      for (int st = 0; st < NS - 1; ++st) {
        if (st < nst) issue(st);
        cp_commit();
      }
    }
    bool neg0 = false;
    for (int st = 0; st < nst; ++st) {
      cp_wait<NS - 2>();
      __syncthreads();
      if (st + NS - 1 < nst) issue(st + NS - 1);
      cp_commit();
      const Stage& S = stg[st % NS];
      if (loadonly) continue;
#pragma unroll 1
      for (int gi = 0; gi < KG; ++gi) {
        if (st * KG + gi >= nsteps) continue;
        switch (ntt) {
          case 1: s5_group<1, S6>(acc, S.w[gi], S.s[gi], S.x[gi], lane, warp); break;
          case 2: s5_group<2, S6>(acc, S.w[gi], S.s[gi], S.x[gi], lane, warp); break;
          case 3: s5_group<3, S6>(acc, S.w[gi], S.s[gi], S.x[gi], lane, warp); break;
          default: s5_group<4, S6>(acc, S.w[gi], S.s[gi], S.x[gi], lane, warp); break;
        }
      }
    }
    if (!dense) {
#pragma unroll
      for (int tt = 0; tt < 4; ++tt)
#pragma unroll
        for (int ci = 0; ci < 4; ++ci)
#pragma unroll
          for (int q = 0; q < 4; ++q) neg0 |= __float_as_uint(acc[tt][ci][q]) == 0x80000000u;
    }
    cp_wait<0>();
    const bool redo = __syncthreads_or(neg0 ? 1 : 0) != 0;
    if (!redo || dense) break;
    if (tid == 0 && p.stats) atomicAdd(p.stats + 0, 1);
    dense = true;
  }
  const float gs4 = mul_ftz(p.GS[e], 4.0f);
#pragma unroll
  for (int tt = 0; tt < 4; ++tt) {
#pragma unroll
    for (int ci = 0; ci < 4; ++ci) {
      const int col = (NTL * P + warp) * 64 + 16 * ci + g8;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int r = 8 * tt + 2 * t + (q & 1);
        const int cc = col + ((q >> 1) ? 8 : 0);
        if (r < nrows) {
          const uint32_t v = to_bf16x2(mul_ftz(acc[tt][ci][q], gs4), 0.0f);
          reinterpret_cast<uint16_t*>(p.C)[(long long)L.rows[r] * p.ldc + cc] = static_cast<uint16_t>(v);
        }
      }
    }
  }
}

template <int KG, int NS, int MINB, int POLICY = 0, bool S6 = false>
static int launch_smem5(const Params& p, cudaStream_t st) {
  prep2_kernel<<<p.max_blocks, 128, 0, st>>>(p);
  cudaError_t e0 = cudaGetLastError();
  if (e0 != cudaSuccess) return (int)e0;
  const int sm = (int)sizeof(SmemStage<KG, NS, 6, 32>) * NS;
  static int attr = 0;
  if (sm > attr) {
    cudaError_t e = cudaFuncSetAttribute(smem5_kernel<KG, NS, MINB, POLICY, S6>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    if (e != cudaSuccess) return (int)e;
    attr = sm;
  }
  dim3 grid(N / 384, p.max_blocks);
  smem5_kernel<KG, NS, MINB, POLICY, S6><<<grid, 192, sm, st>>>(p);
  return (int)cudaGetLastError();
}

// Config dispatch of the units (dispatch_smem4 and dispatch_smem: pzskip_smem.cu, dispatch_smem5:
// pzskip_smem5.cu, dispatch_down64 / dispatch_down32: pzskip_down64.cu / pzskip_down32.cu): config
// wpc's result, or NOT_HANDLED when wpc is not one of the unit's configs.
constexpr int NOT_HANDLED = INT_MIN;
__attribute__((visibility("hidden"))) int dispatch_smem4(int wpc, const Params& p, cudaStream_t st);
__attribute__((visibility("hidden"))) int dispatch_smem5(int wpc, const Params& p, cudaStream_t st);
__attribute__((visibility("hidden"))) int dispatch_smem(int wpc, const Params& p, cudaStream_t st);
__attribute__((visibility("hidden"))) int dispatch_down64(int wpc, const Params& p, cudaStream_t st);
__attribute__((visibility("hidden"))) int dispatch_down32(int wpc, const Params& p, cudaStream_t st);

}  // namespace pzskip

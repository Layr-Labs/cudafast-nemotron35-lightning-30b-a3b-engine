// scrtail: the draft-head screen's cascade after its first streaming dot
// (nemotron_h_mtp_screen.DraftHeadScreen._greedy_row, three-level path) as one CUDA kernel,
// one CTA per 2048-row chunk, writing the same candidate ids and the same fp32 logits.
//
// After the first dot (every row on its 16 most important groups, `scores`), the screen runs
// per chunk: top 1024 of the chunk's scores (_chunk_topk / sctopk), the mid dot over the next 16
// groups for those rows plus their first score, top 256 of those, the rest dot over the other 32
// groups plus the mid score, top 32 of those, and the full-K dot of the 32 survivors (BB-block
// partial terms, then their in-order sum, times the global scale). Here one CTA does all of it
// for its chunk:
//
// * Selections write the ids torch.topk(sorted=False) returns, in its order: every key above
//   the K-th largest in ascending index order, then the K-th largest's ties in ascending index
//   order until K are taken (keys: torch's order-preserving float conversion, -0 below +0,
//   every NaN the largest). The K-th largest key comes from an exact 8-bit radix select.
// * Every dot term is computed as _nvfp4_rows_dot / _nvfp4_rows_dot_part compute it for their
//   (16-row, 64-byte) tile (blocked layout: 8 lanes per row, lane lc holding bytes
//   [8 lc, 8 lc + 8) of the 64-byte block, i.e. one 16-element group): per byte
//   fma(lo, h[2 b], hi * h[2 b + 1]) with E2M1 values decoded by the hardware converter (signed
//   zeros as +0); the group's 8 byte terms added in byte order; then, over the 8 lanes, the
//   product with the group scale (e4m3) and the xor butterfly 4, 2, 1, whose first level adds the
//   partner's rounded product to the lane's own unrounded one (an fma, as compiled). Blocks are
//   accumulated from +0 in K order; the mid and rest scores are fma(gscale, acc, base); the full
//   logits are the in-order sum of the per-block terms (each + 0) times gscale.
// * The lanes of a row hold up to two different values after the butterfly (lanes 0-3 and 4-7);
//   the Triton kernels store one of them through shared memory. WIN selects that lane; the
//   Python side admits this kernel only after its outputs equal the screen's bit for bit.

#include <cuda_runtime.h>
#include <stdint.h>

namespace scrtail {

constexpr int CHUNK = 2048, PRE = 1024, CAS = 256, PER = 32;
constexpr int THREADS = 512, WARPS = THREADS / 32;
constexpr int NB_MID = 128, NB_REST = 256, NB_FULL = 1344;  // bytes per row
constexpr int NKB_FULL = NB_FULL / 64;                      // 21 blocks
constexpr unsigned FULL = 0xffffffffu;

struct Params {
  const float* scores;   // [n] first-dot scores (when the first dot is not fused)
  const float* hp;       // [256] h at the pre columns (fused first dot)
  const uint8_t* w_pre;  // [n, 128] (fused first dot)
  const uint8_t* s_pre;  // [n, 16] e4m3 (fused first dot)
  const float* hm;       // [256] h at the mid columns
  const float* hr;       // [512] h at the rest columns
  const float* h;        // [2688] h
  const uint8_t* w_mid;  // [n, 128]
  const uint8_t* s_mid;  // [n, 16] e4m3
  const uint8_t* w_rest; // [n, 256]
  const uint8_t* s_rest; // [n, 32]
  const uint8_t* w_full; // [n, 1344]
  const uint8_t* s_full; // [n, 168]
  long long* cand;       // [C, 32]
  float* cand_logits;    // [C, 32]
  // optional (admission): the cascade's intermediate buffers, as the screen writes them
  long long* pre_cand;   // [C, 1024]
  float* mid_scores;     // [C, 1024]
  float* scores_out;     // [n]: the mid score scattered at its row
  long long* coarse_cand;// [C, 256]
  float* refine_scores;  // [C, 256]
  const int* gate;       // stop flag at [0], or null
  float gscale;
};

__device__ __forceinline__ uint32_t topk_key(float v) {
  const uint32_t x = __float_as_uint(v);
  uint32_t key = (x & 0x80000000u) ? (x ^ 0xFFFFFFFFu) : (x ^ 0x80000000u);
  if (v != v) key = 0xFFFFFFFFu;
  return key;
}

__device__ __forceinline__ uint2 ld8(const uint8_t* p) {
  uint2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];\n" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t ld1(const uint8_t* p) {
  uint16_t r;
  asm volatile("ld.global.nc.L1::no_allocate.u8 %0, [%1];\n" : "=h"(r) : "l"(p));
  return r;
}

// e4m3 byte -> fp32 through the hardware f16x2 converter, as Triton's fp_to_fp.
__device__ __forceinline__ float e4m3f(uint32_t b) {
  uint32_t pair;
  uint16_t lo16;
  const uint16_t in = static_cast<uint16_t>(b);
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(pair) : "h"(in));
  asm("{ .reg .b16 hi; mov.b32 {%0, hi}, %1; }\n" : "=h"(lo16) : "r"(pair));
  float f;
  asm("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(lo16));
  return f;
}

// one byte's two E2M1 values (low nibble first) as _e2m1_pair: hardware decode, +0 for zeros.
__device__ __forceinline__ void e2m1_pair(uint32_t byte, float& lo, float& hi) {
  uint32_t pair;
  asm("{ .reg .b8 a, b, c, d; mov.b32 {a, b, c, d}, %1; cvt.rn.f16x2.e2m1x2 %0, a; }\n"
      : "=r"(pair) : "r"(byte));
  uint16_t l16, h16;
  asm("mov.b32 {%0, %1}, %2;\n" : "=h"(l16), "=h"(h16) : "r"(pair));
  asm("cvt.f32.f16 %0, %1;\n" : "=f"(lo) : "h"(l16));
  asm("cvt.f32.f16 %0, %1;\n" : "=f"(hi) : "h"(h16));
  lo = (lo == 0.0f) ? 0.0f : lo;
  hi = (hi == 0.0f) ? 0.0f : hi;
}

// One 64-byte block term of a row for lane lc (bytes 8 lc .. 8 lc + 7, scale byte sb):
// the value this lane holds after tl.sum(tl.sum(part, 2) * sc, 1).
__device__ __forceinline__ float block_term(uint2 wb, uint32_t sb, const float* hb) {
  float g = 0.0f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const uint32_t byte = ((j < 4 ? wb.x : wb.y) >> (8 * (j & 3))) & 0xffu;
    float lo, hi;
    e2m1_pair(byte, lo, hi);
    const float p = __fmaf_rn(lo, hb[2 * j], __fmul_rn(hi, hb[2 * j + 1]));
    g = (j == 0) ? p : __fadd_rn(g, p);
  }
  const float s = e4m3f(sb);
  const float prod = __fmul_rn(g, s);
  const float v = __fmaf_rn(g, s, __shfl_xor_sync(FULL, prod, 4));
  const float w = __fadd_rn(v, __shfl_xor_sync(FULL, v, 2));
  return __fadd_rn(w, __shfl_xor_sync(FULL, w, 1));
}

// The first dot's (64-row tile) block term of a row for lane lc: its blocked layout gives a
// thread 4 groups of a row (2 lanes per row) after the byte sums, so the 8 group sums combine as
// (p3 + fma(g2, s2, fma(g0, s0, p1))) + (p7 + fma(g6, s6, fma(g4, s4, p5))), p = rn(g * s).
__device__ __forceinline__ float pre_block_term(uint2 wb, uint32_t sb, const float* hb, int lc) {
  float g = 0.0f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const uint32_t byte = ((j < 4 ? wb.x : wb.y) >> (8 * (j & 3))) & 0xffu;
    float lo, hi;
    e2m1_pair(byte, lo, hi);
    const float p = __fmaf_rn(lo, hb[2 * j], __fmul_rn(hi, hb[2 * j + 1]));
    g = (j == 0) ? p : __fadd_rn(g, p);
  }
  const float s = e4m3f(sb);
  const float p = __fmul_rn(g, s);
  const int base = (threadIdx.x & 31) & ~3;  // lane of group 0 (or 4) of this 4-group half
  const float p1 = __shfl_sync(FULL, p, base + 1);
  const float g2 = __shfl_sync(FULL, g, base + 2);
  const float s2 = __shfl_sync(FULL, s, base + 2);
  const float p3 = __shfl_sync(FULL, p, base + 3);
  const float g0 = __shfl_sync(FULL, g, base);
  const float s0 = __shfl_sync(FULL, s, base);
  (void)lc;
  const float t = __fadd_rn(p3, __fmaf_rn(g2, s2, __fmaf_rn(g0, s0, p1)));
  return __fadd_rn(t, __shfl_xor_sync(FULL, t, 4));
}

struct Smem {
  float sc1[CHUNK];       // first-dot scores of the chunk
  float hp[256];
  float hm[256];
  float hr[512];
  float h[NB_FULL * 2];
  uint16_t pre[PRE];      // local rows, in pre_cand order
  float mid[PRE];
  uint16_t cas[CAS];      // indices into pre[] in coarse order
  float ref[CAS];
  uint16_t fin[PER];      // indices into cas[] in cand order
  float part[PER][NKB_FULL + 1];
  uint32_t hist[256];
  uint32_t scan[4 * 32];
  uint32_t sel[2];
};

#include "scrtail_rest_pipeline.cuh"

// Block-wide exclusive scan of (a, b) over all THREADS threads in thread order.
__device__ __forceinline__ void block_scan2(uint32_t a, uint32_t b, uint32_t& ea, uint32_t& eb,
                                            uint32_t& ta, uint32_t& tb, uint32_t* sh) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint32_t ia = a, ib = b;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t xa = __shfl_up_sync(FULL, ia, o);
    const uint32_t xb = __shfl_up_sync(FULL, ib, o);
    if (lane >= o) { ia += xa; ib += xb; }
  }
  if (lane == 31) { sh[warp] = ia; sh[WARPS + warp] = ib; }
  __syncthreads();
  if (warp == 0) {
    uint32_t wa = lane < WARPS ? sh[lane] : 0u, wb = lane < WARPS ? sh[WARPS + lane] : 0u;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const uint32_t xa = __shfl_up_sync(FULL, wa, o);
      const uint32_t xb = __shfl_up_sync(FULL, wb, o);
      if (lane >= o) { wa += xa; wb += xb; }
    }
    if (lane < WARPS) { sh[2 * WARPS + lane] = wa; sh[3 * WARPS + lane] = wb; }
  }
  __syncthreads();
  const uint32_t pa = warp > 0 ? sh[2 * WARPS + warp - 1] : 0u;
  const uint32_t pb = warp > 0 ? sh[3 * WARPS + warp - 1] : 0u;
  ea = pa + ia - a;
  eb = pb + ib - b;
  ta = sh[2 * WARPS + WARPS - 1];
  tb = sh[3 * WARPS + WARPS - 1];
  __syncthreads();
}

// Output slot (or -1) of each of this thread's E keys (indices t*E .. t*E+E-1, t < N/E) in the
// top-K list torch.topk(sorted=False) writes. All THREADS threads must call it; sm.hist must be
// zero on entry and is zero again on return.
template <int N, int K, int E>
__device__ __forceinline__ void topk_slots(const uint32_t (&key)[E], int (&slot)[E], Smem& sm) {
  constexpr int NT = N / E;
  const int t = threadIdx.x;
  const bool act = t < NT;
  uint32_t prefix = 0u, mask = 0u, left = K;
#pragma unroll 1
  for (int shift = 24; shift >= 0; shift -= 8) {
    // Histogram of the next digit over the keys that match the selected prefix.
    if (act) {
#pragma unroll
      for (int e = 0; e < E; ++e)
        if ((key[e] & mask) == prefix) atomicAdd(&sm.hist[(key[e] >> shift) & 255u], 1u);
    }
    __syncthreads();
    if (t < 32) {
      uint32_t local[8];
      uint32_t s = 0u;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        local[i] = sm.hist[8 * t + i];
        sm.hist[8 * t + i] = 0u;
        s += local[i];
      }
      uint32_t suf = s;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const uint32_t x = __shfl_down_sync(FULL, suf, o);
        if (t + o < 32) suf += x;
      }
      const uint32_t above = suf - s;
      if (above < left && suf >= left) {
        uint32_t acc = above;
        int pick = 8 * t;
#pragma unroll
        for (int i = 7; i >= 0; --i) {
          if (acc + local[i] >= left) { pick = 8 * t + i; break; }
          acc += local[i];
        }
        sm.sel[0] = static_cast<uint32_t>(pick);
        sm.sel[1] = left - acc;
      }
    }
    __syncthreads();
    const uint32_t pick = sm.sel[0];
    left = sm.sel[1];
    prefix |= pick << shift;
    mask |= 255u << shift;
  }
  const uint32_t thr = prefix;
  uint32_t ngt = 0u, neq = 0u;
  if (act) {
#pragma unroll
    for (int e = 0; e < E; ++e) { ngt += key[e] > thr; neq += key[e] == thr; }
  }
  uint32_t egt, eeq, tgt, teq;
  block_scan2(ngt, neq, egt, eeq, tgt, teq, sm.scan);
  const uint32_t need_eq = static_cast<uint32_t>(K) - tgt;
  uint32_t rg = egt, re = eeq;
#pragma unroll
  for (int e = 0; e < E; ++e) {
    int s = -1;
    if (act) {
      if (key[e] > thr) {
        s = static_cast<int>(rg++);
      } else if (key[e] == thr) {
        if (re < need_eq) s = static_cast<int>(tgt + re);
        ++re;
      }
    }
    slot[e] = s;
  }
}

template <bool PDL, int WIN, bool STAGE1, bool REST_PIPE = false>
__global__ void __launch_bounds__(THREADS) scrtail_kernel(Params p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
  if constexpr (PDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" :::);
  }
  const int c = blockIdx.x;
  const int t = threadIdx.x;
  const int lane = t & 31, warp = t >> 5;
  const int lc = lane & 7, sub = lane >> 3;
  const long long base_id = static_cast<long long>(c) * CHUNK;
  if (p.gate != nullptr && *p.gate != 0) {
    // Stopped draft pass: the gated screen's outputs (zero scores, first ids in index order).
    if (STAGE1 && p.scores_out != nullptr)
      for (int j = t; j < CHUNK; j += THREADS) p.scores_out[base_id + j] = 0.0f * p.gscale;
    for (int j = t; j < PER; j += THREADS) {
      p.cand[c * PER + j] = base_id + j;
      p.cand_logits[c * PER + j] = 0.0f * p.gscale;
    }
    if (p.pre_cand != nullptr) {
      for (int j = t; j < PRE; j += THREADS) {
        p.pre_cand[c * PRE + j] = base_id + j;
        p.mid_scores[c * PRE + j] = 0.0f * p.gscale;
      }
      for (int j = t; j < CAS; j += THREADS) {
        p.coarse_cand[c * CAS + j] = base_id + j;
        p.refine_scores[c * CAS + j] = 0.0f * p.gscale;
      }
    }
    return;
  }
  if constexpr (STAGE1)
    for (int i = t; i < 256; i += THREADS) sm.hp[i] = p.hp[i];
  for (int i = t; i < 256; i += THREADS) sm.hm[i] = p.hm[i];
  for (int i = t; i < 512; i += THREADS) sm.hr[i] = p.hr[i];
  for (int i = t; i < NB_FULL * 2; i += THREADS) sm.h[i] = p.h[i];
  for (int i = t; i < 256; i += THREADS) sm.hist[i] = 0u;
  __syncthreads();

  if constexpr (STAGE1) {
    // ---- the first dot: every row of the chunk on its 16 pre groups (2 blocks) -------------
    constexpr int ROWS_IT = WARPS * 4;  // 64 rows per iteration
    constexpr int U = 4;
#pragma unroll 1
    for (int it0 = 0; it0 < CHUNK / ROWS_IT; it0 += U) {
      uint2 wb[U][2];
      uint32_t sb[U][2];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const long long row = base_id + (it0 + u) * ROWS_IT + warp * 4 + sub;
#pragma unroll
        for (int b = 0; b < 2; ++b) {
          wb[u][b] = ld8(p.w_pre + row * 128 + 64 * b + 8 * lc);
          sb[u][b] = ld1(p.s_pre + row * 16 + 8 * b + lc);
        }
      }
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int i = (it0 + u) * ROWS_IT + warp * 4 + sub;
        float acc = 0.0f;
#pragma unroll
        for (int b = 0; b < 2; ++b)
          acc = __fadd_rn(acc, pre_block_term(wb[u][b], sb[u][b], sm.hp + 2 * (64 * b + 8 * lc), lc));
        if (lc == 0) sm.sc1[i] = __fmul_rn(p.gscale, acc);
      }
    }
    __syncthreads();
    if (p.scores_out != nullptr)
      for (int j = t; j < CHUNK; j += THREADS) p.scores_out[base_id + j] = sm.sc1[j];
  }
  // ---- top PRE of the chunk's first scores ------------------------------------------------
  {
    constexpr int E = CHUNK / THREADS;  // 4
    uint32_t key[E];
    int slot[E];
#pragma unroll
    for (int e = 0; e < E; ++e) {
      float v;
      if constexpr (STAGE1) {
        v = sm.sc1[t * E + e];
      } else {
        v = p.scores[base_id + t * E + e];
        sm.sc1[t * E + e] = v;
      }
      key[e] = topk_key(v);
    }
    topk_slots<CHUNK, PRE, E>(key, slot, sm);
#pragma unroll
    for (int e = 0; e < E; ++e)
      if (slot[e] >= 0) sm.pre[slot[e]] = static_cast<uint16_t>(t * E + e);
  }
  __syncthreads();
  if (p.pre_cand != nullptr)
    for (int j = t; j < PRE; j += THREADS) p.pre_cand[c * PRE + j] = base_id + sm.pre[j];

  // ---- mid dot: 1024 rows x 2 blocks, 8 lanes per row ---------------------------------------
  {
    constexpr int ROWS_IT = WARPS * 4;  // 64 rows per iteration
    constexpr int U = 4;                // iterations in flight
#pragma unroll 1
    for (int it0 = 0; it0 < PRE / ROWS_IT; it0 += U) {
      uint2 wb[U][2];
      uint32_t sb[U][2];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int q = (it0 + u) * ROWS_IT + warp * 4 + sub;
        const long long row = base_id + sm.pre[q];
#pragma unroll
        for (int b = 0; b < 2; ++b) {
          wb[u][b] = ld8(p.w_mid + row * NB_MID + 64 * b + 8 * lc);
          sb[u][b] = ld1(p.s_mid + row * (NB_MID / 8) + 8 * b + lc);
        }
      }
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int q = (it0 + u) * ROWS_IT + warp * 4 + sub;
        float acc = 0.0f;
#pragma unroll
        for (int b = 0; b < 2; ++b)
          acc = __fadd_rn(acc, block_term(wb[u][b], sb[u][b], sm.hm + 2 * (64 * b + 8 * lc)));
        const float out = __fmaf_rn(p.gscale, acc, sm.sc1[sm.pre[q]]);
        if (lc == WIN) sm.mid[q] = out;
      }
    }
  }
  __syncthreads();
  if (p.mid_scores != nullptr) {
    for (int j = t; j < PRE; j += THREADS) {
      p.mid_scores[c * PRE + j] = sm.mid[j];
      p.scores_out[base_id + sm.pre[j]] = sm.mid[j];
    }
  }

  // ---- top CAS of the mid scores ------------------------------------------------------------
  {
    constexpr int E = PRE / THREADS;  // 2
    uint32_t key[E];
    int slot[E];
#pragma unroll
    for (int e = 0; e < E; ++e) key[e] = topk_key(sm.mid[t * E + e]);
    topk_slots<PRE, CAS, E>(key, slot, sm);
#pragma unroll
    for (int e = 0; e < E; ++e)
      if (slot[e] >= 0) sm.cas[slot[e]] = static_cast<uint16_t>(t * E + e);
  }
  __syncthreads();
  if (p.coarse_cand != nullptr)
    for (int j = t; j < CAS; j += THREADS) p.coarse_cand[c * CAS + j] = base_id + sm.pre[sm.cas[j]];

  // ---- rest dot: 256 rows x 4 blocks ----------------------------------------------------------
  if constexpr (REST_PIPE) {
    constexpr size_t PACKET_OFFSET = (sizeof(Smem) + 15u) & ~size_t(15u);
    auto& packet = *reinterpret_cast<RestPacket*>(smem_raw + PACKET_OFFSET);
    rest_warp_pipeline<WIN>(p, sm, packet, base_id);
  } else {
    constexpr int ROWS_IT = WARPS * 4;  // 64
#pragma unroll 1
    for (int it = 0; it < CAS / ROWS_IT; ++it) {
      const int q = it * ROWS_IT + warp * 4 + sub;
      const int qi = sm.cas[q];
      const long long row = base_id + sm.pre[qi];
      uint2 wb[4];
      uint32_t sb[4];
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        wb[b] = ld8(p.w_rest + row * NB_REST + 64 * b + 8 * lc);
        sb[b] = ld1(p.s_rest + row * (NB_REST / 8) + 8 * b + lc);
      }
      float acc = 0.0f;
#pragma unroll
      for (int b = 0; b < 4; ++b)
        acc = __fadd_rn(acc, block_term(wb[b], sb[b], sm.hr + 2 * (64 * b + 8 * lc)));
      const float out = __fmaf_rn(p.gscale, acc, sm.mid[qi]);
      if (lc == WIN) sm.ref[q] = out;
    }
  }
  __syncthreads();
  if (p.refine_scores != nullptr)
    for (int j = t; j < CAS; j += THREADS) p.refine_scores[c * CAS + j] = sm.ref[j];

  // ---- top PER of the rest scores -------------------------------------------------------------
  {
    uint32_t key[1];
    int slot[1];
    key[0] = t < CAS ? topk_key(sm.ref[t]) : 0u;
    topk_slots<CAS, PER, 1>(key, slot, sm);
    if (slot[0] >= 0) sm.fin[slot[0]] = static_cast<uint16_t>(t);
  }
  __syncthreads();

  // ---- full dot of the 32 survivors: 32 rows x 21 blocks, one (row, block) per 8 lanes --------
  {
    constexpr int PAIRS = PER * NKB_FULL;  // 672
    constexpr int PAIRS_IT = WARPS * 4;    // 64
    constexpr int IT = (PAIRS + PAIRS_IT - 1) / PAIRS_IT;  // 11
    uint2 wb[IT];
    uint32_t sb[IT];
#pragma unroll
    for (int it = 0; it < IT; ++it) {
      const int u = it * PAIRS_IT + warp * 4 + sub;
      if (u < PAIRS) {
        const int r = u / NKB_FULL, kb = u % NKB_FULL;
        const long long row = base_id + sm.pre[sm.cas[sm.fin[r]]];
        wb[it] = ld8(p.w_full + row * NB_FULL + 64 * kb + 8 * lc);
        sb[it] = ld1(p.s_full + row * (NB_FULL / 8) + 8 * kb + lc);
      } else {
        wb[it] = make_uint2(0u, 0u);
        sb[it] = 0u;
      }
    }
#pragma unroll
    for (int it = 0; it < IT; ++it) {
      const int u = it * PAIRS_IT + warp * 4 + sub;  // uniform per 8 lanes
      if (it * PAIRS_IT + warp * 4 < PAIRS) {         // warp-uniform: shuffles stay full-warp
        const int uu = u < PAIRS ? u : 0;
        const int kb = uu % NKB_FULL;
        const float x = block_term(wb[it], sb[it], sm.h + 2 * (64 * kb + 8 * lc));
        if (u < PAIRS && lc == WIN) sm.part[uu / NKB_FULL][kb] = __fadd_rn(x, 0.0f);
      }
    }
  }
  __syncthreads();
  if (t < PER) {
    float acc = __fadd_rn(sm.part[t][0], 0.0f);
#pragma unroll
    for (int k = 1; k < NKB_FULL; ++k) acc = __fadd_rn(acc, sm.part[t][k]);
    const int qi = sm.cas[sm.fin[t]];
    p.cand[c * PER + t] = base_id + sm.pre[qi];
    p.cand_logits[c * PER + t] = __fmul_rn(acc, p.gscale);
  }
}

template <bool PDL, int WIN, bool STAGE1, bool REST_PIPE = false>
int launch(const Params& prm, int nchunk, cudaStream_t stream, bool pdl) {
  auto kern = scrtail_kernel<PDL, WIN, STAGE1, REST_PIPE>;
  constexpr size_t PACKET_OFFSET = (sizeof(Smem) + 15u) & ~size_t(15u);
  const int smem = static_cast<int>(REST_PIPE ? PACKET_OFFSET + sizeof(RestPacket)
                                             : sizeof(Smem));
  static bool attr_done = false;
  if (!attr_done) {
    cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e != cudaSuccess) return static_cast<int>(e);
    attr_done = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(nchunk);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  return static_cast<int>(cudaLaunchKernelEx(&cfg, kern, prm));
}

}  // namespace scrtail

extern "C" {

int scrtail_smem_bytes() { return static_cast<int>(sizeof(scrtail::Smem)); }

// pointers: scores, hm, hr, h, w_mid, s_mid, w_rest, s_rest, w_full, s_full, cand, cand_logits,
// pre_cand, mid_scores, scores_out, coarse_cand, refine_scores (last five may be null), gate.
// stage1 != 0: hp, w_pre, s_pre are used and the first dot is computed here (scores unused).
int scrtail_run(const void* scores, const void* hm, const void* hr, const void* h,
                const void* w_mid, const void* s_mid, const void* w_rest, const void* s_rest,
                const void* w_full, const void* s_full, void* cand, void* cand_logits,
                void* pre_cand, void* mid_scores, void* scores_out, void* coarse_cand,
                void* refine_scores, const void* gate, const void* hp, const void* w_pre,
                const void* s_pre, float gscale, int nchunk, int win, int pdl, int stage1,
                void* stream) {
  scrtail::Params prm;
  prm.scores = static_cast<const float*>(scores);
  prm.hp = static_cast<const float*>(hp);
  prm.w_pre = static_cast<const uint8_t*>(w_pre);
  prm.s_pre = static_cast<const uint8_t*>(s_pre);
  prm.hm = static_cast<const float*>(hm);
  prm.hr = static_cast<const float*>(hr);
  prm.h = static_cast<const float*>(h);
  prm.w_mid = static_cast<const uint8_t*>(w_mid);
  prm.s_mid = static_cast<const uint8_t*>(s_mid);
  prm.w_rest = static_cast<const uint8_t*>(w_rest);
  prm.s_rest = static_cast<const uint8_t*>(s_rest);
  prm.w_full = static_cast<const uint8_t*>(w_full);
  prm.s_full = static_cast<const uint8_t*>(s_full);
  prm.cand = static_cast<long long*>(cand);
  prm.cand_logits = static_cast<float*>(cand_logits);
  prm.pre_cand = static_cast<long long*>(pre_cand);
  prm.mid_scores = static_cast<float*>(mid_scores);
  prm.scores_out = static_cast<float*>(scores_out);
  prm.coarse_cand = static_cast<long long*>(coarse_cand);
  prm.refine_scores = static_cast<float*>(refine_scores);
  prm.gate = static_cast<const int*>(gate);
  prm.gscale = gscale;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  const bool use_pdl = pdl != 0;
  if (win != 0) return -1;
  if (stage1) return use_pdl ? scrtail::launch<true, 0, true>(prm, nchunk, s, true)
                             : scrtail::launch<false, 0, true>(prm, nchunk, s, false);
  return use_pdl ? scrtail::launch<true, 0, false>(prm, nchunk, s, true)
                 : scrtail::launch<false, 0, false>(prm, nchunk, s, false);
}

// Optional warp-role variant; only the post-first-dot path is supported.
int scrtail_run_restpipe(const void* scores, const void* hm, const void* hr, const void* h,
                const void* w_mid, const void* s_mid, const void* w_rest, const void* s_rest,
                const void* w_full, const void* s_full, void* cand, void* cand_logits,
                void* pre_cand, void* mid_scores, void* scores_out, void* coarse_cand,
                void* refine_scores, const void* gate, const void* hp, const void* w_pre,
                const void* s_pre, float gscale, int nchunk, int win, int pdl, int stage1,
                void* stream) {
  scrtail::Params prm;
  prm.scores = static_cast<const float*>(scores);
  prm.hp = static_cast<const float*>(hp);
  prm.w_pre = static_cast<const uint8_t*>(w_pre);
  prm.s_pre = static_cast<const uint8_t*>(s_pre);
  prm.hm = static_cast<const float*>(hm);
  prm.hr = static_cast<const float*>(hr);
  prm.h = static_cast<const float*>(h);
  prm.w_mid = static_cast<const uint8_t*>(w_mid);
  prm.s_mid = static_cast<const uint8_t*>(s_mid);
  prm.w_rest = static_cast<const uint8_t*>(w_rest);
  prm.s_rest = static_cast<const uint8_t*>(s_rest);
  prm.w_full = static_cast<const uint8_t*>(w_full);
  prm.s_full = static_cast<const uint8_t*>(s_full);
  prm.cand = static_cast<long long*>(cand);
  prm.cand_logits = static_cast<float*>(cand_logits);
  prm.pre_cand = static_cast<long long*>(pre_cand);
  prm.mid_scores = static_cast<float*>(mid_scores);
  prm.scores_out = static_cast<float*>(scores_out);
  prm.coarse_cand = static_cast<long long*>(coarse_cand);
  prm.refine_scores = static_cast<float*>(refine_scores);
  prm.gate = static_cast<const int*>(gate);
  prm.gscale = gscale;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  const bool use_pdl = pdl != 0;
  if (win != 0 || stage1 != 0) return -1;
  return use_pdl ? scrtail::launch<true, 0, false, true>(prm, nchunk, s, true)
                 : scrtail::launch<false, 0, false, true>(prm, nchunk, s, false);
}

}  // extern "C"

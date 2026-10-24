// marlinx: a Marlin-exact NVFP4 W4A16 GEMM for 1 <= M <= 8 bf16 rows.
//
// It reproduces, element for element, what vLLM v0.27.1's dense Marlin kernel
// (csrc/libtorch_stable/quantization/marlin, MarlinNvFp4LinearKernel path)
// computes for bf16 activations, float4_e2m1f weights with e4m3 (Marlin
// S0E5M3-processed) group-16 scales and an fp32 global scale, when Marlin
// picks thread config {thread_k 128, thread_n 128, 256 threads},
// m_block_size_8, group_blocks 1, fp32 global reduce and no atomics. That is
// the configuration Marlin selects for M <= 8 whenever N/128 * 4 > #SMs and
// K % 128 == N % 128 == 0 (see ../README.md for the derivation).
//
// Per output element (token t, feature n in 128-wide column `col`):
//   * Marlin's stripe partition over grid = #SMs cuts the column's k-tiles
//     (128 K each) into segments; each segment is reduced inside one CTA.
//   * Inside a segment, 4 warp rows w = 0..3 each chain
//     mma.m16n8k16.bf16 (weights as the A operand, "mma_trans") over the
//     k16 blocks 8*tau + 2*w and 8*tau + 2*w + 1, tau ascending, from +0.
//   * CTA partial P = acc0 + (acc1 + (acc3 + acc2))  (thread_block_reduce).
//   * Segments are folded bottom-up through fp32: R = P_bottom, R = P + R.
//   * out = bf16_rn(R * global_scale).
// The weight dequant (bit placement, bf16 scale multiply) and the operand
// placement are Marlin's, byte for byte. Only the schedule differs: one CTA of
// 4 warps owns a 64-feature half column for its whole K, streams it through a
// register ring, and does the segment reductions in shared memory.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace marlinx {

struct Params {
  const __nv_bfloat16* A;  // [M, lda] bf16 activations
  long long lda;           // elements
  int M;                   // 1..8
  const uint4* B;          // Marlin-packed fp4 weights, [K/16][N/2] int4
  const uint2* S;          // Marlin-processed e4m3 scales, [K/16][N] bytes
  const float* GS;         // Marlin-processed fp32 global scale (1 value)
  __nv_bfloat16* C;        // [M, ldc] bf16 output
  long long ldc;           // elements
  int N, K;
  int n_tiles, k_tiles;    // N/128, K/128
  int part1_cols;          // columns reduced by a single Marlin block
  int iters;               // Marlin stripe length (k-tiles per block) in part 2
#ifndef MARLINX_NO_KSKIP
  const uint32_t* BK;      // KSkip only: k-row repack of B (py/kskip_layout.py), [K/16][2N] u32
#endif
};

__device__ __forceinline__ uint4 ld_stream_v4(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

__device__ __forceinline__ uint2 ld_nc_v2(const uint2* p) {
  uint2 r;
  asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];\n" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}

__device__ __forceinline__ uint32_t ld_nc_u32(const uint32_t* p) {
  uint32_t r;
  asm volatile("ld.global.nc.u32 %0, [%1];\n" : "=r"(r) : "l"(p));
  return r;
}

__device__ __forceinline__ __nv_bfloat162 as_bf162(uint32_t v) {
  return *reinterpret_cast<__nv_bfloat162*>(&v);
}

__device__ __forceinline__ uint32_t as_u32(__nv_bfloat162 v) {
  return *reinterpret_cast<uint32_t*>(&v);
}

// Marlin dequant<nv_bfloat162, kFE2M1f, dequant_skip_flop = true>.
__device__ __forceinline__ void dequant_fp4(uint32_t q, uint32_t (&out)[2]) {
  uint32_t o1 = (q & 0x80008000u) | ((q & 0x70007000u) >> 6);
  q <<= 4;
  uint32_t o2 = (q & 0x80008000u) | ((q & 0x70007000u) >> 6);
  out[1] = o1;  // reverse indexing as in Marlin
  out[0] = o2;
}

// Marlin dequant_fp8_scales<nv_bfloat162, kFE4M3fn>.
__device__ __forceinline__ void dequant_scales(uint32_t q, uint32_t (&out)[2]) {
  uint32_t o1 = ((q & 0x80008000u) >> 1) | ((q & 0x7F007F00u) >> 4);
  q <<= 8;
  uint32_t o2 = ((q & 0x80008000u) >> 1) | ((q & 0x7F007F00u) >> 4);
  out[1] = o1;
  out[0] = o2;
}

// Marlin mma_trans<bf16>: weights (two n8 tiles) as the A operand, the
// activations of 8 tokens as the B operand.
__device__ __forceinline__ void mma_trans(const uint32_t (&a)[2], const uint32_t (&b)[2],
                                          const uint32_t (&b2)[2], float (&c)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
      : "r"(b[0]), "r"(b2[0]), "r"(b[1]), "r"(b2[1]), "r"(a[0]), "r"(a[1]), "f"(c[0]),
        "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// One k16 block of one warp: Marlin's matmul() body for m_block_size_8.
__device__ __forceinline__ void block_mma(const uint4& bq, const uint2& sq, const uint32_t (&a)[2],
                                          float (&acc)[4][4]) {
  uint32_t fs[4];
  {
    uint32_t t[2];
    dequant_scales(sq.x, t);
    fs[0] = t[0];
    fs[1] = t[1];
    dequant_scales(sq.y, t);
    fs[2] = t[0];
    fs[3] = t[1];
  }
  const uint32_t q[4] = {bq.x, bq.y, bq.z, bq.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    uint32_t b0[2], b1[2];
    dequant_fp4(q[j] << 8, b0);  // b_quant_0 = b_quant_1 << 8
    dequant_fp4(q[j], b1);       // b_quant_1
    const __nv_bfloat162 s = as_bf162(fs[j]);
    const __nv_bfloat162 sx = __bfloat162bfloat162(s.x);
    const __nv_bfloat162 sy = __bfloat162bfloat162(s.y);
    b0[0] = as_u32(__hmul2(as_bf162(b0[0]), sx));
    b0[1] = as_u32(__hmul2(as_bf162(b0[1]), sx));
    b1[0] = as_u32(__hmul2(as_bf162(b1[0]), sy));
    b1[1] = as_u32(__hmul2(as_bf162(b1[1]), sy));
    mma_trans(a, b0, b1, acc[j]);
  }
}

// Walks one column's k16 blocks for one warp chain: segments bottom-up
// (highest K first), tiles ascending inside a segment, k in {0, 1}.
struct Cursor {
  int tau, k, lo, hi;
};

__device__ __forceinline__ void cursor_init(Cursor& c, int first_cut, int k_tiles, int iters) {
  c.hi = k_tiles;
  c.lo = first_cut >= k_tiles ? 0 : first_cut + ((k_tiles - 1 - first_cut) / iters) * iters;
  c.tau = c.lo;
  c.k = 0;
}

// Advances past the current block; true if that block closed its segment.
__device__ __forceinline__ bool cursor_next(Cursor& c, int iters) {
  c.k ^= 1;
  if (c.k) return false;
  if (++c.tau < c.hi) return false;
  c.hi = c.lo;
  c.lo = c.lo - iters > 0 ? c.lo - iters : 0;
  c.tau = c.lo;
  return true;
}

// Segment end: Marlin's thread_block_reduce (P = acc0 + (acc1 + (acc3 + acc2)))
// and global_reduce_fp32 (R = P + R, bottom segment first). red is double-buffered
// by segment parity, so one barrier per segment suffices.
__device__ __forceinline__ void close_segment(float (&red)[2][3][16][32], int par, int w, int lane,
                                              float (&acc)[4][4], float (&R)[4][4], bool first_seg) {
  if (w) {
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) red[par][w - 1][4 * j + e][lane] = acc[j][e];
  }
  __syncthreads();
  if (!w) {
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float a1 = red[par][0][4 * j + e][lane];
        const float a2 = red[par][1][4 * j + e][lane];
        const float a3 = red[par][2][4 * j + e][lane];
        const float pp = acc[j][e] + (a1 + (a3 + a2));
        R[j][e] = first_seg ? pp : pp + R[j][e];
      }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[j][e] = 0.0f;
}

// Preserve the original GEMM BF16 rounding before optional ReLU squared.
// The build retains default IEEE subnormal handling (no --use_fast_math).
template <bool Relu2>
__device__ __forceinline__ __nv_bfloat16 round_output(float scaled) {
  const __nv_bfloat16 rounded = __float2bfloat16_rn(scaled);
  if constexpr (Relu2) {
    float value = __bfloat162float(rounded);
    value = value < 0.0f ? 0.0f : value;
    const float squared = __fmul_rn(value, value);
    return squared != squared ? __ushort_as_bfloat16(0x7fff)
                              : __float2bfloat16_rn(squared);
  }
  return rounded;
}

// Marlin write_result: bf16_rn(R * global_scale), rows < M only (warp 0).
template <bool Relu2>
__device__ __forceinline__ void write_out(const Params& p, int col, int h, int lane,
                                          const float (&R)[4][4]) {
  const float g = __ldg(p.GS);
  const int t0 = 2 * (lane & 3);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const long long f = 128LL * col + 64 * h + 16 * j + (lane >> 2);
    const __nv_bfloat16 v0 = round_output<Relu2>(R[j][0] * g);
    const __nv_bfloat16 v1 = round_output<Relu2>(R[j][1] * g);
    const __nv_bfloat16 v2 = round_output<Relu2>(R[j][2] * g);
    const __nv_bfloat16 v3 = round_output<Relu2>(R[j][3] * g);
    if (t0 < p.M) {
      p.C[t0 * p.ldc + f] = v0;
      p.C[t0 * p.ldc + f + 8] = v2;
    }
    if (t0 + 1 < p.M) {
      p.C[(t0 + 1) * p.ldc + f] = v1;
      p.C[(t0 + 1) * p.ldc + f + 8] = v3;
    }
  }
}

__device__ __forceinline__ int column_first_cut(const Params& p, int col) {
  if (col < p.part1_cols) return p.k_tiles;
  const int l0 = (col - p.part1_cols) * p.k_tiles;
  const int cut = (l0 / p.iters + 1) * p.iters - l0;
  return cut < p.k_tiles ? cut : p.k_tiles;
}

#ifndef MARLINX_NO_KSKIP
// KSkip (shared expert down, N 2688 K 3712): the weights come from BK, the load-time k-row
// repack of B (py/kskip_layout.py; every nibble kept). BK u32 ((kb*(N/64) + unit)*16 + r)*8 + g
// holds k row r of the 64-feature unit for lanes 4g + t: byte j = nibble of feature 16j+g |
// nibble of feature 16j+g+8 << 4. Lane (g, t) loads k rows 2t, 2t+1, 2t+8, 2t+9 (q[0..3]) and
// rebuilds the exact crown int4 word j with kskip_word; every Marlin operation after that is
// unchanged. A k row whose activation is +-0 in every row of the call is not loaded (q = 0):
// its nibbles dequantise to +0, the +0 * scale products stay +0, and they only meet +-0
// activations in the MMA (see the -0 guard in gemm_kernel).
__device__ __forceinline__ uint32_t kskip_word(uint32_t r0, uint32_t r1, uint32_t r8, uint32_t r9,
                                               int j) {
  const int s = 8 * j;
  return ((r0 >> s) & 0xFu) | (((r8 >> s) & 0xFu) << 4) | (((r0 >> (s + 4)) & 0xFu) << 8) |
         (((r8 >> (s + 4)) & 0xFu) << 12) | (((r1 >> s) & 0xFu) << 16) | (((r9 >> s) & 0xFu) << 20) |
         (((r1 >> (s + 4)) & 0xFu) << 24) | (((r9 >> (s + 4)) & 0xFu) << 28);
}

__device__ __forceinline__ uint4 kskip_words(const uint32_t (&q)[4]) {
  uint4 v;
  v.x = kskip_word(q[0], q[1], q[2], q[3], 0);
  v.y = kskip_word(q[0], q[1], q[2], q[3], 1);
  v.z = kskip_word(q[0], q[1], q[2], q[3], 2);
  v.w = kskip_word(q[0], q[1], q[2], q[3], 3);
  return v;
}

// Loads one k16 block's four BK words of this lane. a = this lane's activation words of the
// block (ra layout: a[0] = k 2t | 2t+1 << 16, a[1] = k 2t+8 | 2t+9 << 16, token lane >> 2, 0 for
// rows >= M). Must be called by the whole warp. tm = 0x11111111 << t selects the lanes 4g + t.
__device__ __forceinline__ void kskip_load(uint32_t (&q)[4], const uint32_t* kp,
                                           const uint32_t (&a)[2], uint32_t tm, bool full) {
  const uint32_t b0 = __ballot_sync(0xFFFFFFFFu, (a[0] & 0x7FFFu) != 0u);
  const uint32_t b1 = __ballot_sync(0xFFFFFFFFu, (a[0] & 0x7FFF0000u) != 0u);
  const uint32_t b8 = __ballot_sync(0xFFFFFFFFu, (a[1] & 0x7FFFu) != 0u);
  const uint32_t b9 = __ballot_sync(0xFFFFFFFFu, (a[1] & 0x7FFF0000u) != 0u);
  q[0] = (full || (b0 & tm)) ? ld_nc_u32(kp) : 0u;
  q[1] = (full || (b1 & tm)) ? ld_nc_u32(kp + 8) : 0u;
  q[2] = (full || (b8 & tm)) ? ld_nc_u32(kp + 64) : 0u;
  q[3] = (full || (b9 & tm)) ? ld_nc_u32(kp + 72) : 0u;
}

#endif
// One 64-feature half column (`unit`) per CTA of 4 warps; warp w runs Marlin's
// warp-row chain w through a register ring of P k16 blocks.
#ifndef MARLINX_NO_KSKIP
template <int P, bool Relu2 = false, bool KSkip = false>
#else
template <int P, bool Relu2 = false>
#endif
__global__ void __launch_bounds__(128) gemm_kernel(const Params p) {
  __shared__ float red[2][3][16][32];
  const int unit = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int w = threadIdx.x >> 5;  // Marlin warp row (chain)
  const int col = unit >> 1;
  const int h = unit & 1;  // Marlin warp column (64-feature half)

  // First internal segment cut of this column (k-tiles), k_tiles if none.
  const int first_cut = column_first_cut(p, col);

  const long long nB = p.N >> 1;  // int4 per k16 row
  const long long nS = p.N >> 3;  // uint2 per k16 row
  const uint4* Bp = p.B + 64 * col + 32 * h + lane;
  const uint2* Sp = p.S + 16 * col + 8 * h + (lane >> 2);
  const int tok = lane >> 2;
  const bool arow = tok < p.M;
  const uint32_t* Ap =
      reinterpret_cast<const uint32_t*>(p.A + (arow ? (long long)tok * p.lda : 0)) + (lane & 3);

#ifndef MARLINX_NO_KSKIP
  if constexpr (KSkip) {
    // Same chains, blocks, MMAs, segment folds and epilogue as below; only the weight words come
    // from BK (kskip_load / kskip_words). The activations run P blocks ahead of the weights (ring
    // na, cursor la) because a block's weight loads depend on its activations. Pass 0 skips the
    // all-zero k rows; if any accumulator of the CTA ever reads -0 (the case the MMA equality
    // argument excludes), pass 1 recomputes the whole CTA with every row loaded.
    const long long nK = 2LL * p.N;  // BK u32 per k16 row
    const uint32_t* Kp = p.BK + 128LL * unit + 16 * (lane & 3) + tok;
    const uint32_t tm = 0x11111111u << (lane & 3);
    const int total = 2 * p.k_tiles;
    float acc[4][4];
    float R[4][4];
    for (int pass = 0; pass < 2; ++pass) {
      const bool full = pass != 0;
      uint32_t rq[P][4];
      uint2 rs[P];
      uint32_t ra[P][2];
      uint32_t na[P][2];
      Cursor ld, cs, la;
      cursor_init(ld, first_cut, p.k_tiles, p.iters);
      cs = ld;
      la = ld;
#pragma unroll
      for (int s = 0; s < P; ++s) {
        if (s < total) {
          const int kb = 8 * la.tau + 2 * w + la.k;
          ra[s][0] = arow ? ld_nc_u32(Ap + 8 * kb) : 0u;
          ra[s][1] = arow ? ld_nc_u32(Ap + 8 * kb + 4) : 0u;
          cursor_next(la, p.iters);
        }
      }
#pragma unroll
      for (int s = 0; s < P; ++s) {
        if (s < total) {
          const int kb = 8 * ld.tau + 2 * w + ld.k;
          kskip_load(rq[s], Kp + kb * nK, ra[s], tm, full);
          rs[s] = ld_nc_v2(Sp + kb * nS);
          cursor_next(ld, p.iters);
        }
      }
#pragma unroll
      for (int s = 0; s < P; ++s) {
        if (s + P < total) {
          const int kb = 8 * la.tau + 2 * w + la.k;
          na[s][0] = arow ? ld_nc_u32(Ap + 8 * kb) : 0u;
          na[s][1] = arow ? ld_nc_u32(Ap + 8 * kb + 4) : 0u;
          cursor_next(la, p.iters);
        }
      }
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          acc[j][e] = 0.0f;
          R[j][e] = 0.0f;
        }
      bool first_seg = true;
      int par = 0;
      bool neg0 = false;
      for (int base = 0; base < total; base += P) {
#pragma unroll
        for (int s = 0; s < P; ++s) {
          const int i = base + s;
          if (i < total) {
            block_mma(kskip_words(rq[s]), rs[s], ra[s], acc);
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
              for (int e = 0; e < 4; ++e) neg0 |= __float_as_uint(acc[j][e]) == 0x80000000u;
            const bool seg_end = cursor_next(cs, p.iters);
            if (i + P < total) {
              ra[s][0] = na[s][0];
              ra[s][1] = na[s][1];
              const int kb = 8 * ld.tau + 2 * w + ld.k;
              kskip_load(rq[s], Kp + kb * nK, ra[s], tm, full);
              rs[s] = ld_nc_v2(Sp + kb * nS);
              cursor_next(ld, p.iters);
              if (i + 2 * P < total) {
                const int kb2 = 8 * la.tau + 2 * w + la.k;
                na[s][0] = arow ? ld_nc_u32(Ap + 8 * kb2) : 0u;
                na[s][1] = arow ? ld_nc_u32(Ap + 8 * kb2 + 4) : 0u;
                cursor_next(la, p.iters);
              }
            }
            if (seg_end) {  // uniform across the CTA: every chain has the same length
              close_segment(red, par, w, lane, acc, R, first_seg);
              first_seg = false;
              par ^= 1;
            }
          }
        }
      }
      if (!__syncthreads_or(neg0 ? 1 : 0) || full) break;
    }
    if (!w) write_out<Relu2>(p, col, h, lane, R);
    return;
  }

#endif
  uint4 rb[P];
  uint2 rs[P];
  uint32_t ra[P][2];

  Cursor ld, cs;
  cursor_init(ld, first_cut, p.k_tiles, p.iters);
  cs = ld;
  const int total = 2 * p.k_tiles;

#pragma unroll
  for (int s = 0; s < P; ++s) {
    if (s < total) {
      const int kb = 8 * ld.tau + 2 * w + ld.k;
      rb[s] = ld_stream_v4(Bp + kb * nB);
      rs[s] = ld_nc_v2(Sp + kb * nS);
      ra[s][0] = arow ? ld_nc_u32(Ap + 8 * kb) : 0u;
      ra[s][1] = arow ? ld_nc_u32(Ap + 8 * kb + 4) : 0u;
      cursor_next(ld, p.iters);
    }
  }

  float acc[4][4];
  float R[4][4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      acc[j][e] = 0.0f;
      R[j][e] = 0.0f;
    }
  bool first_seg = true;
  int par = 0;

  for (int base = 0; base < total; base += P) {
#pragma unroll
    for (int s = 0; s < P; ++s) {
      const int i = base + s;
      if (i < total) {
        block_mma(rb[s], rs[s], ra[s], acc);
        const bool seg_end = cursor_next(cs, p.iters);
        if (i + P < total) {
          const int kb = 8 * ld.tau + 2 * w + ld.k;
          rb[s] = ld_stream_v4(Bp + kb * nB);
          rs[s] = ld_nc_v2(Sp + kb * nS);
          ra[s][0] = arow ? ld_nc_u32(Ap + 8 * kb) : 0u;
          ra[s][1] = arow ? ld_nc_u32(Ap + 8 * kb + 4) : 0u;
          cursor_next(ld, p.iters);
        }
        if (seg_end) {  // uniform across the CTA: every chain has the same length
          close_segment(red, par, w, lane, acc, R, first_seg);
          first_seg = false;
          par ^= 1;
        }
      }
    }
  }
  if (!w) write_out<Relu2>(p, col, h, lane, R);
}

// Streaming-read kernel used by the tests as a bandwidth probe and as a
// stand-in for a concurrent memory-bound kernel (dynamic smem reserves SM space).
template <int U>
__global__ void hog_kernel(const uint4* __restrict__ src, long long n16, unsigned* sink) {
  extern __shared__ unsigned hog_smem[];
  unsigned x = 0;
  const long long stride = (long long)gridDim.x * blockDim.x;
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i + (U - 1) * stride < n16; i += U * stride) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = ld_stream_v4(src + i + u * stride);
#pragma unroll
    for (int u = 0; u < U; ++u) x ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  for (; i < n16; i += stride) {
    uint4 v = ld_stream_v4(src + i);
    x ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (x == 0x9e3779b9u) sink[0] = x + (unsigned)(uintptr_t)hog_smem;
}

template <int P, bool Relu2 = false>
static int launch(const Params& p, cudaStream_t stream) {
  gemm_kernel<P, Relu2><<<2 * p.n_tiles, 128, 0, stream>>>(p);
  return (int)cudaGetLastError();
}

#ifndef MARLINX_NO_KSKIP
template <int P>
static int launch_kskip(const Params& p, cudaStream_t stream) {
  gemm_kernel<P, false, true><<<2 * p.n_tiles, 128, 0, stream>>>(p);
  return (int)cudaGetLastError();
}

#endif
}  // namespace marlinx

extern "C" {

// Forces the kernels to load (call outside CUDA graph capture). Returns 0 or a
// cudaError_t value.
int marlinx_init(void) {
  cudaFuncAttributes attr;
  cudaError_t e;
  if ((e = cudaFuncGetAttributes(&attr, marlinx::gemm_kernel<4>)) != cudaSuccess) return (int)e;
  if ((e = cudaFuncGetAttributes(&attr, marlinx::gemm_kernel<8>)) != cudaSuccess) return (int)e;
  if ((e = cudaFuncGetAttributes(&attr, marlinx::gemm_kernel<4, true>)) != cudaSuccess) return (int)e;
  if ((e = cudaFuncGetAttributes(&attr, marlinx::gemm_kernel<8, true>)) != cudaSuccess) return (int)e;
  if ((e = cudaFuncGetAttributes(&attr, marlinx::hog_kernel<8>)) != cudaSuccess) return (int)e;
  return 0;
}

// Registers per thread of gemm_kernel<depth> (depth 4 or 8), or -1.
int marlinx_regs(int depth) {
  cudaFuncAttributes attr;
  cudaError_t e;
  switch (depth) {
    case 4: e = cudaFuncGetAttributes(&attr, marlinx::gemm_kernel<4>); break;
    case 8: e = cudaFuncGetAttributes(&attr, marlinx::gemm_kernel<8>); break;
    default: return -1;
  }
  return e == cudaSuccess ? attr.numRegs : -1;
}

// out[M, N] = Marlin NVFP4 GEMM of A[M, K]. sms is the SM count Marlin uses as
// its grid (cudaDevAttrMultiProcessorCount). Returns 0, a cudaError_t value, or
// a negative code for unsupported arguments.
static int marlinx_gemm_impl(const void* A, long long lda, int M, const void* B, const void* S,
                 const void* GS, void* C, long long ldc, int N, int K, int sms, int depth,
                 void* stream, bool relu2) {
  if (M < 1 || M > 8) return -1;
  if (N <= 0 || K <= 0 || N % 128 || K % 128) return -2;
  if (sms <= 0) return -3;
  if (lda % 8 || (reinterpret_cast<uintptr_t>(A) & 15)) return -4;
  marlinx::Params p;
  p.A = static_cast<const __nv_bfloat16*>(A);
  p.lda = lda;
  p.M = M;
  p.B = static_cast<const uint4*>(B);
  p.S = static_cast<const uint2*>(S);
  p.GS = static_cast<const float*>(GS);
  p.C = static_cast<__nv_bfloat16*>(C);
  p.ldc = ldc;
  p.N = N;
  p.K = K;
  p.n_tiles = N / 128;
  p.k_tiles = K / 128;
  // Marlin's stripe partition with parallel = 1 and grid = sms.
  int part2 = p.n_tiles;
  if (p.n_tiles > sms) {
    part2 = p.n_tiles % sms;
    if (part2 * 3 <= sms) part2 += sms;
  }
  p.part1_cols = p.n_tiles - part2;
  p.iters = (p.k_tiles * part2 + sms - 1) / sms;
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  if (relu2) {
    if (N != 3712 || K != 2688) return -6;
    switch (depth) {
      case 4: return marlinx::launch<4, true>(p, st);
      case 8: return marlinx::launch<8, true>(p, st);
      default: return -5;
    }
  }
  switch (depth) {
    case 4: return marlinx::launch<4>(p, st);
    case 8: return marlinx::launch<8>(p, st);
    default: return -5;
  }
}

// Keep the original ABI and unfused instantiations for every existing call.
int marlinx_gemm(const void* A, long long lda, int M, const void* B, const void* S,
                 const void* GS, void* C, long long ldc, int N, int K, int sms, int depth,
                 void* stream) {
  return marlinx_gemm_impl(A, lda, M, B, S, GS, C, ldc, N, K, sms, depth, stream, false);
}

int marlinx_gemm_relu2(const void* A, long long lda, int M, const void* B, const void* S,
                 const void* GS, void* C, long long ldc, int N, int K, int sms, int depth,
                 void* stream) {
  return marlinx_gemm_impl(A, lda, M, B, S, GS, C, ldc, N, K, sms, depth, stream, true);
}

#ifndef MARLINX_NO_KSKIP
// Loads the KSkip kernels (call outside CUDA graph capture). Returns 0 or a cudaError_t value.
// Separate from marlinx_init so that a failure here only disables KSkip.
int marlinx_kskip_init(void) {
  cudaFuncAttributes attr;
  cudaError_t e;
  if ((e = cudaFuncGetAttributes(&attr, marlinx::gemm_kernel<4, false, true>)) != cudaSuccess) return (int)e;
  if ((e = cudaFuncGetAttributes(&attr, marlinx::gemm_kernel<8, false, true>)) != cudaSuccess) return (int)e;
  return 0;
}

// marlinx_gemm for the shared expert down projection (N 2688, K 3712 only) with the KSkip
// kernel: BK is the k-row repack of B (py/__init__.py _kskip_repack). Same output bits as
// marlinx_gemm (checked at load time). Returns 0, a cudaError_t value, or a negative code.
int marlinx_gemm_kskip(const void* A, long long lda, int M, const void* B, const void* BK,
                       const void* S, const void* GS, void* C, long long ldc, int N, int K,
                       int sms, int depth, void* stream) {
  if (M < 1 || M > 8) return -1;
  if (N != 2688 || K != 3712) return -7;
  if (sms <= 0) return -3;
  if (lda % 8 || (reinterpret_cast<uintptr_t>(A) & 15)) return -4;
  if (BK == nullptr || (reinterpret_cast<uintptr_t>(BK) & 31)) return -8;
  marlinx::Params p;
  p.A = static_cast<const __nv_bfloat16*>(A);
  p.lda = lda;
  p.M = M;
  p.B = static_cast<const uint4*>(B);
  p.S = static_cast<const uint2*>(S);
  p.GS = static_cast<const float*>(GS);
  p.C = static_cast<__nv_bfloat16*>(C);
  p.ldc = ldc;
  p.N = N;
  p.K = K;
  p.n_tiles = N / 128;
  p.k_tiles = K / 128;
  // Marlin's stripe partition with parallel = 1 and grid = sms (as marlinx_gemm_impl).
  int part2 = p.n_tiles;
  if (p.n_tiles > sms) {
    part2 = p.n_tiles % sms;
    if (part2 * 3 <= sms) part2 += sms;
  }
  p.part1_cols = p.n_tiles - part2;
  p.iters = (p.k_tiles * part2 + sms - 1) / sms;
  p.BK = static_cast<const uint32_t*>(BK);
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  switch (depth) {
    case 4: return marlinx::launch_kskip<4>(p, st);
    case 8: return marlinx::launch_kskip<8>(p, st);
    default: return -5;
  }
}

#endif
// Streams nbytes from src with blocks x threads, reserving smem bytes of
// shared memory per CTA. Returns 0 or a cudaError_t value.
int marlinx_hog(int blocks, int threads, int smem, const void* src, long long nbytes, void* sink,
                void* stream) {
  if (smem > 48 * 1024) {
    cudaError_t e = cudaFuncSetAttribute(marlinx::hog_kernel<8>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e != cudaSuccess) return (int)e;
  }
  marlinx::hog_kernel<8><<<blocks, threads, smem, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint4*>(src), nbytes / 16, static_cast<unsigned*>(sink));
  return (int)cudaGetLastError();
}

}  // extern "C"

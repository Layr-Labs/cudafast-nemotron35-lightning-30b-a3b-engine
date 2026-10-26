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

#ifndef MX_LEAN_PW
#define MX_LEAN_PW 8  // MARLINX_LEAN weight ring depth (k16 blocks per warp)
#endif

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
  const uint32_t* Z;       // zero-row skip: k-row-major copy of B, [K/16][16][N/64][8] u32 (or null)
  int part1_cols;          // columns reduced by a single Marlin block
  int iters;               // Marlin stripe length (k-tiles per block) in part 2
  const uint2* SU;         // optional unit-major copy of S, [N/64][K/16][8] uint2 (or null)
};

__device__ __forceinline__ uint4 ld_stream_v4(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

__device__ __forceinline__ uint32_t ld_stream_u32(const uint32_t* p) {
  uint32_t r;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];\n" : "=r"(r) : "l"(p));
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

#ifndef MARLINX_NO_PDL
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

// Activations can be the preceding grid's output. PDL uses coherent loads
// after its memory-clobbering wait, including the activation-derived ZS mask.
template <bool PDL>
__device__ __forceinline__ uint32_t ld_activation_u32(const uint32_t* p) {
  if constexpr (!PDL) return ld_nc_u32(p);
  uint32_t r;
  asm volatile("ld.global.u32 %0, [%1];" : "=r"(r) : "l"(p) : "memory");
  return r;
}
template <bool PDL>
__device__ __forceinline__ uint4 ld_activation_v4(const uint4* p) {
  if constexpr (!PDL) return __ldg(p);
  uint4 r;
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p) : "memory");
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
// Zero-row skip: rebuilds Marlin's weight word j of this lane from the four
// k-row words it read (k = 2t, 2t+1, 2t+8, 2t+9 of the k16 block; byte j of a
// word holds feature 16j+g in its low nibble and 16j+g+8 in its high nibble).
// A row that was not loaded reads as zero nibbles.
__device__ __forceinline__ uint32_t zs_word(const uint4& r, int j) {
  const uint32_t x = (r.x >> (8 * j)) & 255u, y = (r.y >> (8 * j)) & 255u;
  const uint32_t z = (r.z >> (8 * j)) & 255u, w = (r.w >> (8 * j)) & 255u;
  return (x & 15u) | ((z & 15u) << 4) | ((x >> 4) << 8) | ((z >> 4) << 12) |
         ((y & 15u) << 16) | ((w & 15u) << 20) | ((y >> 4) << 24) | ((w >> 4) << 28);
}

template <bool ZS = false>
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
  uint32_t q[4] = {bq.x, bq.y, bq.z, bq.w};
  if constexpr (ZS) {
#pragma unroll
    for (int j = 0; j < 4; ++j) q[j] = zs_word(bq, j);
  }
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

__device__ __forceinline__ __nv_bfloat162 round_output2_relu2(float a, float b) {
  union {
    __nv_bfloat162 h;
    uint32_t u;
  } cvt;
  cvt.h = __floats2bfloat162_rn(a, b);
  uint32_t out;
  asm("{ .reg .b32 x, z;\n"
      "mov.b32 z, 0; max.NaN.bf16x2 x, %1, z;\n"
      "mul.rn.bf16x2 %0, x, x; }\n"
      : "=r"(out) : "r"(cvt.u));
  cvt.u = out;
  return cvt.h;
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
    __nv_bfloat16 v0, v1, v2, v3;
    if constexpr (Relu2) {
      const __nv_bfloat162 v01 = round_output2_relu2(R[j][0] * g, R[j][1] * g);
      const __nv_bfloat162 v23 = round_output2_relu2(R[j][2] * g, R[j][3] * g);
      v0 = __low2bfloat16(v01);
      v1 = __high2bfloat16(v01);
      v2 = __low2bfloat16(v23);
      v3 = __high2bfloat16(v23);
    } else {
      v0 = round_output<false>(R[j][0] * g);
      v1 = round_output<false>(R[j][1] * g);
      v2 = round_output<false>(R[j][2] * g);
      v3 = round_output<false>(R[j][3] * g);
    }
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

// One 64-feature half column (`unit`) per CTA of 4 warps; warp w runs Marlin's
// warp-row chain w through a register ring of P k16 blocks.
template <int P, bool Relu2 = false, bool ZS = false, bool PDL = false>
__global__ void __launch_bounds__(128) gemm_kernel(const Params p) {
  __shared__ float red[2][3][16][32];
  __shared__ uint16_t zmask[ZS ? 512 : 1];
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
  // Unit-major scales (MARLINX_SU): the same scale bytes, stored per 64-feature half
  // column contiguously ([N/64][K/16][8] uint2, py/__init__.py su_repack), so the CTA
  // stages its whole column of scales in shared memory with a few coalesced 2 KB reads
  // instead of one 64-byte read per k16 block (row stride N bytes). Scales are weights,
  // immutable at run time, so a PDL grid may stage them before griddepcontrol.wait.
  extern __shared__ __align__(16) uint2 su_smem[];
  if (p.SU) {
    const uint4* src = reinterpret_cast<const uint4*>(p.SU + (long long)unit * (p.K >> 4) * 8);
    uint4* dst = reinterpret_cast<uint4*>(su_smem);
    for (int i = threadIdx.x; i < (p.K >> 4) * 4; i += 128) dst[i] = ld_stream_v4(src + i);
    __syncthreads();
  }
  auto load_s = [&](int kb) -> uint2 {
    return p.SU ? su_smem[kb * 8 + (lane >> 2)] : ld_nc_v2(Sp + kb * nS);
  };
  const bool arow = tok < p.M;
  const uint32_t* Ap =
      reinterpret_cast<const uint32_t*>(p.A + (arow ? (long long)tok * p.lda : 0)) + (lane & 3);

  uint4 rb[P];
  uint2 rs[P];
  uint32_t ra[P][2];

  Cursor ld, cs;
  cursor_init(ld, first_cut, p.k_tiles, p.iters);
  cs = ld;
  const int total = 2 * p.k_tiles;
#ifndef MARLINX_NO_PDL
  if constexpr (PDL) {
    // B/S are immutable checkpoint tensors. Z is also a weight-only repack,
    // but which Z rows are read depends on A: defer ALL Z/mask reads to wait.
    // Preload only scales for ZS, weights and scales for dense/fused-up/head.
    Cursor pre = ld;
#pragma unroll
    for (int s = 0; s < P; ++s) {
      if (s < total) {
        const int kb = 8 * pre.tau + 2 * w + pre.k;
        if constexpr (!ZS) rb[s] = ld_stream_v4(Bp + kb * nB);
        rs[s] = load_s(kb);
        cursor_next(pre, p.iters);
      }
    }
    pdl_wait();
  }
#endif

  // Zero-row skip: bit k of zmask[kb] is set when activation column 16*kb + k
  // is nonzero in some row m < M; only those weight rows are read.
  const int t4 = lane & 3;
  const long long zstride = (long long)(p.N >> 6) * 8;  // u32 per k row
  const uint32_t* Zp = ZS ? p.Z + (long long)(2 * col + h) * 8 + (lane >> 2) : nullptr;
  if constexpr (ZS) {
    for (int kb = threadIdx.x; kb < (p.K >> 4); kb += 128) {
      uint32_t m = 0;
      for (int r = 0; r < p.M; ++r) {
        const uint4* ar = reinterpret_cast<const uint4*>(p.A + (long long)r * p.lda + 16 * kb);
        const uint4 v0 = ld_activation_v4<PDL>(ar), v1 = ld_activation_v4<PDL>(ar + 1);
        const uint32_t v[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i)
          m |= ((v[i] & 0x7fffu) ? 1u : 0u) << (2 * i) | ((v[i] & 0x7fff0000u) ? 1u : 0u) << (2 * i + 1);
      }
      zmask[kb] = (uint16_t)m;
    }
    __syncthreads();
  }
  auto load_b = [&](int kb) -> uint4 {
    if constexpr (ZS) {
      const uint32_t m = zmask[kb];
      const uint32_t* z = Zp + (long long)kb * 16 * zstride;
      uint4 r;
      r.x = (m >> (2 * t4)) & 1 ? ld_stream_u32(z + (2 * t4) * zstride) : 0u;
      r.y = (m >> (2 * t4 + 1)) & 1 ? ld_stream_u32(z + (2 * t4 + 1) * zstride) : 0u;
      r.z = (m >> (2 * t4 + 8)) & 1 ? ld_stream_u32(z + (2 * t4 + 8) * zstride) : 0u;
      r.w = (m >> (2 * t4 + 9)) & 1 ? ld_stream_u32(z + (2 * t4 + 9) * zstride) : 0u;
      return r;
    } else {
      return ld_stream_v4(Bp + kb * nB);
    }
  };

#pragma unroll
  for (int s = 0; s < P; ++s) {
    if (s < total) {
      const int kb = 8 * ld.tau + 2 * w + ld.k;
      if constexpr (!PDL || ZS) rb[s] = load_b(kb);
      if constexpr (!PDL) rs[s] = load_s(kb);
      ra[s][0] = arow ? ld_activation_u32<PDL>(Ap + 8 * kb) : 0u;
      ra[s][1] = arow ? ld_activation_u32<PDL>(Ap + 8 * kb + 4) : 0u;
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
        block_mma<ZS>(rb[s], rs[s], ra[s], acc);
        const bool seg_end = cursor_next(cs, p.iters);
        if (i + P < total) {
          const int kb = 8 * ld.tau + 2 * w + ld.k;
          rb[s] = load_b(kb);
          rs[s] = load_s(kb);
          ra[s][0] = arow ? ld_activation_u32<PDL>(Ap + 8 * kb) : 0u;
          ra[s][1] = arow ? ld_activation_u32<PDL>(Ap + 8 * kb + 4) : 0u;
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
#ifndef MARLINX_NO_PDL
  // Every thread has waited, and every CTA triggers. Only this grid's C
  // epilogue remains; a dependent must wait before reading any of C.
  if constexpr (PDL) pdl_launch_dependents();
#endif
  if (!w) write_out<Relu2>(p, col, h, lane, R);
}

// ---------------------------------------------------------------------------
// MARLINX_LEAN: the shared expert projections (up 2688->3712, down 3712->2688) with
// gemm_kernel's CTA, grid, warp chains, segment folds, MMA operands and epilogue,
// and a shorter instruction stream per k16 block:
//   * shapes are template constants (strides and the chain length fold into
//     immediates; 32-bit offsets from per-warp base pointers);
//   * activation loads after griddepcontrol.wait are plain coherent loads without a
//     compiler memory clobber (the wait itself orders them);
//   * a segment can only close after a k = 1 block, so only odd ring slots carry
//     the fold code;
//   * zero-row skip (down): no CTA-wide mask pass and no barrier before the first
//     weight load. A lane needs k rows 2t, 2t+1, 2t+8, 2t+9 of each k16 block, and
//     its own activation words of that block hold exactly those columns of its
//     token row, so OR-ing the 4 nonzero bits over the 8 token lanes (3 xor
//     shuffles) gives gemm_kernel's mask bits for those rows. Activations run PW
//     blocks ahead of the weights in their own ring.
//   * zero-row skip copy in "Z2" nibble order (py z2_repack: the same u32 per
//     (k16 block, k row, half column, lane group) as zs_repack, nibbles permuted
//     within the word) so Marlin's operand halves come from 4 byte permutes and
//     shifts instead of rebuilding Marlin's words nibble by nibble. A skipped row
//     keeps whatever finite weights its register held (fp4 x e4m3 is always
//     finite); they multiply activations that are +-0 in every row < M, so every
//     product is a zero and every fp32 sum is unchanged (rows >= M are discarded).

// Marlin's dequant<bf16, fe2m1f, skip_flop> output half for the nibbles at bits 12-15
// and 28-31 of v (dequant_fp4's o1).
__device__ __forceinline__ uint32_t fp4_top(uint32_t v) {
  return (v & 0x80008000u) | ((v & 0x70007000u) >> 6);
}

template <bool PDL>
__device__ __forceinline__ uint32_t ld_act_lean(const uint32_t* p) {
  if constexpr (!PDL) return ld_nc_u32(p);
  uint32_t r;
  asm volatile("ld.global.u32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}

__device__ __forceinline__ void scale_pairs(const uint2& sq, uint32_t (&fs)[4]) {
  uint32_t t[2];
  dequant_scales(sq.x, t);
  fs[0] = t[0];
  fs[1] = t[1];
  dequant_scales(sq.y, t);
  fs[2] = t[0];
  fs[3] = t[1];
}

__device__ __forceinline__ void mma_scaled(uint32_t (&b0)[2], uint32_t (&b1)[2], uint32_t fsj,
                                           const uint32_t (&a)[2], float (&acc)[4]) {
  const __nv_bfloat162 s = as_bf162(fsj);
  const __nv_bfloat162 sx = __bfloat162bfloat162(s.x);
  const __nv_bfloat162 sy = __bfloat162bfloat162(s.y);
  b0[0] = as_u32(__hmul2(as_bf162(b0[0]), sx));
  b0[1] = as_u32(__hmul2(as_bf162(b0[1]), sx));
  b1[0] = as_u32(__hmul2(as_bf162(b1[0]), sy));
  b1[1] = as_u32(__hmul2(as_bf162(b1[1]), sy));
  mma_trans(a, b0, b1, acc);
}

// One k16 block from Z2 row words r = {row 2t, 2t+1, 2t+8, 2t+9}: the same b0/b1
// operands block_mma<ZS> builds from Marlin's word j (b?[0] use only its even
// nibbles = rows 2t/2t+1, b?[1] only its odd nibbles = rows 2t+8/2t+9).
__device__ __forceinline__ void block_mma_z2(const uint4& r, const uint2& sq, const uint32_t (&a)[2],
                                             float (&acc)[4][4]) {
  uint32_t fs[4];
  scale_pairs(sq, fs);
  const uint32_t pa[2] = {__byte_perm(r.x, r.y, 0x5410), __byte_perm(r.x, r.y, 0x7632)};
  const uint32_t pb[2] = {__byte_perm(r.z, r.w, 0x5410), __byte_perm(r.z, r.w, 0x7632)};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t qa = pa[j >> 1], qb = pb[j >> 1];
    uint32_t b0[2], b1[2];
    if (j & 1) {
      b0[0] = fp4_top(qa << 8);
      b1[0] = fp4_top(qa);
      b0[1] = fp4_top(qb << 12);
      b1[1] = fp4_top(qb << 4);
    } else {
      b0[0] = fp4_top(qa << 12);
      b1[0] = fp4_top(qa << 4);
      b0[1] = fp4_top(qb << 8);
      b1[1] = fp4_top(qb);
    }
    mma_scaled(b0, b1, fs[j], a, acc[j]);
  }
}

template <int N, int K, bool Relu2, bool ZS, bool PDL, int PW>
__global__ void __launch_bounds__(128) lean_kernel(const Params p) {
  constexpr int PA = ZS ? 2 * PW : PW;  // activation ring
  constexpr int KT = K / 128;
  constexpr int TOTAL = 2 * KT;         // k16 blocks per warp chain
  constexpr int NB = N / 2;             // uint4 per k16 row of B
  constexpr int NS = N / 8;             // uint2 per k16 row of S
  constexpr int ZST = N / 8;            // u32 per k row of Z
  __shared__ float red[2][3][16][32];
  const int unit = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int w = threadIdx.x >> 5;
  const int col = unit >> 1;
  const int h = unit & 1;
  const int tok = lane >> 2;
  const int t4 = lane & 3;
  const int first_cut = column_first_cut(p, col);
  // per-warp bases; block (tau, k) is at k16 row 8 tau + 2 w + k
  const uint4* Bw = p.B + (64 * col + 32 * h + lane + 2 * w * NB);
  const uint2* Sw = p.S + (16 * col + 8 * h + tok + 2 * w * NS);
  const uint32_t* Zw = nullptr;
  if constexpr (ZS) Zw = p.Z + (unit * 8 + tok + (32 * w + 2 * t4) * ZST);
  const bool arow = tok < p.M;
  const uint32_t* Aw =
      reinterpret_cast<const uint32_t*>(p.A + (arow ? (long long)tok * p.lda : 0)) + (t4 + 16 * w);

  uint4 rb[PW];
  uint2 rs[PW];
  uint32_t ra[PA][2];
#pragma unroll
  for (int s = 0; s < PW; ++s) rb[s] = make_uint4(0u, 0u, 0u, 0u);

  Cursor lw, la, cs;
  cursor_init(lw, first_cut, KT, p.iters);
  la = lw;
  cs = lw;
  auto load_a = [&](int s) {
    const int o = 8 * (8 * la.tau + la.k);
    ra[s][0] = arow ? ld_act_lean<PDL>(Aw + o) : 0u;
    ra[s][1] = arow ? ld_act_lean<PDL>(Aw + o + 4) : 0u;
    cursor_next(la, p.iters);
  };
  // weights (and scales) of the block at lw into slot ws; ZS: rows from act slot as
  auto load_w = [&](int ws, int as, bool scales) {
    const int o = 8 * lw.tau + lw.k;
    if constexpr (ZS) {
      const uint32_t a0 = ra[as][0], a1 = ra[as][1];
      uint32_t nz = ((a0 & 0x7fffu) ? 1u : 0u) | ((a0 & 0x7fff0000u) ? 2u : 0u) |
                    ((a1 & 0x7fffu) ? 4u : 0u) | ((a1 & 0x7fff0000u) ? 8u : 0u);
      nz |= __shfl_xor_sync(0xffffffffu, nz, 4);
      nz |= __shfl_xor_sync(0xffffffffu, nz, 8);
      nz |= __shfl_xor_sync(0xffffffffu, nz, 16);
      const uint32_t* z = Zw + o * (16 * ZST);
      if (nz & 1u) rb[ws].x = ld_stream_u32(z);
      if (nz & 2u) rb[ws].y = ld_stream_u32(z + ZST);
      if (nz & 4u) rb[ws].z = ld_stream_u32(z + 8 * ZST);
      if (nz & 8u) rb[ws].w = ld_stream_u32(z + 9 * ZST);
    } else {
      rb[ws] = ld_stream_v4(Bw + o * NB);
    }
    if (scales) rs[ws] = ld_nc_v2(Sw + o * NS);
    cursor_next(lw, p.iters);
  };

#ifndef MARLINX_NO_PDL
  if constexpr (PDL) {
    // Weights and scales are immutable: preload before the wait (ZS: scales only,
    // its weight rows depend on A).
    Cursor pre = lw;
#pragma unroll
    for (int s = 0; s < PW; ++s) {
      const int o = 8 * pre.tau + pre.k;
      if constexpr (!ZS) rb[s] = ld_stream_v4(Bw + o * NB);
      rs[s] = ld_nc_v2(Sw + o * NS);
      cursor_next(pre, p.iters);
    }
    pdl_wait();
  }
#endif
#pragma unroll
  for (int s = 0; s < PA; ++s) load_a(s);
  if constexpr (ZS || !PDL) {
#pragma unroll
    for (int s = 0; s < PW; ++s) {
      if constexpr (ZS) load_w(s, s, !PDL);
      else load_w(s, s, true);
    }
  } else {
#pragma unroll
    for (int s = 0; s < PW; ++s) cursor_next(lw, p.iters);
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

#pragma unroll 1
  for (int base = 0; base < TOTAL; base += PA) {
#pragma unroll
    for (int s = 0; s < PA; ++s) {
      const int i = base + s;
      if (TOTAL % PA == 0 || i < TOTAL) {
        const int ws = s % PW;
        if constexpr (ZS) block_mma_z2(rb[ws], rs[ws], ra[s], acc);
        else block_mma<false>(rb[ws], rs[ws], ra[s], acc);
        const bool seg_end = cursor_next(cs, p.iters);
        if constexpr (ZS) {
          if (i + PW < TOTAL) load_w(ws, (s + PW) % PA, true);
          if (i + PA < TOTAL) load_a(s);
        } else {
          if (i + PW < TOTAL) {
            load_a(s);
            load_w(ws, s, true);
          }
        }
        if ((s & 1) && seg_end) {  // segments end after a k = 1 block (odd i; base is even)
          close_segment(red, par, w, lane, acc, R, first_seg);
          first_seg = false;
          par ^= 1;
        }
      }
    }
  }
#ifndef MARLINX_NO_PDL
  if constexpr (PDL) pdl_launch_dependents();
#endif
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

template <int P, bool Relu2 = false, bool ZS = false>
static int launch(const Params& p, cudaStream_t stream, bool pdl) {
#ifndef MARLINX_NO_PDL
  if (pdl) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(2 * p.n_tiles);
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = p.SU ? (size_t)(p.K >> 4) * 64 : 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaError_t rc = cudaLaunchKernelEx(&cfg, gemm_kernel<P, Relu2, ZS, true>, p);
    if (rc != cudaSuccess) { cudaGetLastError(); return (int)rc; }
    return (int)cudaGetLastError();
  }
#endif
  gemm_kernel<P, Relu2, ZS><<<2 * p.n_tiles, 128, p.SU ? (size_t)(p.K >> 4) * 64 : 0, stream>>>(p);
  return (int)cudaGetLastError();
}

template <int N, int K, bool Relu2, bool ZS, int PW = 8>
static int launch_lean(const Params& p, cudaStream_t stream, bool pdl) {
#ifndef MARLINX_NO_PDL
  if (pdl) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(2 * p.n_tiles);
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaError_t rc = cudaLaunchKernelEx(&cfg, lean_kernel<N, K, Relu2, ZS, true, PW>, p);
    if (rc != cudaSuccess) { cudaGetLastError(); return (int)rc; }
    return (int)cudaGetLastError();
  }
#endif
  lean_kernel<N, K, Relu2, ZS, false, PW><<<2 * p.n_tiles, 128, 0, stream>>>(p);
  return (int)cudaGetLastError();
}

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
  if ((e = cudaFuncGetAttributes(&attr, marlinx::gemm_kernel<8, false, true>)) != cudaSuccess) return (int)e;
  return 0;
}

#ifndef MARLINX_NO_PDL
// Optional ABI; initialize every PDL specialization before graph capture.
int marlinx_init_pdl(void) {
  cudaFuncAttributes attr;
  cudaError_t e;
  if ((e = cudaFuncGetAttributes(&attr, marlinx::gemm_kernel<4, false, false, true>)) != cudaSuccess) return (int)e;
  if ((e = cudaFuncGetAttributes(&attr, marlinx::gemm_kernel<8, false, false, true>)) != cudaSuccess) return (int)e;
  if ((e = cudaFuncGetAttributes(&attr, marlinx::gemm_kernel<4, true, false, true>)) != cudaSuccess) return (int)e;
  if ((e = cudaFuncGetAttributes(&attr, marlinx::gemm_kernel<8, true, false, true>)) != cudaSuccess) return (int)e;
  if ((e = cudaFuncGetAttributes(&attr, marlinx::gemm_kernel<4, false, true, true>)) != cudaSuccess) return (int)e;
  if ((e = cudaFuncGetAttributes(&attr, marlinx::gemm_kernel<8, false, true, true>)) != cudaSuccess) return (int)e;
  return 0;
}
#endif

// MARLINX_LEAN kernels (shared expert shapes only): every specialization, for loading
// before graph capture. Returns 0 or a cudaError_t value.
int marlinx_init_lean(void) {
  cudaFuncAttributes attr;
  cudaError_t e;
#define MX_LEAN_INIT(...)                                                              \
  if ((e = cudaFuncGetAttributes(&attr, marlinx::lean_kernel<__VA_ARGS__>)) != cudaSuccess) \
    return (int)e;
  MX_LEAN_INIT(3712, 2688, true, false, false, MX_LEAN_PW)
  MX_LEAN_INIT(2688, 3712, false, false, false, MX_LEAN_PW)
  MX_LEAN_INIT(2688, 3712, false, true, false, MX_LEAN_PW)
#ifndef MARLINX_NO_PDL
  MX_LEAN_INIT(3712, 2688, true, false, true, MX_LEAN_PW)
  MX_LEAN_INIT(2688, 3712, false, false, true, MX_LEAN_PW)
  MX_LEAN_INIT(2688, 3712, false, true, true, MX_LEAN_PW)
#endif
#undef MX_LEAN_INIT
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
                 void* stream, bool relu2, const void* Z = nullptr, bool pdl = false,
                 const void* SU = nullptr) {
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
  p.Z = static_cast<const uint32_t*>(Z);
  p.SU = static_cast<const uint2*>(SU);
  if (SU && ((K >> 4) * 64 > 48 * 1024 || (reinterpret_cast<uintptr_t>(SU) & 15))) return -9;
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
      case 4: return marlinx::launch<4, true>(p, st, pdl);
      case 8: return marlinx::launch<8, true>(p, st, pdl);
      default: return -5;
    }
  }
  if (Z) {
    if (N != 2688 || K != 3712 || (K >> 4) > 512) return -7;
    switch (depth) {
      case 4: return marlinx::launch<4, false, true>(p, st, pdl);
      case 8: return marlinx::launch<8, false, true>(p, st, pdl);
      default: return -5;
    }
  }
  switch (depth) {
    case 4: return marlinx::launch<4>(p, st, pdl);
    case 8: return marlinx::launch<8>(p, st, pdl);
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

// Shared expert down with zero-row skip: Z is the k-row-major copy of B made by
// py/__init__.py zs_repack(); A's zero columns (in every row < M) skip their
// weight rows. Same arithmetic as marlinx_gemm.
int marlinx_gemm_zs(const void* A, long long lda, int M, const void* B, const void* S,
                    const void* GS, void* C, long long ldc, int N, int K, int sms, int depth,
                    void* stream, const void* Z) {
  if (!Z) return -8;
  return marlinx_gemm_impl(A, lda, M, B, S, GS, C, ldc, N, K, sms, depth, stream, false, Z);
}

#ifndef MARLINX_NO_PDL
int marlinx_gemm_pdl(const void* A, long long lda, int M, const void* B, const void* S,
                 const void* GS, void* C, long long ldc, int N, int K, int sms, int depth,
                 void* stream) {
  return marlinx_gemm_impl(A, lda, M, B, S, GS, C, ldc, N, K, sms, depth, stream, false, nullptr, true);
}
int marlinx_gemm_relu2_pdl(const void* A, long long lda, int M, const void* B, const void* S,
                 const void* GS, void* C, long long ldc, int N, int K, int sms, int depth,
                 void* stream) {
  return marlinx_gemm_impl(A, lda, M, B, S, GS, C, ldc, N, K, sms, depth, stream, true, nullptr, true);
}
int marlinx_gemm_zs_pdl(const void* A, long long lda, int M, const void* B, const void* S,
                 const void* GS, void* C, long long ldc, int N, int K, int sms, int depth,
                 void* stream, const void* Z) {
  if (!Z) return -8;
  return marlinx_gemm_impl(A, lda, M, B, S, GS, C, ldc, N, K, sms, depth, stream, false, Z, true);
}
#endif

// MARLINX_LEAN: shared up (relu2 = 1: N 3712, K 2688) or shared down (N 2688,
// K 3712; Z2 = py z2_repack(B) for the zero-row skip, or null for dense). Same
// arithmetic and output bits as marlinx_gemm_relu2 / marlinx_gemm_zs / marlinx_gemm.
int marlinx_gemm_lean(const void* A, long long lda, int M, const void* B, const void* S,
                      const void* GS, void* C, long long ldc, int N, int K, int sms,
                      void* stream, const void* Z2, int relu2, int pdl) {
#ifdef MARLINX_NO_PDL
  if (pdl) return -10;
#endif
  if (M < 1 || M > 8) return -1;
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
  p.Z = static_cast<const uint32_t*>(Z2);
  p.SU = nullptr;
  int part2 = p.n_tiles;
  if (p.n_tiles > sms) {
    part2 = p.n_tiles % sms;
    if (part2 * 3 <= sms) part2 += sms;
  }
  p.part1_cols = p.n_tiles - part2;
  p.iters = (p.k_tiles * part2 + sms - 1) / sms;
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  if (relu2) {
    if (N != 3712 || K != 2688 || Z2) return -6;
    return marlinx::launch_lean<3712, 2688, true, false, MX_LEAN_PW>(p, st, pdl != 0);
  }
  if (N != 2688 || K != 3712) return -7;
  if (Z2) return marlinx::launch_lean<2688, 3712, false, true, MX_LEAN_PW>(p, st, pdl != 0);
  return marlinx::launch_lean<2688, 3712, false, false, MX_LEAN_PW>(p, st, pdl != 0);
}

// Unit-major scale variant (any mode): SU = py su_repack(S). Same arithmetic.
int marlinx_gemm_su(const void* A, long long lda, int M, const void* B, const void* S,
                    const void* GS, void* C, long long ldc, int N, int K, int sms, int depth,
                    void* stream, const void* Z, const void* SU, int relu2, int pdl) {
  if (!SU) return -8;
#ifdef MARLINX_NO_PDL
  if (pdl) return -10;
#endif
  return marlinx_gemm_impl(A, lda, M, B, S, GS, C, ldc, N, K, sms, depth, stream, relu2 != 0, Z,
                           pdl != 0, SU);
}

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

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
  const uint32_t* Z;       // highest carrier's byte-preserving k-row weight copy
  uint16_t* mask_out;       // optional fused shared-up metadata
  const uint16_t* mask_in;  // optional same-call shared-down metadata
  int part1_cols;          // columns reduced by a single Marlin block
  int iters;               // Marlin stripe length (k-tiles per block) in part 2
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

// Marlin write_result: bf16_rn(R * global_scale), rows < M only (warp 0).
template <bool Relu2, bool MaskOut = false>
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
    if constexpr (MaskOut) {
      // Each lane owns two tokens of feature g and g+8; OR all eight
      // token columns of this warp's FINAL BF16 ReLU2 values. No atomics.
      const bool lo = (t0 < p.M && (__bfloat16_as_ushort(v0) & 0x7fffu)) ||
                      (t0 + 1 < p.M && (__bfloat16_as_ushort(v1) & 0x7fffu));
      const bool hi = (t0 < p.M && (__bfloat16_as_ushort(v2) & 0x7fffu)) ||
                      (t0 + 1 < p.M && (__bfloat16_as_ushort(v3) & 0x7fffu));
      const unsigned bits = (lo ? 1u << (lane >> 2) : 0u) |
                            (hi ? 1u << ((lane >> 2) + 8) : 0u);
      const unsigned mask = __reduce_or_sync(0xffffffffu, bits);
      if (lane == 0) p.mask_out[4 * (2 * col + h) + j] = (uint16_t)mask;
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
template <int P, bool Relu2 = false, bool ZS = false, bool MaskOut = false, bool MaskIn = false>
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
  const bool arow = tok < p.M;
  const uint32_t* Ap =
      reinterpret_cast<const uint32_t*>(p.A + (arow ? (long long)tok * p.lda : 0)) + (lane & 3);

  uint4 rb[P];
  uint2 rs[P];
  uint32_t ra[P][2];

  // Zero-row skip: bit k of zmask[kb] is set when activation column 16*kb + k
  // is nonzero in some row m < M; only those weight rows are read.
  const int t4 = lane & 3;
  const long long zstride = (long long)(p.N >> 6) * 8;  // u32 per k row
  const uint32_t* Zp = ZS ? p.Z + (long long)(2 * col + h) * 8 + (lane >> 2) : nullptr;
  if constexpr (ZS) {
    for (int kb = threadIdx.x; kb < (p.K >> 4); kb += 128) {
      if constexpr (MaskIn) {
        zmask[kb] = __ldg(p.mask_in + kb);
      } else {
        uint32_t m = 0;
        for (int r = 0; r < p.M; ++r) {
          const uint4* ar = reinterpret_cast<const uint4*>(p.A + (long long)r * p.lda + 16 * kb);
          const uint4 v0 = __ldg(ar), v1 = __ldg(ar + 1);
          const uint32_t v[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
          for (int i = 0; i < 8; ++i)
            m |= ((v[i] & 0x7fffu) ? 1u : 0u) << (2 * i) | ((v[i] & 0x7fff0000u) ? 1u : 0u) << (2 * i + 1);
        }
        zmask[kb] = (uint16_t)m;
      }
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

  Cursor ld, cs;
  cursor_init(ld, first_cut, p.k_tiles, p.iters);
  cs = ld;
  const int total = 2 * p.k_tiles;

#pragma unroll
  for (int s = 0; s < P; ++s) {
    if (s < total) {
      const int kb = 8 * ld.tau + 2 * w + ld.k;
      rb[s] = load_b(kb);
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
        block_mma<ZS>(rb[s], rs[s], ra[s], acc);
        const bool seg_end = cursor_next(cs, p.iters);
        if (i + P < total) {
          const int kb = 8 * ld.tau + 2 * w + ld.k;
          rb[s] = load_b(kb);
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
  if (!w) write_out<Relu2, MaskOut>(p, col, h, lane, R);
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

template <int P, bool Relu2 = false, bool ZS = false, bool MaskOut = false, bool MaskIn = false>
static int launch(const Params& p, cudaStream_t stream) {
  gemm_kernel<P, Relu2, ZS, MaskOut, MaskIn><<<2 * p.n_tiles, 128, 0, stream>>>(p);
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

// Initialize additional mask producer/consumer instances outside capture.
int marlinx_mask_init(void) {
  cudaFuncAttributes attr;
  cudaError_t e;
  if ((e = cudaFuncGetAttributes(&attr, marlinx::gemm_kernel<4, true, false, true>)) != cudaSuccess) return (int)e;
  if ((e = cudaFuncGetAttributes(&attr, marlinx::gemm_kernel<8, true, false, true>)) != cudaSuccess) return (int)e;
  if ((e = cudaFuncGetAttributes(&attr, marlinx::gemm_kernel<4, false, true, false, true>)) != cudaSuccess) return (int)e;
  if ((e = cudaFuncGetAttributes(&attr, marlinx::gemm_kernel<8, false, true, false, true>)) != cudaSuccess) return (int)e;
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
                 void* stream, bool relu2, const void* Z = nullptr, void* mask_out = nullptr, const void* mask_in = nullptr) {
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
  p.mask_out = static_cast<uint16_t*>(mask_out);
  p.mask_in = static_cast<const uint16_t*>(mask_in);
  // Marlin's stripe partition with parallel = 1 and grid = sms.
  int part2 = p.n_tiles;
  if (p.n_tiles > sms) {
    part2 = p.n_tiles % sms;
    if (part2 * 3 <= sms) part2 += sms;
  }
  p.part1_cols = p.n_tiles - part2;
  p.iters = (p.k_tiles * part2 + sms - 1) / sms;
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  if (mask_out) {
    if (!relu2 || Z || mask_in || N != 3712 || K != 2688) return -9;
    switch (depth) {
      case 4: return marlinx::launch<4, true, false, true>(p, st);
      case 8: return marlinx::launch<8, true, false, true>(p, st);
      default: return -5;
    }
  }
  if (mask_in) {
    if (relu2 || !Z || N != 2688 || K != 3712) return -9;
    switch (depth) {
      case 4: return marlinx::launch<4, false, true, false, true>(p, st);
      case 8: return marlinx::launch<8, false, true, false, true>(p, st);
      default: return -5;
    }
  }
  if (relu2) {
    if (N != 3712 || K != 2688) return -6;
    switch (depth) {
      case 4: return marlinx::launch<4, true>(p, st);
      case 8: return marlinx::launch<8, true>(p, st);
      default: return -5;
    }
  }
  if (Z) {
    if (N != 2688 || K != 3712 || (K >> 4) > 512) return -7;
    switch (depth) {
      case 4: return marlinx::launch<4, false, true>(p, st);
      case 8: return marlinx::launch<8, false, true>(p, st);
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

int marlinx_gemm_relu2_mask(const void* A, long long lda, int M, const void* B, const void* S,
                 const void* GS, void* C, long long ldc, int N, int K, int sms, int depth,
                 void* stream, void* mask) {
  if (!mask) return -9;
  return marlinx_gemm_impl(A, lda, M, B, S, GS, C, ldc, N, K, sms, depth, stream, true, nullptr, mask);
}

int marlinx_gemm_zs_mask(const void* A, long long lda, int M, const void* B, const void* S,
                 const void* GS, void* C, long long ldc, int N, int K, int sms, int depth,
                 void* stream, const void* Z, const void* mask) {
  if (!Z || !mask) return -9;
  return marlinx_gemm_impl(A, lda, M, B, S, GS, C, ldc, N, K, sms, depth, stream, false, Z, nullptr, mask);
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

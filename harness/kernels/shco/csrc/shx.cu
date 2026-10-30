// shx: a Marlin-exact NVFP4 W4A16 GEMM for the shared expert at prefill M (496 rows), with a
// footprint small enough to share each SM with three CTAs of humming's routed up projection
// (128 threads, <= 128 registers, ~21 KB shared memory, one CTA per SM).
//
// What stock Marlin computes there (vLLM v0.27.1 marlin_mm / marlin_template.h, thread config
// {thread_k 64, thread_n 128, 128 threads}, thread_m_blocks 4 for the first 448 rows in
// `parallel` 64-row sub-problems and 3 for the 48-row tail, group_blocks 1, fp32 global
// reduce, no atomics), per output element (row r, feature n):
//   * Marlin's stripe partition over grid = #SMs cuts the k-tiles (64 K each) of the column
//     (sub-problem of r, 128-feature n-tile of n) into segments; one CTA reduces a segment.
//   * Inside a segment, k-warp group g = 0, 1 runs one fp32 chain from +0 of
//     mma.m16n8k16.row.col.f32.bf16.bf16.f32 (activations as A, dequantized weights as B) over
//     the k16 blocks 4t + 2g, 4t + 2g + 1 for the segment's tiles t ascending.
//   * P = acc_g0 + acc_g1 (thread_block_reduce).
//   * global_reduce_fp32 in lock order: R = P of the highest-K segment, then R = P + R for each
//     segment going down in K.
//   * out = bf16_rn(R * global_scale) (write_result).
// The host computes the segments (py/__init__.py marlin_units, a transcription of Marlin's
// partition) into a unit table; each unit is 64 rows x 64 features with its segments in fold
// order. The weight dequant, scale multiply, operands and every fp32 add are Marlin's (the same
// code as harness/kernels/marlinx). Optional fused epilogue: bf16(relu(v)^2) of the bf16 output
// v, as Inductor's fused pow/relu computes it.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace shx {

constexpr int kStages = 4;
constexpr int kAStage = 64 * 64;  // 64 rows x 32 k bf16
constexpr int kBStage = 1024;     // 2 kb x 2 halves x 32 lanes x 8 B
constexpr int kSStage = 128;      // 2 kb x 64 scale bytes
constexpr int kStageBytes = kAStage + kBStage + kSStage;
constexpr int kUnitInts = 12;     // row0, nrows, f0, nseg, lo[4], hi[4] (k-tiles of 64)
constexpr int kScratchFloats = 32 * 128;

struct Params {
  const __nv_bfloat16* A;  // [M, lda]
  long long lda;
  int M;
  const uint4* B;      // Marlin fp4 weight, [K/16][N/2] int4
  const uint8_t* S;    // Marlin-processed e4m3 scales, [K/16][N] bytes
  const float* GS;     // Marlin-processed fp32 global scale
  __nv_bfloat16* C;    // [M, ldc]
  long long ldc;
  int N, K;
  const int* units;
  int n_units;
  float* scratch;      // [grid][32][128]
};

__device__ __forceinline__ __nv_bfloat162 as_bf162(uint32_t v) { return *reinterpret_cast<__nv_bfloat162*>(&v); }
__device__ __forceinline__ uint32_t as_u32(__nv_bfloat162 v) { return *reinterpret_cast<uint32_t*>(&v); }

// Marlin dequant<nv_bfloat162, kFE2M1f, dequant_skip_flop = true> (as marlinx).
__device__ __forceinline__ void dequant_fp4(uint32_t q, uint32_t (&out)[2]) {
  uint32_t o1 = (q & 0x80008000u) | ((q & 0x70007000u) >> 6);
  q <<= 4;
  uint32_t o2 = (q & 0x80008000u) | ((q & 0x70007000u) >> 6);
  out[1] = o1;
  out[0] = o2;
}

// Marlin dequant_fp8_scales<nv_bfloat162, kFE4M3fn> (as marlinx).
__device__ __forceinline__ void dequant_scales(uint32_t q, uint32_t (&out)[2]) {
  uint32_t o1 = ((q & 0x80008000u) >> 1) | ((q & 0x7F007F00u) >> 4);
  q <<= 8;
  uint32_t o2 = ((q & 0x80008000u) >> 1) | ((q & 0x7F007F00u) >> 4);
  out[1] = o1;
  out[0] = o2;
}

// Marlin mma<bf16>: activations as A, weights (one n8 tile) as B.
__device__ __forceinline__ void mma(const uint32_t (&a)[4], const uint32_t (&b)[2], float (&c)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]),
        "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void cp16(uint32_t dst, const void* src, bool pred) {
  const int n = pred ? 16 : 0;  // zero-fill rows past the unit
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}

__device__ __forceinline__ void cp8(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(dst), "l"(src));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

template <bool Relu2>
__device__ __forceinline__ __nv_bfloat16 round_output(float scaled) {
  const __nv_bfloat16 rounded = __float2bfloat16_rn(scaled);
  if constexpr (Relu2) {
    float value = __bfloat162float(rounded);
    value = value < 0.0f ? 0.0f : value;
    const float squared = __fmul_rn(value, value);
    return squared != squared ? __ushort_as_bfloat16(0x7fff) : __float2bfloat16_rn(squared);
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

// One stage (two k16 blocks of one chain) of one warp: 2 m16 x 4 n8 tiles.
__device__ __forceinline__ void stage_mma(uint32_t sA, uint32_t sB, uint32_t sS, int rw, int cw, int lane,
                                          float (&acc)[2][4][4]) {
#pragma unroll
  for (int kk = 0; kk < 2; ++kk) {
    uint32_t a[2][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
      const int r = 32 * rw + 16 * mi + (lane & 15);
      const int c = 2 * kk + (lane >> 4);
      ldsm4(a[mi], sA + r * 64 + ((c ^ ((r >> 1) & 3)) << 4));
    }
    uint32_t q0, q1, sq;
    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n"
                 : "=r"(q0), "=r"(q1)
                 : "r"(sB + ((kk * 2 + cw) * 32 + lane) * 8));
    asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(sq) : "r"(sS + kk * 64 + (lane >> 2) * 8 + cw * 4));
    uint32_t fs[2];
    dequant_scales(sq, fs);
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      const uint32_t q = jj ? q1 : q0;
      uint32_t b0[2], b1[2];
      dequant_fp4(q << 8, b0);  // b_quant_0 = b_quant_1 << 8
      dequant_fp4(q, b1);
      const __nv_bfloat162 s = as_bf162(fs[jj]);
      const __nv_bfloat162 sx = __bfloat162bfloat162(s.x);
      const __nv_bfloat162 sy = __bfloat162bfloat162(s.y);
      b0[0] = as_u32(__hmul2(as_bf162(b0[0]), sx));
      b0[1] = as_u32(__hmul2(as_bf162(b0[1]), sx));
      b1[0] = as_u32(__hmul2(as_bf162(b1[0]), sy));
      b1[1] = as_u32(__hmul2(as_bf162(b1[1]), sy));
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        mma(a[mi], b0, acc[mi][2 * jj]);
        mma(a[mi], b1, acc[mi][2 * jj + 1]);
      }
    }
  }
}

template <bool Relu2>
__global__ void __launch_bounds__(128, 4) gemm_kernel(const Params p) {
  __shared__ __align__(128) uint8_t smem[kStages * kStageBytes];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int rw = warp >> 1;  // 32-row half of the unit
  const int cw = warp & 1;   // 32-feature half of the unit
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  float* scr = p.scratch + static_cast<long long>(blockIdx.x) * kScratchFloats + tid;
  const float g = __ldg(p.GS);
  const long long nB = p.N >> 1;  // int4 per k16 row

  for (int u = blockIdx.x; u < p.n_units; u += gridDim.x) {
    const int* U = p.units + kUnitInts * u;
    const int row0 = U[0], nrows = U[1], f0 = U[2], nseg = U[3];
    int lo[4], cnt[4];
    int nst = 0;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      lo[s] = U[4 + s];
      cnt[s] = s < nseg ? 2 * (U[8 + s] - U[4 + s]) : 0;  // stages (2 kb each) per segment
      nst += cnt[s];
    }
    const int col = f0 >> 7, h = (f0 >> 6) & 1;
    const uint4* Bu = p.B + 64 * col + 32 * h;
    const uint8_t* Su = p.S + 128 * col + 64 * h;
    const __nv_bfloat16* Au = p.A + static_cast<long long>(row0) * p.lda;

    // k16 block of stage st (segments in fold order, tiles ascending, 2 kb per stage)
    auto kb_of = [&](int st) -> int {
      int kb = 0;
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        if (st >= 0 && st < cnt[s]) kb = 4 * lo[s] + 2 * st;
        st -= cnt[s];
      }
      return kb;
    };
    auto load = [&](int st) {
      const int kb = kb_of(st);
      const uint32_t slot = sbase + (st % kStages) * kStageBytes;
      // A: 64 rows x 4 chunks (32 k)
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int idx = tid + 128 * i;
        const int r = idx >> 2, c = idx & 3;
        const bool ok = r < nrows;
        const __nv_bfloat16* src = Au + static_cast<long long>(ok ? r : 0) * p.lda + 16 * kb + 8 * c;
        cp16(slot + r * 64 + ((c ^ ((r >> 1) & 3)) << 4), src, ok);
      }
      // B: [kk][half][lane] 8 B
      {
        const int kk = tid >> 6, half = (tid >> 5) & 1, l = tid & 31;
        const uint8_t* src = reinterpret_cast<const uint8_t*>(Bu + (kb + kk) * nB + l) + 8 * half;
        cp8(slot + kAStage + ((kk * 2 + half) * 32 + l) * 8, src);
      }
      // S: [kk][64 B]
      if (tid < 8) {
        const int kk = tid >> 2, c = tid & 3;
        cp16(slot + kAStage + kBStage + kk * 64 + 16 * c, Su + static_cast<long long>(kb + kk) * p.N + 16 * c, true);
      }
    };

    __syncthreads();  // the previous unit's last stage is consumed by every warp
#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
      if (s < nst) load(s);
      cp_commit();
    }

    float accA[2][4][4], accB[2][4][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int t = 0; t < 4; ++t)
#pragma unroll
        for (int e = 0; e < 4; ++e) accA[mi][t][e] = accB[mi][t][e] = 0.0f;

    int seg = 0, seg_end = cnt[0];
    for (int st = 0; st < nst; ++st) {
      cp_wait<kStages - 2>();
      __syncthreads();
      if (st + kStages - 1 < nst) load(st + kStages - 1);
      cp_commit();
      const uint32_t slot = sbase + (st % kStages) * kStageBytes;
      const int kb = kb_of(st);
      if ((kb >> 1) & 1)
        stage_mma(slot, slot + kAStage, slot + kAStage + kBStage, rw, cw, lane, accB);
      else
        stage_mma(slot, slot + kAStage, slot + kAStage + kBStage, rw, cw, lane, accA);
      if (st + 1 == seg_end) {
        // P = acc_g0 + acc_g1; R = P (highest K) then R = P + R
        const bool first = seg == 0, last = seg == nseg - 1;
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
          for (int t = 0; t < 4; ++t)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const int i = (mi * 4 + t) * 4 + e;
              float v = __fadd_rn(accA[mi][t][e], accB[mi][t][e]);
              if (!first) v = __fadd_rn(v, scr[i * 128]);
              if (!last) scr[i * 128] = v;
              accA[mi][t][e] = v;  // the last segment keeps R here for the epilogue
              accB[mi][t][e] = 0.0f;
            }
        if (!last) {
#pragma unroll
          for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int t = 0; t < 4; ++t)
#pragma unroll
              for (int e = 0; e < 4; ++e) accA[mi][t][e] = 0.0f;
        }
        ++seg;
        if (seg < 4) seg_end += cnt[seg];
      }
    }
    cp_wait<0>();

    // epilogue: bf16_rn(R * gs) (+ relu^2), rows < nrows
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
      const int r_lo = 32 * rw + 16 * mi + (lane >> 2);
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int f = f0 + 32 * cw + 8 * t + 2 * (lane & 3);
        // n8 tile t = 2*jj + s covers features 16*jj + 8*s + n of this warp's 32
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int r = r_lo + 8 * hh;
          if (r < nrows) {
            __nv_bfloat162 v;
            if constexpr (Relu2) {
              v = round_output2_relu2(__fmul_rn(accA[mi][t][2 * hh], g),
                                      __fmul_rn(accA[mi][t][2 * hh + 1], g));
            } else {
              v.x = round_output<false>(__fmul_rn(accA[mi][t][2 * hh], g));
              v.y = round_output<false>(__fmul_rn(accA[mi][t][2 * hh + 1], g));
            }
            *reinterpret_cast<__nv_bfloat162*>(p.C + static_cast<long long>(row0 + r) * p.ldc + f) = v;
          }
        }
      }
    }
  }
}


// ---------------------------------------------------------------------------------------- v2
// Units of 64 rows x 128 features (one Marlin column of one sub-problem): half the activation
// reads of v1. Each warp owns 32 rows x 64 features (16 MMAs per k16 block, 64 accumulators),
// so the two k-warp chains of a segment run one after the other (chain 0 over the segment's
// blocks 4t, 4t+1, then chain 1 over 4t+2, 4t+3) and chain 0 waits in thread-private scratch;
// P = chain0 + chain1 and the fold are v1's. A dynamic work counter balances the CTAs (reset by
// the last CTA, so graph replays need no memset). hint = 1 loads activations with L2 evict_last.

constexpr int k2AStage = 64 * 64;   // 64 rows x 32 k bf16
constexpr int k2BStage = 2048;      // 2 kb x 64 int4 (128 features)
constexpr int k2SStage = 256;       // 2 kb x 128 scale bytes
constexpr int k2StageBytes = k2AStage + k2BStage + k2SStage;
constexpr int k2Stages = 4;
constexpr int k2ScratchFloats = 2 * 64 * 128;  // stash + R, per CTA

struct Params2 {
  Params p;
  int* work;  // [2]: next unit, CTAs done
  int hint;
};

__device__ __forceinline__ void cp16_hint(uint32_t dst, const void* src, bool pred, uint64_t pol) {
  const int n = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n" ::"r"(dst), "l"(src), "r"(n),
               "l"(pol));
}

__device__ __forceinline__ void stage_mma2(uint32_t sA, uint32_t sB, uint32_t sS, int rw, int cw, int lane,
                                           float (&acc)[2][8][4]) {
#pragma unroll
  for (int kk = 0; kk < 2; ++kk) {
    uint32_t a[2][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
      const int r = 32 * rw + 16 * mi + (lane & 15);
      const int c = 2 * kk + (lane >> 4);
      ldsm4(a[mi], sA + r * 64 + ((c ^ ((r >> 1) & 3)) << 4));
    }
    uint32_t q[4], sx, sy;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(q[0]), "=r"(q[1]), "=r"(q[2]), "=r"(q[3])
                 : "r"(sB + (kk * 64 + 32 * cw + lane) * 16));
    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n"
                 : "=r"(sx), "=r"(sy)
                 : "r"(sS + kk * 128 + 64 * cw + 8 * (lane >> 2)));
    uint32_t fs[4];
    {
      uint32_t t[2];
      dequant_scales(sx, t);
      fs[0] = t[0];
      fs[1] = t[1];
      dequant_scales(sy, t);
      fs[2] = t[0];
      fs[3] = t[1];
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint32_t b0[2], b1[2];
      dequant_fp4(q[j] << 8, b0);
      dequant_fp4(q[j], b1);
      const __nv_bfloat162 s = as_bf162(fs[j]);
      const __nv_bfloat162 sxx = __bfloat162bfloat162(s.x);
      const __nv_bfloat162 syy = __bfloat162bfloat162(s.y);
      b0[0] = as_u32(__hmul2(as_bf162(b0[0]), sxx));
      b0[1] = as_u32(__hmul2(as_bf162(b0[1]), sxx));
      b1[0] = as_u32(__hmul2(as_bf162(b1[0]), syy));
      b1[1] = as_u32(__hmul2(as_bf162(b1[1]), syy));
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        mma(a[mi], b0, acc[mi][2 * j]);
        mma(a[mi], b1, acc[mi][2 * j + 1]);
      }
    }
  }
}

template <bool Relu2>
__global__ void __launch_bounds__(128, 4) gemm2_kernel(const Params2 q) {
  const Params& p = q.p;
  __shared__ __align__(128) uint8_t smem[k2Stages * k2StageBytes];
  __shared__ int s_unit;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int rw = warp >> 1;  // 32-row half
  const int cw = warp & 1;   // 64-feature half
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  float* stash = p.scratch + static_cast<long long>(blockIdx.x) * k2ScratchFloats + tid;
  float* rbuf = stash + 64 * 128;
  const float g = __ldg(p.GS);
  const long long nB = p.N >> 1;
  uint64_t pol = 0;
  if (q.hint) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));

  for (;;) {
    __syncthreads();  // the previous unit's stages are consumed; s_unit is free
    if (tid == 0) s_unit = atomicAdd(q.work, 1);
    __syncthreads();
    const int u = s_unit;
    if (u >= p.n_units) break;
    const int* U = p.units + kUnitInts * u;
    const int row0 = U[0], nrows = U[1], f0 = U[2], nseg = U[3];
    int lo[4], len[4];
    int nst = 0;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      lo[s] = U[4 + s];
      len[s] = s < nseg ? U[8 + s] - U[4 + s] : 0;  // k-tiles; 2 * len stages (chain 0, then 1)
      nst += 2 * len[s];
    }
    const int col = f0 >> 7;
    const uint4* Bu = p.B + 64 * col;
    const uint8_t* Su = p.S + 128 * col;
    const __nv_bfloat16* Au = p.A + static_cast<long long>(row0) * p.lda;

    auto kb_of = [&](int st) -> int {
      int kb = 0;
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int n2 = 2 * len[s];
        if (st >= 0 && st < n2) {
          const int gg = st >= len[s] ? 1 : 0;
          kb = 4 * (lo[s] + st - gg * len[s]) + 2 * gg;
        }
        st -= n2;
      }
      return kb;
    };
    auto load = [&](int st) {
      const int kb = kb_of(st);
      const uint32_t slot = sbase + (st % k2Stages) * k2StageBytes;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int idx = tid + 128 * i;
        const int r = idx >> 2, c = idx & 3;
        const bool ok = r < nrows;
        const __nv_bfloat16* src = Au + static_cast<long long>(ok ? r : 0) * p.lda + 16 * kb + 8 * c;
        const uint32_t dst = slot + r * 64 + ((c ^ ((r >> 1) & 3)) << 4);
        if (q.hint)
          cp16_hint(dst, src, ok, pol);
        else
          cp16(dst, src, ok);
      }
      {
        const int kk = tid >> 6, i = tid & 63;
        cp16(slot + k2AStage + (kk * 64 + i) * 16, Bu + (kb + kk) * nB + i, true);
      }
      if (tid < 16) {
        const int kk = tid >> 3, c = tid & 7;
        cp16(slot + k2AStage + k2BStage + kk * 128 + 16 * c, Su + static_cast<long long>(kb + kk) * p.N + 16 * c,
             true);
      }
    };

#pragma unroll
    for (int s = 0; s < k2Stages - 1; ++s) {
      if (s < nst) load(s);
      cp_commit();
    }
    float acc[2][8][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int t = 0; t < 8; ++t)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[mi][t][e] = 0.0f;

    int seg = 0, base = 0;
    for (int st = 0; st < nst; ++st) {
      cp_wait<k2Stages - 2>();
      __syncthreads();
      if (st + k2Stages - 1 < nst) load(st + k2Stages - 1);
      cp_commit();
      const uint32_t slot = sbase + (st % k2Stages) * k2StageBytes;
      stage_mma2(slot, slot + k2AStage, slot + k2AStage + k2BStage, rw, cw, lane, acc);
      const int i = st - base;  // stage index inside the segment
      if (i + 1 == len[seg]) {
        // chain 0 closed: park it
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
          for (int t = 0; t < 8; ++t)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              stash[((mi * 8 + t) * 4 + e) * 128] = acc[mi][t][e];
              acc[mi][t][e] = 0.0f;
            }
      } else if (i + 1 == 2 * len[seg]) {
        // chain 1 closed: P = acc_g0 + acc_g1; R = P (highest K) then R = P + R
        const bool first = seg == 0, last = seg == nseg - 1;
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
          for (int t = 0; t < 8; ++t)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const int x = ((mi * 8 + t) * 4 + e) * 128;
              float v = __fadd_rn(stash[x], acc[mi][t][e]);
              if (!first) v = __fadd_rn(v, rbuf[x]);
              if (!last) {
                rbuf[x] = v;
                v = 0.0f;
              }
              acc[mi][t][e] = v;  // the last segment keeps R for the epilogue
            }
        base += 2 * len[seg];
        ++seg;
      }
    }
    cp_wait<0>();
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
      const int r_lo = 32 * rw + 16 * mi + (lane >> 2);
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        const int f = f0 + 64 * cw + 8 * t + 2 * (lane & 3);
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int r = r_lo + 8 * hh;
          if (r < nrows) {
            __nv_bfloat162 v;
            if constexpr (Relu2) {
              v = round_output2_relu2(__fmul_rn(acc[mi][t][2 * hh], g),
                                      __fmul_rn(acc[mi][t][2 * hh + 1], g));
            } else {
              v.x = round_output<false>(__fmul_rn(acc[mi][t][2 * hh], g));
              v.y = round_output<false>(__fmul_rn(acc[mi][t][2 * hh + 1], g));
            }
            *reinterpret_cast<__nv_bfloat162*>(p.C + static_cast<long long>(row0 + r) * p.ldc + f) = v;
          }
        }
      }
    }
  }
  if (tid == 0) {
    __threadfence();
    const int d = atomicAdd(q.work + 1, 1);
    if (d == static_cast<int>(gridDim.x) - 1) {
      // every CTA has taken its last ticket: reset for the next launch
      atomicExch(q.work, 0);
      atomicExch(q.work + 1, 0);
    }
  }
}

#ifndef SHX_NO_PAIR
#include "shx_pair.cuh"
#endif

}  // namespace shx

extern "C" {

int shx_attrs(int relu2, int* out) {
  cudaFuncAttributes a;
  cudaError_t e = relu2 ? cudaFuncGetAttributes(&a, shx::gemm_kernel<true>)
                        : cudaFuncGetAttributes(&a, shx::gemm_kernel<false>);
  if (e != cudaSuccess) return static_cast<int>(e);
  out[0] = a.numRegs;
  out[1] = static_cast<int>(a.sharedSizeBytes);
  out[2] = a.maxThreadsPerBlock;
  out[3] = static_cast<int>(a.localSizeBytes);
  return 0;
}

int shx_scratch_floats(int grid) { return grid * shx::k2ScratchFloats; }  // covers v1 and v2

int shx2_attrs(int relu2, int* out) {
  cudaFuncAttributes a;
  cudaError_t e = relu2 ? cudaFuncGetAttributes(&a, shx::gemm2_kernel<true>)
                        : cudaFuncGetAttributes(&a, shx::gemm2_kernel<false>);
  if (e != cudaSuccess) return static_cast<int>(e);
  out[0] = a.numRegs;
  out[1] = static_cast<int>(a.sharedSizeBytes);
  out[2] = a.maxThreadsPerBlock;
  out[3] = static_cast<int>(a.localSizeBytes);
  return 0;
}

#ifndef SHX_NO_PAIR
int shx2_pair_attrs(int relu2, int* out) {
  cudaFuncAttributes a;
  cudaError_t e = relu2 ? cudaFuncGetAttributes(&a, shx::gemm2_pair_kernel<true>)
                        : cudaFuncGetAttributes(&a, shx::gemm2_pair_kernel<false>);
  if (e != cudaSuccess) return static_cast<int>(e);
  out[0] = a.numRegs;
  out[1] = static_cast<int>(a.sharedSizeBytes);
  out[2] = a.maxThreadsPerBlock;
  out[3] = static_cast<int>(a.localSizeBytes);
  return 0;
}
#endif

// C[M, N] = Marlin NVFP4 GEMM of A[M, K] for the units of py/__init__.py marlin_units.
int shx_gemm(const void* A, long long lda, int M, const void* B, const void* S, const void* GS, void* C,
             long long ldc, int N, int K, const void* units, int n_units, void* scratch, int grid, int relu2,
             void* stream) {
  if (N % 128 || K % 64 || lda % 8 || ldc % 2) return -2;
  if (reinterpret_cast<uintptr_t>(A) & 15) return -4;
  shx::Params p;
  p.A = static_cast<const __nv_bfloat16*>(A);
  p.lda = lda;
  p.M = M;
  p.B = static_cast<const uint4*>(B);
  p.S = static_cast<const uint8_t*>(S);
  p.GS = static_cast<const float*>(GS);
  p.C = static_cast<__nv_bfloat16*>(C);
  p.ldc = ldc;
  p.N = N;
  p.K = K;
  p.units = static_cast<const int*>(units);
  p.n_units = n_units;
  p.scratch = static_cast<float*>(scratch);
  auto st = static_cast<cudaStream_t>(stream);
  if (relu2)
    shx::gemm_kernel<true><<<grid, 128, 0, st>>>(p);
  else
    shx::gemm_kernel<false><<<grid, 128, 0, st>>>(p);
  return static_cast<int>(cudaGetLastError());
}

// v2: units of 64 x 128 from marlin_units(..., unit_features=128); work = int[2] zeroed once.
int shx2_gemm(const void* A, long long lda, int M, const void* B, const void* S, const void* GS, void* C,
              long long ldc, int N, int K, const void* units, int n_units, void* scratch, int grid, int relu2,
              void* work, int hint, void* stream) {
  if (N % 128 || K % 64 || lda % 8 || ldc % 2) return -2;
  if (reinterpret_cast<uintptr_t>(A) & 15) return -4;
  shx::Params2 q;
  shx::Params& p = q.p;
  p.A = static_cast<const __nv_bfloat16*>(A);
  p.lda = lda;
  p.M = M;
  p.B = static_cast<const uint4*>(B);
  p.S = static_cast<const uint8_t*>(S);
  p.GS = static_cast<const float*>(GS);
  p.C = static_cast<__nv_bfloat16*>(C);
  p.ldc = ldc;
  p.N = N;
  p.K = K;
  p.units = static_cast<const int*>(units);
  p.n_units = n_units;
  p.scratch = static_cast<float*>(scratch);
  q.work = static_cast<int*>(work);
  q.hint = hint;
  auto st = static_cast<cudaStream_t>(stream);
  if (relu2)
    shx::gemm2_kernel<true><<<grid, 128, 0, st>>>(q);
  else
    shx::gemm2_kernel<false><<<grid, 128, 0, st>>>(q);
  return static_cast<int>(cudaGetLastError());
}

// Narrow paired entry: host tables contain only row0=0/128, nrows=64, one full
// K segment; each record owns row0 and row0+64 at the same 128-column feature tile.
#ifndef SHX_NO_PAIR
int shx2_pair_gemm(const void* A, long long lda, int M, const void* B, const void* S, const void* GS, void* C,
                   long long ldc, int N, int K, const void* units, int n_units, void* scratch, int grid, int relu2,
                   void* work, int hint, void* stream) {
  if (M != 496 || !((N == 3712 && K == 2688) || (N == 2688 && K == 3712)) ||
      lda < K || lda % 8 || ldc != N || grid < 1 || n_units != 2 * (N / 128)) return -2;
  if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B) |
       reinterpret_cast<uintptr_t>(S) | reinterpret_cast<uintptr_t>(C) |
       reinterpret_cast<uintptr_t>(scratch)) & 15) return -4;
  shx::Params2 q;
  shx::Params& p = q.p;
  p.A = static_cast<const __nv_bfloat16*>(A);
  p.lda = lda;
  p.M = M;
  p.B = static_cast<const uint4*>(B);
  p.S = static_cast<const uint8_t*>(S);
  p.GS = static_cast<const float*>(GS);
  p.C = static_cast<__nv_bfloat16*>(C);
  p.ldc = ldc;
  p.N = N;
  p.K = K;
  p.units = static_cast<const int*>(units);
  p.n_units = n_units;
  p.scratch = static_cast<float*>(scratch);
  q.work = static_cast<int*>(work);
  q.hint = hint;
  auto st = static_cast<cudaStream_t>(stream);
  if (relu2) shx::gemm2_pair_kernel<true><<<grid, 256, 0, st>>>(q);
  else shx::gemm2_pair_kernel<false><<<grid, 256, 0, st>>>(q);
  return static_cast<int>(cudaGetLastError());
}
#endif

}  // extern "C"

// SPDX-License-Identifier: Apache-2.0
// Humming-exact NVFP4 W4A16 routed up (w13) for the decode/verify shape.
// N=1856, K=2688, 128 experts, top-k 6, M=1..8. Same dequant, MMA order and
// FTZ global-scale epilogue as routed_sparse's down kernel, then the bf16
// ReLU-squared the humming relu2 wrapper applies to that bf16 value.
// Every K16 step runs. No mask, no skipped sector: the up activation is dense.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t dq(uint32_t q, __nv_bfloat162 sc) {
  uint32_t lo = q & 7, hi = (q >> 4) & 7;
  lo = lo < 2 ? (lo ? 0x3f00 : 0) : (0x3f00 + (lo << 6));
  hi = hi < 2 ? (hi ? 0x3f00 : 0) : (0x3f00 + (hi << 6));
  uint32_t bits = lo | (hi << 16) | ((q & 8) << 12) | ((q & 128) << 24);
  auto v = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&bits), sc);
  return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ uint32_t ldq(const uint32_t* p) {
  uint32_t q;
  asm("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(q) : "l"(p));
  return q;
}

// Humming is compiled with fast math: its FP32 global-scale multiply flushes
// subnormal operands and results. Match that operation explicitly.
__device__ __forceinline__ float scale_output(float x, float gs) {
  float y;
  asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(x), "f"(gs));
  return y;
}

// bf16(relu(bf16(scale(acc)))^2), same two roundings as humming's epilogue
// followed by the relu2 wrapper (max.NaN, mul.rn, cvt.rn.bf16).
__device__ __forceinline__ __nv_bfloat16 relu2_bf16(float acc, float gs) {
  __nv_bfloat16 rounded = __float2bfloat16_rn(scale_output(acc, gs));
  float x = __bfloat162float(rounded);
  asm("max.NaN.f32 %0, %1, 0f00000000;" : "=f"(x) : "f"(x));
  asm("mul.rn.f32 %0, %1, %1;" : "=f"(x) : "f"(x));
  return __float2bfloat16_rn(x);
}

template <int R, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
dense_up(const __nv_bfloat16* X, const uint32_t* W, const uint16_t* S,
         const float* GS, const int* IDS, __nv_bfloat16* Y, int M) {
  constexpr int N = 1856, K = 2688, topk = 6;
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32, r = lane / 4, t = lane % 4;
  int route = blockIdx.y, e = IDS[route];
  unsigned mask0 = __ballot_sync(0xffffffff, lane < M * topk && IDS[lane] == e);
  unsigned mask1 = __ballot_sync(0xffffffff, lane + 32 < M * topk && IDS[lane + 32] == e);
  unsigned long long matches =
      (unsigned long long)mask0 | ((unsigned long long)mask1 << 32);
  if (route != __ffsll(matches) - 1) return;
  bool active = r < M && ((matches >> (r * topk)) & 63) != 0;
  int tile = blockIdx.x * WARPS + warp, n0 = tile * 16 + r, steps = K / 16;
  const uint32_t* wp = W + ((size_t)e * (N / 16) + tile) * steps * 32 + t * 8 + r;
  const uint16_t* sp = S + ((size_t)e * (N / 16) + tile) * steps * 8 + r;
  uint32_t qw[R], bx0[R], bx1[R];
  __nv_bfloat162 s0[R], s1[R];
  auto load = [&](int i, int step) {
    if (step < steps && tile < N / 16) {
      const uint32_t* xp =
          reinterpret_cast<const uint32_t*>(X + (size_t)r * K + step * 16);
      bx0[i] = active ? xp[t] : 0;
      bx1[i] = active ? xp[t + 4] : 0;
      qw[i] = ldq(wp + step * 32);
      uint16_t ss = sp[step * 8];
      __nv_fp8_e4m3 f0, f1;
      f0.__x = ss & 255;
      f1.__x = ss >> 8;
      s0[i] = __float2bfloat162_rn(float(f0) * 0.25f);
      s1[i] = __float2bfloat162_rn(float(f1) * 0.25f);
    } else {
      qw[i] = bx0[i] = bx1[i] = 0;
      s0[i] = s1[i] = __float2bfloat162_rn(0);
    }
  };
#pragma unroll
  for (int i = 0; i < R; i++) load(i, i);
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
  for (int base = 0; base < steps; base += R) {
#pragma unroll
    for (int i = 0; i < R; i++)
      if (base + i < steps) {
        uint32_t q = qw[i], b0 = bx0[i], b1 = bx1[i];
        auto sf0 = s0[i], sf1 = s1[i];
        load(i, base + i + R);
        uint32_t a0 = dq(q & 255, sf0);
        uint32_t a1 = dq((q >> 8) & 255, sf1);
        uint32_t a2 = dq((q >> 16) & 255, sf0);
        uint32_t a3 = dq(q >> 24, sf1);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      }
  }
  int t0 = 2 * t, t1 = t0 + 1;
  float gs = GS[e] * 4.0f;
  int slot0 = __ffsll((matches >> (t0 * topk)) & 63) - 1;
  int slot1 = __ffsll((matches >> (t1 * topk)) & 63) - 1;
  if (t0 < M && slot0 >= 0) {
    int dst = t0 * topk + slot0;
    if (n0 < N) Y[(size_t)dst * N + n0] = relu2_bf16(d0, gs);
    if (n0 + 8 < N) Y[(size_t)dst * N + n0 + 8] = relu2_bf16(d2, gs);
  }
  if (t1 < M && slot1 >= 0) {
    int dst = t1 * topk + slot1;
    if (n0 < N) Y[(size_t)dst * N + n0] = relu2_bf16(d1, gs);
    if (n0 + 8 < N) Y[(size_t)dst * N + n0 + 8] = relu2_bf16(d3, gs);
  }
}

extern "C" int routed_up_init() {
  cudaFuncAttributes a;
  return int(cudaFuncGetAttributes(&a, dense_up<4, 4>));
}

extern "C" int routed_up_run(const void* x, const void* w, const void* s,
                             const void* gs, const void* ids, void* y, int m,
                             void* stream) {
  if (m < 1 || m > 8) return int(cudaErrorInvalidValue);
  auto st = static_cast<cudaStream_t>(stream);
  dense_up<4, 4><<<dim3(29, m * 6), 128, 0, st>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<const uint32_t*>(w),
      static_cast<const uint16_t*>(s), static_cast<const float*>(gs),
      static_cast<const int*>(ids), static_cast<__nv_bfloat16*>(y), m);
  return int(cudaGetLastError());
}

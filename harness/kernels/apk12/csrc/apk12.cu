// SPDX-License-Identifier: Apache-2.0
// Exact BF16 m16n8k16 chain for 1..8 rows and the two attention projection shapes,
// with the weights streamed from the lossless 12-bit codec planes in fragment order.
// Each output follows one ascending-K chain of k16 MMAs from +0 in fp32 and is rounded
// once to BF16, the reduction F.linear gives at these shapes (the boot guard compares
// every layer). No K split, cross-warp reduction or weight format change.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "apk12_decode.h"
namespace apk12 {
struct P {
  const __nv_bfloat16* x;  // [m, ldx]
  const uint4* sm;         // [N/32][K/64][4 warps][32 lanes]: 16 sign+mantissa bytes
  const uint2* ex;         // same order: 16 exponent-code nibbles
  const uint8_t* meta;     // [N/32][K/128][32 columns]: block base, 0 for an escaped block
  const int32_t* ri;       // [N][K/128]: raw block index, -1 for a coded block
  const uint16_t* raw;     // [escaped blocks][128]: original bits
  __nv_bfloat16* y;        // [m, N]
  int m, ldx;
};
// Weight half of a ring slot: the lane's planes for 64-wide K group G of its warp's
// 8 columns (the load-time pack, never written at run time).
template<int K>
__device__ __forceinline__ void load_w(const P& p, int tile, int warp, int lane, int G,
                                       uint4& s, uint2& x, uint32_t& b) {
  size_t i = (((size_t)tile * (K / 64) + G) * 4 + warp) * 32 + lane;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(s.x), "=r"(s.y), "=r"(s.z), "=r"(s.w) : "l"(p.sm + i));
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];\n"
               : "=r"(x.x), "=r"(x.y) : "l"(p.ex + i));
  b = __ldg(p.meta + ((size_t)tile * (K / 128) + G / 2) * 32 + warp * 8 + lane / 4);
}
// Activation half: A registers 0 and 2 of the group's four k16 steps (row lane/4, zero
// at and past m; K = 16j + 2t and 16j + 8 + 2t). Registers 1 and 3 are rows 8-15: zero.
__device__ __forceinline__ void load_a(const P& p, int lane, int G, uint32_t (&a)[8]) {
  int row = lane / 4;
  const __nv_bfloat16* q = p.x + (size_t)row * p.ldx + G * 64 + (lane % 4) * 2;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    a[2 * j] = row < p.m ? __ldg((const unsigned int*)(q + 16 * j)) : 0u;
    a[2 * j + 1] = row < p.m ? __ldg((const unsigned int*)(q + 16 * j + 8)) : 0u;
  }
}
template<int K>
__device__ __forceinline__ void fragment(const P& p, int tile, int warp, int lane, int G,
                                         const uint4& s, const uint2& x, uint32_t b,
                                         uint32_t (&w)[8]) {
  const uint32_t sv[4] = {s.x, s.y, s.z, s.w}, xv[2] = {x.x, x.y};
  apk_decode(sv, xv, b, w);
  if (b == 0) {
    int n = tile * 32 + warp * 8 + lane / 4;
    int id = __ldg(p.ri + (size_t)n * (K / 128) + G / 2);
    apk_raw(p.raw + (size_t)id * 128, G & 1, lane % 4, w);
  }
}
__device__ __forceinline__ void mma(uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1,
                                    float (&c)[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}
// One warp per 8 columns, four per CTA, R groups (64 K each) in flight per lane.
template<int N, int K, int R>
__global__ __launch_bounds__(128, 3) void gemm(P p) {
  static_assert(N % 32 == 0 && K % 128 == 0 && R <= K / 64, "shape");
  constexpr int GN = K / 64;
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32, tile = blockIdx.x;
  uint4 s[R];
  uint2 x[R];
  uint32_t b[R], a[R][8];
  float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int r = 0; r < R; ++r) {
    load_w<K>(p, tile, warp, lane, r, s[r], x[r], b[r]);
    load_a(p, lane, r, a[r]);
  }
  for (int g0 = 0; g0 < GN; g0 += R) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      int G = g0 + r;
      if (G < GN) {
        uint32_t w[8];
        fragment<K>(p, tile, warp, lane, G, s[r], x[r], b[r], w);
#pragma unroll
        for (int j = 0; j < 4; ++j) mma(a[r][2 * j], a[r][2 * j + 1], w[2 * j], w[2 * j + 1], c);
        if (G + R < GN) {
          load_w<K>(p, tile, warp, lane, G + R, s[r], x[r], b[r]);
          load_a(p, lane, G + R, a[r]);
        }
      }
    }
  }
  int row = lane / 4, col = tile * 32 + warp * 8 + (lane % 4) * 2;
  if (row < p.m) {
    __nv_bfloat162 v;
    v.x = __float2bfloat16_rn(c[0]);
    v.y = __float2bfloat16_rn(c[1]);
    *(__nv_bfloat162*)(p.y + (size_t)row * N + col) = v;
  }
}
// Load-time check: every lane's fragment for every group, through load_w and fragment,
// against the original weight bits at its K positions. Counts differing registers.
template<int N, int K>
__global__ void check(P p, const uint16_t* w, unsigned int* bad) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)N / 32 * (K / 64) * 128) return;
  int lane = i % 32, warp = (i / 32) % 4, G = (i / 128) % (K / 64), tile = i / (128 * (K / 64));
  uint4 s;
  uint2 x;
  uint32_t b, v[8];
  load_w<K>(p, tile, warp, lane, G, s, x, b);
  fragment<K>(p, tile, warp, lane, G, s, x, b, v);
  const uint16_t* q = w + (size_t)(tile * 32 + warp * 8 + lane / 4) * K + G * 64 + (lane % 4) * 2;
  unsigned int n = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint16_t* e = q + 16 * j + 8 * h;
      n += v[2 * j + h] != ((uint32_t)e[0] | ((uint32_t)e[1] << 16));
    }
  if (n) atomicAdd(bad, n);
}
template<int N, int K>
void launch(const P& p, int ring, cudaStream_t st) {
  switch (ring) {
    case 1: gemm<N, K, 1><<<N / 32, 128, 0, st>>>(p); break;
    case 2: gemm<N, K, 2><<<N / 32, 128, 0, st>>>(p); break;
    case 4: gemm<N, K, 4><<<N / 32, 128, 0, st>>>(p); break;
    case 8: gemm<N, K, 8><<<N / 32, 128, 0, st>>>(p); break;
  }
}
template<int N, int K>
cudaError_t init_shape() {
  cudaFuncAttributes attr;
  cudaError_t rc = cudaFuncGetAttributes(&attr, gemm<N, K, 1>);
  if (rc == cudaSuccess) rc = cudaFuncGetAttributes(&attr, gemm<N, K, 2>);
  if (rc == cudaSuccess) rc = cudaFuncGetAttributes(&attr, gemm<N, K, 4>);
  if (rc == cudaSuccess) rc = cudaFuncGetAttributes(&attr, gemm<N, K, 8>);
  return rc;
}
}  // namespace apk12
using apk12::P;
extern "C" int apk12_init() {
  cudaFuncAttributes attr;
  cudaError_t rc = apk12::init_shape<4608, 2688>();
  if (rc == cudaSuccess) rc = apk12::init_shape<2688, 4096>();
  if (rc == cudaSuccess) rc = cudaFuncGetAttributes(&attr, apk12::check<4608, 2688>);
  if (rc == cudaSuccess) rc = cudaFuncGetAttributes(&attr, apk12::check<2688, 4096>);
  return (int)rc;
}
extern "C" int apk12_run(const void* x, const void* sm, const void* ex, const void* meta,
                         const void* ri, const void* raw, void* y, int m, int n, int k,
                         int ldx, int ring, void* stream) {
  if (m < 1 || m > 8 || (ring != 1 && ring != 2 && ring != 4 && ring != 8))
    return (int)cudaErrorInvalidValue;
  P p{(const __nv_bfloat16*)x, (const uint4*)sm, (const uint2*)ex, (const uint8_t*)meta,
      (const int32_t*)ri, (const uint16_t*)raw, (__nv_bfloat16*)y, m, ldx};
  cudaStream_t st = (cudaStream_t)stream;
  if (n == 4608 && k == 2688) apk12::launch<4608, 2688>(p, ring, st);
  else if (n == 2688 && k == 4096) apk12::launch<2688, 4096>(p, ring, st);
  else return (int)cudaErrorInvalidValue;
  return (int)cudaGetLastError();
}
extern "C" int apk12_check(const void* sm, const void* ex, const void* meta, const void* ri,
                           const void* raw, const void* w, void* bad, int n, int k,
                           void* stream) {
  P p{nullptr, (const uint4*)sm, (const uint2*)ex, (const uint8_t*)meta, (const int32_t*)ri,
      (const uint16_t*)raw, nullptr, 0, 0};
  cudaStream_t st = (cudaStream_t)stream;
  int blocks = n / 16 * k / 256;  // one thread per 16 weights, 256 threads per block
  if (n == 4608 && k == 2688)
    apk12::check<4608, 2688><<<blocks, 256, 0, st>>>(p, (const uint16_t*)w, (unsigned int*)bad);
  else if (n == 2688 && k == 4096)
    apk12::check<2688, 4096><<<blocks, 256, 0, st>>>(p, (const uint16_t*)w, (unsigned int*)bad);
  else return (int)cudaErrorInvalidValue;
  return (int)cudaGetLastError();
}

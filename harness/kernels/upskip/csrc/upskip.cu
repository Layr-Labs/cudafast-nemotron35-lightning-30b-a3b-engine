// upskip: humming-exact routed UP (w13) for the decode verify (1..8 tokens).
//
// Same chain as harness/kernels/zskip down4, K and N swapped (K=2688, N=1856,
// 168 k16 groups, 29 tiles of 64 columns), then the fused relu^2 epilogue:
//
//   acc = +0
//   for g in 0..167 ascending:
//       acc = mma.m16n8k16.row.col.f32.bf16.bf16.f32(W_bf16, h_bf16, acc)
//   y   = cvt.rn.bf16(mul.rn.ftz(acc, global_scale * 4))
//   out = cvt.rn.bf16(relu(y)^2)
//
// Weights are zskip's swap repack of checkpoint w13. One warp, one 64-column
// tile, one expert block. Expanded row r reads hidden[r / 6].

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

constexpr int K = 2688;
constexpr int N = 1856;
constexpr int G = K / 16;
constexpr int TILES = N / 64;
constexpr int TOPK = 6;
constexpr int BM = 16;
constexpr int R = 4;

struct Params {
  const __nv_bfloat16* A;
  long long lda;
  int M;
  const int* sorted_ids;
  const int* expert_ids;
  const int* num_tokens_padded;
  const uint8_t* W;
  const uint8_t* S;
  const float* GS;
  __nv_bfloat16* C;
  long long ldc;
  long long w_expert;
  long long w_k;
  long long s_expert;
  long long s_g;
  int max_blocks;
};

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

__device__ __forceinline__ uint16_t relu2_store(float acc, float gs4) {
  const uint32_t v = to_bf16x2(mul_ftz(acc, gs4), 0.0f);
  uint32_t sq;
  asm volatile(
      "{ .reg .b16 lo, hi; .reg .f32 x, z;\n"
      "mov.b32 {lo, hi}, %1;\n"
      "cvt.f32.bf16 x, lo;\n"
      "max.NaN.f32 x, x, 0f00000000;\n"
      "mul.rn.f32 x, x, x;\n"
      "mov.f32 z, 0f00000000;\n"
      "cvt.rn.bf16x2.f32 %0, z, x; }\n"
      : "=r"(sq)
      : "r"(v));
  return static_cast<uint16_t>(sq);
}

__global__ void __launch_bounds__(32) up_kernel(const Params p) {
  const int lane = threadIdx.x & 31;
  const int g8 = lane >> 2;
  const int t = lane & 3;
  const int T = blockIdx.x;
  const int mb = blockIdx.y;
  if (T >= TILES || mb >= p.max_blocks) return;
  if (mb * BM >= *p.num_tokens_padded) return;

  __shared__ int s_rows[8];
  __shared__ int s_nrows;
  __shared__ int s_expert;

  const unsigned id = lane < BM ? static_cast<unsigned>(p.sorted_ids[mb * BM + lane]) : 0xFFFFFFFFu;
  const bool valid = lane < BM && id < static_cast<unsigned>(p.M * TOPK);
  const unsigned ball = __ballot_sync(0xFFFFFFFFu, valid);
  const int rank = __popc(ball & ((1u << lane) - 1u));
  if (valid && rank < 8) s_rows[rank] = static_cast<int>(id);
  if (lane == 0) {
    const int n = __popc(ball);
    s_nrows = n < 8 ? n : 8;
    const int expert = p.expert_ids[mb];
    s_expert = (expert >= 0 && expert < 128) ? expert : -1;
  }
  __syncwarp();

  const int nrows = s_nrows;
  const int e = s_expert;
  if (nrows <= 0 || e < 0) return;

  const int kr0 = 2 * t;
  const int kr1 = kr0 + 1;
  const int kr2 = kr0 + 8;
  const int kr3 = kr0 + 9;
  const __nv_bfloat16* xrow = nullptr;
  if (g8 < nrows) {
    const int sid = s_rows[g8];
    xrow = p.A + (static_cast<long long>(sid / TOPK)) * p.lda;
  }
  const uint8_t* wbase = p.W + e * p.w_expert + static_cast<long long>(T) * 32 + g8 * 4;
  const uint8_t* sbase = p.S + e * p.s_expert + static_cast<long long>(T) * 64 + g8 * 8;

  float acc[4][4];
#pragma unroll
  for (int c = 0; c < 4; ++c)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[c][i] = 0.0f;

  uint32_t wq[R][4];
  uint32_t xb[R][2];
  uint2 sq[R];

  auto load = [&](int slot, int j) {
    if (j < G) {
      const int kb = 16 * j;
      wq[slot][0] = ldg_u32(wbase + static_cast<long long>(kb + kr0) * p.w_k);
      wq[slot][1] = ldg_u32(wbase + static_cast<long long>(kb + kr1) * p.w_k);
      wq[slot][2] = ldg_u32(wbase + static_cast<long long>(kb + kr2) * p.w_k);
      wq[slot][3] = ldg_u32(wbase + static_cast<long long>(kb + kr3) * p.w_k);
      sq[slot] = ldg_u64(sbase + static_cast<long long>(j) * p.s_g);
      xb[slot][0] = xrow ? ldg_u32_l1(xrow + kb + kr0) : 0u;
      xb[slot][1] = xrow ? ldg_u32_l1(xrow + kb + kr2) : 0u;
    }
  };

#pragma unroll
  for (int i = 0; i < R; ++i) load(i, i);

  for (int j0 = 0; j0 < G; j0 += R) {
#pragma unroll
    for (int i = 0; i < R; ++i) {
      const int j = j0 + i;
      if (j < G) {
        const uint32_t w0 = wq[i][0], w1 = wq[i][1], w2 = wq[i][2], w3 = wq[i][3];
        const uint2 sc = sq[i];
        const uint32_t b0 = xb[i][0], b1 = xb[i][1];
        load(i, j + R);
        const uint32_t s01 =
            hmul2_u32(fp8_to_bf16x2(((sc.x << 8) & 0x0000FF00u) | ((sc.x << 16) & 0xFF000000u)), kExp120);
        const uint32_t s23 =
            hmul2_u32(fp8_to_bf16x2(((sc.x >> 8) & 0x0000FF00u) | (sc.x & 0xFF000000u)), kExp120);
        const uint32_t s45 =
            hmul2_u32(fp8_to_bf16x2(((sc.y << 8) & 0x0000FF00u) | ((sc.y << 16) & 0xFF000000u)), kExp120);
        const uint32_t s67 =
            hmul2_u32(fp8_to_bf16x2(((sc.y >> 8) & 0x0000FF00u) | (sc.y & 0xFF000000u)), kExp120);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const uint32_t sp = c == 0 ? s01 : (c == 1 ? s23 : (c == 2 ? s45 : s67));
          const uint32_t slo = __byte_perm(sp, 0, 0x1010);
          const uint32_t shi = __byte_perm(sp, 0, 0x3232);
          const uint32_t sel = c | (c << 4) | ((c + 4) << 8) | ((c + 4) << 12);
          const uint32_t x01 = __byte_perm(w0, w1, sel);
          const uint32_t x23 = __byte_perm(w2, w3, sel);
          const uint32_t a0 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x01 << 12), kExp124), slo);
          const uint32_t a1 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x01 << 8), kExp124), shi);
          const uint32_t a2 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x23 << 12), kExp124), slo);
          const uint32_t a3 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x23 << 8), kExp124), shi);
          mma_bf16(acc[c], a0, a1, a2, a3, b0, b1);
        }
      }
    }
  }

  const float gs4 = mul_ftz(p.GS[e], 4.0f);
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int col = T * 64 + 16 * c + g8;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int cc = col + ((q >> 1) ? 8 : 0);
      const int tok = 2 * t + (q & 1);
      if (tok < nrows && cc < N) {
        reinterpret_cast<uint16_t*>(p.C)[static_cast<long long>(s_rows[tok]) * p.ldc + cc] =
            relu2_store(acc[c][q], gs4);
      }
    }
  }
}

extern "C" int upskip_run(const void* A, long long lda, int M, const void* sorted_ids, const void* expert_ids,
                          const void* num_tokens_padded, int max_blocks, const void* W, const void* S, const void* GS,
                          void* C, long long ldc, long long w_expert, long long w_k, long long s_expert, long long s_g,
                          void* stream_ptr) {
  if (max_blocks <= 0 || M <= 0) return 0;
  Params p{};
  p.A = reinterpret_cast<const __nv_bfloat16*>(A);
  p.lda = lda;
  p.M = M;
  p.sorted_ids = reinterpret_cast<const int*>(sorted_ids);
  p.expert_ids = reinterpret_cast<const int*>(expert_ids);
  p.num_tokens_padded = reinterpret_cast<const int*>(num_tokens_padded);
  p.W = reinterpret_cast<const uint8_t*>(W);
  p.S = reinterpret_cast<const uint8_t*>(S);
  p.GS = reinterpret_cast<const float*>(GS);
  p.C = reinterpret_cast<__nv_bfloat16*>(C);
  p.ldc = ldc;
  p.w_expert = w_expert;
  p.w_k = w_k;
  p.s_expert = s_expert;
  p.s_g = s_g;
  p.max_blocks = max_blocks;
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  up_kernel<<<dim3(TILES, max_blocks), 32, 0, stream>>>(p);
  return 0;
}

// routerbig: the fp32 MoE router GEMM y = x . w^T at prefill M (M = 496, N = 128, K = 2688)
// with the summation tree of the cuBLAS kernel torch.mm picks for it
// (cutlass_80_simt_sgemm_128x128_8x4_tn_align1, grid 4 x 1 x 12, + cublasLt splitKreduce):
//
//   C_s = fma chain over k = 224 s .. 224 s + 223 ascending, from 0     (s = 0 .. 11)
//   y   = ((((C_0 + C_1) + C_2) + ...) + C_11)
//
// recovered by cancellation probes (harness/kernels/smallk/tests/probe.py treeM496N128:
// 12 chains of 224, combined left to right). Pass 1 computes every C_s of a 64-row tile in its
// own CTA (96 CTAs of 256 threads instead of 48 of 128); pass 2 adds the 12 partials of each
// output in order. harness/kernels/routerbig/tests/exact.py checks torch.equal against torch.mm.

#include <cuda_runtime.h>
#include <stdint.h>

#include <type_traits>

namespace {

constexpr int SLICES = 12;
constexpr int SLICE_K = 224;  // K / SLICES
constexpr int BN = 128;       // experts per CTA (all)
constexpr int KS = 16;        // k per smem stage (SLICE_K = 14 stages)
constexpr int PAD = 4;        // smem row padding (transposed stores hit distinct banks)
constexpr int THREADS = 256;  // 16 x 16 threads: TR rows x (4 + 4) experts each

// Pass 1: partial[s][m][n] = C_s(m, n) for a BM-row tile (TR = BM / 16 rows per thread).
template <int BM>
__global__ void __launch_bounds__(THREADS) router_partials(const float* __restrict__ X, const float* __restrict__ W,
                                                          float* __restrict__ P, int M, int K) {
  constexpr int TR = BM / 16;
  static_assert(TR == 2 || TR == 4 || TR == 8, "rows per thread");
  __shared__ __align__(16) float sA[2][KS][BM + PAD];  // transposed: [k][row]
  __shared__ __align__(16) float sW[2][KS][BN + PAD];  // transposed: [k][expert]
  const int tid = threadIdx.x;
  const int ty = tid >> 4, tx = tid & 15;  // rows TR*ty.., experts 4*tx.. and 64+4*tx..
  const int m0 = blockIdx.x * BM;
  const int s = blockIdx.y;
  const int k0 = s * SLICE_K;

  constexpr int TA = BM * KS / 4, TW = BN * KS / 4;  // float4 per stage
  constexpr int NA = (TA + THREADS - 1) / THREADS, NW = TW / THREADS;
  float4 ra[2][NA], rw[2][NW];  // two stages in flight in registers
  auto gload = [&](int set, int kk) {
#pragma unroll
    for (int i = 0; i < NA; ++i) {
      const int idx = tid + i * THREADS;
      const int r = idx >> 2, c = idx & 3;
      const int gm = m0 + r;
      ra[set][i] = (idx < TA && gm < M) ? *reinterpret_cast<const float4*>(X + (size_t)gm * K + kk + c * 4)
                                         : make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (int i = 0; i < NW; ++i) {
      const int idx = tid + i * THREADS;
      const int r = idx >> 2, c = idx & 3;
      rw[set][i] = *reinterpret_cast<const float4*>(W + (size_t)r * K + kk + c * 4);
    }
  };
  auto sstore = [&](int set, int buf) {
#pragma unroll
    for (int i = 0; i < NA; ++i) {
      const int idx = tid + i * THREADS;
      if (idx < TA) {
        const int r = idx >> 2, c = idx & 3;
        sA[buf][c * 4 + 0][r] = ra[set][i].x;
        sA[buf][c * 4 + 1][r] = ra[set][i].y;
        sA[buf][c * 4 + 2][r] = ra[set][i].z;
        sA[buf][c * 4 + 3][r] = ra[set][i].w;
      }
    }
#pragma unroll
    for (int i = 0; i < NW; ++i) {
      const int idx = tid + i * THREADS;
      const int r = idx >> 2, c = idx & 3;
      sW[buf][c * 4 + 0][r] = rw[set][i].x;
      sW[buf][c * 4 + 1][r] = rw[set][i].y;
      sW[buf][c * 4 + 2][r] = rw[set][i].z;
      sW[buf][c * 4 + 3][r] = rw[set][i].w;
    }
  };

  float acc[TR][8];
#pragma unroll
  for (int i = 0; i < TR; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[i][j] = 0.f;

  constexpr int STEPS = SLICE_K / KS;  // 14 (even)
  static_assert(STEPS % 2 == 0, "the loop below handles stages in pairs");
  gload(0, k0);
  sstore(0, 0);
  gload(1, k0 + KS);
  __syncthreads();
  auto stage = [&](auto set_tag, int st) {
    constexpr int SET = decltype(set_tag)::value;
    if (st + 2 < STEPS) gload(SET, k0 + (st + 2) * KS);  // set SET held stage st (stored)
#pragma unroll
    for (int k = 0; k < KS; ++k) {  // ascending k: one fma per output per k
      float av[TR];
      if constexpr (TR == 2) {
        const float2 a = *reinterpret_cast<const float2*>(&sA[SET][k][ty * 2]);
        av[0] = a.x;
        av[1] = a.y;
      } else {
#pragma unroll
        for (int h = 0; h < TR / 4; ++h) {
          const float4 a = *reinterpret_cast<const float4*>(&sA[SET][k][ty * TR + 4 * h]);
          av[4 * h + 0] = a.x;
          av[4 * h + 1] = a.y;
          av[4 * h + 2] = a.z;
          av[4 * h + 3] = a.w;
        }
      }
      const float4 w0 = *reinterpret_cast<const float4*>(&sW[SET][k][tx * 4]);
      const float4 w1 = *reinterpret_cast<const float4*>(&sW[SET][k][64 + tx * 4]);
      const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
      for (int i = 0; i < TR; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = __fmaf_rn(av[i], wv[j], acc[i][j]);
    }
    if (st + 1 < STEPS) {
      sstore(SET ^ 1, SET ^ 1);
      __syncthreads();
    }
  };
#pragma unroll 1
  for (int st = 0; st < STEPS; st += 2) {
    stage(std::integral_constant<int, 0>{}, st);
    stage(std::integral_constant<int, 1>{}, st + 1);
  }

  float* out = P + (size_t)s * M * BN;
#pragma unroll
  for (int i = 0; i < TR; ++i) {
    const int m = m0 + ty * TR + i;
    if (m >= M) continue;
    *reinterpret_cast<float4*>(out + (size_t)m * BN + tx * 4) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
    *reinterpret_cast<float4*>(out + (size_t)m * BN + 64 + tx * 4) =
        make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
  }
}

// Pass 2: y = ((P_0 + P_1) + ...) + P_11, elementwise.
__global__ void router_reduce(const float* __restrict__ P, float* __restrict__ Y, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  float t = P[i];
#pragma unroll
  for (int s = 1; s < SLICES; ++s) t = __fadd_rn(t, P[(size_t)s * total + i]);
  Y[i] = t;
}

}  // namespace

extern "C" {

int routerbig_version() { return 2; }

static int g_bm = 64;  // rows per CTA of pass 1 (32, 64 or 128); routerbig_set_bm changes it

int routerbig_set_bm(int bm) {
  if (bm != 32 && bm != 64 && bm != 128) return -1;
  g_bm = bm;
  return 0;
}

static void launch_partials(const float* x, const float* w, float* p, int M, int K, cudaStream_t s) {
  if (g_bm == 32) {
    router_partials<32><<<dim3((M + 31) / 32, SLICES), THREADS, 0, s>>>(x, w, p, M, K);
  } else if (g_bm == 128) {
    router_partials<128><<<dim3((M + 127) / 128, SLICES), THREADS, 0, s>>>(x, w, p, M, K);
  } else {
    router_partials<64><<<dim3((M + 63) / 64, SLICES), THREADS, 0, s>>>(x, w, p, M, K);
  }
}

int routerbig_partials_only(const void* x, const void* w, void* p, int M, int N, int K, void* stream) {
  if (N != BN || K != SLICES * SLICE_K || M <= 0) return -1;
  launch_partials(static_cast<const float*>(x), static_cast<const float*>(w), static_cast<float*>(p), M, K,
                  static_cast<cudaStream_t>(stream));
  return (int)cudaGetLastError();
}

// X [M, 2688] fp32 row-major, W [128, 2688] fp32 row-major, P scratch [12, M, 128] fp32,
// Y [M, 128] fp32 row-major.
int routerbig_gemm(const void* x, const void* w, void* p, void* y, int M, int N, int K, void* stream) {
  if (N != BN || K != SLICES * SLICE_K || M <= 0) return -1;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  launch_partials(static_cast<const float*>(x), static_cast<const float*>(w), static_cast<float*>(p), M, K, s);
  const int total = M * N;
  router_reduce<<<(total + 255) / 256, 256, 0, s>>>(static_cast<const float*>(p), static_cast<float*>(y), total);
  return (int)cudaGetLastError();
}

}  // extern "C"

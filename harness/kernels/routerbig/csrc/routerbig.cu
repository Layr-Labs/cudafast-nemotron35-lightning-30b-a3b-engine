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

#include <cmath>

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

// Fused prefill routing (route_kernel): the 12-partial sum of router_reduce (same order), the
// grouped top-k of vLLM v0.27.1 single_group_topk_warp_kernel (n_group = topk_group = 1, sigmoid,
// fp32 bias; the arithmetic of harness/kernels/smallk topk_token, copied below), then, in the last
// CTA to finish, moe_align_block_size + count_and_sort_expert_tokens for up to two block sizes
// (sizes, sentinel and -1 fill as vLLM allocates and writes them; within an expert the slot order
// comes from shared-memory atomics, as stock count_and_sort's global atomics leave it).
constexpr int kWarp = 32;
constexpr int kE = BN;          // 128 experts
constexpr int kRouteWarps = 16;  // tokens per CTA
constexpr int kMaxNumel = 4096;
constexpr float kInvalid = -INFINITY;

__device__ __forceinline__ float sigmoid_accurate(float x) { return 0.5f * tanhf(0.5f * x) + 0.5f; }
__device__ __forceinline__ uint32_t twiddle_in(float v) {
  const uint32_t b = __float_as_uint(v);
  return b ^ ((b & 0x80000000u) ? 0xffffffffu : 0x80000000u);
}
__device__ __forceinline__ uint64_t make_key(float v, int idx) {
  return (static_cast<uint64_t>(twiddle_in(v)) << 32) | static_cast<uint64_t>(0xFFFF & (65535 - idx));
}

// One warp ranks one token's 128 experts (smallk topk_token<4>, unchanged).
// s_lane[c] is this lane's logit for expert c * 32 + lane (topk_token read it as s[e]).
__device__ __forceinline__ void topk_token(const float (&s_lane)[kE / kWarp], const float* __restrict__ bias,
                                           int topk, int renormalize, float routed_scaling_factor,
                                           float* __restrict__ out_values, int32_t* __restrict__ out_indices) {
  constexpr int C = kE / kWarp;
  const int lane = threadIdx.x % kWarp;
  uint64_t keys[C];
  float sel[C];
#pragma unroll
  for (int c = 0; c < C; ++c) {
    const int e = c * kWarp + lane;
    const float in = s_lane[c];
    const float b = bias[e];
    float selection = kInvalid;
    if (isfinite(in) && isfinite(b)) {
      const float unbiased = sigmoid_accurate(in);
      const float biased = unbiased + b;
      selection = biased == 0.0f ? 0.0f : biased;
    }
    sel[c] = selection;
    keys[c] = make_key(selection, e);
  }
  unsigned taken = 0;
  float lane_score = kInvalid;
  int lane_expert = -1;
  for (int k = 0; k < topk; ++k) {
    uint64_t best = 0;
    int best_c = -1;
#pragma unroll
    for (int c = 0; c < C; ++c) {
      if (!(taken & (1u << c)) && (best_c < 0 || keys[c] > best)) {
        best = keys[c];
        best_c = c;
      }
    }
    uint64_t wmax = best;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const uint64_t o = __shfl_xor_sync(0xffffffffu, wmax, off);
      wmax = o > wmax ? o : wmax;
    }
    const int widx = 65535 - static_cast<int>(wmax & 0xFFFF);
    if (best_c >= 0 && best == wmax) taken |= 1u << best_c;
    const int owner = widx % kWarp;
    const int owner_c = widx / kWarp;
    float my = kInvalid;
#pragma unroll
    for (int c = 0; c < C; ++c)
      if (c == owner_c) my = sel[c];
    const float wscore = __shfl_sync(0xffffffffu, my, owner);
    if (lane == k) {
      lane_score = wscore;
      lane_expert = widx;
    }
  }
  float lane_unbiased = 0.0f;
  if (lane < topk && lane_expert >= 0 && lane_expert < kE) lane_unbiased = lane_score - bias[lane_expert];
  const bool finite_selection = lane < topk && lane_score != kInvalid;
  lane_unbiased = finite_selection ? lane_unbiased : 0.0f;
  const unsigned finite_mask = __ballot_sync(0xffffffffu, finite_selection);
  float sum = lane_unbiased;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) sum = sum + __shfl_xor_sync(0xffffffffu, sum, off);
  if (lane < topk) {
    float output = 0.0f;
    if (finite_mask == 0) {
      if (renormalize) output = 1.0f / static_cast<float>(topk);
    } else if (finite_selection) {
      float scale = routed_scaling_factor;
      if (renormalize) scale /= sum + 1e-20f;
      output = lane_unbiased * scale;
    }
    out_values[lane] = output;
    out_indices[lane] = lane_expert;
  }
}

struct AlignOut {
  int32_t* sorted;
  int32_t* eids;
  int32_t* ntp;
  int bs, maxpad, maxblk;
};

__global__ void __launch_bounds__(kRouteWarps * kWarp)
    route_kernel(const float* __restrict__ P, int M, const float* __restrict__ bias, int topk, int renormalize,
                 float scale, float* __restrict__ Y, float* __restrict__ vals, int32_t* __restrict__ ids, int nalign,
                 AlignOut a0, AlignOut a1, unsigned* __restrict__ counter) {
  __shared__ int32_t s_ids[kMaxNumel];
  __shared__ int32_t s_cnt[kE];
  __shared__ int32_t s_cum[2][kE + 1];
  __shared__ int32_t s_fill[2][kE];
  __shared__ int s_last;
  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;
  const int token = blockIdx.x * kRouteWarps + warp;
  if (token < M) {
    const int total = M * kE;
    float* y = Y + static_cast<size_t>(token) * kE;
    float lg[kE / kWarp];
#pragma unroll
    for (int c = 0; c < kE / kWarp; ++c) {
      const int i = token * kE + c * kWarp + lane;
      float t = P[i];
#pragma unroll
      for (int sl = 1; sl < SLICES; ++sl) t = __fadd_rn(t, P[static_cast<size_t>(sl) * total + i]);
      lg[c] = t;
      y[c * kWarp + lane] = t;
    }
    topk_token(lg, bias, topk, renormalize, scale, vals + token * topk, ids + token * topk);
  }
  {  // (b) every CTA writes its share of the sentinel fill; the last CTA scatters over it
    const int numel = M * topk;
    const int stride = gridDim.x * blockDim.x;
    for (int k = 0; k < nalign; ++k) {
      const AlignOut a = k == 0 ? a0 : a1;
      for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < a.maxpad; i += stride) a.sorted[i] = numel;
    }
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) s_last = atomicAdd(counter, 1u) == gridDim.x - 1;
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  const int numel = M * topk;
  for (int i = threadIdx.x; i < numel; i += blockDim.x) s_ids[i] = __ldcg(ids + i);
  for (int e = threadIdx.x; e < kE; e += blockDim.x) {
    s_cnt[e] = 0;
    s_fill[0][e] = 0;
    s_fill[1][e] = 0;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < numel; i += blockDim.x) {
    const int e = s_ids[i];
    if (e >= 0 && e < kE) atomicAdd(&s_cnt[e], 1);
  }
  __syncthreads();
  if (warp < nalign) {  // warp k scans the padded counts for block size k (4 experts per lane)
    const int bs = warp == 0 ? a0.bs : a1.bs;
    int v[kE / kWarp], sum = 0;
#pragma unroll
    for (int j = 0; j < kE / kWarp; ++j) {
      const int c = s_cnt[lane * (kE / kWarp) + j];
      v[j] = (c + bs - 1) / bs * bs;
      sum += v[j];
    }
    int incl = sum;
#pragma unroll
    for (int off = 1; off < kWarp; off <<= 1) {
      const int u = __shfl_up_sync(0xffffffffu, incl, off);
      if (lane >= off) incl += u;
    }
    int run = incl - sum;
#pragma unroll
    for (int j = 0; j < kE / kWarp; ++j) {
      s_cum[warp][lane * (kE / kWarp) + j] = run;
      run += v[j];
    }
    if (lane == kWarp - 1) {
      s_cum[warp][kE] = incl;
      *(warp == 0 ? a0.ntp : a1.ntp) = incl;
    }
  }
  __syncthreads();
  for (int k = 0; k < nalign; ++k) {
    const AlignOut a = k == 0 ? a0 : a1;
    const int tot = s_cum[k][kE];
    for (int b = threadIdx.x; b < a.maxblk; b += blockDim.x) {
      const int pos = b * a.bs;
      int e = -1;
      if (pos < tot) {
        int lo = 0, hi = kE - 1;  // the last expert whose range starts at or before pos
        while (lo < hi) {
          const int mid = (lo + hi + 1) / 2;
          if (s_cum[k][mid] <= pos) lo = mid; else hi = mid - 1;
        }
        e = lo;
      }
      a.eids[b] = e;
    }
    for (int i = threadIdx.x; i < numel; i += blockDim.x) {
      const int e = s_ids[i];
      if (e >= 0 && e < kE) a.sorted[s_cum[k][e] + atomicAdd(&s_fill[k][e], 1)] = i;
    }
  }
  if (threadIdx.x == 0) *counter = 0u;
}

}  // namespace

extern "C" {

int routerbig_version() { return 3; }

// Fused routing from pass-1 partials P [12, M, 128]: Y = router_reduce(P), top-k values/indices,
// and moe_align outputs for nalign (0..2) block sizes. counter: one zeroed device uint32 (the
// kernel leaves it zeroed).
int routerbig_route(const void* p, int M, const void* bias, int topk, int renormalize, float scale, void* y,
                    void* vals, void* ids, int nalign, int bs0, void* sorted0, int maxpad0, void* eids0, int maxblk0,
                    void* ntp0, int bs1, void* sorted1, int maxpad1, void* eids1, int maxblk1, void* ntp1,
                    void* counter, void* stream) {
  if (M <= 0 || topk < 1 || topk > kWarp || M * topk > kMaxNumel || nalign < 0 || nalign > 2) return -1;
  AlignOut a0{static_cast<int32_t*>(sorted0), static_cast<int32_t*>(eids0), static_cast<int32_t*>(ntp0), bs0,
              maxpad0, maxblk0};
  AlignOut a1{static_cast<int32_t*>(sorted1), static_cast<int32_t*>(eids1), static_cast<int32_t*>(ntp1), bs1,
              maxpad1, maxblk1};
  route_kernel<<<(M + kRouteWarps - 1) / kRouteWarps, kRouteWarps * kWarp, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(p), M, static_cast<const float*>(bias), topk, renormalize, scale,
      static_cast<float*>(y), static_cast<float*>(vals), static_cast<int32_t*>(ids), nalign, a0, a1,
      static_cast<unsigned*>(counter));
  return (int)cudaGetLastError();
}

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

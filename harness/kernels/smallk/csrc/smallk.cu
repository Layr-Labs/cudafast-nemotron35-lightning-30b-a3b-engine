// smallk: bit-exact replacements for small latency-bound kernels of the Nemotron-3.5 MTP verify.
//
// 1. router GEMM  y[m][n] = sum_k x[m][k] * w[n][k] (fp32, M = 4 rows, the MoE gate) with the exact
//    summation tree of cuBLAS gemmSN_TN_kernel<float, 128, 16, 2, 4, 4, 4, true, ...>, which
//    torch picks for this shape (probed by cancellation, tests/probe.py tree):
//      lane r = k mod 16, tile j = k div 256;
//      C(r, j) = fma chain over k = 256 j + r, 256 j + r + 16, ... ascending, from 0;
//      L(r)    = ((C(r, 0) + C(r, 1)) + C(r, 2)) + ...  (tiles ascending);
//      y       = ((L(0) + L(1)) + L(2)) + ... + L(15)   (lanes ascending).
//    cuBLAS runs 8 outputs per CTA (16 CTAs for 128 experts); here one CTA owns R rows and gives
//    every (tile, lane) chain its own thread, so all weight loads are in flight at once.
//    W16 variant: the gate weight is read from a bf16 shadow that the caller built only if every fp32
//    value is exactly a bf16 value (the checkpoint stores the gate in bf16; vLLM upcasts it on GB10).
//    Each element is widened back with (bits << 16), so the fma chains see the same fp32 operands
//    and the result is bit-identical; the weight read halves (1.38 -> 0.69 MB per gate call).
// 2. grouped top-k (n_group = topk_group = 1, sigmoid scoring, fp32 bias, renormalize) with the
//    arithmetic of vLLM v0.27.1 single_group_topk_warp_kernel: same sigmoid expression, same
//    selection key (score bits, then lower expert index), same unbias/sum/scale rounding steps.
// 3. moe_align_block_size + count_and_sort_expert_tokens in one single-CTA kernel (integers only),
//    alone or fused behind the top-k (route_kernel).
//
// No --use_fast_math: every float operation stays the plain IEEE fp32 operation written here.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#define SK_VERSION 5

namespace {

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

// ------------------------------------------------------------------------------------------------
// router GEMM
// ------------------------------------------------------------------------------------------------
constexpr int kLanes = 16;      // k mod 16
constexpr int kTile = 256;      // k elements per tile (16 per lane)
constexpr int kMaxTiles = 16;   // K <= 4096

// Gate weight element i: fp32 as stored, or (W16) the bf16 shadow widened exactly to fp32.
template <bool W16>
__device__ __forceinline__ float ldw(const void* __restrict__ w, long i) {
#ifndef SMALLK_NO_W16
  if constexpr (W16) {
    return __uint_as_float(static_cast<unsigned int>(__ldg(static_cast<const unsigned short*>(w) + i)) << 16);
  } else
#endif
  {
    return __ldg(static_cast<const float*>(w) + i);
  }
}

template <bool X16>
__device__ __forceinline__ float ldxv(const void* __restrict__ x, long i) {
  if constexpr (X16) {
    return __uint_as_float(static_cast<unsigned int>(__ldg(static_cast<const unsigned short*>(x) + i)) << 16);
  } else {
    return __ldg(static_cast<const float*>(x) + i);
  }
}

// MODE 0: chain = fmaf(w, x, acc) from acc = 0. MODE 1: chain = acc + (w * x) (separate rounding).
template <int R, int M, int MODE, bool W16 = false, bool X16 = false>
__global__ void __launch_bounds__(kLanes * kMaxTiles)
    router_gemm_kernel(const void* __restrict__ x, long ldx, const void* __restrict__ w, int K,
                       float* __restrict__ y, long ldy, int N, const int* __restrict__ vrows) {
  const int t = threadIdx.x;
  const int r = t & (kLanes - 1);
  const int j = t / kLanes;
  const int J = (K + kTile - 1) / kTile;
  const int n0 = blockIdx.x * R;
  __shared__ float part[R * M][kMaxTiles][kLanes];
  __shared__ float lane_sum[R * M][kLanes];

  const int vr = vrows == nullptr ? M : *vrows;
  float wv_pref[kLanes][R];
  if (j < J) {
    const int len_pref = min(kLanes, (K - j * kTile) / kLanes);
    if (len_pref == kLanes) {
      const int base_pref = j * kTile + r;
      long wrow_pref[R];
#pragma unroll
      for (int a = 0; a < R; ++a) {
        const int n = min(n0 + a, N - 1);
        wrow_pref[a] = static_cast<long>(n) * K + base_pref;
      }
#pragma unroll
      for (int i = 0; i < kLanes; ++i)
#pragma unroll
        for (int a = 0; a < R; ++a) wv_pref[i][a] = ldw<W16>(w, wrow_pref[a] + i * kLanes);
    }
  }
  pdl_wait();
  if (j < J) {
    const int len = min(kLanes, (K - j * kTile) / kLanes);
    const int base = j * kTile + r;
    float acc[R][M];
#pragma unroll
    for (int a = 0; a < R; ++a)
#pragma unroll
      for (int m = 0; m < M; ++m) acc[a][m] = 0.0f;
    long wrow[R];
#pragma unroll
    for (int a = 0; a < R; ++a) {
      const int n = min(n0 + a, N - 1);
      wrow[a] = static_cast<long>(n) * K + base;
    }
    if (len == kLanes) {
#pragma unroll
      for (int i = 0; i < kLanes; ++i) {
        float xv[M];
#pragma unroll
        for (int m = 0; m < M; ++m) xv[m] = ldxv<X16>(x, m * ldx + base + i * kLanes);
#pragma unroll
        for (int a = 0; a < R; ++a)
#pragma unroll
          for (int m = 0; m < M; ++m) {
            if (MODE == 0) {
              acc[a][m] = __fmaf_rn(wv_pref[i][a], xv[m], acc[a][m]);
            } else {
              acc[a][m] = __fadd_rn(acc[a][m], __fmul_rn(wv_pref[i][a], xv[m]));
            }
          }
      }
    } else {
      for (int i = 0; i < len; ++i) {
        float xv[M];
#pragma unroll
        for (int m = 0; m < M; ++m) xv[m] = ldxv<X16>(x, m * ldx + base + i * kLanes);
#pragma unroll
        for (int a = 0; a < R; ++a) {
          const float wa = ldw<W16>(w, wrow[a] + i * kLanes);
#pragma unroll
          for (int m = 0; m < M; ++m) {
            if (MODE == 0) {
              acc[a][m] = __fmaf_rn(wa, xv[m], acc[a][m]);
            } else {
              acc[a][m] = __fadd_rn(acc[a][m], __fmul_rn(wa, xv[m]));
            }
          }
        }
      }
    }
#pragma unroll
    for (int a = 0; a < R; ++a)
#pragma unroll
      for (int m = 0; m < M; ++m) part[a * M + m][j][r] = acc[a][m];
  }
  __syncthreads();
  // tiles ascending per lane
  for (int q = t; q < R * M * kLanes; q += blockDim.x) {
    const int am = q / kLanes;
    const int lane = q % kLanes;
    float s = part[am][0][lane];
    for (int jj = 1; jj < J; ++jj) s = __fadd_rn(s, part[am][jj][lane]);
    lane_sum[am][lane] = s;
  }
  __syncthreads();
  // lanes ascending
  if (t < R * M) {
    const int a = t / M;
    const int m = t % M;
    // rows at or past vr take the last valid row's logits bit for bit, so they route to experts
    // that row already loads
    const int src = min(m, max(vr, 1) - 1);
    const int am = a * M + src;
    float s = lane_sum[am][0];
#pragma unroll
    for (int lane = 1; lane < kLanes; ++lane) s = __fadd_rn(s, lane_sum[am][lane]);
    if (n0 + a < N) y[m * ldy + n0 + a] = s;
  }
  pdl_launch_dependents();
}

// ------------------------------------------------------------------------------------------------
// grouped top-k, single group, sigmoid scoring (vLLM v0.27.1 single_group_topk_warp_kernel)
// ------------------------------------------------------------------------------------------------
constexpr int kWarp = 32;
constexpr float kInvalid = -INFINITY;

// Adapted from TensorRT-LLM noAuxTcKernels.cu via vLLM grouped_topk_kernels.cu (Apache-2.0).
__device__ __forceinline__ float sigmoid_accurate(float x) { return 0.5f * tanhf(0.5f * x) + 0.5f; }

__device__ __forceinline__ uint32_t twiddle_in(float v) {
  // cub::Traits<float>::TwiddleIn: negative -> flip all bits, else flip the sign bit
  const uint32_t b = __float_as_uint(v);
  return b ^ ((b & 0x80000000u) ? 0xffffffffu : 0x80000000u);
}

__device__ __forceinline__ uint64_t make_key(float v, int idx) {
  return (static_cast<uint64_t>(twiddle_in(v)) << 32) | static_cast<uint64_t>(0xFFFF & (65535 - idx));
}

// One warp ranks one token's E = 32 * C experts; writes values/indices (and the ids to s_ids).
template <int C>
__device__ __forceinline__ void topk_token(const float* __restrict__ s, const float* __restrict__ bias, int topk,
                                           int renormalize, float routed_scaling_factor,
                                           float* __restrict__ out_values, int32_t* __restrict__ out_indices,
                                           int32_t* s_ids) {
  constexpr int E = C * kWarp;
  const int lane = threadIdx.x % kWarp;
  uint64_t keys[C];
  float sel[C];
#pragma unroll
  for (int c = 0; c < C; ++c) {
    const int e = c * kWarp + lane;
    const float in = s[e];
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
  if (lane < topk && lane_expert >= 0 && lane_expert < E) lane_unbiased = lane_score - bias[lane_expert];
  const bool finite_selection = lane < topk && lane_score != kInvalid;
  lane_unbiased = finite_selection ? lane_unbiased : 0.0f;
  const unsigned finite_mask = __ballot_sync(0xffffffffu, finite_selection);
  // cooperative_groups::reduce(tile32, v, plus): xor butterfly, offsets 16 .. 1
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
    if (s_ids != nullptr) s_ids[lane] = lane_expert;
  }
}

template <int E>
__global__ void __launch_bounds__(256)
    topk_sigmoid_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int num_tokens,
                        int topk, int renormalize, float routed_scaling_factor, float* __restrict__ topk_values,
                        int32_t* __restrict__ topk_indices) {
  const int token = blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp;
  pdl_wait();
  if (token < num_tokens)
    topk_token<E / kWarp>(logits + static_cast<long>(token) * E, bias, topk, renormalize, routed_scaling_factor,
                          topk_values + static_cast<long>(token) * topk,
                          topk_indices + static_cast<long>(token) * topk, nullptr);
  pdl_launch_dependents();
}

// ------------------------------------------------------------------------------------------------
// moe_align_block_size + count_and_sort_expert_tokens (no expert map, 128 experts), one CTA of 128
// threads (thread = expert). Same integers as vLLM's two kernels; within an expert the token slots
// ascend, which is the order stock count_and_sort produces when all ids sit in one warp (numel <= 32;
// the Python guard enforces it).
// ------------------------------------------------------------------------------------------------
constexpr int kAlignExperts = 128;
constexpr int kAlignMaxNumel = 64;

__device__ __forceinline__ void align_e128(const int32_t* s_ids, int numel, int block_size,
                                           int32_t* __restrict__ sorted_ids, int max_num_tokens_padded,
                                           int32_t* __restrict__ expert_ids, int max_num_m_blocks,
                                           int32_t* __restrict__ num_tokens_post_pad, int32_t* s_cum,
                                           int32_t* s_wsum) {
  const int t = threadIdx.x;  // expert
  int c = 0;
  for (int i = 0; i < numel; ++i) c += s_ids[i] == t;
  const int padded = (c + block_size - 1) / block_size * block_size;
  int v = padded;
#pragma unroll
  for (int off = 1; off < kWarp; off <<= 1) {
    const int u = __shfl_up_sync(0xffffffffu, v, off);
    if ((t & (kWarp - 1)) >= off) v += u;
  }
  if ((t & (kWarp - 1)) == kWarp - 1) s_wsum[t / kWarp] = v;
  for (int i = t; i < max_num_tokens_padded; i += kAlignExperts) sorted_ids[i] = numel;
  __syncthreads();
  int base = 0;
  for (int w = 0; w < t / kWarp; ++w) base += s_wsum[w];
  const int incl = base + v;
  const int excl = incl - padded;
  s_cum[t] = excl;
  if (t == kAlignExperts - 1) {
    s_cum[kAlignExperts] = incl;
    num_tokens_post_pad[0] = incl;
  }
  __syncthreads();
  const int total = s_cum[kAlignExperts];
  for (int i = excl; i < incl; i += block_size) expert_ids[i / block_size] = t;
  for (int b = total / block_size + t; b < max_num_m_blocks; b += kAlignExperts) expert_ids[b] = -1;
  if (t < numel) {
    const int e = s_ids[t];
    if (e >= 0 && e < kAlignExperts) {
      int rank = 0;
      for (int q = 0; q < t; ++q) rank += s_ids[q] == e;
      sorted_ids[s_cum[e] + rank] = t;
    }
  }
}

__global__ void __launch_bounds__(kAlignExperts)
    align_kernel(const int32_t* __restrict__ topk_ids, int numel, int block_size, int32_t* __restrict__ sorted_ids,
                 int max_num_tokens_padded, int32_t* __restrict__ expert_ids, int max_num_m_blocks,
                 int32_t* __restrict__ num_tokens_post_pad) {
  __shared__ int32_t s_ids[kAlignMaxNumel];
  __shared__ int32_t s_cum[kAlignExperts + 1];
  __shared__ int32_t s_wsum[kAlignExperts / kWarp];
  pdl_wait();
  if (threadIdx.x < numel) s_ids[threadIdx.x] = topk_ids[threadIdx.x];
  __syncthreads();
  align_e128(s_ids, numel, block_size, sorted_ids, max_num_tokens_padded, expert_ids, max_num_m_blocks,
             num_tokens_post_pad, s_cum, s_wsum);
  pdl_launch_dependents();
}

// top-k of every token (warp per token) and the align of the resulting ids, in one CTA.
__global__ void __launch_bounds__(kAlignExperts)
    route_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int num_tokens, int topk,
                 int renormalize, float routed_scaling_factor, float* __restrict__ topk_values,
                 int32_t* __restrict__ topk_indices, int block_size, int32_t* __restrict__ sorted_ids,
                 int max_num_tokens_padded, int32_t* __restrict__ expert_ids, int max_num_m_blocks,
                 int32_t* __restrict__ num_tokens_post_pad) {
  __shared__ int32_t s_ids[kAlignMaxNumel];
  __shared__ int32_t s_cum[kAlignExperts + 1];
  __shared__ int32_t s_wsum[kAlignExperts / kWarp];
  pdl_wait();
  for (int token = threadIdx.x / kWarp; token < num_tokens; token += kAlignExperts / kWarp)
    topk_token<kAlignExperts / kWarp>(logits + static_cast<long>(token) * kAlignExperts, bias, topk, renormalize,
                                      routed_scaling_factor, topk_values + static_cast<long>(token) * topk,
                                      topk_indices + static_cast<long>(token) * topk, s_ids + token * topk);
  __syncthreads();
  align_e128(s_ids, num_tokens * topk, block_size, sorted_ids, max_num_tokens_padded, expert_ids, max_num_m_blocks,
             num_tokens_post_pad, s_cum, s_wsum);
  pdl_launch_dependents();
}

template <typename Kern, typename... Args>
int launch(Kern kernel, dim3 grid, dim3 block, int pdl, cudaStream_t stream, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaError_t err = cudaLaunchKernelEx(&cfg, kernel, args...);
  return err == cudaSuccess ? 0 : 1000 + static_cast<int>(err);
}

template <int R, int MODE, bool W16 = false, bool X16 = false>
int router_launch(const void* x, long ldx, int M, const void* w, int N, int K, float* y, long ldy, int pdl,
                  const int* vrows, cudaStream_t stream) {
  dim3 grid((N + R - 1) / R);
  dim3 block(kLanes * ((K + kTile - 1) / kTile));
  switch (M) {
    case 4:
      return launch(router_gemm_kernel<R, 4, MODE, W16, X16>, grid, block, pdl, stream, x, ldx, w, K, y, ldy, N, vrows);
    case 5:
      // Follow-on M5 specialization: reuse each independent M4 arithmetic
      // chain; equivalence to the M5 cuBLAS choice requires remote validation.
      return launch(router_gemm_kernel<R, 5, MODE, W16, X16>, grid, block, pdl, stream, x, ldx, w, K, y, ldy, N, vrows);
    default:
      return 2;
  }
}

}  // namespace

extern "C" {

int sk_version() { return SK_VERSION; }

// Loads every kernel into the context (call outside CUDA graph capture).
int sk_init() {
  cudaFuncAttributes a;
  int rc = 0;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<1, 4, 0>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<2, 4, 0>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<4, 4, 0>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<8, 4, 0>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<1, 4, 1>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<2, 4, 1>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<4, 4, 1>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<8, 4, 1>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<1, 5, 0>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<2, 5, 0>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<4, 5, 0>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<8, 5, 0>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<1, 5, 1>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<2, 5, 1>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<4, 5, 1>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<8, 5, 1>) != cudaSuccess;
#ifndef SMALLK_NO_W16
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<1, 4, 0, true>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<2, 4, 0, true>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<4, 4, 0, true>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<8, 4, 0, true>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<1, 5, 0, true>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<2, 5, 0, true>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<4, 5, 0, true>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<8, 5, 0, true>) != cudaSuccess;
#endif
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<1, 4, 0, false, true>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<2, 4, 0, false, true>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<4, 4, 0, false, true>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<8, 4, 0, false, true>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<1, 5, 0, false, true>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<2, 5, 0, false, true>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<4, 5, 0, false, true>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<8, 5, 0, false, true>) != cudaSuccess;
#ifndef SMALLK_NO_W16
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<1, 4, 0, true, true>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<2, 4, 0, true, true>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<4, 4, 0, true, true>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<8, 4, 0, true, true>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<1, 5, 0, true, true>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<2, 5, 0, true, true>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<4, 5, 0, true, true>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, router_gemm_kernel<8, 5, 0, true, true>) != cudaSuccess;
#endif
  rc |= cudaFuncGetAttributes(&a, topk_sigmoid_kernel<128>) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, align_kernel) != cudaSuccess;
  rc |= cudaFuncGetAttributes(&a, route_kernel) != cudaSuccess;
  return rc;
}

// y[M][N] (row stride ldy) = x[M][K] (row stride ldx) . w[N][K]^T; w contiguous; K % 16 == 0, K <= 4096.
// vrows (device int, may be nullptr): rows m >= *vrows get the logits of row max(*vrows, 1) - 1.
int sk_router_gemm_v(const float* x, long ldx, int M, const float* w, int N, int K, float* y, long ldy, int rows,
                     int mode, int pdl, const int* vrows, void* stream) {
  if (K % kLanes != 0 || K > kTile * kMaxTiles || K <= 0 || N <= 0) return 1;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (mode == 0) {
    switch (rows) {
      case 1: return router_launch<1, 0>(x, ldx, M, w, N, K, y, ldy, pdl, vrows, s);
      case 2: return router_launch<2, 0>(x, ldx, M, w, N, K, y, ldy, pdl, vrows, s);
      case 4: return router_launch<4, 0>(x, ldx, M, w, N, K, y, ldy, pdl, vrows, s);
      case 8: return router_launch<8, 0>(x, ldx, M, w, N, K, y, ldy, pdl, vrows, s);
    }
  } else {
    switch (rows) {
      case 1: return router_launch<1, 1>(x, ldx, M, w, N, K, y, ldy, pdl, vrows, s);
      case 2: return router_launch<2, 1>(x, ldx, M, w, N, K, y, ldy, pdl, vrows, s);
      case 4: return router_launch<4, 1>(x, ldx, M, w, N, K, y, ldy, pdl, vrows, s);
      case 8: return router_launch<8, 1>(x, ldx, M, w, N, K, y, ldy, pdl, vrows, s);
    }
  }
  return 3;
}

int sk_router_gemm(const float* x, long ldx, int M, const float* w, int N, int K, float* y, long ldy, int rows,
                   int mode, int pdl, void* stream) {
  return sk_router_gemm_v(x, ldx, M, w, N, K, y, ldy, rows, mode, pdl, nullptr, stream);
}

#ifndef SMALLK_NO_W16
// sk_router_gemm_v with the gate weight read from w16 = bf16(w), which the caller built only when every
// value of w is exactly a bf16 value (bit-identical result; fma chains only, mode 0). vrows may be nullptr.
int sk_router_gemm_w16(const float* x, long ldx, int M, const void* w16, int N, int K, float* y, long ldy, int rows,
                       int pdl, const int* vrows, void* stream) {
  if (K % kLanes != 0 || K > kTile * kMaxTiles || K <= 0 || N <= 0) return 1;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  switch (rows) {
    case 1: return router_launch<1, 0, true>(x, ldx, M, w16, N, K, y, ldy, pdl, vrows, s);
    case 2: return router_launch<2, 0, true>(x, ldx, M, w16, N, K, y, ldy, pdl, vrows, s);
    case 4: return router_launch<4, 0, true>(x, ldx, M, w16, N, K, y, ldy, pdl, vrows, s);
    case 8: return router_launch<8, 0, true>(x, ldx, M, w16, N, K, y, ldy, pdl, vrows, s);
  }
  return 3;
}
#endif  // SMALLK_NO_W16: without it the Python side keeps the fp32 read (no sk_router_gemm_w16 symbol)

int sk_router_gemm_x16(const void* x, long ldx, int M, const void* w, int N, int K, float* y, long ldy, int rows,
                       int pdl, const int* vrows, int w16_flag, void* stream) {
  if (K % kLanes != 0 || K > kTile * kMaxTiles || K <= 0 || N <= 0) return 1;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (w16_flag) {
#ifndef SMALLK_NO_W16
    switch (rows) {
      case 1: return router_launch<1, 0, true, true>(x, ldx, M, w, N, K, y, ldy, pdl, vrows, s);
      case 2: return router_launch<2, 0, true, true>(x, ldx, M, w, N, K, y, ldy, pdl, vrows, s);
      case 4: return router_launch<4, 0, true, true>(x, ldx, M, w, N, K, y, ldy, pdl, vrows, s);
      case 8: return router_launch<8, 0, true, true>(x, ldx, M, w, N, K, y, ldy, pdl, vrows, s);
    }
#else
    return 4;
#endif
  } else {
    switch (rows) {
      case 1: return router_launch<1, 0, false, true>(x, ldx, M, w, N, K, y, ldy, pdl, vrows, s);
      case 2: return router_launch<2, 0, false, true>(x, ldx, M, w, N, K, y, ldy, pdl, vrows, s);
      case 4: return router_launch<4, 0, false, true>(x, ldx, M, w, N, K, y, ldy, pdl, vrows, s);
      case 8: return router_launch<8, 0, false, true>(x, ldx, M, w, N, K, y, ldy, pdl, vrows, s);
    }
  }
  return 3;
}

// logits [M][E] fp32 contiguous, bias [E] fp32; values [M][topk] fp32, indices [M][topk] int32.
int sk_topk_sigmoid(const float* logits, const float* bias, int M, int E, int topk, int renormalize, float scale,
                    float* values, int32_t* indices, int warps_per_cta, int pdl, void* stream) {
  if (E != 128 || topk < 1 || topk > kWarp || topk > E || M <= 0) return 1;
  if (warps_per_cta < 1 || warps_per_cta > 8) return 2;
  dim3 grid((M + warps_per_cta - 1) / warps_per_cta);
  dim3 block(warps_per_cta * kWarp);
  return launch(topk_sigmoid_kernel<128>, grid, block, pdl, static_cast<cudaStream_t>(stream), logits, bias, M,
                topk, renormalize, scale, values, indices);
}

int sk_align(const int32_t* topk_ids, int numel, int num_experts, int block_size, int32_t* sorted_ids,
             int max_num_tokens_padded, int32_t* expert_ids, int max_num_m_blocks, int32_t* num_tokens_post_pad,
             int pdl, void* stream) {
  if (numel <= 0 || numel > kAlignMaxNumel || num_experts != kAlignExperts || block_size <= 0) return 1;
  return launch(align_kernel, dim3(1), dim3(kAlignExperts), pdl, static_cast<cudaStream_t>(stream), topk_ids, numel,
                block_size, sorted_ids, max_num_tokens_padded, expert_ids, max_num_m_blocks, num_tokens_post_pad);
}

int sk_route(const float* logits, const float* bias, int M, int E, int topk, int renormalize, float scale,
             float* values, int32_t* indices, int block_size, int32_t* sorted_ids, int max_num_tokens_padded,
             int32_t* expert_ids, int max_num_m_blocks, int32_t* num_tokens_post_pad, int pdl, void* stream) {
  if (E != kAlignExperts || topk < 1 || topk > kWarp || M <= 0 || M * topk > kAlignMaxNumel || block_size <= 0)
    return 1;
  return launch(route_kernel, dim3(1), dim3(kAlignExperts), pdl, static_cast<cudaStream_t>(stream), logits, bias, M,
                topk, renormalize, scale, values, indices, block_size, sorted_ids, max_num_tokens_padded, expert_ids,
                max_num_m_blocks, num_tokens_post_pad);
}

}  // extern "C"

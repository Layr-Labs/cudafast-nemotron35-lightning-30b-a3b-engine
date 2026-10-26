// sctopk: the draft-head screen's per-chunk top-k (nemotron_h_mtp_screen._chunk_topk) as one CUDA
// kernel per call, writing the same ids in the same order.
//
// _chunk_topk(scores [C, N] fp32, src, out [C, K] int64, N, K, GATHER) returns, per chunk c, the
// indices torch.topk(scores[c], K, sorted=False) returns, in its order (CUDA sbtopk::gatherTopK):
// every key above the K-th largest in ascending index order, then the K-th largest's ties in
// ascending index order until K are taken; written as c * N + i, or src[c, i] with GATHER. Keys are
// torch's TopKTypeConfig<float>::convert (order-preserving unsigned, -0 below +0, every NaN the
// largest). A stopped draft pass (gate[0] != 0, GATED) takes the first K ids in index order.
//
// Here: one CTA per chunk. The K-th largest key t is found by an exact 8-bit radix select (four
// shared-memory histograms over the keys that still match the selected prefix), then every
// element's output slot is its rank among the keys above t (or n_gt + its rank among the ties) in
// index order: two block-wide exclusive scans over the threads' contiguous index ranges. Integers
// only, so the selection, the slots and the written ids are the original kernel's.

#include <cuda_runtime.h>

#include <cstdint>

#define SCT_VERSION 1

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

// nemotron_h_mtp_screen._topk_key, as uint32.
__device__ __forceinline__ uint32_t topk_key(float v) {
  const uint32_t x = __float_as_uint(v);
  uint32_t key = (x & 0x80000000u) ? (x ^ 0xFFFFFFFFu) : (x ^ 0x80000000u);
  if (v != v) key = 0xFFFFFFFFu;
  return key;
}

// Block-wide exclusive scan of two counters (a, b) over THREADS threads in thread order.
template <int THREADS>
__device__ __forceinline__ void block_scan2(uint32_t a, uint32_t b, uint32_t& ea, uint32_t& eb,
                                            uint32_t& ta, uint32_t& tb, uint32_t* sh) {
  constexpr int WARPS = THREADS / 32;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint32_t ia = a, ib = b;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t xa = __shfl_up_sync(0xFFFFFFFFu, ia, o);
    const uint32_t xb = __shfl_up_sync(0xFFFFFFFFu, ib, o);
    if (lane >= o) { ia += xa; ib += xb; }
  }
  if (lane == 31) { sh[warp] = ia; sh[WARPS + warp] = ib; }
  __syncthreads();
  if (warp == 0) {
    uint32_t wa = lane < WARPS ? sh[lane] : 0u, wb = lane < WARPS ? sh[WARPS + lane] : 0u;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const uint32_t xa = __shfl_up_sync(0xFFFFFFFFu, wa, o);
      const uint32_t xb = __shfl_up_sync(0xFFFFFFFFu, wb, o);
      if (lane >= o) { wa += xa; wb += xb; }
    }
    if (lane < WARPS) { sh[2 * WARPS + lane] = wa; sh[3 * WARPS + lane] = wb; }
  }
  __syncthreads();
  const uint32_t pa = warp > 0 ? sh[2 * WARPS + warp - 1] : 0u;
  const uint32_t pb = warp > 0 ? sh[3 * WARPS + warp - 1] : 0u;
  ea = pa + ia - a;
  eb = pb + ib - b;
  ta = sh[2 * WARPS + WARPS - 1];
  tb = sh[3 * WARPS + WARPS - 1];
}

template <int N, int K, int THREADS, bool GATHER>
__global__ void __launch_bounds__(THREADS) chunk_topk_kernel(const float* __restrict__ scores,
                                                             const long long* __restrict__ src,
                                                             long long* __restrict__ out,
                                                             const int* __restrict__ gate) {
  static_assert(N % THREADS == 0 && THREADS % 32 == 0 && THREADS <= 1024, "shape");
  constexpr int E = N / THREADS;  // contiguous indices per thread
  constexpr int WARPS = THREADS / 32;
  __shared__ uint32_t hist[256];
  __shared__ uint32_t scan_sh[4 * 32];
  __shared__ uint32_t sel[2];  // selected digit, remaining count

  pdl_wait();
  pdl_launch_dependents();
  const int c = blockIdx.x;
  const int t = threadIdx.x;
  const long long base_id = static_cast<long long>(c) * N;
  if (gate != nullptr && *gate != 0) {
    // Stopped draft pass: the first K ids in ascending index order.
    for (int j = t; j < K; j += THREADS)
      out[static_cast<long long>(c) * K + j] = GATHER ? src[base_id + j] : base_id + j;
    return;
  }
  uint32_t key[E];
#pragma unroll
  for (int e = 0; e < E; ++e) key[e] = topk_key(scores[base_id + t * E + e]);

  // Exact 8-bit radix select of the K-th largest key.
  uint32_t prefix = 0u, mask = 0u;
  uint32_t left = K;
#pragma unroll 1
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int b = t; b < 256; b += THREADS) hist[b] = 0u;
    __syncthreads();
#pragma unroll
    for (int e = 0; e < E; ++e)
      if ((key[e] & mask) == prefix) atomicAdd(&hist[(key[e] >> shift) & 255u], 1u);
    __syncthreads();
    if (t < 32) {
      // lane l holds bins 8l .. 8l+7; suffix counts from the top bin down
      uint32_t local[8];
      uint32_t s = 0u;
#pragma unroll
      for (int i = 0; i < 8; ++i) { local[i] = hist[8 * t + i]; s += local[i]; }
      // inclusive suffix sum over lanes (lanes above this one)
      uint32_t suf = s;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const uint32_t x = __shfl_down_sync(0xFFFFFFFFu, suf, o);
        if (t + o < 32) suf += x;
      }
      const uint32_t above = suf - s;  // keys in lanes above (higher bins)
      // the lane whose bins contain the left-th key from the top
      const bool mine = above < left && suf >= left;
      const unsigned ball = __ballot_sync(0xFFFFFFFFu, mine);
      if (mine) {
        uint32_t acc = above;
        int pick = 8 * t;
#pragma unroll
        for (int i = 7; i >= 0; --i) {
          if (acc + local[i] >= left) { pick = 8 * t + i; break; }
          acc += local[i];
        }
        sel[0] = static_cast<uint32_t>(pick);
        sel[1] = left - acc;  // keys still to take from the selected bin
      }
      (void)ball;
    }
    __syncthreads();
    const uint32_t pick = sel[0];
    left = sel[1];
    prefix |= pick << shift;
    mask |= 255u << shift;
    __syncthreads();
  }
  const uint32_t thr = prefix;  // the K-th largest key

  uint32_t ngt = 0u, neq = 0u;
#pragma unroll
  for (int e = 0; e < E; ++e) { ngt += key[e] > thr; neq += key[e] == thr; }
  uint32_t egt, eeq, tgt, teq;
  block_scan2<THREADS>(ngt, neq, egt, eeq, tgt, teq, scan_sh);
  (void)teq;
  const uint32_t need_eq = static_cast<uint32_t>(K) - tgt;
  uint32_t rg = egt, re = eeq;
#pragma unroll
  for (int e = 0; e < E; ++e) {
    const int i = t * E + e;
    long long pos = -1;
    if (key[e] > thr) {
      pos = rg++;
    } else if (key[e] == thr) {
      if (re < need_eq) pos = tgt + re;
      ++re;
    }
    if (pos >= 0) out[static_cast<long long>(c) * K + pos] = GATHER ? src[base_id + i] : base_id + i;
  }
}

template <typename F>
int launch(F kernel, int grid, int block, bool pdl, cudaStream_t stream, const float* s,
           const long long* src, long long* out, const int* gate) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(block);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  return static_cast<int>(cudaLaunchKernelEx(&cfg, kernel, s, src, out, gate));
}

}  // namespace

extern "C" {

int sct_version() { return SCT_VERSION; }

// chunk_topk: C chunks of N scores -> K ids each, as nemotron_h_mtp_screen._chunk_topk.
// Supported (N, K): (2048, 1024), (2048, 256), (1024, 256), (256, 32). Returns -1 when not.
int sct_chunk_topk(const void* scores, const void* src, void* out, int C, int N, int K, int gather,
                   const void* gate, int pdl, void* stream) {
  const float* s = static_cast<const float*>(scores);
  const long long* sr = static_cast<const long long*>(src);
  long long* o = static_cast<long long*>(out);
  const int* g = static_cast<const int*>(gate);
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  const bool p = pdl != 0;
#define SCT_CASE(NN, KK, TT)                                                                        \
  if (N == NN && K == KK) {                                                                         \
    return gather ? launch(chunk_topk_kernel<NN, KK, TT, true>, C, TT, p, st, s, sr, o, g)          \
                  : launch(chunk_topk_kernel<NN, KK, TT, false>, C, TT, p, st, s, sr, o, g);        \
  }
  SCT_CASE(2048, 1024, 512)
  SCT_CASE(2048, 256, 512)
  SCT_CASE(1024, 256, 256)
  SCT_CASE(256, 32, 128)
#undef SCT_CASE
  return -1;
}

}  // extern "C"

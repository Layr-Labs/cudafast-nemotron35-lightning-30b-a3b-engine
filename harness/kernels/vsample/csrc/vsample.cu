
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#define VS_VERSION 2

namespace {

constexpr int kThreads = 256;
constexpr int kVec = 8;                         // bf16 per 16-byte load
constexpr int kLoads = 2;                       // 16-byte loads per thread per chunk
constexpr int kChunk = kThreads * kVec * kLoads;  // 4096 elements per CTA
constexpr int kMaxRows = 9;                     // D <= 8
constexpr int kMaxChunks = 64;                  // V <= 262144
constexpr int kVrowsAll = 1 << 30;

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

__device__ __forceinline__ bool wins(float a, int ia, float b, int ib) {
  const bool na = a != a, nb = b != b;
  if (na) return nb ? ia < ib : true;
  if (nb) return false;
  return (a == b) ? ia < ib : a > b;
}

struct Best {
  float v;
  int i;
};

__device__ __forceinline__ void take(Best& b, float v, int i) {
  if (wins(v, i, b.v, b.i)) { b.v = v; b.i = i; }
}

__device__ __forceinline__ Best warp_best(Best b) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float v = __shfl_xor_sync(0xffffffffu, b.v, o);
    const int i = __shfl_xor_sync(0xffffffffu, b.i, o);
    take(b, v, i);
  }
  return b;
}

__device__ __forceinline__ float bf16_bits_to_float(unsigned short h) {
  return __uint_as_float(static_cast<unsigned int>(h) << 16);
}

__global__ void __launch_bounds__(kThreads)
    greedy_verify_kernel(const unsigned short* __restrict__ logits, long ld, int V, int C,
                         const int* __restrict__ target_idx, const int* __restrict__ bonus_idx,
                         const int* __restrict__ cu_num_draft, const int* __restrict__ draft,
                         int* __restrict__ vrows, int* __restrict__ out, int out_len,
                         int* __restrict__ num_accepted, int R, float* __restrict__ pval,
                         int* __restrict__ pidx, unsigned int* __restrict__ counter) {
  const int c = blockIdx.x;
  const int r = blockIdx.y;
  const int t = threadIdx.x;
  __shared__ Best wbest[kThreads / 32];
  __shared__ bool is_last;
  __shared__ int row_arg[kMaxRows];

  pdl_wait();
  Best b{-__int_as_float(0x7f800000), 0x7fffffff};  // (-inf, +big): loses to every element
  const unsigned short* row = logits + static_cast<long>(r) * ld;
  const int base = c * kChunk;
#pragma unroll
  for (int l = 0; l < kLoads; ++l) {
    const int e0 = base + (l * kThreads + t) * kVec;
    if (e0 + kVec <= V) {
      const uint4 q = __ldcg(reinterpret_cast<const uint4*>(row + e0));
      const unsigned int w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
      for (int h = 0; h < 4; ++h) {
        take(b, bf16_bits_to_float(static_cast<unsigned short>(w[h] & 0xffffu)), e0 + 2 * h);
        take(b, bf16_bits_to_float(static_cast<unsigned short>(w[h] >> 16)), e0 + 2 * h + 1);
      }
    } else {
      for (int e = e0; e < V && e < e0 + kVec; ++e) take(b, bf16_bits_to_float(row[e]), e);
    }
  }
  b = warp_best(b);
  if ((t & 31) == 0) wbest[t >> 5] = b;
  __syncthreads();
  if (t < 32) {
    Best x = t < kThreads / 32 ? wbest[t] : Best{-__int_as_float(0x7f800000), 0x7fffffff};
    x = warp_best(x);
    if (t == 0) {
      pval[r * C + c] = x.v;
      pidx[r * C + c] = x.i;
      __threadfence();
      const unsigned int done = atomicAdd(counter, 1u);
      is_last = done == static_cast<unsigned int>(R * C - 1);
    }
  }
  __syncthreads();
  if (!is_last) return;
  __threadfence();
  const int w = t >> 5, lane = t & 31;
  for (int rr = w; rr < R; rr += kThreads / 32) {
    Best x{-__int_as_float(0x7f800000), 0x7fffffff};
    for (int cc = lane; cc < C; cc += 32) take(x, __ldcg(pval + rr * C + cc), __ldcg(pidx + rr * C + cc));
    x = warp_best(x);
    if (lane == 0) row_arg[rr] = x.i;
  }
  __syncthreads();
  if (t == 0) {
    *counter = 0u;  // ready for the next launch (stream order)
    const int vr = vrows == nullptr ? 0 : __ldcg(vrows);
    for (int p = 0; p < out_len; ++p) out[p] = -1;
    const int nd = max(0, min(__ldcg(cu_num_draft), out_len - 1));
    bool rejected = false;
    int written = 0;
    for (int pos = 0; pos < nd; ++pos) {
      int ti = __ldcg(target_idx + pos);
      if (ti < 0) ti += R;  // torch indexing wraps negative indices
      ti = max(0, min(ti, R - 1));
      const int tok = row_arg[ti];
      int d = __ldcg(draft + pos);
      if (vrows != nullptr && pos >= vr - 1) d = -1;
      out[pos] = tok;
      ++written;
      if (d != tok) { rejected = true; break; }
    }
    if (!rejected) {
      int bi = __ldcg(bonus_idx);
      if (bi < 0) bi += R;
      bi = max(0, min(bi, R - 1));
      out[nd] = row_arg[bi];
      ++written;
    }
    if (vrows != nullptr) *vrows = kVrowsAll;
    if (num_accepted != nullptr) *num_accepted = written;
    __threadfence();
  }
  __syncthreads();
  pdl_launch_dependents();
}

}  // namespace

extern "C" {

int vs_version() { return VS_VERSION; }

int vs_max_rows() { return kMaxRows; }

int vs_max_vocab() { return kMaxChunks * kChunk; }

int vs_chunk() { return kChunk; }

int vs_greedy_verify(const void* logits, long ld, int R, int V, const int* target_idx,
                     const int* bonus_idx, const int* cu_num_draft, const int* draft, int* vrows,
                     int* out, int out_len, int* num_accepted, float* pval, int* pidx,
                     unsigned int* counter, int pdl, cudaStream_t stream) {
  if (R < 1 || R > kMaxRows || V < 1 || V > kMaxChunks * kChunk || (V % kVec) != 0 || (ld % kVec) != 0 ||
      out_len < 1 || out_len > kMaxRows || (reinterpret_cast<uintptr_t>(logits) & 15u) != 0)
    return 1;
  const int C = (V + kChunk - 1) / kChunk;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(C, R, 1);
  cfg.blockDim = dim3(kThreads, 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  const cudaError_t e = cudaLaunchKernelEx(&cfg, greedy_verify_kernel,
                                           static_cast<const unsigned short*>(logits), ld, V, C, target_idx,
                                           bonus_idx, cu_num_draft, draft, vrows, out, out_len, num_accepted, R,
                                           pval, pidx, counter);
  return e == cudaSuccess ? 0 : 100 + static_cast<int>(e);
}

}  // extern "C"

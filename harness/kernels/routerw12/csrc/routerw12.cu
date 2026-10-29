// SPDX-License-Identifier: Apache-2.0
// Optional exact W16/W12 packet for the original FP32-input smallk router.
#include <cuda_runtime.h>
#include <cstdint>

namespace rw12 {
constexpr int K = 2688, N = 128, LANES = 16, TILE = 256, MAX_TILES = 16;
constexpr int HEADER = N * sizeof(uint32_t), CODED = 4096, RAW = K * 2;
constexpr int SM_BYTES = K, EX_BYTES = K / 2, BASE_START = SM_BYTES + EX_BYTES;
constexpr int THREADS = LANES * ((K + TILE - 1) / TILE);

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_signal() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

struct Row {
  const uint8_t* data;
  bool coded;
  bool valid;
  unsigned mask;
  int r, j, b0, b1;
};

// Exactly one original sixteen-lane subgroup owns one K256 tile. A physical
// warp may contain two subgroups; the final physical warp contains only16 lanes.
__device__ __forceinline__ Row row(const uint8_t* packet, int bytes, int expert) {
  const int t = threadIdx.x;
  const unsigned mask = __activemask();
  uint32_t word = 0;
  if ((t & 31) == 0) word = __ldg(reinterpret_cast<const uint32_t*>(packet) + expert);
  word = __shfl_sync(mask, word, 0);
  const uint32_t offset = word & ~1u;
  const bool coded = (word & 1u) != 0;
  const int extent = coded ? CODED : RAW;
  const bool valid = offset >= HEADER && (offset & 127u) == 0
      && offset <= static_cast<uint32_t>(bytes - extent);
  Row result{packet + (valid ? offset : HEADER), coded, valid, mask,
             t & (LANES - 1), t / LANES, 0, 0};
  if (valid && coded) {
    // Block20 is the final K128 block; its subgroup never touches block21.
    if (result.r == 0) result.b0 = __ldg(result.data + BASE_START + result.j * 2);
    result.b0 = __shfl_sync(mask, result.b0, 0, LANES);
    if (result.j < 10) {
      if (result.r == 0) result.b1 = __ldg(result.data + BASE_START + result.j * 2 + 1);
      result.b1 = __shfl_sync(mask, result.b1, 0, LANES);
    }
  }
  return result;
}

// This decoder is shared by exhaustive package admission and actual GEMV loads.
// No numerical conversion is used to reconstruct the original BF16 bits.
__device__ __forceinline__ uint16_t bits(const Row& row, int i) {
  const int k = row.j * TILE + row.r + i * LANES;
  if (!row.coded) return __ldg(reinterpret_cast<const uint16_t*>(row.data) + k);
  const unsigned sm = __ldg(row.data + k);
  unsigned ex = 0;
  if ((row.r & 1) == 0) ex = __ldg(row.data + SM_BYTES + k / 2);
  ex = __shfl_sync(row.mask, ex, row.r & ~1, LANES);
  const unsigned code = (ex >> ((row.r & 1) * 4)) & 15u;
  const unsigned base = i < 8 ? row.b0 : row.b1;
  const unsigned exponent = code == 0 ? 0 : base + code - 1;
  return static_cast<uint16_t>(((sm & 128u) << 8) | (exponent << 7) | (sm & 127u));
}

__device__ __forceinline__ float weight(const Row& row, int i) {
  return __uint_as_float(static_cast<unsigned>(bits(row, i)) << 16);
}

__global__ void pack(const uint16_t* __restrict__ source, const uint8_t* __restrict__ bases,
                     uint8_t* __restrict__ packet, int bytes) {
  const int e = blockIdx.x;
  const uint32_t word = reinterpret_cast<const uint32_t*>(packet)[e];
  const uint32_t offset = word & ~1u;
  const bool coded = (word & 1u) != 0;
  const int extent = coded ? CODED : RAW;
  if (offset < HEADER || (offset & 127u) != 0 || offset > static_cast<uint32_t>(bytes - extent)) return;
  uint8_t* dst = packet + offset;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    const uint16_t value = source[e * K + k];
    if (!coded) {
      reinterpret_cast<uint16_t*>(dst)[k] = value;
    } else {
      dst[k] = static_cast<uint8_t>(((value >> 8) & 128u) | (value & 127u));
      if ((k & 1) == 0) {
        const unsigned exp0 = (value >> 7) & 255u;
        const unsigned exp1 = (source[e * K + k + 1] >> 7) & 255u;
        const unsigned base = bases[e * 21 + k / 128];
        const unsigned c0 = exp0 == 0 ? 0 : exp0 - base + 1;
        const unsigned c1 = exp1 == 0 ? 0 : exp1 - base + 1;
        dst[SM_BYTES + k / 2] = static_cast<uint8_t>(c0 | (c1 << 4));
      }
      if ((k & 127) == 0) dst[BASE_START + k / 128] = bases[e * 21 + k / 128];
    }
  }
}

__global__ void decode(const uint8_t* __restrict__ packet, int bytes,
                       uint16_t* __restrict__ result) {
  const int expert = blockIdx.x;
  const Row rd = row(packet, bytes, expert);
  if (!rd.valid) return;
  const int len = rd.j < 10 ? 16 : 8;
  for (int i = 0; i < len; ++i)
    result[expert * K + rd.j * TILE + rd.r + i * LANES] = bits(rd, i);
}

// Original smallk R1 FP32-input MODE0 chains. Only the weight loader differs.
// All rows are computed, including rows the unchanged output rule substitutes.
template <int M>
__global__ void __launch_bounds__(LANES * MAX_TILES)
router(const float* __restrict__ x, long ldx, const uint8_t* __restrict__ packet,
       int bytes, float* __restrict__ y, long ldy, const int* __restrict__ vrows) {
  const int t = threadIdx.x, r = t & (LANES - 1), j = t / LANES;
  constexpr int J = (K + TILE - 1) / TILE;
  __shared__ float part[M][MAX_TILES][LANES];
  __shared__ float lane_sum[M][LANES];
  pdl_wait();
  const int vr = vrows == nullptr ? M : *vrows;
  const Row rd = row(packet, bytes, blockIdx.x);
  if (!rd.valid) return;  // immutable per-CTA metadata: all threads take this together
  if (j < J) {
    const int len = min(LANES, (K - j * TILE) / LANES);
    const int base = j * TILE + r;
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.0f;
    if (len == LANES) {
      float wv[LANES];
#pragma unroll
      for (int i = 0; i < LANES; ++i) wv[i] = weight(rd, i);
#pragma unroll
      for (int i = 0; i < LANES; ++i) {
        float xv[M];
#pragma unroll
        for (int m = 0; m < M; ++m) xv[m] = __ldg(x + m * ldx + base + i * LANES);
#pragma unroll
        for (int m = 0; m < M; ++m) acc[m] = __fmaf_rn(wv[i], xv[m], acc[m]);
      }
    } else {
      for (int i = 0; i < len; ++i) {
        float xv[M];
#pragma unroll
        for (int m = 0; m < M; ++m) xv[m] = __ldg(x + m * ldx + base + i * LANES);
        const float wa = weight(rd, i);
#pragma unroll
        for (int m = 0; m < M; ++m) acc[m] = __fmaf_rn(wa, xv[m], acc[m]);
      }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) part[m][j][r] = acc[m];
  }
  __syncthreads();
  for (int q = t; q < M * LANES; q += blockDim.x) {
    const int m = q / LANES, lane = q % LANES;
    float s = part[m][0][lane];
    for (int jj = 1; jj < J; ++jj) s = __fadd_rn(s, part[m][jj][lane]);
    lane_sum[m][lane] = s;
  }
  __syncthreads();
  if (t < M) {
    const int src = min(t, max(vr, 1) - 1);
    float s = lane_sum[src][0];
#pragma unroll
    for (int lane = 1; lane < LANES; ++lane) s = __fadd_rn(s, lane_sum[src][lane]);
    y[t * ldy + blockIdx.x] = s;
  }
  pdl_signal();
}

template <typename Kernel, typename... Args>
int launch(Kernel kernel, int threads, int pdl, cudaStream_t stream, Args... args) {
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(N);
  config.blockDim = dim3(threads);
  config.stream = stream;
  cudaLaunchAttribute attr = {};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  config.attrs = &attr;
  config.numAttrs = 1;
  const cudaError_t error = cudaLaunchKernelEx(&config, kernel, args...);
  return error == cudaSuccess ? 0 : 1000 + static_cast<int>(error);
}
}  // namespace rw12

extern "C" {
int rw12_version() { return 1; }
int rw12_init() {
  int device = 0;
  cudaDeviceProp p = {};
  if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&p, device) != cudaSuccess) return 1;
  if (p.major != 12 || p.minor != 1) return 2;
  cudaFuncAttributes a = {}, b = {}, c = {}, d = {};
  if (cudaFuncGetAttributes(&a, rw12::router<5>) != cudaSuccess
      || cudaFuncGetAttributes(&b, rw12::router<6>) != cudaSuccess
      || cudaFuncGetAttributes(&c, rw12::pack) != cudaSuccess
      || cudaFuncGetAttributes(&d, rw12::decode) != cudaSuccess) return 3;
  return a.maxThreadsPerBlock >= rw12::THREADS && b.maxThreadsPerBlock >= rw12::THREADS
      && c.maxThreadsPerBlock >= 256 && d.maxThreadsPerBlock >= rw12::THREADS ? 0 : 4;
}
int rw12_pack(const void* source, const void* bases, void* packet, int bytes, void* stream) {
  if (!source || !bases || !packet || bytes < rw12::HEADER + 128 * rw12::CODED
      || bytes >= rw12::N * rw12::RAW) return 1;
  return rw12::launch(rw12::pack, 256, 0, reinterpret_cast<cudaStream_t>(stream),
      static_cast<const uint16_t*>(source), static_cast<const uint8_t*>(bases),
      static_cast<uint8_t*>(packet), bytes);
}
int rw12_decode(const void* packet, int bytes, void* result, void* stream) {
  if (!packet || !result || bytes < rw12::HEADER + 128 * rw12::CODED
      || bytes >= rw12::N * rw12::RAW) return 1;
  return rw12::launch(rw12::decode, rw12::THREADS, 0, reinterpret_cast<cudaStream_t>(stream),
      static_cast<const uint8_t*>(packet), bytes, static_cast<uint16_t*>(result));
}
int rw12_router(const float* x, long ldx, int m, const void* packet, int bytes,
                float* y, long ldy, int pdl, const int* vrows, void* stream) {
  if (!x || !packet || !y || ldx < rw12::K || ldy < rw12::N
      || bytes < rw12::HEADER + 128 * rw12::CODED || bytes >= rw12::N * rw12::RAW) return 1;
  auto s = reinterpret_cast<cudaStream_t>(stream);
  if (m == 5) return rw12::launch(rw12::router<5>, rw12::THREADS, pdl, s,
      x, ldx, static_cast<const uint8_t*>(packet), bytes, y, ldy, vrows);
  if (m == 6) return rw12::launch(rw12::router<6>, rw12::THREADS, pdl, s,
      x, ldx, static_cast<const uint8_t*>(packet), bytes, y, ldy, vrows);
  return 2;
}
}

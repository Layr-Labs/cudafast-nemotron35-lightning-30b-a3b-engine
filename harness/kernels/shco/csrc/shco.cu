// shco: probe kernels for running the shared expert beside humming's routed up projection
// in the 496-token prefill (harness/kernels/shco/tests/step1.py).
//
// dummy_kernel<MODE> is a small-footprint, compute-bound stand-in for the shared expert: it
// issues the shared expert's number of bf16 mma.m16n8k16 (fp32 accumulate) in Marlin's
// large-M inner-loop shape (4 m16 blocks x 2 n8 halves = 8 MMAs per k16 step, 8 independent
// fp32 accumulator chains per warp).
//   MODE 0: MMAs only (operands stay in registers).
//   MODE 1: + per k16 step four ldmatrix.x4 of A fragments from a 16 KB shared buffer and the
//           dequant of one fp4 word into the two B fragments (lop3 + shifts + bf16x2 multiplies
//           by a scale, roughly Marlin's dequant<bf16, fe2m1f> + scale step).
//   MODE 2: MODE 1 + the shared expert's weight bytes streamed from global memory (one 16-byte
//           ld.global.nc per lane every ld_every steps, consumed ld_every steps later).
// No result is meaningful; the accumulators are written only under an impossible condition.
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t lop3_ea(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;  // (a & b) | c
  asm volatile("lop3.b32 %0, %1, %2, %3, 0xEA;\n" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t mul_bf16x2(uint32_t a, uint32_t b) {
  uint32_t r;
  asm volatile("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ uint4 ld_stream(const uint4 *p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

constexpr int kSmemWords = 4096;  // 16 KB

template <int MODE>
__global__ void __launch_bounds__(128) dummy_kernel(int steps, int ld_every, const uint4 *__restrict__ wbuf,
                                                     long long wvecs, const uint32_t *__restrict__ seed,
                                                     float *__restrict__ out) {
  __shared__ __align__(16) uint32_t smem[MODE >= 1 ? kSmemWords : 4];
  const int lane = threadIdx.x & 31;
  const int warps_per_cta = blockDim.x >> 5;
  const int gwarp = blockIdx.x * warps_per_cta + (threadIdx.x >> 5);
  const int nwarps = gridDim.x * warps_per_cta;
  uint32_t a[4][4];
  uint32_t b[2][2];
  float acc[4][2][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) a[i][j] = seed[(lane * 16 + i * 4 + j) & 1023];
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[i][h][j] = 0.f;
  }
  b[0][0] = seed[(lane + 512) & 1023];
  b[0][1] = seed[(lane + 544) & 1023];
  b[1][0] = seed[(lane + 576) & 1023];
  b[1][1] = seed[(lane + 608) & 1023];
  uint32_t q = seed[(lane + 640) & 1023] | 1u;
  const uint32_t scale = seed[(lane + 672) & 1023] & 0x3f803f80u;  // positive bf16x2 in [1, 2)
  uint32_t sbase = 0;
  if (MODE >= 1) {
    for (int i = threadIdx.x; i < kSmemWords; i += blockDim.x) smem[i] = seed[i & 1023];
    __syncthreads();
    sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  }
  uint4 pend = make_uint4(0, 0, 0, 0);
  long long wpos = static_cast<long long>(gwarp) * 32 + lane;
  const long long wstride = static_cast<long long>(nwarps) * 32;
  int ld_count = ld_every - 1;  // first load at step 0
  for (int s = 0; s < steps; ++s) {
    if (MODE >= 1) {
      const uint32_t idx = static_cast<uint32_t>(s & 7) * 4;
#pragma unroll
      for (int i = 0; i < 4; ++i) ldsm4(a[i], sbase + (idx + i) * 512 + lane * 16);
      // fp4 word -> four bf16x2 (sign + e2m1 bits placed into a bf16), times the group scale
      const uint32_t m = 0x81c081c0u;
      b[0][0] = mul_bf16x2(lop3_ea(q << 6, m, 0u), scale);
      b[0][1] = mul_bf16x2(lop3_ea(q << 2, m, 0u), scale);
      b[1][0] = mul_bf16x2(lop3_ea(q >> 2, m, 0u), scale);
      b[1][1] = mul_bf16x2(lop3_ea(q >> 6, m, 0u), scale);
      q = q * 1664525u + 1013904223u;
    }
    if (MODE == 2 && ++ld_count == ld_every) {
      ld_count = 0;
      q ^= pend.x ^ pend.y ^ pend.z ^ pend.w;
      if (wpos < wvecs) pend = ld_stream(wbuf + wpos);
      wpos += wstride;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      mma_bf16(acc[i][0], a[i], b[0][0], b[0][1]);
      mma_bf16(acc[i][1], a[i], b[1][0], b[1][1]);
    }
  }
  float t = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int j = 0; j < 4; ++j) t += acc[i][h][j];
  if (t == 1.2345e-30f && q == 7u) out[blockIdx.x * blockDim.x + threadIdx.x] = t;
}

template <int MODE>
const void *dummy_fn() {
  return reinterpret_cast<const void *>(&dummy_kernel<MODE>);
}

const void *fn_of(int mode) {
  switch (mode) {
    case 0: return dummy_fn<0>();
    case 1: return dummy_fn<1>();
    case 2: return dummy_fn<2>();
    default: return nullptr;
  }
}

}  // namespace

extern "C" {

int shco_version() { return 1; }

// out[0..5]: registers per thread, static smem bytes, max threads per block, local bytes,
// binary version, max dynamic smem
int shco_dummy_attrs(int mode, int *out) {
  const void *f = fn_of(mode);
  if (f == nullptr) return -1;
  cudaFuncAttributes a;
  cudaError_t e = cudaFuncGetAttributes(&a, f);
  if (e != cudaSuccess) return static_cast<int>(e);
  out[0] = a.numRegs;
  out[1] = static_cast<int>(a.sharedSizeBytes);
  out[2] = a.maxThreadsPerBlock;
  out[3] = static_cast<int>(a.localSizeBytes);
  out[4] = a.binaryVersion;
  out[5] = a.maxDynamicSharedSizeBytes;
  return 0;
}

int shco_dummy_occupancy(int mode, int threads, int *blocks_per_sm) {
  const void *f = fn_of(mode);
  if (f == nullptr) return -1;
  return static_cast<int>(cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks_per_sm, f, threads, 0));
}

int shco_dummy(int mode, int grid, int threads, int steps, int ld_every, const void *wbuf, long long wvecs,
               const void *seed, void *out, void *stream) {
  if (ld_every < 1) ld_every = 1;
  auto s = static_cast<cudaStream_t>(stream);
  const auto *w = static_cast<const uint4 *>(wbuf);
  const auto *sd = static_cast<const uint32_t *>(seed);
  auto *o = static_cast<float *>(out);
  switch (mode) {
    case 0: dummy_kernel<0><<<grid, threads, 0, s>>>(steps, ld_every, w, wvecs, sd, o); break;
    case 1: dummy_kernel<1><<<grid, threads, 0, s>>>(steps, ld_every, w, wvecs, sd, o); break;
    case 2: dummy_kernel<2><<<grid, threads, 0, s>>>(steps, ld_every, w, wvecs, sd, o); break;
    default: return -1;
  }
  return static_cast<int>(cudaGetLastError());
}

}  // extern "C"

// The Marlin-exact shared-expert replica (one translation unit, one library).
#include "shx.cu"

// Native-MTP pkslots producer/consumer warp split. Original arithmetic from pkslots.
// Four producer warps decode chunk c+1 to an owner-transposed shared BF16 plane
// while the original four consumer warps accumulate chunk c. No persistent format change.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
namespace pksplit {
constexpr int H = 2688, I = 1856;
__device__ __forceinline__ uint4 ld_w128(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint2 ld_w64(const void* p) {
  uint2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];\n" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}

struct Tile {
  uint32_t words[2][8][128];
  float values[2][128];
};
static_assert(sizeof(Tile) == 9216, "pksplit two-rep tile layout");

template <bool PDL>
__global__ __launch_bounds__(256) void slots_kernel(
    const int32_t* __restrict__ indices, const __nv_bfloat16* __restrict__ values,
    const int32_t* __restrict__ counts, const uint8_t* __restrict__ RSM,
    const uint8_t* __restrict__ REX, const uint8_t* __restrict__ RBASE,
    const int32_t* __restrict__ RRI, const uint16_t* __restrict__ RRAW,
    const uint8_t* __restrict__ SSM, const uint8_t* __restrict__ SEX,
    const uint8_t* __restrict__ SBASE, const int32_t* __restrict__ SRI,
    const uint16_t* __restrict__ SRAW, const int32_t* __restrict__ ids, float* __restrict__ part,
    int topk) {
#ifndef PKSPLIT_NO_PDL
  if constexpr (PDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" :::);
  }
#endif
  __shared__ Tile tile[2];
  __shared__ float red[4][64];
  const int slot = blockIdx.y;
  const bool producer = threadIdx.x >= 128;
  const int owner = threadIdx.x & 127;
  const int lane = owner & 31, warp = owner >> 5;
  const int ct = lane & 3, lr = lane >> 2;
  const int n0 = blockIdx.x * 64 + 16 * ct;
  const int count = counts[slot];
  if (count == 0) {
    if (threadIdx.x < 64) part[(long long)slot * H + blockIdx.x * 64 + threadIdx.x] = 0.0f;
    return; // original empty slot sums only +0*+0 from a +0 accumulator
  }
  const bool routed = slot < topk;
  const long long base = routed ? (long long)ids[slot] * I * H : (long long)(slot - topk) * I * H;
  const uint8_t* SM = routed ? RSM : SSM;
  const uint8_t* EX = routed ? REX : SEX;
  const uint8_t* BASE = routed ? RBASE : SBASE;
  const int32_t* RI = routed ? RRI : SRI;
  const uint16_t* RAW = routed ? RRAW : SRAW;
  const int32_t* ind = indices + (long long)slot * I;
  const __nv_bfloat16* val = values + (long long)slot * I;

  auto produce = [&](int which, int start) {
#pragma unroll
    for (int rep = 0; rep < 2; ++rep) {
      const int pos = start + 32 * rep + 8 * warp + lr;
      uint32_t q[8] = {};
      float v = 0.0f;
      if (pos < count) {
        const int k = ind[pos];
        v = __bfloat162float(val[pos]);
        const long long index = base + (long long)k * H + n0;
        const int rawid = RI[index >> 7];
        if (rawid >= 0) {
          const uint16_t* r = RAW + (long long)rawid * 128 + (index & 127);
          const uint4 a = *reinterpret_cast<const uint4*>(r);
          const uint4 b = *reinterpret_cast<const uint4*>(r + 8);
          q[0]=a.x; q[1]=a.y; q[2]=a.z; q[3]=a.w;
          q[4]=b.x; q[5]=b.y; q[6]=b.z; q[7]=b.w;
        } else {
          const uint4 sm = ld_w128(SM + index);
          const uint2 ex = ld_w64(EX + (index >> 1));
          const uint32_t bs = BASE[index >> 7];
          const uint32_t smw[4] = {sm.x, sm.y, sm.z, sm.w};
          const uint32_t exw[2] = {ex.x, ex.y};
#pragma unroll
          for (int j = 0; j < 16; ++j) {
            const uint32_t s = (smw[j >> 2] >> (8 * (j & 3))) & 0xFFu;
            const uint32_t code = (exw[j >> 3] >> (4 * (j & 7))) & 15u;
            const uint32_t e = code ? (bs + code - 1u) & 0xFFFFu : 0u;
            const uint32_t bits = (((s & 0x80u) << 8) | (e << 7) | (s & 0x7Fu)) & 0xFFFFu;
            q[j >> 1] |= bits << (16 * (j & 1));
          }
        }
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) tile[which].words[rep][j][owner] = q[j];
      tile[which].values[rep][owner] = v;
    }
  };
  // Every CTA thread reaches every barrier, including producer-only warps.
  // Warm stage0. At loop c, producers fill the other stage while consumers
  // read stage c; the single CTA barrier both publishes c+1 and releases c.
  if (producer) produce(0, 0);
  __syncthreads();
  float acc[2][16];
#pragma unroll
  for (int rep = 0; rep < 2; ++rep)
#pragma unroll
    for (int j = 0; j < 16; ++j) acc[rep][j] = 0.0f;
  const int steps = count > 0 ? (count + 63) / 64 : 1;
  for (int step = 0; step < steps; ++step) {
    const int current = step & 1;
    if (producer) {
      if (step + 1 < steps) produce(current ^ 1, (step + 1) * 64);
    } else {
#pragma unroll
      for (int rep = 0; rep < 2; ++rep) {
        const float v = tile[current].values[rep][owner];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const uint32_t q = tile[current].words[rep][j][owner];
          const float lo = __uint_as_float(q << 16);
          const float hi = __uint_as_float(q & 0xFFFF0000u);
          acc[rep][2*j] = __fadd_rn(acc[rep][2*j], __fmul_rn(lo, v));
          acc[rep][2*j+1] = __fadd_rn(acc[rep][2*j+1], __fmul_rn(hi, v));
        }
      }
    }
    __syncthreads();
  }
  if (!producer) {
    float t[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) t[j] = __fadd_rn(acc[0][j], acc[1][j]);
#pragma unroll
    for (int d = 16; d >= 4; d >>= 1)
#pragma unroll
      for (int j = 0; j < 16; ++j) t[j] = __fadd_rn(t[j], __shfl_xor_sync(0xFFFFFFFFu, t[j], d));
    if (lr == 0) {
#pragma unroll
      for (int j = 0; j < 16; ++j) red[warp][16 * ct + j] = t[j];
    }
  }
  __syncthreads();
  if (threadIdx.x < 64) {
    const int c = threadIdx.x;
    const float r = __fadd_rn(__fadd_rn(red[0][c], red[2][c]), __fadd_rn(red[1][c], red[3][c]));
    part[(long long)slot * H + blockIdx.x * 64 + c] = r;
  }
}
} // namespace pksplit
extern "C" {

// part[slots, 2688] = _slots(...) with the (64, 4) tile; slots = topk + 2.
int pksplit_run(const void* indices, const void* values, const void* counts, const void* rsm,
                const void* rex, const void* rbase, const void* rri, const void* rraw,
                const void* ssm, const void* sex, const void* sbase, const void* sri,
                const void* sraw, const void* ids, void* part, int topk, int pdl, void* stream) {
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  const dim3 grid(pksplit::H / 64, topk + 2), block(256);
#ifndef PKSPLIT_NO_PDL
  if (pdl) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = st;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaError_t rc = cudaLaunchKernelEx(
        &cfg, pksplit::slots_kernel<true>, static_cast<const int32_t*>(indices),
        static_cast<const __nv_bfloat16*>(values), static_cast<const int32_t*>(counts),
        static_cast<const uint8_t*>(rsm), static_cast<const uint8_t*>(rex),
        static_cast<const uint8_t*>(rbase), static_cast<const int32_t*>(rri),
        static_cast<const uint16_t*>(rraw), static_cast<const uint8_t*>(ssm),
        static_cast<const uint8_t*>(sex), static_cast<const uint8_t*>(sbase),
        static_cast<const int32_t*>(sri), static_cast<const uint16_t*>(sraw),
        static_cast<const int32_t*>(ids), static_cast<float*>(part), topk);
    if (rc != cudaSuccess) { cudaGetLastError(); return (int)rc; }
    return (int)cudaGetLastError();
  }
#else
  if (pdl) return -1;
#endif
  pksplit::slots_kernel<false><<<grid, block, 0, st>>>(
      static_cast<const int32_t*>(indices), static_cast<const __nv_bfloat16*>(values),
      static_cast<const int32_t*>(counts), static_cast<const uint8_t*>(rsm),
      static_cast<const uint8_t*>(rex), static_cast<const uint8_t*>(rbase),
      static_cast<const int32_t*>(rri), static_cast<const uint16_t*>(rraw),
      static_cast<const uint8_t*>(ssm), static_cast<const uint8_t*>(sex),
      static_cast<const uint8_t*>(sbase), static_cast<const int32_t*>(sri),
      static_cast<const uint16_t*>(sraw), static_cast<const int32_t*>(ids),
      static_cast<float*>(part), topk);
  return (int)cudaGetLastError();
}

}  // extern "C"

// pkslots: the drafter's packed MoE down slot kernel (nemotron_h_mtp_down_packed._slots with
// the admitted (BN, warps) = (64, 4) tile) with fewer registers and a two-deep gather pipeline.
// CPT = 16 output columns per thread is that tile; CPT = 32 gives each thread 32 adjacent columns
// (a 128-column CTA): the same rows per thread and the same per-column operations in the same
// order, so the same fp32 words, from half as many row gathers per byte (PKSLOTS_TILE=64 keeps 16).
//
// _slots, for one slot (a routed expert or one shared-expert half) and 64 output columns, keeps a
// [64 rows, 64 columns] fp32 accumulator in the Triton layout sizePerThread [1, 16], threadsPerWarp
// [8, 4], warpsPerCTA [4, 1]: lane = 4 * lr + ct, row = 32 * rep + 8 * warp + lr, columns
// 16 * ct + j. Each 64-row chunk of the slot's compacted nonzero rows (in order) adds
// w[row, col] * value[row] (masked rows add +0). tl.sum over the rows then adds a thread's two
// reps, reduces the 8 row lanes with the xor butterfly 16, 8, 4, and combines the 4 warps'
// partials as (w0 + w2) + (w1 + w3). This kernel holds the same elements per thread and performs
// those operations in that order, so every fp32 output word is the same.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace pkslots {

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

// A thread's CPT sign/mantissa bytes (CPT / 4 words) and exponent codes (CPT / 8 words) of one row.
template <int CPT>
__device__ __forceinline__ void ld_row(uint32_t* sm, uint32_t* ex, const uint8_t* psm, const uint8_t* pex) {
#pragma unroll
  for (int q = 0; q < CPT / 16; ++q) {
    const uint4 r = ld_w128(psm + 16 * q);
    sm[4 * q] = r.x; sm[4 * q + 1] = r.y; sm[4 * q + 2] = r.z; sm[4 * q + 3] = r.w;
  }
  if constexpr (CPT == 16) {
    const uint2 r = ld_w64(pex);
    ex[0] = r.x; ex[1] = r.y;
  } else {
    const uint4 r = ld_w128(pex);
    ex[0] = r.x; ex[1] = r.y; ex[2] = r.z; ex[3] = r.w;
  }
}

template <int CPT>
struct Stage {
  uint32_t sm[2][CPT / 4];
  uint32_t ex[2][CPT / 8];
  uint32_t bs[2];
  int32_t ri[2];
  float v[2];
  long long idx[2];
};

template <bool PDL, int CPT>
__global__ __launch_bounds__(128) void slots_kernel(
    const int32_t* __restrict__ indices, const __nv_bfloat16* __restrict__ values,
    const int32_t* __restrict__ counts, const uint8_t* __restrict__ RSM,
    const uint8_t* __restrict__ REX, const uint8_t* __restrict__ RBASE,
    const int32_t* __restrict__ RRI, const uint16_t* __restrict__ RRAW,
    const uint8_t* __restrict__ SSM, const uint8_t* __restrict__ SEX,
    const uint8_t* __restrict__ SBASE, const int32_t* __restrict__ SRI,
    const uint16_t* __restrict__ SRAW, const int32_t* __restrict__ ids, float* __restrict__ part,
    int topk) {
#ifndef PKSLOTS_NO_PDL
  if constexpr (PDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" :::);
  }
#endif
  constexpr int TW = 4 * CPT;  // columns per CTA
  __shared__ float red[4][TW];
  const int slot = blockIdx.y;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int ct = lane & 3, lr = lane >> 2;
  const int n0 = blockIdx.x * TW + CPT * ct;
  const int count = counts[slot];
  const bool routed = slot < topk;
  const long long base = routed ? (long long)ids[slot] * I * H : (long long)(slot - topk) * I * H;
  const uint8_t* SM = routed ? RSM : SSM;
  const uint8_t* EX = routed ? REX : SEX;
  const uint8_t* BASE = routed ? RBASE : SBASE;
  const int32_t* RI = routed ? RRI : SRI;
  const uint16_t* RAW = routed ? RRAW : SRAW;
  const int32_t* ind = indices + (long long)slot * I;
  const __nv_bfloat16* val = values + (long long)slot * I;

  auto load = [&](Stage<CPT>& st, int start) {
#pragma unroll
    for (int rep = 0; rep < 2; ++rep) {
      const int pos = start + 32 * rep + 8 * warp + lr;
      if (pos < count) {
        const int k = ind[pos];
        st.v[rep] = __bfloat162float(val[pos]);
        const long long index = base + (long long)k * H + n0;
        st.idx[rep] = index;
        ld_row<CPT>(st.sm[rep], st.ex[rep], SM + index, EX + (index >> 1));
        st.bs[rep] = BASE[index >> 7];
        st.ri[rep] = RI[index >> 7];
      } else {
        st.v[rep] = 0.0f;
        st.idx[rep] = -1;
#pragma unroll
        for (int q = 0; q < CPT / 4; ++q) st.sm[rep][q] = 0u;
#pragma unroll
        for (int q = 0; q < CPT / 8; ++q) st.ex[rep][q] = 0u;
        st.bs[rep] = 0u;
        st.ri[rep] = -1;
      }
    }
  };

  float acc[2][CPT];
#pragma unroll
  for (int rep = 0; rep < 2; ++rep)
#pragma unroll
    for (int j = 0; j < CPT; ++j) acc[rep][j] = 0.0f;

  auto compute = [&](const Stage<CPT>& st) {
#pragma unroll
    for (int rep = 0; rep < 2; ++rep) {
      float w[CPT];
      if (st.idx[rep] < 0) {
#pragma unroll
        for (int j = 0; j < CPT; ++j) w[j] = 0.0f;
      } else if (st.ri[rep] >= 0) {
        const uint16_t* r = RAW + (long long)st.ri[rep] * 128 + (st.idx[rep] & 127);
#pragma unroll
        for (int h = 0; h < CPT / 16; ++h) {
          const uint4 a = *reinterpret_cast<const uint4*>(r + 16 * h);
          const uint4 b = *reinterpret_cast<const uint4*>(r + 16 * h + 8);
          const uint32_t q[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            w[16 * h + 2 * j] = __uint_as_float(q[j] << 16);
            w[16 * h + 2 * j + 1] = __uint_as_float(q[j] & 0xFFFF0000u);
          }
        }
      } else {
#pragma unroll
        for (int j = 0; j < CPT; ++j) {
          const uint32_t s = (st.sm[rep][j >> 2] >> (8 * (j & 3))) & 0xFFu;
          const uint32_t code = (st.ex[rep][j >> 3] >> (4 * (j & 7))) & 15u;
          const uint32_t e = code ? (st.bs[rep] + code - 1u) & 0xFFFFu : 0u;
          const uint32_t bits = (((s & 0x80u) << 8) | (e << 7) | (s & 0x7Fu)) & 0xFFFFu;
          w[j] = __uint_as_float(bits << 16);
        }
      }
#pragma unroll
      for (int j = 0; j < CPT; ++j) acc[rep][j] = __fadd_rn(acc[rep][j], __fmul_rn(w[j], st.v[rep]));
    }
  };

  // Two-deep pipeline: chunk c + 1's gathers are in flight while chunk c is added.
  Stage<CPT> s0, s1;
  load(s0, 0);
  int start = 0;
  for (; start + 64 < count; start += 64) {
    load(s1, start + 64);
    compute(s0);
    s0 = s1;
  }
  compute(s0);  // the last (possibly only) chunk; a zero-count slot adds +0 once, like _slots

  float t[CPT];
#pragma unroll
  for (int j = 0; j < CPT; ++j) t[j] = __fadd_rn(acc[0][j], acc[1][j]);
#pragma unroll
  for (int d = 16; d >= 4; d >>= 1)
#pragma unroll
    for (int j = 0; j < CPT; ++j) t[j] = __fadd_rn(t[j], __shfl_xor_sync(0xFFFFFFFFu, t[j], d));
  if (lr == 0) {
#pragma unroll
    for (int j = 0; j < CPT; ++j) red[warp][CPT * ct + j] = t[j];
  }
  __syncthreads();
  if (threadIdx.x < TW) {
    const int c = threadIdx.x;
    const float r = __fadd_rn(__fadd_rn(red[0][c], red[2][c]), __fadd_rn(red[1][c], red[3][c]));
    part[(long long)slot * H + blockIdx.x * TW + c] = r;
  }
}

}  // namespace pkslots

namespace pkslots {

template <int CPT>
int launch(const void* indices, const void* values, const void* counts, const void* rsm, const void* rex,
           const void* rbase, const void* rri, const void* rraw, const void* ssm, const void* sex,
           const void* sbase, const void* sri, const void* sraw, const void* ids, void* part, int topk,
           int pdl, cudaStream_t st) {
  const dim3 grid(H / (4 * CPT), topk + 2), block(128);
#ifndef PKSLOTS_NO_PDL
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
        &cfg, slots_kernel<true, CPT>, static_cast<const int32_t*>(indices),
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
  slots_kernel<false, CPT><<<grid, block, 0, st>>>(
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

}  // namespace pkslots

extern "C" {

// part[slots, 2688] = _slots(...) with the (64, 4) tile; slots = topk + 2. tile = columns per CTA:
// 64 (16 per thread) or 128 (32 per thread); both give the same fp32 words.
int pkslots_run(const void* indices, const void* values, const void* counts, const void* rsm,
                const void* rex, const void* rbase, const void* rri, const void* rraw,
                const void* ssm, const void* sex, const void* sbase, const void* sri,
                const void* sraw, const void* ids, void* part, int topk, int pdl, int tile,
                void* stream) {
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  if (tile == 128)
    return pkslots::launch<32>(indices, values, counts, rsm, rex, rbase, rri, rraw, ssm, sex, sbase, sri,
                               sraw, ids, part, topk, pdl, st);
  if (tile == 64)
    return pkslots::launch<16>(indices, values, counts, rsm, rex, rbase, rri, rraw, ssm, sex, sbase, sri,
                               sraw, ids, part, topk, pdl, st);
  return -1;
}

}  // extern "C"

// Optional producer/register-fragment variant on entire4ae. Originalfp8ws remains intact.
// Sources: 8b406fa2-8fbf-4867-a9fb-7a8b40915ae3, ownf5f1fee6-0923-4709-ad14-befd2e1199ff.
#pragma once
namespace fp8ws_pipeline {
using namespace fp8ws;
constexpr int NCOMPUTE = fp8ws::NT;
constexpr int PIPE_NT = NCOMPUTE + 32;
// Explicit register fragments. The original four-way unrolled K stage may
// retain many ldmatrix payloads together (official predecessor ptxas: 255 regs,
// zero spills). Two named packets keep the next payload available before the
// current MMA while allowing its storage to be reused as soon as consumed.
// No generated-register or latency improvement is claimed before remote build.
__device__ __forceinline__ void load_fragment(
    uint32_t abase, uint32_t wbase, int ks, int wm, int wn, int lane, int rb0,
    uint32_t (&a)[MW][4], uint32_t (&b)[NW][2]) {
  const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ach = lane >> 4;
  const int br = (lane & 7) + (lane >> 4) * 8, bch = (lane >> 3) & 1;
  const int b2r = lane & 7, b2ch = (lane >> 3) & 1;
#pragma unroll
  for (int i = 0; i < MW; ++i) {
    const int r = wm * MW * 16 + i * 16 + ar;
    if (rb0 + i * 16 < M) ldsm4(abase + r * KT + swz(r, 2 * ks + ach) * 16, a[i]);
  }
#pragma unroll
  for (int j = 0; j < NW / 2; ++j) {
    const int r = wn * NW * 8 + j * 16 + br;
    uint32_t qd[4];
    ldsm4(wbase + r * KT + swz(r, 2 * ks + bch) * 16, qd);
    b[2 * j][0] = qd[0]; b[2 * j][1] = qd[1]; b[2 * j + 1][0] = qd[2]; b[2 * j + 1][1] = qd[3];
  }
  {
    const int r = wn * NW * 8 + (NW - 1) * 8 + b2r;
    ldsm2(wbase + r * KT + swz(r, 2 * ks + b2ch) * 16, b[NW - 1][0], b[NW - 1][1]);
  }
}

__device__ __forceinline__ void consume_fragment(
    const uint32_t (&a)[MW][4], const uint32_t (&b)[NW][2],
    float (&acc)[MW][NW][4], int rb0) {
#pragma unroll
  for (int i = 0; i < MW; ++i)
    if (rb0 + i * 16 < M)
#pragma unroll
      for (int j = 0; j < NW; ++j) mma(a[i], b[j][0], b[j][1], acc[i][j]);
}

__global__ void __launch_bounds__(PIPE_NT, 1) gemm(const __grid_constant__ CUtensorMap tmA, const Params p) {
  extern __shared__ uint8_t sm_raw[];
  __shared__ __align__(8) unsigned long long bars[2 * ST];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint32_t sbase = ((uint32_t)__cvta_generic_to_shared(sm_raw) + 1023u) & ~1023u;
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(&bars[0]), empty0 = full0 + 8 * ST;
  const int grid = gridDim.x;
  if ((int)blockIdx.x >= NTILES) return;
  const int nmine = (NTILES - blockIdx.x + grid - 1) / grid;  // tiles blockIdx.x + i*grid
  const int total = nmine * NKT;
  if (tid == 0) {
    for (int s = 0; s < ST; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NWARP); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  // Loads of iteration gl into stage gl % ST (the producer warp's lane 0);
  // waits until every original compute warp released that stage's previous use.
  auto produce = [&](int gl) {
    const int s = gl % ST, n = gl / ST;
    if (gl >= ST) mbar_wait(empty0 + 8 * s, (n - 1) & 1);
    const int ti = gl / NKT, kt = gl - ti * NKT;
    const int t = blockIdx.x + ti * grid, nt = t / NB, band = t - nt * NB;
    const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB;
    mbar_expect_tx(bar, STB);
    tma2d(dst, &tmA, kt * KT, band * BM, bar);
    bulk(dst + ASZ, p.wp + (size_t)(nt * NKT + kt) * WSZ, WSZ, bar);
  };
  if (warp == NWARP) {
    if (lane == 0) {
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA) : "memory");
      for (int gl = 0; gl < total; ++gl) produce(gl);
      // Drain the last use of each stage. A release means all eight consumer
      // warps completed their shared reads/MMA; no async write is left live
      // when this producer exits. Final epilogues use private accumulators.
      for (int s = 0; s < ST && s < total; ++s) {
        const int last_phase = (total - 1 - s) / ST;
        mbar_wait(empty0 + 8 * s, last_phase & 1);
      }
    }
    return;
  }
  const int wm = warp % WM, wn = warp / WM;
  const float sa = __ldg(p.sa), sb = __ldg(p.sb);
  float acc[MW][NW][4];
#pragma unroll
  for (int i = 0; i < MW; ++i)
#pragma unroll
    for (int j = 0; j < NW; ++j)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
  int kt = 0, ti = 0;
  for (int g = 0; g < total; ++g) {
    const int s = g % ST, n = g / ST;
    const int t = blockIdx.x + ti * grid, nt = t / NB, band = t - nt * NB;
    const int rb0 = band * BM + wm * MW * 16;  // first output row of this warp
    mbar_wait(full0 + 8 * s, n & 1);
    const uint32_t abase = sbase + s * STB, wbase = abase + ASZ;
    static_assert((KT / 32) % 2 == 0, "two explicit fragment packets");
    uint32_t a0[MW][4], a1[MW][4], b0[NW][2], b1[NW][2];
    load_fragment(abase, wbase, 0, wm, wn, lane, rb0, a0, b0);
#pragma unroll 1
    for (int ks = 0; ks < KT / 32; ks += 2) {
      // Original ascending K: packet0 -> packet1. Issue the next shared
      // payload before arithmetic; reuse packet0 only after its MMA chain.
      load_fragment(abase, wbase, ks + 1, wm, wn, lane, rb0, a1, b1);
      consume_fragment(a0, b0, acc, rb0);
      if (ks + 2 < KT / 32)
        load_fragment(abase, wbase, ks + 2, wm, wn, lane, rb0, a0, b0);
      consume_fragment(a1, b1, acc, rb0);
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * s);
    if (++kt == NKT) {
      kt = 0; ++ti;
      const int col0 = nt * BN + wn * NW * 8;
#pragma unroll
      for (int i = 0; i < MW; ++i) {
        epilogue(acc[i], p.out, rb0 + i * 16, col0, lane, wn, sa, sb, rb0 + i * 16 < M);
#pragma unroll
        for (int j = 0; j < NW; ++j)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
      }
    }
  }
}

}  // namespace fp8ws_pipeline
extern "C" int fp8ws_pipeline_init() {
  return static_cast<int>(cudaFuncSetAttribute(fp8ws_pipeline::gemm,
      cudaFuncAttributeMaxDynamicSharedMemorySize, fp8ws::SMEM));
}
extern "C" int fp8ws_run_pipeline(const void* a, const void* wp, const void* sa, const void* sb, void* c, int m, int n,
                         int k, void* stream) {
  using namespace fp8ws_pipeline;
  if (m != M || n != N || k != K || !encode || num_sms <= 0) return (int)cudaErrorInvalidValue;
  if (((uintptr_t)a & 15) || ((uintptr_t)wp & 15) || ((uintptr_t)c & 15)) return (int)cudaErrorInvalidValue;
  CUtensorMap map;
  cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)M};
  cuuint64_t strides[1] = {(cuuint64_t)K};
  cuuint32_t box[2] = {(cuuint32_t)KT, (cuuint32_t)BM};
  cuuint32_t es[2] = {1, 1};
  if (encode(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(a), dims, strides, box, es,
             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
    return -3;
  Params p{(const uint8_t*)wp, (const float*)sa, (const float*)sb, (__nv_bfloat16*)c};
  const int grid = num_sms < NTILES ? num_sms : NTILES;
  fp8ws_pipeline::gemm<<<grid, PIPE_NT, SMEM, (cudaStream_t)stream>>>(map, p);
  return (int)cudaGetLastError();
}

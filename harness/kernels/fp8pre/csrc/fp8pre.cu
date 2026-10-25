// SPDX-License-Identifier: Apache-2.0
// CUTLASS-exact FP8 GEMM for the two Mamba projection shapes at prefill row counts.
// Reads the original E4M3 activation [M,K] and the original column-major weight
// [K,N] (column n is K contiguous bytes); no repack and no new weight buffer.
// Each output follows one ascending-K chain of native e4m3 m16n8k32 MMAs from
// +0, then FP32 scale B, FP32 scale A and RN BF16, matching the pinned SM120
// ScaledEpilogue (the fp8small chain). No K split, reduction, activation
// requantization or weight format change. Rows at or past M are zero-filled in
// shared memory and never stored.
// Several tile/stage variants per shape share that per-output chain; they differ
// only in which CTA and warp own an output and how far ahead operands are staged.
// The Python guard bit-checks and graph-times each one and picks per shape.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
namespace fp8pre {
constexpr int BK = 64, THREADS = 256, WARPS = THREADS / 32;
constexpr int LD = BK + 16;  // 80-byte rows: ldmatrix and cp.async rows are bank-conflict free
struct P {const uint8_t *a,*b;const float *sa,*sb;__nv_bfloat16 *c;int m;};

template<int BM,int BN,int WARPS_M,int STAGES>
struct Tile {
 static constexpr int WARPS_N = WARPS / WARPS_M;
 static constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N, MI = WM / 16, NI = WN / 8;
 static constexpr int A_BYTES = BM * LD, STAGE_BYTES = (BM + BN) * LD, SMEM = STAGES * STAGE_BYTES;
 static_assert(WARPS_M * WARPS_N == WARPS && WM % 16 == 0 && WN % 8 == 0, "warp tiling");
 static_assert(SMEM <= 101376, "dynamic shared memory");
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
 return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src, bool valid) {
 asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template<int N_>
__device__ __forceinline__ void wait_group() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N_)); }
__device__ __forceinline__ void ldsm4(uint32_t addr, uint32_t (&r)[4]) {
 asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
 : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void mma(const uint32_t(&a)[4],uint32_t b0,uint32_t b1,float(&c)[4]) {
 asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
 :"=f"(c[0]),"=f"(c[1]),"=f"(c[2]),"=f"(c[3])
 :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1),"f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]));
}

// One K tile (64 bytes per row) of A rows m0.. and weight columns n0.. into a stage.
template<int K,int BM,int BN,int A_BYTES>
__device__ __forceinline__ void load_stage(const P& p, uint8_t* st, int m0, int n0, int kt) {
 const int t = threadIdx.x;
 const uint32_t sa = smem_u32(st), sb = smem_u32(st + A_BYTES);
 #pragma unroll
 for (int i = 0; i < (BM * 4 + THREADS - 1) / THREADS; ++i) {  // A: BM rows x 4 chunks
  int c = t + i * THREADS;
  if (c < BM * 4) {
   int row = c >> 2, ch = c & 3, gm = m0 + row;
   bool ok = gm < p.m;
   cp16(sa + row * LD + ch * 16, p.a + (size_t)(ok ? gm : 0) * K + kt * BK + ch * 16, ok);
  }
 }
 #pragma unroll
 for (int i = 0; i < (BN * 4 + THREADS - 1) / THREADS; ++i) {  // B: BN columns x 4 chunks
  int c = t + i * THREADS;
  if (c < BN * 4) {
   int col = c >> 2, ch = c & 3;
   cp16(sb + col * LD + ch * 16, p.b + (size_t)(n0 + col) * K + kt * BK + ch * 16, true);
  }
 }
}

template<int N,int K,int BM,int BN,int WARPS_M,int STAGES>
__global__ __launch_bounds__(THREADS,1) void gemm(P p) {
 using T = Tile<BM,BN,WARPS_M,STAGES>;
 constexpr int WM = T::WM, WN = T::WN, MI = T::MI, NI = T::NI;
 constexpr int A_BYTES = T::A_BYTES, STAGE_BYTES = T::STAGE_BYTES;
 extern __shared__ __align__(128) uint8_t smem[];
 const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
 const int wm = warp % WARPS_M, wn = warp / WARPS_M;
 const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;  // M tiles of one N tile run together: weight read once
 float c[MI][NI][4];
 #pragma unroll
 for (int i = 0; i < MI; ++i)
  #pragma unroll
  for (int j = 0; j < NI; ++j)
   #pragma unroll
   for (int e = 0; e < 4; ++e) c[i][j][e] = 0.f;
 constexpr int KT = K / BK;
 #pragma unroll
 for (int s = 0; s < STAGES - 1; ++s) { load_stage<K,BM,BN,A_BYTES>(p, smem + s * STAGE_BYTES, m0, n0, s); commit(); }
 // ldmatrix lane addresses. A x4 for one m16 x k32 step: matrices (rows 0-7, k 0-15),
 // (rows 8-15, k 0-15), (rows 0-7, k 16-31), (rows 8-15, k 16-31) -> a0..a3.
 // B x4 for one n8 column group: k 0-15, 16-31, 32-47, 48-63 -> b0,b1 of step 0, b0,b1 of step 1.
 const int a_row = wm * WM + (lane & 7) + ((lane >> 3) & 1) * 8, a_col = (lane >> 4) * 16;
 const int b_row = wn * WN + (lane & 7), b_col = (lane >> 3) * 16;
 for (int kt = 0; kt < KT; ++kt) {
  wait_group<STAGES - 2>();
  __syncthreads();
  if (kt + STAGES - 1 < KT)
   load_stage<K,BM,BN,A_BYTES>(p, smem + ((kt + STAGES - 1) % STAGES) * STAGE_BYTES, m0, n0, kt + STAGES - 1);
  commit();
  const uint8_t* st = smem + (kt % STAGES) * STAGE_BYTES;
  const uint32_t sa = smem_u32(st), sb = smem_u32(st + A_BYTES);
  uint32_t b[NI][4];
  #pragma unroll
  for (int j = 0; j < NI; ++j) ldsm4(sb + (b_row + j * 8) * LD + b_col, b[j]);
  #pragma unroll
  for (int s = 0; s < 2; ++s) {  // ascending k32 steps inside the tile
   uint32_t a[MI][4];
   #pragma unroll
   for (int i = 0; i < MI; ++i) ldsm4(sa + (a_row + i * 16) * LD + s * 32 + a_col, a[i]);
   #pragma unroll
   for (int i = 0; i < MI; ++i)
    #pragma unroll
    for (int j = 0; j < NI; ++j) mma(a[i], b[j][2 * s], b[j][2 * s + 1], c[i][j]);
  }
 }
 wait_group<0>();
 const float sa = __ldg(p.sa), sb = __ldg(p.sb);
 #pragma unroll
 for (int i = 0; i < MI; ++i) {
  const int row = m0 + wm * WM + i * 16 + (lane >> 2);
  #pragma unroll
  for (int j = 0; j < NI; ++j) {
   const int col = n0 + wn * WN + j * 8 + (lane & 3) * 2;
   #pragma unroll
   for (int h = 0; h < 2; ++h) {
    const int r = row + h * 8;
    if (r < p.m) {
     __nv_bfloat162 v;
     v.x = __float2bfloat16_rn(__fmul_rn(__fmul_rn(c[i][j][2 * h], sb), sa));
     v.y = __float2bfloat16_rn(__fmul_rn(__fmul_rn(c[i][j][2 * h + 1], sb), sa));
     *(__nv_bfloat162*)(p.c + (size_t)r * N + col) = v;
    }
   }
  }
 }
}

template<int N,int K,int BM,int BN,int WARPS_M,int STAGES>
cudaError_t init_v() {
 static_assert(N % BN == 0 && K % BK == 0 && K / BK >= STAGES - 1, "tiles");
 return cudaFuncSetAttribute(gemm<N,K,BM,BN,WARPS_M,STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             Tile<BM,BN,WARPS_M,STAGES>::SMEM);
}
template<int N,int K,int BM,int BN,int WARPS_M,int STAGES>
cudaError_t run_v(const P& p, cudaStream_t stream) {
 dim3 grid((p.m + BM - 1) / BM, N / BN);
 gemm<N,K,BM,BN,WARPS_M,STAGES><<<grid, THREADS, Tile<BM,BN,WARPS_M,STAGES>::SMEM, stream>>>(p);
 return cudaGetLastError();
}

struct Variant {int n,k,bm,bn,stages;cudaError_t (*init)();cudaError_t (*run)(const P&,cudaStream_t);};
#define FP8PRE_V(N,K,BM,BN,WARPS_M,ST) {N,K,BM,BN,ST,&init_v<N,K,BM,BN,WARPS_M,ST>,&run_v<N,K,BM,BN,WARPS_M,ST>}
// in_proj (10304, 2688): 112 and 224 divide N. out_proj (2688, 4096): 112 gives 96 tiles
// (2 full waves on 48 SMs), 224 or 256-row tiles give 48 (1 wave), 128 matches CUTLASS's tile.
constexpr Variant VARIANTS[] = {
 FP8PRE_V(10304, 2688, 128, 112, 4, 3),
 FP8PRE_V(10304, 2688, 128, 112, 4, 4),
 FP8PRE_V(10304, 2688, 128, 224, 2, 3),
 FP8PRE_V(10304, 2688, 256, 112, 4, 3),
 FP8PRE_V(2688, 4096, 128, 112, 4, 3),
 FP8PRE_V(2688, 4096, 128, 112, 4, 5),
 FP8PRE_V(2688, 4096, 128, 128, 4, 4),
 FP8PRE_V(2688, 4096, 256, 112, 4, 3),
 FP8PRE_V(2688, 4096, 128, 224, 2, 3),
};
#undef FP8PRE_V
constexpr int COUNT = sizeof(VARIANTS) / sizeof(VARIANTS[0]);
}
extern "C" int fp8pre_count() { return fp8pre::COUNT; }
// info = {n, k, bm, bn, stages}
extern "C" int fp8pre_info(int v, int* info) {
 if (v < 0 || v >= fp8pre::COUNT) return (int)cudaErrorInvalidValue;
 const fp8pre::Variant& x = fp8pre::VARIANTS[v];
 info[0] = x.n; info[1] = x.k; info[2] = x.bm; info[3] = x.bn; info[4] = x.stages;
 return 0;
}
extern "C" int fp8pre_init(int v) {
 if (v < 0 || v >= fp8pre::COUNT) return (int)cudaErrorInvalidValue;
 return (int)fp8pre::VARIANTS[v].init();
}
extern "C" int fp8pre_run(const void*a,const void*b,const void*sa,const void*sb,void*c,int m,int n,int k,int v,void*stream) {
 using namespace fp8pre;
 if (m < 1 || v < 0 || v >= COUNT || VARIANTS[v].n != n || VARIANTS[v].k != k) return (int)cudaErrorInvalidValue;
 P p{(const uint8_t*)a,(const uint8_t*)b,(const float*)sa,(const float*)sb,(__nv_bfloat16*)c,m};
 return (int)VARIANTS[v].run(p, (cudaStream_t)stream);
}

// SPDX-License-Identifier: Apache-2.0
// bf16ws: the prefill attention projection BF16 GEMMs, C[496,N] = A[496,K] x W[N,K]^T, BF16 out,
// for (N, K) = (4608, 2688) (qkv_proj) and (2688, 4096) (o_proj).
//
// Arithmetic (identical to the cuBLAS kernels F.linear runs at M=496 on these shapes): every output
// starts at +0.0f and follows one ascending-K chain of native
// mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 over the original BF16 values (no split-K,
// no cross-warp reduction), then one RN to BF16.
//
// Layout: the fp8ws kernel (harness/kernels/fp8ws) with the BF16 MMA. Persistent grid (one CTA per
// SM), output tiles of 256 rows x BN columns (2 row bands; BN = 96 for the qkv_proj, 2 x 48 tiles;
// BN = 112 for the o_proj, 2 x 24 tiles), 8 MMA warps of 64 x BN/2 per tile. Per K step of 128 bytes
// (64 BF16 values) the A tile (the producer's row-major activation, rows past 496 zero-filled)
// arrives by a 2-D TMA load with the 128-byte swizzle, and the W tile by one bulk copy from the
// load-time pack Wp[n_tile][k_tile][BN][128 bytes], whose 16-byte chunks are pre-swizzled the same
// way (a byte permutation of the checkpoint weight). Two smem stages, full/empty mbarriers; thread 0
// issues the loads. The epilogue transposes within each lane quad so every lane stores 16
// contiguous bytes.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace bf16ws {
constexpr int M = 496;
constexpr int KT = 128, CPR = KT / 16, ST = 2;  // K step in bytes (64 BF16 values)
constexpr int WM = 4, WN = 2, MW = 4;
constexpr int BM = WM * MW * 16;                // 256
constexpr int NWARP = WM * WN, NT = NWARP * 32;
constexpr int NB = 2;                           // two row bands
constexpr int ASZ = BM * KT;                    // 32 KiB

// (N, K) in BF16 values; NW n8 blocks per warp, BN = 2 * NW * 8 columns per tile.
template <int N, int K, int NW_> struct Shape {
  static constexpr int N_ = N, NW = NW_, BN = WN * NW * 8;
  static constexpr int KB = 2 * K;                       // bytes per row
  static constexpr int NTN = N / BN, NTILES = NB * NTN;
  static constexpr int NKT = KB / KT;
  static constexpr int WSZ = BN * KT, STB = ASZ + WSZ;
  static constexpr int SMEM = ST * STB + 1024;           // + alignment slack
  static_assert(N % BN == 0 && KB % KT == 0 && STB % 1024 == 0, "shape");
};
using QKV = Shape<4608, 2688, 6>;  // BN 96: 2 x 48 tiles, 42 K steps
using OPJ = Shape<2688, 4096, 7>;  // BN 112: 2 x 24 tiles, 64 K steps

// 16-byte chunk c of row r lands at chunk c ^ (r & 7) (TMA SWIZZLE_128B on 128-byte rows,
// 1024-byte aligned stage base).
__host__ __device__ constexpr int swz(int r, int c) { return c ^ (r & 7); }

__device__ __forceinline__ void mbar_init(uint32_t a, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t a, int tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(tx) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t a, int parity) {
  asm volatile(
      "{\n .reg .pred p;\n W_%=:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra W_%=;\n}\n" ::"r"(a), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* map, int x, int y, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
      ::"r"(dst), "l"(map), "r"(x), "r"(y), "r"(bar) : "memory");
}
__device__ __forceinline__ void bulk(uint32_t dst, const void* src, int bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void ldsm4(uint32_t a, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm2(uint32_t a, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ void mma(const uint32_t (&a)[4], uint32_t b0, uint32_t b1, float (&c)[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                 "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
// One RN of the FP32 accumulator to BF16.
__device__ __forceinline__ uint32_t cvt2(float lo, float hi) {
  __nv_bfloat162 v;
  v.x = __float2bfloat16_rn(lo);
  v.y = __float2bfloat16_rn(hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
// c ? a : b without a branch (keeps the warp converged for the shuffles).
__device__ __forceinline__ uint32_t selp(uint32_t a, uint32_t b, bool c) {
  uint32_t r;
  asm("{\n .reg .pred p;\n setp.ne.b32 p, %3, 0;\n selp.b32 %0, %1, %2, p;\n}"
      : "=r"(r) : "r"(a), "r"(b), "r"((uint32_t)c));
  return r;
}
__device__ __forceinline__ uint32_t sel4(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3, int i) {
  const uint32_t lo = selp(v1, v0, i & 1), hi = selp(v3, v2, i & 1);
  return selp(hi, lo, i & 2);
}

// Wp[nt][kt][r][c'] (16-byte chunks) = W[nt*BN + r][kt*KT + swz(r, c')*16 ...]; W is [N][K] BF16.
template <class S>
__global__ void pack_kernel(const uint4* __restrict__ W, uint4* __restrict__ Wp) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)S::N_ * S::KB / 16) return;
  const int cp = i % CPR;
  size_t t = i / CPR;
  const int r = t % S::BN;
  t /= S::BN;
  const int kt = t % S::NKT, nt = (int)(t / S::NKT);
  Wp[i] = W[((size_t)(nt * S::BN + r) * S::KB + kt * KT) / 16 + swz(r, cp)];
}

struct Params {
  const uint8_t* wp; __nv_bfloat16* out;
};

// One pair of n8 blocks (j, j+1): each lane quad holds 4 chunks of 16 bytes (rows r, r+8 x two
// blocks); a 4x4 transpose within the quad gives every lane one chunk.
template <int N>
__device__ __forceinline__ void store_pair(const float (&e0)[4], const float (&e1)[4], __nv_bfloat16* out,
                                           int row0, int col0, int j, int lane, bool st) {
  const int q = lane & 3, rq = lane >> 2;
  const uint32_t v0 = cvt2(e0[0], e0[1]), v1 = cvt2(e0[2], e0[3]);
  const uint32_t v2 = cvt2(e1[0], e1[1]), v3 = cvt2(e1[2], e1[3]);
  const uint32_t r1 = __shfl_xor_sync(0xffffffffu, sel4(v0, v1, v2, v3, q ^ 1), 1);
  const uint32_t r2 = __shfl_xor_sync(0xffffffffu, sel4(v0, v1, v2, v3, q ^ 2), 2);
  const uint32_t r3 = __shfl_xor_sync(0xffffffffu, sel4(v0, v1, v2, v3, q ^ 3), 3);
  const uint32_t own = sel4(v0, v1, v2, v3, q);
  uint4 w;
  w.x = sel4(own, r1, r2, r3, 0 ^ q);
  w.y = sel4(own, r1, r2, r3, 1 ^ q);
  w.z = sel4(own, r1, r2, r3, 2 ^ q);
  w.w = sel4(own, r1, r2, r3, 3 ^ q);
  const int row = row0 + rq + (q & 1) * 8, col = col0 + (j + (q >> 1)) * 8;
  if (st) *reinterpret_cast<uint4*>(out + (size_t)row * N + col) = w;
}
// One lone n8 block j, stored as 8-byte halves.
template <int N>
__device__ __forceinline__ void store_one(const float (&e)[4], __nv_bfloat16* out, int row0, int col0, int j,
                                          int lane, bool st) {
  const int q = lane & 3, rq = lane >> 2;
  const uint32_t x = cvt2(e[0], e[1]), y = cvt2(e[2], e[3]);
  const uint32_t rcv = __shfl_xor_sync(0xffffffffu, selp(x, y, q & 1), 1);
  uint2 w;
  w.x = selp(rcv, x, q & 1);
  w.y = selp(y, rcv, q & 1);
  const int row = row0 + rq + (q & 1) * 8, col = col0 + j * 8 + (q & 2) * 2;
  if (st) *reinterpret_cast<uint2*>(out + (size_t)row * N + col) = w;
}

// One warp's stores for m16 block i. Even NW: pairs (0,1)(2,3)... for both column halves. NW = 7:
// wn==0 stores pairs (0,1)(2,3)(4,5) and block 6 alone, wn==1 block 0 alone and pairs (1,2)(3,4)(5,6),
// so every pair starts on a 32-byte boundary.
template <int N, int NW>
__device__ __forceinline__ void epilogue(const float (&acc)[NW][4], __nv_bfloat16* out, int row0, int col0,
                                         int lane, int wn, bool st) {
  if constexpr (NW % 2 == 0) {
#pragma unroll
    for (int pp = 0; pp < NW / 2; ++pp) store_pair<N>(acc[2 * pp], acc[2 * pp + 1], out, row0, col0, 2 * pp, lane, st);
  } else {
    const bool w1 = wn != 0;
#pragma unroll
    for (int pp = 0; pp < NW / 2; ++pp) {
      float e0[4], e1[4];
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        e0[c] = w1 ? acc[2 * pp + 1][c] : acc[2 * pp][c];
        e1[c] = w1 ? acc[2 * pp + 2][c] : acc[2 * pp + 1][c];
      }
      store_pair<N>(e0, e1, out, row0, col0, 2 * pp + (w1 ? 1 : 0), lane, st);
    }
    float e[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) e[c] = w1 ? acc[0][c] : acc[NW - 1][c];
    store_one<N>(e, out, row0, col0, w1 ? 0 : NW - 1, lane, st);
  }
}

template <class S>
__global__ void __launch_bounds__(NT, 1) gemm(const __grid_constant__ CUtensorMap tmA, const Params p) {
  constexpr int N = S::N_, NW = S::NW, BN = S::BN, NTILES = S::NTILES, NKT = S::NKT;
  constexpr int WSZ = S::WSZ, STB = S::STB;
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
  // Loads of iteration gl into stage gl % ST (thread 0); waits until every warp released that
  // stage's previous use (iteration gl - ST).
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
  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA) : "memory");
    produce(0);
  }
  const int wm = warp % WM, wn = warp / WM;
  float acc[MW][NW][4];
#pragma unroll
  for (int i = 0; i < MW; ++i)
#pragma unroll
    for (int j = 0; j < NW; ++j)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
  // per-lane ldmatrix row and chunk-half offsets
  const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ach = lane >> 4;  // A x4: rows 0-15, k halves
  const int br = (lane & 7) + (lane >> 4) * 8, bch = (lane >> 3) & 1;   // B x4: two n8 blocks
  const int b2r = lane & 7, b2ch = (lane >> 3) & 1;                      // B x2: one n8 block
  int kt = 0, ti = 0;
  for (int g = 0; g < total; ++g) {
    const int s = g % ST, n = g / ST;
    const int t = blockIdx.x + ti * grid, nt = t / NB, band = t - nt * NB;
    const int rb0 = band * BM + wm * MW * 16;  // first output row of this warp
    if (tid == 0 && g + 1 < total) produce(g + 1);
    mbar_wait(full0 + 8 * s, n & 1);
    const uint32_t abase = sbase + s * STB, wbase = abase + ASZ;
#pragma unroll
    for (int ks = 0; ks < KT / 32; ++ks) {  // ascending K within the stage: 16 BF16 values per MMA
      uint32_t a[MW][4], b[NW][2];
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
      if constexpr (NW % 2 == 1) {
        const int r = wn * NW * 8 + (NW - 1) * 8 + b2r;
        ldsm2(wbase + r * KT + swz(r, 2 * ks + b2ch) * 16, b[NW - 1][0], b[NW - 1][1]);
      }
#pragma unroll
      for (int i = 0; i < MW; ++i)
        if (rb0 + i * 16 < M)  // rows 496..511 of the second band do not exist
#pragma unroll
          for (int j = 0; j < NW; ++j) mma(a[i], b[j][0], b[j][1], acc[i][j]);
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * s);
    if (++kt == NKT) {
      kt = 0; ++ti;
      const int col0 = nt * BN + wn * NW * 8;
#pragma unroll
      for (int i = 0; i < MW; ++i) {
        epilogue<N, NW>(acc[i], p.out, rb0 + i * 16, col0, lane, wn, rb0 + i * 16 < M);
#pragma unroll
        for (int j = 0; j < NW; ++j)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
      }
    }
  }
}

typedef CUresult (*EncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                             const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                             CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeFn encode = nullptr;
static int num_sms = 0;

template <class S>
static int pack_shape(const void* w, void* wp, cudaStream_t stream) {
  const size_t chunks = (size_t)S::N_ * S::KB / 16;
  pack_kernel<S><<<(unsigned)((chunks + 255) / 256), 256, 0, stream>>>((const uint4*)w, (uint4*)wp);
  return (int)cudaGetLastError();
}

template <class S>
static int run_shape(const void* a, const void* wp, void* c, cudaStream_t stream) {
  CUtensorMap map;
  // the activation's bytes: [M][2K]
  cuuint64_t dims[2] = {(cuuint64_t)S::KB, (cuuint64_t)M};
  cuuint64_t strides[1] = {(cuuint64_t)S::KB};
  cuuint32_t box[2] = {(cuuint32_t)KT, (cuuint32_t)BM};
  cuuint32_t es[2] = {1, 1};
  if (encode(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(a), dims, strides, box, es,
             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
    return -3;
  Params p{(const uint8_t*)wp, (__nv_bfloat16*)c};
  const int grid = num_sms < S::NTILES ? num_sms : S::NTILES;
  gemm<S><<<grid, NT, S::SMEM, stream>>>(map, p);
  return (int)cudaGetLastError();
}
}  // namespace bf16ws

extern "C" int bf16ws_init() {
  using namespace bf16ws;
  if (!encode) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    cudaError_t e = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q);
    if (e != cudaSuccess || !fn || q != cudaDriverEntryPointSuccess) return e != cudaSuccess ? (int)e : -2;
    encode = (EncodeFn)fn;
  }
  int dev = 0;
  cudaError_t e = cudaGetDevice(&dev);
  if (e == cudaSuccess) e = cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
  if (e == cudaSuccess) e = cudaFuncSetAttribute(gemm<QKV>, cudaFuncAttributeMaxDynamicSharedMemorySize, QKV::SMEM);
  if (e == cudaSuccess) e = cudaFuncSetAttribute(gemm<OPJ>, cudaFuncAttributeMaxDynamicSharedMemorySize, OPJ::SMEM);
  return (int)e;
}

// w: [N][K] BF16 (the linear layer's weight); wp: 2*N*K bytes.
extern "C" int bf16ws_pack(const void* w, void* wp, int n, int k, void* stream) {
  using namespace bf16ws;
  if (n == 4608 && k == 2688) return pack_shape<QKV>(w, wp, (cudaStream_t)stream);
  if (n == 2688 && k == 4096) return pack_shape<OPJ>(w, wp, (cudaStream_t)stream);
  return (int)cudaErrorInvalidValue;
}

extern "C" int bf16ws_run(const void* a, const void* wp, void* c, int m, int n, int k, void* stream) {
  using namespace bf16ws;
  if (m != M || !encode || num_sms <= 0) return (int)cudaErrorInvalidValue;
  if (((uintptr_t)a & 15) || ((uintptr_t)wp & 15) || ((uintptr_t)c & 15)) return (int)cudaErrorInvalidValue;
  if (n == 4608 && k == 2688) return run_shape<QKV>(a, wp, c, (cudaStream_t)stream);
  if (n == 2688 && k == 4096) return run_shape<OPJ>(a, wp, c, (cudaStream_t)stream);
  return (int)cudaErrorInvalidValue;
}

// Independent experimental scale-readiness library. Inherited comments describe original zskip only.
// No local compilation, qualification, resource inspection, activation or timing has been performed.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace zskip {

#ifndef ZSKIP_NO_PDL
// Programmatic dependent launch (sm_90+). Both v4 kernels call these first, before
// any global memory access; launched without the PDL attribute they are no-ops.
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
#endif

constexpr int K = 1856;           // intermediate size (down K)
constexpr int N = 2688;           // hidden size (down N)
constexpr int G = K / 16;         // 116 k16 groups
constexpr int RMAX = 64;          // valid rows per expert block (block_m 16 decode .. 64 prefill)
constexpr int WARP_COLS = 32;
constexpr int NWARPS_TOTAL = N / WARP_COLS;  // 84

enum Mode { DENSE = 0, SKIP = 1, LOADONLY = 2, LOADONLY_DENSE = 3, SETUPONLY = 4 };  // 2-4: timing probes

struct Params {
  const __nv_bfloat16* A;   // [shape_m, lda] relu^2 activations (rows = token*top_k + slot)
  long long lda;            // elements
  int shape_m;              // rows of A and C
  const int* sorted_ids;    // moe_align_block_size (block 16)
  const int* expert_ids;
  const int* num_tokens_padded;
  const uint8_t* W;         // repacked e2m1 nibbles
  const uint8_t* S;         // repacked e4m3 scales
  const float* GS;          // [E] fp32 global scales (weight_scale_2)
  __nv_bfloat16* C;         // [shape_m, ldc]
  long long ldc;            // elements
  long long w_expert, w_tile, w_k;  // byte strides: expert, 128-column tile, k row
  long long s_expert, s_tile, s_g;  // byte strides: expert, 128-column tile, k16 group
  int mode;
  int* stats;               // optional [4]: 0 dense redo items, 1 blocks with > RMAX rows
  uint8_t* ws;              // [max_blocks * BI_BYTES] per-block lists (prep_kernel -> down3_kernel)
  int* queue;               // work-item counter (reset by prep_kernel)
  int max_blocks;           // expert blocks covered (grid of prep_kernel)
  unsigned long long* trace;  // optional timing trace: 8 words per item (see down3_kernel)
  const int* verify_rows;  // optional confidence prefix, paired M5 only
  int c3lpt;  // shape30/config113 only: read same-call prep permutation
};

__device__ __forceinline__ int prefix_row_limit(const Params& p) {
  if (p.verify_rows == nullptr) return p.shape_m;
  const int rows = *p.verify_rows;
  // Avoid multiplying the ALL_ROWS sentinel. Eligible calls have 5 tokens x6.
  return rows >= 5 ? p.shape_m : (rows < 1 ? 1 : rows) * 6;
}

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ unsigned smid() {
  unsigned r;
  asm volatile("mov.u32 %0, %smid;" : "=r"(r));
  return r;
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N_>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N_) : "memory"); }

__device__ __forceinline__ uint32_t lds_u32(const void* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}
__device__ __forceinline__ uint32_t lds_u16(const void* p) {
  return *reinterpret_cast<const uint16_t*>(p);
}

__device__ __forceinline__ uint32_t hmul2_u32(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}

// humming fp_to_fp<Float4E2M1, BFloat16>: nibbles at bits 12..15 and 28..31.
__device__ __forceinline__ uint32_t fp4_to_bf16x2(uint32_t v) {
  return (v & 0x80008000u) | ((v & 0x70007000u) >> 6);
}
// humming fp_to_fp<Float8E4M3, BFloat16>: bytes at bits 8..15 and 24..31.
__device__ __forceinline__ uint32_t fp8_to_bf16x2(uint32_t v) {
  return (v & 0x80008000u) | ((v & 0x7F007F00u) >> 4);
}

// prepare_exp_scale_factor<nv_bfloat162, E>: (E * 0x00010001 + 0x007F007F) << 7
constexpr uint32_t kExp124 = (124u * 0x00010001u + 0x007F007Fu) << 7;  // 2^124 in both halves
constexpr uint32_t kExp120 = (120u * 0x00010001u + 0x007F007Fu) << 7;  // 2^120

__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float mul_ftz(float a, float b) {
  float r;
  asm("mul.rn.ftz.f32 %0, %1, %2;\n" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ uint32_t to_bf16x2(float lo, float hi) {
  __nv_bfloat162 r = __float22bfloat162_rn(make_float2(lo, hi));
  return *reinterpret_cast<uint32_t*>(&r);
}

// bit 2i / 2i+1: bf16 element 2i / 2i+1 of the 16-byte chunk has nonzero magnitude
__device__ __forceinline__ uint32_t nz_mask8(uint4 v) {
  uint32_t m = 0;
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    m |= ((w[i] & 0x7FFFu) != 0u ? 1u : 0u) << (2 * i);
    m |= ((w[i] & 0x7FFF0000u) != 0u ? 1u : 0u) << (2 * i + 1);
  }
  return m;
}


#include "scaleready.cuh"
// Same scalar-pair expression as the original lean_group, once per immutable scale lane.
__global__ void prepare_scales(const uint8_t* source, uint8_t* result, int experts,
                               long long se, long long st, long long sg) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= experts * G * 21 * 8) return;
  const int g8 = q % 8, tile = (q / 8) % 21, group = (q / (8 * 21)) % G;
  const int expert = q / (8 * 21 * G);
  const uint32_t* s = reinterpret_cast<const uint32_t*>(source + expert * se + tile * st + group * sg + g8 * 16);
  uint32_t* d = reinterpret_cast<uint32_t*>(result + expert * (2 * se) + tile * (2 * st) + group * (2 * sg) + g8 * 32);
#pragma unroll
  for (int tt = 0; tt < 2; ++tt) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint32_t sxx = c < 2 ? s[2 * tt] : s[2 * tt + 1];
      d[tt * 4 + c] = hmul2_u32(fp8_to_bf16x2(__byte_perm(sxx, 0u, (c & 1) == 0 ? 0x1404 : 0x3424)), kExp120);
    }
  }
}
}  // namespace zskip

extern "C" {
int scaleready_init() {
  cudaFuncAttributes attr;
  return static_cast<int>(cudaFuncGetAttributes(&attr, zskip::down_lean_kernel<1, 3, 274>));
}
int scaleready_pack(const void* source, void* result, int experts,
                    long long se, long long st, long long sg, void* stream) {
  if (experts != 128 || se != zskip::G * zskip::N || st != 128 || sg != zskip::N || source == nullptr || result == nullptr)
    return -1;
  const int count = experts * zskip::G * 21 * 8;
  zskip::prepare_scales<<<(count + 127) / 128, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const uint8_t*>(source), reinterpret_cast<uint8_t*>(result), experts, se, st, sg);
  return static_cast<int>(cudaGetLastError());
}
int scaleready_down(const void* A, long long lda, int rows, const int* sorted_ids, const int* expert_ids,
                    const int* ntp, int max_blocks, const void* W, const void* S, const float* GS,
                    void* C, long long ldc, long long we, long long wt, long long wk,
                    long long se, long long st, long long sg, int mode, int* stats,
                    void* stream, const int* verify_rows, int pdl) {
  if (rows < 6 || rows > 48 || rows % 6 || max_blocks < 1 || max_blocks > rows || lda % 8 || ldc % 2 ||
      (mode != zskip::DENSE && mode != zskip::SKIP) || (verify_rows != nullptr && (rows != 30 || mode != zskip::SKIP)) ||
      we != zskip::K * (zskip::N / 2) || wt != 64 || wk != zskip::N / 2 ||
      se != 2 * zskip::G * zskip::N || st != 256 || sg != 2 * zskip::N)
    return -2;
  zskip::Params p = {};
  p.A = reinterpret_cast<const __nv_bfloat16*>(A); p.lda = lda; p.shape_m = rows;
  p.sorted_ids = sorted_ids; p.expert_ids = expert_ids; p.num_tokens_padded = ntp;
  p.W = reinterpret_cast<const uint8_t*>(W); p.S = reinterpret_cast<const uint8_t*>(S); p.GS = GS;
  p.C = reinterpret_cast<__nv_bfloat16*>(C); p.ldc = ldc;
  p.w_expert = we; p.w_tile = wt; p.w_k = wk; p.s_expert = se; p.s_tile = st; p.s_g = sg;
  p.mode = mode; p.stats = stats; p.max_blocks = max_blocks; p.verify_rows = verify_rows;
  const cudaStream_t stream_ = reinterpret_cast<cudaStream_t>(stream);
  if (pdl) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(21, max_blocks); cfg.blockDim = dim3(32); cfg.stream = stream_;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    return static_cast<int>(cudaLaunchKernelEx(&cfg, zskip::down_lean_kernel<1, 3, 274>, p));
  }
  zskip::down_lean_kernel<1, 3, 274><<<dim3(21, max_blocks), 32, 0, stream_>>>(p);
  return static_cast<int>(cudaGetLastError());
}
}  // extern C

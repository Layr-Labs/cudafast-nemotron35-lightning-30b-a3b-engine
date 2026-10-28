// Unvalidated optional research, enabled only after private warmup admission.
// Based on the original
// ssd_prefill_fused.py CB expression;
// BF16 fragment mapping follows NVIDIA PTX ISA mma.m16n8k16 documentation.
#include <cuda_runtime.h>
#include <stdint.h>
#include "ssdcb_consumer.cuh"

namespace ssd_cb_research {

// Two adjacent BF16 bit patterns, without any floating-point conversion.
__device__ __forceinline__ uint32_t pair_bits(const uint16_t* p, int64_t stride,
                                              bool valid) {
  if (!valid) return 0;
  uint32_t lo = p[0];
  uint32_t hi = p[stride];
  return lo | (hi << 16);
}

__device__ __forceinline__ void mma(const uint32_t (&a)[4],
                                     const uint32_t (&b)[2], float (&d)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// blockIdx=(row-tile, chunk, group), 16 full warps, CHUNK=128 or 256.
template <int CHUNK>
__device__ void cb_producer(const uint16_t* B, const uint16_t* C, float* cb,
                            int begin, int limit,
                            int64_t bs_token, int64_t bs_group, int64_t bs_state,
                            int64_t cs_token, int64_t cs_group, int64_t cs_state) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int group = blockIdx.z, tile = blockIdx.x;
  constexpr int nbands = CHUNK / 64;
  const bool active = warp < 4 * nbands;
  if (active) {
    const int band_r = warp / nbands, band_c = warp % nbands;
    const int gr = lane >> 2, pair = (lane & 3) * 2;
    const int r0 = tile * 64 + band_r * 16 + gr;
    const int r1 = r0 + 8;
    float sums[8][4] = {};
    // Source-level CB adds four independent 32-wide dot results. Whether
    // Triton lowers this to this exact rounding boundary remains unverified.
    for (int outer = 0; outer < 128; outer += 32) {
      float segment[8][4] = {};
      for (int inner = 0; inner < 32; inner += 16) {
        const int k = outer + inner + pair;
        const uint16_t* cp0 = r0 < limit ? C + (begin + r0) * cs_token + group * cs_group : C;
        const uint16_t* cp1 = r1 < limit ? C + (begin + r1) * cs_token + group * cs_group : C;
        uint32_t a[4] = {
          pair_bits(cp0 + k * cs_state, cs_state, r0 < limit),
          pair_bits(cp1 + k * cs_state, cs_state, r1 < limit),
          pair_bits(cp0 + (k + 8) * cs_state, cs_state, r0 < limit),
          pair_bits(cp1 + (k + 8) * cs_state, cs_state, r1 < limit),
        };
        #pragma unroll
        for (int n8 = 0; n8 < 8; ++n8) {
          const int col = band_c * 64 + n8 * 8 + gr;
          const uint16_t* bp = col < limit ? B + (begin + col) * bs_token + group * bs_group : B;
          uint32_t b[2] = {
            pair_bits(bp + k * bs_state, bs_state, col < limit),
            pair_bits(bp + (k + 8) * bs_state, bs_state, col < limit),
          };
          mma(a, b, segment[n8]);
        }
      }
      #pragma unroll
      for (int n8 = 0; n8 < 8; ++n8)
        #pragma unroll
        for (int i = 0; i < 4; ++i)
          sums[n8][i] = __fadd_rn(sums[n8][i], segment[n8][i]);
    }
    #pragma unroll
    for (int n8 = 0; n8 < 8; ++n8) {
      const int row = band_r * 16 + gr;
      const int col = band_c * 64 + n8 * 8 + pair;
      cb[row * CHUNK + col] = sums[n8][0];
      cb[row * CHUNK + col + 1] = sums[n8][1];
      cb[(row + 8) * CHUNK + col] = sums[n8][2];
      cb[(row + 8) * CHUNK + col + 1] = sums[n8][3];
    }
  }
}

struct LaunchInputs {
  ScanInputs scan;
  const uint16_t* b;
  const void* cu;
  const void* seq;
  int64_t bs0, bs1, bs2;
  int total_tokens, seq_count, chunks;
};

// This exact predicate must also be applied to the original two-CTA path.
// No candidate is safe to enable until the fallback uses its complement.
template <typename I, typename S>
__device__ __forceinline__ bool eligible(const LaunchInputs& p, int chunk,
                                          int chunk_size, int& begin, int& limit,
                                          int& seq, int& prev_seq) {
  int64_t start = static_cast<const I*>(p.cu)[chunk];
  int64_t end = static_cast<const I*>(p.cu)[chunk + 1];
  int64_t cur = static_cast<const S*>(p.seq)[chunk];
  int64_t prev = chunk > 0 ? static_cast<const S*>(p.seq)[chunk - 1] : -1;
  bool ok = start >= 0 && end >= start && end <= p.total_tokens
      && end - start <= chunk_size && cur >= 0 && cur < p.seq_count
      && (chunk == 0 || (prev >= 0 && prev < p.seq_count))
      && (cur != prev || chunk > 0);
  begin = static_cast<int>(start); limit = static_cast<int>(end - start);
  seq = static_cast<int>(cur); prev_seq = static_cast<int>(prev);
  return ok;
}

template <int CHUNK, typename I, typename S>
__global__ __launch_bounds__(512, 1) void fused_cb_scan(LaunchInputs p) {
  extern __shared__ float cb[];
  const int chunk = blockIdx.y, group = blockIdx.z, tile = blockIdx.x;
  int begin, limit, seq, prev_seq;
  if (!eligible<I,S>(p,chunk,CHUNK,begin,limit,seq,prev_seq)) return;
  float early[4][4] = {};
  cb_producer<CHUNK>(p.b,p.scan.c,cb,begin,limit,p.bs0,p.bs1,p.bs2,
                     p.scan.cs0,p.scan.cs1,p.scan.cs2);
  // CHUNK128 uses only the lower eight warps for CB production.  The upper
  // eight can compute their first head's independent state dot before the
  // common barrier, preserving the original MMA and accumulation order.
  if (CHUNK == 128 && threadIdx.x >= 256) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int group_warp = warp & 7;
    const int row_band = group_warp / 2, col_band = group_warp % 2;
    const int lane_group = lane >> 2, pair = (lane & 3) * 2;
    const int r0 = tile * 64 + row_band * 16 + lane_group, r1 = r0 + 8;
    state_dot(p.scan, begin, limit, chunk, group, seq, prev_seq, group * 8 + 4,
              r0, r1, col_band * 32, lane_group, pair, early);
  }
  // All 512 threads participate, including inactive producer warps at CHUNK128.
  __syncthreads();
  consume_cb<CHUNK>(p.scan,cb,begin,limit,chunk,tile,group,seq,prev_seq,early);
}

template <int CHUNK, typename I, typename S>
cudaError_t init_variant() {
  cudaFuncAttributes attr;
  auto rc = cudaFuncGetAttributes(&attr, fused_cb_scan<CHUNK,I,S>);
  if (rc != cudaSuccess) return rc;
  return cudaFuncSetAttribute(fused_cb_scan<CHUNK,I,S>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * CHUNK * sizeof(float));
}
template <int CHUNK, typename I, typename S>
cudaError_t launch_variant(const LaunchInputs& p, cudaStream_t stream) {
  dim3 grid(CHUNK / 64, p.chunks, 8);
  fused_cb_scan<CHUNK,I,S><<<grid,512,64 * CHUNK * sizeof(float),stream>>>(p);
  return cudaGetLastError();
}
}  // namespace ssd_cb_research

// UNWIRED ABI. Any nonzero result must preserve the original implementation.
extern "C" int ssdcb_init() {
  using namespace ssd_cb_research;
  cudaError_t rc;
#define INIT(C,I,S) do { rc=init_variant<C,I,S>(); if(rc!=cudaSuccess) return (int)rc; } while(0)
  INIT(128,int32_t,int32_t); INIT(128,int32_t,int64_t);
  INIT(128,int64_t,int32_t); INIT(128,int64_t,int64_t);
  INIT(256,int32_t,int32_t); INIT(256,int32_t,int64_t);
  INIT(256,int64_t,int32_t); INIT(256,int64_t,int64_t);
#undef INIT
  return 0;
}
extern "C" int ssdcb_run(const ssd_cb_research::LaunchInputs* host,
                         int chunk_size, int cu64, int seq64, void* stream) {
  using namespace ssd_cb_research;
  if (!host || host->chunks <= 0 || host->seq_count <= 0 || host->total_tokens <= 0)
    return (int)cudaErrorInvalidValue;
  LaunchInputs p = *host;
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  cudaError_t rc = cudaErrorInvalidValue;
#define RUN(C,I,S) rc=launch_variant<C,I,S>(p,st)
  if (chunk_size==128) {
    if(cu64) { if(seq64) RUN(128,int64_t,int64_t); else RUN(128,int64_t,int32_t); }
    else { if(seq64) RUN(128,int32_t,int64_t); else RUN(128,int32_t,int32_t); }
  } else if (chunk_size==256) {
    if(cu64) { if(seq64) RUN(256,int64_t,int64_t); else RUN(256,int64_t,int32_t); }
    else { if(seq64) RUN(256,int32_t,int64_t); else RUN(256,int32_t,int32_t); }
  }
#undef RUN
  return (int)rc;
}

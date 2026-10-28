// Optional research implementation, gated at remote model warmup.
// Consumer executes after the producer's CTA-wide shared-CB barrier.
// Restricted to 64 heads, 8 groups, head_dim=64, dstate=128 and Z=None.
// Numerical and boundary equality remain subject to remote validation.
#include <cuda_bf16.h>
#include <stdint.h>

namespace ssd_cb_research {

struct ScanInputs {
  const uint16_t *x, *c;
  const float *dt, *da, *states, *initial, *d;
  uint16_t *out;
  int64_t xs0, xs1, xs2, cs0, cs1, cs2, ts0, ts1, ts2;
  int64_t as0, as1, as2, ss0, ss1, ss2, ss3;
  int64_t is0, is1, is2, is3, os0, os1, os2, ds;
  bool has_initial, has_d, d_vector;
};

__device__ __forceinline__ uint16_t bf16_bits(float x) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(x));
}
__device__ __forceinline__ float from_bf16(uint16_t x) {
  return __bfloat162float(__ushort_as_bfloat16(x));
}
__device__ __forceinline__ uint32_t pack(uint16_t x, uint16_t y) {
  return static_cast<uint32_t>(x) | (static_cast<uint32_t>(y) << 16);
}
__device__ __forceinline__ float approximate_exp(float x) {
  // fast_exp is exp2(log2(e)*x). The chosen PTX and multiplication
  // boundaries must be compared with the original compiler output remotely.
  float z = __fmul_rn(x, 1.4426950408889634f), y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(z));
  return y;
}
__device__ __forceinline__ void consumer_mma(const uint32_t (&a)[4],
                                             const uint32_t (&b)[2],
                                             float (&d)[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint16_t c_value(const ScanInputs& p, int begin,
                                             int group, int row, int k, int limit) {
  return row < limit ? p.c[(int64_t)(begin + row) * p.cs0 + group * p.cs1 + k * p.cs2] : 0;
}
__device__ __forceinline__ uint16_t x_value(const ScanInputs& p, int begin,
                                             int head, int row, int col, int limit) {
  return row < limit ? p.x[(int64_t)(begin + row) * p.xs0 + head * p.xs1 + col * p.xs2] : 0;
}
__device__ __forceinline__ uint16_t prev_value(const ScanInputs& p, int chunk,
                                                int seq, int prev_seq, int head,
                                                int col, int k) {
  if (seq != prev_seq) {
    if (!p.has_initial) return 0;
    return bf16_bits(p.initial[(int64_t)seq * p.is0 + head * p.is1 + col * p.is2 + k * p.is3]);
  }
  return bf16_bits(p.states[(int64_t)(chunk - 1) * p.ss0 + head * p.ss1 + col * p.ss2 + k * p.ss3]);
}

// This dot is independent of the shared CB tile.  Keep the same K and n8
// order for both the pre-barrier and the ordinary consumer schedules.
__device__ __forceinline__ void state_dot(const ScanInputs& p, int begin,
                                         int limit, int chunk, int group,
                                         int seq, int prev_seq, int head,
                                         int r0, int r1, int n0,
                                         int lane_group, int pair,
                                         float (&acc)[4][4]) {
  for (int kbase = 0; kbase < 128; kbase += 16) {
    const int k = kbase + pair;
    uint32_t a[4] = {
      pack(c_value(p, begin, group, r0, k, limit), c_value(p, begin, group, r0, k+1, limit)),
      pack(c_value(p, begin, group, r1, k, limit), c_value(p, begin, group, r1, k+1, limit)),
      pack(c_value(p, begin, group, r0, k+8, limit), c_value(p, begin, group, r0, k+9, limit)),
      pack(c_value(p, begin, group, r1, k+8, limit), c_value(p, begin, group, r1, k+9, limit)),
    };
    #pragma unroll
    for (int n8 = 0; n8 < 4; ++n8) {
      const int col = n0 + n8 * 8 + lane_group;
      uint32_t b[2] = {
        pack(prev_value(p, chunk, seq, prev_seq, head, col, k), prev_value(p, chunk, seq, prev_seq, head, col, k+1)),
        pack(prev_value(p, chunk, seq, prev_seq, head, col, k+8), prev_value(p, chunk, seq, prev_seq, head, col, k+9)),
      };
      consumer_mma(a, b, acc[n8]);
    }
  }
}
template <int CHUNK>
__device__ __forceinline__ uint16_t scaled_cb(const ScanInputs& p, const float* cb,
                                               int chunk, int tile, int head,
                                               int local_row, int key) {
  const int row = tile * 64 + local_row;
  float rm = row < CHUNK ? p.da[head * p.as0 + chunk * p.as1 + row * p.as2] : 0.f;
  float rk = key < CHUNK ? p.da[head * p.as0 + chunk * p.as1 + key * p.as2] : 0.f;
  float tk = key < CHUNK ? p.dt[head * p.ts0 + chunk * p.ts1 + key * p.ts2] : 0.f;
  float decay = approximate_exp(fminf(__fsub_rn(rm, rk), 0.f));
  float value = __fmul_rn(__fmul_rn(cb[local_row * CHUNK + key], decay), tk);
  return row >= key ? bf16_bits(value) : 0;
}

// Called by all 16 warps, after the same CTA-wide barrier. CTA and metadata
// validity, non-overlap, nonnegative sequence identity and all strides must
// be checked by the eventual dispatcher.  Z is deliberately unsupported.
template <int CHUNK>
__device__ void consume_cb(const ScanInputs& p, const float* cb, int begin,
                           int limit, int chunk, int tile, int group,
                           int seq, int prev_seq, const float (&early)[4][4]) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int group_warp = warp & 7, subgroup = warp >> 3;
  const int row_band = group_warp / 2, col_band = group_warp % 2;
  const int lane_group = lane >> 2, pair = (lane & 3) * 2;
  const int r0 = tile * 64 + row_band * 16 + lane_group, r1 = r0 + 8;
  const int local_r0 = row_band * 16 + lane_group, local_r1 = local_r0 + 8;
  const int n0 = col_band * 32;
  for (int h = 0; h < 4; ++h) {
    const int head = group * 8 + subgroup * 4 + h;
    float acc[4][4] = {};
    // The upper warp group has already computed its first state dot while
    // the lower group produced CB; all other heads follow the same helper.
    if (CHUNK == 128 && subgroup == 1 && h == 0) {
      #pragma unroll
      for (int n8 = 0; n8 < 4; ++n8)
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[n8][i] = early[n8][i];
    } else {
      state_dot(p, begin, limit, chunk, group, seq, prev_seq, head,
                r0, r1, n0, lane_group, pair, acc);
    }
    const float d0 = r0 < CHUNK ? p.da[head*p.as0 + chunk*p.as1 + r0*p.as2] : 0.f;
    const float d1 = r1 < CHUNK ? p.da[head*p.as0 + chunk*p.as1 + r1*p.as2] : 0.f;
    #pragma unroll
    for (int n8 = 0; n8 < 4; ++n8) {
      acc[n8][0] = __fmul_rn(acc[n8][0], approximate_exp(d0));
      acc[n8][1] = __fmul_rn(acc[n8][1], approximate_exp(d0));
      acc[n8][2] = __fmul_rn(acc[n8][2], approximate_exp(d1));
      acc[n8][3] = __fmul_rn(acc[n8][3], approximate_exp(d1));
    }
    const int kmax = min((tile + 1) * 64, limit);
    for (int outer = 0; outer < kmax; outer += 64) {
      float segment[4][4] = {};
      for (int kb = 0; kb < 64; kb += 16) {
        const int k = outer + kb + pair;
        uint32_t a[4] = {
          pack(scaled_cb<CHUNK>(p,cb,chunk,tile,head,local_r0,k),scaled_cb<CHUNK>(p,cb,chunk,tile,head,local_r0,k+1)),
          pack(scaled_cb<CHUNK>(p,cb,chunk,tile,head,local_r1,k),scaled_cb<CHUNK>(p,cb,chunk,tile,head,local_r1,k+1)),
          pack(scaled_cb<CHUNK>(p,cb,chunk,tile,head,local_r0,k+8),scaled_cb<CHUNK>(p,cb,chunk,tile,head,local_r0,k+9)),
          pack(scaled_cb<CHUNK>(p,cb,chunk,tile,head,local_r1,k+8),scaled_cb<CHUNK>(p,cb,chunk,tile,head,local_r1,k+9)),
        };
        #pragma unroll
        for (int n8 = 0; n8 < 4; ++n8) {
          const int col = n0 + n8*8 + lane_group;
          uint32_t b[2] = {
            pack(x_value(p,begin,head,k,col,limit),x_value(p,begin,head,k+1,col,limit)),
            pack(x_value(p,begin,head,k+8,col,limit),x_value(p,begin,head,k+9,col,limit)),
          };
          consumer_mma(a,b,segment[n8]);
        }
      }
      #pragma unroll
      for (int n8 = 0; n8 < 4; ++n8)
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[n8][i] = __fadd_rn(acc[n8][i],segment[n8][i]);
    }
    #pragma unroll
    for (int n8 = 0; n8 < 4; ++n8) {
      const int col = n0 + n8*8 + pair;
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row = i < 2 ? r0 : r1, channel = col + (i & 1);
        float v = acc[n8][i];
        if (p.has_d) {
          float d = p.d[head*p.ds + (p.d_vector ? channel : 0)];
          float x = from_bf16(x_value(p,begin,head,row,channel,limit));
          v = __fadd_rn(v,__fmul_rn(x,d));
        }
        if (row < limit) p.out[(int64_t)(begin+row)*p.os0 + head*p.os1 + channel*p.os2] = bf16_bits(v);
      }
    }
  }
}
}  // namespace ssd_cb_research

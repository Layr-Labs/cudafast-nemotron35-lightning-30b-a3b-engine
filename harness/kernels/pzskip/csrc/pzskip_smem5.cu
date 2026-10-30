// pzskip unit: configs 356, 357, 369 and 379 (prep2_kernel, smem5_kernel)
#include "pzskip.cuh"

namespace pzskip {

__global__ void __launch_bounds__(128) prep2_kernel(const Params p) {
  constexpr int BMT = 32;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int mb = blockIdx.x;
  uint8_t* bi = p.ws + (long long)mb * BI_BYTES;
  int* rows = reinterpret_cast<int*>(bi + BI_ROWS);
  uint16_t* mask = reinterpret_cast<uint16_t*>(bi + BI_MASK);
  uint16_t* actg = reinterpret_cast<uint16_t*>(bi + BI_ACTG);
  __shared__ int s_rows[BMT];
  __shared__ int s_n;
  __shared__ uint16_t s_mask[128];
  const bool live = mb * BMT < *p.num_tokens_padded;
  if (tid < BMT) {
    const unsigned id = live ? static_cast<unsigned>(p.sorted_ids[mb * BMT + tid]) : 0xFFFFFFFFu;
    const bool valid = id < static_cast<unsigned>(p.shape_m);
    rows[tid] = valid ? static_cast<int>(id) : -1;
    s_rows[tid] = valid ? static_cast<int>(id) : -1;
  }
  __syncthreads();
  if (tid == 0) {
    int n = 0;
    for (int r = 0; r < BMT; ++r) n += s_rows[r] >= 0 ? 1 : 0;
    s_n = n;
    *reinterpret_cast<int*>(bi + BI_NROWS) = n;
    *reinterpret_cast<int*>(bi + BI_EXPERT) = live ? p.expert_ids[p.half64 ? (mb >> 1) : mb] : -1;
  }
  __syncthreads();
  {
    const int g = tid;  // one k16 group per thread (G = 116 < 128)
    uint32_t m = 0;
    if (g < G && s_n > 0) {
      if (p.dense == 1) {
        m = 0xFFFFu;
      } else {
        for (int r0 = 0; r0 < BMT; r0 += 8) {
          uint4 v[8][2];
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const int row = s_rows[r0 + i];
            if (row >= 0) {
              const uint4* q = reinterpret_cast<const uint4*>(p.A + (long long)row * p.lda + 16 * g);
              v[i][0] = __ldg(q);
              v[i][1] = __ldg(q + 1);
            } else {
              v[i][0] = v[i][1] = make_uint4(0u, 0u, 0u, 0u);
            }
          }
#pragma unroll
          for (int i = 0; i < 8; ++i) m |= nz_mask8(v[i][0]) | (nz_mask8(v[i][1]) << 8);
        }
      }
    }
    s_mask[g] = static_cast<uint16_t>(m);
    mask[g] = static_cast<uint16_t>(m);
  }
  __syncthreads();
  if (warp == 0) {  // ordered list of active groups
    int base = 0;
    for (int c = 0; c < 4; ++c) {
      const int g = 32 * c + lane;
      const bool a = g < G && s_mask[g] != 0;
      const unsigned ball = __ballot_sync(0xFFFFFFFFu, a);
      if (a) actg[base + __popc(ball & ((1u << lane) - 1u))] = static_cast<uint16_t>(g);
      base += __popc(ball);
    }
    if (lane == 0) *reinterpret_cast<int*>(bi + BI_NAG) = base;
  }
}

int dispatch_smem5(int wpc, const Params& p, cudaStream_t st) {
  switch (wpc) {
    case 356: {
#ifndef PZSKIP_NO_CACHE_POLICY
      // As case 333: the M=496 top-6 prefill (2976 routed slots) gets L2 evict-first weight and
      // scale loads unless HUMMING_EVICT_PZ=0; same kernel otherwise, stock 356 on any failure.
      const char* enabled = std::getenv("HUMMING_EVICT_PZ");
      if (p.shape_m == 2976 && (!enabled || enabled[0] != '0')) {
        static int loaded = -1;
        if (loaded < 0) {
          cudaFuncAttributes attr;
          cudaError_t e = cudaFuncGetAttributes(&attr, pzskip::smem5_kernel<1, 4, 3, 1>);
          loaded = e == cudaSuccess;
          if (!loaded) {
            cudaGetLastError();
            std::fprintf(stderr, "pzskip: L2 evict-first load unavailable; stock PZ356 retained\n");
          }
        }
        if (loaded) {
          int rc = pzskip::launch_smem5<1, 4, 3, 1>(p, st);
          if (!rc) return 0;
          loaded = 0;
          cudaGetLastError();
          std::fprintf(stderr, "pzskip: L2 evict-first launch unavailable; stock PZ356 retained\n");
        }
      }
#endif
      return pzskip::launch_smem5<1, 4, 3, 0>(p, st);
    }  // 333's CTA, lean issue (see smem5_kernel)
    case 357: return pzskip::launch_smem5<2, 2, 3>(p, st);
    // LOCAL PATCH (cudafast-nemotron35, pzhalf): 356 on humming's 64-row prefill moe_align
    // layout, read as 32-row halves (block b covers rows 32 b .. 32 b + 31 of the 64-row
    // layout, expert expert_ids[b / 2]). An expert's 32-row chunks are the same rows as in the
    // 32-row layout (both start at its first slot); a half holding only padding has no rows and
    // its CTAs return at once. Same blocks, same arithmetic; the second moe_align is not run.
    case 369: {
      pzskip::Params q = p;
      q.half64 = 1;
#ifndef PZSKIP_NO_CACHE_POLICY
      const char* enabled = std::getenv("HUMMING_EVICT_PZ");
      if (q.shape_m == 2976 && (!enabled || enabled[0] != '0')) {
        int rc = pzskip::launch_smem5<1, 4, 3, 1>(q, st);
        if (!rc) return 0;
        cudaGetLastError();
      }
#endif
      return pzskip::launch_smem5<1, 4, 3, 0>(q, st);
    }
    // LOCAL PATCH (cudafast-nemotron35, pz S6): 369 reading the 6-bit swap scale plane
    // (zskip py to_s6_swap; p.S / s_expert / s_g describe that plane). Same blocks, same arithmetic.
    case 379: {
      pzskip::Params q = p;
      q.half64 = 1;
#ifndef PZSKIP_NO_CACHE_POLICY
      const char* enabled = std::getenv("HUMMING_EVICT_PZ");
      if (q.shape_m == 2976 && (!enabled || enabled[0] != '0')) {
        int rc = pzskip::launch_smem5<1, 4, 3, 1, true>(q, st);
        if (!rc) return 0;
        cudaGetLastError();
      }
#endif
      return pzskip::launch_smem5<1, 4, 3, 0, true>(q, st);
    }
    default: return NOT_HANDLED;
  }
}

}  // namespace pzskip

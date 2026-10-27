// Optional rest-dot warp-role variant; all execution is official-remote-only.
// Include inside namespace scrtail after Params, Smem, ld8/ld1 and block_term.
// Source: entire1c6e2ace-8cfd-4552-935f-a438a54710cc and original scrtail
// a26a82a4-8acb-4821-87a4-9823006eeca6. No arithmetic helper is replaced.
// Barrier design reference: NVIDIA PTX ISA, bar/barrier CTA producer-consumer example.
// https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar
#pragma once

// One packet is shared. The producer preloads the next packet into registers
// while consumers read/compute the current one. A consumed barrier protects
// reuse; no producer writes this storage before all consumers have read it.
// Lane-major planes make each participating warp's stores/loads contiguous.
struct __align__(16) RestPacket {
  uint2 weights[4][THREADS / 2];
  uint32_t scales[4][THREADS / 2];
};

__device__ __forceinline__ void rest_ready_arrive() {
  asm volatile("barrier.cta.arrive 1, 512;" ::: "memory");
}
__device__ __forceinline__ void rest_ready_wait() {
  asm volatile("barrier.cta.sync 1, 512;" ::: "memory");
}
__device__ __forceinline__ void rest_consumed_arrive() {
  asm volatile("barrier.cta.arrive 2, 512;" ::: "memory");
}
__device__ __forceinline__ void rest_consumed_wait() {
  asm volatile("barrier.cta.sync 2, 512;" ::: "memory");
}

template <int WIN>
__device__ __forceinline__ void rest_warp_pipeline(
    const Params& p, Smem& sm, RestPacket& packet, long long base_id) {
  static_assert(THREADS == 512 && WARPS == 16 && CAS == 256,
                "fixed warp roles and named-barrier participant count");
  constexpr int ROLE_WARPS = WARPS / 2;
  constexpr int ROWS_PACKET = ROLE_WARPS * 4;
  constexpr int ITERATIONS = CAS / ROWS_PACKET;
  const int t = threadIdx.x;
  const int warp = t >> 5, lane = t & 31;
  const int lc = lane & 7, sub = lane >> 3;
  if (warp < ROLE_WARPS) {
    // Eight whole producer warps issue each next gathered row's four packed
    // K-block reads before waiting for consumers to release shared storage.
#pragma unroll 1
    for (int it = 0; it < ITERATIONS; ++it) {
      const int q = it * ROWS_PACKET + warp * 4 + sub;
      const int qi = sm.cas[q];
      const long long row = base_id + sm.pre[qi];
      uint2 wb[4];
      uint32_t sb[4];
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        wb[b] = ld8(p.w_rest + row * NB_REST + 64 * b + 8 * lc);
        sb[b] = ld1(p.s_rest + row * (NB_REST / 8) + 8 * b + lc);
      }
      if (it != 0) rest_consumed_wait();
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        packet.weights[b][t] = wb[b];
        packet.scales[b][t] = sb[b];
      }
      rest_ready_arrive();
    }
    // Reset the last consumed phase before all sixteen warps rejoin the
    // original CTA-wide barrier and subsequent top-k stages.
    rest_consumed_wait();
  } else {
    const int role = warp - ROLE_WARPS;
    const int source_lane = t - THREADS / 2;
#pragma unroll 1
    for (int it = 0; it < ITERATIONS; ++it) {
      const int q = it * ROWS_PACKET + role * 4 + sub;
      const int qi = sm.cas[q];
      rest_ready_wait();
      uint2 wb[4];
      uint32_t sb[4];
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        wb[b] = packet.weights[b][source_lane];
        sb[b] = packet.scales[b][source_lane];
      }
      // Release after all shared payload reads, before arithmetic on the
      // private register copies. Producer writes can overlap that arithmetic.
      rest_consumed_arrive();
      float acc = 0.0f;
#pragma unroll
      for (int b = 0; b < 4; ++b)
        acc = __fadd_rn(acc, block_term(wb[b], sb[b], sm.hr + 2 * (64 * b + 8 * lc)));
      const float out = __fmaf_rn(p.gscale, acc, sm.mid[qi]);
      if (lc == WIN) sm.ref[q] = out;
    }
  }
}

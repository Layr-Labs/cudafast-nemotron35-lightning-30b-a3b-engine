// SPDX-License-Identifier: Apache-2.0
// Config683: two cooperating warps own the two old653 64-column halves.
// R3/prefetch274, original dequant/MMA order and one shared ordered group list.
// Only DENSE/SKIP are dispatched. No S6, NR1 specialization or FP-format change.
#ifndef ZSKIP_NO_PAIR

__device__ __forceinline__ void lean64_ldg32_z(uint32_t& x, const void* ptr, uint32_t pred) {
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %2, 0;\n mov.b32 %0, 0;\n"
               " @q ld.global.nc.L1::no_allocate.u32 %0, [%1];\n}\n"
               : "=r"(x) : "l"(ptr), "r"(pred));
}
__device__ __forceinline__ void lean64_neg0(bool& neg0, const float (&acc)[4][4]) {
#pragma unroll
  for (int c = 0; c < 4; ++c) {
#pragma unroll
    for (int q = 0; q < 4; ++q) neg0 |= __float_as_uint(acc[c][q]) == 0x80000000u;
  }
}

__device__ __forceinline__ void lean64_group(
    float (&acc)[4][4], uint32_t (&alo)[4][2], uint32_t (&ahi)[4][2],
    const uint32_t (&w)[4], const uint32_t (&s)[2], const uint32_t (&x)[2],
    uint32_t m, bool& neg0) {
  uint32_t sc[4];
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const uint32_t sxx = c < 2 ? s[0] : s[1];
    sc[c] = hmul2_u32(fp8_to_bf16x2(__byte_perm(sxx, 0u, (c & 1) == 0 ? 0x1404 : 0x3424)), kExp120);
  }
  if (m & 0x00FFu) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint32_t sel = c | (c << 4) | ((c + 4) << 8) | ((c + 4) << 12);
      const uint32_t x01 = __byte_perm(w[0], w[1], sel);
      alo[c][0] = hmul2_lo(hmul2_u32(fp4_to_bf16x2(x01 << 12), kExp124), sc[c]);
      alo[c][1] = hmul2_hi(hmul2_u32(fp4_to_bf16x2(x01), kExp124), sc[c]);
    }
  }  // Empty half keeps the previous group's fragments, exactly like F274.
  if (m & 0xFF00u) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint32_t sel = c | (c << 4) | ((c + 4) << 8) | ((c + 4) << 12);
      const uint32_t x23 = __byte_perm(w[2], w[3], sel);
      ahi[c][0] = hmul2_lo(hmul2_u32(fp4_to_bf16x2(x23 << 12), kExp124), sc[c]);
      ahi[c][1] = hmul2_hi(hmul2_u32(fp4_to_bf16x2(x23), kExp124), sc[c]);
    }
  }
  lean64_neg0(neg0, acc);
#pragma unroll
  for (int i = 0; i < 4; ++i) mma_bf16(acc[i], alo[i][0], alo[i][1], ahi[i][0], ahi[i][1], x[0], x[1]);
}

// No K reordering: each output retains exactly the original ordinal group chain.
__device__ __forceinline__ void lean64_main(
    float (&acc)[4][4], const Params& p, const uint32_t* list, int nag,
    const uint8_t* const (&wr)[4], const uint8_t* sb, const uint8_t* xb,
    unsigned gw, unsigned sg, uint32_t xok, int t, bool dense, int half) {
  constexpr int R = 3, NS = R + 1, F = 274;
  for (int pass = 0; pass < 2; ++pass) {
    const int nsteps = dense ? G : nag;
    uint32_t W_[NS][4], S_[NS][2], X_[NS][2], M_[NS];
#pragma unroll
    for (int i = 0; i < NS; ++i) {
#pragma unroll
      for (int q = 0; q < 4; ++q) W_[i][q] = 0u;
      S_[i][0] = S_[i][1] = X_[i][0] = X_[i][1] = M_[i] = 0u;
    }
    auto issue = [&](int slot, int j) {
      const uint32_t ent = dense ? (0xFFFF0000u | static_cast<uint32_t>(j)) : list[j];
      const unsigned grp = ent & 0xFFFFu;
      const uint32_t mm = ent >> (16 + 2 * t);
      M_[slot] = ent >> 16;
      const unsigned long long go = static_cast<unsigned long long>(grp * gw);
      lean64_ldg32_z(W_[slot][0], wr[0] + go, mm & 1u);
      lean64_ldg32_z(W_[slot][1], wr[1] + go, mm & 2u);
      lean64_ldg32_z(W_[slot][2], wr[2] + go, mm & 0x100u);
      lean64_ldg32_z(W_[slot][3], wr[3] + go, mm & 0x200u);
      ldg64_na(S_[slot][0], S_[slot][1], sb + grp * sg);
      ldg32l1_z(X_[slot][0], xb + 32u * grp, xok);
      ldg32l1_z(X_[slot][1], xb + 32u * grp + 16u, xok);
    };
#pragma unroll
    for (int c = 0; c < 4; ++c) {
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[c][q] = 0.0f;
    }
#pragma unroll
    for (int i = 0; i < R; ++i)
      if (i < nsteps) issue(i, i);
    uint32_t alo[4][2], ahi[4][2];
#pragma unroll
    for (int c = 0; c < 4; ++c) alo[c][0] = alo[c][1] = ahi[c][0] = ahi[c][1] = 0u;
    bool neg0 = false;
    for (int j0 = 0; j0 < nsteps; j0 += NS) {
#pragma unroll
      for (int i = 0; i < NS; ++i) {
        if (j0 + i < nsteps) {
          uint32_t w_[4], s_[2], x_[2];
          const uint32_t m_ = M_[i];
#pragma unroll
          for (int q = 0; q < 4; ++q) w_[q] = W_[i][q];
          s_[0] = S_[i][0];
          s_[1] = S_[i][1];
          x_[0] = X_[i][0];
          x_[1] = X_[i][1];
          // Only warp0 emits the original full-pair prefetch requests.
          const int jp = j0 + i + R + prefetch_distance<F>(nag);
          if (half == 0 && !dense && jp < nsteps) {
            const uint32_t ent = list[jp];
            const unsigned grp = ent & 0xFFFFu;
            const uint32_t mm = ent >> (16 + 2 * t);
            const unsigned long long go = static_cast<unsigned long long>(grp * gw);
            prefetch_l2_pred(wr[0] + go, mm & 1u);
            prefetch_l2_pred(wr[1] + go, mm & 2u);
            prefetch_l2_pred(wr[2] + go, mm & 0x100u);
            prefetch_l2_pred(wr[3] + go, mm & 0x200u);
            prefetch_l2_pred(sb + grp * sg, t == 0);
          }
          // Same R3 refill-before-math ordering, with only this half's payload.
          if (j0 + i + R < nsteps) issue((i + R) % NS, j0 + i + R);
          lean64_group(acc, alo, ahi, w_, s_, x_, m_, neg0);
        }
      }
    }
    if (dense) break;
    lean64_neg0(neg0, acc);
    // Original653 decided redo for BOTH halves together; preserve that union.
    // Every thread has identical nsteps/dense/pass control; no per-group fence.
    if (!__syncthreads_or(neg0)) break;
    if (threadIdx.x == 0 && p.stats) atomicAdd(p.stats + 0, 1);
    dense = true;
  }
}

__global__ void __launch_bounds__(64) down_lean64_kernel(const Params p) {
  // NT32 describes the original one-warp preparation, not this CTA's block size.
  constexpr int NT = 32;
  const int lane = threadIdx.x & 31, half = threadIdx.x >> 5;
  const int g8 = lane >> 2, t = lane & 3;
  const int mb = blockIdx.y, P = blockIdx.x;
  const unsigned long long t0 = p.trace ? gtimer() : 0ull;
#ifndef ZSKIP_NO_PDL
  pdl_wait();
  pdl_launch_dependents();
#endif
  if (mb * 16 >= *p.num_tokens_padded) return;
  __shared__ int s_rows[8];
  __shared__ int s_nrows, s_e, s_nag;
  __shared__ uint16_t s_mask[128];
  __shared__ uint32_t s_list[G];
  if (half == 0) {
    const int e = lane == 0 ? p.expert_ids[mb] : 0;
    const unsigned id = lane < 16 ? static_cast<unsigned>(p.sorted_ids[mb * 16 + lane]) : 0xFFFFFFFFu;
    const bool valid = id < static_cast<unsigned>(p.shape_m);
    const unsigned ball = __ballot_sync(0xFFFFFFFFu, valid);
    const int rank = __popc(ball & ((1u << lane) - 1u));
    if (valid && rank < 8) s_rows[rank] = static_cast<int>(id);
    if (lane == 0) {
      const int n = __popc(ball);
      s_nrows = n < 8 ? n : 8;
      s_e = e;
      if (n > 8 && p.stats && blockIdx.x == 0) atomicAdd(p.stats + 1, 1);
    }
  }
  __syncthreads();
  const int nrows = s_nrows;
  if (nrows == 0) return;
  const int row_limit = prefix_row_limit(p);
  const bool dense0 = p.mode == DENSE;
  if (half == 0) {
    const int tid = lane;
    constexpr int GPT = (128 + NT - 1) / NT;
    uint32_t m[GPT];
#pragma unroll
    for (int i = 0; i < GPT; ++i) m[i] = 0u;
    if (dense0) {
#pragma unroll
      for (int i = 0; i < GPT; ++i) m[i] = tid + NT * i < G ? 0xFFFFu : 0u;
    } else {
      // two rows per round trip (the rows' loads are independent)
      for (int r = 0; r < nrows; r += 2) {
        uint4 v[2][GPT][2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int row = r + h < nrows ? s_rows[r + h] : row_limit;
          const bool ok = row < row_limit;
          const uint4* q = reinterpret_cast<const uint4*>(p.A + (long long)(ok ? row : 0) * p.lda);
#pragma unroll
          for (int i = 0; i < GPT; ++i) {
            const int g = tid + NT * i;
            if (ok && g < G) {
              v[h][i][0] = __ldg(q + 2 * g);
              v[h][i][1] = __ldg(q + 2 * g + 1);
            } else {
              v[h][i][0] = v[h][i][1] = make_uint4(0u, 0u, 0u, 0u);
            }
          }
        }
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int i = 0; i < GPT; ++i) m[i] |= nz_mask8(v[h][i][0]) | (nz_mask8(v[h][i][1]) << 8);
      }
    }
#pragma unroll
    for (int i = 0; i < GPT; ++i)
      if (tid + NT * i < 128) s_mask[tid + NT * i] = static_cast<uint16_t>(m[i]);
    __syncwarp();
    int nag = 0;
#pragma unroll
    for (int c = 0; c < 4; ++c) {  // warp0 publishes one ordered list
      const int g = 32 * c + lane;
      const uint32_t m = s_mask[g];
      const bool a = g < G && m != 0u;
      const unsigned ball = __ballot_sync(0xFFFFFFFFu, a);
      if (a) s_list[nag + __popc(ball & ((1u << lane) - 1u))] = (m << 16) | static_cast<uint32_t>(g);
      nag += __popc(ball);
    }
    if (lane == 0) s_nag = nag;
  }
  __syncthreads();
  const int nag = s_nag;
  const unsigned long long t1 = p.trace ? gtimer() : 0ull;
  const int e = s_e;
  // Original lane's 8-byte raw word split into its low/high 4-byte halves.
  const uint8_t* wb = p.W + e * p.w_expert + P * p.w_tile + g8 * 8 + 4 * half;
  const uint8_t* wr[4];
  wr[0] = wb + (2 * t) * p.w_k;
  wr[1] = wr[0] + p.w_k;
  wr[2] = wr[0] + 8 * p.w_k;
  wr[3] = wr[2] + p.w_k;
  const uint8_t* sb = p.S + e * p.s_expert + P * p.s_tile + g8 * 16 + 8 * half;
  const unsigned sg = static_cast<unsigned>(p.s_g);
  const unsigned gw = static_cast<unsigned>(16 * p.w_k);
  const int xrow_i = g8 < nrows ? s_rows[g8] : -1;
  const uint32_t xok = (xrow_i >= 0 && xrow_i < row_limit) ? 1u : 0u;
  const uint8_t* xb = reinterpret_cast<const uint8_t*>(p.A + (long long)(xok ? xrow_i : 0) * p.lda + 2 * t);
  float acc[4][4];
  lean64_main(acc, p, s_list, nag, wr, sb, xb, gw, sg, xok, t, dense0, half);
  if (p.trace && threadIdx.x == 0) {
    unsigned long long* tr = p.trace + 4ull * (mb * 21 + P);
    tr[0] = t0;
    tr[1] = t1;
    tr[2] = gtimer();
    tr[3] = static_cast<unsigned long long>(nag) | (static_cast<unsigned long long>(nrows) << 16) |
            (static_cast<unsigned long long>(smid()) << 24);
  }
  const float gs4 = mul_ftz(p.GS[e], 4.0f);
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int col = (P * 2 + half) * 64 + 16 * c + g8;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int tok = 2 * t + (q & 1);
      const int cc = col + ((q >> 1) ? 8 : 0);
      if (tok < nrows) {
        const int row = s_rows[tok];
        const uint32_t v = to_bf16x2(mul_ftz(acc[c][q], gs4), 0.0f);
        reinterpret_cast<uint16_t*>(p.C)[(long long)row * p.ldc + cc] =
            row < row_limit ? static_cast<uint16_t>(v) : uint16_t(0);
      }
    }
  }
}

#endif  // ZSKIP_NO_PAIR

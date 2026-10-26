// Lean paired decode (configs 600..): config 119's per-output arithmetic (layout "swap2", one warp per
// 128-column tile pair, per output element the same MMA chain over the block's active k16 groups in
// ascending order, the same dequantisation values, the same epilogue) with a shorter per-group
// instruction stream and an L2 prefetch of whole weight rows:
//  - prep inside the CTA without k lists: the block's rows by ballot, one 16-bit mask per k16 group
//    (two rows' activation loads per round trip), the active groups compacted by ballot into one
//    32-bit (mask << 16 | group) entry each; one warp per CTA: no CTA barrier, no global block info;
//  - per-lane row pointers fixed per warp, one address add per load, a skipped k row's load leaves
//    +0 (no keep masks); an 8-row half whose mask byte is 0 is not dequantised and keeps the
//    fragments of the last non-empty half (finite: the fp4/fp8 bit decodes never give Inf/NaN), as
//    pzskip 336 keeps stale weights: every valid row of the block has a +-0 activation at those k,
//    so those products are +-0 as with 119's literal zeros (F_ZHALF: literal zeros as 119);
//  - the hi-nibble fragments decode x01 itself (fp4_to_bf16x2 reads only bits 12..15 / 28..31,
//    where x01 and x01 << 8 hold the same nibbles); the PRMT duplicates of the scales are folded into
//    the HMUL2 operand selectors (.H0_H0 / .H1_H1): the same multiplies on the same values;
//  - a ring of R + 1 register slots: group j + R is loaded into the slot freed by group j - 1, before
//    group j's math; config 119's -0 guard on every intermediate accumulator state, placed after
//    the next group's dequantisation (before its MMAs) and once after the loop; a warp that sees a
//    -0 recomputes its tiles with every row loaded (stock's computation). NR1 (one valid row) tests
//    token column 0 only (each output element's chain is independent of the other columns);
//  - F_WPF (default 653): with group j's loads each lane requests its own pieces of group j + R + D
//    into L2 (prefetch.global.L2; no registers), so a warp has R groups in registers and D more in
//    flight. F_BPF / F_RPF instead have the warp with P == j % 21 fetch group j + D's whole rows for
//    the block (faster alone, slower beside the decode's concurrent l2pf and shared-expert GEMM).
#ifndef ZSKIP_NO_PAIR

constexpr int F_ZHALF = 1, F_NOSPLIT = 2, F_BPF = 4, F_RPF = 8, F_WPF = 16;
// F_WPF: with group j's loads, each lane requests its own pieces of group j + R + D into L2 (the
// k rows of its mask bits, 8 bytes each, and its 16 scale bytes)
// prefetch distance code in F bits 8..11: 0 none, 1..7 -> 2 * code groups, 8 -> nag / 16, 9 -> nag / 8,
// 10 -> 1 + nag / 32, 11 -> 1, 12 -> 3, 13 -> 5 (each at least 1)
template <int F>
__device__ __forceinline__ int prefetch_distance(int nag) {
  constexpr int f = (F >> 8) & 15;
  int d = f <= 7 ? 2 * f : (f == 8 ? nag >> 4 : (f == 9 ? nag >> 3 : (f == 10 ? 1 + (nag >> 5) : (f == 12 ? 3 : (f == 13 ? 5 : 1)))));
  return d < 1 ? 1 : d;
}
__device__ __forceinline__ void prefetch_l2(const void* ptr) {
  asm volatile("prefetch.global.L2 [%0];\n" ::"l"(ptr));
}
__device__ __forceinline__ void prefetch_l2_pred(const void* ptr, uint32_t pred) {
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %1, 0;\n @q prefetch.global.L2 [%0];\n}\n" ::"l"(ptr), "r"(pred));
}
__device__ __forceinline__ void bulk_prefetch_l2(const void* ptr, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(ptr), "r"(bytes) : "memory");
}
// a skipped row's load leaves +0
__device__ __forceinline__ void ldg64_z(uint32_t& x, uint32_t& y, const void* ptr, uint32_t pred) {
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %3, 0;\n mov.b32 %0, 0;\n mov.b32 %1, 0;\n"
               " @q ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];\n}\n"
               : "=r"(x), "=r"(y) : "l"(ptr), "r"(pred));
}
__device__ __forceinline__ void ldg32l1_z(uint32_t& x, const void* ptr, uint32_t pred) {
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %2, 0;\n mov.b32 %0, 0;\n @q ld.global.nc.u32 %0, [%1];\n}\n"
               : "=r"(x) : "l"(ptr), "r"(pred));
}
__device__ __forceinline__ void ldg128_na(uint32_t (&v)[4], const void* ptr) {
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "l"(ptr));
}

constexpr int COMPUTEONLY = 5;  // timing probe: lean kernel without the weight/scale/activation loads

// a * (low or high half of s duplicated): the PRMT dup of 119 folded into the HMUL2 operand selector
__device__ __forceinline__ uint32_t hmul2_lo(uint32_t a, uint32_t s) {
  const __nv_bfloat162 r = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&a),
                                   __low2bfloat162(*reinterpret_cast<const __nv_bfloat162*>(&s)));
  return *reinterpret_cast<const uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t hmul2_hi(uint32_t a, uint32_t s) {
  const __nv_bfloat162 r = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&a),
                                   __high2bfloat162(*reinterpret_cast<const __nv_bfloat162*>(&s)));
  return *reinterpret_cast<const uint32_t*>(&r);
}

// -0 test of the valid accumulator elements (NR1: token column 0 only)
template <bool NR1>
__device__ __forceinline__ void neg0_check(bool& neg0, const float (&acc)[8][4]) {
#pragma unroll
  for (int c = 0; c < 8; ++c) {
#pragma unroll
    for (int q = 0; q < 4; ++q)
      if (!NR1 || (q & 1) == 0) neg0 |= __float_as_uint(acc[c][q]) == 0x80000000u;
  }
}


// one block's share of group ent's rows into L2: each active k row (1344 bytes) and the scale row
template <int F>
__device__ __forceinline__ void group_prefetch(const Params& p, const uint8_t* wexp, const uint8_t* sexp, uint32_t ent) {
  const int grp = static_cast<int>(ent & 0xFFFFu);
  const int lane = threadIdx.x & 31;
  if constexpr ((F & F_RPF) != 0) {
    uint32_t m = ent >> 16;
    while (m) {
      const int b = __ffs(m) - 1;
      m &= m - 1u;
      if (lane < 11) prefetch_l2(wexp + (long long)(16 * grp + b) * p.w_k + 128 * lane);
    }
    if (lane < 21) prefetch_l2(sexp + (long long)grp * p.s_g + 128 * lane);
  } else {
    if (lane < 16 && ((ent >> (16 + lane)) & 1u))
      bulk_prefetch_l2(wexp + (long long)(16 * grp + lane) * p.w_k, 1344u);
    else if (lane == 16)
      bulk_prefetch_l2(sexp + (long long)grp * p.s_g, 2688u);
  }
}

// One k16 group of the warp's two 64-column tiles. The -0 test of the previous group's accumulators
// sits after this group's dequantisation, just before its MMAs (it must read them before they are
// overwritten); the caller tests the final state after the loop, so every state is tested.
template <bool ZH, bool NR1>
__device__ __forceinline__ void lean_group(float (&acc)[8][4], uint32_t (&alo)[8][2], uint32_t (&ahi)[8][2],
                                           const uint32_t (&w)[8], const uint32_t (&s)[4],
                                           const uint32_t (&x)[2], uint32_t m, bool& neg0) {
  uint32_t sc[8];
#pragma unroll
  for (int tt = 0; tt < 2; ++tt) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint32_t sxx = c < 2 ? s[2 * tt] : s[2 * tt + 1];
      const uint32_t sp = hmul2_u32(fp8_to_bf16x2(__byte_perm(sxx, 0u, (c & 1) == 0 ? 0x1404 : 0x3424)), kExp120);
      sc[tt * 4 + c] = sp;  // low half: column g8, high half: column g8 + 8 (selected in hmul2_lo/hi)
    }
  }
  if (m & 0x00FFu) {
#pragma unroll
    for (int tt = 0; tt < 2; ++tt) {
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const uint32_t sel = c | (c << 4) | ((c + 4) << 8) | ((c + 4) << 12);
        const uint32_t x01 = __byte_perm(w[0 + tt], w[2 + tt], sel);
        alo[tt * 4 + c][0] = hmul2_lo(hmul2_u32(fp4_to_bf16x2(x01 << 12), kExp124), sc[tt * 4 + c]);
        alo[tt * 4 + c][1] = hmul2_hi(hmul2_u32(fp4_to_bf16x2(x01), kExp124), sc[tt * 4 + c]);
      }
    }
  } else if constexpr (ZH) {
#pragma unroll
    for (int i = 0; i < 8; ++i) alo[i][0] = alo[i][1] = 0u;
  }
  if (m & 0xFF00u) {
#pragma unroll
    for (int tt = 0; tt < 2; ++tt) {
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const uint32_t sel = c | (c << 4) | ((c + 4) << 8) | ((c + 4) << 12);
        const uint32_t x23 = __byte_perm(w[4 + tt], w[6 + tt], sel);
        ahi[tt * 4 + c][0] = hmul2_lo(hmul2_u32(fp4_to_bf16x2(x23 << 12), kExp124), sc[tt * 4 + c]);
        ahi[tt * 4 + c][1] = hmul2_hi(hmul2_u32(fp4_to_bf16x2(x23), kExp124), sc[tt * 4 + c]);
      }
    }
  } else if constexpr (ZH) {
#pragma unroll
    for (int i = 0; i < 8; ++i) ahi[i][0] = ahi[i][1] = 0u;
  }
  neg0_check<NR1>(neg0, acc);
#pragma unroll
  for (int i = 0; i < 8; ++i) mma_bf16(acc[i], alo[i][0], alo[i][1], ahi[i][0], ahi[i][1], x[0], x[1]);
}

// The main loop over the block's active groups (dense: all 116 groups, every row loaded).
template <int R, bool NR1, int F>
__device__ __forceinline__ void lean_main(float (&acc)[8][4], const Params& p, const uint32_t* list, int nag,
                                          const uint8_t* const (&wr)[4], const uint8_t* sb, const uint8_t* xb,
                                          unsigned gw, unsigned sg, uint32_t xok, int t, bool dense,
                                          const uint8_t* wexp, const uint8_t* sexp, int P) {
  const bool math = p.mode == DENSE || p.mode == SKIP || p.mode == COMPUTEONLY;
  const bool loads = p.mode != COMPUTEONLY;
  constexpr bool PF = (F & (F_BPF | F_RPF)) != 0 && ((F >> 8) & 15) != 0;
  constexpr bool WPF = (F & F_WPF) != 0 && ((F >> 8) & 15) != 0;
  for (int pass = 0; pass < 2; ++pass) {
    const int nsteps = dense ? G : nag;
    constexpr int NS = R + 1;
    uint32_t W_[NS][8], S_[NS][4], X_[NS][2], M_[NS];
#pragma unroll
    for (int i = 0; i < NS; ++i) {
#pragma unroll
      for (int q = 0; q < 8; ++q) W_[i][q] = 0u;
#pragma unroll
      for (int q = 0; q < 4; ++q) S_[i][q] = 0u;
      X_[i][0] = X_[i][1] = 0u;
      M_[i] = 0u;
    }
    auto issue = [&](int slot, int j) {
      const uint32_t ent = dense ? (0xFFFF0000u | static_cast<uint32_t>(j)) : list[j];
      const unsigned grp = ent & 0xFFFFu;
      const uint32_t mm = ent >> (16 + 2 * t);
      M_[slot] = ent >> 16;
      if (!loads) return;
      const unsigned long long go = static_cast<unsigned long long>(grp * gw);
      ldg64_z(W_[slot][0], W_[slot][1], wr[0] + go, mm & 1u);
      ldg64_z(W_[slot][2], W_[slot][3], wr[1] + go, mm & 2u);
      ldg64_z(W_[slot][4], W_[slot][5], wr[2] + go, mm & 0x100u);
      ldg64_z(W_[slot][6], W_[slot][7], wr[3] + go, mm & 0x200u);
      ldg128_na(S_[slot], sb + grp * sg);
      ldg32l1_z(X_[slot][0], xb + 32u * grp, xok);
      ldg32l1_z(X_[slot][1], xb + 32u * grp + 16u, xok);
    };
#pragma unroll
    for (int c = 0; c < 8; ++c) {
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[c][i] = 0.0f;
    }
    if constexpr (PF) {
      if (!dense && loads) {
        const int D = prefetch_distance<F>(nag);
        for (int jp = P; jp < R + D && jp < nsteps; jp += 21) group_prefetch<F>(p, wexp, sexp, list[jp]);
      }
    }
#pragma unroll
    for (int i = 0; i < R; ++i)
      if (i < nsteps) issue(i, i);
    uint32_t alo[8][2], ahi[8][2];  // the fragments of the last non-empty half
#pragma unroll
    for (int c = 0; c < 8; ++c) alo[c][0] = alo[c][1] = ahi[c][0] = ahi[c][1] = 0u;
    bool neg0 = false;
    for (int j0 = 0; j0 < nsteps; j0 += NS) {
#pragma unroll
      for (int i = 0; i < NS; ++i) {
        if (j0 + i < nsteps) {
          uint32_t w_[8], s_[4], x_[2];
          const uint32_t m_ = M_[i];
#pragma unroll
          for (int q = 0; q < 8; ++q) w_[q] = W_[i][q];
#pragma unroll
          for (int q = 0; q < 4; ++q) s_[q] = S_[i][q];
          x_[0] = X_[i][0];
          x_[1] = X_[i][1];
          if constexpr (PF) {
            const int jp = j0 + i + R + prefetch_distance<F>(nag);
            if (!dense && loads && jp < nsteps && jp % 21 == P) group_prefetch<F>(p, wexp, sexp, list[jp]);
          }
          if constexpr (WPF) {
            const int jp = j0 + i + R + prefetch_distance<F>(nag);
            if (!dense && loads && jp < nsteps) {
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
          }
          // refill before the math: the loads of group j + R start before group j's math
          if (j0 + i + R < nsteps) issue((i + R) % NS, j0 + i + R);
          if (math) {
            lean_group<(F & F_ZHALF) != 0, NR1>(acc, alo, ahi, w_, s_, x_, m_, neg0);
          } else if (p.mode == LOADONLY) {
            // consume the loads so that each slot waits for its data like the math path
            const uint32_t v = w_[0] ^ w_[3] ^ w_[4] ^ w_[7] ^ s_[0] ^ s_[3] ^ x_[0] ^ x_[1];
            if (v == 0x12345678u && p.stats) atomicAdd(p.stats + 3, 1);
          }
        }
      }
    }
    if (!math || dense || p.mode == COMPUTEONLY) break;
    neg0_check<NR1>(neg0, acc);  // the final state
    if (!__any_sync(0xFFFFFFFFu, neg0)) break;
    // a -0 accumulator appeared: recompute this warp's tiles with every row loaded
    if ((threadIdx.x & 31) == 0 && p.stats) atomicAdd(p.stats + 0, 1);
    dense = true;
  }
}

template <int WPC, int R, int F>
__global__ void __launch_bounds__(WPC * 32) down_lean_kernel(const Params p) {
  constexpr int NT = WPC * 32;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g8 = lane >> 2, t = lane & 3;
  const int mb = blockIdx.y;
  const int P = blockIdx.x * WPC + warp;  // 128-column tile pair, 0..20
  const unsigned long long t0 = p.trace ? gtimer() : 0ull;
#ifndef ZSKIP_NO_PDL
  pdl_wait();
  pdl_launch_dependents();
#endif
  if (mb * 16 >= *p.num_tokens_padded) return;
  __shared__ int s_rows[8];
  __shared__ int s_nrows, s_e;
  __shared__ uint16_t s_mask[128];
  __shared__ uint32_t s_list[WPC][G];
  auto bar = [&]() {
    if constexpr (WPC == 1) __syncwarp(); else __syncthreads();
  };
  if (warp == 0) {
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
  bar();
  const int nrows = s_nrows;
  if (nrows == 0) return;
  const int row_limit = prefix_row_limit(p);
  const bool dense0 = p.mode == DENSE || p.mode == LOADONLY_DENSE;
  // masks: bit b of group g set when a valid row below row_limit has a nonzero |a| at k = 16g + b
  {
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
  }
  bar();
  int nag = 0;
#pragma unroll
  for (int c = 0; c < 4; ++c) {  // each warp: its own ordered list of active groups
    const int g = 32 * c + lane;
    const uint32_t m = s_mask[g];
    const bool a = g < G && m != 0u;
    const unsigned ball = __ballot_sync(0xFFFFFFFFu, a);
    if (a) s_list[warp][nag + __popc(ball & ((1u << lane) - 1u))] = (m << 16) | static_cast<uint32_t>(g);
    nag += __popc(ball);
  }
  __syncwarp();
  const unsigned long long t1 = p.trace ? gtimer() : 0ull;
  if (p.mode == SETUPONLY) return;
  const int e = s_e;
  const uint32_t* list = s_list[warp];
  const uint8_t* wb = p.W + e * p.w_expert + P * p.w_tile + g8 * 8;
  const uint8_t* wr[4];
  wr[0] = wb + (2 * t) * p.w_k;
  wr[1] = wr[0] + p.w_k;
  wr[2] = wr[0] + 8 * p.w_k;
  wr[3] = wr[2] + p.w_k;
  const uint8_t* sb = p.S + e * p.s_expert + P * p.s_tile + g8 * 16;
  const unsigned sg = static_cast<unsigned>(p.s_g);
  const unsigned gw = static_cast<unsigned>(16 * p.w_k);
  const int xrow_i = g8 < nrows ? s_rows[g8] : -1;
  const uint32_t xok = (xrow_i >= 0 && xrow_i < row_limit) ? 1u : 0u;
  const uint8_t* xb = reinterpret_cast<const uint8_t*>(p.A + (long long)(xok ? xrow_i : 0) * p.lda + 2 * t);
  const bool math = p.mode == DENSE || p.mode == SKIP || p.mode == COMPUTEONLY;
  float acc[8][4];
  const uint8_t* wexp = p.W + e * p.w_expert;
  const uint8_t* sexp = p.S + e * p.s_expert;
  if (!(F & F_NOSPLIT) && nrows == 1)
    lean_main<R, true, F>(acc, p, list, nag, wr, sb, xb, gw, sg, xok, t, dense0, wexp, sexp, P);
  else
    lean_main<R, false, F>(acc, p, list, nag, wr, sb, xb, gw, sg, xok, t, dense0, wexp, sexp, P);
  if (p.trace && lane == 0) {
    unsigned long long* tr = p.trace + 4ull * (mb * 21 + P);
    tr[0] = t0;
    tr[1] = t1;
    tr[2] = gtimer();
    tr[3] = static_cast<unsigned long long>(nag) | (static_cast<unsigned long long>(nrows) << 16) |
            (static_cast<unsigned long long>(smid()) << 24);
  }
  if (!math || p.mode == COMPUTEONLY) return;
  const float gs4 = mul_ftz(p.GS[e], 4.0f);
#pragma unroll
  for (int tt = 0; tt < 2; ++tt) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int col = (P * 2 + tt) * 64 + 16 * c + g8;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int tok = 2 * t + (q & 1);
        const int cc = col + ((q >> 1) ? 8 : 0);
        if (tok < nrows) {
          const int row = s_rows[tok];
          const uint32_t v = to_bf16x2(mul_ftz(acc[tt * 4 + c][q], gs4), 0.0f);  // humming's cvt.rn.bf16x2.f32
          reinterpret_cast<uint16_t*>(p.C)[(long long)row * p.ldc + cc] =
              row < row_limit ? static_cast<uint16_t>(v) : uint16_t(0);
        }
      }
    }
  }
}

#endif  // ZSKIP_NO_PAIR

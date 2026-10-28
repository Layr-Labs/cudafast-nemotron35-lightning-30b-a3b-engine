// SPDX-License-Identifier: Apache-2.0
// Experimental routed down+sum. The model admits this path only after its
// remote per-layer real-weight, graph and complete-chain timing checks. The
// checked original down and weighted combine remain the fallback.
//
// Intended inclusion point is inside namespace zskip after Params and the
// PDL helpers. A slot is one flattened (token, top-k slot) row of p.A/p.C.
// The map must reproduce the actual sorted block's first-eight selection,
// not assume sorted order or use topk_ids as an expert index directly.

#ifndef ZSKIP_NO_PAIR
enum RoutedSlotKind : int {
  ROUTED_UNASSIGNED = 0,  // original down leaves this output slot untouched
  ROUTED_COMPUTE = 1,     // original down writes an actual BF16 result
  ROUTED_PREFIX_ZERO = 2, // original down writes +0 beyond the live prefix
  ROUTED_AMBIGUOUS = 3,   // duplicate mapping; proposed path must fall back
};

struct RoutedSlotMap {
  int expert;
  int kind;
  int block;
  int rank;
  int nrows;
};

// Cached once per original expert block, not once per output tile or routed
// token. Rows and active group order match down_lean_kernel<1,3,306>.
// The caller must provide max_blocks records, keep them alive throughout the
// graph replay, and serialize this producer before all consumers.
struct RoutedBlockMeta {
  int rows[8];
  int nrows;
  int nag;
  uint32_t active[G];
};

__global__ void __launch_bounds__(32)
prepare_routed_block_meta(const Params p, RoutedBlockMeta* blocks) {
#ifndef ZSKIP_NO_PDL
  pdl_wait();
#endif
  const int lane = threadIdx.x;
  const int mb = blockIdx.x;
  __shared__ int rows[8];
  __shared__ int nrows;
  if (mb * 16 >= *p.num_tokens_padded) {
    if (lane == 0) { blocks[mb].nrows = 0; blocks[mb].nag = 0; }
#ifndef ZSKIP_NO_PDL
    pdl_launch_dependents();
#endif
    return;
  }
  const unsigned id = lane < 16
      ? static_cast<unsigned>(p.sorted_ids[mb * 16 + lane]) : 0xffffffffu;
  const bool valid = id < static_cast<unsigned>(p.shape_m);
  const unsigned ball = __ballot_sync(0xffffffffu, valid);
  const int rank = __popc(ball & ((1u << lane) - 1u));
  if (valid && rank < 8) rows[rank] = static_cast<int>(id);
  if (lane == 0) { const int n = __popc(ball); nrows = n < 8 ? n : 8; }
  __syncwarp();
  if (lane < nrows) blocks[mb].rows[lane] = rows[lane];
  if (lane == 0) blocks[mb].nrows = nrows;
  const int row_limit = prefix_row_limit(p);
  const bool dense0 = p.mode == DENSE || p.mode == LOADONLY_DENSE;
  int nag = 0;
  for (int c = 0; c < 4; ++c) {
    const int group = 32 * c + lane;
    uint32_t mask = 0;
    if (group < G && dense0) mask = 0xffffu;
    else if (group < G) {
      for (int r = 0; r < nrows; ++r) {
        const int row = rows[r];
        if (row < row_limit) {
          const uint4* a = reinterpret_cast<const uint4*>(
              p.A + static_cast<long long>(row) * p.lda);
          mask |= nz_mask8(__ldg(a + 2 * group)) |
                  (nz_mask8(__ldg(a + 2 * group + 1)) << 8);
        }
      }
    }
    const bool active = group < G && mask != 0;
    const unsigned active_ballot = __ballot_sync(0xffffffffu, active);
    if (active) {
      const int offset = __popc(active_ballot & ((1u << lane) - 1u));
      blocks[mb].active[nag + offset] = (mask << 16) | group;
    }
    nag += __popc(active_ballot);
  }
  if (lane == 0) blocks[mb].nag = nag;
#ifndef ZSKIP_NO_PDL
  pdl_launch_dependents();
#endif
}

// One warp maps one flattened expert slot. All lanes participate in each
// ballot; lanes 16..31 are padding, matching down_lean_kernel's selection.
// The caller must arrange a device-safe whole-operation fallback if ANY map
// reports ROUTED_AMBIGUOUS. This kernel alone never authorizes fast execution.
__global__ void __launch_bounds__(192)
prepare_routed_slot_map(const Params p, RoutedSlotMap* map) {
#ifndef ZSKIP_NO_PDL
  pdl_wait();
#endif
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 6 + warp;
  if (row >= p.shape_m) {
#ifndef ZSKIP_NO_PDL
    pdl_launch_dependents();
#endif
    return;
  }
  int expert = -1;
  int owner_block = -1, owner_rank = -1, owner_nrows = 0;
  int matches = 0;
  const int padded = *p.num_tokens_padded;
  for (int mb = 0; mb < p.max_blocks && mb * 16 < padded; ++mb) {
    const unsigned id = lane < 16
        ? static_cast<unsigned>(p.sorted_ids[mb * 16 + lane]) : 0xffffffffu;
    const bool valid = id < static_cast<unsigned>(p.shape_m);
    const unsigned valid_ballot = __ballot_sync(0xffffffffu, valid);
    const int rank = __popc(valid_ballot & ((1u << lane) - 1u));
    const bool selected = valid && rank < 8 && id == static_cast<unsigned>(row);
    const unsigned selected_ballot = __ballot_sync(0xffffffffu, selected);
    if (selected_ballot) {
      const int count = __popc(selected_ballot);
      if (matches == 0) {
        const int selected_lane = __ffs(selected_ballot) - 1;
        expert = p.expert_ids[mb];
        owner_block = mb;
        owner_rank = __shfl_sync(0xffffffffu, rank, selected_lane);
        const int n = __popc(valid_ballot);
        owner_nrows = n < 8 ? n : 8;
      }
      matches += count;
    }
  }
  if (lane == 0) {
    RoutedSlotMap result;
    result.expert = expert;
    result.block = owner_block;
    result.rank = owner_rank;
    result.nrows = owner_nrows;
    result.kind = matches > 1 ? ROUTED_AMBIGUOUS
        : matches == 0 ? ROUTED_UNASSIGNED
        : row >= prefix_row_limit(p) ? ROUTED_PREFIX_ZERO : ROUTED_COMPUTE;
    map[row] = result;
  }
#ifndef ZSKIP_NO_PDL
  pdl_launch_dependents();
#endif
}

// One CTA produces a replay-local branch flag after BOTH metadata producers
// have completed. The fallback launch must consult this same flag and use the
// original down+combine path; the fast launch below must skip on fallback.
// Use an ordinary stream dependency or a correctly wired PDL wait between
// kernels. A stale flag or a per-slot fallback is not sufficient.
__global__ void __launch_bounds__(32)
classify_routed_operation(const Params p, const RoutedSlotMap* map,
                          const RoutedBlockMeta* blocks, int* fallback,
                          int force_fallback) {
#ifndef ZSKIP_NO_PDL
  pdl_wait();
#endif
  const int lane = threadIdx.x;
  bool bad = false;
  for (int row = lane; row < p.shape_m; row += 32) {
    const RoutedSlotMap m = map[row];
    if (m.kind == ROUTED_AMBIGUOUS || m.kind < ROUTED_UNASSIGNED ||
        m.kind > ROUTED_PREFIX_ZERO) bad = true;
    if (m.kind == ROUTED_COMPUTE || m.kind == ROUTED_PREFIX_ZERO) {
      if (m.expert < 0 || m.expert >= 128 || m.block < 0 ||
          m.block >= p.max_blocks || m.rank < 0 || m.rank >= m.nrows ||
          m.nrows < 1 || m.nrows > 8) {
        bad = true;
      } else {
        const RoutedBlockMeta* meta = blocks + m.block;
        if (meta->nrows != m.nrows || meta->rows[m.rank] != row ||
            meta->nag < 0 || meta->nag > G) bad = true;
      }
    }
  }
  const bool any_bad = __any_sync(0xffffffffu, bad);
  if (lane == 0) *fallback = (any_bad || force_fallback) ? 1 : 0;
#ifndef ZSKIP_NO_PDL
  pdl_launch_dependents();
#endif
}

// Exact config-663 baseline body, executed only on the classified fallback.
// This independent copy is uncompiled and must be compared to the original
// whenever the base changes. Graph launch ordering is the caller's duty.
__global__ void __launch_bounds__(32)
guarded_routed_baseline_prototype(const Params p, const int* fallback) {
  constexpr int WPC = 1, R = 3, F = 306;
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
  if (!*fallback) return;
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
  const uint8_t* sb = p.S + e * p.s_expert + P * p.s_tile + ((F & F_S6) != 0 ? g8 * 8 : g8 * 16);
  const uint8_t* sb2 = p.S + e * p.s_expert + P * p.s_tile + 64 + g8 * 4;  // F_S6 only
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
    lean_main<R, true, F>(acc, p, list, nag, wr, sb, xb, gw, sg, xok, t, dense0, wexp, sexp, P, sb2);
  else
    lean_main<R, false, F>(acc, p, list, nag, wr, sb, xb, gw, sg, xok, t, dense0, wexp, sexp, P, sb2);
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

// Prototype only: one CTA owns (token, 128 output columns); its six warps
// compute the six expert slots, stage the original BF16 epilogues in shared
// memory, and warp zero reduces slots in their original order. The final
// tensor must be separate from p.A and p.C: the caller later copies it to the
// contract output only after the entire producer grid has completed.
//
// The whole-operation classifier and guarded baseline are experimental.
// Remote admission compares fma and skip semantics to the real chain,
// including NaNs, signed zero, and effects on the down-output workspace.
__global__ void __launch_bounds__(192)
routed_down_sum_prototype(const Params p, const RoutedSlotMap* map,
                          const RoutedBlockMeta* blocks,
                          const int* fallback, const float* topk_weights,
                          __nv_bfloat16* scratch) {
#ifndef ZSKIP_NO_PDL
  pdl_wait();
#endif
  if (*fallback) {
#ifndef ZSKIP_NO_PDL
    pdl_launch_dependents();
#endif
    return;
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int token = blockIdx.y;
  const int pair = blockIdx.x;
  const int row = token * 6 + warp;
  const int g8 = lane >> 2;
  const int t = lane & 3;
  __shared__ uint32_t active_list[6][G];
  __shared__ int original_rows[6][8];
  __shared__ uint16_t parts[6][128];
  const RoutedSlotMap owner = map[row];

  if (owner.kind == ROUTED_COMPUTE && owner.expert >= 0 && owner.expert < 128) {
    // Reuse the original block's rows and ordered union mask. Each token CTA
    // still recomputes the full block's MMA, so the overall tradeoff is unknown.
    const RoutedBlockMeta* meta = blocks + owner.block;
    if (lane < owner.nrows) original_rows[warp][lane] = meta->rows[lane];
    const int nag = meta->nag;
    for (int i = lane; i < nag; i += 32) active_list[warp][i] = meta->active[i];
    __syncwarp();
    const int row_limit = prefix_row_limit(p);
    const bool dense0 = p.mode == DENSE || p.mode == LOADONLY_DENSE;
    const int e = owner.expert;
    const uint8_t* wb = p.W + e * p.w_expert + pair * p.w_tile + g8 * 8;
    const uint8_t* wr[4];
    wr[0] = wb + (2 * t) * p.w_k;
    wr[1] = wr[0] + p.w_k;
    wr[2] = wr[0] + 8 * p.w_k;
    wr[3] = wr[2] + p.w_k;
    const uint8_t* sb = p.S + e * p.s_expert + pair * p.s_tile + g8 * 8;
    const uint8_t* sb2 = p.S + e * p.s_expert + pair * p.s_tile + 64 + g8 * 4;
    const int xrow_i = g8 < owner.nrows ? original_rows[warp][g8] : -1;
    const uint32_t xok = xrow_i >= 0 && xrow_i < row_limit ? 1u : 0u;
    const uint8_t* xb = reinterpret_cast<const uint8_t*>(
        p.A + static_cast<long long>(xok ? xrow_i : 0) * p.lda + 2 * t);
    const uint8_t* wexp = p.W + e * p.w_expert;
    const uint8_t* sexp = p.S + e * p.s_expert;
    float acc[8][4];
    // Config 663 is (WPC=1, R=3, flags=306); NOSPLIT keeps NR1=false.
    lean_main<3, false, 306>(acc, p, active_list[warp], nag, wr, sb, xb,
                             static_cast<unsigned>(16 * p.w_k),
                             static_cast<unsigned>(p.s_g), xok, t, dense0,
                             wexp, sexp, pair, sb2);
    const float gs4 = mul_ftz(p.GS[e], 4.0f);
    for (int tt = 0; tt < 2; ++tt) {
      for (int c = 0; c < 4; ++c) {
        const int col = tt * 64 + 16 * c + g8;
        for (int q = 0; q < 4; ++q) {
          const int tok = 2 * t + (q & 1);
          const int cc = col + ((q >> 1) ? 8 : 0);
          if (tok == owner.rank) {
            const uint32_t v = to_bf16x2(mul_ftz(acc[tt * 4 + c][q], gs4), 0.f);
            parts[warp][cc] = static_cast<uint16_t>(v);
            // Preserve down_output's observable workspace contents for later
            // operations; fast path removes the combine read, not this write.
            reinterpret_cast<uint16_t*>(p.C)[
                static_cast<long long>(row) * p.ldc + pair * 128 + cc] =
                static_cast<uint16_t>(v);
          }
        }
      }
    }
  } else {
    // No valid assignment leaves the original p.C row unchanged. A selected
    // row beyond the live prefix is explicitly +0 in the original down path.
    for (int j = lane; j < 128; j += 32) {
      parts[warp][j] = owner.kind == ROUTED_PREFIX_ZERO ? uint16_t(0)
          : reinterpret_cast<const uint16_t*>(p.C)[
              static_cast<long long>(row) * p.ldc + pair * 128 + j];
      if (owner.kind == ROUTED_PREFIX_ZERO)
        reinterpret_cast<uint16_t*>(p.C)[
            static_cast<long long>(row) * p.ldc + pair * 128 + j] = 0;
    }
  }
  __syncthreads();
  if (warp == 0) {
    for (int j = lane; j < 128; j += 32) {
      float sum = 0.f;
      for (int slot = 0; slot < 6; ++slot) {
        const __nv_bfloat16 value = *reinterpret_cast<const __nv_bfloat16*>(
            &parts[slot][j]);
        sum = __fmaf_rn(__bfloat162float(value), topk_weights[token * 6 + slot], sum);
      }
      scratch[static_cast<long long>(token) * N + pair * 128 + j] =
          __float2bfloat16_rn(sum);
    }
  }
#ifndef ZSKIP_NO_PDL
  pdl_launch_dependents();
#endif
}

// After the fast grid or guarded original down grid finishes, write the
// contract output. Input activation and contract output may alias, so this
// kernel MUST NOT begin before the complete chosen producer grid completes.
// Fallback uses the original down buffer in the original top-k order; the
// fast path copies its disjoint scratch. Caller must preserve _l2pf.join and
// only use this prototype with contiguous fp32 weights and top-k == 6.
__global__ void __launch_bounds__(128)
finish_routed_sum_prototype(const Params p, const int* fallback,
                            const float* topk_weights,
                            const __nv_bfloat16* scratch,
                            __nv_bfloat16* output) {
#ifndef ZSKIP_NO_PDL
  pdl_wait();
#endif
  const int token = blockIdx.y;
  const int col = blockIdx.x * 128 + threadIdx.x;
  if (*fallback) {
    float acc = 0.f;
    for (int slot = 0; slot < 6; ++slot) {
      const __nv_bfloat16 v = p.C[(static_cast<long long>(token) * 6 + slot) * p.ldc + col];
      acc = __fmaf_rn(__bfloat162float(v), topk_weights[token * 6 + slot], acc);
    }
    output[static_cast<long long>(token) * N + col] = __float2bfloat16_rn(acc);
  } else {
    output[static_cast<long long>(token) * N + col] =
        scratch[static_cast<long long>(token) * N + col];
  }
#ifndef ZSKIP_NO_PDL
  pdl_launch_dependents();
#endif
}
#endif

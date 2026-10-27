// SPDX-License-Identifier: Apache-2.0
// Config693: one originalNT32 mask/list producer per expertblock.
// Original653 one-warp/R3/F274 math consumes this same-call metadata.
// DENSE/SKIP only; qualification and execution run only on the official remote.
#ifndef ZSKIP_NO_PAIR

struct LeanPrepared {
  int rows[8];
  int nrows, expert, nag;
  uint32_t list[G];
  unsigned long long start, ready;
};
static_assert(sizeof(LeanPrepared) <= BI_BYTES, "existing per-block workspace capacity");

__global__ void __launch_bounds__(32) lean_prepare_kernel(const Params p) {
  constexpr int NT = 32;
  const int tid = threadIdx.x, lane = tid;
  const int mb = blockIdx.x;
  const unsigned long long t0 = p.trace ? gtimer() : 0ull;
#ifndef ZSKIP_NO_PDL
  pdl_wait();
  pdl_launch_dependents();
#endif
  LeanPrepared* meta = reinterpret_cast<LeanPrepared*>(p.ws + (long long)mb * BI_BYTES);
  if (lane < 8) meta->rows[lane] = 0;
  if (mb * 16 >= *p.num_tokens_padded) {
    if (lane == 0) meta->nrows = meta->nag = meta->expert = 0;
    return;
  }
  __shared__ int s_rows[8], s_nrows;
  __shared__ uint16_t s_mask[128];
  const unsigned id = lane < 16 ? static_cast<unsigned>(p.sorted_ids[mb * 16 + lane]) : 0xFFFFFFFFu;
  const bool valid = id < static_cast<unsigned>(p.shape_m);
  const unsigned ball = __ballot_sync(0xFFFFFFFFu, valid);
  const int rank = __popc(ball & ((1u << lane) - 1u));
  __syncwarp();  // finish unused-row initialization before publishing valid rows
  if (valid && rank < 8) {
    s_rows[rank] = static_cast<int>(id);
    meta->rows[rank] = static_cast<int>(id);
  }
  if (lane == 0) {
    const int n = __popc(ball);
    s_nrows = n < 8 ? n : 8;
    meta->nrows = s_nrows;
    meta->expert = p.expert_ids[mb];
    // Old653 did this once on pair0; this producer runs once perexpertblock.
    if (n > 8 && p.stats) atomicAdd(p.stats + 1, 1);
  }
  __syncwarp();
  const int nrows = s_nrows;
  if (nrows == 0) {
    if (lane == 0) meta->nag = 0;
    return;
  }
  const int row_limit = prefix_row_limit(p);
  const bool dense0 = p.mode == DENSE;
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
  __syncwarp();
  int nag = 0;
#pragma unroll
  for (int c = 0; c < 4; ++c) {  // one producer writes one ordered list
    const int g = 32 * c + lane;
    const uint32_t m = s_mask[g];
    const bool a = g < G && m != 0u;
    const unsigned ball = __ballot_sync(0xFFFFFFFFu, a);
    if (a) meta->list[nag + __popc(ball & ((1u << lane) - 1u))] = (m << 16) | static_cast<uint32_t>(g);
    nag += __popc(ball);
  }
  __syncwarp();
  if (lane == 0) {
    meta->nag = nag;
    if (p.trace) {
      meta->start = t0;
      meta->ready = gtimer();
    }
  }
}

__global__ void __launch_bounds__(32) down_lean_prepared_kernel(const Params p) {
  constexpr int F = 274, R = 3;
  const int lane = threadIdx.x;
  const int g8 = lane >> 2, t = lane & 3;
  const int mb = blockIdx.y, P = blockIdx.x;
#ifndef ZSKIP_NO_PDL
  pdl_wait();
  pdl_launch_dependents();
#endif
  if (mb * 16 >= *p.num_tokens_padded) return;
  const LeanPrepared* meta = reinterpret_cast<const LeanPrepared*>(p.ws + (long long)mb * BI_BYTES);
  const int nrows = meta->nrows;
  if (nrows == 0) return;
  const int nag = meta->nag;
  const int row_limit = prefix_row_limit(p);
  const bool dense0 = p.mode == DENSE;
  const unsigned long long t0 = p.trace ? meta->start : 0ull;
  const unsigned long long t1 = p.trace ? meta->ready : 0ull;
  const int e = meta->expert;
  const uint32_t* list = meta->list;
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
  const int xrow_i = g8 < nrows ? meta->rows[g8] : -1;
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
          const int row = meta->rows[tok];
          const uint32_t v = to_bf16x2(mul_ftz(acc[tt * 4 + c][q], gs4), 0.0f);  // humming's cvt.rn.bf16x2.f32
          reinterpret_cast<uint16_t*>(p.C)[(long long)row * p.ldc + cc] =
              row < row_limit ? static_cast<uint16_t>(v) : uint16_t(0);
        }
      }
    }
  }
}

#endif  // ZSKIP_NO_PAIR

// Included inside namespace shx. Two original 64x128 units with identical full K segments.
// Each four-warp band keeps gemm2's original instruction/accumulator ownership. Only the
// immutable B/scale cp.async requests are shared; A, accumulators and scratch remain separate.
constexpr int kPairStageBytes = 2 * k2AStage + k2BStage + k2SStage;
constexpr int kPairScratchFloats = 2 * k2ScratchFloats;

template <bool Relu2>
__global__ void __launch_bounds__(256, 1) gemm2_pair_kernel(const Params2 q) {
  const Params& p = q.p;
  __shared__ __align__(128) uint8_t smem[k2Stages * kPairStageBytes];
  __shared__ int s_unit;
  const int tid = threadIdx.x;
  const int band = tid >> 7;
  const int lt = tid & 127;
  const int lane = lt & 31;
  const int warp = lt >> 5;
  const int rw = warp >> 1, cw = warp & 1;
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  float* stash = p.scratch + (static_cast<long long>(blockIdx.x) * 2 + band) * k2ScratchFloats + lt;
  float* rbuf = stash + 64 * 128;
  const float g = __ldg(p.GS);
  const long long nB = p.N >> 1;
  uint64_t pol = 0;
  if (q.hint) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));

  for (;;) {
    // All eight warps have finished the previous unit, including output/scratch stores.
    __syncthreads();
    if (tid == 0) s_unit = atomicAdd(q.work, 1);
    __syncthreads();
    const int u = s_unit;
    if (u >= p.n_units) break;
    const int* U = p.units + kUnitInts * u;
    const int row0 = U[0] + 64 * band, nrows = U[1], f0 = U[2], nseg = U[3];
    int lo[4], len[4];
    int nst = 0;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      lo[s] = U[4 + s];
      len[s] = s < nseg ? U[8 + s] - U[4 + s] : 0;
      nst += 2 * len[s];
    }
    const int col = f0 >> 7;
    const uint4* Bu = p.B + 64 * col;
    const uint8_t* Su = p.S + 128 * col;
    const __nv_bfloat16* Au = p.A + static_cast<long long>(row0) * p.lda;
    auto kb_of = [&](int st) -> int {
      int kb = 0;
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int n2 = 2 * len[s];
        if (st >= 0 && st < n2) {
          const int gg = st >= len[s] ? 1 : 0;
          kb = 4 * (lo[s] + st - gg * len[s]) + 2 * gg;
        }
        st -= n2;
      }
      return kb;
    };
    auto load = [&](int st) {
      const int kb = kb_of(st);
      const uint32_t slot = sbase + (st % k2Stages) * kPairStageBytes;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int idx = lt + 128 * i;
        const int r = idx >> 2, c = idx & 3;
        const bool ok = r < nrows;
        const __nv_bfloat16* src = Au + static_cast<long long>(ok ? r : 0) * p.lda + 16 * kb + 8 * c;
        const uint32_t dst = slot + band * k2AStage + r * 64 + ((c ^ ((r >> 1) & 3)) << 4);
        if (q.hint) cp16_hint(dst, src, ok, pol);
        else cp16(dst, src, ok);
      }
      // Unique B/S producers. Band 1's per-thread async wait alone is insufficient:
      // every producer waits below, then the full CTA barrier publishes both operands.
      if (band == 0) {
        const int kk = lt >> 6, i = lt & 63;
        cp16(slot + 2 * k2AStage + (kk * 64 + i) * 16, Bu + (kb + kk) * nB + i, true);
        if (lt < 16) {
          const int sk = lt >> 3, c = lt & 7;
          cp16(slot + 2 * k2AStage + k2BStage + sk * 128 + 16 * c,
               Su + static_cast<long long>(kb + sk) * p.N + 16 * c, true);
        }
      }
    };
#pragma unroll
    for (int s = 0; s < k2Stages - 1; ++s) {
      if (s < nst) load(s);
      cp_commit();
    }
    float acc[2][8][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int t = 0; t < 8; ++t)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[mi][t][e] = 0.0f;
    int seg = 0, base = 0;
    for (int st = 0; st < nst; ++st) {
      cp_wait<k2Stages - 2>();
      // Also prevents reusing a slot while either band is still consuming it.
      __syncthreads();
      if (st + k2Stages - 1 < nst) load(st + k2Stages - 1);
      cp_commit();
      const uint32_t slot = sbase + (st % k2Stages) * kPairStageBytes;
      stage_mma2(slot + band * k2AStage, slot + 2 * k2AStage,
                 slot + 2 * k2AStage + k2BStage, rw, cw, lane, acc);
      const int i = st - base;
      if (i + 1 == len[seg]) {
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
          for (int t = 0; t < 8; ++t)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              stash[((mi * 8 + t) * 4 + e) * 128] = acc[mi][t][e];
              acc[mi][t][e] = 0.0f;
            }
      } else if (i + 1 == 2 * len[seg]) {
        const bool first = seg == 0, last = seg == nseg - 1;
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
          for (int t = 0; t < 8; ++t)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const int x = ((mi * 8 + t) * 4 + e) * 128;
              float v = __fadd_rn(stash[x], acc[mi][t][e]);
              if (!first) v = __fadd_rn(v, rbuf[x]);
              if (!last) {
                rbuf[x] = v;
                v = 0.0f;
              }
              acc[mi][t][e] = v;
            }
        base += 2 * len[seg];
        ++seg;
      }
    }
    cp_wait<0>();
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
      const int r_lo = 32 * rw + 16 * mi + (lane >> 2);
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        const int f = f0 + 64 * cw + 8 * t + 2 * (lane & 3);
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int r = r_lo + 8 * hh;
          if (r < nrows) {
            __nv_bfloat162 v;
            if constexpr (Relu2) {
              v = round_output2_relu2(__fmul_rn(acc[mi][t][2 * hh], g),
                                      __fmul_rn(acc[mi][t][2 * hh + 1], g));
            } else {
              v.x = round_output<false>(__fmul_rn(acc[mi][t][2 * hh], g));
              v.y = round_output<false>(__fmul_rn(acc[mi][t][2 * hh + 1], g));
            }
            *reinterpret_cast<__nv_bfloat162*>(p.C + static_cast<long long>(row0 + r) * p.ldc + f) = v;
          }
        }
      }
    }
  }
  // The preceding full-CTA barrier is reached after every output store, even for the
  // final unit. Keep the original last-worker reset protocol, with a private pair counter.
  if (tid == 0) {
    __threadfence();
    const int d = atomicAdd(q.work + 1, 1);
    if (d == static_cast<int>(gridDim.x) - 1) {
      atomicExch(q.work, 0);
      atomicExch(q.work + 1, 0);
    }
  }
}

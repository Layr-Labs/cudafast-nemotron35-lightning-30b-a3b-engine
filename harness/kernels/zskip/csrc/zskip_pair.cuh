// Port of dadon 12f98e36 (31889a4), decode config 113 only. Main's BI layout and
// prep are deliberately retained. Fused=true composes fkiene bc524594's prep.
#ifndef ZSKIP_NO_PAIR
__device__ __forceinline__ uint4 ldg_u128(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::evict_first.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

// TPW = 64-column tiles per warp. TPW = 1 reads layout "swap"; TPW = 2 reads layout "swap2", where
// the two tiles of a 128-column pair are interleaved so a lane's words of both tiles are adjacent
// (one 8-byte load per k row: 64-byte segments per row, 16-byte scale loads).
// NB = token n-blocks of 8 per warp: an expert block of up to 8 * NB rows; the dequantised
// weight fragments of a group serve all NB n-blocks.
// HALF (TPW = 1): the warp's 64-column tile is one half of a "swap2" pair (4-byte words, 8-byte
// scales at the tile's offset in the pair) instead of a "swap" tile.
// HalfSkip (configs 114, 119): an 8-row half of a group (k 0..7 or 8..15) whose mask byte is zero
// has keep01 or keep23 zero on every lane, so its A fragments are +0 whatever the weights; they
// are not dequantised and the same MMAs take literal +0 there. m is the same on every lane.
template <bool Fused, bool HalfSkip>
__global__ void __launch_bounds__(32) down_pair_kernel(const Params p) {
  constexpr int TPW = 2, WPC = 1, R = 3, NB = 1;
  constexpr bool HALF = false;
  static_assert(TPW == 1 || TPW == 2, "tiles per warp");
  static_assert(NB >= 1 && NB <= 4, "n-blocks per warp");
  static_assert(!HALF || TPW == 1, "half pairs are single tiles");
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g8 = lane >> 2, t = lane & 3;
  const int mb = blockIdx.y;
  const int P = blockIdx.x * WPC + warp;  // tile (TPW = 1) or tile pair (TPW = 2)
#ifndef ZSKIP_NO_PDL
  pdl_wait();
  pdl_launch_dependents();
#endif
  if (mb * 16 >= *p.num_tokens_padded) return;
  const uint8_t* bi = p.ws + (long long)mb * BI_BYTES;
#ifndef ZSKIP_NO_FUSED
  extern __shared__ __align__(16) uint8_t smem_pair[];
  if constexpr (Fused) {
    fused_prep<32, 16, 8>(p, mb, smem_pair);
    bi = smem_pair;
  }
#endif
  const BlockHdr* hdr = reinterpret_cast<const BlockHdr*>(bi);
  const int nrows = hdr->nrows;
  if (nrows == 0 || p.mode == SETUPONLY) return;
  if (nrows > 8 * NB) {  // the launcher's config covers every block it is given; never taken
    if (lane == 0 && p.stats) atomicAdd(p.stats + 1, 1);
    return;
  }
  const int e = hdr->expert;
  const uint16_t* gmask = reinterpret_cast<const uint16_t*>(bi + BI_MASK);
  const uint16_t* gact = reinterpret_cast<const uint16_t*>(bi + BI_ACTG);
  const bool math = p.mode == DENSE || p.mode == SKIP;
  bool dense = p.mode == DENSE || p.mode == LOADONLY_DENSE;
  const int row_limit = prefix_row_limit(p);
  const __nv_bfloat16* xrow[NB];
#pragma unroll
  for (int nb = 0; nb < NB; ++nb) {
    const int r = nb * 8 + g8;
    xrow[nb] = r < nrows && hdr->rows[r] < row_limit
        ? p.A + (long long)hdr->rows[r] * p.lda : nullptr;
  }
  const uint8_t* wbase = HALF ? p.W + e * p.w_expert + (P >> 1) * 64 + g8 * 8 + (P & 1) * 4
                              : p.W + e * p.w_expert + P * (32 * TPW) + g8 * (4 * TPW);
  const uint8_t* sbase = HALF ? p.S + e * p.s_expert + (P >> 1) * 128 + g8 * 16 + (P & 1) * 8
                              : p.S + e * p.s_expert + P * (64 * TPW) + g8 * (8 * TPW);
  // the k-slot pair of this lane in a group: rows 2t, 2t+1 (low k) and 2t+8, 2t+9 (high k)
  const int kr[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};

  float acc[4 * TPW][NB][4];
  for (int pass = 0; pass < 2; ++pass) {
    const int nsteps = dense ? G : hdr->nag;
    uint32_t wq[R][4][TPW], sq[R][2 * TPW], xb[R][NB][2], mk[R];
    auto load = [&](int slot, int j) {
      if (j < nsteps) {
        const int grp = dense ? j : gact[j];
        const uint32_t m = dense ? 0xFFFFu : gmask[grp];
        const int kb = 16 * grp;
#pragma unroll
        for (int h = 0; h < 4; ++h) {
          const uint8_t* src = wbase + (long long)(kb + kr[h]) * p.w_k;
          if constexpr (TPW == 1) {
            wq[slot][h][0] = ((m >> kr[h]) & 1u) ? ldg_u32(src) : 0u;
          } else {
            const uint2 v = ((m >> kr[h]) & 1u) ? ldg_u64(src) : make_uint2(0u, 0u);
            wq[slot][h][0] = v.x;
            wq[slot][h][1] = v.y;
          }
        }
        if constexpr (TPW == 1) {
          const uint2 v = ldg_u64(sbase + (long long)grp * p.s_g);
          sq[slot][0] = v.x;
          sq[slot][1] = v.y;
        } else {
          const uint4 v = ldg_u128(sbase + (long long)grp * p.s_g);
          sq[slot][0] = v.x;
          sq[slot][1] = v.y;
          sq[slot][2] = v.z;
          sq[slot][3] = v.w;
        }
#pragma unroll
        for (int nb = 0; nb < NB; ++nb) {
          xb[slot][nb][0] = xrow[nb] ? ldg_u32_l1(xrow[nb] + kb + 2 * t) : 0u;
          xb[slot][nb][1] = xrow[nb] ? ldg_u32_l1(xrow[nb] + kb + 8 + 2 * t) : 0u;
        }
        mk[slot] = m;
      } else {
        mk[slot] = 0u;
      }
    };
#pragma unroll
    for (int c = 0; c < 4 * TPW; ++c) {
#pragma unroll
      for (int nb = 0; nb < NB; ++nb) {
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[c][nb][i] = 0.0f;
      }
    }
#pragma unroll
    for (int i = 0; i < R; ++i) load(i, i);
    bool neg0 = false;
    for (int j0 = 0; j0 < nsteps; j0 += R) {
#pragma unroll
      for (int i = 0; i < R; ++i) {
        if (j0 + i < nsteps) {
          const uint32_t m = mk[i];
          uint32_t w[4][TPW], s[2 * TPW];
#pragma unroll
          for (int h = 0; h < 4; ++h) {
#pragma unroll
            for (int tt = 0; tt < TPW; ++tt) w[h][tt] = wq[i][h][tt];
          }
#pragma unroll
          for (int q = 0; q < 2 * TPW; ++q) s[q] = sq[i][q];
          uint32_t bb[NB][2];
#pragma unroll
          for (int nb = 0; nb < NB; ++nb) {
            bb[nb][0] = xb[i][nb][0];
            bb[nb][1] = xb[i][nb][1];
          }
          load(i, j0 + i + R);
          if (math) {
            const uint32_t m01 = (m >> (2 * t)) & 3u;
            const uint32_t m23 = (m >> (2 * t + 8)) & 3u;
            const uint32_t keep01 = ((m01 & 1u) ? 0x0000FFFFu : 0u) | ((m01 & 2u) ? 0xFFFF0000u : 0u);
            const uint32_t keep23 = ((m23 & 1u) ? 0x0000FFFFu : 0u) | ((m23 & 2u) ? 0xFFFF0000u : 0u);
#pragma unroll
            for (int tt = 0; tt < TPW; ++tt) {
              // scales: byte 2c + h of (s[2tt], s[2tt+1]) -> column 16c + g8 + 8h of tile tt;
              // humming's fp8 bit placement, x 2^120
              const uint32_t sx = s[2 * tt], sy = s[2 * tt + 1];
              const uint32_t s01 = hmul2_u32(fp8_to_bf16x2(((sx << 8) & 0x0000FF00u) | ((sx << 16) & 0xFF000000u)), kExp120);
              const uint32_t s23 = hmul2_u32(fp8_to_bf16x2(((sx >> 8) & 0x0000FF00u) | (sx & 0xFF000000u)), kExp120);
              const uint32_t s45 = hmul2_u32(fp8_to_bf16x2(((sy << 8) & 0x0000FF00u) | ((sy << 16) & 0xFF000000u)), kExp120);
              const uint32_t s67 = hmul2_u32(fp8_to_bf16x2(((sy >> 8) & 0x0000FF00u) | (sy & 0xFF000000u)), kExp120);
              if constexpr (!HalfSkip) {
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                  const uint32_t sp = c == 0 ? s01 : (c == 1 ? s23 : (c == 2 ? s45 : s67));
                  const uint32_t slo = __byte_perm(sp, 0, 0x1010);  // column g8 (h = 0)
                  const uint32_t shi = __byte_perm(sp, 0, 0x3232);  // column g8 + 8 (h = 1)
                  const uint32_t sel = c | (c << 4) | ((c + 4) << 8) | ((c + 4) << 12);
                  const uint32_t x01 = __byte_perm(w[0][tt], w[1][tt], sel);  // bytes: w0[c], w0[c], w1[c], w1[c]
                  const uint32_t x23 = __byte_perm(w[2][tt], w[3][tt], sel);
                  const uint32_t a0 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x01 << 12), kExp124), slo) & keep01;
                  const uint32_t a1 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x01 << 8), kExp124), shi) & keep01;
                  const uint32_t a2 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x23 << 12), kExp124), slo) & keep23;
                  const uint32_t a3 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x23 << 8), kExp124), shi) & keep23;
#pragma unroll
                  for (int nb = 0; nb < NB; ++nb) {
                    // n-blocks without valid rows (warp-uniform) issue no MMA
                    if (nb == 0 || nrows > nb * 8) mma_bf16(acc[tt * 4 + c][nb], a0, a1, a2, a3, bb[nb][0], bb[nb][1]);
                  }
                }
              } else {
                // the same operations as above for each half that has a set mask bit
                uint32_t slo[4], shi[4], alo[4][2];
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                  const uint32_t sp = c == 0 ? s01 : (c == 1 ? s23 : (c == 2 ? s45 : s67));
                  slo[c] = __byte_perm(sp, 0, 0x1010);
                  shi[c] = __byte_perm(sp, 0, 0x3232);
                }
                if (m & 0x00FFu) {
#pragma unroll
                  for (int c = 0; c < 4; ++c) {
                    const uint32_t sel = c | (c << 4) | ((c + 4) << 8) | ((c + 4) << 12);
                    const uint32_t x01 = __byte_perm(w[0][tt], w[1][tt], sel);
                    alo[c][0] = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x01 << 12), kExp124), slo[c]) & keep01;
                    alo[c][1] = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x01 << 8), kExp124), shi[c]) & keep01;
                  }
                } else {
#pragma unroll
                  for (int c = 0; c < 4; ++c) alo[c][0] = alo[c][1] = 0u;
                }
                if (m & 0xFF00u) {
#pragma unroll
                  for (int c = 0; c < 4; ++c) {
                    const uint32_t sel = c | (c << 4) | ((c + 4) << 8) | ((c + 4) << 12);
                    const uint32_t x23 = __byte_perm(w[2][tt], w[3][tt], sel);
                    const uint32_t a2 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x23 << 12), kExp124), slo[c]) & keep23;
                    const uint32_t a3 = hmul2_u32(hmul2_u32(fp4_to_bf16x2(x23 << 8), kExp124), shi[c]) & keep23;
#pragma unroll
                    for (int nb = 0; nb < NB; ++nb) {
                      if (nb == 0 || nrows > nb * 8)
                        mma_bf16(acc[tt * 4 + c][nb], alo[c][0], alo[c][1], a2, a3, bb[nb][0], bb[nb][1]);
                    }
                  }
                } else {
#pragma unroll
                  for (int c = 0; c < 4; ++c) {
#pragma unroll
                    for (int nb = 0; nb < NB; ++nb) {
                      if (nb == 0 || nrows > nb * 8)
                        mma_bf16(acc[tt * 4 + c][nb], alo[c][0], alo[c][1], 0u, 0u, bb[nb][0], bb[nb][1]);
                    }
                  }
                }
              }
            }
            if (!dense) {
#pragma unroll
              for (int c = 0; c < 4 * TPW; ++c) {
#pragma unroll
                for (int nb = 0; nb < NB; ++nb) {
#pragma unroll
                  for (int q = 0; q < 4; ++q) neg0 |= __float_as_uint(acc[c][nb][q]) == 0x80000000u;
                }
              }
            }
          }
        }
      }
    }
    if (!__any_sync(0xFFFFFFFFu, neg0) || dense || !math) break;
    // a -0 accumulator appeared: recompute this warp's tiles with every row loaded
    if (lane == 0 && p.stats) atomicAdd(p.stats + 0, 1);
    dense = true;
  }
  if (!math) return;

  const float gs4 = mul_ftz(p.GS[e], 4.0f);
#pragma unroll
  for (int tt = 0; tt < TPW; ++tt) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int col = (P * TPW + tt) * 64 + 16 * c + g8;
#pragma unroll
      for (int nb = 0; nb < NB; ++nb) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int tok = nb * 8 + 2 * t + (q & 1);
          const int cc = col + ((q >> 1) ? 8 : 0);
          if (tok < nrows) {
            const uint32_t v = to_bf16x2(mul_ftz(acc[tt * 4 + c][nb][q], gs4), 0.0f);  // humming's cvt.rn.bf16x2.f32
            reinterpret_cast<uint16_t*>(p.C)[(long long)hdr->rows[tok] * p.ldc + cc] =
                hdr->rows[tok] < row_limit ? static_cast<uint16_t>(v) : uint16_t(0);
          }
        }
      }
    }
  }
}


#endif  // ZSKIP_NO_PAIR

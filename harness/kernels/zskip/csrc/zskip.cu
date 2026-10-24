// zskip: humming-exact routed-expert DOWN projection (NVFP4 W4A16, N = 2688, K = 1856) for
// calls of 1..8 tokens (the MTP verify), that does not load the weight rows whose activations
// are exactly zero. ../README.md documents what humming 0.1.10 computes and the evidence.
//
// Per output element humming computes acc = +0; for each k16 group g ascending:
// acc = mma.sync.m16n8k16.f32.bf16.bf16.f32 over k = 16g..16g+15; out = cvt.rn.bf16(
// mul.ftz(acc, mul.ftz(gs[e], 4))), with the weights dequantised as
// hmul2(hmul2(fp4bits, 2^124), hmul2(fp8bits, 2^120)) = e2m1 * e4m3 / 4. These kernels do the
// same operations on the same values in the same k-slots and order, with the weights as the
// MMA A operand and the activations as B (token slots as MMA columns). For each expert block
// and k16 group the 16-bit mask of k with a nonzero activation in some valid row selects the
// weight rows to load; skipped rows enter the MMA as +0 and groups with mask 0 issue no MMA.
// Device-checked properties (tests/zskip_exact.py --phase probe): with acc != -0,
// mma(acc, a, b) == mma(acc, a, b') when b' is +0 where a is zero; mma(acc, 0, b) == acc for
// acc not -0 / NaN. A warp whose accumulator ever becomes -0 recomputes its tiles densely.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace zskip {

// Programmatic dependent launch (sm_90+, taken from main's zskip): both kernels of a call wait for
// the preceding grid before any global memory access and let the next grid launch; launched
// without the PDL attribute both instructions are no-ops.
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_launch_dependents() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

constexpr int K = 1856;           // intermediate size (down K)
constexpr int N = 2688;           // hidden size (down N)
constexpr int G = K / 16;         // 116 k16 groups
constexpr int RMAX = 32;          // valid rows per expert block (16 or 32 slots)
constexpr int WARP_COLS = 32;
constexpr int NWARPS_TOTAL = N / WARP_COLS;  // 84

enum Mode { DENSE = 0, SKIP = 1, LOADONLY = 2, LOADONLY_DENSE = 3, SETUPONLY = 4 };  // 2-4: timing probes

struct Params {
  const __nv_bfloat16* A;   // [shape_m, lda] relu^2 activations (rows = token*top_k + slot)
  long long lda;            // elements
  int shape_m;              // rows of A and C
  const int* sorted_ids;    // moe_align_block_size (block 16)
  const int* expert_ids;
  const int* num_tokens_padded;
  const uint8_t* W;         // repacked e2m1 nibbles
  const uint8_t* S;         // repacked e4m3 scales
  const float* GS;          // [E] fp32 global scales (weight_scale_2)
  __nv_bfloat16* C;         // [shape_m, ldc]
  long long ldc;            // elements
  long long w_expert, w_tile, w_k;  // byte strides: expert, 128-column tile, k row
  long long s_expert, s_tile, s_g;  // byte strides: expert, 128-column tile, k16 group
  int mode;
  int* stats;               // optional [4]: 0 dense recomputes, 1 blocks with more rows than supported
  uint8_t* ws;              // [max_blocks * BI_BYTES] per-block lists (prep_kernel -> down4_kernel)
  int max_blocks;           // expert blocks covered (grid y)
  int slots;                // sorted_ids slots per block (16, or 32 for half of humming's 64)
  int ediv;                 // block b's expert is expert_ids[b / ediv]
};

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N_>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N_) : "memory"); }

__device__ __forceinline__ uint32_t lds_u32(const void* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}
__device__ __forceinline__ uint32_t lds_u16(const void* p) {
  return *reinterpret_cast<const uint16_t*>(p);
}

__device__ __forceinline__ uint32_t hmul2_u32(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}

// humming fp_to_fp<Float4E2M1, BFloat16>: nibbles at bits 12..15 and 28..31.
__device__ __forceinline__ uint32_t fp4_to_bf16x2(uint32_t v) {
  return (v & 0x80008000u) | ((v & 0x70007000u) >> 6);
}
// humming fp_to_fp<Float8E4M3, BFloat16>: bytes at bits 8..15 and 24..31.
__device__ __forceinline__ uint32_t fp8_to_bf16x2(uint32_t v) {
  return (v & 0x80008000u) | ((v & 0x7F007F00u) >> 4);
}

// prepare_exp_scale_factor<nv_bfloat162, E>: (E * 0x00010001 + 0x007F007F) << 7
constexpr uint32_t kExp124 = (124u * 0x00010001u + 0x007F007Fu) << 7;  // 2^124 in both halves
constexpr uint32_t kExp120 = (120u * 0x00010001u + 0x007F007Fu) << 7;  // 2^120

__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float mul_ftz(float a, float b) {
  float r;
  asm("mul.rn.ftz.f32 %0, %1, %2;\n" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ uint32_t to_bf16x2(float lo, float hi) {
  __nv_bfloat162 r = __float22bfloat162_rn(make_float2(lo, hi));
  return *reinterpret_cast<uint32_t*>(&r);
}

// bit 2i / 2i+1: bf16 element 2i / 2i+1 of the 16-byte chunk has nonzero magnitude
__device__ __forceinline__ uint32_t nz_mask8(uint4 v) {
  uint32_t m = 0;
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    m |= ((w[i] & 0x7FFFu) != 0u ? 1u : 0u) << (2 * i);
    m |= ((w[i] & 0x7FFF0000u) != 0u ? 1u : 0u) << (2 * i + 1);
  }
  return m;
}

template <int N_MAX>
__device__ __forceinline__ void cp_async_wait_dyn(int n) {
  // waits until at most n of this thread's commit groups are pending (n >= 15 waits for 15)
  switch (n < 0 ? 0 : n) {
    case 0: cp_async_wait<0>(); break;
    case 1: cp_async_wait<1>(); break;
    case 2: cp_async_wait<2>(); break;
    case 3: cp_async_wait<3>(); break;
    case 4: cp_async_wait<4>(); break;
    case 5: cp_async_wait<5>(); break;
    case 6: cp_async_wait<6>(); break;
    case 7: cp_async_wait<7>(); break;
    case 8: cp_async_wait<8>(); break;
    case 9: cp_async_wait<9>(); break;
    case 10: cp_async_wait<10>(); break;
    case 11: cp_async_wait<11>(); break;
    case 12: cp_async_wait<12>(); break;
    case 13: cp_async_wait<13>(); break;
    case 14: cp_async_wait<14>(); break;
    default: cp_async_wait<15>(); break;
  }
}

// ---------------------------------------------------------------------------
// Per expert block info, built once per call by prep_kernel into p.ws (BI_BYTES per block):
//   int nrows, nact, nag, expert; int rows[RMAX] (the valid sorted ids in block order);
//   u16 mask[128] (bit b of group g: k = 16g + b nonzero in some valid row); u16 rbase[128]
//   (active rows before group g); u16 gbase[128] (active groups before g); u16 act_g[128] (active
//   groups in order); u16 act_k[K] (active k in order).
constexpr int BI_HDR = 256;
constexpr int BI_MASK = BI_HDR;
constexpr int BI_RBASE = BI_MASK + 256;
constexpr int BI_GBASE = BI_RBASE + 256;
constexpr int BI_ACTG = BI_GBASE + 256;
constexpr int BI_ACTK = BI_ACTG + 256;
constexpr int BI_USED = BI_ACTK + K * 2;  // 4992
constexpr int BI_BYTES = 4992;

struct BlockHdr {
  int nrows, nact, nag, expert;
  int rows[RMAX];
};
static_assert(sizeof(BlockHdr) <= BI_HDR, "block header");

// Builds the lists for one expert block into dst (shared or global memory). mask_of(g) returns the
// group's 16-bit mask. Needs all NT threads; ends with a barrier.
template <int NT>
__device__ __forceinline__ void build_lists(uint8_t* dst, int nrows_valid, const Params& p, const int* rows,
                                            bool dense) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  uint16_t* mask = reinterpret_cast<uint16_t*>(dst + BI_MASK);
  uint16_t* rbase = reinterpret_cast<uint16_t*>(dst + BI_RBASE);
  uint16_t* gbase = reinterpret_cast<uint16_t*>(dst + BI_GBASE);
  uint16_t* act_g = reinterpret_cast<uint16_t*>(dst + BI_ACTG);
  uint16_t* act_k = reinterpret_cast<uint16_t*>(dst + BI_ACTK);
  __shared__ int s_tot[2];
  for (int g = tid; g < 128; g += NT) {
    uint32_t m = 0;
    if (g < G) {
      if (dense) {
        m = 0xFFFFu;
      } else {
        for (int r = 0; r < nrows_valid; ++r) {
          const uint4* q = reinterpret_cast<const uint4*>(p.A + (long long)rows[r] * p.lda + 16 * g);
          m |= nz_mask8(__ldg(q)) | (nz_mask8(__ldg(q + 1)) << 8);
        }
      }
    }
    mask[g] = static_cast<uint16_t>(m);
  }
  __syncthreads();
  if (warp == 0) {
    uint32_t cr[4], cg[4], sr = 0, sg = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t m = mask[4 * lane + i];
      cr[i] = __popc(m);
      cg[i] = m ? 1u : 0u;
      sr += cr[i];
      sg += cg[i];
    }
    uint32_t ir = sr, ig = sg;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const uint32_t vr = __shfl_up_sync(0xFFFFFFFFu, ir, d), vg = __shfl_up_sync(0xFFFFFFFFu, ig, d);
      if (lane >= d) {
        ir += vr;
        ig += vg;
      }
    }
    uint32_t er = ir - sr, eg = ig - sg;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      rbase[4 * lane + i] = static_cast<uint16_t>(er);
      gbase[4 * lane + i] = static_cast<uint16_t>(eg);
      er += cr[i];
      eg += cg[i];
    }
    if (lane == 31) {
      s_tot[0] = static_cast<int>(ir);
      s_tot[1] = static_cast<int>(ig);
    }
  }
  __syncthreads();
  for (int g = tid; g < G; g += NT) {
    uint32_t m = mask[g];
    if (m) act_g[gbase[g]] = static_cast<uint16_t>(g);
    int i = rbase[g];
    while (m) {
      const int b = __ffs(m) - 1;
      act_k[i++] = static_cast<uint16_t>(16 * g + b);
      m &= m - 1u;
    }
  }
  if (tid == 0) {
    BlockHdr* h = reinterpret_cast<BlockHdr*>(dst);
    h->nact = s_tot[0];
    h->nag = s_tot[1];
  }
  __syncthreads();
}

// one CTA per expert block (grid = max_blocks): rows, masks and lists into p.ws (v4)
__global__ void __launch_bounds__(128) prep_kernel(const Params p) {
  const int tid = threadIdx.x, lane = tid & 31;
  const int mb = blockIdx.x;
  uint8_t* bi = p.ws + (long long)mb * BI_BYTES;
  BlockHdr* h = reinterpret_cast<BlockHdr*>(bi);
  __shared__ int s_rows[RMAX];
  __shared__ int s_nrows;
  pdl_wait();
  pdl_launch_dependents();
  if (mb * p.slots >= *p.num_tokens_padded) {
    if (tid == 0) {
      h->nrows = 0;
      h->nact = 0;
      h->nag = 0;
    }
    return;
  }
  if (tid < 32) {
    const unsigned id = lane < p.slots ? static_cast<unsigned>(p.sorted_ids[mb * p.slots + lane]) : 0xFFFFFFFFu;
    const bool valid = lane < p.slots && id < static_cast<unsigned>(p.shape_m);
    const unsigned ball = __ballot_sync(0xFFFFFFFFu, valid);
    const int rank = __popc(ball & ((1u << lane) - 1u));
    const bool keep = valid && rank < RMAX;
    if (keep) {
      s_rows[rank] = static_cast<int>(id);
      h->rows[rank] = static_cast<int>(id);
    }
    if (lane == 0) {
      const int n = __popc(ball);
      s_nrows = n < RMAX ? n : RMAX;
      h->nrows = s_nrows;
      h->expert = p.expert_ids[mb / p.ediv];
      if (n > RMAX && p.stats) atomicAdd(p.stats + 1, 1);
    }
  }
  __syncthreads();
  const bool dense = p.mode == DENSE || p.mode == LOADONLY_DENSE;
  build_lists<128>(bi, s_nrows, p, s_rows, dense);
}

// ---------------------------------------------------------------------------
// v4: one warp per (expert block, 64-column tile), weights as the MMA A operand (16 output
// columns x 16 k), activations as B (16 k x 8 token slots), one fp32 chain per output element
// over the k16 groups in ascending order (the same products in the same k-slots as humming; the
// operand roles of the two bf16 inputs are swapped). Layout "swap" (repack): per expert and k row
// 1344 bytes = 42 tiles x 32 bytes; byte 4g + c of a tile = nibbles of columns 64T + 16c + g (low)
// and 64T + 16c + g + 8 (high); scales per k16 group 2688 bytes = 42 tiles x 64 bytes, byte
// 8g + 2c + h = scale of column 64T + 16c + g + 8h. Lane (g = lane/4, t = lane%4) loads the
// 4-byte words of k rows 2t, 2t+1, 2t+8, 2t+9 of each active group (inactive rows are not
// loaded: one k row of a tile is one 32-byte sector), its 8 scale bytes and its two activation
// words, through a register ring of R active groups.

__device__ __forceinline__ uint32_t ldg_u32(const void* p) {
  uint32_t v;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ uint2 ldg_u64(const void* p) {
  uint2 v;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t ldg_u32_l1(const void* p) {  // activations: L2 resident, reused
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}

__device__ __forceinline__ uint4 ldg_u128(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
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
template <int TPW, int WPC, int R, int NB, bool HALF>
__global__ void __launch_bounds__(WPC * 32) down4_kernel(const Params p) {
  static_assert(TPW == 1 || TPW == 2, "tiles per warp");
  static_assert(NB >= 1 && NB <= 4, "n-blocks per warp");
  static_assert(!HALF || TPW == 1, "half pairs are single tiles");
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g8 = lane >> 2, t = lane & 3;
  const int mb = blockIdx.y;
  const int P = blockIdx.x * WPC + warp;  // tile (TPW = 1) or tile pair (TPW = 2)
  pdl_wait();
  pdl_launch_dependents();
  if (mb * p.slots >= *p.num_tokens_padded) return;
  const uint8_t* bi = p.ws + (long long)mb * BI_BYTES;
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
  const __nv_bfloat16* xrow[NB];
#pragma unroll
  for (int nb = 0; nb < NB; ++nb) {
    const int r = nb * 8 + g8;
    xrow[nb] = r < nrows ? p.A + (long long)hdr->rows[r] * p.lda : nullptr;
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
            reinterpret_cast<uint16_t*>(p.C)[(long long)hdr->rows[tok] * p.ldc + cc] = static_cast<uint16_t>(v);
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Device probes of the tensor-core semantics the skip relies on.

__device__ __forceinline__ uint32_t rnd(uint64_t& s) {
  s = s * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<uint32_t>(s >> 32);
}

// a realistic B value: (e2m1 * e4m3) / 4 through the kernel's own dequant ops
__device__ __forceinline__ uint32_t gen_b_real(uint64_t& s) {
  uint32_t nib = rnd(s) & 0xF;
  uint32_t byte;
  do { byte = rnd(s) & 0xFF; } while ((byte & 0x7F) == 0x7F);  // no NaN scale
  uint32_t b = fp4_to_bf16x2(nib << 12);
  b = hmul2_u32(b, kExp124);
  uint32_t sv = hmul2_u32(fp8_to_bf16x2(byte << 8), kExp120);
  return hmul2_u32(b, __byte_perm(sv, 0, 0x1010)) & 0xFFFFu;
}

__device__ __forceinline__ uint32_t gen_bf16_finite(uint64_t& s) {
  uint32_t v;
  do { v = rnd(s) & 0xFFFF; } while ((v & 0x7F80u) == 0x7F80u);
  return v;
}

// A value of row r, k-slot k for trial kind `kind` (k not in the forced-zero set)
__device__ __forceinline__ uint32_t gen_a(uint64_t& s, int kind) {
  const uint32_t r = rnd(s);
  switch (kind) {
    case 0: {  // relu^2-like: positive, exponent 2^-40 .. 2^14, 60% zeros
      if ((r & 0xFF) < 154) return 0u;
      const uint32_t ex = 127 - 40 + ((r >> 8) % 55);
      return (ex << 7) | ((r >> 16) & 0x7F);
    }
    case 1: return gen_bf16_finite(s);  // full finite range, both signs, subnormals
    case 4: {  // bf16 subnormals and tiny normals
      if ((r & 3) == 0) return 0u;
      const uint32_t ex = (r >> 2) % 3;  // 0 = subnormal
      return ((r >> 31) << 15) | (ex << 7) | ((r >> 8) & 0x7F);
    }
    default: return gen_bf16_finite(s);
  }
}

__device__ __forceinline__ float gen_c(uint64_t& s, int kind) {
  const uint32_t r = rnd(s);
  const uint32_t sel = r & 7;
  if (sel < 2) return 0.0f;  // +0
  if (sel == 2) {            // tiny / subnormal fp32
    return __uint_as_float(((r >> 31) << 31) | (rnd(s) & 0x00FFFFFFu));
  }
  if (sel == 3 && kind == 6) {  // non-finite C
    const uint32_t v = (rnd(s) & 1) ? 0x7F800000u : 0x7FC00000u | (rnd(s) & 0x3FFFFF);
    return __uint_as_float(((r >> 30) & 1) << 31 | v);
  }
  // random sign, exponent 2^-60 .. 2^40
  const uint32_t ex = 127 - 60 + ((r >> 8) % 101);
  return __uint_as_float(((r >> 31) << 31) | (ex << 23) | (rnd(s) & 0x7FFFFF));
}

// counts[kind*8 + i]: 0 elements, 1 mismatches with C != -0, 2 mismatches with C == -0,
// 3 elements with C == -0, 4 full-zero-A identity violations (C not -0 / NaN),
// 5 full-zero-A elements checked, 6 mma(-0, a, b) != mma(+0, a, b), 7 elements checked for 6.
__global__ void probe_kernel(uint64_t seed, int iters, unsigned long long* counts, uint32_t* samples) {
  const int lane = threadIdx.x & 31;
  const uint64_t gw = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  uint64_t su = seed ^ (gw * 0x9E3779B97F4A7C15ULL);                 // warp-uniform
  uint64_t sl = seed ^ ((gw * 32 + lane + 1) * 0xD1B54A32D192ED03ULL);  // per lane
  unsigned long long n[7][8] = {};
  const int t = lane & 3;
  for (int it = 0; it < iters; ++it) {
    const int kind = static_cast<int>(rnd(su) % 7);
    // zero-k set Z (warp-uniform): density varies per trial; sometimes all 16
    const uint32_t zr = rnd(su);
    uint32_t Z;
    const uint32_t zsel = zr & 7;
    if (zsel == 0) Z = 0xFFFFu;
    else if (zsel == 1) Z = 0u;
    else {
      Z = 0;
      const uint32_t th = 64 + (rnd(su) & 0x7F);  // 25% .. 75% zero per slot
      for (int k = 0; k < 16; ++k) Z |= ((rnd(su) & 0xFF) < th ? 1u : 0u) << k;
    }
    const bool neg_zero_a = (rnd(su) & 3) == 0;  // forced zeros of A as -0
    // the k-slots of this lane: 2t, 2t+1 (reg 0/1 low k), 2t+8, 2t+9
    const int ks[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};
    const int c8 = lane >> 2;
    uint32_t ah[4][2];  // [reg][half]: reg0 row c k lo, reg1 row c+8 k lo, reg2 row c k hi, reg3 row c+8 k hi
    uint32_t bh[2][2], bph[2][2];
    // cancellation trials (kind 2): A[r][k1] = A[r][k2] = v(r), B[k1][n] = -1, B[k2][n] = +1
    const int k1 = static_cast<int>(rnd(su) & 15), k2 = static_cast<int>(rnd(su) & 15);
    const uint64_t vseed = su;
    const uint32_t big = 0x7000u | (rnd(su) & 0xFFu);  // bf16 >= 2^97 (kind 3)
    for (int reg = 0; reg < 4; ++reg) {
      const int row = c8 + ((reg & 1) ? 8 : 0);
      for (int h = 0; h < 2; ++h) {
        const int k = ks[(reg >> 1) * 2 + h];
        uint32_t v;
        if ((Z >> k) & 1u) v = neg_zero_a ? 0x8000u : 0u;
        else if (kind == 2) {
          uint64_t rs = vseed ^ (0x51ED27ULL * (row + 1));
          const uint32_t rv = rnd(rs);
          v = (k == k1 || k == k2) ? ((rv & 0x8000u) | (0x3000u + (rv & 0x0FFFu))) : 0u;
        } else if (kind == 3) v = (rnd(sl) & 1) ? (0x0800u | (rnd(sl) & 0x7FFu)) : 0u;  // ~2^-111
        else if (kind == 5) v = (rnd(sl) & 3) ? 0u : gen_a(sl, 0);
        else v = gen_a(sl, kind);
        ah[reg][h] = v;
      }
    }
    for (int reg = 0; reg < 2; ++reg) {
      for (int h = 0; h < 2; ++h) {
        const int k = ks[reg * 2 + h];
        uint32_t v;
        if (kind == 1 || kind == 4) v = gen_bf16_finite(sl);
        else v = gen_b_real(sl);
        if (kind == 2 && k1 != k2 && k == k1) v = 0xBF80u;  // -1
        if (kind == 2 && k1 != k2 && k == k2) v = 0x3F80u;  // +1
        if (kind == 3 && ((Z >> k) & 1u)) v = big | ((rnd(sl) & 1u) << 15);
        bh[reg][h] = v;
        bph[reg][h] = ((Z >> k) & 1u) ? 0u : v;
      }
    }
    uint32_t a[4], b[2], bp[2];
    for (int reg = 0; reg < 4; ++reg) a[reg] = ah[reg][0] | (ah[reg][1] << 16);
    for (int reg = 0; reg < 2; ++reg) {
      b[reg] = bh[reg][0] | (bh[reg][1] << 16);
      bp[reg] = bph[reg][0] | (bph[reg][1] << 16);
    }
    float c0[4], d1[4], d2[4];
    for (int i = 0; i < 4; ++i) {
      c0[i] = gen_c(sl, kind);
      if ((rnd(sl) & 15) == 0) c0[i] = -0.0f;
      d1[i] = c0[i];
      d2[i] = c0[i];
    }
    mma_bf16(d1, a[0], a[1], a[2], a[3], b[0], b[1]);
    mma_bf16(d2, a[0], a[1], a[2], a[3], bp[0], bp[1]);
    // property Q: the sign of a zero accumulator never matters: mma(-0, a, b) == mma(+0, a, b),
    // with the original and with the substituted b
    float qp[4] = {0.0f, 0.0f, 0.0f, 0.0f}, qm[4] = {-0.0f, -0.0f, -0.0f, -0.0f};
    float qp2[4] = {0.0f, 0.0f, 0.0f, 0.0f}, qm2[4] = {-0.0f, -0.0f, -0.0f, -0.0f};
    mma_bf16(qp, a[0], a[1], a[2], a[3], b[0], b[1]);
    mma_bf16(qm, a[0], a[1], a[2], a[3], b[0], b[1]);
    mma_bf16(qp2, a[0], a[1], a[2], a[3], bp[0], bp[1]);
    mma_bf16(qm2, a[0], a[1], a[2], a[3], bp[0], bp[1]);
    for (int i = 0; i < 4; ++i) {
      n[kind][7] += 2;
      if (__float_as_uint(qp[i]) != __float_as_uint(qm[i])) n[kind][6]++;
      if (__float_as_uint(qp2[i]) != __float_as_uint(qm2[i])) n[kind][6]++;
    }
    for (int i = 0; i < 4; ++i) {
      const uint32_t cu = __float_as_uint(c0[i]);
      const uint32_t u1 = __float_as_uint(d1[i]), u2 = __float_as_uint(d2[i]);
      n[kind][0]++;
      if (cu == 0x80000000u) {
        n[kind][3]++;
        if (u1 != u2) n[kind][2]++;
      } else if (u1 != u2) {
        n[kind][1]++;
        const unsigned slot = atomicAdd(samples, 1u);
        if (slot < 64) {
          uint32_t* sp = samples + 1 + slot * 8;
          sp[0] = kind; sp[1] = cu; sp[2] = u1; sp[3] = u2; sp[4] = Z; sp[5] = a[i & 1 ? 2 : 0];
          sp[6] = b[0]; sp[7] = b[1];
        }
      }
      if (Z == 0xFFFFu) {
        const bool nan = (cu & 0x7FFFFFFFu) > 0x7F800000u;
        if (cu != 0x80000000u && !nan) {
          n[kind][5]++;
          if (u1 != cu) n[kind][4]++;
        }
      }
    }
  }
  for (int k = 0; k < 7; ++k)
    for (int i = 0; i < 8; ++i)
      if (n[k][i]) atomicAdd(&counts[k * 8 + i], n[k][i]);
}

// epilogue probe: out[i] = cvt.rn.bf16(mul.ftz(x[i], gs4)) | (mul result bits stored in out2)
__global__ void epi_kernel(const uint32_t* x, int n, float gs, uint32_t* out_bf16, uint32_t* out_mul) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float gs4 = mul_ftz(gs, 4.0f);
  const float y = mul_ftz(__uint_as_float(x[i]), gs4);
  out_mul[i] = __float_as_uint(y);
  out_bf16[i] = to_bf16x2(y, y) & 0xFFFFu;
}

}  // namespace zskip

using namespace zskip;

static int g_pdl = 0;  // zskip_set_pdl: launch the pair with programmatic dependent launch

template <typename Kern>
static cudaError_t launch_pdl(Kern kernel, dim3 grid, dim3 block, cudaStream_t stream, const Params& p) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, kernel, p);
}

template <int TPW, int WPC, int R, int NB, bool HALF>
static int launch4(const Params& p, cudaStream_t stream) {
  if (p.slots > 8 * NB && p.slots > 16) return -5;  // a block could hold more rows than the warp covers
  if (g_pdl) {
    cudaError_t e = launch_pdl(prep_kernel, dim3(p.max_blocks), dim3(128), stream, p);
    if (e == cudaSuccess)
      e = launch_pdl(down4_kernel<TPW, WPC, R, NB, HALF>, dim3(42 / (TPW * WPC), p.max_blocks), dim3(WPC * 32),
                     stream, p);
    if (e != cudaSuccess) {
      cudaGetLastError();
      return (int)e;
    }
    return (int)cudaGetLastError();
  }
  prep_kernel<<<p.max_blocks, 128, 0, stream>>>(p);
  down4_kernel<TPW, WPC, R, NB, HALF><<<dim3(42 / (TPW * WPC), p.max_blocks), WPC * 32, 0, stream>>>(p);
  return (int)cudaGetLastError();
}

// config id, 64-column tiles per warp, warps per CTA, ring depth, token n-blocks of 8 per warp,
// half pair (tiles per warp 1: layout "swap", or "swap2" when half; 2: layout "swap2")
#define ZSKIP4_CONFIGS(X)                                                                          \
  X(100, 1, 2, 4, 1, false) X(110, 2, 1, 4, 1, false) X(113, 2, 1, 3, 1, false)                     \
  X(212, 2, 1, 2, 2, false) X(300, 1, 2, 3, 4, true) X(301, 1, 2, 2, 4, true)                       \
  X(302, 1, 6, 3, 4, true) X(303, 1, 2, 4, 4, true)

extern "C" {

// Loads the kernels and sets their shared-memory limits (call outside CUDA graph capture).
int zskip_init(void) {
  cudaFuncAttributes attr;
  cudaError_t e;
  if ((e = cudaFuncGetAttributes(&attr, prep_kernel)) != cudaSuccess) return (int)e;
#define X4(ID, TPW_, W_, R_, NB_, H_) \
  if ((e = cudaFuncGetAttributes(&attr, down4_kernel<TPW_, W_, R_, NB_, H_>)) != cudaSuccess) return (int)e;
  ZSKIP4_CONFIGS(X4)
#undef X4
  if ((e = cudaFuncGetAttributes(&attr, probe_kernel)) != cudaSuccess) return (int)e;
  if ((e = cudaFuncGetAttributes(&attr, epi_kernel)) != cudaSuccess) return (int)e;
  return 0;
}

int zskip_set_pdl(int on) {
  g_pdl = on ? 1 : 0;
  return 0;
}

// info[0] = registers, [1] = dynamic smem bytes, [2] = warps per CTA, [3] = CTAs per SM
// returns 0, or -2 for an unknown id
int zskip_config(int id, int* info) {
  cudaFuncAttributes attr;
#define X4(ID, TPW_, W_, R_, NB_, H_)                                                            \
  if (id == ID) {                                                                                \
    if (cudaFuncGetAttributes(&attr, down4_kernel<TPW_, W_, R_, NB_, H_>) != cudaSuccess) return -1; \
    info[0] = attr.numRegs;                                                                      \
    info[1] = 0;                                                                                 \
    info[2] = W_;                                                                                \
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&info[3], down4_kernel<TPW_, W_, R_, NB_, H_>, W_ * 32, 0) != \
        cudaSuccess)                                                                             \
      return -1;                                                                                 \
    return 0;                                                                                    \
  }
  ZSKIP4_CONFIGS(X4)
#undef X4
  return -2;
}

int zskip_down(const void* A, long long lda, int shape_m, const int* sorted_ids, const int* expert_ids,
               const int* num_tokens_padded, int max_blocks, const void* W, const void* S, const float* GS,
               void* C, long long ldc, long long w_expert, long long w_tile, long long w_k, long long s_expert,
               long long s_tile, long long s_g, int mode, int* stats, int config, void* ws, int slots, int ediv,
               void* stream) {
  Params p;
  p.A = reinterpret_cast<const __nv_bfloat16*>(A);
  p.lda = lda;
  p.shape_m = shape_m;
  p.sorted_ids = sorted_ids;
  p.expert_ids = expert_ids;
  p.num_tokens_padded = num_tokens_padded;
  p.W = reinterpret_cast<const uint8_t*>(W);
  p.S = reinterpret_cast<const uint8_t*>(S);
  p.GS = GS;
  p.C = reinterpret_cast<__nv_bfloat16*>(C);
  p.ldc = ldc;
  p.w_expert = w_expert;
  p.w_tile = w_tile;
  p.w_k = w_k;
  p.s_expert = s_expert;
  p.s_tile = s_tile;
  p.s_g = s_g;
  p.mode = mode;
  p.stats = stats;
  p.ws = reinterpret_cast<uint8_t*>(ws);
  p.max_blocks = max_blocks;
  p.slots = slots;
  p.ediv = ediv;
  if (max_blocks <= 0) return 0;
  if ((slots != 16 && slots != 32) || ediv < 1) return -3;
  if ((lda % 8) || (ldc % 2)) return -3;
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
#define X4(ID, TPW_, W_, R_, NB_, H_) \
  if (config == ID) return launch4<TPW_, W_, R_, NB_, H_>(p, st);
  ZSKIP4_CONFIGS(X4)
#undef X4
  return -2;
}

int zskip_probe(unsigned long long seed, int blocks, int iters, unsigned long long* counts, uint32_t* samples,
                void* stream) {
  probe_kernel<<<blocks, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(seed, iters, counts, samples);
  return (int)cudaGetLastError();
}

int zskip_epi(const uint32_t* x, int n, float gs, uint32_t* out_bf16, uint32_t* out_mul, void* stream) {
  epi_kernel<<<(n + 255) / 256, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(x, n, gs, out_bf16, out_mul);
  return (int)cudaGetLastError();
}

}  // extern "C"

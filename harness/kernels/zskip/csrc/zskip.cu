// zskip C API (kernels, config lists and per-config entry points: zskip.cuh). zskip_down dispatches a
// config id to the entry points that the zskip_<part>.cu units instantiate. This unit compiles the
// kernels that several configs launch (prep_kernel, c3lpt_order_kernel) and the probes.

#include "zskip.cuh"

namespace zskip {

// C3LPT: stable descending active-K16 order of expert blocks. All21 column
// pairs of a block have identical masks, so sorting30 blocks is tile-level LPT.
// queue[4..33] is same-call scratch, overwritten on every admitted invocation.
__global__ void __launch_bounds__(32) c3lpt_order_kernel(const Params p) {
#ifndef ZSKIP_NO_PDL
  pdl_wait();
  pdl_launch_dependents();
#endif
  const int lane = threadIdx.x;
  int key = -1;
  if (lane < p.max_blocks) {
    const BlockHdr* h = reinterpret_cast<const BlockHdr*>(p.ws + (long long)lane * BI_BYTES);
    key = (h->nag << 5) | (31 - lane);
  }
#pragma unroll
  for (int k = 2; k <= 32; k *= 2) {
#pragma unroll
    for (int j = k / 2; j > 0; j /= 2) {
      const int other = __shfl_xor_sync(0xFFFFFFFFu, key, j);
      const bool high = ((lane & j) == 0) == ((lane & k) == 0);
      key = high ? max(key, other) : min(key, other);
    }
  }
  if (lane < p.max_blocks)
    p.queue[4 + lane] = 31 - (key & 31);
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
// 5 full-zero-A elements checked.
__global__ void probe_kernel(uint64_t seed, int iters, unsigned long long* counts, uint32_t* samples) {
  const int lane = threadIdx.x & 31;
  const uint64_t gw = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  uint64_t su = seed ^ (gw * 0x9E3779B97F4A7C15ULL);                 // warp-uniform
  uint64_t sl = seed ^ ((gw * 32 + lane + 1) * 0xD1B54A32D192ED03ULL);  // per lane
  unsigned long long n[7][6] = {};
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
    for (int i = 0; i < 6; ++i)
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

int g_sms = 0;
static int g_c3lpt = 1;  // Python C3LPT=0 restores vsc order
int g_rds_ring2 = 0;
int g_pdl = 0;

template <int BM, int CAP>
void launch_prep(const Params& p, cudaStream_t stream) {
  prep_kernel<BM, CAP><<<p.max_blocks, 128, 0, stream>>>(p);
}
#ifndef ZSKIP_NO_PDL
template <int BM, int CAP>
cudaError_t launch_prep_pdl(const Params& p, cudaStream_t stream) {
  return launch_pdl(prep_kernel<BM, CAP>, dim3(p.max_blocks), dim3(128), stream, p);
}
#endif
// the block sizes and row caps (8 NH) of the config lists
template void launch_prep<16, 8>(const Params&, cudaStream_t);
template void launch_prep<16, 16>(const Params&, cudaStream_t);
template void launch_prep<32, 32>(const Params&, cudaStream_t);
template void launch_prep<64, 64>(const Params&, cudaStream_t);
#ifndef ZSKIP_NO_PDL
template cudaError_t launch_prep_pdl<16, 8>(const Params&, cudaStream_t);
template cudaError_t launch_prep_pdl<16, 16>(const Params&, cudaStream_t);
template cudaError_t launch_prep_pdl<32, 32>(const Params&, cudaStream_t);
template cudaError_t launch_prep_pdl<64, 64>(const Params&, cudaStream_t);
#endif

#ifndef ZSKIP_NO_PAIR
static bool is_lean_config(int id) {
#define XL(ID, W_, R_, F_) if (id == ID) return true;
  ZSKIP_LEAN_CONFIGS(XL)
#undef XL
  return false;
}
#endif

extern "C" {

int zskip_set_c3lpt(int on) { g_c3lpt = on ? 1 : 0; return 0; }

int zskip_set_rds_ring2(int on) { g_rds_ring2 = on ? 1 : 0; return 0; }

#ifndef ZSKIP_NO_PDL
int zskip_set_pdl(int on) {
  g_pdl = on ? 1 : 0;
  return 0;
}
#endif

int zskip_init(void) {
  cudaFuncAttributes attr;
  cudaError_t e;
#define X4(ID, W_, R_, NH_, BM_) \
  if ((e = load4<W_, R_, NH_, BM_>()) != cudaSuccess) return (int)e;
  ZSKIP4_CONFIGS(X4)
#undef X4
  if ((e = cudaFuncGetAttributes(&attr, prep_kernel<16, 8>)) != cudaSuccess) return (int)e;

#ifndef ZSKIP_NO_PAIR
  if ((e = load_pair<false, false>()) != cudaSuccess) return (int)e;
  if ((e = load_pair<false, true>()) != cudaSuccess) return (int)e;
#ifndef ZSKIP_NO_FUSED
  if ((e = load_pair<true, false>()) != cudaSuccess) return (int)e;
  if ((e = load_pair<true, true>()) != cudaSuccess) return (int)e;
#endif
#endif
  return 0;
}

// info[0] = registers, [1] = dynamic smem bytes, [2] = warps, [3] = CTAs per SM; 0, or -2 for an unknown id
int zskip_config(int id, int* info) {
#define X(ID, W_, RB_, NR_, QB_, NQ_) \
  if (id == ID) return info3<W_, RB_, NR_, QB_, NQ_>(info);
  ZSKIP_CONFIGS(X)
#undef X
#define X4(ID, W_, R_, NH_, BM_) \
  if (id == ID) return info4<W_, R_, NH_, BM_>(info);
  ZSKIP4_CONFIGS(X4)
#undef X4
#ifndef ZSKIP_NO_FUSED
#define X4F(ID, W_, R_, NH_, BM_) \
  if (id == ID) return info4f<W_, R_, NH_, BM_>(info);
  ZSKIP4F_CONFIGS(X4F)
#undef X4F
#endif
#ifndef ZSKIP_NO_PAIR
#define PAIR_INFO(ID, FUSED, HSKIP) \
  if (id == ID) return info_pair<FUSED, HSKIP>(info);
  PAIR_INFO(113, false, false)
  PAIR_INFO(114, false, true)
#ifndef ZSKIP_NO_FUSED
  PAIR_INFO(118, true, false)
  PAIR_INFO(119, true, true)
#endif
#undef PAIR_INFO
#define XL(ID, W_, R_, F_) \
  if (id == ID) return info_lean<W_, R_, F_>(info);
  ZSKIP_LEAN_CONFIGS(XL)
#undef XL
#endif
  return -2;
}

static int zskip_down_impl(const void* A, long long lda, int shape_m, const int* sorted_ids, const int* expert_ids,
               const int* num_tokens_padded, int max_blocks, const void* W, const void* S, const float* GS,
               void* C, long long ldc, long long w_expert, long long w_tile, long long w_k, long long s_expert,
               long long s_tile, long long s_g, int mode, int* stats, int config, void* ws, int* queue,
               int ctas_per_sm, void* trace, void* stream, const int* verify_rows) {
  if (verify_rows != nullptr && ((shape_m != 30 && shape_m != 36) || mode != SKIP ||
      (config != 113 && config != 114 && config != 118 && config != 119
#ifndef ZSKIP_NO_PAIR
       && !is_lean_config(config)
#endif
       ))) return -5;
  Params p;
  p.verify_rows = verify_rows;
  p.c3lpt = g_c3lpt && shape_m == 30 && max_blocks == 30 && config == 113 && mode == SKIP;
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
  p.queue = queue;
  p.max_blocks = max_blocks;
  p.trace = reinterpret_cast<unsigned long long*>(trace);
  if (max_blocks <= 0) return 0;
  if ((lda % 8) || (ldc % 2)) return -3;
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
#define X(ID, W_, RB_, NR_, QB_, NQ_) \
  if (config == ID) return launch3<W_, RB_, NR_, QB_, NQ_>(p, st, ctas_per_sm);
  ZSKIP_CONFIGS(X)
#undef X
#define X4(ID, W_, R_, NH_, BM_) \
  if (config == ID) return launch4<W_, R_, NH_, BM_>(p, st);
  ZSKIP4_CONFIGS(X4)
#undef X4
#ifndef ZSKIP_NO_FUSED
#define X4F(ID, W_, R_, NH_, BM_) \
  if (config == ID) return launch4f<W_, R_, NH_, BM_>(p, st);
  ZSKIP4F_CONFIGS(X4F)
#undef X4F
#endif
#define X5(ID, W_, S_, NH_, BM_) \
  if (config == ID) return launch5<W_, S_, NH_, BM_>(p, st);
  ZSKIP5_CONFIGS(X5)
#undef X5
#define X6(ID, W_, R_, NH_, BM_) \
  if (config == ID) return launch6<W_, R_, NH_, BM_>(p, st);
  ZSKIP6_CONFIGS(X6)
#undef X6
#define X7(ID, W_, R_, NH_, BM_) \
  if (config == ID) return launch7<W_, R_, NH_, BM_>(p, st);
  ZSKIP7_CONFIGS(X7)
#undef X7
#define X8(ID, W_, R_, NH_, BM_) \
  if (config == ID) return launch8<W_, R_, NH_, BM_>(p, st);
  ZSKIP8_CONFIGS(X8)
#undef X8
#ifndef ZSKIP_NO_PAIR
  if (config == 113) return launch_pair<false>(p, st);
  if (config == 114) return launch_pair<false, true>(p, st);
#ifndef ZSKIP_NO_FUSED
  if (config == 118) return launch_pair<true>(p, st);
  if (config == 119) return launch_pair<true, true>(p, st);
#endif
#define XL(ID, W_, R_, F_) \
  if (config == ID) return launch_lean<W_, R_, F_>(p, st);
  ZSKIP_LEAN_CONFIGS(XL)
#undef XL
#endif
  return -2;
}

int zskip_down(const void* A, long long lda, int shape_m, const int* sorted_ids, const int* expert_ids,
               const int* num_tokens_padded, int max_blocks, const void* W, const void* S, const float* GS,
               void* C, long long ldc, long long w_expert, long long w_tile, long long w_k, long long s_expert,
               long long s_tile, long long s_g, int mode, int* stats, int config, void* ws, int* queue,
               int ctas_per_sm, void* trace, void* stream) {
  return zskip_down_impl(A, lda, shape_m, sorted_ids, expert_ids, num_tokens_padded, max_blocks, W, S, GS, C, ldc, w_expert, w_tile, w_k, s_expert, s_tile, s_g, mode, stats, config, ws, queue, ctas_per_sm, trace, stream, nullptr);
}

int zskip_down_prefix(const void* A, long long lda, int shape_m, const int* sorted_ids, const int* expert_ids,
               const int* num_tokens_padded, int max_blocks, const void* W, const void* S, const float* GS,
               void* C, long long ldc, long long w_expert, long long w_tile, long long w_k, long long s_expert,
               long long s_tile, long long s_g, int mode, int* stats, int config, void* ws, int* queue,
               int ctas_per_sm, void* trace, void* stream, const int* verify_rows) {
  return zskip_down_impl(A, lda, shape_m, sorted_ids, expert_ids, num_tokens_padded, max_blocks, W, S, GS, C, ldc, w_expert, w_tile, w_k, s_expert, s_tile, s_g, mode, stats, config, ws, queue, ctas_per_sm, trace, stream, verify_rows);
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

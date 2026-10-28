// SPDX-License-Identifier: Apache-2.0
// Experimental path, included inside namespace bf16ws after gemm and
// epilogue definitions. Qualification and measurements are remote only.
// QKV M496/N4608/K2688: two original eight-warp row-band groups consume the
// same W tile. A separate two-slot W pipeline permits the next W transfer to
// overlap current consumption; one A buffer per band is released by all 16
// consumer warps before it is refilled. The design is not yet validated.

constexpr int TWIN_WARPS = 16;
constexpr int TWIN_THREADS = TWIN_WARPS * 32;
constexpr int TWIN_SMEM = 2 * ASZ + 2 * QKV::WSZ + 1024;
static_assert(TWIN_SMEM == QKV::SMEM, "two-band staging must fit original QKV shared allocation");
static_assert(QKV::NW % 2 == 0 && QKV::NTN == 48 && QKV::WSZ % 1024 == 0,
              "prototype uses QKV's even paired MMA blocks and aligned weight stages");

__global__ void __launch_bounds__(TWIN_THREADS, 1)
gemm_qkv_twinband_prototype(const __grid_constant__ CUtensorMap tmA, const Params p) {
  constexpr int N = QKV::N_, NW = QKV::NW, BN = QKV::BN;
  constexpr int NKT = QKV::NKT, WSZ = QKV::WSZ;
  extern __shared__ uint8_t sm_raw[];
  __shared__ __align__(8) unsigned long long a_full_bar, a_empty_bar;
  __shared__ __align__(8) unsigned long long w_full_bars[2], w_empty_bars[2];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int band = warp / NWARP, local_warp = warp % NWARP;
  const int wm = local_warp % WM, wn = local_warp / WM;
  const int nt = blockIdx.x;
  if (nt >= QKV::NTN) return;
  const uint32_t sbase = ((uint32_t)__cvta_generic_to_shared(sm_raw) + 1023u) & ~1023u;
  const uint32_t afull = (uint32_t)__cvta_generic_to_shared(&a_full_bar);
  const uint32_t aempty = (uint32_t)__cvta_generic_to_shared(&a_empty_bar);
  const uint32_t wfull = (uint32_t)__cvta_generic_to_shared(&w_full_bars[0]);
  const uint32_t wempty = (uint32_t)__cvta_generic_to_shared(&w_empty_bars[0]);
  if (tid == 0) {
    mbar_init(afull, 1);
    mbar_init(aempty, TWIN_WARPS);
    for (int s = 0; s < 2; ++s) {
      mbar_init(wfull + 8 * s, 1);
      mbar_init(wempty + 8 * s, TWIN_WARPS);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA) : "memory");
    mbar_expect_tx(wfull, WSZ);
    bulk(sbase + 2 * ASZ, p.wp + (size_t)nt * NKT * WSZ, WSZ, wfull);
  }
  float acc[MW][NW][4];
#pragma unroll
  for (int i = 0; i < MW; ++i)
#pragma unroll
    for (int j = 0; j < NW; ++j)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
  const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ach = lane >> 4;
  const int br = (lane & 7) + (lane >> 4) * 8, bch = (lane >> 3) & 1;
  const int rb0 = band * BM + wm * MW * 16;
  for (int kt = 0; kt < NKT; ++kt) {
    if (tid == 0) {
      const int next = kt + 1;
      if (next < NKT) {
        const int ns = next & 1;
        if (next >= 2) mbar_wait(wempty + 8 * ns, ((next / 2) - 1) & 1);
        mbar_expect_tx(wfull + 8 * ns, WSZ);
        bulk(sbase + 2 * ASZ + ns * WSZ,
             p.wp + (size_t)(nt * NKT + next) * WSZ, WSZ, wfull + 8 * ns);
      }
      if (kt != 0) mbar_wait(aempty, (kt - 1) & 1);
      mbar_expect_tx(afull, 2 * ASZ);
      tma2d(sbase, &tmA, kt * KT, 0, afull);
      tma2d(sbase + ASZ, &tmA, kt * KT, BM, afull);
    }
    mbar_wait(afull, kt & 1);
    mbar_wait(wfull + 8 * (kt & 1), (kt / 2) & 1);
    const uint32_t abase = sbase + band * ASZ;
    const uint32_t wbase = sbase + 2 * ASZ + (kt & 1) * WSZ;
#pragma unroll
    for (int ks = 0; ks < KT / 32; ++ks) {
      uint32_t a[MW][4], b[NW][2];
#pragma unroll
      for (int i = 0; i < MW; ++i) {
        const int r = wm * MW * 16 + i * 16 + ar;
        if (rb0 + i * 16 < M) ldsm4(abase + r * KT + swz(r, 2 * ks + ach) * 16, a[i]);
      }
#pragma unroll
      for (int j = 0; j < NW / 2; ++j) {
        const int r = wn * NW * 8 + j * 16 + br;
        uint32_t qd[4];
        ldsm4(wbase + r * KT + swz(r, 2 * ks + bch) * 16, qd);
        b[2 * j][0] = qd[0]; b[2 * j][1] = qd[1];
        b[2 * j + 1][0] = qd[2]; b[2 * j + 1][1] = qd[3];
      }
#pragma unroll
      for (int i = 0; i < MW; ++i)
        if (rb0 + i * 16 < M)
#pragma unroll
          for (int j = 0; j < NW; ++j) mma(a[i], b[j][0], b[j][1], acc[i][j]);
    }
    __syncwarp();
    if (lane == 0) {
      mbar_arrive(aempty);
      mbar_arrive(wempty + 8 * (kt & 1));
    }
  }
  const int col0 = nt * BN + wn * NW * 8;
#pragma unroll
  for (int i = 0; i < MW; ++i)
    epilogue<N, NW>(acc[i], p.out, rb0 + i * 16, col0, lane, wn, rb0 + i * 16 < M);
}

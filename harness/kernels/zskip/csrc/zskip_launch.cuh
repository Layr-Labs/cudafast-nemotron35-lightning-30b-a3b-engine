// zskip per-config entry points (declared in zskip.cuh): the kernel launches of zskip_down and the
// kernel attribute queries of zskip_config and zskip_init. Each zskip_<part>.cu includes this file and
// instantiates them for the configs of its lists (ZSKIP_INST* below), which compiles their kernels in
// that unit.
#pragma once
#include "zskip.cuh"

template <int WPC, int R, int NH, int BM>
int launch4(const Params& p, cudaStream_t stream) {
#ifndef ZSKIP_NO_PDL
  if (g_pdl) {
    // A rejected launch is returned (after clearing it) so the caller can retry without PDL.
    cudaError_t e = launch_prep_pdl<BM, 8 * NH>(p, stream);
    if (e == cudaSuccess)
      e = launch_pdl(down4_kernel<WPC, R, NH, BM>, dim3(42 / WPC, p.max_blocks), dim3(WPC * 32), stream, p);
    if (e != cudaSuccess) {
      cudaGetLastError();
      return (int)e;
    }
    return (int)cudaGetLastError();
  }
#endif
  launch_prep<BM, 8 * NH>(p, stream);
  down4_kernel<WPC, R, NH, BM><<<dim3(42 / WPC, p.max_blocks), WPC * 32, 0, stream>>>(p);
  return (int)cudaGetLastError();
}

#ifndef ZSKIP_NO_FUSED
template <int WPC, int R, int NH, int BM>
int launch4f(const Params& p, cudaStream_t stream) {
#ifndef ZSKIP_NO_PDL
  if (g_pdl) {
    cudaError_t e = launch_pdl(down4_kernel<WPC, R, NH, BM, true>, dim3(42 / WPC, p.max_blocks),
                               dim3(WPC * 32), stream, p, BI_BYTES);
    if (e != cudaSuccess) {
      cudaGetLastError();
      return (int)e;
    }
    return (int)cudaGetLastError();
  }
#endif
  down4_kernel<WPC, R, NH, BM, true><<<dim3(42 / WPC, p.max_blocks), WPC * 32, BI_BYTES, stream>>>(p);
  return (int)cudaGetLastError();
}
#endif

#ifndef ZSKIP_NO_PAIR
template <bool Fused, bool HalfSkip, int R>
static int launch_pair_impl(const Params& p, cudaStream_t stream) {
  constexpr int SM = Fused ? BI_BYTES : 0;
#ifndef ZSKIP_NO_PDL
  if (g_pdl) {
    cudaError_t e = cudaSuccess;
    if constexpr (!Fused)
      e = launch_prep_pdl<16, 8>(p, stream);
    if (e == cudaSuccess && p.c3lpt)
      e = launch_pdl(c3lpt_order_kernel, dim3(1), dim3(32), stream, p);
    if (e == cudaSuccess)
      e = launch_pdl(down_pair_kernel<Fused, HalfSkip, R>, dim3(21, p.max_blocks), dim3(32), stream, p, SM);
    if (e != cudaSuccess) {
      cudaGetLastError();
      return (int)e;
    }
    return (int)cudaGetLastError();
  }
#endif
  if constexpr (!Fused) launch_prep<16, 8>(p, stream);
  if (p.c3lpt) c3lpt_order_kernel<<<1, 32, 0, stream>>>(p);
  down_pair_kernel<Fused, HalfSkip, R><<<dim3(21, p.max_blocks), 32, SM, stream>>>(p);
  return (int)cudaGetLastError();
}
template <bool Fused, bool HalfSkip>
int launch_pair(const Params& p, cudaStream_t stream) {
  return g_rds_ring2 ? launch_pair_impl<Fused, HalfSkip, 2>(p, stream)
                     : launch_pair_impl<Fused, HalfSkip, 3>(p, stream);
}

template <int WPC, int R, int F>
int launch_lean(const Params& p, cudaStream_t stream) {
#ifndef ZSKIP_NO_PDL
  if (g_pdl) {
    cudaError_t e = launch_pdl(down_lean_kernel<WPC, R, F>, dim3(21 / WPC, p.max_blocks), dim3(WPC * 32), stream, p);
    if (e != cudaSuccess) {
      cudaGetLastError();
      return (int)e;
    }
    return (int)cudaGetLastError();
  }
#endif
  down_lean_kernel<WPC, R, F><<<dim3(21 / WPC, p.max_blocks), WPC * 32, 0, stream>>>(p);
  return (int)cudaGetLastError();
}
#endif

template <int WPC, int S, int NH, int BM>
int launch5(const Params& p, cudaStream_t stream) {
  constexpr int SM = WPC * S * Stage5<NH>::BYTES;
  static bool attr = false;
  if (!attr) {
    if (SM > 48 * 1024 &&
        cudaFuncSetAttribute(down5_kernel<WPC, S, NH, BM>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM) != cudaSuccess)
      return -5;
    attr = true;
  }
  launch_prep<BM, 8 * NH>(p, stream);
  down5_kernel<WPC, S, NH, BM><<<dim3(42 / WPC, p.max_blocks), WPC * 32, SM, stream>>>(p);
  return (int)cudaGetLastError();
}
template <int WPC, int R, int NH, int BM>
int launch6(const Params& p, cudaStream_t stream) {
  launch_prep<BM, 8 * NH>(p, stream);
  down6_kernel<WPC, R, NH, BM><<<dim3(42 / WPC, p.max_blocks), WPC * 32, 0, stream>>>(p);
  return (int)cudaGetLastError();
}
template <int WPC, int R, int NH, int BM>
int launch7(const Params& p, cudaStream_t stream) {
  launch_prep<BM, 8 * NH>(p, stream);
  down7_kernel<WPC, R, NH, BM><<<dim3(42 / WPC, p.max_blocks), WPC * 32, 0, stream>>>(p);
  return (int)cudaGetLastError();
}
template <int WPC, int R, int NH, int BM>
int launch8(const Params& p, cudaStream_t stream) {
  launch_prep<BM, 8 * NH>(p, stream);
  down8_kernel<WPC, R, NH, BM><<<dim3(42 / WPC, p.max_blocks), WPC * 32, 0, stream>>>(p);
  return (int)cudaGetLastError();
}

template <int W, int RB, int NR, int QB, int NQ>
int launch3(const Params& p, cudaStream_t stream, int ctas_per_sm) {
  using L = Layout3<W, RB, NR, QB, NQ>;
  static int occ = 0;
  if (occ == 0) {
    cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, down3_kernel<W, RB, NR, QB, NQ>, W * 32,
                                                                  L::BYTES);
    if (e != cudaSuccess) return (int)e;
    if (occ < 1) return -4;
  }
  if (g_sms == 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, dev);
  }
  const int per_sm = ctas_per_sm > 0 ? min(ctas_per_sm, occ) : occ;
  launch_prep<16, 8>(p, stream);
  down3_kernel<W, RB, NR, QB, NQ><<<g_sms * per_sm, W * 32, L::BYTES, stream>>>(p);
  return (int)cudaGetLastError();
}

// zskip_config: info[0] = registers, [1] = dynamic smem bytes, [2] = warps, [3] = CTAs per SM; 0, or -1
template <int W, int RB, int NR, int QB, int NQ>
int info3(int* info) {
  cudaFuncAttributes attr;
  if (cudaFuncGetAttributes(&attr, down3_kernel<W, RB, NR, QB, NQ>) != cudaSuccess) return -1;
  info[0] = attr.numRegs;
  info[1] = Layout3<W, RB, NR, QB, NQ>::BYTES;
  info[2] = W;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&info[3], down3_kernel<W, RB, NR, QB, NQ>,
                                                    W * 32, Layout3<W, RB, NR, QB, NQ>::BYTES) != cudaSuccess)
    return -1;
  return 0;
}
template <int WPC, int R, int NH, int BM>
int info4(int* info) {
  cudaFuncAttributes attr;
  if (cudaFuncGetAttributes(&attr, down4_kernel<WPC, R, NH, BM>) != cudaSuccess) return -1;
  info[0] = attr.numRegs;
  info[1] = 0;
  info[2] = WPC;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&info[3], down4_kernel<WPC, R, NH, BM>, WPC * 32, 0) != cudaSuccess)
    return -1;
  return 0;
}
#ifndef ZSKIP_NO_FUSED
template <int WPC, int R, int NH, int BM>
int info4f(int* info) {
  cudaFuncAttributes attr;
  if (cudaFuncGetAttributes(&attr, down4_kernel<WPC, R, NH, BM, true>) != cudaSuccess) return -1;
  info[0] = attr.numRegs;
  info[1] = BI_BYTES;
  info[2] = WPC;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&info[3], down4_kernel<WPC, R, NH, BM, true>,
                                                    WPC * 32, BI_BYTES) != cudaSuccess)
    return -1;
  return 0;
}
#endif
#ifndef ZSKIP_NO_PAIR
template <bool Fused, bool HalfSkip>
int info_pair(int* info) {
  constexpr int SM = Fused ? BI_BYTES : 0;
  auto kernel = g_rds_ring2 ? down_pair_kernel<Fused, HalfSkip, 2> : down_pair_kernel<Fused, HalfSkip, 3>;
  cudaFuncAttributes attr;
  if (cudaFuncGetAttributes(&attr, kernel) != cudaSuccess) return -1;
  info[0] = attr.numRegs; info[1] = SM; info[2] = 1;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&info[3], kernel, 32, SM) != cudaSuccess)
    return -1;
  return 0;
}
template <int WPC, int R, int F>
int info_lean(int* info) {
  cudaFuncAttributes attr;
  if (cudaFuncGetAttributes(&attr, down_lean_kernel<WPC, R, F>) != cudaSuccess) return -1;
  info[0] = attr.numRegs; info[1] = 0; info[2] = WPC;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&info[3], down_lean_kernel<WPC, R, F>, WPC * 32, 0) != cudaSuccess)
    return -1;
  return 0;
}
#endif

// zskip_init: load the kernels
template <int WPC, int R, int NH, int BM>
cudaError_t load4() {
  cudaFuncAttributes attr;
  return cudaFuncGetAttributes(&attr, down4_kernel<WPC, R, NH, BM>);
}
#ifndef ZSKIP_NO_PAIR
template <bool Fused, bool HalfSkip>
cudaError_t load_pair() {
  cudaFuncAttributes attr;
  cudaError_t e;
  if ((e = cudaFuncGetAttributes(&attr, down_pair_kernel<Fused, HalfSkip, 2>)) != cudaSuccess) return e;
  return cudaFuncGetAttributes(&attr, down_pair_kernel<Fused, HalfSkip, 3>);
}
#endif

// The entry points of one config, for its unit; arguments as in the config lists.
#define ZSKIP_INST3(ID, W_, RB_, NR_, QB_, NQ_)                                   \
  template int launch3<W_, RB_, NR_, QB_, NQ_>(const Params&, cudaStream_t, int); \
  template int info3<W_, RB_, NR_, QB_, NQ_>(int*);
#define ZSKIP_INST4(ID, W_, R_, NH_, BM_)                                 \
  template int launch4<W_, R_, NH_, BM_>(const Params&, cudaStream_t); \
  template int info4<W_, R_, NH_, BM_>(int*);                           \
  template cudaError_t load4<W_, R_, NH_, BM_>();
#define ZSKIP_INST4F(ID, W_, R_, NH_, BM_)                                 \
  template int launch4f<W_, R_, NH_, BM_>(const Params&, cudaStream_t); \
  template int info4f<W_, R_, NH_, BM_>(int*);
#define ZSKIP_INST5(ID, W_, S_, NH_, BM_) template int launch5<W_, S_, NH_, BM_>(const Params&, cudaStream_t);
#define ZSKIP_INST6(ID, W_, R_, NH_, BM_) template int launch6<W_, R_, NH_, BM_>(const Params&, cudaStream_t);
#define ZSKIP_INST7(ID, W_, R_, NH_, BM_) template int launch7<W_, R_, NH_, BM_>(const Params&, cudaStream_t);
#define ZSKIP_INST8(ID, W_, R_, NH_, BM_) template int launch8<W_, R_, NH_, BM_>(const Params&, cudaStream_t);
#define ZSKIP_INST_PAIR(FUSED, HSKIP)                                    \
  template int launch_pair<FUSED, HSKIP>(const Params&, cudaStream_t); \
  template int info_pair<FUSED, HSKIP>(int*);                          \
  template cudaError_t load_pair<FUSED, HSKIP>();
#define ZSKIP_INST_LEAN(ID, W_, R_, F_)                                 \
  template int launch_lean<W_, R_, F_>(const Params&, cudaStream_t); \
  template int info_lean<W_, R_, F_>(int*);

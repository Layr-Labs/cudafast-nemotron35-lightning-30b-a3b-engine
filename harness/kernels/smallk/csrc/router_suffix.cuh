// SPDX-License-Identifier: Apache-2.0
// Optional source-dead row-chain pruning; the original FP32 kernel stays intact.
#pragma once

// MODE 0: chain = fmaf(w, x, acc) from acc = 0. MODE 1: chain = acc + (w * x) (separate rounding).
template <int R, int M, int MODE, bool W16 = false>
__global__ void __launch_bounds__(kLanes * kMaxTiles)
    router_gemm_suffix_kernel(const float* __restrict__ x, long ldx, const void* __restrict__ w, int K,
                       float* __restrict__ y, long ldy, int N, const int* __restrict__ vrows) {
  const int t = threadIdx.x;
  const int r = t & (kLanes - 1);
  const int j = t / kLanes;
  const int J = (K + kTile - 1) / kTile;
  const int n0 = blockIdx.x * R;
  __shared__ float part[R * M][kMaxTiles][kLanes];
  __shared__ float lane_sum[R * M][kLanes];

  pdl_wait();
  // verify-row count (nullptr: every row), loaded early so its latency hides under the main loop
  const int vr = vrows == nullptr ? M : *vrows;
  const int live = min(M, max(vr, 1));
  if (j < J) {
    const int len = min(kLanes, (K - j * kTile) / kLanes);
    const int base = j * kTile + r;
    float acc[R][M];
#pragma unroll
    for (int a = 0; a < R; ++a)
#pragma unroll
      for (int m = 0; m < M; ++m) acc[a][m] = 0.0f;
    long wrow[R];
#pragma unroll
    for (int a = 0; a < R; ++a) {
      const int n = min(n0 + a, N - 1);
      wrow[a] = static_cast<long>(n) * K + base;
    }
    if (len == kLanes) {
      float wv[kLanes][R];
#pragma unroll
      for (int i = 0; i < kLanes; ++i)
#pragma unroll
        for (int a = 0; a < R; ++a) wv[i][a] = ldw<W16>(w, wrow[a] + i * kLanes);
#pragma unroll
      for (int i = 0; i < kLanes; ++i) {
        float xv[M];
#pragma unroll
        for (int m = 0; m < M; ++m) {
          xv[m] = 0.0f;
          if (m < live) xv[m] = __ldg(x + m * ldx + base + i * kLanes);
        }
#pragma unroll
        for (int a = 0; a < R; ++a)
#pragma unroll
          for (int m = 0; m < M; ++m) {
            if (m < live && MODE == 0) {
              acc[a][m] = __fmaf_rn(wv[i][a], xv[m], acc[a][m]);
            } else if (m < live) {
              acc[a][m] = __fadd_rn(acc[a][m], __fmul_rn(wv[i][a], xv[m]));
            }
          }
      }
    } else {
      for (int i = 0; i < len; ++i) {
        float xv[M];
#pragma unroll
        for (int m = 0; m < M; ++m) {
          xv[m] = 0.0f;
          if (m < live) xv[m] = __ldg(x + m * ldx + base + i * kLanes);
        }
#pragma unroll
        for (int a = 0; a < R; ++a) {
          const float wa = ldw<W16>(w, wrow[a] + i * kLanes);
#pragma unroll
          for (int m = 0; m < M; ++m) {
            if (m < live && MODE == 0) {
              acc[a][m] = __fmaf_rn(wa, xv[m], acc[a][m]);
            } else if (m < live) {
              acc[a][m] = __fadd_rn(acc[a][m], __fmul_rn(wa, xv[m]));
            }
          }
        }
      }
    }
#pragma unroll
    for (int a = 0; a < R; ++a)
#pragma unroll
      for (int m = 0; m < M; ++m) part[a * M + m][j][r] = acc[a][m];
  }
  __syncthreads();
  // tiles ascending per lane
  for (int q = t; q < R * M * kLanes; q += blockDim.x) {
    const int am = q / kLanes;
    const int lane = q % kLanes;
    float s = part[am][0][lane];
    for (int jj = 1; jj < J; ++jj) s = __fadd_rn(s, part[am][jj][lane]);
    lane_sum[am][lane] = s;
  }
  __syncthreads();
  // lanes ascending
  if (t < R * M) {
    const int a = t / M;
    const int m = t % M;
    // rows at or past vr take the last valid row's logits bit for bit, so they route to experts
    // that row already loads
    const int src = min(m, max(vr, 1) - 1);
    const int am = a * M + src;
    float s = lane_sum[am][0];
#pragma unroll
    for (int lane = 1; lane < kLanes; ++lane) s = __fadd_rn(s, lane_sum[am][lane]);
    if (n0 + a < N) y[m * ldy + n0 + a] = s;
  }
  pdl_launch_dependents();
}

template <int R, bool W16>
int router_suffix_launch(const float* x, long ldx, int m, const void* w,
                         float* y, long ldy, int pdl, const int* vrows,
                         cudaStream_t stream) {
  dim3 grid(128 / R), block(kLanes * ((2688 + kTile - 1) / kTile));
  if (m == 5) return launch(router_gemm_suffix_kernel<R,5,0,W16>, grid, block, pdl,
                            stream, x, ldx, w, 2688, y, ldy, 128, vrows);
  if (m == 6) return launch(router_gemm_suffix_kernel<R,6,0,W16>, grid, block, pdl,
                            stream, x, ldx, w, 2688, y, ldy, 128, vrows);
  return 2;
}

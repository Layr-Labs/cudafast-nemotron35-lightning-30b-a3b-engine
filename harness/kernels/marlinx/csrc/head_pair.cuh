// Optional lm_head-only experiment. Included inside namespace marlinx.
// Each original four-warp chain computes both h=0/1 of the SAME original col.
// The two B/S/acc/R chains and segment folds remain separate; only A is shared.
// This changes scheduling/register pressure, not the declared weight format.
#ifndef MARLINX_NO_HEAD_PAIR
template <bool PDL>
__global__ void __launch_bounds__(128) head_pair_kernel(const Params p) {
  constexpr int P = 8;
  __shared__ float red0[2][3][16][32];
  __shared__ float red1[2][3][16][32];
  extern __shared__ __align__(16) uint2 pair_su[];
  const int col = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int w = threadIdx.x >> 5;
  const int first_cut = column_first_cut(p, col);
  const long long nB = p.N >> 1;
  const uint4* B0 = p.B + 64 * col + lane;
  const uint4* B1 = B0 + 32;
  const int su_half = (p.K >> 4) * 8;  // uint2, 10,752 bytes per original half
  const uint4* su_src = reinterpret_cast<const uint4*>(
      p.SU + (long long)(2 * col) * su_half);
  uint4* su_dst = reinterpret_cast<uint4*>(pair_su);
  // The two existing adjacent unit-major SU records, with no new weight pack.
  for (int i = threadIdx.x; i < (p.K >> 4) * 8; i += 128)
    su_dst[i] = ld_stream_v4(su_src + i);
  __syncthreads();
  auto load_s0 = [&](int kb) -> uint2 { return pair_su[kb * 8 + (lane >> 2)]; };
  auto load_s1 = [&](int kb) -> uint2 {
    return pair_su[su_half + kb * 8 + (lane >> 2)];
  };
  const int tok = lane >> 2;
  const bool arow = tok < p.M;
  const uint32_t* Ap = reinterpret_cast<const uint32_t*>(
      p.A + (arow ? (long long)tok * p.lda : 0)) + (lane & 3);
  uint4 rb0[P], rb1[P];
  uint2 rs0[P], rs1[P];
  uint32_t ra[P][2];
  Cursor ld, cs;
  cursor_init(ld, first_cut, p.k_tiles, p.iters);
  cs = ld;
  const int total = 2 * p.k_tiles;
#ifndef MARLINX_NO_PDL
  if constexpr (PDL) {
    Cursor pre = ld;
#pragma unroll
    for (int s = 0; s < P; ++s) {
      if (s < total) {
        const int kb = 8 * pre.tau + 2 * w + pre.k;
        rb0[s] = ld_stream_v4(B0 + kb * nB);
        rb1[s] = ld_stream_v4(B1 + kb * nB);
        rs0[s] = load_s0(kb);
        rs1[s] = load_s1(kb);
        cursor_next(pre, p.iters);
      }
    }
    // Only immutable weights/scales were read above. Every A read is below.
    pdl_wait();
  }
#endif
#pragma unroll
  for (int s = 0; s < P; ++s) {
    if (s < total) {
      const int kb = 8 * ld.tau + 2 * w + ld.k;
      if constexpr (!PDL) {
        rb0[s] = ld_stream_v4(B0 + kb * nB);
        rb1[s] = ld_stream_v4(B1 + kb * nB);
        rs0[s] = load_s0(kb);
        rs1[s] = load_s1(kb);
      }
      ra[s][0] = arow ? ld_activation_u32<PDL>(Ap + 8 * kb) : 0u;
      ra[s][1] = arow ? ld_activation_u32<PDL>(Ap + 8 * kb + 4) : 0u;
      cursor_next(ld, p.iters);
    }
  }
  float acc0[4][4], acc1[4][4], R0[4][4], R1[4][4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      acc0[j][e] = 0.0f;
      acc1[j][e] = 0.0f;
      R0[j][e] = 0.0f;
      R1[j][e] = 0.0f;
    }
  bool first_seg = true;
  int par = 0;
  for (int base = 0; base < total; base += P) {
#pragma unroll
    for (int s = 0; s < P; ++s) {
      const int i = base + s;
      if (i < total) {
        block_mma<false>(rb0[s], rs0[s], ra[s], acc0);
        block_mma<false>(rb1[s], rs1[s], ra[s], acc1);
        const bool seg_end = cursor_next(cs, p.iters);
        if (i + P < total) {
          const int kb = 8 * ld.tau + 2 * w + ld.k;
          rb0[s] = ld_stream_v4(B0 + kb * nB);
          rb1[s] = ld_stream_v4(B1 + kb * nB);
          rs0[s] = load_s0(kb);
          rs1[s] = load_s1(kb);
          ra[s][0] = arow ? ld_activation_u32<PDL>(Ap + 8 * kb) : 0u;
          ra[s][1] = arow ? ld_activation_u32<PDL>(Ap + 8 * kb + 4) : 0u;
          cursor_next(ld, p.iters);
        }
        if (seg_end) {
          // Both calls are CTA-uniform. Each retains the original four-warp
          // pp tree, bottom-up fold, and parity for its own red storage.
          close_segment(red0, par, w, lane, acc0, R0, first_seg);
          close_segment(red1, par, w, lane, acc1, R1, first_seg);
          first_seg = false;
          par ^= 1;
        }
      }
    }
  }
#ifndef MARLINX_NO_PDL
  if constexpr (PDL) pdl_launch_dependents();
#endif
  if (!w) {
    write_out<false>(p, col, 0, lane, R0);
    write_out<false>(p, col, 1, lane, R1);
  }
}

static int launch_head_pair(const Params& p, cudaStream_t stream, bool pdl) {
  const size_t smem = (size_t)(p.K >> 4) * 128;  // two existing SU halves
#ifndef MARLINX_NO_PDL
  if (pdl) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(p.n_tiles);
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaError_t rc = cudaLaunchKernelEx(&cfg, head_pair_kernel<true>, p);
    if (rc != cudaSuccess) return (int)rc;
    return (int)cudaGetLastError();
  }
#endif
  head_pair_kernel<false><<<p.n_tiles, 128, smem, stream>>>(p);
  return (int)cudaGetLastError();
}
#endif

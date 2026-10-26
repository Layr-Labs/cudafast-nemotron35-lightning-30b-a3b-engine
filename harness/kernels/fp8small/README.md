# Native FP8 small-M projections

This package accelerates the two fixed Mamba projection shapes on GB10:
`(N,K)=(10304,2688)` and `(2688,4096)`, for one through eight FP8 rows.
It preserves the pinned vLLM 0.27.1 SM120 CUTLASS computation:

1. Original E4M3 activation and weight bytes enter native
   `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32`.
2. Each output starts at positive FP32 zero and follows one ascending-K chain,
   with no split-K or cross-warp reduction.
3. Multiply the accumulator by the original scalar weight scale, then the
   original scalar activation scale, each in FP32; round once to BF16.

The scaling order follows upstream `ScaledEpilogue` in
`csrc/libtorch_stable/cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp`.
The source is pinned by the challenge image. The prototype's generic Triton dot
lowered to FP16 MMA and produced rare bit differences; that implementation is
not part of this package.

The checkpoint-derived weight pack only permutes four-byte groups. Its layout
is `[N/32,K/32,4 warps,8 columns,4 K-lanes,2 K-halves,4 bytes]`. Each warp reads
one contiguous `uint2` per lane. The inverse byte permutation is checked in
the differential proof. No weights are calibrated or requantized. The original
column-major weight remains available to CUTLASS for larger batches.

The kernel uses four warps per CTA and a 16-step register prefetch ring.
No-allocation L1 reads for streamed weights leave room for the small
activation working set. Shape specializations make indexing constants without
changing the arithmetic chain. Two kernel entry points are loaded before graph
capture. Allocation and execution use the current PyTorch CUDA stream.

`py/__init__.py` validates the GPU, dtype, shape, stride and alignment at weight
preparation. The vLLM adapter adds a custom-op boundary so a symbolic compiled
range can dispatch on real M during CUDA-graph capture. Eligibility is based
only on static quantization configuration and tensor metadata. Prefill and
ineligible metadata retain the original CUTLASS call, including its original
activation quantization and output handling.

Setup discovers `build.sh` through the existing kernel-package builder. The
source is compiled inside the pinned image for sm_121a, without fast math.
No installed container files are modified. No runtime tuning or prompt data
is stored. The only persistent new tensor is a byte-preserving weight pack.

Before integration, the selected kernel passed 1,472 real-weight comparisons
across all 23 Mamba layers, both projections, M=1..8 and four activation sets:
random, small, clipped extreme and all FP8 bit patterns. A separate earlier
ring sweep passed 2,944 cases. These tests compare integer BF16 output bits,
including NaNs, with the actual CUTLASS operator. Isolated timings are not
full-model or official leaderboard results.

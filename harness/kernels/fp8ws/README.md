# fp8ws: prefill Mamba in_proj and out_proj FP8 GEMMs

The 496-token prefill runs the Mamba `in_proj` projection, (N,K) = (10304,2688), and
the `out_proj` projection, (N,K) = (2688,4096), as FP8 GEMMs with M = 496 on every
Mamba layer. This package replaces the pinned CUTLASS call for exactly these two
shapes with a persistent kernel that keeps the CUTLASS arithmetic:

1. The original E4M3 activation and weight bytes enter native
   `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32`.
2. Every output starts at positive FP32 zero and follows one ascending-K chain:
   no split-K, no cross-warp reduction.
3. The accumulator is multiplied by the scalar weight scale, then by the scalar
   activation scale, each in FP32, and rounded once to BF16 (`ScaledEpilogue`,
   as in fp8small).

Layout. One CTA per SM claims output tiles of 256 rows x 112 columns (two row
bands; 92 column tiles for the in_proj, 24 for the out_proj) and runs eight MMA
warps of 64 x 56 over them. The kernel is one template instantiated for the two
(N, K) pairs; everything but the tile counts and the K step count is shared. Per K step
of 128 bytes a TMA load brings the activation tile (the producer's row-major
tensor, unchanged; rows past 496 are zero-filled and never stored) with the
128-byte swizzle, and one bulk copy brings the weight tile. Two shared-memory
stages are synchronized by full/empty mbarriers. The epilogue transposes within
each lane quad so every lane stores 16 contiguous output bytes.

Weight pack. At load the column-major weight is copied once into
`[n_tile][k_tile][112][128]` tiles whose 16-byte chunks are XOR-swizzled by row
(chunk c of row r at c ^ (r & 7)). This is a byte permutation, input-independent;
`unpack_reference` inverts it with tensor ops only and the admission compares the
result with the original bytes. No weight is calibrated or requantized. The
original weight stays: CUTLASS serves every other M.

Admission (vllm `kernels/linear/scaled_mm/fp8ws.py`). Each layer's pack is used
only if, at load, its output equals `ops.cutlass_scaled_mm` bit for bit on five
synthetic activation sets (four Gaussian scales and all E4M3 bit patterns
including NaNs) with the layer's own scales and two other scale pairs; any
mismatch or failure keeps CUTLASS for that layer. Dispatch sits in the fp8c and
fp8small custom ops' M > 8 fallback, so it runs on real tensor metadata inside
torch.compile and is recorded into the prefill CUDA graph. `FP8WS=0` disables
the package at load. `harness/tests/fp8ws_exact.py` is the GPU proof.

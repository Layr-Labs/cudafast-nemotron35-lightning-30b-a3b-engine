# bf16ws: prefill attention projection BF16 GEMMs

The 496-token prefill runs each attention layer's `qkv_proj`, (N,K) = (4608,2688), and
`o_proj`, (N,K) = (2688,4096), as BF16 GEMMs with M = 496 (`F.linear`, which cuBLAS
serves with an nvjet kernel on these shapes). This package replaces that call for
exactly these two shapes with a persistent kernel that keeps the same arithmetic:

1. The original BF16 activation and weight values enter native
   `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.
2. Every output starts at positive FP32 zero and follows one ascending-K chain:
   no split-K, no cross-warp reduction.
3. The accumulator is rounded once to BF16.

Layout. The kernel is the fp8ws kernel (`harness/kernels/fp8ws`) with the BF16 MMA:
one CTA per SM claims output tiles of 256 rows x BN columns (two row bands; BN = 96
and 48 column tiles for the `qkv_proj`, BN = 112 and 24 column tiles for the
`o_proj`) and runs eight MMA warps of 64 x BN/2 over them. Per K step of 128 bytes
(64 BF16 values) a TMA load brings the activation tile (the producer's row-major
tensor, unchanged; rows past 496 are zero-filled and never stored) with the 128-byte
swizzle, and one bulk copy brings the weight tile. Two shared-memory stages are
synchronized by full/empty mbarriers. The epilogue transposes within each lane quad
so every lane stores 16 contiguous output bytes. The BF16 m16n8k16 fragments occupy
the same bytes as the E4M3 m16n8k32 fragments of fp8ws, so the ldmatrix addressing is
the same.

Weight pack. At load the row-major [N, K] BF16 weight is copied once into
`[n_tile][k_tile][BN][128 bytes]` tiles whose 16-byte chunks are XOR-swizzled by row
(chunk c of row r at c ^ (r & 7)). This is a byte permutation, input-independent;
`unpack_reference` inverts it with tensor ops only and the admission compares the
result with the original bytes. No weight changes value. The original weight stays:
`F.linear` and the packed decode path serve every other row count.

Admission (vllm `models/nemotron_h_attn_packed.py`, `_prepare_ws`). Each projection's
pack is used only if, at load, its output equals `F.linear` bit for bit on five
synthetic activation sets (four Gaussian scales and every BF16 bit pattern including
infinities and NaNs); any mismatch or failure keeps `F.linear` for that projection.
Dispatch sits in the `nemo_attn_pk_linear` custom op for 496 rows, so it runs on real
tensor metadata inside torch.compile and is recorded into the prefill CUDA graph.
`BF16WS=0` disables the package at load. `harness/tests/bf16ws_exact.py` is the GPU
proof.

# Exact sparse routed down projection

This package builds an sm_121a CUDA kernel for the checkpoint's 128 NVFP4
experts, each with a 2688-by-1856 down projection. It supports one through eight
tokens with six distinct routed experts per token and BF16 activations. The
ordinary Humming kernel handles prefill and unsupported layouts.

At weight loading, weights and group scales are permuted without changing any
byte. Original FP32 global scales are copied unchanged. The additional static
storage is approximately 7.7 GiB across the 23 routed layers. No dequantized
weight cache is stored, and no input-dependent data survives a call.

For each invocation, a small kernel computes four sector bits per K16 group,
combining all current tokens routed to the same expert. A sector can omit its
weight load only when all four corresponding BF16 input values are exactly
zero for every such token. Both signs of zero qualify. These masks are rebuilt
on the current stream before every GEMM, including every CUDA graph replay.

Weights form the A operand and activations the B operand of native BF16
`mma.sync.m16n8k16`. Each output retains one FP32 accumulator and every K16
step in ascending order. The kernel decodes the original FP4 values, applies
the original FP8 group scale divided by four in BF16, then applies four times
the original global scale to the FP32 accumulator before RN BF16 output. This
matches the existing Humming representation and rounding. Its FP32 global-scale
multiply explicitly uses `mul.rn.ftz.f32`, matching Humming's fast-math epilogue
for subnormal operands and results while preserving signed zero. The first
integration test caught this edge case; the arithmetic was corrected rather
than weakening the bit comparison. Zero sectors still
execute their MMA steps. Routing weights and the final expert reduction use
the existing implementation.

The selected kernel uses four warps, a four-step register prefetch ring and
four mask-generation CTAs per routed expert. The initial prototype, reusable
mask variant and 24 launch variants were tested separately. Timing reports
are diagnostic kernel timings, not official model scores.

`harness/tests/routed_sparse_exact.py` checks reversible packing, actual use of
the guarded path, complete expert outputs and large-token fallback against
the original Humming sequence using the pinned checkpoint. Full-model golden
streams, repeated independent prompts and a fresh control/candidate benchmark
are required before drawing an end-to-end performance conclusion.

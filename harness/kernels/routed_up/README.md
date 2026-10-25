# routed_up: Humming-exact NVFP4 routed up for M <= 8

Replaces the humming indexed w13 launch on the verify (1 to 8 tokens, top-k 6).
Prefill stays on humming. The pack is a byte permutation of the checkpoint
`(128, 1856, 1344)` uint8 weight and `(128, 1856, 168)` FP8 group scales, plus
a copy of the FP32 per-expert global scale. No new numeric format.

Arithmetic, per output element, matches humming's batch-invariant chain and
then the existing relu2 wrapper:

1. acc starts at +0.
2. For each K16 group in ascending order, one
   `mma.sync.m16n8k16.row.col.f32.bf16.bf16.f32`. Weights are the A operand.
   FP4 nibbles and FP8 group scales are decoded the same way as `routed_sparse`
   (group scale / 4 in bf16).
3. `mul.rn.ftz.f32` by `global_scale * 4`, then RN to bf16.
4. `max.NaN` against +0, `mul.rn` square, RN to bf16.

`ROUTED_UP=0` or any shape miss keeps the humming relu2 epilogue. The launch
allocates nothing, so a captured verify graph replays it.

## Additional remote admission in this composition

The original whole0a8fd90c-8958-4d41-8417-d8ebc7d882c0 later measured2.21223178900872, below highest41 by5.7754%; it is not a baseline. This source is only an optional new variant. Before enabling it for a real immutable pack/shape, official remote startup compares private activations at three magnitudes plus signed-zero/subnormal patterns against the original real-weight Humming fused-ReLU2 kernel using the same current structural route. It caches only immutable pack identity, metadata and the result, not route values or request inputs/outputs. A capture miss or ordinary mismatch retains Humming.

After correctness, seven alternating cold private timings must show the native variant faster than0.96times the original median. A64MiB private flush precedes timing, GPU events measure only startup trials, and a failed performance admission retains the original path. All executable validation is official remote. No local compile/import/test/timing was performed by this composition. These guards do not prove a whole-bundle speed gain; the official golden/three-pair result decides the combination.

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

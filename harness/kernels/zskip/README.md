# zskip: humming-exact routed down projection that skips zero activation rows

`zskip` replaces humming's routed-expert down projection (w2, NVFP4 W4A16, N = 2688,
K = 1856, 128 experts) for calls of 1 to 8 tokens (the MTP verify: 5 rows at depth 4).
Every bf16 output element is bit-identical to humming's. The MoE activation is relu^2, so
most activations are exact zeros: on real decode windows 88.8% of the (expert block, k)
pairs are zero in every row routed to the expert. Such k rows of W_down only ever multiply
exact zeros, and the kernel does not load them.

`vllm/.../fused_moe/experts/humming_zskip.py` calls it from `HummingIndexedExperts.apply`;
`ZSKIP=0` disables it (calls of up to 8 tokens then use `routed_sparse`); prefill keeps
humming.

## What humming computes for this call

humming-kernels 0.1.10, indexed GEMM, batch-invariant tuned config for up to 24 rows
(`humming_tuned_moe.py`: block 16x128x32, warp 16x32x32, no stream-K, unpadded K = 1856,
NVRTC with `--use_fast_math`). For each output element (row p of an expert block, column
n):

    acc = +0
    for g = 0 .. 115 (k16 groups, ascending):
        acc = mma.sync.m16n8k16.row.col.f32.bf16.bf16.f32(A[p, 16g:16g+16], B[16g:16g+16, n], acc)
    out = cvt.rn.bf16x2.f32(mul.ftz(acc, mul.ftz(gs[e], 4)))

A is the bf16 relu^2 activation in natural k order. B[k, n] is built by bit placement and
two exact bf16 multiplies: `hmul2(hmul2(fp4bits(nibble), 2^124), hmul2(fp8bits(scale), 2^120))`
(`fp_to_fp`, `may_apply_bs_and_zp_on_b` and the exponent offsets 124/120/2 of
`exp_offset.cuh`), i.e. e2m1 x e4m3 / 4 exactly.

## What zskip computes

Kernel `down4_kernel`, production config 113 (two 64-column tiles per warp, one warp per CTA, a
register ring of 3 groups) on layout `swap2`:

- Load-time repack `repack(..., layout="swap2")` of the checkpoint tensors (every byte kept,
  357 MB per layer): per expert and k row, 1344 bytes = 21 pairs of 128 columns x 64 bytes.
  Byte 64P + 8g + 4u + c holds the nibbles of columns 128P + 64u + 16c + g (low) and + 8
  (high), so lane (g, t) of a warp reads the A-fragment nibbles of both its tiles with one
  8-byte load and a k row is one 64-byte segment per warp. Scales: per expert and k16 group,
  2688 bytes, byte 128P + 16g + 8u + 2c + h = the scale of column 128P + 64u + 16c + g + 8h.
  (Layout `swap`, config 100: the same with one 64-column tile per warp, 32-byte segments.)
- Per call, `prep_kernel` (one CTA per expert block) reads the block's rows of the
  activation and writes the 16-bit mask of each group (bit b set when some valid row has a
  nonzero activation at k = 16g + b; +0 and -0 count as zero) and the list of non-empty
  groups.
- One warp per (expert block, 128-column pair) walks the non-empty groups in ascending order.
  Lane (g, t) loads the words of k rows 2t, 2t+1, 2t+8, 2t+9 only when the mask bit is set,
  dequantises with humming's operations, sets the halves of skipped rows to +0, and issues one
  `mma.sync.m16n8k16` per 16 columns with the weights as the A operand and the activations
  (token slots as columns) as B: per output element the same 16 products in the same k-slots
  and the same fp32 chain.
- Epilogue: humming's `mul.rn.ftz.f32` by `gs * 4` and `cvt.rn.bf16x2.f32`.

Exactness rests on these properties, each checked on the device (`tests/zskip_exact.py
--phase probe,epi`): with an accumulator other than -0, `mma(acc, a, b) == mma(acc, a, b')`
when b' is +0 at k-slots where a is zero (1.07e9 MMA outputs over relu^2-like, full-range,
exact-cancellation, huge-B-at-zero, subnormal, sparse and non-finite trials: no mismatch,
and none with acc = -0 either); `mma(-0, a, b) == mma(+0, a, b)` (1.07e9 pairs, with the
original and the substituted b, no mismatch); `mma(acc, 0, b) == acc` for acc other than -0 and NaN
(omitted MMAs of empty groups); the epilogue equals humming's NVRTC fast-math code on all
2^32 fp32 inputs for 8 global scales. As insurance, a warp whose accumulator ever becomes
-0 recomputes its tile with every row loaded (humming's computation without substitution);
it never triggered on any test input.

## Prefill

The 496-token prefill call runs `down4_kernel` config 212 (two 64-column tiles per warp, two
8-row n-blocks per warp sharing each dequantised weight fragment, ring of 2) on a block-16
`moe_align` of the call, so each block holds at most 16 rows; the union mask of a block's rows
selects the k rows to load and L2 deduplicates the re-reads of an expert's blocks. humming's
batch-invariant w2 computes each output with the same ascending k16 chain at every block size,
and the prefill outputs are bit-identical to humming's M=496 w2 on all 23 layers
(nemo-exec-zspf1/zspf2/zsvm). Per layer on captured prompts (nemo-exec-zsvm): 1043-1504 us vs
humming 1528-1680 us and main's config 405 1157-1631 us; over the 22 full layers 26.5 ms vs
35.3 ms (humming) and 29.7 ms (config 405 on its 15 layers, humming elsewhere). Variants with
32-row blocks on humming's own alignment and four n-blocks per warp were exact but slower
(1307-1887 us per layer).

## Evidence

`tests/zskip_exact.py --phase exact` on all 23 MoE layers of the snapshot, against stock
humming (`HummingMethod.forward_layer`, bitwise): real decode windows of the 7 pool
sequences captured from an eager engine (M = 1..8), embedding/random/edge inputs routed by
the real router, 11 adversarial activation sets (92% sparse, -0, subnormal, tiny signed,
signed, huge, NaN/Inf, zero rows, one-hot, all-zero, dense) with random routing, and every
kernel configuration and layout, M = 1..8: all bit-identical (runs nemo-exec-zs-v4, zs-a9, zs4b, zs4c, zs4d).
`humming_zskip.finalize` repeats one comparison per layer at load and falls back to humming
if it differs.

Time per call (CUDA graph over the 23 layers, cold weights, real decode windows; M = 4: 16.4
expert blocks per layer, bytes dense 46.1 MB, zskip 8.2 MB, routed_sparse sectors 17.5 MB;
M = 5: 19.2 blocks, 53.8 / 9.6 / 20.4 MB):

| kernel | M = 4, us per call | M = 5, us per call |
|---|---|---|
| humming (stock) | 191 | 221 |
| routed_sparse (main) | 154 | 180 |
| zskip config 100 (`swap`, 1 tile per warp) | 70.6 | 90.6 |
| zskip config 110 (`swap2`, 2 tiles per warp, ring 4) | 66.7 | 80.9 |
| zskip config 113 (`swap2`, 2 tiles per warp, ring 3) | 66.8 | 73.3 |
| zskip config 110, loads only | 48.2 | |

Variants measured and dropped (all bit-identical): smem-staged cp.async with a persistent
item queue (114 us at M = 4), a single launch with per-warp masks and a per-warp cp.async
ring (115-141 us), L2 prefetch of the rows several groups ahead (77-96 us), and removing the
-0 checks and skipped-row masks (slower: 68.5 and 77 us).

## Files

- `csrc/zskip.cu`: `prep_kernel`, `down4_kernel`, the probes; C ABI `zskip_down`,
  `zskip_init`, `zskip_config`, `zskip_probe`, `zskip_epi`.
- `csrc/gather.cu`: gather-rate probe (`tests/gather_bench.py`).
- `py/__init__.py`: `repack`, `down`, `init`, `scratch`, probes.
- `tests/zskip_exact.py`: probe, epilogue, capture, exactness and timing phases;
  `tests/sequences.json`: the pool sequences for the capture.

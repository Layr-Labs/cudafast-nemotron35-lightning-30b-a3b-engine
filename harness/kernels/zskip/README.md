# zskip: humming-exact routed down projection that skips zero activation rows

`zskip` replaces humming's routed-expert down projection (w2, NVFP4 W4A16, N = 2688,
K = 1856, 128 experts) for calls of 1 to 8 tokens (the MTP verify is 4).
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

Kernel v4 (`down4_kernel`, config 100: 2 warps per CTA, a register ring of 4 groups):

- Load-time repack `repack(..., layout="swap")` of the checkpoint tensors (every byte kept,
  357 MB per layer): per expert and k row, 1344 bytes = 42 tiles of 64 columns x 32 bytes.
  Byte 4g + c of a tile holds the nibbles of columns 64T + 16c + g (low) and + 8 (high).
  Scales: per expert and k16 group, 2688 bytes, byte 64T + 8g + 2c + h = the scale of column
  64T + 16c + g + 8h.
- Per call, `prep_kernel` (one CTA per expert block) reads the block's rows of the
  activation and writes the 16-bit mask of each group (bit b set when some valid row has a
  nonzero activation at k = 16g + b; +0 and -0 count as zero) and the list of non-empty
  groups.
- One warp per (expert block, 64-column tile) walks the non-empty groups in ascending order.
  Lane (g, t) loads the 4-byte words of k rows 2t, 2t+1, 2t+8, 2t+9 only when the mask bit is
  set: the 8 lanes that share a k row read one 32-byte sector. It dequantises with humming's
  operations, sets the halves of skipped rows to +0, and issues one `mma.sync.m16n8k16` per
  16 columns with the weights as the A operand and the activations (token slots as columns)
  as B: per output element the same 16 products in the same k-slots and the same fp32 chain.
- Epilogue: humming's `mul.rn.ftz.f32` by `gs * 4` and `cvt.rn.bf16x2.f32`.

Exactness rests on three properties, each checked on the device (`tests/zskip_exact.py
--phase probe,epi`): with an accumulator other than -0, `mma(acc, a, b) == mma(acc, a, b')`
when b' is +0 at k-slots where a is zero (1.07e9 MMA outputs over relu^2-like, full-range,
exact-cancellation, huge-B-at-zero, subnormal, sparse and non-finite trials: no mismatch,
and none with acc = -0 either); `mma(acc, 0, b) == acc` for acc other than -0 and NaN
(omitted MMAs of empty groups); the epilogue equals humming's NVRTC fast-math code on all
2^32 fp32 inputs for 8 global scales. As insurance, a warp whose accumulator ever becomes
-0 recomputes its tile with every row loaded (humming's computation without substitution);
it never triggered on any test input.

## Evidence

`tests/zskip_exact.py --phase exact` on all 23 MoE layers of the snapshot, against stock
humming (`HummingMethod.forward_layer`, bitwise): real decode windows of the 7 pool
sequences captured from an eager engine (M = 1..4), embedding/random/edge inputs routed by
the real router, 11 adversarial activation sets (92% sparse, -0, subnormal, tiny signed,
signed, huge, NaN/Inf, zero rows, one-hot, all-zero, dense) with random routing, and every
kernel configuration and layout: all bit-identical (runs nemo-exec-zs-v2, zs-v3, zs-v4).
`humming_zskip.finalize` repeats one comparison per layer at load and falls back to humming
if it differs.

Time per call (CUDA graph over the 23 layers, cold weights, real M = 4 windows; 16.4 expert
blocks per layer; bytes: dense 46.1 MB, zskip 8.2 MB, routed_sparse sectors 17.5 MB):

| kernel | us per call |
|---|---|
| humming (stock) | 192.7 |
| routed_sparse (main bce7a11) | 170.2 |
| zskip v3 (smem staging, config 1) | 113.9 |
| zskip v4 (config 100) | 70.7 |
| zskip v4, loads only | 57.9 |

## Files

- `csrc/zskip.cu`: `prep_kernel`, `down4_kernel` (production), `down3_kernel` (cp.async
  staging variant, configs 0..9), the probes; C ABI `zskip_down`, `zskip_init`,
  `zskip_config`, `zskip_probe`, `zskip_epi`.
- `csrc/gather.cu`: gather-rate probe (`tests/gather_bench.py`).
- `py/__init__.py`: `repack`, `down`, `init`, `scratch`, probes.
- `tests/zskip_exact.py`: probe, epilogue, capture, exactness and timing phases;
  `tests/sequences.json`: the pool sequences for the capture.

## New composition experiment (GPT 6.1 Sol / Codex / ultra)

The complete source tree of pending a1c083d7-215a-4b8e-abeb-ea12e7cad022
(tree 4b6706eb4bc5ac901b64ba6211a2d1f13ba3987c) was preserved before composing.
Immediately before upload a1 became the official highest at1.92418420867247;
its complete source is retained. C2 c2a77c1c-7c50-4cd9-9521-ca5cdd5ec009
is its predecessor at1.8354630280356752. The timing/probe evidence above is the original author's
report; this solver performed no local execution. Official validation decides
the full combination's correctness and speed.

New combination: depth4 native BF16 MTP, M5 target router and exact5 verify graph
(with the tiny draft graphs retained), plus the final-layer single-logits-row
prefill routing method from pending81f5f6b9-af28-4c98-b08d-cba5e0eeb588.
The previous whole depth4/M5/slot-graph bundle5eb1514c-5de3-49fa-987d-614e7aa2f787
scored1.8144433342994297 and is not a baseline. Its B3slot graphs are excluded.
This experiment changes the down-projection cost substantially by using zskip;
it does not assert that depth4 is independently beneficial. M5's previous
whole-bundle token pass is not a standalone bit-identity proof against cuBLAS.

The last-MoE method changes only unread final-layer rows during the existing
single-row-logits graph capture; prompt-logprobs and auxiliary-output requests
retain ordinary execution. The selected row, all earlier KV/SSM updates and
all request input refreshes remain. A1 is now the official highest. The whole81 carrier scored1.80685685849703
and is excluded as a baseline; only its prefill-only BundleB is selected for
a fresh combination, with no isolated gain claim. This new combination remains
experimental until official validation finishes.

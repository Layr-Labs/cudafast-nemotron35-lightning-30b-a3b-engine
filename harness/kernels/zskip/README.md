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


## Selective dadon second-pass port on 3e7a4ee (static candidate)

Source: dadon `12f98e36-35d1-41c0-ba8d-8fd101ffeee1`, commit `31889a4`.
Only decode c113 is ported: paired 64-column tiles, one warp/CTA, ring 3,
`swap2` layout. Config 118 adds main's fkiene `bc524594` fused prep to that
kernel. Both use main's unchanged Params ABI, block header, 5120-byte block
info layout, `prep_kernel<16,8>`, dequantisation helpers and epilogue helpers.
Main's configs 100/108, prefill 405, all existing checks and pzskip stay intact.

The paired decode copy is made from the original `swap` buffer at load by a
byte permutation (`to_swap2`), with no dequantisation or numerical conversion.
The original copy remains for pzskip and the original decode fallback. This
costs 359,202,816 additional bytes/layer (about 8.262 GB over 23 layers), plus
bounded load-time permutation temporaries. Capacity must be checked on GB10.
No routed_sparse repack is removed.

At load, original checks run first. c113 then passes a separate bitwise
comparison against humming at M4/M5/M8 with spread and crowded top-6 routing,
signed zeros, an empty k16 group and an all-zero row. c118 repeats those cases;
failure keeps checked c113. A missing paired build, failed c113 check, or
allocation error keeps main's checked c108/c100. Only M5 dispatch uses the new
copy/config; prefill and all other decode sizes retain main's choices.
`ZSKIP_DADON=0` removes the new buffers/path. `ZSKIP_FUSED_PREP=0` disables
both old and new fused prep, leaving c113 if its guard passes. `ZSKIP=0`,
`ZSKIP_PDL=0`, `PZSKIP=0` and prefill switches retain their original roles.

PDL uses the existing launch helper and Python rejection/retry logic. Every
new kernel waits before touching global memory and signals dependents after
that wait. The fused grid has no prep/down edge. `-DZSKIP_NO_PAIR` removes
all paired instantiations; build.sh tries the original four fallback flag
combinations with the pair and then without it. `zskip_init` is unchanged,
so optional paired-kernel loading cannot disable main's checked kernels.

Run the existing device test with `--pair --time-ms 5 --time-configs
100,108,113,118` to cover the new configs alongside main. `--phase exact`
includes real/synthetic/adversarial M1..8 tests in dense and skip modes;
`--phase capture,exact,time --capture all` uses captured decode windows.
These require the pinned CUDA image and snapshot. This port has NOT been
CUDA-compiled, GPU-run, token-verified or timed locally. The new fused variant
has no published timing. Dadon's c113 timing cannot be claimed for c118.


## Lean paired decode (configs 600.., ZSKIP_LEAN, config 653 for M=5)

`csrc/zskip_lean.cuh`, `down_lean_kernel`: config 119's per-output computation (swap2 layout, one
warp per 128-column tile pair, the same MMA chain over the active k16 groups, the same dequantised
values and epilogue), with a shorter instruction stream and L2 prefetch:

- prep in the CTA without k lists (one 32-bit `mask << 16 | group` entry per active group, two
  activation rows per round trip, no barrier); one address add per load; skipped k rows load +0;
- an empty 8-row half keeps the previous finite fragments (as pzskip 336 keeps stale weights;
  F_ZHALF gives 119's literal zeros); hi nibbles decoded from x01 directly; scale duplicates folded
  into the HMUL2 operand selectors; a ring of R + 1 slots refilled before each group's math;
- 119's -0 guard on every intermediate accumulator state (placed after the next group's
  dequantisation, plus the final state), dense recompute on a -0;
- F_WPF (653: R = 3, D = 2): each lane prefetches its own pieces of group j + R + D into L2.
  F_BPF (613) bulk-prefetches whole rows per block instead.

Bitwise: 5152 captured real decode windows (23 layers x 7 sequences x 4 windows x M=1..8) equal
stock humming's w2 output and config 119; 5888 adversarial cases (all -0, every zero as -0, 0.5/2/8%
sparse, mixed signed zeros, subnormal, huge, NaN/Inf, zero rows, one-hot, empty groups, dense) equal
119 in SKIP and DENSE; 3220 verify-prefix cases equal 119. `humming_zskip._prepare_lean` repeats
the M=1..8 and prefix checks against humming and the checked pair config at load.
It is on by default (`ZSKIP_LEAN=0` keeps the checked pair config); `ZSKIP_LEAN_CONFIG` picks another lean config.

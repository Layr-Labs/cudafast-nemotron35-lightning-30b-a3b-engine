# marlinx: Marlin-exact NVFP4 W4A16 GEMM for M <= 8

`marlinx` replaces vLLM's dense Marlin kernel (`ops.marlin_gemm`, reached through
`MarlinNvFp4LinearKernel.apply_weights`) for the Nemotron-3.5 NVFP4 W4A16 dense
layers when a call has 1 to 8 rows: the 23 target shared experts (up 2688->3712,
down 3712->2688) and `lm_head` (2688->131072). Every bf16 output element is
bit-identical to stock Marlin. It reads the same Marlin-format tensors the layer
already holds (no repack, no extra memory). `MARLINX=0` disables it. Calls outside
the guard in `py/__init__.py` (shape allowlist, 1 <= M <= 8, bf16, 16-byte aligned
rows, no bias, not under `torch.compile` tracing) run stock Marlin, so prefill
(M ~ 496) is untouched.

## What stock Marlin computes for these calls

Sources: vLLM v0.27.1 (commit 6e448d0) `csrc/libtorch_stable/quantization/marlin/`
`marlin.cu`, `marlin_template.h`, `dequant.h`, `marlin_mma.h`, and
`model_executor/layers/quantization/utils/marlin_utils_fp4.py`.

**Configuration.** `apply_fp4_marlin_linear` calls `marlin_gemm` with
`use_fp32_reduce=True` and `use_atomic_add=False` (N >= 2048 and
`VLLM_MARLIN_USE_ATOMIC_ADD` unset). For M <= 16, `marlin_mm` takes the small-batch
thread configs; `{thread_k 128, thread_n 128, 256 threads}` is valid for all three
shapes and the `{128, 64, 128}` override needs `N/128 * 4 <= #SMs`, false here. So
the kernel is `Marlin<bf16, fe2m1f, bf16, fe4m3fn, 256, 1, 8, 8, m_block_size_8 =
(M <= 8), stages 4, group_blocks 1>` with grid = #SMs (48 on GB10). M = 1..8 run
the same kernel; output column t depends only on activation row t, so Marlin is
row-invariant on this range. M >= 9 switches to `m_block_size_8 = false` and is
not covered.

**Partition (per 128-feature column).** `k_tiles = K/128`, `n_tiles = N/128`.
If `n_tiles > grid`: `part2 = n_tiles % grid`, plus `grid` when `part2*3 <= grid`;
otherwise `part2 = n_tiles`. `iters = ceil(k_tiles*part2 / grid)`. Columns
`< n_tiles - part2` are reduced whole by one CTA. In the last `part2` columns,
k-tile `r` of column `c'` has the linear index `L = c'*k_tiles + r`, and CTA `b`
owns `[b*iters, (b+1)*iters)`, so a column is cut into segments at multiples of
`iters`.

| layer | k_tiles | n_tiles | part2 | iters | segments per column |
|---|---|---|---|---|---|
| shared up | 21 | 29 | 29 | 13 | 2 or 3 |
| shared down | 29 | 21 | 21 | 13 | 3 or 4 |
| lm_head | 21 | 1024 | 64 | 28 | 1 (cols 0..959), 1 or 2 (cols 960..1023) |

**Weights.** For k16 block `kb` and 64-feature half `h` of column `col`, lane `l`
of the warp at warp column `h` reads the int4 `B[kb][64*col + 32*h + l]` of the
`[K/16][N/2]` int4 weight and the 8 scale bytes at byte `kb*N + 128*col + 64*h +
8*(l/4)` of the `[K/16][N]` scales. Scales: `dequant_fp8_scales<bf16, fe4m3fn>`
(bit placement of Marlin's S0E5M3 bytes into bf16), giving 4 bf16x2 `fs[0..3]`.
For `j = 0..3`, word `q = int4[j]`: `frag_b0 = dequant<bf16, fe2m1f, skip_flop>(q
<< 8)`, `frag_b1 = dequant(q)` (bit placement, value = fp4 * 2^-126), then
`frag_b0 = __hmul2(frag_b0, {fs[j].x, fs[j].x})`, `frag_b1 = __hmul2(frag_b1,
{fs[j].y, fs[j].y})` (exact products). The fp32 global scale already carries
`2^(126-7)` and Marlin's power-of-two rescale.

**MMA.** `mma_trans`: `mma.sync.m16n8k16.row.col.f32.bf16.bf16.f32` with A =
`{frag_b0[0], frag_b1[0], frag_b0[1], frag_b1[1]}` (16 features x 16 k) and B =
the activations of 8 tokens, `{A[l/4][16kb + 2(l%4) + {0,1}], A[l/4][16kb + 8 +
2(l%4) + {0,1}]}` (ldmatrix x2 layout). Output `c0, c1` = features `16j + l/4`,
tokens `2(l%4) + {0,1}`; `c2, c3` = features `+8`.

**Per output element** (token t, feature n of column `col`):

1. Each segment `[r0, r1)` of the column is computed by one CTA whose 4 warp rows
   `w = 0..3` each run one fp32 chain from +0 over k16 blocks `8*tau + 2*w` then
   `8*tau + 2*w + 1`, `tau = r0 .. r1-1` ascending.
2. `thread_block_reduce`: `P = acc0 + (acc1 + (acc3 + acc2))`.
3. `global_reduce_fp32` in lock order: the CTA owning the highest-K segment is
   `slice_idx 0`; `R = P_bottom`, then `R = P + R` for each segment going up.
4. `write_result`: `bf16_rn(R * global_scale)`.

No bias, no per-column scale (group_blocks = 1), no fp16 reduce, no atomics.

## What marlinx changes

Only the schedule: one CTA of 4 warps per 64-feature half column (grid = 2 *
N/128: 58, 42 and 2048 CTAs). Warp `w` runs chain `w` of every segment of its half
column, streaming its int4/scale/activation words from global memory through a
register ring of 8 k16 blocks. Segments are
processed bottom-up, so the fold `R = P + R` happens as each segment closes,
through shared memory (double-buffered, one barrier per segment). There are no
inter-CTA reductions, locks or workspace. A CTA has 128 threads and 12 KB of static
shared memory, where a Marlin CTA has 256 threads and requests the full 99 KB, so
marlinx CTAs share SMs with the routed-expert kernel when the shared expert runs on
its aux stream. Weight-dequant, MMA operands and every fp32 add and
multiply are Marlin's, in Marlin's order.

## Files

- `csrc/marlinx.cu`: the kernel (`gemm_kernel<depth>`, depth 8 by default, 4
  available through `MARLINX_DEPTH`), `marlinx_gemm` (C ABI), and a streaming-read
  probe (`marlinx_hog`) used by the tests.
- `py/__init__.py`: ctypes loader, `prepare(layer)`, `maybe_apply(layer, x)`.
- `build.sh`: builds `libmarlinx.so` into `$KERNELS_OUT/marlinx` (called by
  `harness/kernels/build.sh`).
- `tests/marlinx_exact.py`: bit-identity against `ops.marlin_gemm` on the snapshot
  weights for M = 1..8, and timings (`bin/atom nemo exec SHA
  harness/kernels/marlinx/tests/marlinx_exact.py`).

## Measurements (GB10, 48 SMs)

Isolated, CUDA graph, weights not L2-resident (`tests/marlinx_exact.py`):

| call | stock Marlin | marlinx |
|---|---|---|
| lm_head M=1 | 851-861 us (230-233 GB/s) | 798-809 us (245-248 GB/s) |
| lm_head M=4 | 858-883 us (224-231 GB/s) | 813-822 us (241-244 GB/s) |
| shared up M=4 | 27.9 us | 30.4 us |
| shared down M=4 | 27.9 us | 35.7 us |

End to end at depth 3 (7 prompts x 3 reps, base 31d126c): decode 8.756 -> 8.604
ms/token, exact vs the oracle. lm_head alone accounts for about half of it
(8.681 ms/token with the shared expert left on Marlin).

Variants tried and dropped (all bit-identical): a per-warp cp.async ring in shared
memory makes the shared expert faster in isolation (26.5-27 us) but helps less end
to end (8.732 ms/token), because it takes more bandwidth from the routed-expert
kernel that runs beside it; it is slower for lm_head (850-870 us). L2 prefetch
ahead of the register ring and a persistent grid were slower for lm_head. Ring
depths 4-16 perform the same.

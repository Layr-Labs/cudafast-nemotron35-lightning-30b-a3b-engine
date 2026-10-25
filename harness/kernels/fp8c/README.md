# fp8c: lossless 7-bit weight code for the Mamba FP8 projections

This package reads the two Mamba projection weights on GB10 from a lossless
7-bit coding of their FP8 E4M3 bytes, for one through eight rows. The shapes
are `(N,K)=(10304,2688)` (in_proj) and `(2688,4096)` (out_proj). Decode is
DRAM-bound, and these weights are about a fifth of the bytes one verify cycle
reads.

The arithmetic is fp8small's (`harness/kernels/fp8small`). Each lane rebuilds,
in registers, the exact `uint32` B-fragment words that fp8small loads. It hands
them to the same `mma.sync.aligned.kind::f8f6f4.m16n8k32` e4m3 instruction, in
the same ascending-K chain from +0, then applies the same FP32 scale B, FP32
scale A and RN BF16 epilogue. The output bits are equal by construction, and
the tests check them anyway.

## Format (load-time, input-independent)

Each lane computes one output column, so the code is per column. The encoder
starts from the fp8small pack.

- **Window.** It takes the window of eight consecutive exponents
  `[e0, e0+7]` that holds most of the column.
- **Code.** A byte `s|eeee|mmm` with `e` in the window becomes the 7-bit code
  `s|(e-e0)|mmm`.
- **Escape.** The column's least frequent window code is the escape. Bytes
  outside the window, and bytes equal to the escape value, are coded as the
  escape, and their original byte goes to the exception area.

On the pinned checkpoint, 1.3-2.0% of the weights are escapes. A single
per-tensor window failed on small-magnitude columns of in_proj.

A 32-column tile is the work of one CTA. Each tile has one region, laid out in
the order the CTA reads it:

| Part | Content |
|---|---|
| header (256 B in_proj, 384 B out_proj) | `u16` exception offsets per (chunk, warp) plus the final total; `u16 e0 \| esc<<8` per column in the last 64 bytes |
| `C` chunks x 3584 B | a chunk is 4 K-steps (128 K); planes A (4 words per lane, `v4` load), B (2 words, `v2`), D (1 word) |
| exception bytes | ordered (chunk, warp, lane, word, byte); per-tensor capacity |

Per lane and chunk, fp8small reads 8 words `R0..R7`, where `R[2s+h]` is step
`s`, half `h`. Here they are 7 words: byte `j` of word `i<7` holds
`code(R_i byte j) | bit i of code(R_7 byte j) << 7`.

The 46 tensors total 800.5 MB against 890.3 MB, or 7.194 bits per weight
(7.15-7.25 per tensor).

## Decoder

Per chunk, each lane:

- unpacks its 8 code words;
- flags escapes with a SIMD byte compare;
- expands codes with one masked add and one shift-or per word;
- counts its escapes;
- takes its exclusive prefix from three bit-plane ballots.

Two shuffles of the warp-chunk's exception window, one word per lane and up to
32 words, bring the lane's escape bytes. Patching is branch-free: the escape
byte is replicated by `prmt` and merged under a 0xff-per-escape byte mask with
one LOP3. One pass handles the lowest escape of each word. A second pass runs
only when some lane has two in one word. The per-escape shuffle loop is a rare
warp-uniform fallback for three or more in a word, more than 7 in a chunk, or
bytes beyond the two words.

A timing probe found that divergent per-word loops cost about twice the
L2-resident runtime, so none remain on the fast path. The decode of chunk c+1
is issued after the MMAs of chunk c, so the two latencies overlap.

The ring holds 3 chunks for in_proj and 4 for out_proj. `FP8C_RING=3|4|6`
overrides it. in_proj gains more from occupancy: 128 registers, 4 CTAs per SM.
The PDL variant issues the header, the planes and the windows before
`griddepcontrol.wait`; activations, scales and stores come after it, as in
fp8small.

## Load-time guard

`prepare(pack)` encodes the tensor, then runs `fp8c_decode` over the whole
buffer. That kernel uses the same loads and the same `decode()`, with stores in
place of MMAs. Every byte is compared with the fp8small pack. `prepare` returns
`None`, the layer keeps fp8small, and the reason is logged, on any of:

- a mismatch;
- an unsupported GPU or shape;
- a warp-chunk with more than 124 escapes;
- a buffer above 93% of the original bytes.

The original column-major weight stays for CUTLASS (prefill and M > 8). The
fp8small pack is dropped for an accepted layer. Encoding and checking all 46
tensors takes about 2 s at load.

## Integration

- **Custom op.** `vllm/model_executor/kernels/linear/scaled_mm/fp8c.py`
  registers `fp8c_scaled_mm`. As for fp8small, the guard runs on real metadata
  inside the custom op.
- **Weight.** `cutlass.py` registers `_fp8c_weight` on the layer.
- **Prefetch.** The l2pf hooks prefer `_fp8c_weight`: the Mamba out_proj
  window, the MoE window to the next in_proj, and the runner's layer-0
  prefetch. The buffer is tile-major, so its head is what the kernel reads
  first.
- **Boot log.** The first call logs one line:
  `fp8c active: 46 Mamba FP8 projections ... (890.3 MB -> 800.5 MB, 7.194 bits/weight, 0 kept fp8small)`.
- **Switches.** `FP8C=0` restores fp8small; `FP8C_PDL=0` turns off the PDL
  launch.

## Proof and timings

`tests/fp8c_exact.py` runs on the real weights of all 23 layers and both
projections, at M=1, 4, 5 and 8. It uses random, small, extreme and
all-bit-pattern activations, ring 3/4/6, PDL on and off, and CUDA graph replay
with changed inputs. Its 1840 cases compare integer BF16 bits against fp8small
and against CUTLASS, with 0 mismatches. `tests/fp8c_stats.py` reports the
escape statistics of the candidate windows.

Timings are in microseconds per call, at commit 97862b7, from a CUDA graph.
"cold" cycles 12 layers. "head" adds the engine's l2pf prefetch of the buffer
head before each call: 12 MB for in_proj, 5 MB for out_proj, the same prefetch
time in both columns. "warm" repeats one layer. The default rings are in_proj 3
and out_proj 4.

| shape | M | cold fp8small | cold fp8c | head fp8small | head fp8c | warm fp8small | warm fp8c |
|---|---|---|---|---|---|---|---|
| in_proj | 1 | 118.9 | 109.0 (+9.1%) | 117.5 | 107.5 (+9.3%) | 72.7 | 56.8 (+28.0%) |
| in_proj | 4 | 120.3 | 109.8 (+9.5%) | 120.4 | 107.6 (+11.9%) | 71.9 | 58.9 (+22.1%) |
| in_proj | 5 | 119.9 | 109.1 (+9.9%) | 117.7 | 110.9 (+6.1%) | 72.8 | 58.4 (+24.7%) |
| in_proj | 8 | 120.0 | 109.3 (+9.8%) | 118.6 | 111.6 (+6.3%) | 73.1 | 57.5 (+27.2%) |
| out_proj | 1 | 48.0 | 45.3 (+5.9%) | 47.3 | 45.0 (+5.1%) | 9.0 | 19.0 (-52.7%) |
| out_proj | 4 | 48.0 | 45.4 (+5.6%) | 47.4 | 45.0 (+5.2%) | 9.0 | 19.1 (-53.2%) |
| out_proj | 5 | 48.0 | 45.3 (+5.8%) | 47.2 | 45.1 (+4.7%) | 9.1 | 19.3 (-52.9%) |
| out_proj | 8 | 48.2 | 45.9 (+5.0%) | 47.2 | 45.1 (+4.8%) | 10.4 | 19.7 (-47.1%) |

The in_proj warm column benefits from the smaller buffer: 24.8 MB nearly fits
in the 24 MB L2, while fp8small's 27.7 MB does not. Once out_proj is fully
L2-resident, decode is issue-bound and costs more than the bytes it saves.
These are isolated kernel timings, not full-model or official results.

# Exact FP8C mixed-six/seven packet, isolated c41 experiment

Effort: ultra. The actual native turn context identifies gpt-6.1-sol and Codex.
This is an unexecuted static prototype. No local candidate import, syntax or AST
execution, compilation, setup, test, profile, benchmark or preflight was used.
No submission, cancellation or official state mutation was made by this author.
Only an official remote run may execute the new library, private admission and
the complete model. A frozen source commit, if later authorized, is a source
snapshot and is not proof that this feature activates or improves the score.

## Baseline and scope

The complete baseline is c41fc629-8c06-4bf2-8cb8-cad0edea9875, officially scored
2.77776232031352. Its source is cdea0e5bb85ae36999f23c1f9de15de5901b8e45,
with complete Git tree 3f3a3fa2636eb7c71bf871462134b3b18fbc5259. The local
baseline snapshot is 4726d88aef2bfa8e04c110015568a9c68a840b30. All its existing
features remain present. This prototype adds no retired whole bundle and is
independent of the head-pair experiment that was already validating when this
work started. A newer highest result must be checked by the submitting agent
before any future upload; this document makes no live leaderboard assertion.

This is explicitly a new selection strategy in the old six-bit FP8C family,
not a newly invented lossless codec. The retired whole-layer experiment
0fde1e7f-cced-4bea-bfe7-cc2eece540be scored 2.74656547661021, approximately
0.948622% below its de4975c1-0182-4fe4-8a7f-70ac7e5f8af1 base. That result is
negative whole evidence. Its production activation and isolated component
causality were not observed. This candidate keeps c41, not that whole, as its
baseline and reuses only the disclosed exact code math and relevant admission
ideas. There is no statement that a new selection strategy repairs the old
performance result.

## Mechanism and real source traffic

The existing FP8C consumer assigns one 128-thread CTA to each original group of
32 output columns. Four warps reconstruct eight original FP8 words per K128
chunk. The current seven-bit main region is 3,584 bytes per tile and chunk;
the old six-bit decoder needs 3,072. The six-bit version selects a four-exponent
window per column, with an explicit raw-byte escape stream, whereas seven bits
select eight exponents. Exact FP8 bytes, including unusual bit patterns, are
reconstructed before the original native E4M3 MMA. No declared FP8 format,
quantization scale, checkpoint weight or numerical precision changes.

Old whole-layer six-bit admission rejected an entire layer when any warp/chunk
needed more than 124 escapes. This candidate chooses independently at the
original N32 tile boundary. A six-bit tile is eligible only when every one of
its warp/chunk records satisfies that same window bound, its offsets fit, and
its source requests improve after descriptor overhead. Unqualified tiles keep
the original seven-bit header, main bytes and actual escape bytes. Production
requires both C6 > 0 and C7 > 0: all-six and all-seven outcomes retain the old
whole layer. This prevents presenting the retired all-six experiment as new.

The packet is one dedicated contiguous allocation. A 256-byte-aligned prefix
contains one uint32 descriptor per original tile. Each descriptor contains an
aligned payload offset and one mode bit; all other low bits remain zero. A
warp's lane0 reads the original tile descriptor and broadcasts it. Logical tile
number still determines the output columns and decoded output location; the
physical payload offset does not alter indexing. Each payload has the original
header and either original main layout, followed by bounded escape data and
zero padding. Offsets are narrowed only after the selected tile is known safe.
The final escape load may read a four-byte word beyond the final valid byte;
the exception area therefore has capacity max(round_up(total,16),16), and the
complete payload end is rounded to256. Cursors must remain below2^31.

The source-request estimate counts main bytes, the original 16*C header-offset
bytes, 256 parameter bytes, and each original load_x four-byte request. For an
escape record starting at offset q with length l, that term is
4*ceil(((q mod 4)+l)/4). Even a zero-length record can issue a four-byte load
when its start is unaligned. The candidate adds 16 logical descriptor bytes
per CTA, one uint32 per warp. It does not claim savings from deleting fixed
padding that the consumer never loaded. A layer needs at least 1% less total
source-request accounting and a smaller stored packet before any executable
admission is attempted. These are source-level request bounds, not measured
transactions, cache misses, DRAM bytes, or timings.

If every main tile could use six bits with no extra escapes, the two matrices
could omit at most (10304*2688 + 2688*4096)/8 = 4,838,400 main bytes per Mamba
layer, or 111,283,200 bytes for 23 layers. The candidate intentionally rejects
the all-six case, and actual escape costs and C6 coverage lower that bound.
The true compression distribution is unknown. L2 reuse and physical backing
compression can make a logical reduction yield no DRAM or latency reduction.

## Production ownership and shared selection

Preparation occurs after target/drafter packing and CMP2, before compile and
graph capture. It uses only the exact supported target Cutlass FP8C objects,
the two actual projection shapes, static scalar FP32 scales, and a single
sm121 BF16 Nemotron engine. TP/PP/DP/context-parallel sizes must all be one;
expert parallel, DBO/ubatching, EPLB, LoRA, weight transfer and weight offload
are excluded. Unsupported objects keep their complete original code path.

The original registered seven-bit buffer is retained and unchanged. The
actual logical B parameter also remains, as required by the original prefill
and fallback. The old seven-bit decoder's output is inverted by views and
permutation and compared bytewise with actual B; another full B clone is not
created. A new packet is compared against that decoded original using its
own native decoder. After the probes both packets are decoded and rechecked.

The model layer owns the admitted packet as a nonpersistent registered buffer
and owns its entry. Global indexing and tensor/owner references are weak. Every
selection checks object identity, pointer/storage/layout/dtype/device/mutation
version of actual owners, the actual kernel and layer links, configuration,
reference/native library identities, original ring setting, admitted PDL mode,
and unchanged prefetch settings. Inference tensors without a mutation-version
proof are refused. Owner changes or reloads are not supported as a concurrent
operation; metadata changes make subsequent selections retain original FP8C.
This is an immutable, serial, fixed-engine experiment, not a general model
hot-reload or concurrency implementation.

The identical selector is used by prefetch, paced prefetch and fused multi-range
prefetch, and by the consumer inside every existing fp8c_scaled_mm guard. Thus
the runner may retain its original first-target tensor and still prefetch the
same admitted packet that GEMV consumes. The selector allocates or qualifies
nothing at request time. It introduces no consumer-only M/input/scale-pointer
selection rule. The original M1..8, alignment, FP8, device and scalar-scale
guards remain. Unsupported paths keep original FP8C, FP8WS or CUTLASS behavior.

Existing prefetch range counts, byte budgets, side-stream fork/join and router
reservation remain. A shorter packet can make the unchanged budget reach more
useful tiles, but this is not credited as measured reuse. The new allocation
has offset zero because existing l2pf intentionally reads untyped storage,
not a view's element slice. The packet is created after CMP2 and is not assumed
to receive CMP2 physical compression. Both original and mixed encoded weights
remain resident. Extra memory, allocator pressure and changed cache competition
can outweigh the logical request reduction, including across other layers.

## Native arithmetic, graph and PDL

Both decoder helper bodies are copied exactly from their attributed sources.
The same FP8 activation loads and m16n8k32 MMA are used in the same ascending
K order with the same original ring depth (default in_proj 3, out_proj 4; an
existing 3/4/6 setting is retained). The existing out_proj ordering decodes
before refilling activation; the in_proj ordering refills at its original
earlier point. The output is still scaleB multiplication, scaleA multiplication
and round-to-nearest BF16. Eight reconstructed words are not a new ring-depth
claim. There is no altered top-k, sampler, model graph configuration or state.

For PDL, immutable packet loads can precede the original wait. Current A and
the FP32 scales are consumed after that wait, and the original dependent-launch
point is retained. A CTA's mode is uniform across all its original warps. The
normal variant and every enabled PDL variant must be admitted; a library built
without PDL cannot replace a currently PDL-enabled FP8C layer. A production
native error propagates; no fallback is attempted after an operation might
have written output. This prevents a failed capture or partial launch from
silently changing production state and then retrying the old implementation.

Static same-order source is not a device bit-equality proof. The extra uniform
branch and simultaneously compiled decoder paths can alter registers, spills,
occupancy, instruction scheduling and compiler lowering. Actual compiled
attributes and complete bytewise remote comparisons are required. There is
no local compilation or statement that the native TU successfully builds.

## Remote-only private admission

No request tokens, benchmark prompts, KV, conv state or SSM state are used by
admission. Every real prompt prefill still executes as before. Synthetic raw
weights exercise all 256 FP8 bytes, zero/unaligned escape records, counts around
word boundaries through124, tile ends, and all-six/all-seven/mixed decoding.
Those forced modes are confined to private decoder checks; production uses the
strict mixed-only selection. Prefix and suffix guards surround decoded output.

Every actual M1..8 is compared to the original direct native ABI using the
actual weights, finite activation patterns, signed zeros and separate all-byte
patterns. Current actual scales, zeros, negative zero, infinities, NaNs,
subnormal and very large FP32 scales are checked. Output comparison is raw BF16
bytes; A and output bounds are checked. The reference bypasses the old Python
eager PDL retry, so admission cannot mutate the baseline's global PDL mode.

Independent producer graphs have separate A and scalar-scale destinations for
reference and candidate. Before each replay both destinations are poisoned;
the candidate graph runs first. The native producer signals dependents before
writing current A and scales and includes a delay. There is no prefetch join
in this test to hide a missing dependent wait. Changed finite inputs/scales
and guard values are checked for each graph, along with actual output bytes.
Finite expected results are required in this producer test, so a blanket NaN
cannot mask stale reads. No source-level test can promise a race will manifest;
the unchanged source wait is also reviewed directly.

Private whole-chain timing separately captures original or selected packet
prefetch, unchanged range budgets, actual next-router range where present,
side-stream join and complete GEMV. Warm and64MiB pressure samples use alternating
order for M1/5/6/8. Every sampled condition must improve, with a median and
aggregate ratio below0.99. Resource admission also rejects local memory and
requires a resident block within a conservative register limit. These tests
do not replay production conv/MoE work inside the overlap window and do not
establish its real hiding, total latency or hit distribution. Whole official
correctness and score are still required and may reject the candidate.

All graph/tensor/native-pointer owners remain in the same admission frame.
Nested finally blocks drain both private main and side streams before any
owner is released, including during exceptions. A healthy private refusal
retains original FP8C. An unhealthy drain propagates and stops loading; it does
not retry a possibly written operation. The new packet and global entry are
published only after completed admission and drain. An optional build failure
is confined to a separate TU/package and cannot delete the original library.

## Evidence limits and attribution

At the time of authoring there is no actual C6/C7 distribution, accepted owner
count, packet byte ratio, kernel activation, native resource result, output
comparison, private timing or official whole result for this candidate. A zero
admission count is a valid full-fallback outcome and a whole score alone cannot
show activation. Public logs may not expose private diagnostics. No other
author's self-reported device measurements are treated as our observations.

Material sources, with complete submission identities:

- c41fc629-8c06-4bf2-8cb8-cad0edea9875: complete verified baseline and all its
  carried source and provenance, retained in full.
- fc04335b-c894-41f9-b73b-aa9b5aac1fb5: original exact seven-bit FP8C native
  codec, original pack contract and bounded exception decoder.
- 0fde1e7f-cced-4bea-bfe7-cc2eece540be: exact six-bit native/helper and encoder
  math, plus negative whole-family evidence described above.
- 4929b3f5-29ce-4a7c-93b5-f2f441fd4d29: original out_proj decode/refill ordering
  carried by c41 and preserved by the new consumer.
- 297e3412-dc4a-45ae-a0f1-b6c3afd76c1b: original prefetch-window mechanism
  whose budgets and ownership remain, with shared packet selection added.
- 110cff4b-d86d-4fe9-ae30-f6ce550d005f: actual-owner, private qualification and
  immutable startup/capture safety patterns; no A16 runtime feature imported.
  Whole score2.74300427046762 was approximately1.2513% below c41 and retired.
- 77f74985-a845-4963-bf29-b860e8daa7ef: dynamic private graph and owner-retirement
  validation ideas; no suffix arithmetic imported. Whole2.75184607067041 was
  approximately0.933% below c41 and retired.
- 8e0a3de6-17c7-4667-8804-ee17507eb41a: single exact encoded-packet
  prefetch/consumer selection and mixed/fallback lifecycle ideas; no W12 router
  code imported. Whole2.76663609402143 was approximately0.40055% below c41 and
  retired. These negative wholes do not isolate component effects.

All ancestor entries in the baseline VENDOR remain. The appended entry and
the dependency metadata identify actual material reuse rather than presenting
the retired experiments as a winning baseline. No source from unrelated
pending bundles is included. A submission should be considered an unverified
whole experiment until official remote correctness and scoring finish.

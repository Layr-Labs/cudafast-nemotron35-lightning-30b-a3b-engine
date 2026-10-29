# Target head partial-max prototype, retaining complete logits

This is an unvalidated source experiment on the complete verified CUDA tree of
`c41fc629-8c06-4bf2-8cb8-cad0edea9875`. Its official source is
`cdea0e5bb85ae36999f23c1f9de15de5901b8e45`, tree
`3f3a3fa2636eb7c71bf871462134b3b18fbc5259`; the local equivalent parent is
`4726d88aef2bfa8e04c110015568a9c68a840b30`. The parent's official complete score
is 2.77776232031352. No result for this combination is claimed. It includes
neither the pending depth-five plan-bank change nor any retired optimization.

No local candidate execution, import, syntax/AST check, compilation, setup,
preflight, tests, profiling or benchmark was performed. Only source editing,
text reads and Git metadata/diff inspection were used. This is a local draft
snapshot and has not been submitted. Independent source review and all runtime
validation remain outstanding.

## Scope and expected traffic

The target's eager small-M head currently writes an M x 131072 BF16 logits
matrix. The strict greedy speculative sampler scans that matrix, finds each
row's vocabulary winner, and performs rejection, bonus selection, verify-row
reset and accepted-count publication. The full logits remain useful for all
noneligible sampling paths.

The experiment retains every full-logits store and the original head callback.
It adds a partial maximum to the head epilogue, then permits the strict sampler
to read those partials rather than the whole vocabulary matrix. The target
head still streams the same weights, uses the same Marlin partition and MMA
chains, and rounds/stores the same requested BF16 outputs in source. There is
no weight-traffic reduction claim.

Each 64-column unit emits one 8-byte record per token row. At M6, the baseline
logits scan requests 6 x 131072 x 2 = 1572864 bytes. The 6 x 2048 x 8-byte
partial array adds 98304 bytes of stores and 98304 bytes of reads. Thus the
maximum removed logical data traffic is 1376256 bytes, or 1.3125 MiB, per
eligible verify. Complete logits stores are unchanged. This is not a DRAM
measurement: the original scan may hit L2. Added allocations, CPU guards,
epilogue instructions, registers or a different compiler schedule can make the
complete chain slower.

## Producer ownership and numerical boundary

The existing MarlinX GEMM assigns one CTA of four warps to one `unit` of 64
output columns. Warp zero owns the final reduced R values and writes C. With
`unit=2*col+h`, its lane `l` writes token pair `2*(l&3)` and `2*(l&3)+1`.
For j=0..3, it writes columns `64*unit+16*j+(l>>2)` and that column plus eight.
For each token row these four pairs from each of eight lanes cover exactly
the 64 distinct vocabulary IDs of that unit.

The new compile-time `Partials` branch consumes the SAME v0..v3 values used in
the existing C stores, after R times the global scale and RN-to-BF16. It never
compares unrounded accumulators or substitutes a different dot product. Each
lane reduces its eight values separately for the two token rows. XOR shuffles
at offsets 4, 8 and 16 keep `lane&3` unchanged, so they combine exactly the
eight lanes belonging to each token pair. Only lanes with `lane>>2==0` publish
the pair. Their stores are masked independently by `row<M`.

The output address is `row*(N/64)+unit`. A different CTA has a different unit;
a different writer lane has a different token pair. There is precisely one
writer for every one of M x 2048 records. No grid-wide atomic max, global
initialization, inter-CTA numerical reduction or borrowed static output is
introduced. The packed uint64 contains the FP32 bits of the rounded BF16
value in its low 32 bits and the vocabulary ID in the high 32 bits.

The comparator has the original vsample order: NaN wins over numeric values;
among NaNs the smaller token ID wins; otherwise the larger value wins and
equal values choose the smaller ID. This treats +0 and -0 as equal. Infinity
and all-negative-infinity rows use the same order. A `(-inf,INT_MAX)`
initializer loses to every real vocabulary entry, including -inf. The new
consumer uses the original comparator too. The parent Marlin close-segment
order, scale multiplication, BF16 stores, weight formats and K traversal are
retained in source. These facts do not establish compiled device bit equality.

The new producer ABI is restricted to N131072/K2688/M2..6, aligned inputs,
ldc=N, a nonnull 8-byte-aligned partial output, the existing depth4/depth8
loader choices and no ReLU2 or zero-skip head specialization. The old exports
and default nonpartial instantiations remain. The existing unit-major scale
copy is passed unchanged. The producer follows the parent's current PDL
enablement. Its early dependent-launch signal still precedes the epilogue;
the consumer must wait for the complete producer before reading its records.

## Consumer and unchanged rejection semantics

The optional vsample input ABI reads 2048 uint64 records per row. Eight
256-thread CTAs cover a row, with one record per thread. It reuses the
original warp comparator, shared partial publication, thread fences and
last-CTA counter protocol. Its second phase and all rejection/bonus/count/reset
statements are shared with the original full-logits kernel specialization.
The original per-(device,stream) scratch has capacity for 64 chunks per row;
the new eight chunks fit. Ordered launches on that stream leave the same
counter reset discipline intact.

The Python wrapper first applies all existing vsample metadata, dtype,
alignment, row-count and no-capture guards. The new input further requires
owning contiguous int64 data of shape [M,2048] on the logits device, M2..6,
and full vocabulary N131072. An absent optional ABI or incompatible partial
layout falls back to the existing complete-logits consumer. An accepted
native launch that returns an error is raised visibly, not silently retried
with another head calculation or published as a partial result.

## Original callback, guards and single-frame ticket

The runner retains `self.model.compute_logits(sample_hidden_states)` and the
original hidden gather. A thread-local, one-shot collector scopes just that
call. MarlinX collects only when its `maybe_apply` sees the exact expected
lm_head object and the new head shape; no other layer can redirect the scope.
A nested collector disables collection. `finally` restores the previous
thread-local value on success or exception. Duplicate head calls refuse the
ticket. No process-global tensor registry is created.

The producer guard requires one request, native MTP, TP/PP/DP size one, no
microbatching, no LoRA, GPU BF16 hidden rows of the exact shape and no active
capture. The raw owner must be the stock NemotronHForCausalLM, with stock
LogitsProcessor and ParallelLMHead, one of the two actual ModelOpt NVFP4
linear methods, an actual MarlinNvFp4LinearKernel and a prepared MarlinX
head. The original bound compute_logits method is checked. Logits must not
be supplied as input; soft-cap is absent, scale is one, vocabulary is 131072,
and no head dtype change is applied. An early call to the original strict
greedy-argmax guard avoids producing unused partials for known ineligible
sampling. It does not replace the later guard.

After the callback, the collector checks the final logits device, dtype,
pointer, shape, stride and storage offset against its owning C allocation.
This permits the original no-op vocabulary slice but refuses a different
returned tensor layout. The ticket then holds the exact final logits object,
its immutable layout description, the per-call partial allocation and the
producer stream. ExecuteModelState carries this ticket once. It is not
indexed by request ID and cannot retrieve a result from a prior request.

At sample entry, the ticket is handled BEFORE grammar. Any grammar output
discards the partial route. The original CPU output-token update,
spec-token bookkeeping and draft-probability retrieval in `_sample` still run.
Only then does the existing RejectionSampler's current `_can_use_greedy_argmax`
guard run. A false guard keeps the original sampler and complete logits.
There is no cached sampling-eligibility flag. The ticket's `take` is single
use and rechecks exact final-logits identity/layout, stream and capture state.
All original verify-prefix rejection, accepted count and output-copy handling
remain. The token output is a new ordinary allocation, as in the parent.

## Streams, graph replay, abort and reload

The new path only produces outside capture. A target FULL graph can still
produce the hidden states, but the original gather and eager head run after
it. Fresh-prefill graph logits bypass this helper. The partial experiment
does not capture another graph, alter a pool or retain a hidden source view.
Each eligible head allocates new owning C and partial storage for this frame.
The next model graph replay or head call cannot overwrite them as static
graph output. The ticket keeps them alive while it may be consumed.

For the ordinary same-stream path, producer and sampler enqueue in order;
PDL consumers execute their dependency wait before loads. Torch's stream-
ordered allocator protects released temporary storage from subsequent reuse
on that same stream. If sampling changes streams, the ticket records an event
on its producer stream, makes the current stream wait and records the owning
logits on the current stream before grammar or baseline sampling. Partial
consumption is refused on that path, so no cross-stream partial lease exists.
The event is created only for this exceptional fallback, not every frame.

Aborted frames drop their own tensor references; no output can be looked up
by a future frame. Sentinel clearing of ExecuteModelState also drops the
ticket. The collector holds no state beyond its scope. The existing model
shutdown synchronization and output lifetimes remain applicable, and no new
private graph or permanent model/weight owner is retained. Weight reloads
cannot cause the ticket to recompute from a different checkpoint: it only
represents the logits already calculated for that frame. Future calls check
the current stock backend and prepared head again. CUDA errors remain visible.

## Official-only evidence still needed

Static review must independently recheck writer coverage, shuffle row
ownership, both C ABI parameter lists, model guard reachability and the
sampler's side effects. No local compilation or syntax checker is allowed.
The official worker must establish that the new CUDA variants compile, that
all complete logits match the parent at the BF16 boundary, that NaN/tie/zero
behavior and changed-input calls are correct, and that frame/grammar/stream
fallbacks preserve results. Complete timed tokens and all official gates are
the final correctness authority.

Performance must compare complete original head+vsample against complete
partial-producing head+partial consumer. Isolated max timing omits the added
head epilogue and allocations. The first invocation's lazy kernel loading and
all guards are ordinary work; nothing assumes they happened before timing.
There is no runtime speed chooser in this first prototype. A complete official
regression above the user's 0.1% threshold excludes this whole candidate.

The benchmark publishes only sealed score data. Contract section3.6 withholds
engine stderr and engine-chosen log values; benchmark.yml uploads only the
sealed score.json. Private counters or printf therefore cannot establish a
public scored-path hit, and no such evidence is claimed. Requesting a trusted
official diagnostic export is separate from changing this candidate. The
source guards establish static reachability, not observed activation.

## Material sources

- `c41fc629-8c06-4bf2-8cb8-cad0edea9875`: complete verified parent, retained.
- `adab284a-bf1d-4ce2-8e90-6720fa0f5435`: original MarlinX partition, GEMM,
  scale/rounding and Python dispatch used by the new producer specialization.
- `b86d9048-2e9d-4a70-9373-abe8d09b77d4`: existing vsample comparator,
  reduction, rejection, accepted-count and verify-row-reset body.

The negative `cf78769f-e926-475b-b9b5-4d202453b663` redux experiment was read
as comparison evidence only. Its alternate ordered-integer comparator,
qualification and warmup changes are not included. This prototype changes
the producer/consumer data boundary while retaining the parent's comparator.

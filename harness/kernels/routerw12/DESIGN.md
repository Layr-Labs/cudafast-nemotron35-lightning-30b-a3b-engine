# Optional exact router-weight packet, design before implementation

This is an independent source-only experiment on the complete verified c41 tree
`3f3a3fa2636eb7c71bf871462134b3b18fbc5259`, locally represented by
`4726d88aef2bfa8e04c110015568a9c68a840b30`. The official baseline submission is
`c41fc629-8c06-4bf2-8cb8-cad0edea9875`, score2.77776232031352. The suffix whole
`77f74985-a845-4963-bf29-b860e8daa7ef` scored2.75184607067041 and is excluded.
No suffix arithmetic or A16 input conversion is selected.

All executable admission below is proposed for the official remote engine only.
No actual weight statistics, compression ratio, numerical pass, path engagement,
compiler/resource success, latency or score is claimed. No candidate has been
executed, imported, parsed, compiled, tested or benchmarked locally.

## Actual source boundaries

The c41 early-router target uses the original external BF16-to-FP32 conversion,
then calls the original smallk opaque body on FP32 input. Its eligible shapes
include M5/M6,N128,K2688. Original smallk loads its actual BF16 shadow only after
the original PDL wait, widens each uint16 with bits<<16, performs the per-lane
ascending explicit FP32 FMA chain, folds K256 tiles ascending, and folds sixteen
lanes ascending. The new variant must retain all those arithmetic and control
sites, including vrows' source-row substitution. It does not omit invalid rows.

The preceding Mamba or attention operation already calls l2pf.gate_weight, which
selects that same W16 shadow and reserves at most a quarter of the prefetch
window. The projection receives the rest of the window. Therefore replacing
only the router loader would fetch two different representations; returning a
smaller reservation would refill the saved traffic with projection prefetch.
Both would invalidate a claim that the proposed source deletes the intended
prefetch requests.

## One immutable mixed packet

Only an actual gate accepted by the existing exact gate_w16 policy is eligible.
Its original FP32 parameter and its existing BF16 shadow stay available and are
not cloned into persistent new weight owners. The optional representation is
one contiguous CUDA uint8 tensor with a 512-byte header and row payloads. Header
storage is independently allocated, with storage_offset zero and storage size
equal to the packet length. The native consumer and prefetch storage base are
therefore the same address; a contiguous slice is not sufficient. Each header
entry e is an unsigned32 byte offset, with low bit1 meaning coded. All offsets
are multiples of128 before the low-bit tag. Each expert row is either:

- raw:5376 bytes, containing its original2688 BF16 words, or
- coded:4096 bytes:2688 sign/mantissa bytes,1344 paired exponent nibbles,
  21 base bytes, and43 zero-padding bytes.

For each contiguous128-value block, exponent-zero values receive code0 and keep
their sign/mantissa. Other exponents must lie in [base,base+14], where base is
the minimum nonzero exponent; an all-zero-exponent block uses base1. Codes1..15
reconstruct exponent=base+code-1. The original uint16 is reconstructed with
integer shifts/OR, then the original uint16-to-FP32 bit widening is used. This
retains signed zero, subnormals, infinity and NaN payload bits whenever the
actual row fits. There is no quantization, dropped outlier or arithmetic
conversion in the representation.

A row is coded only if all21 blocks fit. Otherwise the entire row is raw. This
avoids per-block escape pointers and speculative unused payload planes. Actual
fit bits are computed from the actual W16 words at remote startup; they are not
predicted from dtype, distribution assumptions or another tensor's statistics.
Let C be the actual number of coded rows. Total bytes are688640-1280*C. C0 is
refused before the final packet allocation. C1 gives a smaller packet, but no
speed is inferred; the later complete-path timing gate can refuse it.

The packet is a second optional weight view, so resident memory increases while
the baseline parameter and W16 fallback remain. The design intentionally avoids
another full FP32 weight clone. Temporary statistics, decode guards and the
final packet are bounded to one small router at a time; temporary data and
private graphs are drained before their owners are released.

## Common producer/consumer selection and byte budget

Preparation runs at the existing model-level post-load hook after all per-layer
quant/attention/HPC finalization and before normal model compilation/capture.
Only fixed immutable single-worker target early-router layers qualify: native
Nemotron, no latent/sequence-parallel router, no bias, TP/PP/DP/CP one, uni
executor, one sequence, no ubatching/EPLB/LoRA/offload/weight or KV transfer.
The optional native library and the original smallk W16/vrows support must be
present. First version is R1,SM121,FP32 input,M5/M6 only. The original parameter,
exact shadow, device, layout, pointer, identity and available version counters,
plus relevant immutable smallk/l2pf configuration, are part of admission.
Inference tensors without a version counter must be safely refused rather
than reading _version. This may refuse every real layer; actual engagement is
unknown.

A positive entry owns only its immutable packet and decisions, and weakly
identifies the original parameter/shadow. The gate/model owns the entry through
its graph lifetime; a global lookup must not create a new strong model-weight
or graph-pool owner. No request data, output or execution graph is cached for
production. Fixed-engine teardown ordering is inherited; hot reload or general
concurrent mutation is outside this experiment.

l2pf.gate_weight receives the current row count from its two actual Mamba and
attention callers. It and smallk's production dispatcher look up the same
already-published entry, with the same M5/M6 configuration signature. A missing,
rejected, changed or unsupported entry uses the original shadow/FP32 dispatch.
The actual gated target input, row-count buffer and stream are checked again by
the consumer. A missing count keeps the original NULL==all-rows rule and uses
the same packet; this codec does not require a prefix count. This avoids a
producer/consumer mismatch on the original first allocation of the count.
No count allocation or lifetime is changed. No preparation or GPU-to-CPU data
read occurs in production.

For an admitted entry, the producer returns the packet tensor while preserving
the original W16 reservation. Existing prefetch_many clamps the request to the
packet's smaller complete storage; its projection budget remains identical to
the baseline. Qualification requires the current default windows to cover the
whole original688128-byte shadow within their quarter budget. A reduced custom
window is outside the optional policy. The immutable packet is complete before
the entry is published, and prefetch and GEMV capture the same tensor address.
The existing side-stream fork/join is retained. There is no new production
stream, event, scratch, producer ticket, allocation or retry.

## Decoder and lane ownership

The candidate uses the original R1 grid128, block176. For each original
sixteen-lane subgroup, K256 contains two K128 exponent blocks. Every lane reads
its own SM byte. Only even lanes load a paired EX byte, then share that byte
with their odd neighbor through width16 shuffles. Each subgroup leader loads a
base once per K128 block and broadcasts it. All participating lanes execute
the same shuffle calls, using the actual active mask; the final physical warp
contains only16 threads. Never use a hardcoded full-warp mask there.

The final j10 tile has only128 values/8 terms per lane, so it reads only block20
and never block21. The raw-row branch executes the original uint16 load and
bit widening. Row mode is immutable and uniform for each expert CTA, so no
barrier is conditional on a row's representation. All original output writers,
partial stores, two CTA barriers, source-row rule and PDL instructions remain.
Changing the loader may still change remote register allocation, spills,
instruction issue or compiler behavior; it is not a device-bit proof.

## Required private admission and failure ownership

All native calls, package construction and measurements run solely in the
official remote engine. Prepare must complete all checks before publishing a
positive entry. It must establish actual fit count, smaller real storage, full
uint16 decode equality through the same native loader used by the GEMV, and
unchanged original weight/shadow/package contents with explicit canaries.

Reference is the original smallk W16 router, not the new wrapper. Compare every
FP32 output bit, including duplicated suffix rows, for each eligible M and null
count plus INT32_MIN,-1,0,1..M,M+1,VROWS_ALL,INT32_MAX. Inputs include finite
scales, cancellations, signed zeros, subnormals, finite extremes and nonfinite
raw-word patterns. Guards and readonly owners must retain their bits. Private
old/new graphs copy changing inputs/counts into independent fixed buffers,
include the actual same-packet prefetch fork/join, and replay alternate count
orders. The package and all graph/tensor owners must remain in the same frame
until an inner finally drains all participating streams, including exceptions.

Actual old-prefetch+original-router and new-packet-prefetch+new-router must be
compared under alternating warm and cache-pressure conditions. The projection
reservation is unchanged. A pure codec or GEMV microbenchmark is insufficient
to approve the prefetch integration. Passing private timing does not establish
performance with the whole concurrent shared/routed expert chain. The official
whole golden and final score remain authoritative.

Missing optional library/symbol/resource support, no coded rows, bit mismatch or
speed refusal leaves the original kernel and original prefetch selected.
Native launch/capture/replay/synchronization errors are visible; do not swallow
them and retry after any production write. A separate optional build package
must not make a failed new compilation remove the original smallk library.

## Source-level bound and evidence limits

Original W16 shadow bytes are688128 per gate and15826944 for23 layers. If every
row in every layer fits, a mixed packet including its512-byte header occupies
524800 bytes per gate and12070400 across23 layers. The maximum package/prefetch
range reduction is3756544 bytes, about3.58MiB per verify. This corrects the
earlier simpler4096-row+128-mode-table estimate, because a mixed compact packet
uses a512-byte offset table. Actual byte requests, cache sectors and physical
DRAM traffic differ; the source has no actual hardware measurement.

Existing exact-router tests only record W16 gate eligibility and compare input
patterns; wide-exponent patterns describe activations, not gate weights.
Existing MTP pack logs describe MTP tensors. Neither establishes a router fit
rate. Engine stderr is sealed and official score artifacts do not expose this
private statistic. No histogram should be encoded in score/error/public output.
A successful whole run may have declined the optional path entirely.

## Material sources and explicit exclusions

- c41 complete baseline: c41fc629-8c06-4bf2-8cb8-cad0edea9875.
- Original smallk FP32 chains/schedule:100549d4-51c4-4f56-b3ca-2604731ab8a4.
- Exact W16 shadow:0852152c-9025-499d-83bd-684cf825232e.
- Original lossless exponent-window codec:3d51cab6-7079-4fb2-98e4-f2a238077897.
- Grouped exponent-load method:c0ce642d-0f7b-4c1f-a168-f8edb153aa4e and
  59b3dbb4-6919-44ca-85f1-fcbbc0f1d975. The proposed CUDA width16 mapping is
  newly adapted to the router; this is not a claim that the earlier code did it.
- Next-router prefetch:297e3412-dc4a-45ae-a0f1-b6c3afd76c1b.
- Existing M5/M6 reference:f38ce1ad-87fa-4cfb-8a0c-ac46587810f7 and
  eb48e2f0-50e6-4c39-a1f3-0ac305d17c5b.
- Private post-load/lifetime/refusal design only:
  110cff4b-d86d-4fe9-ae30-f6ce550d005f and
  77f74985-a845-4963-bf29-b860e8daa7ef. Both whole scores are lower than c41;
  neither whole source is a baseline. A16 and suffix arithmetic are excluded.

The old X16 whole failures4ef72848-19b4-492e-a948-e9d44a94bef0 and
528c6b75-628b-427b-9dab-07f2e7a9a3c1 do not diagnose this design, and their
input/cast changes are not imported. The retired raw-pipeline whole
cdf8aecf-594a-48ec-bc02-d03a379ce1c2 is not recycled. Public-note text searches
did not find the same router codec; that is not an exhaustive all-source proof.

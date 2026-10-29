# Exact router weight packet shared by the prefetch and GEMV

Effort: ultra. Implementation used GPT 6.1 Sol through Codex, confirmed from
the actual native turn context. This is an unvalidated source-only experiment.
No candidate import, syntax/AST parsing, execution, setup, compilation, AOT,
test, benchmark, profiling, preflight or model startup was performed locally.
All executable admission described below is authored only for the official
remote engine. No compression ratio, path engagement, numerical pass, timing or
score is claimed.

## Complete baseline and isolated change

The verified baseline is `c41fc629-8c06-4bf2-8cb8-cad0edea9875`, official score
2.77776232031352, source `cdea0e5bb85ae36999f23c1f9de15de5901b8e45`, complete
tree `3f3a3fa2636eb7c71bf871462134b3b18fbc5259`. The local reconstruction
`4726d88aef2bfa8e04c110015568a9c68a840b30` has that exact tree. The complete
model, MTP, graph, sampler, prefill, state/cache formats and protocol remain.
The target's original BF16-to-FP32 conversion stays outside the MoE opaque
boundary at its original source location. This is not an A16 input or a suffix
row-skipping experiment. Every original router row is still computed.

The only optional production mechanism is an input-independent, bit-preserving
router-weight repack. It replaces the existing exact BF16 shadow stream by a
smaller immutable mixed raw/coded packet when actual-weight admission succeeds.
Both the existing previous-layer prefetch and the original-order FP32 router
read that same packet. This is a genuine change to their source byte ranges,
not a different claimed FLOP count, routing rule or reuse across requests.

There is no evidence yet that any actual gate row fits this codec or that any
gate passes private timing. An official whole pass can still have used the
baseline throughout. Engine stderr is sealed and the public score artifact
does not expose private weight statistics or optional kernel engagement.

## The existing weight and prefetch path

Each target gate has shape128x2688. The current smallk policy first checks that
its finalized FP32 parameter equals the FP32 widening of a BF16 shadow at every
raw bit. A qualifying gate's ordinary router reads688128 bytes of BF16 words;
otherwise it retains the original FP32 path. The new code re-proves the actual
finalized parameter against the actual shadow before encoding, because the
baseline shadow cache itself only checks pointer/shape.

The original router already shares each expert's weights across all M rows in
its CTA. This experiment does not invent another cross-token read saving. Its
first optional scope is R1,M5/M6,N128,K2688 on SM121. It changes only the
representation read for that existing stream.

The preceding eligible Mamba or attention op already prefetches this gate's
W16 shadow through l2pf.gate_weight, beside its following projection weights.
It reserves at most a quarter of the existing window, and gives the rest to
the projection. Merely changing the GEMV loader would read the old shadow as
well as the new planes. Merely reducing the reservation would refill the
saved request range with more projection bytes. Neither is the proposed delta.

The modified gate_weight receives the current row count from its two original
callers, looks up the same admitted packet as the router, and preserves the
old W16 reservation. Existing prefetch_many clamps the actual transfer to the
packet's smaller storage length. The projection receives exactly its original
budget. The ordinary side stream, fork/join, native prefetch3 kernel and all
projection ranges remain. No new production prefetch launch is added.

## Exact mixed representation

The packet is one dedicated contiguous CUDA uint8 allocation with zero storage
offset and storage size equal to its length. Native GEMV uses data_ptr and
prefetch uses the storage base; these must be the same address. A contiguous
view of a guarded or larger allocation would not satisfy this condition.

Its first512 bytes contain128 uint32 row entries. Each entry stores an aligned
byte offset and a low-bit coded flag. Every row payload starts at a128-byte
boundary. A raw row contains the original2688 uint16 words,5376 bytes. A coded
row occupies4096 bytes:2688 sign/mantissa bytes,1344 exponent-nibble bytes,
21 base bytes and43 zero-padding bytes.

Each original128-value block uses code0 for exponent-zero values, retaining
their original sign/mantissa bits. Other exponents must lie within14 of the
minimum nonzero exponent. Codes1..15 reconstruct base+code-1; an all-zero-
exponent block uses base1. Integer shifts/OR reconstruct the exact original
BF16 word, and the same bits<<16 widening supplies the original FP32 operand.
Signed zero, subnormal, infinity and NaN payload bits are not rounded or
canonicalized by this codec. A row is coded only if every one of its21 blocks
fits. Any failed block keeps that entire row as raw W16 in the same packet.
There is no per-block raw escape pointer or speculative unused payload plane.

The actual fit flags are derived at remote post-load from the real shadow.
Let C be the number of coded expert rows. Total packet length is
688640-1280*C bytes. C0 is refused before allocating the final packet; even C1
is smaller than the old shadow, but is not presumed faster. The original FP32
parameter and existing W16 shadow remain available as the complete fallback.
They are not replaced or freed. Thus a positive packet increases resident
memory even though its active representation is smaller. Only one actual
router's temporary statistics, native decode guards and comparison copies are
live at a time, and private graphs and temporaries are drained before release.

The contract's load-time repack rule requires input independence and exact
value preservation. This codec is designed to meet those conditions; the
actual native decoder must prove every reconstructed word before use. It does
not change target activation, expert, FP8 projection, KV, SSM or MTP formats.

## Original arithmetic and lane mapping

The separate optional native library retains the original router's R1 grid128
and176-thread block. Its sixteen-lane subgroup owns one K256 tile. Both full
K128 halves remain in the same ascending per-lane FMA chain. The last tile has
only128 values/eight terms per lane and accesses base block20, never block21.

For a coded row, each lane loads its own sign/mantissa byte; even lanes load a
paired exponent byte and share it with their odd neighbor. A subgroup leader
loads one base per128-value block. Width16 shuffles use the actual active mask,
including the final physical warp with only16 active threads. Every member
executes the same shuffle calls. A raw row uses the original uint16 read and
bit widening. Row mode is immutable and uniform for an expert CTA.

The original PDL wait precedes all activation, count and weight reads. The
original explicit __fmaf_rn chain, ascending K-tile fold, ascending lane fold,
positive-zero initialization, both CTA barriers, unique output writers and
final PDL signal remain. The current source-row substitution remains
src=min(m,max(vr,1)-1); negative/zero counts still select row0. NULL still means
all M rows. No suffix arithmetic is removed. The optional code supports NULL
also during the original first creation of the count buffer, so producer and
consumer do not select different weight representations on that transition.
It does not allocate or alter the original count buffer.

These are source properties, not a claim of identical generated instructions
or device outputs. New integer/shuffle operations can change register
allocation, spills, scheduling and compiler behavior. Full native output-bit
checks and the official complete golden remain necessary.

## Preparation and immutable owners

A final model-level post-load hook runs after the existing per-layer quant,
attention and HPC finalization. It recognizes actual target early-router MoE
layers and their real registered predecessor. Mamba projection ranges are
derived from the actual fp8c/fp8small/raw out-proj selected by the baseline
helper. Attention ranges are the actual APK planes or ordinary weight selected
by the baseline. The original688128 reservation is subtracted in each case.
These owners are borrowed only for private admission; the new entry does not
retain another projection, graph pool or model weight owner.

The supported engine is immutable, one sequence, single uni worker, with
TP/PP/DP/prefill-CP/decode-CP all one, no ubatching, EPLB, latent or sequence-
parallel router, LoRA, CPU/prefetch offload, weight transfer or KV transfer.
The original smallk W16/vrows entries, optional new native library and actual
prefetch3 entry must exist. Both configured quarter windows must cover the
whole original shadow. Unsupported configurations keep the complete baseline.

M5 and M6 are admitted separately. Original parameter/shadow identities,
pointers, available versions, dtypes, shapes, strides and offsets, the packet
version and relevant smallk/l2pf configuration form the lookup guard.
Inference tensors without version counters are refused before reading
_version. That may reject every real owner; no engagement is inferred from
the source. The actual parameter/shadow key must also remain unchanged across
private preparation.

The gate owns the positive entry; the entry owns the immutable packet and
weakly identifies its gate, parameter and shadow. A global index holds only a
weak reference. A second post-load attempt refuses to replace the owner,
preserving an old captured packet's lifetime; hot reload is unsupported and a
changed identity/version rejects the live Python lookup. No general graph
invalidation or concurrent weight mutation guarantee is claimed. Normal fixed
engine graph lifetime and synchronized teardown remain required.

Production performs metadata/decision lookup only. The new path uses the
existing router_gemm fresh output allocation, with no added GPU allocation,
private qualification, graph, stream, event, scratch or counter. It requires
target row_mask=True; MTP or ordinary GateLinear calls without that marker
keep their original dispatch. Input/output/read-storage overlap refuses the
optional path. Non-NULL counts must be same-device contiguous int32 scalar
buffers. Unsupported input/output layout, grouping, mode or changed flags
returns to the original kernel. Once a native operation has launched, its
errors are visible and are not followed by a production retry.

## Authored remote admission

The new package first derives actual128x21 block statistics. C0 leaves no
positive entry. The native packer writes the new row layout, and a native full
decode uses the same row loader as the real GEMV. Every original BF16 word
must match; decode output has explicit left/right guards, and the header and
storage properties are checked. No weight histogram or private diagnostic is
exported to public score/error channels.

Each actual gate/M then uses separate original/new private FP32 inputs, guards,
outputs and int32 count buffers. The original reference directly calls the
unmodified smallk W16 entry, so it cannot recursively select W12. Every output
bit is compared, including duplicated suffix rows. Counts include
INT32_MIN,-1,zero, every1..M,M+1,VROWS_ALL,INT32_MAX and NULL. Inputs cover two
finite scales, cancellation patterns, signed zero, subnormals, finite extremes,
infinities and quiet/signaling NaNs through explicit raw-word patterns.
Input/count contents, shadow bits, packet bytes and guards must remain intact.

Private old/new graph replays copy changing input/count sources into their own
fixed current buffers, then execute their actual immutable-weight prefetch
ranges and router. The prefetch uses the baseline native prefetch3 ABI, the
real predecessor's projection tensors and unchanged projection budget on a
dedicated private side stream. The main stream joins it before the router.
No temporary positive entry is published to test this path. Replays alternate
increasing/decreasing/negative/zero/all-valid counts and change input values.

All private tensors and graphs remain owned in the function frame. An inner
finally attempts to drain both the side and main streams before owners unwind,
including a failure before the ordinary join. The outer loading stream also
rejoins the private main stream. An allocation OOM after that drain may refuse
the optional feature. Native CUDA launch/capture/replay/synchronization errors
propagate rather than pretending that a damaged context safely fell back.

## Authored timing and limitations

Private timing captures eight original or new prefetch-plus-router pairs,
using the real shadow/packet and actual projection ranges. It samples counts
1,M-1,M,VROWS_ALL under warm and64-MiB-pressure conditions, with five alternating
old/new orderings. The authored criterion is median candidate/reference below
0.995 and maximum ratio below1.01, plus final raw output/owner checks. This is a
conservative admission rule, not an observed production count distribution.

The private prefetch/router pair does not run the preceding Mamba/attention
compute window or the whole concurrent shared/routed expert chain. Thus even
private admission would not establish full-engine benefit. Python metadata
lookup and first-use/setup costs also remain. Only the official complete
golden and whole score can decide usefulness under the user's0.1% rule.

The isolated optional library is built separately. A failed compile deletes
its partial output and leaves baseline smallk intact; a missing library,
symbol, resource eligibility, exact shadow, coded row, numerical equality or
speed decision preserves the original kernel and prefetch. SMALLK_ROUTER_W12=0
disables the feature. No successful whole is evidence of optional engagement
if compilation or admission declined it.

## Source-scale bound and unknowns

For all128 rows fitting in all23 target gates, each packet is524800 bytes,
against688128 bytes of W16. The total immutable package and complete gate
prefetch-range reduction is3,756,544 bytes, about3.58MiB per verify. The original
router stream is15,826,944 bytes across23 layers; the all-fit packet total is
12,070,400 bytes including all512-byte headers. This supersedes the earlier
simple fixed-row/mode-table estimate by accounting for the actual mixed-row
offset table. Fewer admitted rows or layers reduce the bound.

This is a source range/storage bound. It does not assert equal HBM savings,
cache miss reduction, bandwidth, latency or score gain. Existing prefetch can
already make router weights L2-resident. Nibble/base decoding, dependent header
loads, shuffles, additional resident memory and changed cache interference can
make the smaller packet slower. Actual checkpoint exponent spans are unknown.
The existing exact-router test's wide-exponent pattern is for activations, not
weight statistics; existing MTP compression logs are for other matrices.

## Complete provenance and negative history

DEPENDENCIES.json lists12 complete material UUIDs:

- `c41fc629-8c06-4bf2-8cb8-cad0edea9875`: complete verified baseline.
- `100549d4-51c4-4f56-b3ca-2604731ab8a4`: original router chain/schedule.
- `0852152c-9025-499d-83bd-684cf825232e`: exact W16 policy and fallback.
- `3d51cab6-7079-4fb2-98e4-f2a238077897`: lossless exponent-window codec.
- `c0ce642d-0f7b-4c1f-a168-f8edb153aa4e` and
  `59b3dbb4-6919-44ca-85f1-fcbbc0f1d975`: grouped exponent-load method.
- `297e3412-dc4a-45ae-a0f1-b6c3afd76c1b`: next-router prefetch window.
- `f38ce1ad-87fa-4cfb-8a0c-ac46587810f7`: inherited M5 reference.
- `eb48e2f0-50e6-4c39-a1f3-0ac305d17c5b`: inherited M6/depth-five engine.
- `ca5c19dd-da37-4328-880a-152f8c7b3651`: current device-count protocol.
- `110cff4b-d86d-4fe9-ae30-f6ce550d005f`: private post-load/refusal/owner
  design only. Whole2.74300427046762 regressed1.2513% against c41, excluded.
- `77f74985-a845-4963-bf29-b860e8daa7ef`: private full-output/count/graph
  guard and stream-drain design only. Whole2.75184607067041 regressed about
  0.933%, excluded. Actual activation and isolated cause remain unknown.

No A16 input, changed cast, suffix skip or retired whole is imported. Historical
X16 wholes `4ef72848-19b4-492e-a948-e9d44a94bef0` and
`528c6b75-628b-427b-9dab-07f2e7a9a3c1` had whole golden token mismatches with no
score and unisolated causes; this is not claimed to diagnose or fix them.
The retired raw-pipeline whole `cdf8aecf-594a-48ec-bc02-d03a379ce1c2` is not
recycled. Public-note searches found no same gate codec, but do not prove that
all past source lacks it.

## Static receipt and reproduction boundary

An independent actual-source review checked all eight runtime files, then the
public note,12 complete dependency IDs and appended VENDOR entry. It found no
remaining statically demonstrated blocker in the stated narrow scope. The
final helper blob is e8ed94af5a92c23cf2dd13ae4f6bd04122afe287 and native blob is
e95d260658f132d0d177691dbac3fb5192d8c26e. Static source review is not compilation,
numeric validation, production engagement or a measured gain.

The official remote setup already discovers the separate package through
`bash harness/kernels/build.sh`, which invokes
`bash harness/kernels/routerw12/build.sh` with KERNELS_OUT and the pinned CUDA
architecture. That optional build never replaces libsmallk. The standard model
loader subsequently invokes the new final-postprocess hook. No local setup or
test command is prescribed or was run in this experiment. Local checks were
Git text diffs, `git diff --cached --check`, file/blob metadata and byte totals.
The frozen commit/tree receipt can be supplied outside the source snapshot so
it does not create a self-referential commit field. Before an official upload,
the integration task must refresh the whole frontier, material attribution and
slot conditions. Compilation, actual compression/admission, both-stream graph
execution and whole-model validation remain official remote work.

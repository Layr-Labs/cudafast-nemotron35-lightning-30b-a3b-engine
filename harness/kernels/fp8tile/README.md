# Exact mixed FP8 tile codec (optional)

Whole base: `422827d4-d094-4eb3-a0d6-cffb2b04b229`, official source
`96d0eecc4074f66034650af7b8d9de8b6e181ffe`, tree
`78b3a21770699ba6d49b64e72906d066733d61b0`. Its sealed complete score is
2.422929186279726. That whole result does not establish which optional codec
was admitted, or an isolated dictionary benefit.

The existing exact six-bit dictionary pads every tile to a layer-wide worst
exception capacity. This candidate uses immutable per-tile exception counts
to search partitions by actual maximum group stride. A six-bit group contains
only dictionary-compatible tiles; the complement retains the existing exact
seven-bit encoding. Each group trims its raw exception tail to its own maximum.
One real payload allocation owns both groups and their int32 tile IDs, counting
IDs and alignment in the size admission. Existing prefetches address this real
allocation. Retained original references increase resident memory; the size
criterion measures hot encoded bytes, not total resident allocation.

Two disjoint tile-ID grids preserve the original per-column ascending-K32 FP8
MMA, FP32 scale-B then scale-A and RN BF16 epilogue. There is one host entry and
two grid launches, with a skipped empty group. Both grids use the same PDL
policy; immutable weight loads may precede the dependency wait, while mutable
activation/scales use coherent loads after it. Output columns use original
tile IDs. Runtime failure is propagated, never retried with another codec.

Preparation first preserves the already admitted seven-bit reference and then
the baseline optional whole-layer six-bit choice. New mixed storage must beat
that **actual current choice** by more than 0.5%, decode every actual packed
byte identically, match actual BF16 output bytes for M=1/4/5/8 with zero,
nonuniform finite and all raw FP8 activation patterns, and match changing
activation/scale-producer CUDA graph replays. A private nine-round alternating
cold graph comparison (64 MiB flush outside timing) must beat the actual
current complete call by 4%. Captured input/scale copies are the same on both
sides of that comparison. An isolated child first covers all-six, all-seven,
alternating and first/last tile partitions for both supported shapes. All
execution, including compilation and these admissions, is official remote
only. No local candidate execution, syntax/AST evaluation or benchmark.

`FP8TILE=0` keeps the actual baseline choice. Unknown buffers use the original
FP8 CUTLASS path in the integration. Strong owners retain the anchor, payload
views, ID views and original reference; copied/moved unknown pointers are never
decoded using a guessed format. Failed optional build or preparation retains
the actual current codec. No numerical format, MTP depth, confidence, token
acceptance, request/phase policy, or trusted measurement source is changed.

Direct material: `422827d4-d094-4eb3-a0d6-cffb2b04b229` six-bit dictionary,
encoder and exact decoder/ring; `fc04335b-c894-41f9-b73b-aa9b5aac1fb5`
original seven-bit encoder/decoder and FP8 MMA; entire carrier
`b91cd8e6-5d46-4cc9-95b6-47e5d2e3b6bd`;
`927589d0-dd7f-4257-9bef-30129a89b7ae` coherent PDL pattern;
`a33bc79f-84b0-4244-8ab9-e3c43f45f935` and
`b0d0b0b6-428a-4ff1-9e25-1192a3259930` private raw-byte/changing-graph
qualification ideas. Their lower whole experiments are not adopted as bases.
Full inherited lineage is declared in the public submission note/dependencies.
Useful pending `90a47912-89d3-4cc3-88db-10c20e6d3af9` proposes the existing
six-chunk seven-bit out-projection loader ring. We apply that method only to
the new mixed candidate's seven-bit out group (explicit FP8C_RING overrides
remain effective), with its entire-call cold4% and exact-byte gates. The
baseline whole-layer fp8c defaults remain unchanged. The ring2 idea in
`29cb7f6e-f6cb-44ac-a053-836858fb77af` was removed after its fresh whole
422+ring2 result2.39213140293325 regressed more than0.1%; all three zskip
files are again exact whole422. No ring2 runtime is included.

Limitations: two-grid launch overhead and reduced grid size can offset traffic
savings; temporary encoder and retained fallback allocations cost memory;
dictionary tails may be too large; every optional layer may reject admission.
Only the official whole-bundle result decides whether this combination becomes
the next baseline. No isolated gain, admission coverage or runtime correctness
is inferred from source inspection.

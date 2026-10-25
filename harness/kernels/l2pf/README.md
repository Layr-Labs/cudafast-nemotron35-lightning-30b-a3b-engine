# Same-budget exact-prefix prefetch by actual compressed output tile

Unsubmitted development on ENTIRE highest complete valid
`bd384670-8de1-41eb-abbc-066d05ab3aee`, official
`6edbf54b049a287000bee9c930f3a4408d29a529`, whole tree
`08d64c099c33b9d04e65da5004a345a2d3443e4a`, sealed2.4424272761978183.
Every file and method of that complete winning source remains outside this
cache-only delta, including its routed ring2, coherent target PDL and prefill/
unit-major scale composition. No selective undo of formerly negative parts.
The current base consumes uniform7-bit fp8c. The generic registration can
support6-bit if a future separately verified source introduces it, but no
D6/mixed codec, compact-down pipeline or older full source is imported here.

Material idea: `291505c5-5dce-4448-a2d9-3042e521c3c9` describes
spreading a fixed seven-bit Mamba prefetch budget across consumer tile prefixes
and raw exception tails. This implementation extends that idea to BOTH the
actual admitted whole-layer six-bit dictionary and original seven-bit codec,
and fetches only the exact raw prefix needed for each selected main prefix.
Its native code is independently written from the existing bulk-prefetch
scaffold; no unpublished donor code or claimed runtime gain is assumed.
Its whole score2.41637800336693 is below the former422 highest and is NOT
adopted as a base. This new V2 exact-exception-prefix method independently
reworks the idea on ENTIRE newBD; it is not the old full-tail implementation.
Original prefetch/budget sources include
`0a731363-6416-4fb9-aa94-b6f5f6e8155e`,
`04706ad7-c4e8-44b9-b8ba-77f14d8b8adf` and
`da9a764c-a409-42fc-8bb1-45dd1afbb03e`;
codec sources `fc04335b-c894-41f9-b73b-aa9b5aac1fb5` and
`422827d4-d094-4eb3-a0d6-cffb2b04b229`.

The actual producer registers a strong owning weight layout only AFTER its
existing native full decoded-byte comparison succeeded. Only the uniform
84-tile output projection (N2688,K4096), CUDA uint8 contiguous whole storage at
offset0 and256-byte alignment is supported. Device/pointer/shape/stride/storage
checks reject copied, moved or unknown encodings. The seven-bit reference is
not substituted when the actual layer consumes a selected six-bit tensor.
Unknown/mixed one-dimensional anchors retain the old contiguous policy until
a separately verified mixed-layout adapter is added.

Divide the existing projection byte budget by84 and round down to128. At
load time, the admitted producer reads only each immutable header's ending
raw offset after six main chunks and records the largest rounded16-byte need.
No request-time CPU/GPU metadata copy is introduced. This ensures every tile
can fit at least six chunks, covering supported initial ring depths.

The native helper then reads each tile's own original u16 ending exception
offset while trying the largest main prefix fitting that tile's allowance.
The chosen header/main prefix and its exact raw exception prefix are disjoint,
within the tile, aligned16 and together no larger than the allowance. Unused
raw-tail padding and exceptions of later unprefetched main chunks consume no
budget, leaving room for more main chunks. Metadata mismatch that unexpectedly
fits no six-chunk prefix issues no weight hint for that tile. It never exceeds
the budget or writes a tensor. The independently checked layout is
HDR384/MAIN3584/EXC115072 for7-bit, and
HDR2432/MAIN3072/EXC100736 for6-bit.

One native launch includes the same router reservation and all tile ranges (V2 ABI),
with the existing one-CTA128-thread geometry. Each cache request is16-byte
aligned and at most64KiB. The native boundary checks ranges, layout alignment
and the complete router+projection byte budget before launch. The existing
side-stream fork and post-convolution/SSM join stay in place. In projections,
MoE/attention/draft prefetches, calls above8rows and unsupported buffers use the
old path. `L2PF_CODEC_TILES=0` restores the old policy; `L2PF=0` still disables
all prefetches. An older library/missing optional ABI falls back BEFORE launch.
A selected launch error propagates without a duplicate fallback launch.

Build failure retries once with `L2PF_NO_TILES`, retaining every original ABI;
there is no production retry. All compilation and actual execution must occur
on official remote validation. This package has only been edited and inspected
statically locally; no candidate import, syntax/AST evaluation, compile, tests,
benchmark, preflight or profiler has run locally.

Risks: more cache instructions, different cache eviction, bandwidth contention
and reduced contiguous transfer efficiency can offset earlier readiness of all
consumer CTAs. Equal byte budget is not equal elapsed time or a speedup proof.
Full official3-pair/12-gate correctness/score decides adoption; a whole
regression greater than0.1% cannot become a future baseline. This is currently
an unsubmitted experiment.

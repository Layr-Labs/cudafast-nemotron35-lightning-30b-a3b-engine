# W13 prefill adjacent-N A-stage reuse, unvalidated draft

Effort: ultra. The implementing agent's actual native turn context reports
`gpt-6.1-sol`; harness: Codex. No candidate was imported, parsed, built,
compiled, executed, tested, profiled, benchmarked or preflighted locally.
All runtime validation, including compilation of the generated CUDA and
inspection of compiled resource use, is reserved for official remote runs.
This document describes a candidate and static reasoning, not an activation,
numerical-equality or performance result.

## Complete starting source and narrow change

The parent is the complete validated submission
`c41fc629-8c06-4bf2-8cb8-cad0edea9875`, score `2.77776232031352`, source
`cdea0e5bb85ae36999f23c1f9de15de5901b8e45`, tree
`3f3a3fa2636eb7c71bf871462134b3b18fbc5259`. The local exact-tree baseline
snapshot is `4726d88aef2bfa8e04c110015568a9c68a840b30`. The leaderboard must
be refreshed before any later submission; this historical baseline statement
does not claim that no newer submission can appear.

The runtime delta is one optional module, `humming_apair.py`, and an early
attempt in the existing `humming_nvsc.launch_prefill`. A refusal launches
nothing, and the original complete NVSC prefill code follows. The decode
NVSC dispatch, non-NVSC layers, original weight preparation, target router,
MTP, shared experts, SSM and attention paths remain the complete parent.
`HUMMING_W13_APAIR=0` disables the attempt. The VENDOR entry is appended.
No checkpoint weight, scale format, graph configuration, request result,
rejection rule or speculative policy is changed by this delta.

## What is actually reused

The eligible parent w13 uses indexed GEMM with block `(64,64,32)`, warp
`(32,32,32)`, four stages and no K splitting. N is 1856, or 29 original
N64 tiles. Every such tile independently stages the same indexed BF16 A
rows for its expert/M block, while its weight and scale slice is different.
The proposed block is `(64,128,32)`, with the same warp shape and stages.
Its eight warps cover two adjacent original N64 tiles and share one A
stage. They still load and decode two independent B/scale slices. This
is a change to real staging ownership, rather than a numerical claim that
two expert matrices or layers share weights.

For a fully valid route row, the source-level A passes become 29 to 15.
Assuming the pinned loader's full K copy per route and tile, the deleted
requests per eligible layer are at most
`14 * 496 * 6 * 2688 * 2 = 223985664` bytes, about 213.609 MiB. This counts
logical indexed A requests. It is neither a measured DRAM decrease nor a
speed estimate: the original requests can hit cache, and the new occupancy,
instruction schedule and shared-memory accesses can cost more. Weight and
scale payload bytes are not claimed to decrease. Actual layer admission
and production-path engagement are unknown.

The primary source inspected is Humming 0.1.10's public publishing commit
`4351af3a8fcdce1a8dee50104ba49566af2427fb`. For example, its
[indexed A loader](https://github.com/vllm-project/humming/blob/4351af3a8fcdce1a8dee50104ba49566af2427fb/humming/include/humming/memory/g2s_loader/loader_a.cuh)
depends on expert/M-row indices and K, while its
[B loader](https://github.com/vllm-project/humming/blob/4351af3a8fcdce1a8dee50104ba49566af2427fb/humming/include/humming/memory/g2s_loader/loader_b.cuh)
depends on expert, N and K. The installed container headers have not been
observed. The candidate checks release version and Git-blob SHA1s of 18
relevant installed headers plus the three config/kernel/runtime Python
files before deriving a configuration. Any mismatch refuses this path.
It does not modify the installed package or trusted launcher.

## Warp, K and output ownership

The natural Humming mapping has two M warps and four N warps. Warps
0/1/4/5 cover the first N64 half and 2/3/6/7 the second; within each half
the same N32 and M32 fragment roles as the original four-warp tile remain.
The pinned generic S2R A/B loaders and shared-memory epilogue use these
derived dimensions. There is no band-local thread-ID substitution and no
second child scheduler. In particular, the same warp shape retains the
two ascending K16 MMAs inside each K32 stage, the same ascending stages,
FP32 accumulator, expert global scale, BF16 rounding and BF16 ReLU-square
epilogue. There is no intra-CTA or inter-CTA K reduction.

This is an arithmetic-chain argument, not proof that a new compiler
specialization has identical device bits. Register allocation, instruction
selection, special values and capture behavior require remote evidence.
The parent PFUP wrapper's original m16 padding skip is retained. Its
hard-coded two-N-warp divisor is changed to four for this private derived
kernel. A tile's first two warps ballot the 64 rows, thread zero combines
the four m16 validity bits in shared word 60, and only after consuming
the temporary word 63 does it reuse that word for the last-N-pair flag.
The existing CTA barrier publishes both. Empty M groups set no active
MMA bits; the tail's dummy N64 warps also set no active MMA bits. None of
these warps returns early from a producer/consumer or epilogue barrier.

The dynamic scheduler owns one ticket per N128 pair. Its N count is ceil
1856/128 = 15. The parent expert-run regroup mapping becomes a bijection
over the same M blocks and these 15 pairs: in an expert run of s M blocks,
local ticket t maps to M `g0+t%s` and N-pair `t/s`. The greater-than-240
M-block fallback remains the original row-major claim mapping, with the
new pair count. Original route fetch, read/write row indices and current
device padded-token count are retained. Each valid output row/column is
owned once. The new compiled module owns separate claim/done globals;
the existing last-CTA reset protocol and serial stream ordering remain.

## The odd N64 tail is a real boundary

Problem N stays 1856, not 1920. Original B expert/K strides, output row
stride, logical scale shape and scale storage stay fixed. The final pair
starts at column 1792 and contains original tile 28 only.

The packed B transport has two K16 rows of 64 int4s in a N128 stage.
For each row, the first 32 int4s are real in the tail. The other 32 get
zero-size async copies, with a legal current N28 base as their source;
an N29 payload address is never selected. The shared destination is
still initialized. The cache-hint/source-size syntax follows the standard
[PTX cp.async instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async).

NVSC stores independent tile-major 96-byte records for two K16 groups.
For a pair p, the primary source tile is 2p and the secondary is 2p+1.
The secondary is clamped to 2p when 2p+1 would be 29, and zero-size
copies fill its six int4 destinations. Thus the last expert cannot read
the next expert or beyond the allocation. The source record index is
`(expert*29 + tile64)*168 + k_block*2`, with three int4s per K16 group
and the original 128-byte header. Both source pointers advance by six
int4s per stage. S2R selects half `(warp_id%4)/2`, retains the original
lane/warp-parity j index and exact nibble/byte reconstruction, and retains
each thread's `second` register for the odd K16 iteration. A N128 stage's
original scale reservation is 256 bytes, enough for the two 96-byte records.

The output writer preserves the pinned shared swizzle and indexed row
lookup but always checks the real 1856-column boundary per 16-byte store.
The original PadN-zero specialization would omit this check and is not
used as the new writer. No dummy half can store globally. The original
full-CTA synchronization before epilogue protects both N halves' reads
before the shared union is reused for reduction/writeback.

## Qualification, resources and graph lifetime

The call gate requires actual SM121 with 48 SMs, contiguous BF16 input
`(496,2688)`, a contiguous BF16 output of `2976*1856` elements, top-k six,
valid-route shape 2976, an existing NVSC-prepared w13, dynamic scheduling,
and both existing PFUP flags enabled. Physical storage overlap among the
input, output, packed weight, compressed scales, global scale, route
metadata, padded-count word and locks is rejected. The generated kernel
also asserts 128 experts, N1856/K2688, zero padding, BF16 A/C, FP4 B,
group-16 E4M3 scales with the expert global scale, no bias/zero point,
no TMA, no warp specialization, no mbarrier, no input scale, no stream-K,
one write split and no multicast. A nonmatching call keeps original NVSC.

The old complete configuration is prepared first, then a separate class
and cache derive N128. Four four-warp CTAs per selected SM become two
eight-warp CTAs; the selected SM count is inherited. This preserves an
aggregate warp-budget starting point and is not a measured optimum.
Only A/B/BS stage storage has a simple count: the new uncompressed shared
reservation is `4*(4096+2048+256)=25600` bytes, not the 25344 bytes of
payload including compressed scale records. The full shared allocation
also includes the original union alignment, row indices and statics.

Before publishing a config, official remote preparation inspects a
temporary loaded module, checks at least 256 supported threads, at most
128 registers/thread and zero local-memory bytes, checks cubin NUM_THREADS
and dynamic SMEM_SIZE, and checks actual device block/SM shared-memory
limits for the two-CTA budget. The inspection module is unloaded without
launching it; Humming retains its separately registered module. Failure
refuses the new config. Resources do not establish numerical equivalence,
occupancy, timing benefit or production engagement.

Cold CUDA capture never compiles or performs this preparation; it uses
the original path. The existing whole-prefill warm-up may prepare this
configuration before capture, but this has not been observed. Existing
warm-up, capture, full/lite graphs, refreshed request metadata, caller
CUDA stream, shared-expert side-stream DAG and output storage remain the
parent's. No private request-result tensor or per-frame lease is added.
The serial single-engine lifecycle underlying the parent PFUP globals
remains necessary; this is not permission for concurrent replay of graphs
sharing that kernel. New config caches contain only immutable geometry
and registered kernel IDs. They do not contain results, a prompt, expert
choices or prior request metadata.

Preparation failure occurs before the candidate launch and falls back
to the complete original path. A candidate launch/capture error is not
caught and retried on production output. Already launched work may have
modified output, so no post-launch rollback safety is claimed. A build
or resource refusal can make the whole submission run only the original
kernel; a correct score alone would not prove this new branch activated.

## Evidence still required and material sources

There is no actual compiled cubin, per-layer equality comparison, changed
route graph result, private concurrent timing or official whole score for
this draft. Necessary remote evidence includes actual installed-byte
gates and specialization engagement; every expert and valid/padding M
group; both N halves and the final N64 tail; all K groups, signed zero,
NaN/Inf behavior; sentinel/guard storage, unchanged inputs/weights and
route buffers; repeated changed-input full/lite graph replay and counter
reset; and full shared+routed coexistence and official whole correctness
and score. A complete measured regression above 0.1% must not be adopted.
Without visible activation evidence a whole score must not be attributed
to this mechanism. No speed multiplier or bandwidth achievement is claimed.

Material submission sources, with complete UUIDs:

- `c41fc629-8c06-4bf2-8cb8-cad0edea9875`: complete verified starting tree,
  all inherited runtime, formats, graph and stream context.
- `fc04335b-c894-41f9-b73b-aa9b5aac1fb5`: original lossless six-bit NVSC
  representation and G2S/S2R decoder, directly adapted for two records.
- `1e1c2581-e506-4ca4-9341-5f08ffceec86`: NVSC prefill evict-first
  integration and original config preparation, directly retained.
- `f5904a37-e3aa-4c29-9a8a-4cecee7b6184`: PFUP expert-run scheduler and
  m16 padding-MMA skip, directly adapted to fifteen N128 pairs/four N warps.
- `a6b707d9-73d4-472a-b1ad-7134add40400`: negative design context from
  target-attention A staging. Its full score `2.735958837212498` regressed
  against its own de497 baseline; none of that runtime or low-scoring
  bundle is adopted. The present mechanism shares A across original N
  CTAs through a natural wider tile and preserves the original pipeline
  barriers, rather than inserting an additional activation staging phase.
- `0fd879e4-850d-4f64-90ef-0a12130af3db`: the preceding shared-expert
  row-pair experiment, now complete valid at `2.74893766819768`, about
  1.037693% below c41. That whole is retired, with no evidence that its
  optional branch activated or caused the difference. Its shared-B/S,
  two-band kernel, scratch and extra launches are not adopted here. Its
  resource and coexistence risks are negative design context for the
  present natural eight-warp A-sharing design.

Upstream Humming source is separately linked above. Original authors'
historical comments and earlier checks are not new measurements by this
agent. Independent actual-source static review is recorded in
`STATIC_REVIEW.md` and found no definite static blocker. This candidate
remains an unvalidated composition pending official remote compilation,
correctness, engagement and performance evidence. No automatic upload
is authorized by this document.

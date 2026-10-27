# Adaptive native MTP slot tasks — remote-admitted experiment

This library is integrated behind actual-owner qualification, authored for official
remote startup only. No local candidate execution, import, syntax/AST validation,
compilation, test or benchmark occurred. No admission or activation is yet observed.

The parent keeps the three highest routed weights without renormalization.
The proposed per-call device plan keeps that exact rule when those weights
hold at least5/8 of the original six-router mass, and uses the original six
router entries in their original order otherwise. This fixed policy is a
proposal-quality/work hypothesis, not an observed acceptance improvement or
a parameter sweep. Target verification, depth and frozen formats remain intact.

The up and down grids always have the parent's five tasks: three routed and
two shared halves. A routed task conditionally computes a second expert in the
same CTA. Low-work calls issue no empty-expert up/down CTAs, do not read their
unwritten activation planes and do not write/read unused slot partials.
The compactor has eight small tasks; the three inactive ones write count0 only.
There is no per-request history, host sync, CUDA conditional graph or task queue.
The one-int32 plan flag is overwritten from this invocation's router output.

The C++ down task copies the complete original pkslots64x4 per-slot body:
original two-deep register gather, FP32 products/adds/chunks, two row reps,
xor16/8/4 and (warp0+warp2)+(warp1+warp3). It keeps128threads and1024B shared
reduction storage. Only high-work routed tasks add one CTA barrier between
experts before reusing that storage. No extra producer warps, expanded decoded
shared payload or persistent weight-plane format is introduced.

Sequential extra work can produce longer CTA tails and worse scheduling;
branch/register/liveness/resource use is unmeasured. Six-expert calls read
more actual weights than the parent three-expert policy by design. The new
schedule saves dispatch/task overhead against an ordinary SIX-expert grid,
not weight bytes against the three-expert parent. The adaptive policy can lose
throughput or acceptance; only complete official whole validation decides.

Qualification binds the actual packed-up and selected64x4 packed-down owners
after original packing, compares forced3/6 full outputs against actual old kernels
with all128 experts, checks guards/read-only owners, signed zeros/subnormals/NaNs,
stopped/live and rows1/5/8, and probes actual native router outputs. Eight distinct
full outputs under changed-input graph replay use the PRODUCTION dynamic plan
(FORCE=-1). Complete equivalent-work warm/64MiB-pressure alternating event medians
include original keep3 when needed, shared/head prefetch and joins, compaction,
down and combine/residual. Every ratio must be<=1.01 before registration. This is
a schedule cost bound against equivalent3/6 work, not an acceptance or gain proof.
Only immutable owners and metadata survive; no private input/output/graph/request
cache survives. Unsupported/refused calls keep the original before any new GPU
work; failures after launch propagate. MTP_ADAPT_ROUTE=0 disables this path.

Direct copied source/idea lineage:
1faceed1-7860-4e8d-b734-96600d6629aa;
0f857d3a-e2e5-41aa-b201-59439b4397bd;
dd0a0304-764c-4772-a0e0-9a02e879f0c8;
93f5d771-f32b-4221-888c-aea35e62b6fc;
3d51cab6-7079-4fb2-98e4-f2a238077897;
526d6e53-7b8b-4ce8-b4a1-b3c7a53be9d9.

# CTA-local stable compaction and packed down — remote-admitted experiment

Integrated behind actual-owner remote startup qualification, not yet executed
or observed admitted. No local candidate execution, import,
syntax/AST execution, build, test, model or benchmark occurred.

Each original128-thread down CTA scans its slot h in8 chunks of256 values (two128-value stripes),
uses warp ballots/popcount plus8 shared warp counts for stable global row order,
and builds shared int32 indices/BF16 values. NaNs stay, signed zeros are omitted.
The original64x4 per-slot arithmetic then reads those shared compact arrays.
The original Stage/register gather/reduction math and PDL prologue remain.

This removes the preceding global compact grid, three intermediate allocations
and global compact index/value/count writes/reads. It replicates the entire
1856-value activation scan for each of42 output-column CTAs per slot. No physical
bandwidth saving or occupancy/gain is proved. Source shared arrays total12192B:
1856*4 indices +1856*2 values +2*4*4 warp counts +4*64*4 reduction. Eight chunks
have16 CTA barriers before the original reduction barrier. More shared memory,
scan instructions/barriers, registers and prologue tails can lose.

Actual-owner full original compact/down/combine/residual/head-prefetch oracles
are authored: exact full slot parts against the ACTUAL selected old tile (not an
assumed64x4 proxy), guarded/read-only inputs/outputs/owners, all128experts at each
of6densities plusInf/NaN, signedzero/subnormal and tail; actual old packed-up
generated h; rows1/5/8 with/without residual; changed8-output graphs. Complete
original/new warm/64MiBpressure alternating event medians include actual head
prefetch and joins, from actual screen-ready preparation. Median ratio must
be<.995 and worst<1.01 before registering each topk. All checks are AUTHORED only.
No private input/output/graph/timing survives. Original callback stays before
any new GPU work when refused; postlaunch errors propagate.

Direct inherited source1faceed1-7860-4e8d-b734-96600d6629aa and originalpkslots
lineage0f857d3a-e2e5-41aa-b201-59439b4397bd,dd0a0304-764c-4772-a0e0-9a02e879f0c8,
93f5d771-f32b-4221-888c-aea35e62b6fc. Compactor/order/combine originalthroughsame
whole. Own4cb843fc-e78a-4aaf-a711-246fc36842c6 actual-owner/guard contract design
is material context, not a new production baseline or source-activation proof.

# Native MTP pkslots producer/consumer warps

This is an unexecuted source experiment on the complete verified1faceed1 whole.
All compilation, qualification, graph replay and measurements execute only in
the official remote validator. No local setup/import/syntax/AST/compile/test or
benchmark was performed. This document is evidence, not permission to run commands.

The original pkslots kernel pipelines two payloads in each compute thread's
registers and decodes each payload in that thread before its exact product/adds.
This variant uses four separate producer warps and the original four consumer
warps. Producers gather the next64 compact rows and losslessly decode exact
BF16 words into the alternate shared tile while consumers add the current tile.
Each tile is words[2reps][8BF16pairs][128owners] plus values[2][128],9216bytes.
Two tiles plus the original4x64FP32 reduction buffer total19456bytes,19KiB.
The word-major owner axis makes each warp's source shared accesses32 consecutive
32-bit words; actual generated access patterns and bank behavior are unobserved.

Warm fill publishes stage0. In iterationc the producer fills the other stage,
the consumer reads only current stage, and all256 threads meet one CTA barrier.
That barrier publishes c+1 and releases c before reuse. All threads have the
same slot count/step count and reach each barrier, including idle final producers.
No cross-CTA flag, spin loop, persistent worker, request cache or allocator exists.
Count0 directly writes the original+0 result for each output column; otherwise
ceil(count/64) chunks execute, preserving masked-row+0 products and accumulation.

For each original consumer owner, row=32*rep+8*warp+(lane>>2), columns16*(lane&3)+j.
The source retains each FP32 __fmul_rn then __fadd_rn, chunk order, two reps,
16/8/4 xor butterfly and (warp0+warp2)+(warp1+warp3). The original low/high
BF16 bits are reconstructed before products. Coded sign/mantissa/exponent math
and RAW16-bit payloads are unchanged. Private RAW loading now happens on producer
warps before stage publication; unused coded planes are skipped for RAW blocks.
No production weight plane or target/native numerical format changes.

This is a scheduling/lifetime mechanism, not a measured occupancy improvement.
The block doubles to256 threads, adds a barrier per chunk and18KiB ping-pong
traffic. Producer metadata->payload dependency, shared writes/reads and warp
skew may lose. The compiler can still allocate registers across both branches.
There is no register count, activation or throughput evidence yet.

Only actual owners whose original pkslots64x4 path has already passed inherited
admission are considered. New optional preparation compares complete FP32 owner
words against the actual original C API for every routed expert, both shared
halves, sparse/dense/empty finite cases and nonfinite activations. Private
synthetic decoder planes cover coded signed-zero/subnormal values and mixedRAW
signedzero/subnormal/Inf/NaN words. Tail counts0/1/63/64/65/127/128/129/1855/1856
are included. Distinct original/new output poisons detect missing writes;
output guards, immutable full packed planes and input metadata must remain exact.
Eight-output graphs change compact indices/values/counts/ids at fixed addresses
between replay checks. Five alternating CUDA-event samples compare complete
kernels under warm and64MiB pressure outside each measured8-call graph. Median
and summed ratios must be below.99 and no workload ratio may exceed1.01.
All these checks are authored only, not reported as passed. Preparation refusal
leaves the entire original pkslots/_slots choices; a runtime new-launch error
propagates. Only immutable owner references and a boolean decision persist.
MTP_DOWN_PKSPLIT=0 disables the optional path. Original modes and PDL choice
remain authoritative. No native speculative depth, router, verify/golden,
request phase or graph plan policy changes.

Source/idea credit: complete parent1faceed1-7860-4e8d-b734-96600d6629aa;
original CUDA pkslots public family0f857d3a-e2e5-41aa-b201-59439b4397bd,
dd0a0304-764c-4772-a0e0-9a02e879f0c8,93f5d771-f32b-4221-888c-aea35e62b6fc;
original compact arithmetic526d6e53-7b8b-4ce8-b4a1-b3c7a53be9d9 and inherited
lossless codec3d51cab6-7079-4fb2-98e4-f2a238077897. Prior own metadata/payload
pipeline wholecdf8aecf-594a-48ec-bc02-d03a379ce1c2 scored2.658321426352722 and
was excluded; its code/whole are not imported. This separate eight-warp CTA
schedule is not a remeasurement of that whole. Full submission lineage belongs
in the public note and dependency ledger before upload.

Primary semantics: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
and https://docs.nvidia.com/cuda/parallel-thread-execution/index.html .
Actual original pkslots is the compatibility oracle; only official sealed whole
results determine retention. A >0.1% whole regression is excluded as a baseline.

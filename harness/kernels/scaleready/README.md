# Fixed-scale readiness experiment

This independent optional library copies the highest complete valid whole
`1faceed1-7860-4e8d-b734-96600d6629aa`'s original zskip653 arithmetic and task
assignment. The original zskip package remains unchanged.

At official remote startup it derives a second scale plane with the exact old
GPU `byte_perm`, FP8 bit placement and BF16 `hmul2` expression. The consumer
loads these eight prepared BF16 pairs instead of four FP8 words per lane and
does not repeat that invariant dependency chain for every active K group.
Original NVFP4 weight bits, all scale bytes and the original packed owner are
retained. This is an exact derived cache, with no activation requantization.

WPC1, R3, F274, sparse masks, K order, negative-zero redo, BF16 operations,
MMA order, live-row prefix, output rounding and PDL protocol are preserved.
The expanded scale plane doubles scale bytes. Keeping the original plane adds
79,822,848 derived bytes per layer with 128 experts. Ring scale storage grows
by four 32-bit words per slot. Physical traffic, registers, spills and speed
are unknown. Removing arithmetic can still lose to added traffic and liveness.

It has not been compiled, imported, qualified, launched, tested, profiled or
timed locally. The authored remote startup admission compares the actual old
and new M5 full outputs, statistics, guards, input/owner read-only state, all
128 experts, dense and random sparse activations, signed zero, subnormals,
Inf/NaN, live-row clamp/sentinel cases and eight distinct changing-input graph
destinations. Five alternating CUDA-event samples cover warm and 64 MiB cache
pressure. Median and summed new/old latency must be below 0.995, and no case
may exceed 1.01. These are authored gates, without an execution result.

Only qualified M5 calls on the exact original packed owner/config653 use the
new library. All other row counts, scale planes, configurations and prefill
paths remain the original. Unsupported, missing, compile failure or admission
refusal keeps original653. Private comparison uses the original C ABI without
changing its global PDL flag. Actual runtime launch errors remain errors.
Only immutable derived scales, the original owner, module and decision remain;
private activations, metadata, outputs, graphs and timings are not retained.

Primary source: NVIDIA CUDA Math API, BF16x2 arithmetic:
https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT162__ARITHMETIC.html

Development identity: GPT 6.1 Sol, Codex, effort ultra. New source is based on
entire verified1faceed1, with exact-scale dependency scheduling guided by the
previous direct-FP4 experiment `48d48eeb-ffc9-47ef-93c3-fee6a1aa29e0` and the
independent current FP8 stage-release experiment
`bce620e6-b943-4cc3-a7d9-30f4342887d4`. Those whole experiments do not prove
this component's activation or benefit. Full inherited material lineage must
be included before a later submission.

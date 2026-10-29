# A16 target-router input: isolated experiment in the earlier X16 family

Effort: ultra. The implementation turn used GPT 6.1 Sol through Codex. This is an unvalidated source draft. No candidate code, imports, syntax or AST checks, setup, compilation, AOT, tests, benchmark, profiler, preflight or model startup were run locally. Every executable admission check described below is newly authored code for the official remote workflow, not a result observed by the author.

## Baseline and scope

The complete verified baseline is submission `c41fc629-8c06-4bf2-8cb8-cad0edea9875`, score 2.77776232031352. Its official source is `cdea0e5bb85ae36999f23c1f9de15de5901b8e45`, whole tree `3f3a3fa2636eb7c71bf871462134b3b18fbc5259`; the independently preserved local reconstruction is `4726d88aef2bfa8e04c110015568a9c68a840b30`. The experiment starts from that whole tree. No older failed source is substituted for it.

The fixture is the pinned Nemotron 3.5 Lightning model on DGX Spark GB10, SM121. The current baseline uses native MTP depth five; a target verify therefore includes up to six rows. This draft only offers the new input representation to target MoE blocks at M5 or M6. It does not change the MTP model, draft head, sampler, target head, prefill path, expert kernels, quantization, gate weights, top-k, confidence rule, SSM state or any model result cache.

The intended mechanism is to avoid an otherwise materialized BF16-to-FP32 router input and to read the same activation elements as BF16 inside the existing FP32 FMA/reduction program. It is activation and intermediate traffic work. It does not reduce the model's weight bytes. All observed scores in this document belong to the named official submissions, never to this draft.

## Earlier X16 results and what is isolated here

This is a continuation of an earlier X16 method, not a newly discovered or previously untested algorithm. Material provenance includes both:

- `4ef72848-19b4-492e-a948-e9d44a94bef0`, source `6f1a572ce6fecd444c3d896f2139f1cd6ff7fe26`. Its complete official run failed the sealed free-run decode golden with a token mismatch and no score. The source also changed other target/MTP code and included additional unrelated files. The contribution of BF16 router input was not isolated.
- `528c6b75-628b-427b-9dab-07f2e7a9a3c1`, official source `9ecd2eb99e2ea1c7ca8941c46d56f02d7c516e2d`, tree `787ab923725c9d8985a1d835a83fd6b6304dbaf9`. This was our earlier admitted-X16, seventh-Gluon and plan-readiness combination. It too failed the sealed free-run decode golden with no score. The router, Gluon and readiness contributions were not isolated. Its prior prepared-source record is superseded by that terminal result.

Neither failed whole source is adopted as the baseline or described as a valid optimization. This experiment selects the BF16 input idea with explicit attribution, drops early immutable-weight prefetch, drops the Gluon changes and drops plan-readiness changes. It preserves the c41 router's post-wait weight loading as well as its activation and verify-count ordering. The new post-load admission and rejection boundary are specific changes to the experiment; they do not prove a cause or repair of either historical mismatch.

The earlier unsubmitted reserve `9b12379d78e4a4a86485d806f39402eeb5027601` had no independent official result. It is not positive numerical evidence. The tested historical combination above must not be replaced by that reserve's unvalidated status when describing the record.

## Original input and output ownership

In the c41 target early-router path, `NemotronHMoE.forward` passes the original BF16 hidden state to the expert operation and a separate FP32 conversion to its router-input parameter. The runner computes router logits after the existing shared-expert stream fork. With no routed input transform, the shared expert already receives the same BF16 hidden tensor as the routed input.

The new producer guard requires the actual pure `torch.ops.vllm.moe_forward_shared` entry, existing early-router eligibility, shared experts, no latent transform, no routed input/output transform, no runner-owned gate, no sequence parallelism, no MTP prefix, H2688 and E128. It does not extend the mutable `moe_forward` entry to a new alias relationship. The BF16 router input is read-only, and the output is a fresh M-by-128 FP32 allocation.

The existing auxiliary stream's record_stream, producer wait, shared-expert work and main-stream join are unchanged. The A16 router uses the original main stream. No production stream, event, scratch arena, output alias or asynchronous host copy is added. The input and all current-frame results remain governed by the original caller's graph and frame lifetime.

## Admission occurs before the external cast is selected

A new model-level `process_weights_after_loading` hook uses the loader's existing final hook. The loader invokes it after layer quantization processing, attention processing and HPC module processing. This matters because a load_weights-only hook would run before some actual parameter finalization.

The hook first clears each block's admitted-shape tuple. It refuses distributed or mutable configurations: TP/PP/DP/prefill-CP/decode-CP must all be one; no EPLB, no ubatching, the executor must be uni, max_num_seqs must be one, no LoRA, no weight transfer, no KV transfer and no CPU or prefetch weight offload. Actual weights must also pass the helper's CUDA, contiguous FP32, exact 128-by-2688, current-device and SM121 checks.

For each eligible real gate, only shapes whose private qualification returns true are placed in a static M5/M6 tuple before compilation or capture. The forward chooses between the BF16 tensor and the original external FP32 conversion before evaluating the cast. A refused shape therefore preserves the old cast outside the opaque expert operation. It does not merely recast after changing every compiled call's boundary.

Production calls only look up a pre-existing admitted entry. They do not launch qualification, allocate private test graphs, perform CPU equality checks or initiate capture. The entry key includes device, M, actual weight pointer/version, exact W16-shadow pointer/version, router-row grouping and PDL setting. Weak owner identities reject pointer reuse without adding a new strong reference to another model's weights. Original baseline W16 ownership is unchanged.

This is a fixed immutable resident-engine experiment. A post-load or configuration change after capture is not supported as a general hot-reload protocol. The model's hook clears shape choices on a supported new post-load cycle, and cache keys reject changed versions; an already captured graph cannot be transparently rewritten by those Python guards. The full official engine lifecycle must still be validated.

## Private checks authored for the official remote engine

No production request, prompt, live hidden-state buffer, routing output or sampler result enters admission. The private inputs are synthetic and the weights are the actual loaded gate parameters.

First, a per-device oracle enumerates all 65,536 BF16 bit patterns. It compares the raw FP32 bits from Torch's BF16 widening with the new native conversion utility. Zeros, signed zeros, subnormals, infinities and all NaN payloads are included. Any difference refuses the device's A16 path. In particular, a NaN-payload difference can cause every layer to keep the baseline; there is no evidence that this exhaustive conversion gate will pass on the remote image. The comparison is not relaxed to make a path engage.

For each actual gate and M, private FP32 outputs and left/right guards are explicitly allocated as FP32. This is important because the model loader runs under the model's default dtype, which can be BF16. The new C ABI always writes FP32 outputs.

Private eager comparisons use two finite ranges, signed zero patterns and a raw BF16 edge set including subnormals, maximum finite values, infinities and quiet/signaling NaNs of both signs. They compare raw bits of the entire guarded output for every verify-row count from zero through M. Inputs, weights and the existing shadow are checked for unintended mutation. The reference is the original FP32-input smallk call with the same row grouping, PDL and exact shadow choice.

Separate private graphs exercise changed input producers and changing verify-row counts. Both the reference and candidate graph contain their own current-input copy before their router. Replays with changed finite patterns are compared. This is stronger than replaying an unchanged output, but it is still finite testing and does not establish the compiled model's whole golden stream.

Finally, private pure-kernel timing graphs contain eight original or candidate router calls. The reference FP32 activation is already materialized, so no speculative cast-launch benefit is added to the admission comparison. Warm and 64-MiB-pressure runs alternate ordering. The authored rule requires a median candidate/reference ratio below 0.995 and every sampled ratio below 1.01. These are local per-kernel admission criteria inside the remote engine, not observed measurements or a guarantee of full shared+routed concurrency benefit.

## Private stream and error lifetime

Private allocations and graph execution use one nondefault stream. It waits for the loading stream, so actual finalized weights and any existing W16 shadow are ready. Every local Tensor, graph and graph pool remains alive in the same function frame through a finally block that synchronizes the private stream. Only after that drain can the local owners be released. The loading stream also rejoins the private stream in the outer finally.

This drain is required on exceptions during capture, replay, event recording or comparison, not only on the successful end of a timing loop. No request-owned state is modified. A numerical or speed refusal returns false after private work is complete and retains the original external cast. A native launch, capture or replay error remains visible after cleanup; the implementation does not retry the failed operation on production state or assume the CUDA context is healthy after an asynchronous error.

## Kernel and ABI changes

The original FP32 router kernel and its C entry points remain intact. The separate `router_a16.cuh` is a source derivative: only the activation pointer width and the two activation load sites change, plus the separate function name. It uses a native BF16-to-FP32 conversion, then the same explicit FP32 FMA chain and addition program. There is no new tensor-core reduction or reduced-precision accumulation.

The K256 tiles, 16 per-lane terms, ascending tile fold, ascending lane fold, output ownership, verify-row substitution, PDL wait and dependent launch are retained. All mutable activation and verify-count reads, and all weight reads, remain after the original PDL wait. M5/M6, N128 and K2688 are explicit in the optional ABI. Router grouping R1/R2/R4/R8 is selected exactly as in the baseline. Mode zero is the production variant. The shadow branch consumes the baseline's existing bit-checked W16 representation; otherwise the original FP32 gate weights are read.

The optional Python ctypes declaration has eleven arguments in the same order as `sk_router_gemm_a16`: input pointer, input row stride, M, weight pointer, FP32 output pointer, output row stride, R, W16 flag, PDL flag, optional verify-count pointer and CUDA stream. There is a separate version/init and a bounded conversion-oracle entry. No original public ABI changes.

The new kernel can still compile differently because its operand loads and register liveness differ. Source-level FMA and fold preservation is not device binary equality or a proof of bit equality. The official remote compiler, complete model golden and score are the deciding evidence.

## Build and fallback

The optional header and its new C exports are excluded by SMALLK_NO_A16. The official setup build tries the original combined build, then without A16, then without W16, then without both. Every attempt removes the previous library artifact and checks the actual pipeline status and nonempty output. A failed optional compilation therefore cannot leave a stale partial shared library reported as a successful build.

A library missing the A16 symbols keeps the original FP32 behavior. SMALLK_ROUTER_A16=0 also keeps empty admitted-shape tuples. Production refusal converts BF16 to FP32 before any legacy smallk/routerbig/Torch consumer. This defensive conversion preserves their input dtype; after an admitted graph has already been compiled, it does not by itself restore that graph's earlier external cast boundary. Only the pre-compile rejected-shape path preserves that boundary.

A whole official pass after a no-A16 fallback would establish only that the whole package is valid. It would not prove that A16 was compiled, admitted or used. No stdout, counters or path-hit evidence has been observed for this draft.

## Source-level scale and limits

For grouping R, the original N128 router has 128/R CTAs. Each CTA loads the M-by-2688 activation once. Replacing a four-byte activation load with two bytes has a source request delta of 2 * (128/R) * M * 2688 bytes per layer. Avoiding the FP32 intermediate write adds 4 * M * 2688 bytes. At M6 and R1, that is 4,193,280 bytes per layer, or 96,445,440 bytes over 23 target MoE layers. The unique removed FP32 tensor alone is only 64,512 bytes per layer.

These are source-level repeated load requests and intermediate writes. They are not measured DRAM transactions. The repeated activation reads can hit L1/L2, a conversion may fuse with the preceding norm, and overlap with shared experts can hide any saved work. The experiment does not assert a saved launch, a speedup, a production hit rate or an expected score. A private speed admission does not model full-model contention.

## Complete material attribution

The separate dependency file lists complete submission UUIDs. Additional carried implementation provenance remains in the unchanged baseline VENDOR entries.

- `c41fc629-8c06-4bf2-8cb8-cad0edea9875`: complete verified highest baseline, including all current weight, target, MTP, sampler and graph methods.
- `4ef72848-19b4-492e-a948-e9d44a94bef0`: earlier BF16 router-input/X16 idea and actual failed whole-source evidence; not a valid baseline.
- `528c6b75-628b-427b-9dab-07f2e7a9a3c1`: earlier own admitted-X16/Gluon/readiness combination and actual failed whole-source evidence; causes not isolated and unrelated changes not copied.
- `100549d4-51c4-4f56-b3ca-2604731ab8a4`: the original smallk FP32 router chain, early-router handoff and source structure from which this optional activation specialization is derived.
- `0852152c-9025-499d-83bd-684cf825232e`: original exact W16 shadow/dispatch already present in c41 and retained, rather than a new gate-weight format.
- `f38ce1ad-87fa-4cfb-8a0c-ac46587810f7`: M5 router instantiation carried through the baseline.
- `85890747-0e71-4c01-aaf1-0252b06d368b`: current inherited smallk router implementation/dispatch carrier.
- `eb48e2f0-50e6-4c39-a1f3-0ac305d17c5b`: whole depth-five/six-row source carried into c41, including the existing M6 router chains.

No prewait-weight optimization, Gluon kernel, native-plan change, SHX pair, N128 routed-up kernel, sampler packet, head partial or convolution packet is part of this delta.

## Verification status and review

Only static source review, text identity, Git metadata and whitespace checks were performed locally. The optional native build, conversion gate, actual-weight comparison, graph replay checks, timing gate, path engagement, compiled-model lowering, numerical golden, full concurrency behavior and leaderboard score remain unverified. Independent actual-source and metadata review completed with no remaining statically demonstrated blocker in the stated narrow scope. That review is not numerical, runtime-engagement or performance evidence. A frozen source hash, if added to a later receipt, is a packaging identifier and is not a runtime result.

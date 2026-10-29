# Delete source-dead target router suffix chains, retaining the FP32 boundary

Effort: ultra. This implementation turn used GPT 6.1 Sol through Codex, confirmed from the actual native turn context. This is a source-only experiment on the complete c41 baseline. No candidate imports, syntax/AST parsing or execution, setup, builds, compilation, AOT, tests, benchmark, profiler, preflight or GPU/model startup were run locally. Executable checks described below are authored solely for the official remote engine; no pass, activation or speed is claimed.

## Complete baseline and isolated scope

The verified baseline is `c41fc629-8c06-4bf2-8cb8-cad0edea9875`, official score 2.77776232031352, source `cdea0e5bb85ae36999f23c1f9de15de5901b8e45`, whole tree `3f3a3fa2636eb7c71bf871462134b3b18fbc5259`. The independent local reconstruction `4726d88aef2bfa8e04c110015568a9c68a840b30` has that exact whole tree. This experiment starts from the complete reconstruction and keeps the c41 model, MTP, sampler, graph, weight, quantization and protocol stack.

The only production mechanism is deletion of router input-load/FMA chains whose results the existing router already discards. The original target input remains FP32. The original BF16-to-FP32 conversion stays at the same outside-the-opaque-op source location. No A16 input specialization, compressed W12 gate representation, expert work policy, weight prefetch, new native-plan behavior or different sampler is selected.

The intended path is a single-request target verify with M5 or M6, N128 expert logits and K2688, on SM121. The baseline is native MTP depth five, so M6 is a real configured verify shape. Current confidence gating can lower a device verify-row count. The source establishes that this path and control exist; no official artifact reveals the real production distribution of counts or whether this optional kernel is admitted.

## Why the discarded arithmetic is identifiable

In the original `smallk.cu`, the kernel loads `vr` after its PDL wait. It then computes all M independent FP32 dot products per expert. At the final output it selects

`src = min(m, max(vr, 1) - 1)`.

Thus outputs at or past the live prefix already receive the last live row's logits. The source accumulator and partial sums of those suffix rows are not read by any output. This is independent of their values, including non-finite values. For zero or negative counts, row0 remains the selected source; for counts at least M and for the original large sentinel, every row remains its own source.

The new optional kernel computes `live = min(M, max(vr, 1))` from the same single count read. It retains every valid row's original ascending per-lane FMA chain. For m >= live it initializes the same accumulator to positive zero, omits that row's activation loads and FMA chain, then keeps the original complete shared partial writes, fold, barriers, final source selection and output stores.

All weight loads stay at their original source positions. At least row0 is always evaluated, so the original weight stream remains necessary even when the count is zero. The original full-tile weight preload and partial-tile weight load are not moved or masked. The original FP32 and exact W16 weight representations are unchanged.

This is different from proposing to skip supposedly distinct expert weights for inactive rows: those rows already use a live row's routing, as the existing strategy ledger notes. This experiment makes no expert-weight saving claim.

## Exact source boundaries retained

The original FP32 router kernel and all original C exports remain intact. The optional header is derived from that body with a different kernel name, the integer live clamp, conditions at the two activation-load sites and conditions at the two FMA sites. A static text comparison that reverses only these changes recovers the complete original 3,469-character body. This is source evidence, not a compiler or device-bit proof.

The K256 tile progression, 16 per-lane terms, per-expert weight addressing, accumulator initialization, ascending tile fold, ascending lane fold, current output source rule and unique output ownership are retained. No thread exits before a barrier. Every original part entry is still initialized and written; an inactive row's zero partials are not exposed to the output selector.

The producer and fake interfaces are unchanged. `NemotronHMoE.forward`, `GateLinear`, `MoERunner` and the original outer cast are not edited. In `NemotronHForCausalLM` only a model-level post-load hook is added. No forward dtype, tensor layout, alias contract or compiler output boundary is changed by that hook.

The new kernel still may compile differently: dynamic integer predicates, register allocation, load speculation, instruction issue and predication are controlled by the remote compiler. Source-level work deletion does not prove that fewer physical memory transactions or instructions will be observed. Any numerical difference in the official complete golden remains a failure regardless of source reasoning.

## Existing device control and production qualification

The original native-MTP depth-gate end publishes kept drafts plus one into the existing verify-row buffer, with the existing all-rows sentinel and optional emit-budget clamp. The original `live_verify_rows` refusal when a captured target gate missed the buffer is retained. The new code does not publish, replace, reset or change the lifetime of this buffer.

The new dispatcher is reached only after the original smallk metadata checks and row-buffer handling. It additionally requires the target `row_mask` flag, a non-None contiguous CUDA int32 shape-(1,) count on the same device, M5 or M6, contiguous FP32 input of width2688 and exact contiguous FP32 gate shape128x2688. It requires SM121, the current device, the original row-count entry and the optional suffix ABI. MTP gates with row_mask false, absent control buffers, larger calls and unsupported layouts keep the original dispatcher.

Production is metadata and prior-admission lookup only. It does not read a device scalar on the CPU, run private tests or create/capture private graphs. The count remains a current device value consumed by the kernel after its original PDL wait. A changed count between graph replays therefore controls that replay's live prefix, rather than becoming a Python capture constant.

Cache keys include actual weight identity, pointer/version, exact W16 shadow identity/pointer/version, device, M, router grouping and PDL. Inference tensors without version counters are refused before reading _version; this also preserves fallback if an unsupported configuration first creates its baseline shadow during inference instead of post-load preparation. Weak identities reject address reuse without adding a permanent model-weight or graph-pool owner. No request inputs, logits, token choices, graph results or state snapshots are cached.

The original shared-expert stream fork/join is untouched. The production kernel runs on the original current main stream with the original launch geometry. It adds no production graph, stream, event, global scratch, counter, allocation beyond the original fresh output, or mutable state. The original count producer and wait are the same synchronization proof required by the baseline.

## Model post-load admission

The existing model-loader utility invokes a model-level post-load hook after quant, attention and HPC weight processing. The new hook uses that final point, after the actual gate weights have been finalized and before normal model compilation/capture. It does not move prefill or request computation into startup.

The hook is restricted to the existing target early-router blocks in an immutable resident configuration: TP/PP/DP/prefill-CP/decode-CP all one, uni executor, one sequence, no ubatching, no EPLB, no LoRA, no weight transfer, no KV transfer, and no CPU/prefetch weight offload. Actual CUDA weight and ABI eligibility is checked again by the helper.

A false numerical or speed admission leaves no positive cache entry and keeps the original kernel. Missing optional symbols or an explicitly disabled feature also keep the original kernel. General concurrent serving or weight/configuration mutation after capture is not supported by this fixed-engine experiment; a captured graph does not reinterpret new Python decisions.

## Authored private numerical checks, official remote only

Each actual gate and M uses separate private FP32 input/output buffers and separate int32 count buffers for the old and new kernels. Output buffers have different poisons and explicit FP32 left/right guards. Explicit dtypes are required because the loader can run with BF16 as the default dtype.

The current weight policy is used for both paths: the original exact W16 shadow if present, otherwise the original FP32 gate. There is no new conversion or format. The original reference is called directly through `router_gemm`, so it cannot recursively dispatch to the optional suffix path.

Counts include INT32_MIN, -1, zero, every prefix1..M, M+1, VROWS_ALL and INT32_MAX. These cover the negative/zero, partial and all-valid semantic regions and both signed boundaries. Inputs cover two finite scales, signed zeros and FP32 raw-word edge patterns containing subnormals, maximum finite values, infinities and quiet/signaling NaNs of both signs.

Every comparison checks the raw bits of the entire Mx128 output, not only the live rows. Thus the baseline's duplicated suffix outputs must also match. Guards must keep their own distinct original values. Each private count is compared against an independent expected count, and input, weight and shadow bits are checked for unintended mutation. No request data participates.

## Graph replay checks and private owner lifetime

Old and new private graphs each record copies from the same private input source and count source into their own current input/count buffers before their router. Replays alternate increasing, decreasing, negative, zero, sentinel and boundary counts and change the input pattern, instead of replaying an unchanged fixture. Outputs, guards, sources and copied current buffers are checked again.

Private work uses one nondefault stream that waits for the loading stream. Every Tensor and CUDAGraph remains strongly owned in the same function frame until a finally block synchronizes that private stream. The loading stream also rejoins in an outer finally. This cleanup is attempted before owners unwind on capture, replay, event or comparison exceptions.

No production state is modified by private checks. A normal mismatch returns false after private work drains. Native CUDA, capture, replay or synchronization errors propagate; they are not swallowed and retried on production state. An unrecoverable CUDA error may prevent a successful drain and abort startup. It is not described as a safe numerical-refusal fallback.

## Authored timing admission and its limits

Private kernel-only graphs contain eight reference or candidate calls. Neither timing path includes or receives credit for removing the unchanged external cast. Inputs are already FP32 in both cases.

The rule samples every live count1..M and the all-rows sentinel, under warm and 64-MiB-pressure conditions, with five alternating old/new orderings. The authored criterion is median candidate/reference below0.995 and the maximum observed ratio below1.01. This count mixture is an explicit admission rule, not an observed or assumed production distribution. Full guards and input/weight bits are checked again after timing.

The rule may reject every gate due to noise or full-prefix predicate overhead. Conversely, passing a pure-kernel gate does not establish a win while shared and routed experts run concurrently, and does not establish whole-model improvement. No actual private timing, admission result or path-hit evidence has been observed.

## Build and fallback behavior

The original smallk single translation unit includes the optional header only without SMALLK_NO_SUFFIX. Both added exports and the optional init are guarded by the same macro. Official setup first attempts the combined build, then without suffix, then without W16, then without both. Each attempt removes a stale/partial library and checks pipeline status and a nonempty output.

A suffix compilation failure can therefore produce a library containing the baseline only. Missing optional exports are detected before candidate dispatch. SMALLK_ROUTER_SUFFIX=0 disables the optional path. No failed production launch is retried, including inside capture. A complete official pass after build or admission fallback would not prove that the new kernel was compiled, selected or executed.

The new ctypes call has eleven ordered arguments: FP32 input pointer, row stride, M, selected weight pointer, FP32 output pointer, output row stride, R, W16 flag, PDL flag, count pointer and CUDA stream. The C wrapper rejects missing pointers and unsupported M/strides/grouping before launching. Original public ABIs are unchanged.

## Source-level amount and unknown effect

Let L=clamp(vr,1,M). The deleted FMA count per target gate is (M-L)*128*2688. Deleted FP32 activation-load requests per gate are (M-L)*(128/R)*2688 values, each four bytes. No weight-load byte is deleted.

For M6, L2 and R1 over23 target MoE layers, the source totals are31,653,888 fewer FMA operations and126,615,552 bytes of repeated FP32 activation requests. This is a conditional example and an upper bound for that count, not a measured common case. The original FP32 cast still materializes every row. Input reuse can make most router reads L1/L2 hits, and the gate overlaps shared expert work. There is no HBM, bandwidth, launch-saving, latency or score prediction.

## Provenance and earlier results

- `c41fc629-8c06-4bf2-8cb8-cad0edea9875`: complete verified baseline, retaining its entire source and VENDOR credit.
- `100549d4-51c4-4f56-b3ca-2604731ab8a4`: original smallk router FMA/fold chain, stream schedule and source structure.
- `ca5c19dd-da37-4328-880a-152f8c7b3651`: original confidence stop, device verify-count and forced-rejection protocol. This experiment consumes that existing count without changing it.
- `0852152c-9025-499d-83bd-684cf825232e`: exact W16 shadow/weight-read policy, retained.
- `f38ce1ad-87fa-4cfb-8a0c-ac46587810f7`: inherited M5 router chains.
- `eb48e2f0-50e6-4c39-a1f3-0ac305d17c5b`: inherited complete depth-five/M6 implementation through c41.
- `110cff4b-d86d-4fe9-ae30-f6ce550d005f`: material reuse of the just-reviewed private post-load admission, weak identity/version cache, same-frame stream drain and optional build-fallback design. Its complete official score2.74300427046762 is1.2513% below c41 and the whole source is excluded. Admission, production engagement and cause are unknown. No A16 input, kernel, changed outer cast or whole bundle is selected here.

Historical X16 context also includes `4ef72848-19b4-492e-a948-e9d44a94bef0` and `528c6b75-628b-427b-9dab-07f2e7a9a3c1`, both official whole golden token mismatches with no score and unisolated causes. They are not positive evidence or baselines. The current outer cast is deliberately retained, but this is not claimed to diagnose or fix either historical failure.

The W12 router-weight idea, pkslots raw-pipeline `cdf8aecf-594a-48ec-bc02-d03a379ce1c2`, SHX pair and N128 w13 experiments are not part of this delta. No lower or failed whole is adopted.

## Current status

Independent actual-source and metadata review found no remaining statically demonstrated blocker in the stated scope. The final inference-tensor metadata refusal was separately reviewed before freezing. Local work was text/Git/file-metadata inspection only. Compilation, resource feasibility, predicate realization, numerical equality, graph behavior, admission, real count distribution, engagement, concurrency and full score remain unverified. Only the official whole result can decide whether to retain this experiment under the user's0.1% regression rule.

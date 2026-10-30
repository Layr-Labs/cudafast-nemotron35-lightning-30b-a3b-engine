# Target lm_head: share one activation ring between the original two N64 halves

Effort: ultra

Implementation authoring used GPT 6.1 Sol with Codex. The actual native turn context for this batch reports `gpt-6.1-sol` and `ultra`. This document describes a static, unverified candidate. No local candidate import, syntax/AST check, build, setup, CUDA execution, model run, test, benchmark, profiler or preflight was performed. The validation code described below is intended to execute only inside official remote validation. There is no claimed score or observed production hit for this candidate.

## Verified baseline and experiment boundary

The complete baseline is `c41fc629-8c06-4bf2-8cb8-cad0edea9875`, official score **2.77776232031352**, official source `cdea0e5bb85ae36999f23c1f9de15de5901b8e45`, whole tree `3f3a3fa2636eb7c71bf871462134b3b18fbc5259`. Local parent `4726d88aef2bfa8e04c110015568a9c68a840b30` preserves that whole tree. The proposed patch is an independent overlay on that baseline. The router W12, suffix, A16, Humming N128 and SHX pair whole bundles are not used as baselines.

A separate packaging receipt must attach the candidate source hash and independent actual-code review before any official submission. This in-tree method document avoids a self-referential candidate commit hash and does not authorize an upload. The current benchmark manifest permits `vllm`, `harness` and `mtp-head.manifest.json`; every candidate change is inside those paths. Trusted fixture, benchmark envelope, scoring and golden verification files remain unchanged.

The experiment changes only the actual prepared **MarlinX target lm_head** at `(N,K)=(131072,2688)`, with **M5 or M6**, ring depth eight, existing unit-major scale storage, no bias, and a fixed sm_121 device with 48 SMs. It does not select a provider. Other providers, absence of the existing SU pack, unsupported shapes/layouts, compiled tracing, refused admission or changed owners/configuration use the original path. M1 through M4, M7/M8 and prefill retain their existing implementations. The native MTP head and its screening/packed projections are unchanged. All logits remain materialized as before; no argmax, sampler, partial-logit protocol or draft policy changes.

## Source-level mechanism

The original `harness/kernels/marlinx/csrc/marlinx.cu` assigns one original four-warp CTA to a 64-column half. Its `col=unit>>1` and `h=unit&1` split each original 128-column tile into two CTAs. `column_first_cut` depends only on `col`. Both halves initialize the same `Cursor`, traverse the same descending segment order and ascending K sequence inside each segment, and obtain exactly the same `kb` for each warp and ring position.

The activation address is `A + tok*lda` plus the original lane offset; it has no dependence on `h`. Therefore the two halves repeat the same activation register-ring loads. Their weight addresses and scales differ by half and are genuinely different data; those loads cannot be shared.

The new optional `csrc/head_pair.cuh` assigns one original four-warp CTA to the complete original `col`. It contains two separate `rb/rs/acc/R` chains and separate original double-buffered reduction storage, but one `ra` activation ring. Every original K position feeds the same two activation words to both independent half-column MMA chains. The grid changes from 2048 to 1024 CTAs for this head. The original `N/128`, `k_tiles`, `sms`, `part1_cols` and `iters` remain the stripe calculation inputs; the reduced new CTA count is deliberately **not** substituted into the original stripe formula.

The existing SU layout is `[N/64][K/16][64]` bytes. Each new CTA copies two adjacent existing records, one per original half, into shared memory. There is no second persistent scale pack and no new weight representation. Each half has its own B and scale operands. The new path calls the existing `block_mma<false>`, `close_segment` and `write_out<false>` helpers, rather than introducing another dequant, MMA instruction, reduction expression or BF16 rounding helper.

For each half, the intended arithmetic sequence stays four original warp-row chains, each original `mma.sync` operand order, the original `acc0 + (acc1 + (acc3 + acc2))` segment reduction, bottom-segment-first FP32 fold, global FP32 scale multiplication and BF16 RN store. Interleaving two independent chains can still change compiler allocation/scheduling. Source preservation is not device-bit proof; all output elements must pass the real-weight remote comparison before publication of the entry.

## What the byte arithmetic does and does not mean

The deleted source requests are one duplicate BF16 A traversal for each of the 1024 original 128-column tiles. The formula is `1024 * M * 2688 * 2` bytes:

| Call | Duplicate A source-request upper bound removed |
|---|---:|
| M5 | 27,525,120 B = 26.25 MiB |
| M6 | 33,030,144 B = 31.5 MiB |

This is logical source-request arithmetic. It is **not** physical HBM savings, a measured transaction count, or an expected latency reduction. The entire M6 input is only 32,256 bytes. Original B loads use L1-no-allocate hints; A may already be well cached. Removing repeated loads can save instructions/cache traffic while saving very little external memory bandwidth.

The **176,160,768 bytes of packed FP4 weights** and **22,020,096 bytes of scales** are unchanged. Every actual weight, scale, output and sampling result remains computed. The full M6 BF16 output is still 1,572,864 bytes, and its existing downstream reads remain. This candidate is not a weight-traffic codec and does not claim the savings of the retired W12 experiment.

## Resource risk and optional build boundary

Two original reduction buffers occupy `2 * 12288 = 24576` bytes. Two existing SU halves occupy `2 * 10752 = 21504` bytes, for **46,080 bytes** before compiler-reported static additions/alignment. The native resource query checks actual compiled function attributes, zero local-memory size, total static plus dynamic shared memory no larger than 48 KiB, sufficient block capacity, and at least one resident CTA per SM.

P8 explicit array state grows approximately from 96 to 176 32-bit values per thread before pointers and temporaries. This is not a compiled register count. It may produce spills, reduce active CTAs, increase instruction latency or make the kernel unattractive. The resource gate is followed by a private speed gate. Neither establishes a complete-model benefit.

The optional template is protected by `MARLINX_NO_HEAD_PAIR`; its native symbols are independent of the original MarlinX initialization. Official setup first tries the complete library. If that compilation fails it deletes the partial output and recompiles with `MARLINX_NO_HEAD_PAIR`, preserving original PDL. If that also fails, it deletes the partial output and builds the original no-PDL library. Build status, rather than merely an existing output file, determines success. A missing pair ABI declines the feature. A complete official score after build fallback would not prove that the new kernel ran.

No local compilation was attempted. The optional native template, launch API, register usage and resource query may fail in the pinned official image. Compilation fallback is separate from runtime failure: once production launches the new grid, a nonzero return raises; there is no retry against a destination that may already have been written.

## Admission and original fallback

`NemotronHForCausalLM.process_weights_after_loading` is a final model hook. The existing model-loader utility invokes it after per-module quantization, attention and HPC postprocessing. It examines only `self.lm_head`; it does not rerun the existing Marlin pack or SU preparation. The actual kernel class must be exactly `MarlinNvFp4LinearKernel`, `_marlinx_ok` must already be true, and actual current B/S/GS/SU owners must match their expected dtypes, shapes, strides, devices and layout. Unsupported providers decline without changing their selection.

Scope additionally requires one worker, TP/PP/DP/context parallel size one, no expert parallel/EPLB/ubatching, one scheduler sequence, BF16 Nemotron-H configuration, no LoRA, no weight transfer and no CPU/prefetch offload. Only versioned tensors are accepted. Inference tensors without version counters are refused; that conservative rule may prevent all actual admissions and no hit is claimed. Existing SU bytes are checked against the original scale tensor's byte-permutation view, with no second full persistent copy.

Each actual weight/scale/SU identity, storage address, byte size, offset, shape, stride, dtype, device and version is recorded. The entry retains weak tensor references and is owned by the head; it does not add another persistent full-weight owner. Configuration and library identity, SM count and the admitted PDL state are also retained. Production only looks up this entry. It performs no qualification, resource query, GPU scalar read, synchronization or weight allocation. A declined lookup reaches the original `gemm` call and its existing fresh output. An admitted call makes one fresh full output allocation, exactly the same output ownership pattern as the old call.

Post-load qualification is one-shot. A later post-load invocation invalidates the optional entry and refuses requalification. Hot weight replacement while already-captured CUDA graphs are live, general concurrent model mutation and offload are not supported claims. The intended engine owns immutable checkpoint weights for its lifetime. Every production call checks live metadata; existing captured graphs still depend on that immutable-owner contract.

The original `_pdl_active` flag is not modified by private reference calls. Both original and new native operations receive explicit PDL modes, avoiding the original eager auto-retry wrapper during qualification. Production requires the mode admitted at startup; a later global PDL change declines the pair path. Unsupported or rejected calls do not change the original global readiness state.

## Official-remote-only qualification program

The code authors the following private startup qualification; **none of these checks has run locally or supplied an observed result for this candidate**:

1. Query actual compiled pair attributes for the active PDL mode, and also ordinary mode when PDL is active. Refuse spills, excessive shared memory, unsupported thread capacity or no resident CTA.
2. Compare actual existing SU bytes with the original scale view. Do not clone the full FP4 checkpoint matrix or make a second SU pack.
3. For M5 and M6, compute the original SU ABI and new ABI using the same actual immutable checkpoint B/S/GS/SU. Compare every output element as raw int16 BF16 bits, including all 131072 columns and columns on both sides of internal stripe cuts. Check separate output boundary poison words and unchanged private input words.
4. Input sets include small/normal/large finite activations, zeros, negative zero, BF16 subnormals, signed rows, tail zeros, infinities, largest finite values and distinct NaN payloads. Any bit difference refuses the complete optional entry; NaN equality is not weakened to numerical closeness.
5. Capture two independent private-stream graphs for each row count. Each graph contains an actual input copy producer followed directly by the new head grid. Replay with changed source contents, then run the original reference and compare full raw outputs plus guards and copied input. Placing the reference after replay prevents the reference from serializing away the candidate's input-producer edge.
6. Run warm and 64-MiB cache-pressure ABBA kernel timings on the actual immutable weights, separately for M5/M6. The pressure buffer is touched before each measured kernel; event timing excludes that touch. Warm new median must be at least 0.5% faster, and pressure median must not regress. Both row counts must pass. These private timings are noisy admission evidence, not official whole-model timings.

All graph objects, tensors, output storage and streams are held in the same private probe frame. Its nested `finally` attempts to synchronize both streams even if the first drain raises. The successful-drain marker is set only after both return. A safe private mismatch, unsupported capture, OOM or other private failure may decline only after both drains completed; failure to drain propagates and does not continue into the production fallback. CUDA launch or capture faults may therefore fail official boot. This is preferable to retrying after uncertain writes or freeing in-flight private owners.

Only private synthetic activations and outputs are used for admission; no prompt, model forward, KV cache, SSM state or scored prefill is computed or retained. After qualification the graphs and scratch leave scope. The entry contains only immutable owner metadata and host evidence. This is not cross-request or cross-phase result reuse.

## Stream and output ownership

Each new grid uses the caller's current stream. There is no persistent activation, output or reduction workspace shared between calls. Reductions and SU stage storage are CTA-local. Every original output column has exactly one writer: `(col,h=0)` writes its original 64 features and `(col,h=1)` writes the other64; `write_out` retains its original row mask and N stride. No output is borrowed from a prior frame.

For PDL, only immutable B/SU loads may run before `griddepcontrol.wait`. All A-ring reads are textually below the memory-clobbering wait and use the existing coherent activation helper. The new CTA signals dependents once after both half chains finish; its original two BF16 epilogues remain after that signal, as in the original PDL protocol. Downstream kernels retain their original stream dependencies. The optional helper adds no new model-level stream fork, join, event, or sampler ticket.

Full BF16 logits preserve the original downstream interface. The logits processor, grammar/logprob paths, first-index ties and sampler are not replaced. The qualification insists on raw output equality because preserving source formulas alone does not establish NaN payloads, compiler scheduling or device rounding.

## Provenance and historical results

Full API declarations are in `HEAD_PAIR_DEPENDENCIES.json`. Material sources are:

| Full submission UUID | Material used |
|---|---|
| `c41fc629-8c06-4bf2-8cb8-cad0edea9875` | Complete verified baseline and all its inherited functionality |
| `100549d4-51c4-4f56-b3ca-2604731ab8a4` | Original MarlinX ring/Cursor/MMA/fold/BF16 helper semantics |
| `c3e16bbb-c76b-42e2-84d6-3a7260e3d478` | Existing SU unit-major scale layout and staging |
| `3d51cab6-7079-4fb2-98e4-f2a238077897` | Material carrier of original MarlinX PDL preload/wait/signal behavior |
| `e7f42e20-4e1b-4a8e-80b5-8c2c49369615` | Adjacent-N A-sharing idea and negative resource evidence; no Humming code port |
| `110cff4b-d86d-4fe9-ae30-f6ce550d005f` | Final-postload, versioned weak-owner and private cleanup design only |
| `77f74985-a845-4963-bf29-b860e8daa7ef` | Raw-output/boundary/changed-graph qualification principles only |
| `8e0a3de6-17c7-4667-8804-ee17507eb41a` | Private two-stream owner/drain and fixed-real-weight admission lessons only |

The earlier Humming adjacent-N experiment used a natural eight-warp N128 prefill w13 tile, PFUP routing and NVSC tails. This new head experiment uses the existing four-warp head layout, two original independent chain sets, a shared register A ring, exact divisible N and no Humming scheduler or routed metadata. The earlier idea remains material even though its whole result regressed. It is not represented as validated positive evidence.

Historical complete whole results versus c41: Humming N128 `e7f42e20-4e1b-4a8e-80b5-8c2c49369615` **2.75254912647548 (-0.90768%)**; A16 `110cff4b-d86d-4fe9-ae30-f6ce550d005f` **2.74300427046762 (-1.2513%)**; suffix `77f74985-a845-4963-bf29-b860e8daa7ef` **2.75184607067041 (-0.9330%)**; W12 `8e0a3de6-17c7-4667-8804-ee17507eb41a` **2.76663609402143 (-0.40055%)**. Those wholes are retired and are not adopted. Their optional path activation and isolated causes are unknown. Reusing a validation/lifecycle lesson does not reuse their model arithmetic or claim that a prior regression has been fixed.

The pending description `cac45ecb-6d5f-4ba9-95d1-4761499100b1` was read for exclusion. Its confidence0.51/minimum-cycle2 defaults already exist in c41; the remaining described items are existing conditional-step, attention PDL and shared-up lean switches. No code or idea from that pending is added here, and its unverified author claims are not a benchmark result for this experiment.

## Reproduction and remaining evidence

An official worker stages the complete candidate editable tree and runs the benchmark's normal setup. The modified `harness/kernels/marlinx/build.sh` builds the optional template or its original-library fallback. Model loading invokes the final private head qualification. The official full correctness, token-stream and paired score result remains the acceptance authority. Runtime investigation must establish whether this exact path actually activated; a passing whole score alone cannot establish activation, register counts, or isolated benefit.

For archival source review only, the parent is the complete verified c41 tree. The intended diff is the optional head header/ABI/helper, original helper lookup, final model post-load hook, optional build fallback and public provenance files. A later source receipt should record the exact commit/tree and file hashes, created with Git hooks and signing disabled. No candidate commands were executed to construct this draft. Local checks were limited to source reads/edits, `git diff --check`, file metadata and Git hash/receipt operations.

Remaining uncertainty includes actual provider and SU eligibility, inference-tensor refusal, compiled function resource limits, native build/CUDA-graph support, NaN payload equality, private timing noise, Python metadata-key overhead, real M5/M6 distribution, allocator effects after private capture, changes in kernel occupancy/PDL overlap, and the official complete-model score. No speedup estimate follows from the 31.5-MiB logical source-request bound.

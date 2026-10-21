# Engineering log — Nemotron 3.5 Lightning

The organizer's engineering record for this track: how it was derived, what was
measured on the Spark cluster, and what was changed as a result.

This repository keeps the rule set of
`Layr-Labs/cudafast-qwen38-125b-a6b-engine` @
`5707d4f23362a7208483028e6d34519267266729`. It swaps the engine and the model.

## 1. What carried over unchanged

- **Scoring:** composite `prefill^0.25 * decode^0.75`, per-box paired
  baseline, floors 0.95 and ceiling 5.0, batch size 1, 128 decode steps, the
  10% token-tolerance gate, and depths 1 to 6. (Now: 3 pairs, one per live
  golden, the score the mean of the per-pair composites, and a 496-token chat
  seed. The 2-pair, 4096-token setup of sections 7.7 to 7.11 is history; see
  section 7.14.)
- **The surface:** Engine Protocol v1 and the adapter's verb logic, counters
  and fail-closed checks (`harness/protocol-adapter/src/{protocol,adapter,engine,mock,tests,resident}.rs`).
- **The resident:** the resident socket wire and the per-leg
  `serve-up.sh --boot/--stop` contract with benchd.
- **The pipeline:** the trusted pipeline: preflight, measure-and-score,
  calibrate-box, stage-baseline-workspace, fetch-benchd, fetch-goldens, the
  static review and the manifest lint.
- **The job:** the ranked job's shape, holding no credential, taking the GPU
  lock and waiting for quiescence.

## 2. What changed, and why

| Area | Reference | This track | Why |
|---|---|---|---|
| Engine | vendored ds4 (C/CUDA, GGUF only) | vendored vLLM v0.27.1 Python source over the pinned image's compiled build | ds4 has no Mamba-2, no ModelOpt NVFP4 and no `nemotron_h`; vLLM runs all three and is NVIDIA's recommended stack on GB10 |
| Weights | a GGUF conversion | the official ModelOpt safetensors, 60 files pinned | traceable to NVIDIA's checkpoint; no conversion step to trust |
| MTP head | a separate Q8_0 GGUF | `mtp.*` inside shard 52, BF16 | that is how NVIDIA ships it |
| Resident | `ds4-resident` (C) | `harness/vllm_resident` (Python), same wire | the Rust client is unchanged |
| Backend Rust | `ds4_backend.rs` + FFI | `session_backend.rs`, resident only | the in-process FFI path had no vLLM equivalent |
| Isolation | host process | pinned container, `--network none`, read-only snapshot and tree, fresh caches per leg | Python engine code can reach anything the host allows |
| Editable paths | `ds4`, `harness`, `mtp-head`, `mtp-head.manifest.json` | `vllm`, `harness`, `mtp-head.manifest.json` | `mtp-head/` held no weights and was never meant to be editable |
| Custom kernels | inside `ds4/` | `harness/kernels/` | inside benchd's harness-hash roster, so kernel edits are in the sealed provenance |
| KV backend | `contiguous` | `vllm-paged` | vLLM is paged; the benchd branch must accept the value |
| Build cache | `tools/ds4/build-cache.sh` | none | the build is cargo plus a file copy, and a cache would only add risk |
| Box runner | macOS/launchd | Linux/systemd | the boxes are DGX Sparks |
| Launch declaration | MTP depth 1 | serial | the reference's own contract calls the launch candidate serial; the null control must be about 1.000 |

## 3. Reference defects fixed here

These were in the reference and must not come back. `tools/check-doc-consistency.py`
holds the first three in CI.

1. **Editable paths.** TASK.md said there were 3 editable paths and that the
   engine was read-only, while `benchmark.json` listed 4, including the engine.
2. **Head re-quantization.** TASK.md said the head could be re-quantized, and
   the contract and `mtp-head/README.md` said it could not.
3. **Depth coverage.** The preflight and `serve_configuration.depths` covered
   only depths 1 to 3, while 1 to 6 were permitted and had oracles.
4. **Category.** `category: "swift"`.
5. **The launch candidate.** The contract called it serial, but the shipped
   declaration enabled MTP.
6. **Stale docs.** The contract described a `results.json` emitter; benchd
   seals `score.json` itself.
7. **A phantom sealed field.** The docs cited `effective_mean_draft_len`, which
   benchd does not seal.
8. **Section references.** Code comments cited contract section 11.4(c), which
   did not exist.
9. **Stray markers.** The adapter's `lib.rs` carried stray `__REDRAW_MARK_*`
   constants.
10. **Macintosh leftovers.** macOS runner docs and a Seatbelt profile on a Linux
    track.
11. **Silent socket-path overflow.** A long workspace path failed only at the
    resident's bind(). `serve-up.sh` now refuses by name.
12. **Declaration cap.** The pinned head (2.67 GB) was larger than the 2 GiB
    declaration cap. The cap is now 3 GiB, and the head's own size is recorded
    in the fixture.

## 4. Decisions a reviewer should know about

- **Activations are part of the frozen format.** The checkpoint declares its
  experts W4A16 (`hf_quant_config.json`: `W4A16_NVFP4`), so FP4 activation
  quantization is a format change and is prohibited. On GB10, vLLM runs these
  layers through Marlin W4A16 kernels; faster W4A16 kernels are the intended
  kernel lever.
- **The MTP head may not be re-quantized** (`mtp_head.requantization_permitted:
  false`). It is BF16, about 2.5 GiB. A re-quantized head is a large decode
  lever, and allowing it would be a deliberate rules change.
- **Prefix caching is off in the reference engine configuration,** and reuse
  across phases is prohibited by rule (contract section 8). As on the reference
  track, the engine is editable, so this is a rule plus review, not a runtime
  gate.
- **benchd's env var names are kept:** `MLXFAST_QWEN_MTP_TRACK_ID` and
  `DS4_RESIDENT_SOCKET`. benchd reads and injects them under those names.

## 5. Open items (external)

State on 2026-10-20 (section 7.14 has the record).

1. **benchd — Layr-Labs bench-dev `main`.** `tools/fetch-benchd.sh` pins dist
   commit `43a9a1642cc22c3fb2f71b9c707699cd3a9602c7` (bench pull request 425),
   built from source `b65af98c2a060df12f2ced849c68528b825fecdb`, and refuses
   any other `source_commit`. The source adds the stricter sealing of bench
   pull requests 421 to 424 to the cool gate and prefill median of section
   7.13. A no-cache rebuild matches the dist byte for byte
   ([reproduce.md, section 1](reproduce.md#1-rebuild-benchd-from-source)).
2. **Goldens — recorded, pinned and staged.** The live pool is `pool-9`,
   `pool-10` and `pool-11`: three 496-token chat-template coding prompts, each
   a serial golden plus six per-depth oracles, 21 files pinned in the fixture
   (PR #33). They are staged on Spark box 1 to Spark box 4 and on the jump box. R2 is
   not used: the live pool is kept on the jump box, and the held-out pool and
   the prompt texts in the organizer archive (see
   docs/pool-refresh-and-final-rescore.md, "Where the goldens live").
3. **`baseline_reference_commit` — pinned** at
   `5be35aa395283b14d1887cec70a37d6e6bdeac9c` (2026-10-07, the resident
   phase-close fix), in this repository.
4. **Timing noise — addressed.** The prefill phase is one warm-up pass followed
   by the median of five timed passes, and the gate is 60 °C (section 7.7).
   Every box is calibrated with a CV limit of 2% (`calibration_max_cv_percent`,
   PR #34); the measured CVs are all under 1% (section 7.14).
5. **GB10 temperature — resolved.** `nvidia-smi` reports a GPU temperature on
   all eight Sparks, and the cool gate is 60 °C (section 7.5).
6. **Logit ties — open.** The current benchd has no timed reference replay, so
   the timed window is an exact match (sections 7.8 and 7.13).

## 6. Verification on the Spark cluster (before arming)

1. **Bring-up, per box:**
   - `tools/box-runner/install.sh` and the systemd unit;
   - `docker pull` of the image by digest;
   - `./tools/fetch-benchd.sh && ./setup.sh`, with every pin passing.
2. **Engine sanity:** boot a serial resident and check that the greedy output
   on the public prompt is coherent and agrees with an HF transformers
   reference. Boot at depths 1 to 6 and check that every depth runs and
   `spec_acceptance_rate` is sane.
3. **Protocol:**
   - `tools/three-flows-dry-run.sh` against the real resident
     (`SERVE_UP_LAUNCHER` unset);
   - `./tools/nemotron35-measure-and-score.sh --preflight-only`.
4. **Determinism:**
   - serial twice gives identical tokens;
   - serial against mtp1 to mtp6 stays inside the 10% budget;
   - the same holds across boxes.
5. **Calibration:** `tools/calibrate-box.sh` on every box, 3 passes, a CV of
   2% or less. The fixture field `calibration_max_cv_percent` sets this limit.
6. **Null control:** the reference against itself scores about 1.000 on every
   box. The same tree with MTP depth 3 declared scores above 1.
7. **Negative tests:** each of these must be refused:
   - an edit outside `editablePaths` (static review);
   - an over-budget surface;
   - depth 7;
   - `"source": "remote"`;
   - a resident that tries the network;
   - a preset `VLLM_*`, `HF_*` or `RESIDENT_*`;
   - a candidate that fails the gate or regresses more than 5%.
8. **End to end:** a `workflow_dispatch` of `benchmark.yml` on one runner, then
   on the fleet, with `score.json` sealed and printed to the job log. Arm only
   then.

## 7. Findings

### 7.1 Engine bring-up on GB10 (2026-09-26, Sparks 5–8)

- **First boot.** The first real `VllmBackend` boot worked as written: about
  220 s for load, FlashInfer autotune and CUDA graph capture, and about 290 s
  with MTP. The resident loads 20.1 GiB of weights and gives the KV cache
  32 GiB inside the 0.45 reservation.
- **Greedy output.** Greedy output on the public long-copy prompt is coherent.
  The model copies the passage.
- **MTP acceptance.** It is high on this prompt. The mean committed tokens per
  verify cycle were 1.97, 2.91, 3.66, 4.57, 5.57 and 6.40 for depths 1 to 6.

### 7.2 Determinism: the stock kernels are not reproducible (fixed)

- **The symptom.** With vLLM's default kernels, the same 1024-token prefill
  gave different logits run to run, within one resident and across boots. The
  seed logits swung by up to 0.6. Greedy streams forked a few tokens in, which
  would make the serial control leg fail its own golden at random.
- **Bisection.** Per-layer output hashes across identical prefills located
  the first difference at `model.layers.29.mixer.experts`. Every earlier layer
  was bit-identical, including 13 earlier MoE layers. The router output and the
  shared experts were also identical, and so were the routing decisions
  (expert ids and weights).
- **The cause.** The Marlin NVFP4 MoE gives different outputs when it is
  replayed on one fixed captured input. The checkpoint's FP8-quantized
  activations then amplify a last-bit difference into a changed argmax.
- **What did not help:** eager mode (no compile, no CUDA graphs), Triton
  attention, moving the FP8 GEMMs off FlashInfer, and the humming MoE without
  its batch-invariant flag. The global `VLLM_BATCH_INVARIANT` cannot start at
  all: it forces the W4A16 linears onto CUTLASS W4A4.
- **The fix, now the reference engine:**
  - the humming W4A16 MoE (`moe_backend="humming"`) with its batch-invariant
    kernels, via a vendored-vLLM patch that adds `VLLM_HUMMING_BATCH_INVARIANT`;
  - the Mamba FP8 projections on CUTLASS (`VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel`);
  - a second vendored patch that gives humming a 273 GB/s bandwidth fallback,
    because GB10's NVML exposes no memory clock.
- **Two more sources of box-to-box drift, both pinned:**
  - *The KV cache size.* vLLM profiles it at boot, and the profile differs box
    to box (the CUDA graph memory estimate varies). A different cache geometry
    changed the numerics. The fixture now pins
    `RESIDENT_KV_CACHE_MEMORY_BYTES` = 4 GiB, which `backends.py` passes as
    `kv_cache_memory_bytes`.
  - *Triton autotune of the Mamba-2 SSD kernels.* It picked different tile
    configs on different boots. A third vendored patch,
    `layers/mamba/ops/ssd_pinned_configs.py`, keeps exactly one config for
    each of the five SSD kernels and refuses to start if that config is
    missing. `VLLM_MAMBA_SSD_AUTOTUNE=1` restores autotune for experiments.
- **Result.** On all four Sparks, 10 of 10 serial runs and 10 of 10 MTP-3 runs
  give one stream. The stream is bit-identical ACROSS the four boxes, and the
  MTP-3 stream equals the serial stream exactly (0 of 129 tokens differ).
- **The cost.** Serial decode is about 4.5% slower than the nondeterministic
  Marlin path (78.1 against 81.9 tok/s).
- **What it means for participants.** A faster kernel that breaks
  reproducibility fails the gate at random. `tools/determinism-probe.py`
  checks it on the box.

### 7.3 Timing noise

- **The cold first prefill.** The first 1024-token prefill after a boot was
  about 2.8× slower than the rest, from lazy JIT and autotune. benchd times the
  first prefill of every leg and the fixture declares 0 warm-up runs, so the
  resident now runs an input-independent warm-up before it reports ready: three
  rounds of a synthetic 1024-token prefill plus 64 decode steps.
- **Back-to-back runs in one resident** (client timing, 10–12 runs, three boxes):

  | Phase | Speed | CV |
  |---|---|---|
  | decode | 81.5–81.9 tok/s | 0.40–0.61% |
  | prefill | 5.2–5.4k tok/s | 2.0–2.7% |

- **What drives the prefill spread (profiled).** A torch.profiler run of one
  in-process prefill shows it is GPU-bound. At 1024 tokens, 196 ms of the
  201 ms wall is kernel time, and the humming W4A16 MoE alone is 112 ms. At 64
  tokens the MoE still takes 53 ms, because nearly every expert is touched.
  So a prefill streams most of the NVFP4 expert set from memory: a ~80 ms
  floor at 273 GB/s, plus ~0.12 ms per token. Capturing piecewise CUDA graphs
  up to 1024 tokens changed nothing, which confirms that launch overhead is
  not the cost.
- **Idle makes it worse.** In one resident, back-to-back 1024-token prefills
  run at 203–209 ms. After an idle of 60–240 s they run at 199–239 ms, while
  decode is unchanged (12.4–13.2 ms/token). The likely mechanism is clock
  ramp-up (DVFS) after idle: GB10's NVML exposes no memory clock to confirm
  it. benchd's order for every timed window is an ungated warm-up leg, then
  the cool gate (minutes of idle), then ONE timed prefill. So every measured
  prefill is a post-idle prefill.
- **benchd calibration, 4 passes, first rounds (2026-09-26):**

  | Box | Passes done | Prefill s/tok per pass | Prefill CV | Decode CV |
  |---|---|---|---|---|
  | Spark box 4 | 4/4 | 204.3, 208.0, 195.4, 207.9 µs | 2.9% | ~0.4% |
  | Spark box 8 | 4/4 | 194.8, 214.4, 201.7, 214.6 µs | 4.7% | ~0.8% |
  | Spark box 8 (round 1) | 4/4 | last pass 197.9 µs | 8.4% | — |
  | Spark box 3 | 1/4 | 182.8 µs | — | — |
  | Spark boxes 1, 2 and 7 | 0/4 | cool gate refused pass 1 (section 7.5) | — | — |

  Decode met the fixed maximum for this experiment. Prefill did not, on any
  box. benchd refused the calibration (`CALIBRATION-CV-EXCEEDED`), as it
  should. The current `calibration_max_cv_percent` fixture field sets a 2%
  maximum.
- **Options considered (the gate was not loosened):**
  1. A longer timed prefill, for example a 4096-token seed. The fixed cost
     and the ramp are then a smaller share of the window. This changes
     `seed_tokens` from the reference rule set's 1024, and every golden would
     be re-recorded.
  2. A benchd change to time the prefill as the median of several prefills
     inside the window. This belongs to benchd.
  3. A per-phase engine-side "re-warm" before the timed prefill. It is NOT
     recommended: it runs work outside the timed window to beat the cool
     gate's intent, and participants would copy it.

### 7.4 Goldens (2026-09-26)

- **What was recorded.** `tools/record-goldens.sh --apply` ran independently on
  Spark box 5 and Spark box 6 and wrote 15 files: the 8 timed-pool goldens, the live
  golden at depths 1 to 6 (`botany.mtpN`), and the public golden. All 15 are
  **byte-identical across the two boxes**, and both boxes pinned identical
  fixtures. The set is staged in `~/nemotron35/goldens` on all eight Sparks.
- **Depths 1–3 reproduce serial exactly.** Their goldens are the same bytes as
  the serial `botany` golden.
- **Depths 4–6 diverge from serial.** They diverge early in the decode
  window, and most of the 128 decode tokens differ from serial after that. The divergence
  is deterministic: it is the same on both boxes, and depths 4, 5 and 6 share
  one file. A verify pass of 5 or more tokens evidently takes a kernel path
  whose numerics differ from the 1-token step. The per-depth goldens exist to
  absorb exactly this, so each depth is gated against its own golden. It is
  still an open engine-side improvement: a participant who makes the
  wide-verify path agree with serial loses nothing.

### 7.5 The cool gate on GB10: an idle resident holds the die near 50 °C

- **The symptom.** benchd runs a cool gate (≤ 50 °C on CUDA) before every timed
  phase, in calibration and in the ranked path alike. It aborts when the GPU
  is above 50 °C and has not dropped 0.25 °C in 90 s, after 180 s of waiting.
  In the first two calibration rounds, 4 of 7 boxes were refused on pass 1:
  `GPU is hot and not cooling down`, stalled at 51–52 °C.
- **The cause is the platform, not the engine.** On driver 580.159.03, ANY
  CUDA context holds the GB10 in P0, at about 10.7 W and 2.4 GHz, even with
  zero utilization. A one-element torch tensor does it. The bare box idles in
  P8 at 3.8–4.0 W and 45–46 °C. With a context the die plateaus at 47–49 °C
  on a quiet box, and at 51–52 °C after the boot's heat soak with neighbours
  busy. The loaded vLLM resident's CPU side is idle during the gate (below
  0.5% per thread).
- **Upstream sees it too.** benchd's CUDA runbook records the same thing on the
  Qwen fleet: "the loaded resident held the die at 51 to 52 C". The runbook's
  claim that the ranked path has no gate is out of date: `official_cool_gate`
  gates every ranked phase.
- **What was not done.** The gate was not loosened, clocks were not
  locked, and persistence mode was not changed. The gate and its temperature are benchd's, and clock or
  driver settings are box configuration. Whether a box passes is decided by
  its cooling and ambient. The resolution needs a benchd change, or better
  airflow on the boxes that stall.

### 7.6 Experiment: gate temperature × prefill length (2026-09-26, all 8 Sparks)

- **Method.** `gateprobe.py` (scratch, not in the tree) repeats benchd's timed
  window against one booted serial resident per box:
  1. an ungated warm-up (a prefill of length L, then 128 decode steps);
  2. a wait until the GPU is at or below T °C, with benchd's stall rule
     tracked (180 s floor, 90 s without a 0.25 °C drop);
  3. one timed prefill of L tokens and 128 timed decode steps.

  It runs 4 reps per condition per box: the same sample count as a
  calibration. The limits: timing is client-side, and one resident per box
  means no boot-to-boot variation. So this ranks the options; it does not
  replace a benchd calibration.
- **Results** (per-box CV across 4 reps; a box counts as passing under the
  former gate when both axes are inside it):

  | Gate | Prefill | Prefill CV median / max | Decode CV max | Boxes passing | Gate refusals | Mean prefill |
  |---|---|---|---|---|---|---|
  | 50 °C | 1024 | 2.82% / 4.84% | 1.00% | 0/8 | 18/32 | 208 ms |
  | 50 °C | 2048 | 1.75% / 2.54% | 0.71% | 2/8 | 22/32 | 305 ms |
  | 50 °C | 4096 | 0.89% / 1.69% | 1.00% | 5/8 | 20/32 | 501 ms |
  | 60 °C | 1024 | 2.05% / 2.45% | 0.75% | 1/8 | 0/32 | 203 ms |
  | 60 °C | 2048 | 1.18% / 2.42% | 0.80% | 1/8 | 0/32 | 306 ms |
  | 60 °C | 4096 | 0.68% / 1.97% | 0.64% | 6/8 | 0/32 | 508 ms |

- **Reading.**
  - A 60 °C gate removes every refusal: every wait was 0 s, starting at
    56–57 °C. At 50 °C only Spark box 4 and Spark box 8 never refused.
  - The longer prefill is what brings the prefill CV down.
  - 4096 at 60 °C is the best combination. Spark box 3 (1.26%) and Spark box 4
    (1.97%) were still above the former gate, and a 4-sample CV is itself
    noisy.
  - Decode is under about 1% everywhere, and 4096-token context costs about
    0.2 ms per decode step.
- **Not done in this experiment.** A real benchd calibration at 60 °C: the
  gate temperature is a compiled benchd constant. Section 7.7 does it with a
  rebuilt benchd.

### 7.7 The fix: a track-specific benchd, a 4096-token seed, one warm-up and the median of five (2026-09-27)

- **What changed, and where.**

  | Change | Where | Why |
  |---|---|---|
  | GB10 cool gate 50 °C → 60 °C | benchd fork `gumbii/nemotron35-own` | A loaded resident holds the die at 49–52 °C in P0 (section 7.5). `GPU T.Limit Temp` is a margin (~96 °C limit); thermal-slowdown counters stayed at 0 µs. |
  | `seed_tokens` 1024 → 4096 | fixture | The ~80 ms bandwidth floor is a smaller share of a longer prefill (section 7.3). |
  | `official_prefill_warmup_runs` 0 → 1 | fixture; the adapter accepts a repeated `prefill` in one prefill phase | An unmeasured pass after the gate idle brings the clocks back up before timing. |
  | `official_prefill_timed_runs` = 5 (new, optional field; median) | benchd fork plus fixture | It damps per-sample noise and drops one slow pass. |

- **The benchd build.** `scripts/build-dist-linux.sh` builds it on a GB10.
  Rebuilding with `BENCHD_DIST_NO_CACHE=1` gives the same bytes. Dist
  `1dfdf28`, source `ad17f32`, benchd sha256 `f95d5949…`. The workspace suite
  passes (1207).
- **Goldens.** All 15, at 4096 tokens, were recorded independently on Spark box 5
  and Spark box 6 and are byte-identical. The public prompt was extended to 4201
  tokens, and the private excerpts to about 5000 words (the old excerpts are
  byte-identical prefixes).
- **Calibration** (`benchd calibrate-baseline`, 4 passes, a fresh resident
  per pass):

  | Round | Setup | Boxes under the former gate on both axes | Prefill CV |
  |---|---|---|---|
  | 1 | 1024 seed, 50 °C gate | 0 of 2 that finished (4 of 7 refused by the gate) | 2.9–8.4% |
  | 2 | 4096 seed, 1 warm-up, 60 °C, 1 timed pass | 3 of 6 | 0.37–2.70% |
  | 3 | 4096 seed, 1 warm-up, 60 °C, **median of 5** | **8 of 8** | **0.17–0.89%** |

  In round 3, decode CV was 0.25–0.84%, with no gate refusals. The reference
  measures prefill at 126–127 µs/token (about 0.52 s per 4096-token prefill)
  and decode at 16.6–16.9 ms/token (benchd-timed, 4096-token context).
- **Status of those files.** The round-3 calibrations
  (`~/nemotron35/cal4096.json` on each box) were measured against a reference
  clone at `4e4c0be`. They become the official per-box calibrations once the
  published reference commit is pinned. Rerun `tools/calibrate-box.sh` then;
  the numbers should not move.

### 7.8 Logit ties make the timed window's exact-match check brittle (fork replay, not in the current benchd)

- **Status (2026-10-07).** The reference replay below was built in the
  GumbiiDigital fork benchd. The current benchd (Layr-Labs bench main, section
  7.13) does not have it, and the fixture no longer declares
  `timed_replay_max_logit_gap` or `timed_replay_off_argmax_per_thousand`. The
  timed decode window is an exact match against the per-depth oracle. Main has
  its own timed token tolerance, which this track does not enable.
- **What was found.**
  - The target's logits come out of the NVFP4 `lm_head` in BF16. Near the top
    of the distribution their resolution is 1/16 to 1/8, so the top two
    tokens often tie exactly. At a tie the lowest token id wins.
  - Along the reference's own 128-step continuation, every pool prompt has
    at least one exact tie, and several have four or more.
  - With the 4096 seed, every MTP depth splits from serial at an early tie in
    the calibration prompt's continuation. That is why the per-depth goldens differ from serial. The divergence
    is deterministic and byte-identical across boxes.
- **Why it matters.** benchd checks the TIMED free-run decode with an EXACT
  match on every committed token: a mismatch is a hard failure, and the 10%
  tolerance applies only to the correctness gate. So a kernel change that
  moves the last bit of the top logits at a tie position fails the timed
  window, even if it is a legitimate speed-up. The engine is reproducible, so
  the reference itself is safe. Participants, though, are held to bit-exact
  logits at those positions.
- **The decision: reference replay.** A precomputed tie list was
  tried first and does not scale. One prompt's main stream has 16 near-ties
  within one BF16 step. The tree of reference continuations grew 16 → 101 → 283
  branches by depth 3 and still did not cover the reference's own MTP streams.
- **What was built** (benchd fork source `196e586`, dist `cefe79f`):
  - The fixture declares `timed_replay_max_logit_gap` 2.0 and
    `timed_replay_off_argmax_per_thousand` 100.
  - A candidate's timed divergence (prefill token, seed or committed token) is
    deferred.
  - After the pairs, benchd boots the reference once and teacher-forces each
    deferred stream through it, then applies contract section 5.4.1.
  - Control legs and calibration never defer. The single-leg and non-official
    paths refuse a deferred divergence by name.
  - The golden format is unchanged, so no re-recording was needed.
- **How the numbers were set.** The reference's own MTP-3 and MTP-6 streams on
  all 8 pool prompts were replayed through its serial engine. The worst
  in-top-8 gap was 1.625 logits, there were at most 9 off-argmax positions in
  129, and one token fell outside the top 8. The verify pass quantizes FP8
  activations over the whole batch, so it moves logits by more than the last
  bit. The policy leaves margin over all of it.
- **Tests.**
  - In bench-core: the policy, the BF16 step, and the judgement.
  - In bench-runner: deferral at the prefill, seed and token positions, the
    exact match when deferral is off, and the refusal on the non-replay path.
  - In benchd: replay accept, gap refusal, budget refusal, and prefix positions
    skipped.
  - The workspace passes 1219 tests.

### 7.9 Eight-hour load test on all 8 Sparks (2026-09-27)

- **Setup.**
  - Window 03:30–11:30 EDT, all 8 Sparks, one JIT runner per box
    (`tools/box-runner/`, stable name `nemotron35-box-<host>`).
  - The organizer's router (`tools/router/router.py`) stands in for the
    intake service: it keeps only `editablePaths`, overlays them onto the
    armed base branch, pushes `sub/<id>` and dispatches `benchmark.yml`.
  - `tools/router/loadgen.py`, seed 20260927, keeps 6 submissions queued: null
    40%, MTP 1–6 20%, Marlin 8%, CUDA graphs 7%, garbage 10%, negatives 15%.
- **Throughput.**
  - 191 submitted, 186 finished, 5 expired unrun at the window's end.
  - 21–24 jobs per box. A ranked job takes 23 min (median), 29 min (p90).
  - About 23 jobs an hour across the fleet.
  - No infrastructure failure after the fixes below (from 04:30 on).
- **Outcomes against expectations.**

  | Variant | Outcome |
  |---|---|
  | null | 68 pass, 2 refused by the +2% acceptance band |
  | MTP 1–6 | 41 below the 0.95 floor, 1 band drift, 1 boot refused |
  | Marlin | 17 below the 0.95 floor (prefill about 0.91) |
  | CUDA graphs to 4096 | 13 pass, 1 boot refused (memory plan) |
  | garbage | 12 `TIMED-REPLAY-REJECTED`, 2 stopped first by the acceptance band |
  | 3 MB file | 10 refused by the static review |
  | depth 7 | 3 refused by the preflight |
  | trusted paths smuggled | 11 pass (the router dropped them), 1 band drift |
  | engine sets a `VLLM_*` variable | 3 pass (review-only, recorded) |

- **Fairness: the null score.**
  - 68 passes: mean 0.9959, sd 0.0058, range 0.983 to 1.009.
  - The candidate leg reads about 0.4% slow. The spread is about ±1%.
  - Per-box means run from 0.993 (Spark box 3) to 1.000 (Spark box 2).
- **MTP on the 4096 botany prompt.** Decode speed-up by depth, against serial:
  about 1.03 at depth 1, 0.94 at depth 2, 0.97 at depth 3, 1.03 at depth 4,
  0.98 at depth 5 and 0.96 at depth 6. Prefill costs 8% at depth 1 and up to
  18% at depth 6. No depth clears the 0.95 prefill floor.
- **Findings.**
  1. **The +2% acceptance band refuses unchanged code.** 2 of 70 null runs
     were refused, with decode 2.2% slower than the reference. With an sd of
     0.6% plus heat, a 2% band is about 3 sd. Options: a 3% band, or one retry
     on a band miss.
  2. **Spark box 6 drifted slow.** It caused 3 of the band refusals:
     its decode ran 2–3% slower than its calibration. It was power-drained
     and re-calibrated afterwards.
  3. **Memory plan after a heavy job.** One control leg's resident refused to
     boot on Spark box 4: the memory plan came out short, likely because memory
     was not yet returned from the previous job. It happened once.
- **Bugs found and fixed during the test** (all committed):
  - The supervisor did not export `BOX_RUNNER_APP_ENV` to the mint helper.
  - The JIT config file sat in a root-owned directory the runner could not
    clean; it now goes over stdin.
  - The default `work_folder` did not exist; it is now `_work`.
  - `fetch-benchd.sh` chmod-ed the read-only staged pair.
  - The idle-recycle `timeout` killed every job at 60 minutes.
  - `SERVE_UP_SOCKET_DIR=/tmp` is hidden by the container's own `/tmp`, so
    the job hung until the 3600 s health timeout. `serve-up.sh` now refuses it.
  - A cancelled job left its resident container on the GPU. The supervisor
    now removes `serve-up-*` containers on every path that ends a job.
  - The router read no box name, and did not parse the log score.
- **No hosted runners, one artifact.** The pipeline uses no GitHub-hosted
  runner: the surface check runs on a box runner. The score reaches the
  dispatcher as the sealed `score.json` artifact Yukon reads
  (`benchmark-results-<run_id>/score.json`) and in the job log
  (`tools/router/README.md`). `ci.yml` is manual-only.
- **Participant code ran as `runner` (resolved).** During this test untrusted
  build and engine code ran as the box's `runner` user, which is
  root-equivalent through `docker`. Participant code now runs as an
  unprivileged sandbox user; see `docs/sandbox-design.md`.

### 7.10 Rotation pool goes live: oracles, cross-prompt calibration, held-out pool (2026-09-28)

- **Oracles.** The 42 per-depth oracles (MTP-1 to MTP-6) for the seven other
  pool prompts were recorded on Spark box 5 and Spark box 6 independently. All 56
  files per box (8 serial goldens, 48 oracles) are byte-identical across
  the two boxes. The 14 already-pinned goldens re-recorded to their exact
  pins.
- **One transient.** Spark box 5 failed once while loading weights, with
  `CUDA error: operation not permitted` in `nvfp4_marlin_process_scales`.
  A resume (`--only … --steps 2,3,4,5,6`) on the same box then recorded
  every file identically.
- **Does one band per box hold for every prompt?** A full calibration was
  run on one non-botany prompt per box, on Spark boxes 1, 2, 3, 4, 7 and 8.
  - Decode lands within 0.43% of botany on every box (−0.43% to 0.00%).
  - Prefill lands within 0.64% on five boxes, and at −1.22% on Spark box 3
    (the medicine prompt).
  - Each score is a ratio against a control leg run on the same prompt in
    the same job, and the prefill band is ±5%. So one botany band per box
    stays valid, and no per-prompt bands are needed.
  - The first Spark box 3 attempt was refused on its CV (decode 1.10%, above
    the then-current maximum). A rerun passed at CV 0.58%.
- **Held-out final pool.** 8 stitched prompts × 7 goldens, each agreed byte
  for byte by two boxes (Spark box 1, 2, 3, 4, 7, 8). The pool is kept only in
  `.mlxfast-private/final/` on the organizer workstation, and the box-side
  recordings were deleted.
- **Recorder socket path.** It must stay short: a per-job path under the
  refresh folder came to 108 bytes, over the 107-byte AF_UNIX limit.

### 7.11 Rotation smoke test and 3-hour soak (2026-09-28)

- **Smoke test.** 2 unchanged engines passed (1.006 and 1.002) and the
  garbage engine was replay-rejected. The three jobs drew three different
  prompts (republic, travel, essays).
- **Soak, 06:47–09:47 plus drain.** 81 submissions on 8 prompts:

  | Group | Submissions | Outcome |
  |---|---|---|
  | Unchanged engines | 31 | 30 passed, mean 0.9959, range 0.982–1.009 |
  | Garbage | 10 | all replay-rejected |
  | Depth 7 | 8 | all refused by preflight |
  | 3 MB file | 2 | both refused by the surface check |
  | Edits to trusted paths | 2 | dropped by the router; both passed (about 1.00) |
  | Engine sets `VLLM_*` | 5 | all passed (0.995; observe-only) |
  | CUDA graphs | 3 | all passed (1.007) |
  | Marlin | 2 | floor failed |
  | Reference MTP | 14 | floor failed |
  | Still queued at the stop | 6 | expired, never dispatched |

  There were no infrastructure failures and no watchdog cancels.
- **One reference-leg divergence.**
  - One job on Spark box 8 (republic) failed with
    `SERIAL-CONTROL-LEG-FAILED`: partway through decode, the trusted serial
    control returned a different token from the golden.
  - Three later serial recordings of republic on the same box all
    reproduced the pin (`ed507fd…`) byte for byte. Republic's control leg
    passed on the other boxes.
  - So this was a one-off, not a box or golden defect: 1 in about 100
    control legs over the soak and the smoke test.
  - A control-leg failure is never the submission's fault. Such a job
    is re-run once automatically rather than scored as a failure: the
    measure script re-runs the whole measurement once, only when the trusted
    engine's own tokens diverged, after clearing every per-job directory
    (contract section 5.5). Any other control-leg failure, or a second
    divergence, ends as an infrastructure outcome (`control_failed`).

### 7.12 Adversarial review of the whole setup, and its fixes (2026-09-30 to 10-02)

An adversarial review covered five surfaces: the sandbox and container, the
scoring path and workflow, intake and the router, the box runner and
calibration, and benchd. No path let a submission forge a passing score or
read hidden data. Three paths let a team make bad runs disappear, so only
good draws reached the ranking. All three are fixed and tested on the boxes.

- **Fake score lines in the job log.**
  - The attack: participant output reaches the job log through `build.rs`
    warnings, the kernels build, and the resident log that `serve-up.sh`
    prints when a boot fails. A second copy of benchd's score marker made the
    router discard the run as infrastructure.
  - The fix: commit-reveal.
    - A per-job nonce is drawn into `RUNNER_TEMP` before any participant code
      runs, and the score is revealed after sandbox cleanup in a block tagged
      with it.
    - The parser takes only the last tagged END line, with a sha256 match
      (`docs/submission-restriction-spec.md` §10).
    - A run whose code ran but sealed no valid score is `unsealed`, the
      team's failure.
  - On the boxes: the fake marker and a complete fake block, printed as raw
    log lines, were ignored and the real score was kept.
- **`/dev/shm` between legs.**
  - The attack: `/dev/shm` is a 61 GB tmpfs in unified memory, shared through
    `--ipc host`, and it was wiped only at job start and end. A slow candidate
    could fill it so pair 2's reference leg could not boot: a final
    control-leg failure, recorded as infrastructure.
  - The fix: the candidate leg's `serve-up.sh --stop` kills the sandbox
    user's processes, runs the wipe and waits for memory. A wipe also runs
    before the measurement, and the adapter runs under `prlimit` caps.
  - On a box: a 50 GiB flood by `nemosub` took available memory from 106.8 to
    57.9 GiB, below the reference leg's 64.8 GiB. The real stop path removed
    it and restored 108.7 GiB in 4.7 s.
- **The final mean.**
  - The attack: `rescore_report` averaged only passed runs, so an engine could
    fail on purpose on slow draws.
  - The fix: exactly K slots per team.
    - A failure caused by the submission counts as 0.
    - Infrastructure failures are re-run up to a cap.
    - Reference-leg or band failures that persist after re-runs count as 0.
    - Contest selection counts failures the same way.
- **Sealed refusal text.** benchd's out-of-band refusal no longer seals
  control-leg timings, calibration numbers or the box name in the text. A
  refused run seals no timings at all: benchd source `9403e9f`, dist
  `a8fafeb`. This was a fork change. The current benchd does not have it
  (section 7.13).
- **The egress rule did not survive reboots.**
  - What happened: a wifi-controller firmware update power-cycled all 8
    boxes. The rule that rejects all `nemosub` traffic had been persisted only
    through `netfilter-persistent`, which the boxes lack, so it was gone.
  - Impact: the ranked preflight probes egress and refused every job, so it
    failed closed.
  - The fix: `nemotron35-sandbox-egress.service` runs
    `/etc/nemotron35-sandbox/egress.sh` at every boot, which always puts the
    rule first in OUTPUT. The box runner `Requires` the unit and checks the
    rule in `ExecStartPre`.
  - Verified on all 8 boxes. The `nemosub` uid differs per box (981, 995,
    996), so checks match the rule by name.
- **Fairness after all fixes.**
  - 40 unchanged engines across all 8 boxes: 39 passed, mean 0.9956, sd
    0.0049. That matches the 0.9959 measured before the fixes.
  - An earlier sample of 7 runs at 0.989 was noise.
  - The `/dev/shm` wipe before each reference leg adds a few seconds of idle.
    The 60 C cool gate does not equalise start temperatures, but the soak
    shows no bias from it.
- **Candidate boot failures from GPU transients.**
  - What happened: one unchanged engine failed to boot with `CUDA error: an
    illegal instruction`, the same family as the Spark box 5 transient in 7.10.
  - Rule: a pair-1 candidate boot failure (no completed pair is sealed, so
    `paired_legs` is absent) shares the job's single re-run.
  - Rule: in the final re-score, each slot may be re-dispatched once after a
    `boot_failed`; a second one counts as 0.
  - A pair-2 boot failure is never re-run in the job: the engine has seen its
    own timing.
- **Router PATH.** After a `gh` upgrade, the router found no `gh` and
  dispatched nothing for 4 hours. Its PATH now uses mise's stable shims.

### 7.13 benchd moves from the fork to Layr-Labs bench main (2026-10-07)

(History. Section 7.14 supersedes the pin and the sealing rows below: since
PR #35 the pinned benchd has the stricter sealing again.)

- **What changed.** `tools/fetch-benchd.sh` resolves Layr-Labs bench-dev
  `main`, dist `e4e3110`, source `44f1fda`. Before, it resolved the
  GumbiiDigital fork, branch `gumbii/nemotron35-own`, dist `a8fafeb`, source
  `9403e9f`.
- **What the new benchd adds.**

  | Change | Effect on this track |
  |---|---|
  | Decode is the decode window (`ecd66fe`) | Decode seconds per token = the time of `free_decode_run(N)` / N, with no seed prefill. The decode gain, the band and the floors use it. |
  | Calibration file version 3 (`ecd66fe`) | benchd and `tools/ranked-box-preflight.sh` refuse versions 1 and 2. Every box must recalibrate. |
  | `official_pair_combine` (`1aa7721`) | The fixture declares `mean`. The score is the mean of the 3 per-pair composites, and every pair is gated. |
  | 3 calibration passes | `tools/calibrate-box.sh` defaults to 3 passes, not 4 (organizer ruling, 2026-10-06). |

- **What the fork had and main does not have.**

  | Fork change | Now |
  |---|---|
  | Timed-divergence reference replay (section 7.8) | Removed from the fixture. The timed window is an exact match. |
  | `iterate --live-prompt` rotation | Not used. Every job measures every live golden. |
  | 0.97–1.03 decode band default | Main's default is 0.98–1.02. The fixture still declares a 3% up tolerance, which main reads. |
  | Stricter sealing (section 7.12) | Not in main. The sealed record can carry engine-chosen values. |
  | This track's contract-oracle entry | Not in main. benchd reads the fixture. |

- **Kept.** The 60 °C GB10 cool gate and `official_prefill_timed_runs` (the
  median of 5 timed prefill passes) are in main at `44f1fda`.
- **Fixture parse check.** `bench-core` at `44f1fda` parses the fixture and
  certifies it: `official_pair_combine` = mean, 3 pairs, 5 timed prefill passes
  and 1 warm-up, floors 0.95, bands 5% prefill and 3% decode (up only).

### 7.14 Dev refresh and ranked acceptance (2026-10-19/20)

The commands for benchd, the goldens, calibration and a ranked score are in
[reproduce.md](reproduce.md). The depth runs and the load test used the
workflow on test branches and `tools/router/loadgen.py --hours 3 --seed 20260927
--backlog 4` on the router (`tools/router/README.md`).

- **Change ledger** (all merged into this repository's `main` with signed
  merge commits).

  | PR | Merge | Change |
  |---|---|---|
  | #28 | `1c94f52` | measure-and-score retries a pair-1 serial-control boot failure once, then exits 75 (contract section 5.5) |
  | #30, #31, #32 | `3063107`, `ea3e0d9`, `c4b03d4` | README: the ranked box toolchain, how to submit; docs: the score artifact upload, the sealing, the fleet state |
  | #33 | `78c0067` | new `pool-9/10/11` goldens: the organizer's own three coding prompts, same names and 496-token length; only the 21 pins change |
  | #34 | `49b5068` | the calibration CV limit is the fixture field `calibration_max_cv_percent`, set to 2.0 (it was a fixed 1% in benchd; a Spark box 3 calibration was refused at 1.20%) |
  | #35 | `c1ef8de` | benchd pin: bench-dev dist `43a9a16` (bench PR 425), source `b65af98`, which adds bench PRs 421-424 (engine text, audit fields, hello identity and engine-derived numbers kept out of the sealed output) and the fixture CV limit |
  | #36 | `e9dfccb` | the organizer router records a pair-1 serial-control boot failure as `control_failed` (infrastructure); a `neg_boot_fail` load-test variant |
  | #37 | `c39f90c` | `tools/pool-refresh.sh --count` (default: the fixture's `live_goldens`) and `--texts` (chat prompts); the count persists across resumed steps |

- **Pinned execution set.** Engine image `vllm/vllm-openai:v0.27.1` @
  `sha256:1c8e60a0…b228c`; snapshot `bee75962…16b8`; reference tree
  `5be35aa`; benchd dist `43a9a16` / source `b65af98`; 3 pairs, one per live
  golden, mean of the per-pair composites; seed 496 tokens; 1 warm-up and the
  median of 5 timed prefill passes; 128 decode steps; floors 0.95, ceiling
  5.0; control health band 5% prefill and 3% decode (slow side only).

- **benchd rebuild.** A no-cache rebuild of the Linux aarch64 pair from source
  `b65af98` (`scripts/build-dist-linux.sh`, 2026-10-20) gave `benchd` sha256
  `ff4017e9…338c` (3,898,160 bytes) and `record-correctness-golden`
  `bbe18e01…c596` (1,640,408 bytes), equal to the published dist.

- **Live pool and custody.** Three coding tasks (Python, Rust, TypeScript),
  each one user message through the pinned chat template, exactly 496 tokens
  (`tools/author-prompts.py --chat`). Recorded on Spark box 3 and Spark box 4
  (serial plus MTP 1 to 6): 21 files, byte-identical on both boxes, pinned by
  #33. Input hashes are in reproduce.md, section 2. The texts, token files
  and goldens are in the organizer archive; the live goldens are also on the
  jump box. On 2026-10-20 the copies on Spark box 1 to Spark box 4 and the jump box
  were checked: the same 21 files on all five.
  On 2026-10-20 the pool was recorded again from the same token files, with
  the pinned benchd `b65af98`, on Spark box 4 and Spark box 2: all 21 files equal the
  fixture pins on both boxes.

- **Calibration** (version 3, 3 passes, reference `5be35aa`, benchd
  `b65af98`). CV in % (prefill / decode):

  | Box | Runner name | pool-9 | pool-10 | pool-11 |
  |---|---|---|---|---|
  | Spark box 1 | (organizer runner) | 0.32 / 0.27 | 0.80 / 0.69 | 0.27 / 0.25 |
  | Spark box 2 | (box label) | 0.57 / 0.91 | 0.14 / 0.53 | 0.62 / 0.19 |
  | Spark box 3 | (organizer runner) | 0.65 / 0.36 | 0.47 / 0.23 | 0.36 / 0.36 |
  | Spark box 4 | (organizer runner) | 0.64 / 0.08 | 0.56 / 0.19 | 0.43 / 0.43 |

  The box name must be the runner name: a band captured under the box's host
  label was refused by the preflight on the runner, and the box was
  recalibrated under the runner name.

- **Ranked acceptance** (unchanged engine, `main` declares MTP depth 3).

  | Run | Box | Result |
  |---|---|---|
  | dev repository run, `c1ef8de` | Spark box 2 | every step succeeded; `passed: true`, score 1.3099; decode gain 1.445, prefill gain 0.980; artifact uploaded |
  | organizer test repository run, same tree | Spark box 1 | `passed: true`, score 1.3095 |
  | organizer test repository, depths 1 to 6 | Spark box 3, Spark box 4 | 1.164 / 1.114 / 1.316 / 1.189 / 1.131 / 1.104, all pass correctness |
  | organizer test repository run, wrong tokens | Spark box 4 | refused, no score |
  | Yukon participant test: a submission to the dev row, depth 2, built by Yukon as `80581d1` on `main` `c39f90c` | Spark box 2 | `passed: true`, score 1.0970; artifact = reveal line; Yukon: "did not improve current best" |

  The same engine scored 1.4376 to 1.4427 on this repository before #33 (four
  runs on the previous prompts). The pool changed, not the engine: the MTP
  decode gain depends on the text the head drafts.

- **Held-out final pool.** The September pool (8 stitched prompts, 4096
  tokens) did not fit the 3-pair, 496-token fixture, so it was retired. The
  new pool is three organizer-written coding tasks (`final-1` Go, `final-2`
  TypeScript, `final-3` Python), each exactly 496 tokens through the chat
  template. Recorded on 2026-10-20 on Spark box 4 and Spark box 2 at `c1ef8de` with
  benchd `b65af98`: 21 goldens, byte-identical on both boxes. It is kept in
  the organizer's private state and archive and is not on any golden
  directory of a box; the rescore reads its `pins.json`.

- **Load test** (2026-10-20 16:21-19:21 UTC, then drain; `tools/router/loadgen.py`
  seed 20260927, backlog 4, base = dev `c1ef8de`; organizer test repository;
  Spark boxes 1, 3 and, from 17:35, 4).

  | Variant | Outcome |
  |---|---|
  | null (unchanged engine) | 11 pass: mean 1.3142, sd 0.0043, range 1.3077 to 1.3216; per box 1.3106 (Spark box 1), 1.3140 (Spark box 3), 1.3181 (Spark box 4) |
  | MTP | 3 pass (1.1535, 1.3102, 1.1016) |
  | trusted paths smuggled | 1 pass (1.3124): the router dropped the paths |
  | 3 MB file | 2 refused by the static review |
  | Marlin | 3 failed correctness |
  | CUDA graphs | 1 failed correctness |
  | garbage | 1 failed correctness (the fixture has no timed replay, so a token mismatch is the refusal) |

  22 jobs finished and 4 expired unrun at the window's end. No
  infrastructure failure (no `infra_failed`, `control_failed`, `band_refused`
  or `unsealed`). The Marlin and CUDA-graphs variants are measured, not
  expected to pass; under the exact-match timed window they now fail
  correctness, where in the September soak (4096-token seed, timed replay) most
  CUDA-graphs runs passed. A dev-row baseline on Spark box 2 the same evening
  scored 1.2911 (Yukon dev run), 1.4% under the Spark box 2 run of the
  morning; not yet explained.

- **Checks.** At `c1ef8de`, on Spark box 4 (and an organizer workstation for the three
  checks whose tools that box lacks): adapter `cargo fmt`, `clippy -D
  warnings`, `cargo test` and the release build; the resident and router unit
  tests; the manifest lint; `check-doc-consistency.py`; the workflow egress
  scan; and every `tools/test-*.sh` and box-runner test. All pass. The
  preflight's section 8 paired-baseline cases need a checkout with history
  (they skip in an archive copy) and pass in a full clone.

- **Limits.** Branch
  protection on `main` is the org's signed-commit ruleset only. The organizer
  archive is a private repository; a reviewer needs read access to it to
  check the prompt texts.

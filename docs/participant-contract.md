# Nemotron 3.5 Lightning 30B-A3B NVFP4 — participant contract

This document states the terms that bind a submission to track
`nemotron3.5-lightning-30b-a3b-cuda-v1`.

`benchmark.json` is the Yukon track manifest.
`fixtures/nemotron35_lightning_30b_a3b_track.json` is the track contract fixture.
Both carry values, paths, commands and pins only. This document explains them.
It never overrides them.

## 1. Order of authority

Apply these in order. The higher entry wins.

1. The ranked run on the official runner. It is the authority on any score.
2. `fixtures/nemotron35_lightning_30b_a3b_track.json` and `benchmark.json`.
3. This document.
4. `README.md`, `TASK.md` and `AGENTS.md`.

If a configuration file disagrees with this document on a plain value, the
configuration file wins, and CI (`tools/check-doc-consistency.py`) is meant to
catch the drift first. If either disagrees with the benchmarker about
measurement, the benchmarker wins.

The benchmarker is a prebuilt `benchd` binary. `./tools/fetch-benchd.sh`
resolves it from the bench repository's release channel for this track, and
verifies it against the channel's `benchd.manifest.json` (`{branch,
source_commit, sha256, bytes}`). A submission cannot change what measures it.
The channel is the `main` branch of the public `Layr-Labs/benchd`
repository, and the fetch needs no token. A verified pair through
`BENCHD_DIST_LOCAL` needs no network.

## 2. What the track measures

The track measures single-stream prefill and decode speed of the target on a
DGX Spark (GB10, sm_121, 128 GB unified memory).

The target is `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` at revision
`bee7596271d1495f6992ae224aefde4410e816b8`: a ModelOpt mixed-precision
checkpoint of `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`.

| Property | Value |
|---|---|
| Architecture | `NemotronHForCausalLM`, `model_type` `nemotron_h` |
| Layers | 52: 23 Mamba-2, 23 MoE, 6 attention (indices 5, 12, 19, 26, 33, 42) |
| Hidden size | 2688 |
| Attention | 32 query heads, 2 KV heads, head dimension 128 |
| Mamba-2 | 64 heads of dimension 64, SSM state 128, conv kernel 4, 8 groups |
| MoE | 128 routed experts, 6 per token, intermediate 1856, plus a shared expert of 3712 |
| Vocabulary | 131072, untied embeddings |
| Tokens | BOS 1; EOS `</s>` 2 and `<|im_end|>` 11 |
| MTP | 1 native next-token layer (`mtp.*`: attention + MoE), BF16 |

The checkpoint's `hf_quant_config.json` is the authority on formats:

| Tensors | Format |
|---|---|
| Routed and shared experts, `lm_head` | NVFP4, weight-only (W4A16): 16-element blocks, FP8 E4M3 block scale, FP32 global scale |
| Mamba `in_proj` / `out_proj` | FP8 |
| KV cache | FP8 |
| Mamba SSM state | float32 (`mamba_ssm_cache_dtype`) |
| MTP head | BF16 |

The engine is vLLM `v0.27.1`. The speculative arm is vLLM's `mtp` method over
the native MTP layer. The KV backend is `vllm-paged`, pinned in the fixture and
in `benchmark.json`.

## 3. What you may edit

`benchmark.json` `editablePaths` is the authority: `vllm`, `harness` and
`mtp-head.manifest.json`. A submission archive has REPLACE semantics over those
paths.

| Path | Contents |
|---|---|
| `vllm/` | vLLM v0.27.1's Python package (`vllm/vllm/`), `vllm/VENDOR.json`, `vllm/LICENSE` |
| `harness/` | `protocol-adapter/` (Rust), `vllm_resident/` (Python), optional `kernels/` |
| `mtp-head.manifest.json` | the speculative declaration (section 4) |

### 3.1 Optional paths

`mtp-head.manifest.json` is optional. An archive that omits it runs serial.

### 3.2 The byte budget

The static review (`.github/scripts/submission-static-review-checks.sh`) caps
the editable surface, and `benchmark.json` `editableSurfaceByteBudget` declares
the caps:

| Cap | Value | Derivation |
|---|---|---|
| `maxTotalBytes` | 36,047,190 | the surface at rest, 34,998,614, plus 1 MiB |
| `maxFileBytes` | 2,558,837 | the largest file at rest, 1,510,261, plus 1 MiB |
| `maxGrowthBytes` | 36,047,190 | equal to the total: the vendoring commit is the growth |

The cap exists so that source cannot hide lookup tables. It also refuses
symlinks and special files anywhere under an editable path.

### 3.3 What you may not edit

Everything else: `tools/`, `fixtures/`, `setup.sh`, `benchmark.sh`,
`benchmark.json`, `correctness_prompts/`, `.github/`, the docs, the pinned
snapshot, the pinned engine image and benchd.

### 3.4 The target's formats are frozen

Do not re-quantize, re-represent, or change the numerical format of any target
weight, activation path, cache or recurrent state. Section 2 lists the formats.
This holds even when the result passes the correctness gate.

- **What is legal:** kernels that compute the SAME formats faster. That includes
  a fused NVFP4 dequant-GEMM or MoE, a better Mamba-2 scan or state update,
  attention kernels, CUDA graphs, fusion, and scheduling and memory layout.
- **Activations are part of the format.** The experts are declared W4A16, so
  quantizing their activations to FP4 or FP8 is a change of format. On GB10 this
  rules out native FP4 MMA with on-the-fly activation quantization.
- **A load-time repack is also legal**, under two conditions. It must be
  input-independent (for example a weight layout swizzle), and it must preserve
  every value exactly.

### 3.5 Editing the engine

`vllm/vllm/` is the engine's Python and Triton source. `tools/stage-vllm-overlay.sh`
copies the pinned image's installed vLLM to `.build/vllm-overlay/` and layers
your `vllm/vllm/` over it file by file. Your source wins; the base's compiled
extensions (`_C.abi3.so`, `_moe_C`, the flash-attention build) and generated
files survive. A file you delete is NOT deleted from the base.

New CUDA/C++ code goes in `harness/kernels/`. When `harness/kernels/build.sh`
exists, setup runs it inside the pinned image with no network. The script gets
`KERNELS_OUT` (put importable modules there) and `TORCH_CUDA_ARCH_LIST=12.1a`.
`KERNELS_OUT` is the only place it may write: the checkout is mounted
read-only, so a build that writes into the tree fails. The build runs under
CPU, memory and process limits. `KERNELS_OUT` is second on the resident's
`PYTHONPATH`, after the overlay.

The resident (`harness/vllm_resident/`) owns everything vLLM is asked to do
beyond the fixture's `engine_env`. That includes the engine arguments, CUDA
graphs, the compile level and the scheduler settings. It runs:

- inside the pinned image, with `--network none`;
- with the snapshot and the leg's tree mounted read-only;
- with a fresh `HOME` and fresh vLLM, Triton and inductor caches per leg.

benchd seals a harness hash over a fixed root set that includes `vllm` and
`harness`, so every engine edit is visible in the sealed provenance.

**Your engine must stay REPRODUCIBLE.** The same prompt must give the same
tokens on every run and on every box. The serial control leg and every timed
leg are checked against recorded tapes, and on this checkpoint a last-bit
difference becomes a different greedy token within a few steps: the
activations are FP8-quantized, which amplifies it. vLLM's stock Marlin MoE is
not reproducible on GB10, which is why the reference engine runs the humming
MoE in its batch-invariant mode (`docs/nemotron35-port-notes.md` section 7.2).
Check any kernel change with `tools/determinism-probe.py` before submitting:
it must report one stream.

### 3.6 Where your code runs

On a ranked box every piece of participant code runs as an unprivileged
sandbox user, not as the runner. That covers the adapter build and its
`build.rs`, `harness/kernels/build.sh`, the resident and the vLLM overlay, and
the adapter at run time. That user has no network, not even loopback. It can
read the checkout (read-only) and write only its own per-job directories. It
cannot reach the goldens, the trusted tools, docker or anything else on the
box.

- **Cargo dependencies come from crates.io only.** Every source in `Cargo.lock`
  must be the crates.io index. A `path` dependency must stay inside `harness/`.
  `[patch]`, `[replace]`, git sources and other registries are refused. The
  trusted side vendors the crates first, and then the build runs offline, so
  a `build.rs` that fetches anything fails.
- **What comes back is the score, nothing else.** The adapter's stderr is kept
  private and never returned, and job logs are not shown to participants. The
  score record carries no text your engine supplied. An engine error appears
  as a failure class plus the sha256 and byte count of its message. Identity
  strings and acceptance lengths are summarised, not copied.
- **Reported digests must be well-formed.** `head_provenance_sha256` and
  `runner_manifest_sha256` must be 64 lowercase hex characters, or they seal
  as `invalid`.
- **Nothing persists between jobs or legs.** Temp files, shared memory, IPC
  objects and message queues the sandbox user leaves are wiped, every process
  it started is killed after its leg, and it cannot schedule jobs. Do not rely
  on anything surviving from a previous leg or job; section 8 forbids the
  reuse anyway.

## 4. The MTP head

The head is the `mtp.*` tensors inside the pinned
`model-00052-of-00052.safetensors`. The fixture's `mtp_head` block records it:
270 tensors, BF16, 2,670,652,160 bytes. No submission carries head weights.

### 4.1 What you may declare

```json
{
  "version": 1,
  "source": "pinned",
  "max_bytes": 3221225472,
  "spec": { "enabled": true, "num_speculative_tokens": 3 }
}
```

`tools/spec-declaration.sh` is the one trusted reader of the declaration.

- **Allowed keys.** It accepts `version`, `source`, `max_bytes`, `bytes`,
  `sha256` and `spec`, and refuses any other key.
- **Source.** It refuses any `source` except `"pinned"`, which rules out
  `"remote"` and `"in_branch"`.
- **Depth.** It refuses an enabled depth outside `mtp_head.permitted_draft_depths`,
  which is 1 to 6.
- **Serial.** A declaration that is absent, disabled, or has 0 tokens derives
  serial.
- **Recorded, not enforced.** `max_bytes`, `bytes` and `sha256` are recorded
  only. The fixture's `declaration_max_bytes` is 3 GiB, so the pinned head
  fits the cap.

The shipped declaration enables MTP at depth 3 (`enabled: true`,
`num_speculative_tokens: 3`). The reference tree runs serial, and the
reference scored against itself is a null control of about 1.000.

### 4.2 What you may not do

- Replace the head, or load any other drafter (DSpark, DFlash, Eagle,
  n-gram).
- Re-quantize the head: `mtp_head.requantization_permitted` is `false`.
- Ship head weights, or fetch anything at run time; the resident has no
  network.

### 4.3 What the size cap does and does not do

The cap is a declaration limit only. Nothing loads from a declaration. The head
always comes from the verified snapshot.

### 4.4 What the head does in a run

At depth d, vLLM runs the one MTP layer recursively to propose d tokens per
cycle, and the target verifies them in one forward. The target decides every
emitted token.

The adapter reports the cycles as `acceptance_lengths`, and the drafted,
accepted and committed totals. benchd seals `effective_spec_mode`,
`effective_spec_depth`, `spec_rounds`, `spec_drafted_total`,
`spec_accepted_total` and `spec_acceptance_rate`.

## 5. Scoring

### 5.1 The formula

```text
composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75
gain      = baseline_aggregate / candidate_aggregate
```

The run is PAIRED on one box, and the pairs run one after the other, never
overlapping:

- **Pairs.** A ranked job runs `official_pairs` = 3 pairs.
- **Legs.** Each pair is one SERIAL CONTROL leg on the organizer-staged
  reference tree at `baseline_reference_commit`, then one CANDIDATE leg on the
  submission at its declared depth.
- **Aggregation.** benchd computes one composite per pair. The fixture
  declares `official_pair_combine: "mean"`, so the score is the arithmetic
  mean of the 3 per-pair composites. The floors and the bands gate every pair
  on its own, each against its own control leg. The first pair that fails
  stops the run. Every pair stays in `metrics.paired_legs` as measured.

No baseline is stored anywhere; the control leg measures it next to the
candidate.

**Direction, rounding and ties.** Higher is better: `benchmark.json` sets
`direction` to `+`. benchd does not round the score: `score.json` holds the full
double-precision value, so two runs tie only when the two values are equal. The
platform, not this repository, decides whether a submission improves the
current best.

**What decode means.** Decode seconds per token is the decode window: the time
of `free_decode_run(N)` divided by N. The seed prefill of the prompt is not
part of it. Prefill is its own phase, timed on its own. `decode_gain` is the
control leg's decode window per token divided by the candidate leg's. The
calibration band and the floors use the same figure. benchd seals the seed
prefill window as a report-only figure under
`*_seed_prefill_window_seconds_per_token`.

#### 5.1.1 Where the prefill window is, and why you cannot move work out of it

benchd splits its own clock at the verb boundary:

| Window | From | To |
|---|---|---|
| prefill | `free_decode_begin` sent | the validated `seed_token` comes back |
| decode | there | `free_decode_run(N)` returns |

What the engine owes:

1. `free_decode_begin` runs the FULL 496-token seed prefill with the requested
   spec resolved, and replies only after that work has completed. The reply is
   the greedy seed token and the echoed `effective_spec`.
2. `free_decode_run` never re-prefills. It decodes from the state `begin` left.
3. Nothing prefills before `free_decode_begin` arrives.

**Moving work between windows is a scoring defect, not an optimisation.**
Deferred seed work lands in the decode window and makes decode slower. Decode
work done early, inside `free_decode_begin`, leaves the decode window and is
not scored, which breaks obligations 1 and 2 above.

**Enforced by the workflow.** The ranked workflow reads benchd's private audit.
The candidate-over-control limit depends on the declared depth: 1.2 at
depth 0, 1.35 at 1, 1.5 at 2 and 3, 1.6 at 4 and 1.65 at 5 and 6
(the fixture key `seed_window_cand_over_control_limit_by_depth`). The
control-over-candidate limit stays 1.5.
The seed-window figures stay private. A refusal names only the rule
(`tools/score-plausibility-gate.sh`).

The seed token's index: `seed_token` is `expected_decode_seed_token`;
`free_decode_run` returns the N tokens after it, so `tokens[0]` = `expected_decode_tokens[0]`.

### 5.2 The measured window

| Quantity | Value |
|---|---|
| Seed tokens per stream | 496 |
| Unmeasured prefill warm-up passes before the timed prefill | 1 (the pinned public warm-up golden, computed in full and never scored) |
| Timed prefill passes per phase | 1; that pass supplies the phase's prefill time |
| Checked decode steps | 128 |
| Golden shape | 496 `prompt_tokens` and an answer of at most 1024 `expected_tokens` per case (the answer ends at the first end-of-turn token), plus the free-run `benchmark` block |
| Streams per window | 1 |
| Timed prompts per leg | 1; pair k measures `live_goldens[(k - 1) mod 3]` (section 5.2.1) |
| Prompts in the pinned correctness pool | 3 |
| Pairs per ranked job | 3 |
| Legs per ranked job | 6 |

Every timed phase waits behind benchd's GPU cool-down gate. The ranked job
waits for the box to go quiescent first, and refuses to run with no
temperature reader.

### 5.2.1 Every job measures every live golden

The fixture's `live_goldens` lists three hidden coding prompts. Each one is a
user message rendered through the pinned `chat_template.jinja` with thinking
off, and is exactly `seed_tokens` long. Every job measures every live golden:
pair k measures `live_goldens[(k - 1) mod 3]`, and both legs of a pair measure
the same prompt, so the ratio compares like with like. The score is the mean
of the per-pair composites. `metrics.paired_legs` identifies each pair's
prompt by its sha256.

Why: one pair per prompt makes the score depend on all three prompts, and the
mean gives each prompt the same weight.
Every live golden carries its own serial golden and one timed oracle per draft
depth, and the run refuses a declared depth that has no oracle.

The per-box calibration file carries one band per live golden, and benchd
gates each control leg with the band of the prompt that leg measured.

### 5.3 The parameters

| Parameter | Value |
|---|---|
| `scoredBatchSize` | 1 |
| `prefillGainExponent` | 0.25 |
| `decodeGainExponent` | 0.75 |
| `pairsPerCohort` | 3 |
| `minPairsPerCohort` | 3 |
| `decodeSpeedupFloor` | 0.95 |
| `prefillSpeedupFloor` | 0.95 |
| `decodeSpeedupCeiling` | 5.0 |
| `kvBackend` | `vllm-paged` |

The score is the mean of the three per-pair composites
(`official_pair_combine`: `mean`). The floors and the ceiling gate every pair;
the workflow gate (`tools/score-plausibility-gate.sh`) enforces the 5.0 decode
ceiling on every pair.
Each pair has one timed prefill pass after one unmeasured warm-up pass.
The warm-up uses the pinned public golden and is never scored.

**Prefill ceiling (workflow gate).** A ranked run whose prefill gain is above
1.5 in any pair fails and gets no sealed score: no reveal block and no score
artifact (`tools/score-plausibility-gate.sh`, run after sandbox cleanup and
before the reveal). This is a policy ceiling: real prefill work on this engine
scores near 1.0, and a gain far above that usually means the timed passes
reused earlier work, which section 8 bans. The gate does not detect all reuse;
section 8 still applies below the ceiling. The same script also enforces the
seed-prefill window ceilings described in section 5.1.1.

### 5.3.1 Anti-cheat guards

benchd applies these guards only when the fixture enables them.

- Prefill gain cannot exceed 1.5. Decode gain cannot exceed 5.0.
  The refusal is `SPEEDUP-ABOVE-CEILING`.
- Each seed window must exist. It cannot exceed 1.7 times its timed prefill.
  Refusals are `SEED-WINDOW-NOT-OBSERVED` and `SEED-WINDOW-ABOVE-LIMIT`.
- All goldens must resolve inside a fresh private directory.
  Refusals are `PRIVATE-DIR-REQUIRED` and `GOLDEN-OUTSIDE-PRIVATE-DIR`.
- Sandbox override variables are not permitted.
  The refusal is `SANDBOX-OVERRIDE-REFUSED`.
- The warm-up uses a pinned public golden. This golden is never scored.
- Golden `g` supplies prefill. Golden `g+1` supplies decode, with wraparound.
- benchd checks each leg teardown. The refusal is `LEG-TEARDOWN-FAILED`.
- A reused prompt fails with `PROMPT-REUSE-REFUSED`.

### 5.4 Token fidelity

benchd applies a per-stream token-tolerance gate with a **10% budget** against
the serial tape. Every per-depth golden carries the SAME serial `cases[]`
(`tools/record-goldens.sh` splices them), so fidelity is always measured
against serial. The gate accepts similar output. It does not certify lossless
output.

### 5.4.1 The timed window: exact match

The timed decode window is checked token by token against the timed golden for
your declared depth. Every committed token must match. A token that differs
fails the run. The ranked run and the local modes apply the same rule.

The logits of this checkpoint are BF16, so the reference's top two tokens tie
exactly, or sit one BF16 step apart, several times in every 128 tokens. A
faster kernel, or a speculative verify that quantizes activations over several
tokens at once, can resolve such a tie differently and fail the run.

The benchmarker has no timed-divergence reference replay. The fixture declares
no timed token tolerance, so none applies.

### 5.5 Arming

`official_scoring_enabled` in the fixture is the single arm authority. benchd
refuses to seal an official artifact while it is `false` or absent, and so does
`tools/ranked-box-preflight.sh`.

The fixture ships ARMED (`official_scoring_enabled: true`). It was armed only
after four steps were done:

1. every golden is recorded on a Spark and pinned (the pending entries carry the
   sentinel `NEMOTRON35-LIGHTNING-CUDA-PENDING-ORGANIZER`, which the preflight
   refuses);
2. `baseline_reference_commit` names the promoted reference;
3. every box is calibrated;
4. the null control, the reference scored against itself, lands at about 1.000.

The control leg is serial whatever the candidate declares. benchd boots each
leg's resident itself, from that leg's own tree, with
`tools/serve-up.sh --boot --spec serial|mtp --draft-len N --socket-out FILE`,
and stops it with `--stop --socket PATH`.

A control leg that fails (`SERIAL-CONTROL-LEG-FAILED`) is not a fault of your
code, and the router records it as `control_failed`. How it counts: in the
contest, a finished post-setup `control_failed` or `band_refused` run counts
as 0.0 toward selection, with no automatic re-dispatch; in the final re-score
its slot is re-dispatched, and it counts as 0.0 only if it still fails after
the retries (section 5.7). A serial-control boot failure in a later pair is
final (`refused`). The measurement driver keeps one retry budget per job.
It re-runs the whole measurement, both legs of every pair, exactly once when
one of three causes fires: (1) the reference's own token divergence
(`SERIAL-CONTROL-LEG-FAILED: benchmark prefill|free-run decode seed|free-run
decode token mismatch`, in the timed or the warm-up leg); (2) the pair-1
candidate resident fails to boot before any candidate timing is sealed
(`LEG-SERVE-BOOT-FAILED: the candidate leg's ...` with an empty or absent
`metrics.paired_legs` audit trail); or (3) the pair-1 serial-control resident
fails to boot (`LEG-SERVE-BOOT-FAILED: the serial-control leg's ...`, with an
empty or absent `metrics.paired_legs`). A boot failure in a later pair is never
retried: pair 1 is already in `paired_legs`, so a second attempt would be a
best-of-two on your engine. Whichever cause fires first consumes the budget;
for example, a control divergence on attempt 1 followed by a pair-1 boot
failure on attempt 2 is final. In every retriable case the driver wipes the
sandbox, including all five per-job directories, before the second run. The
second attempt alone is the result; the first attempt's timings are discarded,
never compared. A second control-leg token divergence, or a second pair-1
serial-control boot failure, exits 75 (an infrastructure outcome), and the router records it as `control_failed`. Every other failure
is final, including every other control-leg failure (spawn, session,
protocol, cool gate, memory), because the next pair's control leg runs after
your leg. The log records `control_leg_attempts=1` or `2`; the sealed
`score.json` does not change.

#### 5.5.1 The per-box calibration

Each ranked box carries a calibration file, named by
`MLXFAST_BASELINE_CALIBRATION` and written by `tools/calibrate-box.sh` (3
passes, one resident boot per pass). It is a HEALTH BAND for the control leg,
never a denominator. A control leg outside the band fails the run by name. A box
whose passes spread by more than 2% gets no band. The fixture field
`calibration_max_cv_percent` sets this limit.

`tools/ranked-box-preflight.sh` section 8b refuses a calibration file in any of
these cases:

- its `version` is not 3. Versions 1 and 2 recorded decode as the whole window
  (seed prefill plus decode, over N), and the refusal says to recalibrate;
- it names another track, box or reference commit, or it has no `prompts[]`
  entry for one of the fixture's `live_goldens` (one band per live golden);
- it is dated before the reference commit or in the future;
- its band cannot fail.

The band's widths come from the fixture. The control leg's per-token time may
run at most `prefill_band_up_tolerance` = 5% (prefill) and
`decode_band_up_tolerance` = 3% (decode) slower than the box's calibration. The
fast-side tolerances (`prefill_band_down_tolerance` and
`decode_band_down_tolerance`, 5% each) are declared but disabled
(`prefill_band_down_enabled` and `decode_band_down_enabled` are `false`).

A failure of the trusted control leg (outside its band, or off its golden,
`SERIAL-CONTROL-LEG-FAILED`) is an infrastructure fault, not the submission's:
the control leg runs the organizer's reference tree, which the submission
cannot change.


### 5.6 Which goldens you can hold

| Object | Where it lives | Can you have it? |
|---|---|---|
| `correctness_prompts/public_longcopy_gate_english_4096.txt` | git | Yes: the public prompt text |
| `correctness_prompts/public-longcopy-gate-english-496.golden.json` | git | Yes: for local serial runs |
| `correctness_prompts/public-longcopy-gate-english-496.mtp<N>.golden.json`, one per depth 1 to 6 | git | Yes: for local runs at depth N |
| `timed_prompt_pool[]`: the 3 serial tapes that `live_goldens` (`pool-9`, `pool-10`, `pool-11`) names, three 496-token chat-template coding prompts | organizer storage, staged on the box | No |
| `speculative_oracles{}`, 6 per-depth oracles per live golden (18 in all) | organizer storage, staged on the box | No |
| `hidden_correctness_golden` | the default live golden's serial tape, pinned by digest | No |

The organizer stages the set on each box, and the preflight checks every
file's byte count and sha256 against its pin before a ranked run. The fixture's
`r2_path` fields name the files; they are not fetched at run time.

### 5.7 Pool refresh and final ranking

- **Per job.** Each ranked job measures every live golden (section 5.2.1).
- **Refresh.** The live goldens are replaced by prompts no submission has seen
  every 3 to 4 days during the contest.
- **Submissions** are unlimited.
- **Final ranking.** At the close, each team's best submission is taken: the
  mean contest score counts every contest run of that submission that scored or
  failed because of the submission (a team failure counts as 0.0; a
  post-setup reference-leg or band failure counts as 0.0; other infrastructure
  runs are excluded). Ties go to the most recent submission. The top N teams
  (default 10) are selected. Each selected submission is re-measured K times
  (default 5), one run per slot, across different boxes on a held-out pool
  that was never staged during the contest. Each slot is its first run that
  ends with a score or a team failure; infrastructure outcomes and off-pool
  prompts are re-run, not scored. In the final re-score only, the first candidate
  resident boot failure in a slot may be re-dispatched once; a second boot
  failure in the same slot counts as 0.0. A run that fails because of the
  submission, or a reference-leg or band failure that persists after re-runs,
  counts as 0.0 for that slot. The final ranking is the mean of the K slot values (the score, or
  0.0 for a team failure or penalized post-setup infrastructure slot). A team
  with any slot still pending or unresolved is incomplete and is not ranked
  above a complete team. Contest scores decide only who is re-measured.

## 6. Running the benchmark

`benchmarkCommand` runs `tools/nemotron35-yukon-run.sh`. That script selects
one of two paths:

| Condition | Path |
|---|---|
| `GITHUB_ACTIONS` is `true`, or one or more of `MLXFAST_NEMOTRON35_GOLDEN_DIR`, `MLXFAST_BASELINE_WORKSPACE` and `MLXFAST_BASELINE_CALIBRATION` is set | Ranked. The script replaces itself with `tools/nemotron35-measure-and-score.sh`. |
| None of these | Local. The script runs `tools/local-baseline.sh`, which runs `./benchmark.sh --local-iterate` over the public golden at the declared depth. The result has `"score": null`. |

The ranked workflow calls `tools/nemotron35-measure-and-score.sh` directly.
Both scripts are trusted-side tooling.

The ranked script builds one `benchd iterate --mode official` command. It
passes one `--golden` triple (the timed golden of the declared depth) and one
`--control-golden` triple (that prompt's serial tape) for every live golden,
in the fixture's `live_goldens` order (`pool-9`, `pool-10`, `pool-11`), so
pair k measures `live_goldens[k-1]`. With depth 3 declared, the shape is:

```text
benchd iterate \
  --engine <adapter binary> --weights $MLXFAST_TARGET_SNAPSHOT_DIR \
  --golden $GOLDEN_DIR/pool-9.mtp3.golden.json  --golden-sha256 <pin> --golden-bytes <pin> \
  --golden $GOLDEN_DIR/pool-10.mtp3.golden.json --golden-sha256 <pin> --golden-bytes <pin> \
  --golden $GOLDEN_DIR/pool-11.mtp3.golden.json --golden-sha256 <pin> --golden-bytes <pin> \
  --mode official --score-path score.json \
  --contract fixtures/nemotron35_lightning_30b_a3b_track.json \
  --baseline-workspace $MLXFAST_BASELINE_WORKSPACE \
  --baseline-calibration $MLXFAST_BASELINE_CALIBRATION \
  --control-golden $GOLDEN_DIR/pool-9.golden.json  --control-golden-sha256 <pin> --control-golden-bytes <pin> \
  --control-golden $GOLDEN_DIR/pool-10.golden.json --control-golden-sha256 <pin> --control-golden-bytes <pin> \
  --control-golden $GOLDEN_DIR/pool-11.golden.json --control-golden-sha256 <pin> --control-golden-bytes <pin> \
  --box $RUNNER_NAME --mtp-depth 3
```

The pins come from the fixture. Do not type this command by hand to
reproduce a score: dispatch `.github/workflows/benchmark.yml`, which adds the
GPU lock, the sandbox, the cleanup and the nonce-bound reveal
([reproduce.md, section 4](reproduce.md#4-reproduce-a-ranked-score)).

benchd seals `score.json`, its `.sha256` sidecar and the integrity sidecar
itself. The script converts nothing.
Before any participant code runs, the job draws a 64-hex nonce into
runner-private storage. After sandbox cleanup, **Reveal the sealed score**
prints that nonce once, the sealed `score.json` if it exists, and an end line
that binds the nonce to the file digest and length. The intake service parses
that block only ([submission-restriction-spec.md](submission-restriction-spec.md) §10); arbitrary text in the same log does not count.

When the reveal step prints a sealed `score.json`, the ranked workflow then
uploads that same file as the one artifact `benchmark.json` `scoreArtifact`
names (`benchmark-results-<run_id>/score.json`). This is the score Yukon reads.
No other file is uploaded (`tools/ci-workflow-egress-scan.sh`).

The rest of the command surface:

- **`preSubmitCommand`** runs the same script with `--preflight-only`, which
  checks the arm gate and the golden pin without measuring.
- **The ranked pipeline** is `.github/workflows/benchmark.yml`, dispatched
  only by `workflow_dispatch`. It runs on
  `[self-hosted, Linux, ARM64, nemotron3.5-lightning-30b-a3b-cuda-v1]`, one
  ranked run per box at a time.
- **Sockets.** Nothing in the job may export `DS4_RESIDENT_SOCKET` or
  `BENCH_WORKER_RESIDENT_SOCKET` (benchd's names), and nothing may preset
  `RESIDENT_*`, `VLLM_*`, `HF_*`, `TRANSFORMERS_*`, `TORCHINDUCTOR_*` or
  `TRITON_*`. The preflight refuses each of them by name.

## 7. The pinned artifacts

| Artifact | Identity |
|---|---|
| Target | 60 files at the pinned revision, each by bytes and sha256 (`target.files`) |
| MTP head | `mtp.*` in `model-00052-of-00052.safetensors` |
| Engine source | vLLM `v0.27.1` @ `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` (`vllm/VENDOR.json`) |
| Engine image | `docker.io/vllm/vllm-openai:v0.27.1` @ `sha256:1c8e60a0841b333c700488cb029d3664807249da0c071e862191b00fe34b228c` |
| Driver floor | `580.159.03` |
| benchd | `Layr-Labs/benchd` dist commit `80f8819dde25b55f1f5c5dc1c85863498135f87e`, built from source `10afddd4af99bcec1cbf87e57242f224df0bdac7` (public snapshot `6b6f7cf1fe1423e60cdf0ff936ac2d3f26ac348b`); `dist/linux-aarch64/benchd.manifest.json` there gives each binary's sha256 and bytes. `tools/fetch-benchd.sh` (`PINNED_SOURCE_COMMIT` / `PINNED_PUBLIC_SOURCE_COMMIT`) is the authority. |

## 8. Prohibited techniques

A submission that uses any of these is disqualified, whether or not a gate
catches it:

- **Reuse across requests.** Any cache, memo or reuse keyed on a request's
  input tokens, or carried between requests or phases, whose only possible hit
  is the harness repeating a computation. That includes turning vLLM's prefix
  caching back on, since benchd's phases replay the same prompt. It also
  includes keeping KV or Mamba state across a `sync`, or answering from a
  recorded stream. Bit-identical output does not make it legitimate.
  It also covers the prefill warm-up pass. The warm-up uses a separate public
  golden, and each pass must be computed in full. Carrying warm-up work into a
  timed pass is this offence. The workflow's prefill ceiling can refuse such a
  run when the gain shows it.
- **Hardcoded content.** Hidden prompts, token ids or answers.
- **Timing and protocol tricks.** Timing shortcuts, protocol injection, moving
  work across the prefill/decode boundary (section 5.1.1), network access, and
  filesystem exfiltration.
- **Format changes.** Any change to a frozen format (section 3.4) or to the MTP
  head (section 4.2).
- **Out-of-surface edits.** Any change outside `editablePaths`.

Input-independent work stays legal: weight repacks, precomputed shape-keyed
tables, CUDA graph capture at boot, and KV and state reuse WITHIN one request.
Keep every change prompt-independent and model-general. The hidden prompts
differ from the public one.

## 9. Submitting

Use the Yukon CLI for every submission operation. A submission archive packages
only `editablePaths`, and it rejects symlinks and generated artifacts. Keep
`harness/protocol-adapter/target/` and `__pycache__/` out of it, since they
count against the byte cap. The ranked run is the only gate that ranks a
submission.

## 10. License

The pinned checkpoint is NVIDIA's, under the OpenMDW License Agreement v1.1
(OpenMDW-1.1) that ships with it at the pinned revision. vLLM is Apache-2.0 (`vllm/LICENSE`). This
repository distributes no model weights.

## 11. Speculative decoding and the serial stream

11.1 The target decides every token. A speculative cycle commits only drafts
equal to the target's own greedy choice, plus the target's bonus token.

11.2 The verify forward runs d + 1 positions in one pass, and the serial path
runs one. They are different kernels over the same weights. On this checkpoint
they also quantize the activations to FP8 with scales computed over the whole
batch, so their logits differ by more than the last bits: replayed through the
serial engine, the reference's own MTP streams sit up to 1.6 logits below the
serial top logit.

11.3 Such a difference changes a token only where the top logits are close. BF16
logits make exact ties common (section 5.4.1).

11.4 Four consequences follow for the timed oracles:
(a) the serial tape is the fidelity oracle for every depth (section 5.4);
(b) the timed free-run oracle is authored under the same spec the leg replays,
one golden per depth, so a leg is never checked against another depth's
trajectory;
(c) a free-run trajectory can fork from the serial one at a near-tie, which is
why each depth has its own timed oracle and why the gate prices divergence
rather than forbidding it;
(d) your engine can fork from the reference's trajectory at a near-tie, and the
timed window then fails, because it requires an exact match (section 5.4.1).

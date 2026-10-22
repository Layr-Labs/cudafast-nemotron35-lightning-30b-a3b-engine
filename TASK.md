# The task — Nemotron 3.5 Lightning 30B-A3B NVFP4 on DGX Spark

Make the target model run faster on a DGX Spark (GB10, sm_121), single stream.

The target is `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` at revision
`bee7596271d1495f6992ae224aefde4410e816b8`. It is a 52-layer hybrid: 23
Mamba-2 layers, 23 MoE layers (128 routed experts, 6 per token, plus a shared
expert) and 6 attention layers, with one native MTP layer embedded in the
checkpoint.

The engine is vLLM v0.27.1. Its Python source is vendored at `vllm/vllm/`, and
it runs on top of the compiled extensions in the pinned engine container. The
scored binary is the Engine Protocol v1 adapter (`harness/protocol-adapter`).
It attaches to one vLLM resident per leg (`harness/vllm_resident`).

The ranked track is `nemotron3.5-lightning-30b-a3b-cuda-v1`. Read
[README.md](README.md) for setup. Read
[`docs/participant-contract.md`](docs/participant-contract.md) for the reasons
behind every rule.

## What you may change

`benchmark.json` `editablePaths` is the authority. Its editable paths are
`vllm`, `harness` and `mtp-head.manifest.json`. A submission archive replaces
those paths and nothing else.

| Path | What it is |
|---|---|
| `vllm/` | The vendored vLLM Python and Triton source: model code (`nemotron_h.py`, `nemotron_h_mtp.py`), Mamba and MoE layers, attention backends, the scheduler, the speculative-decoding loop. |
| `harness/` | The Rust protocol adapter, the Python resident, and optional custom CUDA/C++ kernels under `harness/kernels/`. A `harness/kernels/build.sh` runs in the pinned container at setup, and its output goes on the resident's `PYTHONPATH`. |
| `mtp-head.manifest.json` | The speculative-decoding declaration: `spec.enabled` and `spec.num_speculative_tokens`. |

The rule behind the list is simple. Code that **proposes** tokens or
**computes** the forward pass is editable. Code that **verifies**, **measures**
or **ledgers** stays trusted: `tools/`, `fixtures/`, `setup.sh`,
`benchmark.sh`, `.github/`, the pinned snapshot, the pinned container and
benchd.

## What stays fixed

> **WARNING — the target's numerical formats are frozen.**
> The checkpoint's own `hf_quant_config.json` declares the formats. Routed and
> shared experts and `lm_head` are NVFP4 weight-only, which means W4A16:
> 16-element blocks with an FP8 E4M3 scale and an FP32 global scale. The Mamba
> in/out projections are FP8. The KV cache is FP8. The Mamba SSM state is
> float32.
>
> Do not re-quantize, re-represent or change the format of any target weight,
> activation path or cache. That includes quantizing activations to FP4.
> This holds even when the result passes the correctness gate. You may write
> faster kernels that compute the same formats: fused dequant-GEMM, a better
> MoE, a better Mamba scan, CUDA graphs, fusion, scheduling.

> **WARNING — the MTP head is the organizer's pinned weights.**
> The head is the `mtp.*` tensors in the pinned checkpoint: BF16, 270 tensors,
> 2,670,652,160 bytes. You may not replace it, re-quantize it, or ship head
> weights of your own. `mtp-head.manifest.json` accepts `"source": "pinned"`
> only.

- **The scored batch size is 1.** It is not a tunable.
- **The only speculative arm is the native MTP head,** run as vLLM's `mtp`
  method. DSpark, DFlash and Eagle drafters are not part of this track.
- **The draft depth is yours to choose, from 1 to 6.** A disabled declaration
  or depth 0 runs serial. The non-editable envelope refuses 7 and above. Each
  run seals `effective_spec_mode` and `effective_spec_depth`, along with the
  per-cycle `acceptance_lengths` and `spec_acceptance_rate`.
- **The engine configuration comes from the fixture:**
  `serve_configuration.engine_env` sets the context of 8192 tokens, the
  batched-token budget and the memory fraction. The resident's own code decides
  everything else vLLM does. The resident runs with no network, with the
  snapshot and your tree mounted read-only, and with a fresh cache directory
  for every leg.
- **The KV backend is `vllm-paged`.**

## How to run it

```bash
./tools/fetch-benchd.sh
```

This command resolves and verifies the pinned benchmarker binary.

```bash
./setup.sh
```

This command builds the adapter and pulls the pinned engine image by digest. It
stages the vLLM overlay (the container's vLLM with your `vllm/vllm/` on top),
builds `harness/kernels/` if present, and downloads and verifies the target
snapshot. It needs cargo, docker with the NVIDIA container runtime, jq,
python3, curl and sha256sum.

```bash
mkdir -p "$HOME/rsock" && export SERVE_UP_SOCKET_DIR="$HOME/rsock"
./tools/local-baseline.sh
```

This command runs one local leg through benchd on the public golden. The
resident's socket must not be under `/tmp`, which the container hides.

```bash
cargo test --manifest-path harness/protocol-adapter/Cargo.toml
python3 -m unittest discover -s harness/vllm_resident
```

These commands run the adapter's and the resident's own tests with no GPU.

## How it scores

```text
composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75
gain      = baseline_aggregate / candidate_aggregate
```

The score is serial-anchored, so a faster candidate scores above 1.

- **Pairs.** A ranked run measures 3 pairs on one box in one job. Each pair is
  one serial-control leg on the organizer-staged reference tree, then one
  candidate leg at your declared depth.
- **Workload.** Every leg is single-stream: a 496-token prompt (one unmeasured
  warm-up prefill, then five timed prefills of the same prompt, scored by their
  median) and a 128-step decode. The prompts are the fixture's `live_goldens`:
  three hidden coding prompts. Each one is a user message in the model's own
  chat template, with thinking off.
- **Every job measures every live golden.** Pair k measures prompt
  (k - 1) mod 3. Both legs of a pair measure the same prompt, so the ratio
  stays comparable.
- **Decode.** Decode time per token is the decode window: the time of
  `free_decode_run(N)` divided by N. The seed prefill is not part of it.
  Prefill is its own phase.
- **Aggregation.** Each pair has its own composite. The score is the mean of
  the 3 per-pair composites. Every pair must clear both floors.
- **Bounds.** Both floors are 0.95, so a candidate that regresses prefill or
  decode by more than 5 percent is refused. The ceiling is 5.0.
- **Correctness.** The per-stream token-tolerance gate prices divergence from
  the serial tape against a 10% budget.

> **WARNING — the gate accepts similar output, not identical output.** The
> track does not require token-for-token equality with the serial trajectory,
> but a change that moves the numerics far enough fails the gate.

## Ranking

- **Each ranked job measures every live golden** (one pair per prompt).
- **The live goldens are replaced every 3 to 4 days** by prompts no submission
  has seen.
- **Submissions are unlimited.**
- **The final ranking is a re-measurement.** At the close, each team's best
  submission is taken: the mean contest score counts every run that scored or
  failed because of the submission (a team failure counts as 0.0; a post-setup
  reference-leg or band failure counts as 0.0; other infrastructure runs are
  excluded). The top N teams (default 10) are selected. Each selected
  submission is re-measured K times (default 5), one run per slot, across
  different boxes on a held-out pool that was never staged during the contest.
  Each slot is its first run that ends with a score or a team failure;
  infrastructure failures are re-run. In the final re-score only, one
  re-dispatch is allowed after the first candidate boot failure per slot. A run
  that fails because of the submission,
  or a reference-leg or band failure that persists after re-runs, counts as 0.0
  for that slot. The final ranking is the mean of the K slot values (the score,
  or 0.0 for a team failure or penalized post-setup infrastructure slot).

[The contract, section 5.7](docs/participant-contract.md#57-pool-refresh-and-final-ranking)
states the same rules.

## Local runs are directional

A local run uses your machine, not a ranked Spark. Treat a local score as a
smoke signal, not as a prediction. The ranked box run is the authority.

## Authorities

| Question | File |
|---|---|
| Editable paths, commands, scoring values | `benchmark.json` |
| Pins, engine configuration, depths, scoring semantics | `fixtures/nemotron35_lightning_30b_a3b_track.json` |
| Why each rule says what it says | `docs/participant-contract.md` |
| What a measured run executes | the channel benchmarker (`tools/fetch-benchd.sh`) |

Where this document and the contract fixture disagree, the fixture wins. Where
either disagrees with the benchmarker about measurement, the benchmarker wins.
CI (`tools/check-doc-consistency.py`) fails when this document drifts from
`benchmark.json` or the fixture.

# Nemotron 3.5 Lightning on DGX Spark: inference speed challenge

A ranked engineering challenge: make NVIDIA's
`Nemotron-3.5-Lightning-30B-A3B-NVFP4` prefill and decode faster on a DGX Spark,
single stream, without changing what the model computes.

**[Overview](OVERVIEW.md)** · **[Task and rules](TASK.md)** ·
**[Participant contract](docs/participant-contract.md)** ·
**[Sandbox design](docs/sandbox-design.md)** ·
**[Pool refresh and final ranking](docs/pool-refresh-and-final-rescore.md)**

## At a glance

| | |
|---|---|
| **Model** | [`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4), pinned by revision and per-file sha256 |
| **Hardware** | NVIDIA DGX Spark (GB10, sm_121, 128 GB unified memory) |
| **Engine** | vLLM v0.27.1 in a pinned container; teams edit `vllm/`, `harness/` and the speculation depth |
| **Score** | `prefill_gain^0.25 × decode_gain^0.75`, each gain measured against a trusted reference run on the same box, same prompt, same job |
| **Correctness** | output must match the reference golden within a 10% token budget; the timed decode window must match its per-depth oracle exactly |
| **Fairness** | run-to-run spread is small: the unchanged engine (MTP 3) scores 1.314 on average, sd 0.004 (11 runs, 3 boxes, 2026-10-20); the reference against itself scored 0.996, sd 0.005 (September soak, 8 boxes) |
| **Integrity** | sandboxed, network-less execution; hidden prompts rotated every 3–4 days; final ranking re-measured on a held-out pool |
| **Track** | `nemotron3.5-lightning-30b-a3b-cuda-v1` |

[OVERVIEW.md](OVERVIEW.md) explains each of these in one page: how scoring
works, what prevents cheating, and how the system was tested.

## What runs

```text
benchd (pinned prebuilt, trusted)
  │  spawns per phase, Engine Protocol v1 (NDJSON over stdio)
  ▼
.build/release/mlxfast-runtime-worker      ← harness/protocol-adapter (Rust, editable)
  │  NDJSON over a Unix socket
  ▼
vLLM resident, one per leg                 ← harness/vllm_resident (Python, editable)
  │  runs inside the pinned container: --network none, snapshot and tree read-only
  ▼
vLLM v0.27.1: compiled extensions from the pinned image,
Python/Triton source from vllm/vllm/       ← vendored (editable)
  ▼
the pinned safetensors snapshot            ← trusted, verified by bytes + sha256
```

- **The resident loads the model once per leg.** benchd spawns a fresh adapter
  for each phase (warmup, timed prefill, timed decode, correctness), and each
  adapter connects instead of loading.
- **benchd boots each leg's resident itself,** through that leg's own
  `tools/serve-up.sh --boot`, and stops it with `--stop`.
- **The adapter reports raw counters only.** benchd does all timing and
  scoring.

## Setup

The toolchain needed:

- cargo
- docker with the NVIDIA container runtime
- jq, python3, curl, sha256sum

A DGX Spark has all of them after the standard DGX OS setup plus rustup.

```bash
./tools/fetch-benchd.sh
```

This command resolves the pinned benchd from its release channel and verifies
the bytes against the channel manifest. The channel is the `main` branch of the
public `Layr-Labs/benchd` repository; the fetch needs no token. The organizer's
development channel publishes the same bytes. A verified pair in `benchd-bin/`
or in `BENCHD_DIST_LOCAL` needs no network.

```bash
./setup.sh
```

This command runs `tools/engine/build.sh` and then verifies the target
snapshot. The engine build does four things:

1. builds the adapter and stages it at `.build/release/mlxfast-runtime-worker`;
2. pulls the pinned image `vllm/vllm-openai:v0.27.1` by its arm64 digest;
3. stages `.build/vllm-overlay`: the image's installed vLLM, with this tree's
   `vllm/vllm/` copied over it;
4. builds `harness/kernels/` in the container when
   `harness/kernels/build.sh` exists.

On a ranked box the adapter build (and any `build.rs`) and the kernels build
run as an unprivileged sandbox user with no network. Cargo dependencies must
come from crates.io, and are vendored before an offline build; `build.sh` may
write only to `$KERNELS_OUT`. [The contract, section
3.6](docs/participant-contract.md#36-where-your-code-runs) lists the rules.

The snapshot step downloads the 60 pinned files (about 21.6 GB) into
`reference_weights/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/`. Each file
must match its pinned byte count and sha256. On a ranked box the snapshot is
pre-staged and named by `MLXFAST_TARGET_SNAPSHOT_DIR`, and `setup.sh` only
verifies it.

```bash
mkdir -p "$HOME/rsock" && export SERVE_UP_SOCKET_DIR="$HOME/rsock"
./tools/local-baseline.sh
```

The resident's socket must not be under `/tmp`, which the container hides, so
set `SERVE_UP_SOCKET_DIR` first (it also applies to `yukon run`). This command boots a resident for your declared depth. It then runs benchd's
local-iterate leg over the public golden of that depth, which the organizer
recorded on a Spark: `correctness_prompts/public-longcopy-gate-english-496.golden.json`
for serial, `correctness_prompts/public-longcopy-gate-english-496.mtp<N>.golden.json`
for depth N. On this engine a depth-N run takes another token path than the
serial run, so each depth has its own golden. A local run seals `score: null`:
only the paired ranked path produces a score.

```bash
yukon setup
```

This command runs `setupCommand`: `./tools/fetch-benchd.sh`, then
`./setup.sh`.

```bash
yukon run
```

This command runs `benchmarkCommand`. Off a ranked box it runs
`./tools/local-baseline.sh` and seals `score.json` with `"score": null`. Only
the ranked runner gives a score.

## Iterating without a GPU

```bash
cargo test --manifest-path harness/protocol-adapter/Cargo.toml
cargo build --release --manifest-path harness/protocol-adapter/Cargo.toml
python3 -m unittest discover -s harness/vllm_resident -v
```

These commands test the adapter against its mock engine, and the resident's
wire and bookkeeping against a fake backend. `test_adapter_e2e` drives the real
adapter binary against the real resident at every depth from 0 to 6.

```bash
tools/three-flows-dry-run.sh
```

This command runs the real benchd binaries (the pinned `main` channel) over the
same fake resident. It covers golden authoring, a local leg at serial and at an
MTP depth, and the paired lifecycle.

## Repository structure

| Path | Trust | What it is |
|---|---|---|
| `vllm/vllm/` | editable | vLLM v0.27.1 Python source (`vllm/VENDOR.json` records the base) |
| `harness/protocol-adapter/` | editable | Engine Protocol v1 adapter (Rust) |
| `harness/vllm_resident/` | editable | the resident: `resident.py` (wire, bookkeeping), `backends.py` (vLLM + fake) |
| `harness/kernels/` | editable, optional | custom CUDA/C++ kernels built by `harness/kernels/build.sh` |
| `mtp-head.manifest.json` | editable | the speculative-decoding declaration |
| `fixtures/nemotron35_lightning_30b_a3b_track.json` | trusted | the track contract: pins, engine configuration, depths, scoring semantics |
| `benchmark.json` | trusted | editable paths, commands, byte budget, scoring constants |
| `setup.sh`, `benchmark.sh`, `tools/` | trusted | build, serve, preflight, measure, calibrate, stage, organizer tools |
| `correctness_prompts/` | trusted | the public prompt text and the public golden |
| `.github/` | trusted | CI (`ci.yml`) and the ranked pipeline (`benchmark.yml`) |
| `docs/` | — | the contract, the resident wire, CI coverage, the sandbox design, the submission restriction spec, pool refresh and final re-score, the engineering log |

## Submitting

Use the Yukon CLI for every account operation and every submission operation.

```bash
export PATH="${HOME}/.local/bin:${PATH}"
```

This command puts `yukon` on your path.

```bash
yukon login <api-key> --api https://api.yukon.org
```

This command authenticates you.

```bash
yukon clone GumbiiDigital/cudafast-nemotron35-lightning-30b-a3b
```

This command clones the benchmark repository.

Before you submit, remove build artifacts from the editable paths. The archive
packages `vllm/`, `harness/` and `mtp-head.manifest.json` as they sit on disk.
The unchanged editable surface is 35,021,070 bytes; the byte budget
(`editableSurfaceByteBudget.maxTotalBytes` in `benchmark.json`) is 36,047,190
bytes. A stale adapter `target/` tree or a `__pycache__/` directory uses that
room and count against the budget.

```bash
cargo clean --manifest-path harness/protocol-adapter/Cargo.toml
find vllm harness -name __pycache__ -type d -prune -exec rm -rf {} +
```

These commands remove the adapter's build tree and Python bytecode caches.
`./setup.sh` and the ranked box rebuild what they need from source.

```bash
yukon submit --model "<exact model name>" --harness "<harness name>" --note-file submission-note.md
```

This command uploads your editable-path archive. The note is public markdown.

```bash
yukon submissions
```

This command lists your submissions.

A submission archive replaces the editable paths. The static review enforces
the byte budget and refuses symlinks and special files, and the intake keeps
only the editable paths
([docs/submission-restriction-spec.md](docs/submission-restriction-spec.md)). `yukon submit` does not run a local test first, and no local
run blocks the upload. Run `yukon run` yourself before you submit.

## The pinned artifacts

| Artifact | Pin |
|---|---|
| Target | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` @ `bee7596271d1495f6992ae224aefde4410e816b8`, 60 files by bytes + sha256 |
| MTP head | the `mtp.*` tensors inside `model-00052-of-00052.safetensors` (BF16, 2,670,652,160 bytes) |
| Engine source | vLLM `v0.27.1` @ `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Engine image | `docker.io/vllm/vllm-openai:v0.27.1` @ `sha256:1c8e60a0841b333c700488cb029d3664807249da0c071e862191b00fe34b228c` (arm64) |
| benchd | `Layr-Labs/benchd` dist `92da41a15ed7e80896c67033dbcf813a88fa9dc8`, built from source `918b8acaf122fb4bedef1f79f580c7c23c65b905` (public snapshot `0aacae37f70e681d2465fff23856665a04bc44cd`; `PINNED_COMMIT`, `PINNED_SOURCE_COMMIT` and `PINNED_PUBLIC_SOURCE_COMMIT` in `tools/fetch-benchd.sh`); the dist's `dist/linux-aarch64/benchd.manifest.json` pins each binary's sha256 and bytes |
| Goldens | `timed_prompt_pool[]` (the 3 serial goldens that `live_goldens` names: `pool-9`, `pool-10`, `pool-11`, three 496-token chat-template coding prompts), `speculative_oracles` (18 per-depth oracles) and `hidden_correctness_golden` (the `pool-9` serial pin) in the fixture: 21 files pinned by sha256 and bytes (staged on each box, never committed; see [OVERVIEW.md](OVERVIEW.md)) |

## The ranked run

`.github/workflows/benchmark.yml` is dispatched once per submission. It runs
two jobs:

1. **A `surface-check` on a Spark runner** (no GPU, no credential). It runs
   the manifest lint and the static review of the editable surface, which
   enforces the byte budget and refuses symlinks and special files.
2. **A `ranked` job on a self-hosted Spark** labelled
   `[self-hosted, Linux, ARM64, nemotron3.5-lightning-30b-a3b-cuda-v1]`. It
   runs these steps in order:
   - `tools/ranked-box-preflight.sh`
   - `./setup.sh`
   - `tools/draw-live-golden.sh`: the fixture declares `live_goldens`, so
     nothing is drawn. Every job measures every live golden, one pair per
     prompt; `score.json` identifies a golden only by its sha256.
   - under the box-wide GPU lock and a quiescence wait,
     `tools/nemotron35-measure-and-score.sh`, which execs
     `benchd iterate --mode official` and seals `score.json`
   - after sandbox cleanup (no participant process remains), **Reveal the
     sealed score** prints a commit-reveal block to the job log: a per-job
     nonce drawn before sandbox start (runner-private, never printed earlier),
     the exact bytes of `score.json` if present, and an end line binding
     nonce to the file's sha256 and byte length (or `none` when absent).
   - **Upload score artifact** then uploads that same `score.json` as
     `benchmark-results-<run_id>/score.json` (the `benchmark.json`
     `scoreArtifact`), only when the reveal step printed it after a clean
     sandbox cleanup. The reveal block stays the tamper-evident record
     ([docs/submission-restriction-spec.md](docs/submission-restriction-spec.md)
     §10).

The ranked job holds no credential. Goldens, the reference workspace, the
calibration file and the benchd pair are staged on the box out of band.

Participant code (the builds, the resident and the adapter) runs as an
unprivileged sandbox user with no network, reading the checkout read-only and
writing only its own per-job directories. The adapter's stderr is not
returned, the score record carries no engine-supplied text, and nothing
persists between jobs or legs.

### The ranked box toolchain

| | |
|---|---|
| **Hardware** | NVIDIA DGX Spark (GB10, aarch64) |
| **OS** | Ubuntu 24.04.4 LTS, kernel 6.17.0-1026-nvidia |
| **Driver** | NVIDIA 580.159.03 (the fixture's `driver_min`), CUDA driver API 13.0; Docker 29.2.1, NVIDIA Container Toolkit 1.19.1 |
| **Engine image** | `docker.io/vllm/vllm-openai:v0.27.1` @ `sha256:1c8e60a0841b333c700488cb029d3664807249da0c071e862191b00fe34b228c` (Ubuntu 22.04.5 LTS) |
| **In the image** | vLLM 0.27.1, PyTorch 2.13.0+cu130, FlashInfer 0.6.16.post3, Python 3.12.13; CUDA 13.0 (nvcc V13.0.88), gcc 11.4.0 |
| **CUDA arch** | `sm_121` (the fixture's `serve_configuration.toolchain.cuda_arch`) |
| **Rust on the host** | 1.98.1 (`1.98.1-aarch64-unknown-linux-gnu`), offline `cargo build --release --locked` on vendored crates.io dependencies |

The host has no nvcc. `harness/kernels/build.sh`, when present, runs inside the
pinned image. The adapter (`harness/protocol-adapter`) builds on the host. The
vLLM Python and Triton source in `vllm/vllm/` runs on the compiled extensions
of the pinned image.

The hosted CI does not run the engine on a GPU. A green hosted CI does not
prove that your kernels compile for `sm_121` with nvcc 13.0, or that the
resident boots. The ranked job builds your candidate in `./setup.sh`; a build
failure there fails the submission before any scoring.

## Organizer runbook

1. **Stage each Spark.**
   - `tools/box-runner/install.sh` sets up the runner supervisor
     (`tools/box-runner/README.md`).
   - `sudo tools/box-runner/setup-sandbox-user.sh` creates the sandbox user
     once per box ([sandbox design, section 6](docs/sandbox-design.md#6-operator-steps)).
   - Pre-pull the engine image by digest.
   - Stage the snapshot and export `MLXFAST_TARGET_SNAPSHOT_DIR`.
   - Stage benchd with `tools/fetch-benchd.sh`.
2. **Stage the goldens** on every box. The fixture pins every golden of the
   live pool (serial golden plus six per-depth oracles per prompt);
   `tools/pool-refresh.sh` records, stages and pins a new generation, and
   a box is restaged from the jump box copy
   ([where the goldens live](docs/pool-refresh-and-final-rescore.md#where-the-goldens-live)).
3. **Stage the reference tree** on every box with
   `tools/stage-baseline-workspace.sh`, at the fixture's
   `baseline_reference_commit`.
4. **Calibrate every box** with `tools/calibrate-box.sh <runner name> <out>`
   (3 passes; the box name must be the box's runner name, which the ranked
   job passes as `--box`). Every per-prompt CV must be at most the fixture's
   `calibration_max_cv_percent` (2%), or the file gets no band
   ([reproduce, section 3](docs/reproduce.md#3-recalibrate-a-box)).
5. **Arm the track:** set `official_scoring_enabled: true` once the null
   control lands at about 1.000 ([contract, section 5.5](docs/participant-contract.md#55-arming)).
6. **Run the contest.** The intake service dispatches `benchmark.yml` once per
   submission. Every 3 to 4 days, replace the live pool; at the close, freeze
   and run the final re-score on the held-out pool. Both procedures are in
   [`docs/pool-refresh-and-final-rescore.md`](docs/pool-refresh-and-final-rescore.md).
   `tools/router/` is the organizer's own dispatcher, used for load tests and
   for the final re-score (`tools/router/README.md`).

[`docs/nemotron35-port-notes.md`](docs/nemotron35-port-notes.md) (the
engineering log) records the cluster verification and the load-test findings.

## Documentation

| Question | File |
|---|---|
| Editable paths, commands, scoring values | `benchmark.json` |
| Pins, engine configuration, scoring semantics | `fixtures/nemotron35_lightning_30b_a3b_track.json` |
| Why the rules say what they say | `docs/participant-contract.md` |
| The resident wire | `docs/resident.md` |
| What CI covers | `docs/ci-coverage.md` |
| How participant code is sandboxed on a box | `docs/sandbox-design.md` |
| What a submission may contain, and who enforces it | `docs/submission-restriction-spec.md` |
| Pool refresh and final re-score | `docs/pool-refresh-and-final-rescore.md` |
| How this track was derived, and what was measured | `docs/nemotron35-port-notes.md` (engineering log) |
| How to reproduce benchd, the goldens, a calibration and a ranked score | `docs/reproduce.md` |
| What a measured run executes | benchd, resolved from the pinned channel by `tools/fetch-benchd.sh` |

# What CI covers

`.github/workflows/ci.yml` runs on manual dispatch only (`workflow_dispatch`):
its jobs need GitHub-hosted runners, so run the same checks locally before a
push (`AGENTS.md`, "Checks before you submit"). It holds no credential.

## `lint` (x86 ubuntu)

| Step | What it proves |
|---|---|
| egress scan | no workflow references a secret, inherits one, asks for write permission, or uploads anything but the ranked `score.json` (the ranked workflow uploads that one file, after sandbox cleanup) |
| actionlint | the workflows parse and type-check (pinned binary, sha256-verified) |
| shellcheck | every tracked `*.sh` outside `vllm/` passes at warning severity |
| manifest lint | `benchmark.json` against the fixture: schema, paths, commands, scoring constants, byte-budget fallbacks, per-leg serve verbs |
| benchd platform / recorder | `tools/fetch-benchd.sh` refuses a wrong platform or an unverified second binary |
| adapter | `cargo fmt`, `clippy -D warnings` and `cargo test` (mock and scripted sessions) |
| docs agree with the manifest | `tools/check-doc-consistency.py` |
| participant engine edit | the editable list, benchd's harness-hash roster, and that a `vllm/vllm/` edit reaches the staged overlay while compiled extensions survive |
| serve-up boot/stop, memory plan | the real `serve-up.sh` and the real resident (fake backend) |
| facade, local baseline, setup onboarding | the entry points, through a real resident boot |
| yukon run | `benchmarkCommand` reaches the ranked script on a ranked box and `tools/local-baseline.sh` off one |
| spec declaration | the declaration envelope |
| ranked preflight | every environment refusal, and the section 8 paired-baseline refusals on a scaffolded box |
| calibrator, stager, golden fetcher, box runner | their mechanics, with stub collaborators |

## `resident` (x86 ubuntu)

The Python resident's unit and wire tests, and `test_adapter_e2e.py`. That test
runs the real `cuda-engine` against the real resident at depths 0 to 6 and
requires the serial stream at every depth.

## `engine-overlay` (arm64 ubuntu, no GPU)

This job does three things:

1. pulls the pinned engine image by digest;
2. runs `tools/stage-vllm-overlay.sh`;
3. imports `vllm.model_executor.models.nemotron_h` and `nemotron_h_mtp` through
   the overlay, asserting that vLLM resolved from the overlay.

## What CI cannot cover

Anything that needs a GPU:

- the engine's numerics;
- speed;
- the goldens;
- calibration;
- the ranked run.

The engineering log (`docs/nemotron35-port-notes.md`) section 6 lists the
on-box verification, and section 7 the measured findings.
`tools/three-flows-dry-run.sh` runs the real benchd binaries over the fake
resident. It is not in CI, because it downloads benchd.

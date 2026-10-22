# Agent guide — Nemotron 3.5 Lightning 30B-A3B NVFP4 engine

This file is the working contract for coding agents in this repository.
`CLAUDE.md` is a symbolic link to it.

Read [README.md](README.md) first and [TASK.md](TASK.md) second. This file adds
the operational rules for iterating here. The ranked track is
`nemotron3.5-lightning-30b-a3b-cuda-v1`.

## Goal

Make the target prefill and decode faster on a DGX Spark, single stream, without
changing what it computes beyond the 10% token-tolerance budget.

## Authorities

| Question | File |
|---|---|
| What you may edit | `benchmark.json` `editablePaths`: `vllm`, `harness`, `mtp-head.manifest.json` |
| Pins, engine configuration, depths, scoring | `fixtures/nemotron35_lightning_30b_a3b_track.json` |
| Why the rules say what they say | `docs/participant-contract.md` |
| The resident wire | `docs/resident.md` |

## Where to work

| Lever | Where |
|---|---|
| Speculation depth | `mtp-head.manifest.json` `spec` (1 to 6) |
| Engine arguments (CUDA graphs, compile level, scheduler, async scheduling) | `harness/vllm_resident/backends.py` `VllmBackend` |
| Per-step overhead between benchd and vLLM | `harness/vllm_resident/resident.py`, `harness/protocol-adapter/src/` |
| Model code | `vllm/vllm/model_executor/models/nemotron_h.py`, `nemotron_h_mtp.py` |
| Mamba-2 | `vllm/vllm/model_executor/layers/mamba/` |
| MoE (NVFP4 W4A16) | `vllm/vllm/model_executor/layers/fused_moe/`, `.../quantization/` |
| Speculative loop | `vllm/vllm/v1/spec_decode/`, `vllm/vllm/v1/worker/gpu/spec_decode/` |
| New CUDA kernels | `harness/kernels/` + `harness/kernels/build.sh` |

## Rules

- **Never edit outside `editablePaths`.** That covers `tools/`, `fixtures/`,
  `setup.sh`, `benchmark.sh`, `benchmark.json`, `.github/` and
  `correctness_prompts/`, all trusted. A submission that touches them is
  refused.
- **Never change a numerical format.** Section 3.4 of the contract lists them.
  Keep NVFP4 W4A16 experts and `lm_head`, FP8 Mamba projections, the FP8 KV
  cache, the fp32 SSM state and the BF16 MTP head exactly as declared.
- **Never reuse work across requests or phases.** That rules out prefix
  caching, memoized prompts and state kept across `sync`. The resident's
  `invalidate` must leave the engine with nothing from the previous phase.
  The timed prefill follows one unmeasured warm-up prefill of the same prompt
  in the same phase; compute both in full and carry nothing between them.
- **Never move prefill work out of `free_decode_begin`** (contract section 5.1.1).
- **The timed window demands an exact match.** If your timed stream leaves
  the golden at any token, even at a logit tie, the run fails (contract
  section 5.4.1). The ranked run and the local modes apply the same rule.
- **Keep the engine reproducible.** Run `tools/determinism-probe.py` against a
  booted resident; it must report one stream at serial and at your depth. The
  stock Marlin MoE is NOT reproducible on GB10
  (`docs/nemotron35-port-notes.md` section 7.2).
- **Keep the resident wire intact.** The Rust client in
  `harness/protocol-adapter/src/resident.rs` and `resident.py` must agree.
  `test_adapter_e2e.py` proves it at every depth.
- **Keep `IMPLEMENTED_DEPTH` in `resident.py` and `ENGINE_IMPLEMENTED_DEPTH` in
  `session_backend.rs` equal.** Tests on both sides check it.
- **Report raw counters only.** The adapter never reports a rate, a time or a
  ratio. benchd does all measurement.
- **Keep `harness/protocol-adapter/target/` and `__pycache__/` out of a
  submission.** They count against the byte cap.

## Checks before you submit

```bash
cargo fmt --manifest-path harness/protocol-adapter/Cargo.toml -- --check
cargo clippy --manifest-path harness/protocol-adapter/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path harness/protocol-adapter/Cargo.toml
cargo build --release --manifest-path harness/protocol-adapter/Cargo.toml
python3 -m unittest discover -s harness/vllm_resident
python3 tools/lint-benchmark-manifest.py
```

These commands check the adapter, the resident and the manifest with no GPU.

```bash
mkdir -p "$HOME/rsock" && export SERVE_UP_SOCKET_DIR="$HOME/rsock"
./setup.sh && ./tools/local-baseline.sh
```

This command runs one real leg. It needs a Spark or another sm_121 box with the
pinned image.

## Environment rules on a ranked box

Never export the following in a ranked job; `tools/ranked-box-preflight.sh`
refuses each of them by name:

- **Resident sockets:** `DS4_RESIDENT_SOCKET` and
  `BENCH_WORKER_RESIDENT_SOCKET`. benchd injects them per leg, and an inherited
  one is a third resident.
- **Engine configuration:** anything prefixed `RESIDENT_`, `VLLM_`, `HF_`,
  `TRANSFORMERS_`, `TORCHINDUCTOR_` or `TRITON_`. The fixture's `engine_env`
  is the only engine configuration.
- **Test seams:** `SERVE_UP_LAUNCHER` and `STAGE_VLLM_OVERLAY_BASE`.
- **Measurement skips:** `BENCHD`, `MLXFAST_SKIP_*` and
  `MLXFAST_LOCAL_COOL_GATE`.

## Notes

Keep your own notes, such as what sped the engine up and what the gate refused,
in an untracked local file such as `notes.local.md` (add it to your local git
ignore, for example `.git/info/exclude`). Never commit anything outside
`editablePaths`: that includes this file and `docs/`.

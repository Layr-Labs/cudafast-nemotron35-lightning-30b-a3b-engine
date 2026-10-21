# harness/

The participant-editable engine adapter.

| Path | What it is |
|---|---|
| `protocol-adapter/` | The Engine Protocol v1 adapter, in Rust. benchd spawns it once per phase as `.build/release/mlxfast-runtime-worker` and speaks NDJSON to it over stdio. It attaches to the leg's resident through the socket benchd injects (`BENCH_WORKER_RESIDENT_SOCKET`, or the legacy `DS4_RESIDENT_SOCKET`). With no socket, it serves a deterministic mock and says so. |
| `vllm_resident/` | The resident, one process per leg. `resident.py` owns the socket wire and the frontier bookkeeping. `backends.py` holds `VllmBackend`, the scored engine, and `FakeBackend`, for tests. `tools/serve-up.sh` boots it in the pinned image. |
| `kernels/` (optional) | Custom CUDA/C++ kernels. `kernels/build.sh` runs in the pinned image at setup with `KERNELS_OUT` and `TORCH_CUDA_ARCH_LIST` set. Its output is importable from the resident. |

The adapter reports raw counters only: tokens, `acceptance_lengths`, and the
drafted, accepted and committed totals. benchd times every verb itself.
`docs/resident.md` documents the socket wire, and the protocol schema is
`protocol-adapter/reference/engine-protocol-v1.schema.json`.

## Tests (no GPU)

```bash
cargo test --manifest-path harness/protocol-adapter/Cargo.toml
cargo build --release --manifest-path harness/protocol-adapter/Cargo.toml
python3 -m unittest discover -s harness/vllm_resident -v
```

These commands test each side on its own. `test_adapter_e2e.py` then runs the
real `cuda-engine` against the real resident (fake backend) at depths 0 to 6,
and requires the committed stream to equal the serial stream.

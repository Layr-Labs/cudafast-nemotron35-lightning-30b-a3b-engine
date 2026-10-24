#!/usr/bin/env bash
# Diagnostic live-weight launch timing; never invoked by the resident.
set -euo pipefail
base=$1
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cd "${2:-$PWD}"
exec 9>/tmp/mtplx-gpu-exclusive.lock
flock 9
image=$(jq -r '.serve_configuration | .container_image+"@"+.container_digest' fixtures/nemotron35_lightning_30b_a3b_track.json)
test=${MTP_SWEEP_TEST:-mtp_compact_launch_sweep}
run="$PWD/.local-run/$test"
mkdir -p "$run"
docker run --rm --gpus all --ipc=host --network none --user "$(id -u):$(id -g)" \
  -v "$base:$base" -w "$PWD" -e HOME="$run" -e XDG_CACHE_HOME="$run/cache" \
  -e PYTHONPATH="$PWD/.build/vllm-overlay:$PWD/.build/kernels" \
  -e NEMO_SNAPSHOT="$base/reference_weights/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4" \
  -e NEMO_RESULT="$run/result.json" -e VLLM_BATCH_INVARIANT=1 -e VLLM_NO_USAGE_STATS=1 \
  --entrypoint python3 "$image" "harness/tests/$test.py"

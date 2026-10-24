#!/usr/bin/env bash
set -euo pipefail
base=$1
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH" PYTHONDONTWRITEBYTECODE=1
exec 9>/tmp/mtplx-gpu-exclusive.lock; flock 9
until [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do sleep 5; done
run="$PWD/.local-run/screenstop-probe"
mkdir -p "$run"
test ! -e "$run/started"; date -u > "$run/started"
image=$(jq -r '.serve_configuration | .container_image+"@"+.container_digest' fixtures/nemotron35_lightning_30b_a3b_track.json)
control="$base/.local-run/frontier-7fac-draft2"
docker run --rm --gpus all --ipc=host --network none --user "$(id -u):$(id -g)" \
 -v "$base:$base" -w "$PWD" -e HOME="$run" -e XDG_CACHE_HOME="$run/cache" \
 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH="$control/.build/vllm-overlay:$control/.build/kernels" \
 -e NEMO_CONTROL="$control" -e NEMO_SNAPSHOT="$base/reference_weights/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4" \
 -e NEMO_RESULT="$run/result.json" -e VLLM_BATCH_INVARIANT=1 -e VLLM_NO_USAGE_STATS=1 \
 --entrypoint python3 "$image" harness/tests/mtp_screenstop_compare.py

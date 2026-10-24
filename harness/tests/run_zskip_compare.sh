#!/usr/bin/env bash
set -euo pipefail
base=$1
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cd "$base/.local-run/race-4d5563"
exec 9>/tmp/mtplx-gpu-exclusive.lock
flock 9
image=$(jq -r '.serve_configuration | .container_image+"@"+.container_digest' fixtures/nemotron35_lightning_30b_a3b_track.json)
mkdir -p .local-run/zskip-compare-cache
# Keep normal module imports on the untouched live carrier; candidate is loaded
# explicitly from its separate ctypes library for the direct paired comparison.
docker run --rm --gpus all --ipc=host --network none --user "$(id -u):$(id -g)" \
 -v "$base:$base" -w "$PWD" \
 -e HOME="$PWD/.local-run/zskip-compare-cache" -e XDG_CACHE_HOME="$PWD/.local-run/zskip-compare-cache" \
 -e PYTHONPATH="$PWD/.build/vllm-overlay:$PWD/.build/kernels" \
 -e VLLM_BATCH_INVARIANT=1 -e VLLM_HUMMING_BATCH_INVARIANT=1 -e VLLM_NO_USAGE_STATS=1 \
 --entrypoint python3 "$image" harness/tests/zskip_race_compare.py \
 --weights "$base/reference_weights/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4" \
 --control .build/kernels/zskip/libzskip.so --candidate .local-run/libzskip-prmt.so \
 --output .local-run/zskip-race-comparison.json

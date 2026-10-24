#!/usr/bin/env bash
# One paired engine comparison; no timing redraws.
set -euo pipefail
base=/home/frosty40/comps/nemo/cudafast-nemotron35-lightning-30b-a3b
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH" PYTHONDONTWRITEBYTECODE=1
exec 9>/tmp/mtplx-gpu-exclusive.lock; flock 9
until [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do sleep 5; done
export MLXFAST_TARGET_SNAPSHOT_DIR="$base/reference_weights/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4" SERVE_UP_WEIGHTS_DIR="$base/reference_weights/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
if [[ $1 == AB ]]; then order='control coarse16'; else order='coarse16 control'; fi
for flavor in $order; do
 cd "$base/.local-run/frontier-7fac-$flavor"
 mkdir -p .local-run "$HOME/rsock/nemo-7fac-coarse16-$flavor"
 test ! -e .local-run/coarse16-pair.started; date -u > .local-run/coarse16-pair.started
 export SERVE_UP_SOCKET_DIR="$HOME/rsock/nemo-7fac-coarse16-$flavor" MLXFAST_SCORE_PATH=.local-run/score.coarse16-pair.json
 ./setup.sh > .local-run/setup.coarse16-pair.log 2>&1
 cmp vllm/vllm/model_executor/models/nemotron_h_mtp_screen.py .build/vllm-overlay/vllm/model_executor/models/nemotron_h_mtp_screen.py
 ./tools/local-baseline.sh > .local-run/engine.coarse16-pair.log 2>&1
 jq -e '.passed and .metrics.passed_correctness' "$MLXFAST_SCORE_PATH" >/dev/null
 printf '%s %s ' "$(hostname)" "$flavor"
 jq -c '{passed,prefill:.metrics.prefill_seconds_per_token,decode:.metrics.decode_seconds_per_token}' "$MLXFAST_SCORE_PATH"
done

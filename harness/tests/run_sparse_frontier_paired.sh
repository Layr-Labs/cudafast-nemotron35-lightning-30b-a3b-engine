#!/usr/bin/env bash
# Public fixture paired probe. Run in a dedicated setup checkout on a Spark.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
base=${1:?base checkout required}
cd "$base/.local-run/engine-bce7-sparse"
export MLXFAST_TARGET_SNAPSHOT_DIR="$base/reference_weights/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
export SERVE_UP_WEIGHTS_DIR="$MLXFAST_TARGET_SNAPSHOT_DIR"
export SERVE_UP_SOCKET_DIR="$HOME/rsock/nemo-sparse-prmt"
mkdir -p "$SERVE_UP_SOCKET_DIR"
exec 9>/tmp/mtplx-gpu-exclusive.lock
flock -n 9
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || { echo GPU_BUSY; exit 73; }
source=harness/kernels/routed_sparse/csrc/routed_sparse.cu
restore() { cp .local-run/routed-sparse-a9-control.cu "$source"; }
trap restore EXIT
for arm in ${SPARSE_ARMS:-control candidate candidate2 control2}; do
  if [[ "$arm" == control* ]]; then
    cp .local-run/routed-sparse-a9-control.cu "$source"
  else
    cp .local-run/routed-sparse-a9-prmt.cu "$source"
  fi
  ./setup.sh > ".local-run/setup.sparse-prmt-$arm.log" 2>&1
  export MLXFAST_SCORE_PATH=".local-run/score.sparse-prmt-$arm.json"
  ./tools/local-baseline.sh > ".local-run/sparse-prmt-$arm.log" 2>&1
  jq -e '.passed and .metrics.passed_correctness' "$MLXFAST_SCORE_PATH"
  jq '{passed,metrics:{passed_correctness:.metrics.passed_correctness,prefill_seconds_per_token:.metrics.prefill_seconds_per_token,decode_seconds_per_token:.metrics.decode_seconds_per_token}}' "$MLXFAST_SCORE_PATH"
done
printf 'PAIRED_COMPLETE\n'

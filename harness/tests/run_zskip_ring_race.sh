#!/usr/bin/env bash
# Isolated register-ring-depth screen; preserve the supplied complete carrier.
set -euo pipefail
base=$1
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cd "${2:-$PWD}"
export MLXFAST_TARGET_SNAPSHOT_DIR="$base/reference_weights/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
export SERVE_UP_WEIGHTS_DIR="$MLXFAST_TARGET_SNAPSHOT_DIR"
export SERVE_UP_SOCKET_DIR="$HOME/rsock/nemo-zskip-ring-race"
mkdir -p "$SERVE_UP_SOCKET_DIR"
exec 9>/tmp/mtplx-gpu-exclusive.lock
flock 9
round=0
for config in ${RING_CONFIGS:-101 100 100 101}; do
  python3 - "$config" <<'PY'
import re,sys
from pathlib import Path
p=Path('vllm/vllm/model_executor/layers/fused_moe/experts/humming_zskip.py');s=p.read_text();s,n=re.subn(r'^CONFIG = [0-9]+$',f'CONFIG = {sys.argv[1]}',s,flags=re.M);assert n==1;p.write_text(s)
PY
  round=$((round+1))
  tag="zskip-ring-$config-$round"
  ./setup.sh > ".local-run/setup.$tag.log" 2>&1
  export MLXFAST_SCORE_PATH=".local-run/score.$tag.json"
  ./tools/local-baseline.sh > ".local-run/$tag.log" 2>&1
  jq -e '.passed and .metrics.passed_correctness' "$MLXFAST_SCORE_PATH"
  jq '{passed,metrics:{passed_correctness:.metrics.passed_correctness,prefill_seconds_per_token:.metrics.prefill_seconds_per_token,decode_seconds_per_token:.metrics.decode_seconds_per_token}}' "$MLXFAST_SCORE_PATH"
done
echo ZSKIP_RING_ENGINE_RACE_PASS

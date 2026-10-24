#!/usr/bin/env bash
# Live-frontier single-host depth race. No scoring/golden changes.
set -euo pipefail
base=$1
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cd "${2:-$PWD}"
export MLXFAST_TARGET_SNAPSHOT_DIR="$base/reference_weights/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
export SERVE_UP_WEIGHTS_DIR="$MLXFAST_TARGET_SNAPSHOT_DIR"
export SERVE_UP_SOCKET_DIR="$HOME/rsock/nemo-zskip-race"
mkdir -p "$SERVE_UP_SOCKET_DIR"
exec 9>/tmp/mtplx-gpu-exclusive.lock
flock 9
for depth in ${RACE_DEPTHS:-3 2 2 3}; do
 python3 - "$depth" <<'PY'
import json,sys
from pathlib import Path
p=Path('mtp-head.manifest.json');d=json.loads(p.read_text());d['spec']['num_speculative_tokens']=int(sys.argv[1]);p.write_text(json.dumps(d,indent=2)+'\n')
PY
 round=${round:-0}; round=$((round+1))
 ./setup.sh > ".local-run/setup.zskip-depth${depth}-${round}.log" 2>&1
 export MLXFAST_SCORE_PATH=".local-run/score.zskip-depth${depth}-${round}.json"
 ./tools/local-baseline.sh > ".local-run/zskip-depth${depth}-${round}.log" 2>&1
 jq -e '.passed and .metrics.passed_correctness' "$MLXFAST_SCORE_PATH"
 jq '{passed,metrics:{passed_correctness:.metrics.passed_correctness,prefill_seconds_per_token:.metrics.prefill_seconds_per_token,decode_seconds_per_token:.metrics.decode_seconds_per_token}}' "$MLXFAST_SCORE_PATH"
done
echo ZSKIP_DEPTH_RACE_PASS

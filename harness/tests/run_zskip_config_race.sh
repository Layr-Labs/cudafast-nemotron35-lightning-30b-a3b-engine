#!/usr/bin/env bash
set -euo pipefail
base=$1
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cd "$base/.local-run/race-config-compact"
export MLXFAST_TARGET_SNAPSHOT_DIR="$base/reference_weights/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
export SERVE_UP_WEIGHTS_DIR="$MLXFAST_TARGET_SNAPSHOT_DIR"
export SERVE_UP_SOCKET_DIR="$HOME/rsock/nemo-zskip-config"
mkdir -p "$SERVE_UP_SOCKET_DIR"
exec 9>/tmp/mtplx-gpu-exclusive.lock
flock 9
round=0
for config in ${ZSKIP_CONFIGS:-100 105 105 100}; do
  python3 - "$config" <<'PY'
import json,re,sys
from pathlib import Path
p=Path('mtp-head.manifest.json');d=json.loads(p.read_text());d['spec']['num_speculative_tokens']=3;p.write_text(json.dumps(d,indent=2)+'\n')
p=Path('vllm/vllm/model_executor/layers/fused_moe/experts/humming_zskip.py');s=p.read_text();s,n=re.subn(r'^CONFIG = [0-9]+$',f'CONFIG = {sys.argv[1]}',s,flags=re.M);assert n==1;p.write_text(s)
PY
  round=$((round+1))
  ./setup.sh > ".local-run/setup.zskip-config$config-$round.log" 2>&1
  export MLXFAST_SCORE_PATH=".local-run/score.zskip-config$config-$round.json"
  ./tools/local-baseline.sh > ".local-run/zskip-config$config-$round.log" 2>&1
  jq -e '.passed and .metrics.passed_correctness' "$MLXFAST_SCORE_PATH"
  jq '{passed,metrics:{passed_correctness:.metrics.passed_correctness,prefill_seconds_per_token:.metrics.prefill_seconds_per_token,decode_seconds_per_token:.metrics.decode_seconds_per_token}}' "$MLXFAST_SCORE_PATH"
done
echo ZSKIP_CONFIG_RACE_PASS

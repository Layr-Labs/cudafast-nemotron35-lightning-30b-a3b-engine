#!/usr/bin/env bash
# Whole-engine coarse drafter-screen comparison; unchanged exact target.
set -euo pipefail
base=$1
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cd "${2:-$PWD}"
export MLXFAST_TARGET_SNAPSHOT_DIR="$base/reference_weights/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
export SERVE_UP_WEIGHTS_DIR="$MLXFAST_TARGET_SNAPSHOT_DIR"
export SERVE_UP_SOCKET_DIR="$HOME/rsock/nemo-screen-frontier-race"
mkdir -p "$SERVE_UP_SOCKET_DIR"
exec 9>/tmp/mtplx-gpu-exclusive.lock
flock 9
cp .local-run/mtp-compact-frontier.py vllm/vllm/model_executor/models/nemotron_h_mtp_compact.py
python3 - <<'PY'
import json,re
from pathlib import Path
p=Path('mtp-head.manifest.json');d=json.loads(p.read_text());d['spec']['num_speculative_tokens']=4;p.write_text(json.dumps(d,indent=2)+'\n')
p=Path('vllm/vllm/model_executor/layers/fused_moe/experts/humming_zskip.py');s,n=re.subn(r'^CONFIG = [0-9]+$','CONFIG = 100',p.read_text(),flags=re.M);assert n==1;p.write_text(s)
PY
round=0
for groups in ${SCREEN_GROUPS:-16 32 32 16}; do
  python3 - "$groups" <<'PY'
import re,sys
from pathlib import Path
p=Path('vllm/vllm/model_executor/models/nemotron_h_mtp_screen.py')
s,n=re.subn(r'^CASCADE_FRAC_GROUPS = [0-9]+$',f'CASCADE_FRAC_GROUPS = {int(sys.argv[1])}',p.read_text(),flags=re.M);assert n==1;p.write_text(s)
PY
  round=$((round+1))
  tag="screen-frontier-$groups-$round"
  ./setup.sh > ".local-run/setup.$tag.log" 2>&1
  export MLXFAST_SCORE_PATH=".local-run/score.$tag.json"
  ./tools/local-baseline.sh > ".local-run/$tag.log" 2>&1
  jq -e '.passed and .metrics.passed_correctness' "$MLXFAST_SCORE_PATH"
  jq '{passed,metrics:{passed_correctness:.metrics.passed_correctness,prefill_seconds_per_token:.metrics.prefill_seconds_per_token,decode_seconds_per_token:.metrics.decode_seconds_per_token}}' "$MLXFAST_SCORE_PATH"
done
echo MTP_SCREEN_FRONTIER_ENGINE_RACE_PASS

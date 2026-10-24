#!/usr/bin/env bash
# Full-engine compact native MTP launch ABBA. Ordinary local trusted benchmark.
set -euo pipefail
base=$1
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cd "${2:-$PWD}"
export MLXFAST_TARGET_SNAPSHOT_DIR="$base/reference_weights/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
export SERVE_UP_WEIGHTS_DIR="$MLXFAST_TARGET_SNAPSHOT_DIR"
export SERVE_UP_SOCKET_DIR="$HOME/rsock/nemo-compact-launch-race"
mkdir -p "$SERVE_UP_SOCKET_DIR"
exec 9>/tmp/mtplx-gpu-exclusive.lock
flock 9
round=0
for warps in ${COMPACT_WARPS:-4 2 2 4}; do
  python3 - "$warps" <<'PY'
import json,re,sys
from pathlib import Path
p=Path('mtp-head.manifest.json');d=json.loads(p.read_text());d['spec']['num_speculative_tokens']=3;p.write_text(json.dumps(d,indent=2)+'\n')
p=Path('vllm/vllm/model_executor/layers/fused_moe/experts/humming_zskip.py');s=p.read_text();s,n=re.subn(r'^CONFIG = [0-9]+$', 'CONFIG = 100',s,flags=re.M);assert n==1;p.write_text(s)
p=Path('vllm/vllm/model_executor/models/nemotron_h_mtp_compact.py');s=p.read_text();s,n=re.subn(r'TOPK=6, S=2, BN=32, BK=64, num_warps=[0-9]+, num_stages=2,',f'TOPK=6, S=2, BN=32, BK=64, num_warps={sys.argv[1]}, num_stages=2,',s);assert n==1;p.write_text(s)
PY
  round=$((round+1))
  tag="compact-w${warps}-${round}"
  ./setup.sh > ".local-run/setup.$tag.log" 2>&1
  export MLXFAST_SCORE_PATH=".local-run/score.$tag.json"
  ./tools/local-baseline.sh > ".local-run/$tag.log" 2>&1
  jq -e '.passed and .metrics.passed_correctness' "$MLXFAST_SCORE_PATH"
  jq '{passed,metrics:{passed_correctness:.metrics.passed_correctness,prefill_seconds_per_token:.metrics.prefill_seconds_per_token,decode_seconds_per_token:.metrics.decode_seconds_per_token}}' "$MLXFAST_SCORE_PATH"
done
echo MTP_COMPACT_ENGINE_RACE_PASS

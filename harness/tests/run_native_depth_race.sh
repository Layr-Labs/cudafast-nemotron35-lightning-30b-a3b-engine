#!/usr/bin/env bash
# Whole-engine native depth comparison, using the exact selected carrier.
set -euo pipefail
base=$1
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cd "${2:-$PWD}"
export MLXFAST_TARGET_SNAPSHOT_DIR="$base/reference_weights/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
export SERVE_UP_WEIGHTS_DIR="$MLXFAST_TARGET_SNAPSHOT_DIR"
export SERVE_UP_SOCKET_DIR="$HOME/rsock/nemo-native-depth-race"
mkdir -p "$SERVE_UP_SOCKET_DIR"
exec 9>/tmp/mtplx-gpu-exclusive.lock
flock 9
# Reset only the earlier experimental down/helper and ring config. Frontier
# start projection/M5 support and all other target/drafter methods remain.
cp .local-run/mtp-compact-frontier.py vllm/vllm/model_executor/models/nemotron_h_mtp_compact.py
python3 - <<'PY'
from pathlib import Path
import re
p=Path('vllm/vllm/model_executor/layers/fused_moe/experts/humming_zskip.py')
s,n=re.subn(r'^CONFIG = [0-9]+$','CONFIG = 100',p.read_text(),flags=re.M);assert n==1;p.write_text(s)
# Same no-user-input synthetic prefill capture at the newly allowed depth.
# Decode verify-width capture is already derived from depth by the carrier.
p=Path('vllm/vllm/v1/worker/gpu_model_runner.py');s=p.read_text()
assert 'and self.num_spec_tokens in (3, 4)' in s or 'and self.num_spec_tokens in (3, 4, 5, 6)' in s
s=s.replace('and self.num_spec_tokens in (3, 4)','and self.num_spec_tokens in (3, 4, 5, 6)');p.write_text(s)
PY
round=0
for depth in ${NATIVE_DEPTHS:-5 4 4 5}; do
  python3 - "$depth" <<'PY'
import json,sys
from pathlib import Path
p=Path('mtp-head.manifest.json');d=json.loads(p.read_text());d['spec']['num_speculative_tokens']=int(sys.argv[1]);p.write_text(json.dumps(d,indent=2)+'\n')
PY
  round=$((round+1))
  tag="native-depth-$depth-$round"
  ./setup.sh > ".local-run/setup.$tag.log" 2>&1
  export MLXFAST_SCORE_PATH=".local-run/score.$tag.json"
  ./tools/local-baseline.sh > ".local-run/$tag.log" 2>&1
  jq -e '.passed and .metrics.passed_correctness' "$MLXFAST_SCORE_PATH"
  jq '{passed,metrics:{passed_correctness:.metrics.passed_correctness,prefill_seconds_per_token:.metrics.prefill_seconds_per_token,decode_seconds_per_token:.metrics.decode_seconds_per_token}}' "$MLXFAST_SCORE_PATH"
done
echo NATIVE_DEPTH_ENGINE_RACE_PASS

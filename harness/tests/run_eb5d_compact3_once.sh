#!/usr/bin/env bash
set -euo pipefail
base=$1
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export MLXFAST_TARGET_SNAPSHOT_DIR="$base/reference_weights/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
export SERVE_UP_WEIGHTS_DIR="$MLXFAST_TARGET_SNAPSHOT_DIR"
export SERVE_UP_SOCKET_DIR="$HOME/rsock/nemo-eb5d-compact3"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$SERVE_UP_SOCKET_DIR" .local-run
exec 9>/tmp/mtplx-gpu-exclusive.lock
flock 9
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || { echo GPU_BUSY;exit 1; }
./setup.sh > .local-run/setup.compact3.log 2>&1
root=$PWD;overlay=$(readlink -f .build/vllm-overlay)
image=$(jq -r '.serve_configuration | .container_image+"@"+.container_digest' fixtures/nemotron35_lightning_30b_a3b_track.json)
docker run --rm --gpus all --ipc=host --network none --user "$(id -u):$(id -g)" -v "$base:$base" -w "$root" \
 -e PYTHONDONTWRITEBYTECODE=1 -e HOME="$root/.local-run" -e PYTHONPATH="$overlay:$root/.build/kernels" -e NEMO_SNAPSHOT="$MLXFAST_TARGET_SNAPSHOT_DIR" \
 --entrypoint python3 "$image" harness/tests/compact3_dispatch_check.py > .local-run/compact3-check.log 2>&1
python3 - <<'PY'
import hashlib,json
from pathlib import Path
r={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for root in ('vllm/vllm','harness/kernels','harness/vllm_resident') for f in Path(root).rglob('*') if f.is_file() and '__pycache__' not in str(f)};p=Path('mtp-head.manifest.json');r[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();Path('.local-run/source.compact3.json').write_text(json.dumps(r,sort_keys=True)+'\n')
PY
export MLXFAST_SCORE_PATH=.local-run/score.compact3.json
./tools/local-baseline.sh > .local-run/engine.compact3.log 2>&1
jq -e '.passed and .metrics.passed_correctness' "$MLXFAST_SCORE_PATH"
jq '{passed,metrics:{passed_correctness:.metrics.passed_correctness,prefill_seconds_per_token:.metrics.prefill_seconds_per_token,decode_seconds_per_token:.metrics.decode_seconds_per_token}}' "$MLXFAST_SCORE_PATH"
printf 'EB5D_COMPACT3_ONE_CANDIDATE_DONE\n'

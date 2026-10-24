#!/usr/bin/env bash
# Reproducibility, not a performance rerun. Existing tested composition stays staged.
set -euo pipefail
base=$1
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export SERVE_UP_WEIGHTS_DIR="$base/reference_weights/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
export SERVE_UP_SOCKET_DIR="$HOME/rsock/nemo-native-compact3-probe"
mkdir -p "$SERVE_UP_SOCKET_DIR" .local-run/native-compact3-probe
exec 9>/tmp/mtplx-gpu-exclusive.lock
flock 9
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || { echo 'GPU busy; refusing'; exit 1; }
python3 - <<'PY'
import json
j=json.load(open('correctness_prompts/public-longcopy-gate-english-496.mtp4.golden.json'))
assert len(j['cases'])>0
c=j['cases'][0]
print('case keys',sorted(c))
tokens=c.get('prompt_tokens',c.get('prompt'))
assert isinstance(tokens,list) and all(isinstance(t,int) for t in tokens)
json.dump(tokens,open('.local-run/native-compact3-probe/prompt.json','w'))
# Trusted probe has a checkout-specific adapter path; change only that path in a diagnostic copy.
s=open('tools/determinism-probe.py').read()
s=s.replace('os.path.expanduser("~/nemotron35/repo/.build/release/mlxfast-runtime-worker")','os.path.abspath(".build/release/mlxfast-runtime-worker")')
open('.local-run/native-compact3-probe/probe.py','w').write(s)
PY
python3 - <<'PYSET'
import json
from pathlib import Path
p=Path('mtp-head.manifest.json');j=json.loads(p.read_text());j['spec']['num_speculative_tokens']=4;p.write_text(json.dumps(j,indent=2)+'\n')
PYSET
export MLXFAST_TARGET_SNAPSHOT_DIR="$SERVE_UP_WEIGHTS_DIR"
./setup.sh > .local-run/native-compact3-probe/setup.log 2>&1
for mode in serial mtp; do
 depth=0;spec=''
 if [[ $mode == mtp ]]; then depth=4;spec='{"mode":"mtp","mtp":{"depth":4}}';fi
 out=".local-run/native-compact3-probe/$mode.socket"
 ./tools/serve-up.sh --boot --spec "$mode" --draft-len "$depth" --socket-out "$out" > ".local-run/native-compact3-probe/boot-$mode.log" 2>&1
 sock=$(cat "$out")
 trap './tools/serve-up.sh --stop --socket "$sock"' EXIT
 PROBE_STREAMS_OUT=".local-run/native-compact3-probe/streams-$mode.json" python3 .local-run/native-compact3-probe/probe.py "$sock" .local-run/native-compact3-probe/prompt.json 2 "$spec" | tee ".local-run/native-compact3-probe/probe-$mode.log"
 ./tools/serve-up.sh --stop --socket "$sock"
 trap - EXIT
 python3 - "$mode" <<'PY'
import json,sys
s=json.load(open('.local-run/native-compact3-probe/streams-'+sys.argv[1]+'.json'))
assert len(s)==2 and len({tuple(t) for t in s})==1
print('NATIVE_COMPACT3_DETERMINISM_'+sys.argv[1].upper()+'_PASS',len(s[0]),flush=True)
PY
done

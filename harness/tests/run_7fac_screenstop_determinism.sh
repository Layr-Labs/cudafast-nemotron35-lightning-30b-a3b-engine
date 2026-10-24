#!/usr/bin/env bash
# Reproducibility only, not a score or a throughput rerun.
set -euo pipefail
base=$1
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH" PYTHONDONTWRITEBYTECODE=1
export SERVE_UP_WEIGHTS_DIR="$base/reference_weights/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
export SERVE_UP_SOCKET_DIR="$HOME/rsock/nemo-7fac-screenstop-probe"
mkdir -p "$SERVE_UP_SOCKET_DIR" .local-run/7fac-screenstop-probe
exec 9>/tmp/mtplx-gpu-exclusive.lock; flock 9
until [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do sleep 5; done
cmp vllm/vllm/model_executor/models/nemotron_h_mtp_screen.py .build/vllm-overlay/vllm/model_executor/models/nemotron_h_mtp_screen.py
python3 - <<'PY'
import json
from pathlib import Path
p=Path('.local-run/7fac-screenstop-probe')
j=json.loads(Path('correctness_prompts/public-longcopy-gate-english-496.mtp4.golden.json').read_text());c=j['cases'][0];t=c.get('prompt_tokens',c.get('prompt'));assert isinstance(t,list) and all(isinstance(i,int) for i in t);(p/'prompt.json').write_text(json.dumps(t)+'\n')
s=Path('tools/determinism-probe.py').read_text();s=s.replace('os.path.expanduser("~/nemotron35/repo/.build/release/mlxfast-runtime-worker")','os.path.abspath(".build/release/mlxfast-runtime-worker")');(p/'probe.py').write_text(s)
PY
for mode in serial mtp; do
 depth=0; spec=''; if [[ $mode == mtp ]]; then depth=4;spec='{"mode":"mtp","mtp":{"depth":4}}';fi
 out=".local-run/7fac-screenstop-probe/$mode.socket"
 ./tools/serve-up.sh --boot --spec "$mode" --draft-len "$depth" --socket-out "$out" > ".local-run/7fac-screenstop-probe/boot-$mode.log" 2>&1
 sock=$(cat "$out")
 trap './tools/serve-up.sh --stop --socket "$sock"' EXIT
 PROBE_STREAMS_OUT=".local-run/7fac-screenstop-probe/streams-$mode.json" python3 .local-run/7fac-screenstop-probe/probe.py "$sock" .local-run/7fac-screenstop-probe/prompt.json 3 "$spec" | tee ".local-run/7fac-screenstop-probe/probe-$mode.log"
 ./tools/serve-up.sh --stop --socket "$sock"
 trap - EXIT
 python3 - "$mode" <<'PY'
import json,sys
s=json.load(open('.local-run/7fac-screenstop-probe/streams-'+sys.argv[1]+'.json'));assert len(s)==3 and len({tuple(t) for t in s})==1 and all(len(t)==129 for t in s);print('FAC7_SCREENSTOP_DETERMINISM_'+sys.argv[1].upper()+'_PASS',len(s),len(s[0]),flush=True)
PY
done
python3 - <<'PY'
import json
from pathlib import Path
p=Path('.local-run/7fac-screenstop-probe')
# Contract section 11.4: serial and MTP have separate timed trajectories.
# Compare each reproducible stream with its OWN public per-depth oracle.
for mode,suffix in (('serial',''),('mtp','.mtp4')):
    j=json.loads(Path('correctness_prompts/public-longcopy-gate-english-496'+suffix+'.golden.json').read_text())['benchmark']
    expected=[j['expected_decode_seed_token']]+j['expected_decode_tokens'][:128]
    streams=json.loads((p/('streams-'+mode+'.json')).read_text())
    assert len(streams)==3 and all(t==expected for t in streams),mode
    print('FAC7_SCREENSTOP_OWN_ORACLE_'+mode.upper()+'_EXACT_PASS',len(streams),len(expected),flush=True)
PY

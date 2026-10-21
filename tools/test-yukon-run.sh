#!/usr/bin/env bash
# Hermetic test for tools/nemotron35-yukon-run.sh. No GPU, no benchd.
#
# The script copies into a temporary tree with the real benchmark.json. Stubs
# replace the ranked script and tools/local-baseline.sh, and each stub records
# how it was called.
#
#   1. GITHUB_ACTIONS=true goes to the ranked script with all arguments.
#   2. each ranked input alone goes to the ranked script.
#   3. benchmark.json's benchmarkCommand with no ranked input goes to
#      tools/local-baseline.sh with no arguments and MLXFAST_SCORE_PATH set to
#      scorePath.
#   4. the local path refuses an argument.
#
# tools/test-local-baseline.sh covers the path from tools/local-baseline.sh to
# benchd at the declared depth.
#
# Usage: tools/test-yukon-run.sh
# Exit:  0 all cases pass, 1 a case failed (printed with a FAIL prefix)
set -uo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
fails=0
fail() { echo "FAIL $*" >&2; fails=$((fails + 1)); }
pass() { echo "ok   $*"; }

# make_tree <dir>
make_tree() {
  local t="$1"
  mkdir -p "${t}/tools"
  cp "${SRC}/tools/nemotron35-yukon-run.sh" "${t}/tools/"
  cp "${SRC}/benchmark.json" "${t}/"
  cat > "${t}/tools/nemotron35-measure-and-score.sh" <<'EOF'
#!/usr/bin/env bash
echo "ranked $*" >> "$(dirname "$0")/../calls"
EOF
  cat > "${t}/tools/local-baseline.sh" <<'EOF'
#!/usr/bin/env bash
echo "local score=${MLXFAST_SCORE_PATH:-} cwd=${PWD} args=$#" >> "$(dirname "$0")/../calls"
EOF
  chmod +x "${t}/tools/"*.sh
}

clean_env=(env -u GITHUB_ACTIONS -u MLXFAST_NEMOTRON35_GOLDEN_DIR -u MLXFAST_BASELINE_WORKSPACE -u MLXFAST_BASELINE_CALIBRATION -u MLXFAST_SCORE_PATH)

# 1
t="${WORK}/c1"; make_tree "${t}"
"${clean_env[@]}" GITHUB_ACTIONS=true "${t}/tools/nemotron35-yukon-run.sh" --preflight-only
if [[ "$(cat "${t}/calls")" == "ranked --preflight-only" ]]; then pass "case 1: GITHUB_ACTIONS=true is ranked"; else fail "case 1: $(cat "${t}/calls")"; fi

# 2
for var in MLXFAST_NEMOTRON35_GOLDEN_DIR MLXFAST_BASELINE_WORKSPACE MLXFAST_BASELINE_CALIBRATION; do
  t="${WORK}/c2-${var}"; make_tree "${t}"
  "${clean_env[@]}" "${var}=/nonexistent" "${t}/tools/nemotron35-yukon-run.sh"
  if [[ "$(cat "${t}/calls")" == "ranked " ]]; then pass "case 2: ${var} alone is ranked"; else fail "case 2 ${var}: $(cat "${t}/calls")"; fi
done

# 3
t="${WORK}/c3"; make_tree "${t}"
cmd="$(jq -r '.benchmarkCommand[2]' "${t}/benchmark.json")"
score="$(jq -r '.scorePath' "${t}/benchmark.json")"
(cd "${t}" && "${clean_env[@]}" bash -c "${cmd}") 2>/dev/null
want="local score=${score} cwd=$(cd "${t}" && pwd) args=0"
if [[ "$(cat "${t}/calls")" == "${want}" ]]; then pass "case 3: benchmarkCommand off a ranked box runs tools/local-baseline.sh"; else fail "case 3: $(cat "${t}/calls")"; fi

# 4
t="${WORK}/c4"; make_tree "${t}"
"${clean_env[@]}" "${t}/tools/nemotron35-yukon-run.sh" --preflight-only 2>/dev/null; rc=$?
if [[ "${rc}" == "2" && ! -e "${t}/calls" ]]; then pass "case 4: the local path refuses an argument"; else fail "case 4: rc=${rc}"; fi

[[ "${fails}" == "0" ]] || exit 1
echo "all cases passed"

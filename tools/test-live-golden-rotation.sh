#!/usr/bin/env bash
# test-live-golden-rotation.sh -- which prompts a job measures, hermetically.
#
# Covers tools/draw-live-golden.sh (uniform over a rotation pool, refusal of a
# pool name outside timed_prompt_pool, fallback to live_golden when no rotation
# is declared, and no draw at all under live_goldens) and
# tools/nemotron35-measure-and-score.sh under the shipped live_goldens: every
# live golden handed to benchd in fixture order, with its serial control golden
# and, at a declared depth, its per-depth oracle; a drawn prompt refused; and
# the one measurement retry. benchd is a stub that records its argv. No GPU, no
# network.
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
FIXTURE="${REPO_ROOT}/fixtures/nemotron35_lightning_30b_a3b_track.json"
DRAW="${REPO_ROOT}/tools/draw-live-golden.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok() { echo "ok    $*"; }

mapfile -t NAMES < <(jq -r '.live_goldens[]' "${FIXTURE}")
FIRST="${NAMES[0]}"

# A fixture copy with synthetic pins: serial golden i is sha256 "a"x64 and
# 100+i bytes, and every per-depth oracle mtp<d> is sha256 "<d>"x64 and
# 1000+d bytes (only measure-and-score reads them here).
jq '(.timed_prompt_pool[0].r2_path | split("/")[:-1] | join("/")) as $prefix
    | .warmup_golden = {sha256: "5edc020ba47b6e035dfd5584b2020627c9c2c68597800f4e65d69115d695e58c", bytes: 28357}
    | .require_private_dir = true
    | .timed_prompt_pool = [.live_goldens | to_entries[]
        | {r2_path: "\($prefix)/\(.value).golden.json", sha256: ("a" * 64), bytes: (100 + .key)}]
    | .speculative_oracles = (reduce .live_goldens[] as $n ({};
        .[$n] = (reduce range(1; 7) as $d ({};
          .["mtp\($d)"] = {r2_path: "\($prefix)/\($n).mtp\($d).golden.json",
                           sha256: ("\($d)" * 64), bytes: (1000 + $d)}))))' \
  "${FIXTURE}" > "${WORK}/full.json"
# The same prompts as a per-job rotation pool, the draw's other shape.
jq --arg first "${FIRST}" '.live_golden_rotation = {mode: "per_job_random", pool: .live_goldens, pending: []}
    | .live_golden = $first | del(.live_goldens)' "${WORK}/full.json" > "${WORK}/rot.json"

# --- the draw -----------------------------------------------------------------
draws=600
counts="$(for _ in $(seq "${draws}"); do "${DRAW}" "${WORK}/rot.json"; done | sort | uniq -c)"
distinct="$(wc -l <<<"${counts}" | tr -d ' ')"
[[ "${distinct}" -eq "${#NAMES[@]}" ]] \
  && ok "draw: ${draws} draws hit all ${#NAMES[@]} pool prompts" \
  || fail "draw: ${draws} draws hit ${distinct} of ${#NAMES[@]} prompts: ${counts}"
# Expected 200 per prompt of 3, sd ~11.5; [140, 260] is a five-sigma band.
while read -r c name; do
  (( c >= 140 && c <= 260 )) || fail "draw: '${name}' drawn ${c} times of ${draws}; not uniform"
done <<<"${counts}"
ok "draw: every prompt drawn within five sigma of uniform"

jq 'del(.live_golden_rotation)' "${WORK}/rot.json" > "${WORK}/norot.json"
[[ "$("${DRAW}" "${WORK}/norot.json")" == "${FIRST}" ]] \
  && ok "draw: no rotation block falls back to live_golden" \
  || fail "draw: no rotation block did not fall back to live_golden"
drawn="$("${DRAW}")"
[[ $? -eq 0 && -z "${drawn}" ]] \
  && ok "draw: the shipped fixture declares live_goldens and draws nothing" \
  || fail "draw: the shipped fixture drew '${drawn}'"
jq '.live_golden_rotation = {mode: "per_job_random", pool: .live_goldens}' "${FIXTURE}" > "${WORK}/both.json"
out="$("${DRAW}" "${WORK}/both.json" 2>&1)" && fail "draw: live_goldens with a rotation block drew '${out}'" \
  || { grep -q "both live_goldens and live_golden_rotation" <<<"${out}" && ok "draw: live_goldens with a rotation block is refused by name" || fail "draw: wrong refusal: ${out}"; }

jq '.live_golden_rotation.pool += ["nosuch"]' "${WORK}/rot.json" > "${WORK}/bad.json"
out="$("${DRAW}" "${WORK}/bad.json" 2>&1)" && fail "draw: a pool name outside timed_prompt_pool was drawn from" \
  || { grep -q "'nosuch' has no timed_prompt_pool entry" <<<"${out}" && ok "draw: a pool name outside timed_prompt_pool is refused by name" || fail "draw: wrong refusal: ${out}"; }
jq '.live_golden_rotation.pool = []' "${WORK}/rot.json" > "${WORK}/empty.json"
"${DRAW}" "${WORK}/empty.json" >/dev/null 2>&1 && fail "draw: an empty pool drew something" || ok "draw: an empty pool is refused"
jq '.live_golden_rotation.mode = "round_robin"' "${WORK}/rot.json" > "${WORK}/mode.json"
"${DRAW}" "${WORK}/mode.json" >/dev/null 2>&1 && fail "draw: an unknown mode drew something" || ok "draw: an unknown mode is refused"

# --- measure-and-score measures every live golden ------------------------------
ROOT="${WORK}/root"
mkdir -p "${ROOT}/tools" "${ROOT}/fixtures" "${ROOT}/correctness_prompts"
for entry in "${REPO_ROOT}"/tools/*; do ln -s "${entry}" "${ROOT}/tools/$(basename "${entry}")"; done
if [[ -n "${MEASURE_SCRIPT_UNDER_TEST:-}" ]]; then
  rm "${ROOT}/tools/nemotron35-measure-and-score.sh"
  ln -s "${MEASURE_SCRIPT_UNDER_TEST}" "${ROOT}/tools/nemotron35-measure-and-score.sh"
fi
ln -s "${REPO_ROOT}/benchmark.json" "${ROOT}/benchmark.json"
cp "${REPO_ROOT}/correctness_prompts/public-longcopy-gate-english-496.golden.json" "${ROOT}/correctness_prompts/"
jq '.spec = {enabled: false, num_speculative_tokens: 0}' "${REPO_ROOT}/mtp-head.manifest.json" > "${ROOT}/mtp-head.manifest.json"
jq '.spec = {enabled: true, num_speculative_tokens: 3}' "${REPO_ROOT}/mtp-head.manifest.json" > "${WORK}/mtp3.manifest.json"
CONTRACT="${ROOT}/fixtures/nemotron35_lightning_30b_a3b_track.json"

GOLDENS="${WORK}/goldens"
mkdir -p "${GOLDENS}" "${WORK}/ws" "${WORK}/weights"
for n in "${NAMES[@]}"; do
  : > "${GOLDENS}/${n}.golden.json"
  : > "${GOLDENS}/${n}.mtp3.golden.json"
done
chmod 0440 "${GOLDENS}"/*.golden.json
: > "${WORK}/cal.json"
printf '#!/usr/bin/env bash\nexit 0\n' > "${WORK}/engine"
chmod +x "${WORK}/engine"

# benchd stub: `iterate --help` advertises the flags (STUB_FLAGS), a real run
# records its argv. With STUB_PLAN (a file of outcomes, one per run: pass,
# control, control2, control_noaudit, control_false, control_object,
# control_string, prefill, spawn, floor, teardown) it also seals
# score.json like benchd: the
# JSON on stdout, the marker on stderr, exit 1 unless it passed. Run n appends
# its outcome to STUB_RUNS and keeps its argv as STUB_RUNS.argv.n. With
# STUB_JOBDIR it logs how many files the per-job sandbox dirs hold
# (STUB_RUNS.seen), then plants one in each, as a memoizing candidate would.
cat > "${WORK}/benchd" <<'EOF'
#!/usr/bin/env bash
if [[ "$2" == --help ]]; then echo "${STUB_FLAGS:---control-golden --mtp-depth --live-prompt}"; exit 0; fi
printf '%s\n' "$@" > "${STUB_ARGV}"
if [[ -n "${STUB_PRIVATE_LOG:-}" ]]; then
  printf 'dir=%s\n' "${MLXFAST_PRIVATE_DIR:-}" > "${STUB_PRIVATE_LOG}"
  if [[ -n "${MLXFAST_PRIVATE_DIR:-}" ]]; then
    printf 'mode=%s\n' "$(stat -c %a "${MLXFAST_PRIVATE_DIR}")" >> "${STUB_PRIVATE_LOG}"
    "${STUB_REAL_FIND:-find}" "${MLXFAST_PRIVATE_DIR}" -mindepth 1 -maxdepth 1 -type f -printf 'file=%f\n' | sort >> "${STUB_PRIVATE_LOG}"
  fi
fi
[[ -n "${STUB_PLAN:-}" ]] || exit 0
n=$(( $(wc -l < "${STUB_RUNS}") + 1 ))
what="$(sed -n "${n}p" "${STUB_PLAN}")"
echo "${what}" >> "${STUB_RUNS}"
cp "${STUB_ARGV}" "${STUB_RUNS}.argv.${n}"
if [[ -n "${STUB_JOBDIR:-}" ]]; then
  find "${STUB_JOBDIR}" -mindepth 2 | wc -l >> "${STUB_RUNS}.seen"
  for d in cargo-home target kernels-out sock adapter-home; do echo memo > "${STUB_JOBDIR}/${d}/memo-${n}"; done
  [[ -z "${STUB_RUNNER_SOCK:-}" || "${n}" -ne 1 ]] || echo runner > "${STUB_JOBDIR}/sock/runner-sock-leftover"
fi
score="$(grep -A1 -xF -- --score-path <<<"$(printf '%s\n' "$@")" | tail -1)"
case "${what}" in
  pass) json="{\"passed\": true, \"score\": 1.${n}, \"metrics\": {\"error\": \"\"}}" ;;
  control) json="{\"passed\": false, \"score\": null, \"metrics\": {\"error\": \"SERIAL-CONTROL-LEG-FAILED: benchmark free-run decode token mismatch at step 47: expected oracle token 1278, engine returned 1799\"}}" ;;
  control2|control_noaudit|control_false|control_object|control_string) json="{\"passed\": false, \"score\": null, \"metrics\": {\"error\": \"SERIAL-CONTROL-LEG-FAILED: benchmark free-run decode token mismatch at step 47: expected oracle token 1278, engine returned 1799\"}}" ;;
  prefill) json="{\"passed\": false, \"score\": null, \"metrics\": {\"error\": \"SERIAL-CONTROL-LEG-FAILED: reference warmup leg failed: benchmark prefill token mismatch at step 0: expected oracle token 1, engine returned 2\"}}" ;;
  spawn) json="{\"passed\": false, \"score\": null, \"metrics\": {\"error\": \"SERIAL-CONTROL-LEG-FAILED: worker spawn failed\"}}" ;;
  floor) json="{\"passed\": false, \"score\": null, \"metrics\": {\"error\": \"performance floor failed: decode_speedup=0.9\"}}" ;;
  teardown) json="{\"passed\": false, \"score\": null, \"metrics\": {\"error\": \"LEG-TEARDOWN-FAILED: the candidate leg stop failed\"}}" ;;
  boot) json="{\"passed\": false, \"score\": null, \"metrics\": {\"error\": \"LEG-SERVE-BOOT-FAILED: the candidate leg's resident did not boot (1): abcd1234\", \"paired_legs\": []}}" ;;
  boot2) json="{\"passed\": false, \"score\": null, \"metrics\": {\"error\": \"LEG-SERVE-BOOT-FAILED: the candidate leg's resident did not boot (1): abcd1234\", \"paired_legs\": [{\"pair\": 1, \"prompt_sha256\": \"deadbeef\", \"control_prefill_seconds_per_token\": 1.0, \"control_decode_seconds_per_token\": 2.0, \"candidate_prefill_seconds_per_token\": 1.1, \"candidate_decode_seconds_per_token\": 2.1}]}}" ;;
  boot_ctl) json="{\"passed\": false, \"score\": null, \"metrics\": {\"error\": \"LEG-SERVE-BOOT-FAILED: the serial-control leg's resident did not boot (1): abcd1234\", \"paired_legs\": []}}" ;;
  boot_ctl2) json="{\"passed\": false, \"score\": null, \"metrics\": {\"error\": \"LEG-SERVE-BOOT-FAILED: the serial-control leg's resident did not boot (1): abcd1234\", \"paired_legs\": [{\"pair\": 1, \"prompt_sha256\": \"deadbeef\", \"control_prefill_seconds_per_token\": 1.0, \"control_decode_seconds_per_token\": 2.0, \"candidate_prefill_seconds_per_token\": 1.1, \"candidate_decode_seconds_per_token\": 2.1}]}}" ;;
esac
audit="{\"passed\": false, \"score\": null, \"metrics\": {\"paired_legs\": []}}"
if [[ "${what}" == control2 ]]; then
  audit="{\"passed\": false, \"score\": null, \"metrics\": {\"paired_legs\": [{\"pair\": 1, \"candidate_prefill_seconds_per_token\": 1.1, \"candidate_decode_seconds_per_token\": 2.1}]}}"
fi
case "${what}" in
  control_false) audit='{"passed": false, "metrics": {"paired_legs": false}}' ;;
  control_object) audit='{"passed": false, "metrics": {"paired_legs": {}}}' ;;
  control_string) audit='{"passed": false, "metrics": {"paired_legs": "none"}}' ;;
esac
if [[ "${what}" != control_noaudit && -n "${BENCHD_PRIVATE_AUDIT_PATH:-}" ]]; then
  printf '%s\n' "${audit}" > "${BENCHD_PRIVATE_AUDIT_PATH}"
fi
echo "stub benchd run ${n}: ${what}" >&2
printf '%s\n' "${json}" | tee "${score}"
: > "${score}.sha256"
echo "benchd iterate: wrote ${score} (passed=$(jq -r .passed "${score}"), score=$(jq -r .score "${score}"))" >&2
[[ "${what}" == pass ]]
EOF
chmod +x "${WORK}/benchd"

measure() { # measure FIXTURE [env...]  -> stdout+stderr; argv in ${WORK}/argv
  cp "$1" "${CONTRACT}"; shift
  rm -f "${WORK}/argv"
  env -u MLXFAST_LIVE_GOLDEN_CHOICE -u SANDBOX_USER -u MLXFAST_QWEN_MTP_TRACK_ID -u MLXFAST_PRIVATE_DIR \
    BENCHD="${WORK}/benchd" STUB_ARGV="${WORK}/argv" STUB_PRIVATE_LOG="${WORK}/private.log" STUB_REAL_FIND="$(command -v find)" \
    BENCHD_PRIVATE_AUDIT_PATH="${WORK}/private-audit.json" \
    MLXFAST_BASELINE_WORKSPACE="${WORK}/ws" MLXFAST_BASELINE_CALIBRATION="${WORK}/cal.json" \
    MLXFAST_NEMOTRON35_GOLDEN_DIR="${GOLDENS}" MLXFAST_ENGINE_BIN="${WORK}/engine" \
    MLXFAST_WEIGHTS_PATH="${WORK}/weights" SPEC_DECLARATION_MANIFEST="${ROOT}/mtp-head.manifest.json" \
    "$@" "${ROOT}/tools/nemotron35-measure-and-score.sh" ${MS_ARGS[@]+"${MS_ARGS[@]}"} 2>&1
}
MS_ARGS=()
argv_has() { grep -qxF -- "$1" "${WORK}/argv" 2>/dev/null; }
argv_after() { grep -A1 -xF -- "$1" "${WORK}/argv" | tail -1; }

argv_all() { grep -A1 -xF -- "$1" "${WORK}/argv" | grep -vxF -- "$1" | grep -vxF -- --; }
want_list() { local f="$1"; shift; local n; for n in "${NAMES[@]}"; do printf '%s\n' "${GOLDENS}/${n}${f}"; done; }
want_private_list() { local d="$1" f="$2" n; for n in "${NAMES[@]}"; do printf '%s\n' "${d}/${n}${f}"; done; }

out="$(measure "${WORK}/full.json")"
private_dir="$(sed -n 's/^dir=//p' "${WORK}/private.log")"
if [[ "$(argv_all --golden)" == "$(want_private_list "${private_dir}" .golden.json)" ]] \
  && [[ "$(argv_all --control-golden)" == "$(want_private_list "${private_dir}" .golden.json)" ]] \
  && [[ "$(argv_all --golden-bytes)" == "$(seq 100 $((99 + ${#NAMES[@]})))" ]] \
  && ! argv_has --live-prompt && ! argv_has --mtp-depth; then
  ok "measure: serial: every live golden, in fixture order, is a --golden and its own --control-golden, with its pin; no --live-prompt"
else
  fail "measure: serial run did not hand benchd every live golden; out: $(tail -2 <<<"${out}"); argv: $(tr '\n' ' ' < "${WORK}/argv" 2>/dev/null)"
fi
if [[ "$(argv_after --warmup-golden)" == "${private_dir}/public-longcopy-gate-english-496.golden.json" ]] \
  && [[ "$(argv_after --warmup-golden-sha256)" == "5edc020ba47b6e035dfd5584b2020627c9c2c68597800f4e65d69115d695e58c" ]] \
  && [[ "$(argv_after --warmup-golden-bytes)" == "28357" ]] \
  && grep -qx 'mode=700' "${WORK}/private.log" \
  && [[ "$(grep '^file=' "${WORK}/private.log")" == $'file=pool-10.golden.json\nfile=pool-11.golden.json\nfile=pool-9.golden.json\nfile=public-longcopy-gate-english-496.golden.json' ]] \
  && [[ ! -d "${private_dir}" ]]; then
  ok "measure: private dir has only passed goldens and warm-up, carries its pin, has mode 0700, and is removed"
else
  fail "measure: private golden staging is wrong: $(tr '\n' ' ' < "${WORK}/private.log" 2>/dev/null); argv: $(tr '\n' ' ' < "${WORK}/argv" 2>/dev/null)"
fi

mkdir "${WORK}/bin"
cat > "${WORK}/bin/cp" <<'EOF'
#!/usr/bin/env bash
/bin/cp "$@" || exit
if [[ -n "${STUB_CP_CHANGE_SOURCE:-}" && "$3" == */nemotron35-private.*/* && "$2" == *pool-9.golden.json ]]; then
  chmod u+w "$2"
  printf different >> "$2"
fi
EOF
chmod +x "${WORK}/bin/cp"
out="$(measure "${WORK}/full.json" PATH="${WORK}/bin:${PATH}" STUB_CP_CHANGE_SOURCE=1)"
if [[ ! -f "${WORK}/argv" ]] \
  && grep -q "private golden 'pool-9.golden.json' has different bytes" <<<"${out}"; then
  ok "measure: a same-named control golden with different bytes is refused by basename"
else
  fail "measure: a same-named control golden with different bytes was not refused: $(tail -2 <<<"${out}")"
fi
chmod u+w "${GOLDENS}/${NAMES[0]}.golden.json"
printf '' > "${GOLDENS}/${NAMES[0]}.golden.json"
chmod 0440 "${GOLDENS}/${NAMES[0]}.golden.json"

jq 'del(.warmup_golden, .require_private_dir)' "${WORK}/full.json" > "${WORK}/no-private.json"
out="$(measure "${WORK}/no-private.json")"
if [[ "$(argv_all --golden)" == "$(want_list .golden.json)" ]] \
  && ! argv_has --warmup-golden \
  && [[ "$(sed -n 's/^dir=//p' "${WORK}/private.log")" == "" ]]; then
  ok "measure: an old fixture keeps direct golden paths and passes no warm-up flag"
else
  fail "measure: an old fixture did not keep the old interface: $(tr '\n' ' ' < "${WORK}/argv" 2>/dev/null)"
fi

out="$(measure "${WORK}/full.json" SPEC_DECLARATION_MANIFEST="${WORK}/mtp3.manifest.json")"
private_dir="$(sed -n 's/^dir=//p' "${WORK}/private.log")"
if [[ "$(argv_all --golden)" == "$(want_private_list "${private_dir}" .mtp3.golden.json)" ]] \
  && [[ "$(argv_all --golden-sha256 | sort -u)" == "$(printf '3%.0s' $(seq 64))" ]] \
  && [[ "$(argv_all --control-golden)" == "$(want_private_list "${private_dir}" .golden.json)" ]] \
  && [[ "$(argv_all --control-golden-sha256 | sort -u)" == "$(printf 'a%.0s' $(seq 64))" ]] \
  && [[ "$(argv_after --mtp-depth)" == 3 ]] && ! argv_has --live-prompt; then
  ok "measure: at mtp3 each candidate oracle is speculative_oracles.<prompt>.mtp3 and each control golden is the serial tape"
else
  fail "measure: mtp3 run did not use the per-depth oracles; out: $(tail -2 <<<"${out}"); argv: $(tr '\n' ' ' < "${WORK}/argv" 2>/dev/null)"
fi
if [[ "$(grep '^file=' "${WORK}/private.log" | wc -l)" -eq 7 ]] \
  && [[ "$(grep -c '^file=.*\.mtp3\.golden\.json$' "${WORK}/private.log")" -eq 3 ]] \
  && [[ "$(grep -c '^file=pool-.*\.golden\.json$' "${WORK}/private.log")" -eq 6 ]] \
  && grep -qx 'file=public-longcopy-gate-english-496.golden.json' "${WORK}/private.log" \
  && [[ ! -d "${private_dir}" ]]; then
  ok "measure: mtp3 private dir has exactly three timed, three control, and one warm-up golden"
else
  fail "measure: mtp3 private dir did not contain exactly the seven passed goldens: $(tr '\n' ' ' < "${WORK}/private.log")"
fi

jq --arg n "${NAMES[1]}" 'del(.speculative_oracles[$n].mtp3)' "${WORK}/full.json" > "${WORK}/no-oracle.json"
out="$(measure "${WORK}/no-oracle.json" SPEC_DECLARATION_MANIFEST="${WORK}/mtp3.manifest.json")"
[[ ! -f "${WORK}/argv" ]] && grep -q "declared spec 'mtp3' has no speculative_oracles\['${NAMES[1]}'\] entry" <<<"${out}" \
  && ok "measure: a declared depth with no oracle for one live golden is refused by name" \
  || fail "measure: a missing oracle was not refused: $(tail -2 <<<"${out}")"

out="$(measure "${WORK}/full.json" MLXFAST_LIVE_GOLDEN_CHOICE="${FIRST}")"
[[ ! -f "${WORK}/argv" ]] && grep -q "every job measures every live golden" <<<"${out}" \
  && ok "measure: a drawn prompt is refused: every job measures every live golden" \
  || fail "measure: a drawn prompt was honoured: $(tail -2 <<<"${out}")"

jq 'del(.live_goldens)' "${WORK}/full.json" > "${WORK}/no-live.json"
out="$(measure "${WORK}/no-live.json")"
[[ ! -f "${WORK}/argv" ]] && grep -q "declares no live_goldens" <<<"${out}" \
  && ok "measure: a fixture with no live_goldens is refused" \
  || fail "measure: a fixture with no live_goldens was scored: $(tail -2 <<<"${out}")"

rm -f "${GOLDENS}/${NAMES[2]}.golden.json"
out="$(measure "${WORK}/full.json")"
: > "${GOLDENS}/${NAMES[2]}.golden.json"
[[ ! -f "${WORK}/argv" ]] && grep -q "live golden not found at ${GOLDENS}/${NAMES[2]}.golden.json" <<<"${out}" \
  && ok "measure: an unstaged live golden is refused by name" \
  || fail "measure: an unstaged live golden was not refused: $(tail -2 <<<"${out}")"

# --- the one control-leg retry ---------------------------------------------------
# plan OUTCOME... -> runs measure with that benchd plan; sets rc, out, runs.
printf 'MemTotal: %d kB\nMemAvailable: %d kB\n' $((100 * 1024 * 1024)) $((100 * 1024 * 1024)) > "${WORK}/meminfo"
PLAN_ENV=()
plan() {
  printf '%s\n' "$@" > "${WORK}/plan"
  : > "${WORK}/runs"
  rm -f "${ROOT}/score.json" "${ROOT}/score.json.sha256" "${WORK}"/runs.* "${WORK}/private-audit.json"
  out="$(measure "${WORK}/full.json" STUB_PLAN="${WORK}/plan" \
    STUB_RUNS="${WORK}/runs" SERVE_UP_MEMINFO="${WORK}/meminfo" WAIT_FOR_MEMORY_TIMEOUT_S=0 \
    ${PLAN_ENV[@]+"${PLAN_ENV[@]}"})"
  rc=$?
  runs="$(tr '\n' ' ' < "${WORK}/runs")"
}
markers() { grep -c "benchd iterate: wrote" <<<"${out}"; }

plan pass
[[ "${rc}" -eq 0 && "${runs}" == "pass " && "$(markers)" -eq 1 ]] \
  && grep -q "control_leg_attempts=1" <<<"${out}" && jq -e '.score == 1.1' "${ROOT}/score.json" >/dev/null \
  && ok "retry: a passing run runs once, attempts=1, one marker" \
  || fail "retry: pass: rc=${rc} runs=${runs} out: $(tail -3 <<<"${out}")"

plan control pass
[[ "${rc}" -eq 0 && "${runs}" == "control pass " && "$(markers)" -eq 1 ]] \
  && grep -q "control_leg_attempts=2" <<<"${out}" && grep -q "(passed=true, score=1.2)" <<<"${out}" \
  && ! grep -q '"score": null' <<<"${out}" && jq -e '.score == 1.2' "${ROOT}/score.json" >/dev/null \
  && ok "retry: control failure then pass is scored from attempt 2 alone, attempts=2, one marker" \
  || fail "retry: control+pass: rc=${rc} runs=${runs} out: $(tail -5 <<<"${out}")"

plan control2 pass
[[ "${rc}" -eq 75 && "${runs}" == "control2 " && "$(markers)" -eq 1 ]] \
  && grep -q "control_leg_attempts=1" <<<"${out}" \
  && grep -q "retry was withheld because candidate pairs were already measured" <<<"${out}" \
  && ok "retry: pair-2 control divergence after candidate timing is final infrastructure" \
  || fail "retry: pair-2 control divergence: rc=${rc} runs=${runs} out: $(tail -5 <<<"${out}")"

plan control_noaudit pass
[[ "${rc}" -eq 75 && "${runs}" == "control_noaudit " && "$(markers)" -eq 1 ]] \
  && grep -q "control_leg_attempts=1" <<<"${out}" \
  && grep -q "retry was withheld because the private audit was missing or unreadable" <<<"${out}" \
  && ok "retry: a missing private audit withholds the retry" \
  || fail "retry: missing private audit: rc=${rc} runs=${runs} out: $(tail -5 <<<"${out}")"

for malformed in false object string; do
  plan "control_${malformed}" pass
  [[ "${rc}" -eq 75 && "${runs}" == "control_${malformed} " && "$(markers)" -eq 1 ]] \
    && grep -q "control_leg_attempts=1" <<<"${out}" \
    && grep -q "retry was withheld because the private audit was missing or unreadable" <<<"${out}" \
    && ok "retry: paired_legs ${malformed} withholds the retry" \
    || fail "retry: paired_legs ${malformed}: rc=${rc} runs=${runs} out: $(tail -5 <<<"${out}")"
done

REAL_JQ="$(command -v jq)"
mkdir -p "${WORK}/jq-fail-bin"
cat > "${WORK}/jq-fail-bin/jq" <<EOF
#!/usr/bin/env bash
if [[ "\${STUB_JQ_FAIL_AUDIT_PREDICATE:-0}" == 1 && "\$*" == *'any('* \
      && "\${!#}" == "\${BENCHD_PRIVATE_AUDIT_PATH:-}" ]]; then
  exit 2
fi
exec ${REAL_JQ@Q} "\$@"
EOF
chmod +x "${WORK}/jq-fail-bin/jq"
PLAN_ENV=(PATH="${WORK}/jq-fail-bin:${PATH}" STUB_JQ_FAIL_AUDIT_PREDICATE=1)
plan control pass
[[ "${rc}" -eq 75 && "${runs}" == "control " && "$(markers)" -eq 1 ]] \
  && grep -q "control_leg_attempts=1" <<<"${out}" \
  && grep -q "retry was withheld because the private audit was missing or unreadable" <<<"${out}" \
  && ok "retry: a jq error while reading candidate timing withholds the retry" \
  || fail "retry: private-audit jq error: rc=${rc} runs=${runs} out: $(tail -5 <<<"${out}")"
PLAN_ENV=()

plan control control pass
[[ "${rc}" -eq 75 && "${runs}" == "control control " && "$(markers)" -eq 1 ]] \
  && grep -q "Serial control leg diverged twice" <<<"${out}" && grep -q "control_leg_attempts=2" <<<"${out}" \
  && ok "retry: two control failures end as infrastructure (exit 75), no third run" \
  || fail "retry: control twice: rc=${rc} runs=${runs} out: $(tail -3 <<<"${out}")"

plan floor pass
[[ "${rc}" -eq 1 && "${runs}" == "floor " && "$(markers)" -eq 1 ]] \
  && grep -q "control_leg_attempts=1" <<<"${out}" \
  && ok "retry: a candidate failure is never retried" \
  || fail "retry: floor: rc=${rc} runs=${runs} out: $(tail -3 <<<"${out}")"

TEARDOWN_MARKER="${WORK}/teardown-failed"
rm -f "${TEARDOWN_MARKER}"
PLAN_ENV=(TEARDOWN_FAILED_MARKER="${TEARDOWN_MARKER}")
plan teardown pass
[[ "${rc}" -eq 75 && "${runs}" == "teardown " && "$(markers)" -eq 1 ]] \
  && grep -q "control_leg_attempts=1" <<<"${out}" \
  && grep -q "Leg teardown failed" <<<"${out}" \
  && [[ -f "${TEARDOWN_MARKER}" && "$(stat -c %a "${TEARDOWN_MARKER}")" == 600 ]] \
  && ok "teardown: a LEG-TEARDOWN-FAILED seal exits 75 and writes a mode-600 marker" \
  || fail "teardown: rc=${rc} runs=${runs} out: $(tail -5 <<<"${out}")"
rm -f "${TEARDOWN_MARKER}"
PLAN_ENV=()

plan control floor
[[ "${rc}" -eq 1 && "${runs}" == "control floor " ]] && jq -e '.metrics.error | startswith("performance floor")' "${ROOT}/score.json" >/dev/null \
  && ok "retry: a candidate failure on attempt 2 is the result (no third run)" \
  || fail "retry: control+floor: rc=${rc} runs=${runs}"
cmp -s "${WORK}/runs.argv.1" "${WORK}/runs.argv.2" && grep -qE "/${FIRST}\.golden\.json$" "${WORK}/runs.argv.2" \
  && ok "retry: attempt 2 runs benchd with the identical argv (the same goldens)" \
  || fail "retry: attempt 2's argv differs from attempt 1's"

plan prefill pass
[[ "${rc}" -eq 0 && "${runs}" == "prefill pass " ]] \
  && ok "retry: a reference warmup prefill-token divergence is retried" \
  || fail "retry: prefill divergence: rc=${rc} runs=${runs}"

plan spawn pass
[[ "${rc}" -eq 1 && "${runs}" == "spawn " ]] && grep -q "control_leg_attempts=1" <<<"${out}" \
  && ok "retry: a control-leg failure that is not a token divergence (worker spawn) is final" \
  || fail "retry: spawn: rc=${rc} runs=${runs} out: $(tail -3 <<<"${out}")"

plan boot pass
[[ "${rc}" -eq 0 && "${runs}" == "boot pass " && "$(markers)" -eq 1 ]] \
  && grep -q "control_leg_attempts=2" <<<"${out}" && grep -q "pair-1 candidate resident did not boot" <<<"${out}" \
  && jq -e '.score == 1.2' "${ROOT}/score.json" >/dev/null \
  && ok "retry: pair-1 candidate boot failure then pass is scored from attempt 2" \
  || fail "retry: boot+pass: rc=${rc} runs=${runs} out: $(tail -5 <<<"${out}")"

plan boot2 pass
[[ "${rc}" -eq 1 && "${runs}" == "boot2 " && "$(markers)" -eq 1 ]] \
  && grep -q "control_leg_attempts=1" <<<"${out}" \
  && ok "retry: pair-2 candidate boot failure (paired_legs non-empty) is never retried" \
  || fail "retry: boot2: rc=${rc} runs=${runs} out: $(tail -3 <<<"${out}")"

plan boot_ctl pass
[[ "${rc}" -eq 0 && "${runs}" == "boot_ctl pass " && "$(markers)" -eq 1 ]] \
  && grep -q "control_leg_attempts=2" <<<"${out}" && grep -q "pair-1 serial-control resident did not boot" <<<"${out}" \
  && jq -e '.score == 1.2' "${ROOT}/score.json" >/dev/null \
  && ok "retry: pair-1 serial-control boot failure then pass is scored from attempt 2" \
  || fail "retry: boot_ctl+pass: rc=${rc} runs=${runs} out: $(tail -5 <<<"${out}")"

plan boot_ctl boot_ctl
[[ "${rc}" -eq 75 && "${runs}" == "boot_ctl boot_ctl " && "$(markers)" -eq 1 ]] \
  && grep -q "Serial-control leg boot failed twice" <<<"${out}" && grep -q "control_leg_attempts=2" <<<"${out}" \
  && ok "retry: two pair-1 serial-control boot failures end as infrastructure (exit 75)" \
  || fail "retry: boot_ctl twice: rc=${rc} runs=${runs} out: $(tail -3 <<<"${out}")"

plan boot_ctl2 pass
[[ "${rc}" -eq 1 && "${runs}" == "boot_ctl2 " && "$(markers)" -eq 1 ]] \
  && grep -q "control_leg_attempts=1" <<<"${out}" \
  && ok "retry: pair-2 serial-control boot failure (paired_legs non-empty) is never retried" \
  || fail "retry: boot_ctl2: rc=${rc} runs=${runs} out: $(tail -3 <<<"${out}")"

plan control boot
[[ "${rc}" -eq 1 && "${runs}" == "control boot " && "$(markers)" -eq 1 ]] \
  && grep -q "control_leg_attempts=2" <<<"${out}" \
  && ok "retry: control divergence then pair-1 boot failure uses the one retry budget (no third run)" \
  || fail "retry: control+boot: rc=${rc} runs=${runs} out: $(tail -5 <<<"${out}")"

cat > "${WORK}/benchd-noseal" <<'EOF'
#!/usr/bin/env bash
if [[ "$2" == --help ]]; then echo "--control-golden --mtp-depth --live-prompt"; exit 0; fi
echo run >> "${STUB_RUNS}"; exit 1
EOF
chmod +x "${WORK}/benchd-noseal"
echo '{"passed": false, "score": null, "metrics": {"error": "SERIAL-CONTROL-LEG-FAILED: benchmark free-run decode token mismatch at step 1"}}' > "${ROOT}/score.json"
: > "${WORK}/runs"
out="$(measure "${WORK}/full.json" STUB_RUNS="${WORK}/runs" \
  SERVE_UP_MEMINFO="${WORK}/meminfo" WAIT_FOR_MEMORY_TIMEOUT_S=0 BENCHD="${WORK}/benchd-noseal")"
rc=$?
[[ "${rc}" -eq 1 && "$(wc -l < "${WORK}/runs")" -eq 1 && ! -e "${ROOT}/score.json" ]] \
  && ok "retry: a stale control-divergence seal from before the run never triggers a retry" \
  || fail "retry: stale seal: rc=${rc} runs=$(wc -l < "${WORK}/runs")"

# The sandboxed teardown, with sudo and ipcs stubbed (the sandbox user is the
# current user; the stub ipcs lists nothing, so no real IPC object is touched).
SB="${WORK}/sb"
JOBDIR="${SB}/root/42-1"
mkdir -p "${SB}/bin" "${SB}/tmp" "${SB}/home" "${SB}/runner-temp" \
  "${SB}/shm" "${SB}/mqueue" "${SB}/sysvipc"
for d in cargo-home target kernels-out sock adapter-home; do mkdir -p "${JOBDIR}/${d}"; done
printf '%s\n' "${SB}/tmp" > "${SB}/wipe-dirs"
printf '%s\n' 'key shmid perms size cpid lpid nattch uid gid cuid cgid atime dtime ctime rss swap' > "${SB}/sysvipc/shm"
printf '%s\n' 'key semid perms nsems uid gid cuid cgid otime ctime' > "${SB}/sysvipc/sem"
printf '%s\n' 'key msqid perms cbytes qnum lspid lrpid uid gid cuid cgid stime rtime ctime' > "${SB}/sysvipc/msg"
REAL_FIND="$(command -v find)"
cat > "${SB}/bin/sudo" <<EOF
#!/usr/bin/env bash
[[ "\$1 \$2" == "-n -u" && "\$4" == "--" ]] || exit 97
shift 4
[[ "\$1" == kill ]] && exit 0
[[ "\$1 \$2 \$3" == "test -w ${SB}/home" ]] && exit 1
if [[ "\$1" == find ]]; then
  # nemosub cannot delete runner-owned socket dirs left by a crashed resident.
  shift
  echo 'zz-sudo-find-stderr-forbidden' >&2
  args=()
  for a in "\$@"; do
    if [[ "\$a" == -delete ]]; then
      args+=( ! -name 'runner-sock-leftover' ! -name 'zz-candidate-log-forbidden-dir' ! -name 'zz-candidate-log-forbidden-child' -delete )
    else
      args+=( "\$a" )
    fi
  done
  exec ${REAL_FIND@Q} "\${args[@]}"
fi
exec "\$@"
EOF
cat > "${SB}/bin/stat" <<EOF
#!/usr/bin/env bash
[[ "\$1 \$2 \$3 \$4" == "-Lc %u -- ${SB}/home" ]] && { echo 0; exit 0; }
exec $(command -v stat) "\$@"
EOF
cat > "${SB}/bin/find" <<EOF
#!/usr/bin/env bash
args=()
for a in "\$@"; do
  if [[ "\$a" == -delete ]]; then
    echo 'zz-find-stderr-forbidden' >&2
    args+=( ! -name 'zz-candidate-log-forbidden-dir' ! -name 'zz-candidate-log-forbidden-child' -delete )
  elif [[ "\$a" == -printf ]]; then
    echo 'zz-find-stderr-forbidden' >&2
    args+=( "\$a" )
  else
    args+=( "\$a" )
  fi
done
exec ${REAL_FIND@Q} "\${args[@]}"
EOF
chmod +x "${SB}/bin/find"
cat > "${SB}/bin/pgrep" <<EOF
#!/usr/bin/env bash
[[ "\$1 \$2" == "-u $(id -un)" ]] && exit 1
exec $(command -v pgrep) "\$@"
EOF
printf '#!/usr/bin/env bash\nexit 0\n' > "${SB}/bin/ipcs"
chmod +x "${SB}/bin/"*
SB_ENV=(PATH="${SB}/bin:${PATH}" SANDBOX_USER="$(id -un)" SANDBOX_ROOT="${SB}/root" SANDBOX_JOB=42-1
        RUNNER_TEMP="${SB}/runner-temp" SANDBOX_WIPE_LIST="${SB}/wipe-dirs" SANDBOX_WIPE_HOME="${SB}/home"
        SANDBOX_SYSVIPC_DIR="${SB}/sysvipc" SANDBOX_DEV_SHM="${SB}/shm" SANDBOX_DEV_MQUEUE="${SB}/mqueue")
env "${SB_ENV[@]}" "${ROOT}/tools/stage-cuda-engine.sh" --print-shim > "${SB}/engine"
cp "${WORK}/engine" "${SB}/engine.sandboxed"
chmod +x "${SB}/engine" "${SB}/engine.sandboxed"
echo setup-leftover > "${SB}/tmp/setup-leftover"
PLAN_ENV=("${SB_ENV[@]}" MLXFAST_ENGINE_BIN="${SB}/engine")
echo adapter-state > "${JOBDIR}/adapter-home/state"
mkdir "${JOBDIR}/sock/stale-leg"
echo socket-state > "${JOBDIR}/sock/stale-leg/state"
plan pass
[[ "${rc}" -eq 75 && -z "${runs}" ]] && grep -q "Pre-measurement sandbox reset failed" <<<"${out}" \
  && ok "measure: stale adapter-home or socket state fails before attempt 1" \
  || fail "measure: stale per-leg state: rc=${rc} runs=${runs} out: $(tail -3 <<<"${out}")"
rm -f "${JOBDIR}/adapter-home/state" "${JOBDIR}/sock/stale-leg/state"
rmdir "${JOBDIR}/sock/stale-leg"

chmod g+w "${SB}/engine.sandboxed"
plan pass
[[ "${rc}" -eq 75 && -z "${runs}" ]] && grep -q "Pre-measurement sandbox state check failed" <<<"${out}" \
  && ok "measure: a group-writable staged engine fails before attempt 1" \
  || fail "measure: writable build output: rc=${rc} runs=${runs} out: $(tail -3 <<<"${out}")"
chmod g-w "${SB}/engine.sandboxed"

plan pass
[[ "${rc}" -eq 0 && "${runs}" == "pass " && ! -e "${SB}/tmp/setup-leftover" ]] \
  && ok "measure: a file planted in shared tmp before the run is gone when attempt 1 starts" \
  || fail "measure: pre-measurement wipe: rc=${rc} runs=${runs} leftover=$(test -e "${SB}/tmp/setup-leftover" && echo yes || echo no)"
PLAN_ENV=("${SB_ENV[@]}" MLXFAST_ENGINE_BIN="${SB}/engine" STUB_JOBDIR="${JOBDIR}")
plan control pass
[[ "${rc}" -eq 0 && "${runs}" == "control pass " && "$(tr '\n' ' ' < "${WORK}/runs.seen")" == "0 0 " ]] \
  && ok "retry: every file attempt 1 planted in the five per-job sandbox dirs is gone when attempt 2 starts" \
  || fail "retry: sandboxed teardown: rc=${rc} runs=${runs} seen=$(tr '\n' ' ' < "${WORK}/runs.seen" 2>/dev/null) out: $(tail -4 <<<"${out}")"
for d in cargo-home target kernels-out sock adapter-home; do rm -rf "${JOBDIR:?}/${d}/"*; done

TEARDOWN_MARKER="${SB}/runner-temp/teardown-failed"
rm -f "${TEARDOWN_MARKER}"
PLAN_ENV=("${SB_ENV[@]}" MLXFAST_ENGINE_BIN="${SB}/engine" STUB_JOBDIR="${JOBDIR}" STUB_RUNNER_SOCK=1
          TEARDOWN_FAILED_MARKER="${TEARDOWN_MARKER}")
plan boot_ctl pass
[[ "${rc}" -eq 0 && "${runs}" == "boot_ctl pass " && ! -e "${JOBDIR}/sock/runner-sock-leftover" \
    && ! -e "${TEARDOWN_MARKER}" ]] \
  && ok "retry: clear_sandbox_job_dirs removes runner-owned leftovers the sandbox user could not delete" \
  || fail "retry: runner-owned sock cleanup: rc=${rc} runs=${runs} leftover=$(test -e "${JOBDIR}/sock/runner-sock-leftover" && echo yes || echo no)"
for d in cargo-home target kernels-out sock adapter-home; do rm -rf "${JOBDIR:?}/${d}/"*; done

LOG_FORBIDDEN_DIR='zz-candidate-log-forbidden-dir'
LOG_FORBIDDEN_CHILD='zz-candidate-log-forbidden-child'
mkdir "${JOBDIR}/target/${LOG_FORBIDDEN_DIR}"
echo x > "${JOBDIR}/target/${LOG_FORBIDDEN_DIR}/${LOG_FORBIDDEN_CHILD}"
PLAN_ENV=("${SB_ENV[@]}" MLXFAST_ENGINE_BIN="${SB}/engine" STUB_JOBDIR="${JOBDIR}"
          TEARDOWN_FAILED_MARKER="${TEARDOWN_MARKER}")
plan boot_ctl pass
[[ "${rc}" -eq 75 && "${runs}" == "boot_ctl " ]] \
  && [[ -f "${TEARDOWN_MARKER}" && "$(stat -c %a "${TEARDOWN_MARKER}")" == 600 ]] \
  && grep -q 'sandbox job subdirectory target is not empty after cleanup (2 leftover entries)' <<<"${out}" \
  && ! grep -qF "${LOG_FORBIDDEN_DIR}" <<<"${out}" \
  && ! grep -qF "${LOG_FORBIDDEN_CHILD}" <<<"${out}" \
  && ! grep -qF "${JOBDIR}" <<<"${out}" \
  && ! grep -qE 'zz-(sudo-)?find-stderr-forbidden' <<<"${out}" \
  && ok "retry: fail-closed cleanup logs only the job sub-dir name and a count, never entry names" \
  || fail "retry: fail-closed cleanup log: rc=${rc} runs=${runs} out: $(tail -5 <<<"${out}")"

rm -f "${TEARDOWN_MARKER}"
rm -f "${JOBDIR}/target/${LOG_FORBIDDEN_DIR}/${LOG_FORBIDDEN_CHILD}"
rmdir "${JOBDIR}/target/${LOG_FORBIDDEN_DIR}"
for d in cargo-home target kernels-out sock adapter-home; do rm -f "${JOBDIR}/${d}"/*; done
PLAN_ENV=("${SB_ENV[@]}" SANDBOX_WIPE_LIST="${SB}/no-such-list" MLXFAST_ENGINE_BIN="${SB}/engine")
plan pass
[[ "${rc}" -eq 75 && "${runs}" == "" ]] && grep -q "Pre-measurement sandbox reset failed" <<<"${out}" \
  && ok "measure: a pre-measurement wipe failure exits 75 before attempt 1" \
  || fail "measure: pre-measurement wipe failure: rc=${rc} runs=${runs} out: $(tail -3 <<<"${out}")"
WIPE_REAL="${REPO_ROOT}/tools/sandbox-wipe.sh"
WIPE_COUNT="${SB}/wipe-count"
rm -f "${ROOT}/tools/sandbox-wipe.sh"
cat > "${ROOT}/tools/sandbox-wipe.sh" <<EOF
#!/usr/bin/env bash
n=0
[[ -f "${WIPE_COUNT}" ]] && n=\$(<"${WIPE_COUNT}")
echo \$((n + 1)) > "${WIPE_COUNT}"
[[ "\${SANDBOX_WIPE_FAIL_ON:-0}" -eq \$((n + 1)) ]] \
  && { echo "sandbox-wipe.sh: deliberate test failure" >&2; exit 1; }
exec bash "${WIPE_REAL}" "\$@"
EOF
chmod +x "${ROOT}/tools/sandbox-wipe.sh"
rm -f "${WIPE_COUNT}"
PLAN_ENV=("${SB_ENV[@]}" MLXFAST_ENGINE_BIN="${SB}/engine" STUB_JOBDIR="${JOBDIR}" SANDBOX_WIPE_FAIL_ON=2)
plan control pass
ln -sf "${WIPE_REAL}" "${ROOT}/tools/sandbox-wipe.sh"
[[ "${rc}" -eq 75 && "${runs}" == "control " ]] && grep -q "retry teardown failed" <<<"${out}" \
  && ok "retry: a teardown that cannot reset the box exits 75 without a second run" \
  || fail "retry: teardown failure: rc=${rc} runs=${runs} out: $(tail -3 <<<"${out}")"
PLAN_ENV=()

MS_ARGS=(--preflight-only)
out="$(measure "${WORK}/full.json")"
MS_ARGS=()
grep -q "official_scoring_enabled" <<<"${out}" \
  && ok "measure: --preflight-only resolves every live golden and reaches the arm gate" \
  || fail "measure: --preflight-only did not reach the arm gate: $(tail -2 <<<"${out}")"

if [[ "${failures}" -eq 0 ]]; then
  echo "test-live-golden-rotation.sh: all cases passed"
  exit 0
fi
echo "test-live-golden-rotation.sh: ${failures} case(s) failed" >&2
exit 1

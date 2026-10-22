#!/usr/bin/env bash
#
# test-score-plausibility-gate.sh -- tools/score-plausibility-gate.sh
#
# Hermetic: temp score.json files only.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE="${REPO_ROOT}/tools/score-plausibility-gate.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

passed=0
failed=0

ok() {
  echo "ok    $1"
  passed=$((passed + 1))
}

bad() {
  echo "FAIL  $1"
  echo "      $2"
  failed=$((failed + 1))
}

write_score() {
  printf '%s' "$1" > "${WORK}/score.json"
}

run_gate() {
  "${GATE}" "${WORK}/score.json" 2>&1
}

# (a) passed true, three pairs, gain ~0.97 -> exit 0
write_score '{"passed": true, "score": 1.0, "metrics": {"paired_legs": [
  {"pair": 1, "control_prefill_seconds_per_token": 0.000271, "candidate_prefill_seconds_per_token": 0.000280},
  {"pair": 2, "control_prefill_seconds_per_token": 0.000271, "candidate_prefill_seconds_per_token": 0.000280},
  {"pair": 3, "control_prefill_seconds_per_token": 0.000271, "candidate_prefill_seconds_per_token": 0.000280}
]}}'
out="$(run_gate)"
rc=$?
if [[ "${rc}" -eq 0 && "${out}" == *"ok (max prefill gain"* ]]; then
  ok "(a) three plausible pairs"
else
  bad "(a) three plausible pairs" "rc=${rc} out=${out}"
fi

# (b) passed true, one pair at gain 2.0 -> exit 1, REFUSED names pair
write_score '{"passed": true, "score": 9.0, "metrics": {"paired_legs": [
  {"pair": 7, "control_prefill_seconds_per_token": 2.0, "candidate_prefill_seconds_per_token": 1.0}
]}}'
out="$(run_gate)"
rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"REFUSED pair 7:"* ]]; then
  ok "(b) excessive prefill gain refuses and names pair"
else
  bad "(b) excessive prefill gain refuses and names pair" "rc=${rc} out=${out}"
fi

# (c) passed false with huge gain -> exit 0
write_score '{"passed": false, "score": null, "metrics": {"paired_legs": [
  {"pair": 1, "control_prefill_seconds_per_token": 10.0, "candidate_prefill_seconds_per_token": 0.001}
]}}'
out="$(run_gate)"
rc=$?
if [[ "${rc}" -eq 0 && "${out}" == *"run did not pass"* ]]; then
  ok "(c) failed run skips gain check"
else
  bad "(c) failed run skips gain check" "rc=${rc} out=${out}"
fi

# (d) file missing -> exit 0
rm -f "${WORK}/score.json"
out="$("${GATE}" "${WORK}/score.json" 2>&1)"
rc=$?
if [[ "${rc}" -eq 0 && "${out}" == *"no score.json"* ]]; then
  ok "(d) missing score file"
else
  bad "(d) missing score file" "rc=${rc} out=${out}"
fi

# (e) passed true, empty paired_legs -> exit 1
write_score '{"passed": true, "score": 1.0, "metrics": {"paired_legs": []}}'
out="$(run_gate)"
rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"REFUSED"* ]]; then
  ok "(e) empty paired_legs refused"
else
  bad "(e) empty paired_legs refused" "rc=${rc} out=${out}"
fi

# (f) candidate_prefill_seconds_per_token 0 -> exit 1
write_score '{"passed": true, "score": 1.0, "metrics": {"paired_legs": [
  {"pair": 2, "control_prefill_seconds_per_token": 0.001, "candidate_prefill_seconds_per_token": 0}
]}}'
out="$(run_gate)"
rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"REFUSED"* ]]; then
  ok "(f) zero candidate prefill refused"
else
  bad "(f) zero candidate prefill refused" "rc=${rc} out=${out}"
fi

# (g) invalid JSON -> exit 1
write_score 'not-json'
out="$(run_gate)"
rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"REFUSED"* ]]; then
  ok "(g) invalid JSON refused"
else
  bad "(g) invalid JSON refused" "rc=${rc} out=${out}"
fi

# (h) gain exactly 1.5 -> exit 0
write_score '{"passed": true, "score": 1.0, "metrics": {"paired_legs": [
  {"pair": 1, "control_prefill_seconds_per_token": 1.5, "candidate_prefill_seconds_per_token": 1.0}
]}}'
out="$(run_gate)"
rc=$?
if [[ "${rc}" -eq 0 && "${out}" == *"ok (max prefill gain 1.5000"* ]]; then
  ok "(h) gain exactly at ceiling passes"
else
  bad "(h) gain exactly at ceiling passes" "rc=${rc} out=${out}"
fi

echo "score-plausibility-gate: ${passed} passed, ${failed} failed"
[[ "${failed}" -eq 0 ]]

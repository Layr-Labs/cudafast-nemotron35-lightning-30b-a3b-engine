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

write_audit() {
  printf '%s' "$1" > "${WORK}/audit.json"
}

run_gate() {
  if [[ -f "${WORK}/audit.json" ]]; then
    "${GATE}" "${WORK}/score.json" "${WORK}/audit.json" 2>&1
  else
    "${GATE}" "${WORK}/score.json" 2>&1
  fi
}

# Honest private audit: same score and pairs, seed windows 1:1.
write_audit_three_ok() {
  write_audit '{"score": 1.0, "metrics": {"paired_legs": [
  {"pair": 1, "control_seed_prefill_window_seconds_per_token": 0.001, "candidate_seed_prefill_window_seconds_per_token": 0.001},
  {"pair": 2, "control_seed_prefill_window_seconds_per_token": 0.001, "candidate_seed_prefill_window_seconds_per_token": 0.001},
  {"pair": 3, "control_seed_prefill_window_seconds_per_token": 0.001, "candidate_seed_prefill_window_seconds_per_token": 0.001}
]}}'
}

# (a) passed true, three pairs, gain ~0.97 -> exit 0
write_score '{"passed": true, "score": 1.0, "metrics": {"paired_legs": [
  {"pair": 1, "control_prefill_seconds_per_token": 0.000271, "candidate_prefill_seconds_per_token": 0.000280},
  {"pair": 2, "control_prefill_seconds_per_token": 0.000271, "candidate_prefill_seconds_per_token": 0.000280},
  {"pair": 3, "control_prefill_seconds_per_token": 0.000271, "candidate_prefill_seconds_per_token": 0.000280}
]}}'
write_audit_three_ok
out="$(run_gate)"
rc=$?
if [[ "${rc}" -eq 0 && "${out}" == *"ok (max prefill gain"* && "${out}" == *"seed windows within limits"* ]]; then
  ok "(a) three plausible pairs"
else
  bad "(a) three plausible pairs" "rc=${rc} out=${out}"
fi

# (b) passed true, one pair at gain 2.0 -> exit 1, REFUSED names pair
write_score '{"passed": true, "score": 9.0, "metrics": {"paired_legs": [
  {"pair": 7, "control_prefill_seconds_per_token": 2.0, "candidate_prefill_seconds_per_token": 1.0}
]}}'
write_audit '{"score": 9.0, "metrics": {"paired_legs": [
  {"pair": 7, "control_seed_prefill_window_seconds_per_token": 0.001, "candidate_seed_prefill_window_seconds_per_token": 0.001}
]}}'
out="$(run_gate)"
rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"REFUSED pair 7:"* ]]; then
  ok "(b) excessive prefill gain refuses and names pair"
else
  bad "(b) excessive prefill gain refuses and names pair" "rc=${rc} out=${out}"
fi

# (c) passed false with huge gain -> exit 0
rm -f "${WORK}/audit.json"
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
rm -f "${WORK}/score.json" "${WORK}/audit.json"
out="$("${GATE}" "${WORK}/score.json" 2>&1)"
rc=$?
if [[ "${rc}" -eq 0 && "${out}" == *"no score.json"* ]]; then
  ok "(d) missing score file"
else
  bad "(d) missing score file" "rc=${rc} out=${out}"
fi

# (e) passed true, empty paired_legs -> exit 1
write_score '{"passed": true, "score": 1.0, "metrics": {"paired_legs": []}}'
write_audit '{"score": 1.0, "metrics": {"paired_legs": []}}'
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
write_audit '{"score": 1.0, "metrics": {"paired_legs": [
  {"pair": 2, "control_seed_prefill_window_seconds_per_token": 0.001, "candidate_seed_prefill_window_seconds_per_token": 0.001}
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
write_audit '{"score": 1.0, "metrics": {"paired_legs": []}}'
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
write_audit '{"score": 1.0, "metrics": {"paired_legs": [
  {"pair": 1, "control_seed_prefill_window_seconds_per_token": 0.001, "candidate_seed_prefill_window_seconds_per_token": 0.001}
]}}'
out="$(run_gate)"
rc=$?
if [[ "${rc}" -eq 0 && "${out}" == *"ok (max prefill gain 1.5000"* ]]; then
  ok "(h) gain exactly at ceiling passes"
else
  bad "(h) gain exactly at ceiling passes" "rc=${rc} out=${out}"
fi

# (i) honest audit, seed ratios 1.0 and 1.1 -> ok
write_score '{"passed": true, "score": 2.5, "metrics": {"paired_legs": [
  {"pair": 1, "control_prefill_seconds_per_token": 1.0, "candidate_prefill_seconds_per_token": 1.0},
  {"pair": 2, "control_prefill_seconds_per_token": 1.1, "candidate_prefill_seconds_per_token": 1.0}
]}}'
write_audit '{"score": 2.5, "metrics": {"paired_legs": [
  {"pair": 1, "control_seed_prefill_window_seconds_per_token": 0.002, "candidate_seed_prefill_window_seconds_per_token": 0.002},
  {"pair": 2, "control_seed_prefill_window_seconds_per_token": 0.0022, "candidate_seed_prefill_window_seconds_per_token": 0.002}
]}}'
out="$(run_gate)"
rc=$?
if [[ "${rc}" -eq 0 && "${out}" == *"seed windows within limits"* ]]; then
  ok "(i) honest audit seed ratios 1.0 and 1.1"
else
  bad "(i) honest audit seed ratios 1.0 and 1.1" "rc=${rc} out=${out}"
fi

# (j) candidate seed 3x control -> REFUSED, no seed digits in output
write_score '{"passed": true, "score": 1.0, "metrics": {"paired_legs": [
  {"pair": 4, "control_prefill_seconds_per_token": 1.0, "candidate_prefill_seconds_per_token": 1.0}
]}}'
write_audit '{"score": 1.0, "metrics": {"paired_legs": [
  {"pair": 4, "control_seed_prefill_window_seconds_per_token": 0.003, "candidate_seed_prefill_window_seconds_per_token": 0.009}
]}}'
out="$(run_gate)"
rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"REFUSED pair 4:"* && "${out}" == *"free_decode_begin"* ]]; then
  if [[ "${out}" == *"0.003"* || "${out}" == *"0.009"* || "${out}" == *"3.000"* ]]; then
    bad "(j) candidate seed 3x control" "seed window digits leaked: ${out}"
  else
    ok "(j) candidate seed 3x control"
  fi
else
  bad "(j) candidate seed 3x control" "rc=${rc} out=${out}"
fi

# (k) candidate seed control/3 -> REFUSED (reused seed prefill)
write_score '{"passed": true, "score": 1.0, "metrics": {"paired_legs": [
  {"pair": 5, "control_prefill_seconds_per_token": 1.0, "candidate_prefill_seconds_per_token": 1.0}
]}}'
write_audit '{"score": 1.0, "metrics": {"paired_legs": [
  {"pair": 5, "control_seed_prefill_window_seconds_per_token": 0.003, "candidate_seed_prefill_window_seconds_per_token": 0.001}
]}}'
out="$(run_gate)"
rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"REFUSED pair 5:"* && "${out}" == *"reused earlier work"* ]]; then
  ok "(k) control seed 3x candidate refused"
else
  bad "(k) control seed 3x candidate refused" "rc=${rc} out=${out}"
fi

# (l) passed true with no audit argument -> REFUSED
write_score '{"passed": true, "score": 1.0, "metrics": {"paired_legs": [
  {"pair": 1, "control_prefill_seconds_per_token": 1.0, "candidate_prefill_seconds_per_token": 1.0}
]}}'
rm -f "${WORK}/audit.json"
out="$("${GATE}" "${WORK}/score.json" 2>&1)"
rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"private benchd audit missing"* ]]; then
  ok "(l) passed true without audit refused"
else
  bad "(l) passed true without audit refused" "rc=${rc} out=${out}"
fi

# (m) audit score differs from score.json -> REFUSED
write_score '{"passed": true, "score": 1.0, "metrics": {"paired_legs": [
  {"pair": 1, "control_prefill_seconds_per_token": 1.0, "candidate_prefill_seconds_per_token": 1.0}
]}}'
write_audit '{"score": 2.0, "metrics": {"paired_legs": [
  {"pair": 1, "control_seed_prefill_window_seconds_per_token": 0.001, "candidate_seed_prefill_window_seconds_per_token": 0.001}
]}}'
out="$(run_gate)"
rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"private audit does not match score.json"* ]]; then
  ok "(m) audit score mismatch refused"
else
  bad "(m) audit score mismatch refused" "rc=${rc} out=${out}"
fi

# (n) audit missing seed fields -> REFUSED
write_score '{"passed": true, "score": 1.0, "metrics": {"paired_legs": [
  {"pair": 1, "control_prefill_seconds_per_token": 1.0, "candidate_prefill_seconds_per_token": 1.0}
]}}'
write_audit '{"score": 1.0, "metrics": {"paired_legs": [
  {"pair": 1}
]}}'
out="$(run_gate)"
rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"seed-prefill window missing from the private audit"* ]]; then
  ok "(n) audit missing seed fields refused"
else
  bad "(n) audit missing seed fields refused" "rc=${rc} out=${out}"
fi

# (o) audit pair order differs -> REFUSED
write_score '{"passed": true, "score": 1.0, "metrics": {"paired_legs": [
  {"pair": 1, "control_prefill_seconds_per_token": 1.0, "candidate_prefill_seconds_per_token": 1.0},
  {"pair": 2, "control_prefill_seconds_per_token": 1.0, "candidate_prefill_seconds_per_token": 1.0}
]}}'
write_audit '{"score": 1.0, "metrics": {"paired_legs": [
  {"pair": 2, "control_seed_prefill_window_seconds_per_token": 0.001, "candidate_seed_prefill_window_seconds_per_token": 0.001},
  {"pair": 1, "control_seed_prefill_window_seconds_per_token": 0.001, "candidate_seed_prefill_window_seconds_per_token": 0.001}
]}}'
out="$(run_gate)"
rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"paired_legs does not match score.json"* ]]; then
  ok "(o) audit pair order mismatch refused"
else
  bad "(o) audit pair order mismatch refused" "rc=${rc} out=${out}"
fi

echo "score-plausibility-gate: ${passed} passed, ${failed} failed"
[[ "${failed}" -eq 0 ]]

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
GATE_DECLARED_DEPTH=3

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
  printf '%s' "$1" | jq '.metrics.effective_spec_depth //= 3' > "${WORK}/audit.json"
}

write_audit_without_depth() {
  printf '%s' "$1" > "${WORK}/audit.json"
}

run_gate() {
  local audit="" args
  [[ ! -f "${WORK}/audit.json" ]] || audit="${WORK}/audit.json"
  args=("${WORK}/score.json" "${audit}" "${GATE_DECLARED_DEPTH:-3}")
  [[ -z "${GATE_CONTRACT:-}" ]] || args+=("${GATE_CONTRACT}")
  "${GATE}" "${args[@]}" 2>&1
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
  {"pair": 1, "control_prefill_seconds_per_token": 0.000271, "candidate_prefill_seconds_per_token": 0.000280, "control_decode_seconds_per_token": 0.0134, "candidate_decode_seconds_per_token": 0.0113},
  {"pair": 2, "control_prefill_seconds_per_token": 0.000271, "candidate_prefill_seconds_per_token": 0.000280, "control_decode_seconds_per_token": 0.0134, "candidate_decode_seconds_per_token": 0.0113},
  {"pair": 3, "control_prefill_seconds_per_token": 0.000271, "candidate_prefill_seconds_per_token": 0.000280, "control_decode_seconds_per_token": 0.0134, "candidate_decode_seconds_per_token": 0.0113}
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
  {"pair": 7, "control_prefill_seconds_per_token": 2.0, "candidate_prefill_seconds_per_token": 1.0, "control_decode_seconds_per_token": 0.0134, "candidate_decode_seconds_per_token": 0.0113}
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
  {"pair": 1, "control_prefill_seconds_per_token": 10.0, "candidate_prefill_seconds_per_token": 0.001, "control_decode_seconds_per_token": 0.0134, "candidate_decode_seconds_per_token": 0.0113}
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
  {"pair": 2, "control_prefill_seconds_per_token": 0.001, "candidate_prefill_seconds_per_token": 0, "control_decode_seconds_per_token": 0.0134, "candidate_decode_seconds_per_token": 0.0113}
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
  {"pair": 1, "control_prefill_seconds_per_token": 1.5, "candidate_prefill_seconds_per_token": 1.0, "control_decode_seconds_per_token": 0.0134, "candidate_decode_seconds_per_token": 0.0113}
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
  {"pair": 1, "control_prefill_seconds_per_token": 1.0, "candidate_prefill_seconds_per_token": 1.0, "control_decode_seconds_per_token": 0.0134, "candidate_decode_seconds_per_token": 0.0113},
  {"pair": 2, "control_prefill_seconds_per_token": 1.1, "candidate_prefill_seconds_per_token": 1.0, "control_decode_seconds_per_token": 0.0134, "candidate_decode_seconds_per_token": 0.0113}
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
  {"pair": 4, "control_prefill_seconds_per_token": 1.0, "candidate_prefill_seconds_per_token": 1.0, "control_decode_seconds_per_token": 0.0134, "candidate_decode_seconds_per_token": 0.0113}
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
  {"pair": 5, "control_prefill_seconds_per_token": 1.0, "candidate_prefill_seconds_per_token": 1.0, "control_decode_seconds_per_token": 0.0134, "candidate_decode_seconds_per_token": 0.0113}
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
  {"pair": 1, "control_prefill_seconds_per_token": 1.0, "candidate_prefill_seconds_per_token": 1.0, "control_decode_seconds_per_token": 0.0134, "candidate_decode_seconds_per_token": 0.0113}
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
  {"pair": 1, "control_prefill_seconds_per_token": 1.0, "candidate_prefill_seconds_per_token": 1.0, "control_decode_seconds_per_token": 0.0134, "candidate_decode_seconds_per_token": 0.0113}
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
  {"pair": 1, "control_prefill_seconds_per_token": 1.0, "candidate_prefill_seconds_per_token": 1.0, "control_decode_seconds_per_token": 0.0134, "candidate_decode_seconds_per_token": 0.0113}
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
  {"pair": 1, "control_prefill_seconds_per_token": 1.0, "candidate_prefill_seconds_per_token": 1.0, "control_decode_seconds_per_token": 0.0134, "candidate_decode_seconds_per_token": 0.0113},
  {"pair": 2, "control_prefill_seconds_per_token": 1.0, "candidate_prefill_seconds_per_token": 1.0, "control_decode_seconds_per_token": 0.0134, "candidate_decode_seconds_per_token": 0.0113}
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

# (p) decode gain 6.0 -> refused (the contract's 5.0 decode ceiling)
write_score '{"passed": true, "score": 2.0, "metrics": {"paired_legs": [
  {"pair": 1, "control_prefill_seconds_per_token": 0.000271, "candidate_prefill_seconds_per_token": 0.000280, "control_decode_seconds_per_token": 0.012, "candidate_decode_seconds_per_token": 0.002}
]}}'
write_audit '{"score": 2.0, "metrics": {"paired_legs": [
  {"pair": 1, "control_seed_prefill_window_seconds_per_token": 0.001, "candidate_seed_prefill_window_seconds_per_token": 0.001}
]}}'
out="$(run_gate)"; rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"decode gain 6.0000 exceeds the ceiling 5"* ]]; then ok "(p) decode gain above 5.0 refused"; else bad "(p) decode gain above 5.0 refused" "rc=${rc} out=${out}"; fi

# (q) decode gain exactly 5.0 -> ok
write_score '{"passed": true, "score": 2.0, "metrics": {"paired_legs": [
  {"pair": 1, "control_prefill_seconds_per_token": 0.000271, "candidate_prefill_seconds_per_token": 0.000280, "control_decode_seconds_per_token": 0.01, "candidate_decode_seconds_per_token": 0.002}
]}}'
out="$(run_gate)"; rc=$?
if [[ "${rc}" -eq 0 ]]; then ok "(q) decode gain exactly 5.0 passes"; else bad "(q) decode gain exactly 5.0 passes" "rc=${rc} out=${out}"; fi

# (r) depth 3, candidate seed 1.33x control -> ok under its 1.5 limit
write_score '{"passed": true, "score": 1.0, "metrics": {"paired_legs": [
  {"pair": 1, "control_prefill_seconds_per_token": 1.0, "candidate_prefill_seconds_per_token": 1.0, "control_decode_seconds_per_token": 1.0, "candidate_decode_seconds_per_token": 1.0}
]}}'
write_audit '{"score": 1.0, "metrics": {"effective_spec_depth": 3, "paired_legs": [
  {"pair": 1, "control_seed_prefill_window_seconds_per_token": 1.0, "candidate_seed_prefill_window_seconds_per_token": 1.33}
]}}'
out="$(run_gate)"; rc=$?
if [[ "${rc}" -eq 0 ]]; then ok "(r) depth 3 seed ratio 1.33 passes"; else bad "(r) depth 3 seed ratio 1.33 passes" "rc=${rc} out=${out}"; fi

# (s) depth 6, candidate seed 1.47x control -> ok under its 1.7 limit
write_audit '{"score": 1.0, "metrics": {"effective_spec_depth": 6, "paired_legs": [
  {"pair": 1, "control_seed_prefill_window_seconds_per_token": 1.0, "candidate_seed_prefill_window_seconds_per_token": 1.47}
]}}'
GATE_DECLARED_DEPTH=6; out="$(run_gate)"; rc=$?
if [[ "${rc}" -eq 0 ]]; then ok "(s) depth 6 seed ratio 1.47 passes"; else bad "(s) depth 6 seed ratio 1.47 passes" "rc=${rc} out=${out}"; fi

# (s2) depth 6, candidate seed 1.6x control -> ok under its 1.7 limit (the old fixed 1.5 would refuse it)
write_audit '{"score": 1.0, "metrics": {"effective_spec_depth": 6, "paired_legs": [
  {"pair": 1, "control_seed_prefill_window_seconds_per_token": 1.0, "candidate_seed_prefill_window_seconds_per_token": 1.6}
]}}'
GATE_DECLARED_DEPTH=6; out="$(run_gate)"; rc=$?
if [[ "${rc}" -eq 0 ]]; then ok "(s2) depth 6 seed ratio 1.6 passes"; else bad "(s2) depth 6 seed ratio 1.6 passes" "rc=${rc} out=${out}"; fi

# (t) depth 6, candidate seed 1.75x control -> refused above its 1.7 limit
write_audit '{"score": 1.0, "metrics": {"effective_spec_depth": 6, "paired_legs": [
  {"pair": 1, "control_seed_prefill_window_seconds_per_token": 1.0, "candidate_seed_prefill_window_seconds_per_token": 1.75}
]}}'
GATE_DECLARED_DEPTH=6; out="$(run_gate)"; rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"REFUSED pair 1:"* ]]; then ok "(t) depth 6 seed ratio 1.75 refused"; else bad "(t) depth 6 seed ratio 1.75 refused" "rc=${rc} out=${out}"; fi

# (u) a passed audit without the candidate depth -> refused
write_audit_without_depth '{"score": 1.0, "metrics": {"paired_legs": [
  {"pair": 1, "control_seed_prefill_window_seconds_per_token": 1.0, "candidate_seed_prefill_window_seconds_per_token": 1.0}
]}}'
GATE_DECLARED_DEPTH=3
out="$(run_gate)"; rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"effective_spec_depth"* ]]; then ok "(u) missing candidate depth refused"; else bad "(u) missing candidate depth refused" "rc=${rc} out=${out}"; fi

# (u2) the audit depth cannot select a limit that differs from the declaration.
write_audit '{"score": 1.0, "metrics": {"effective_spec_depth": 6, "paired_legs": [
  {"pair": 1, "control_seed_prefill_window_seconds_per_token": 1.0, "candidate_seed_prefill_window_seconds_per_token": 1.0}
]}}'
out="$(run_gate)"; rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"does not match declared depth 3"* ]]; then ok "(u2) audit depth mismatch refused"; else bad "(u2) audit depth mismatch refused" "rc=${rc} out=${out}"; fi

# (u3) serial accepts an absent or zero depth, but no other value.
write_audit_without_depth '{"score": 1.0, "metrics": {"paired_legs": [
  {"pair": 1, "control_seed_prefill_window_seconds_per_token": 1.0, "candidate_seed_prefill_window_seconds_per_token": 1.0}
]}}'
GATE_DECLARED_DEPTH=0; out="$(run_gate)"; rc=$?
if [[ "${rc}" -eq 0 ]]; then ok "(u3) serial audit with absent depth passes"; else bad "(u3) serial audit with absent depth passes" "rc=${rc} out=${out}"; fi
write_audit '{"score": 1.0, "metrics": {"effective_spec_depth": 1, "paired_legs": [
  {"pair": 1, "control_seed_prefill_window_seconds_per_token": 1.0, "candidate_seed_prefill_window_seconds_per_token": 1.0}
]}}'
GATE_DECLARED_DEPTH=0; out="$(run_gate)"; rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"does not match declared depth 0"* ]]; then ok "(u3) serial audit with depth one refused"; else bad "(u3) serial audit with depth one refused" "rc=${rc} out=${out}"; fi

# (v) a depth with no configured limit -> refused
write_audit '{"score": 1.0, "metrics": {"effective_spec_depth": 7, "paired_legs": [
  {"pair": 1, "control_seed_prefill_window_seconds_per_token": 1.0, "candidate_seed_prefill_window_seconds_per_token": 1.0}
]}}'
GATE_DECLARED_DEPTH=7
out="$(run_gate)"; rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"no valid seed-window limit"* ]]; then ok "(v) missing depth limit refused"; else bad "(v) missing depth limit refused" "rc=${rc} out=${out}"; fi

# (w) an old fixture keeps the fixed 1.5 limit and does not require a depth
OLD_FIXTURE="${WORK}/old-fixture.json"
jq 'del(.seed_window_cand_over_control_limit_by_depth)' \
  "${REPO_ROOT}/fixtures/nemotron35_lightning_30b_a3b_track.json" > "${OLD_FIXTURE}"
write_audit_without_depth '{"score": 1.0, "metrics": {"paired_legs": [
  {"pair": 1, "control_seed_prefill_window_seconds_per_token": 1.0, "candidate_seed_prefill_window_seconds_per_token": 1.6}
]}}'
write_audit '{"score": 1.0, "metrics": {"effective_spec_depth": 3, "paired_legs": [
  {"pair": 1, "control_seed_prefill_window_seconds_per_token": 1.0, "candidate_seed_prefill_window_seconds_per_token": 1.6}
]}}'
GATE_DECLARED_DEPTH=3
GATE_CONTRACT="${OLD_FIXTURE}"
out="$(run_gate)"; rc=$?
unset GATE_CONTRACT
if [[ "${rc}" -eq 1 && "${out}" == *"REFUSED pair 1:"* && "${out}" == *"fixed 1.5 limit"* && "${out}" != *"Traceback"* ]]; then
  ok "(w) old fixture ratio 1.6 refuses cleanly"
else
  bad "(w) old fixture ratio 1.6 refuses cleanly" "rc=${rc} out=${out}"
fi

echo "score-plausibility-gate: ${passed} passed, ${failed} failed"
[[ "${failed}" -eq 0 ]]

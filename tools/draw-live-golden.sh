#!/usr/bin/env bash
# draw-live-golden.sh -- draw the scored prompt for ONE ranked job.
#
# TRUSTED, runner-side. .github/workflows/benchmark.yml runs it after Setup and
# before the GPU-exclusive measurement step, writes the name to a runner-private
# file under RUNNER_TEMP, and hands it to tools/nemotron35-measure-and-score.sh
# as MLXFAST_LIVE_GOLDEN_CHOICE. Both legs of every pair of the job measure the
# one name drawn here.
#
# WHY A DRAW. Submissions are unlimited and a candidate engine sees the prompt
# at run time. One fixed scored prompt is a target a team can reconstruct over
# many runs through whatever side channel is left; a prompt drawn per job from
# the hidden pool is a moving one.
#
# THE SOURCE IS /dev/urandom, never a run id, a sha, a clock or anything else a
# participant can see or steer. Rejection sampling keeps the draw exactly
# uniform over the pool (no modulo bias).
#
# The pool is the fixture's live_golden_rotation.pool. A fixture with no
# rotation block prints live_golden, exactly as before rotation existed.
#
# A fixture that declares live_goldens scores EVERY listed prompt in every job
# (one pair per prompt, tools/nemotron35-measure-and-score.sh), so there is
# nothing to draw: this prints nothing, and the measurement refuses a non-empty
# MLXFAST_LIVE_GOLDEN_CHOICE.
#
# Usage: tools/draw-live-golden.sh [FIXTURE]     prints one prompt name, or nothing
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
FIXTURE="${1:-${ROOT}/fixtures/nemotron35_lightning_30b_a3b_track.json}"
die() { printf 'draw-live-golden: %s\n' "$*" >&2; exit 1; }

[[ -f "${FIXTURE}" ]] || die "fixture not found: ${FIXTURE}"
command -v jq >/dev/null 2>&1 || die "jq is required"

if [[ "$(jq -r 'has("live_goldens")' "${FIXTURE}")" == "true" ]]; then
  [[ "$(jq -r 'has("live_golden_rotation")' "${FIXTURE}")" != "true" ]] \
    || die "the fixture declares both live_goldens and live_golden_rotation; a job either measures every live golden or draws one prompt"
  exit 0
fi
if [[ "$(jq -r 'has("live_golden_rotation")' "${FIXTURE}")" != "true" ]]; then
  live="$(jq -r '.live_golden // empty' "${FIXTURE}")"
  [[ -n "${live}" ]] || die "the fixture declares neither live_golden_rotation nor live_golden"
  printf '%s\n' "${live}"
  exit 0
fi

mode="$(jq -r '.live_golden_rotation.mode // empty' "${FIXTURE}")"
[[ "${mode}" == "per_job_random" ]] || die "live_golden_rotation.mode is '${mode}'; only per_job_random is defined"
mapfile -t POOL < <(jq -r '.live_golden_rotation.pool // [] | .[]' "${FIXTURE}")
n="${#POOL[@]}"
[[ "${n}" -ge 1 ]] || die "live_golden_rotation.pool is empty; there is nothing to draw"
# Every name must be a timed_prompt_pool entry; the full pin check (serial
# golden plus every per-depth oracle) is tools/ranked-box-preflight.sh's.
for name in "${POOL[@]}"; do
  [[ "${name}" =~ ^[a-z0-9_-]+$ ]] || die "rotation pool name '${name}' is not [a-z0-9_-]+"
  jq -e --arg f "/${name}.golden.json" 'any(.timed_prompt_pool[]; .r2_path | endswith($f))' "${FIXTURE}" >/dev/null \
    || die "rotation pool name '${name}' has no timed_prompt_pool entry"
done

# Uniform on [0, n): draw 32 bits, reject the top partial bucket.
limit=$(( (1 << 32) - ((1 << 32) % n) ))
while :; do
  r="$(od -An -N4 -tu4 /dev/urandom | tr -d '[:space:]')"
  [[ "${r}" =~ ^[0-9]+$ ]] || die "could not read /dev/urandom"
  (( r < limit )) && break
done
printf '%s\n' "${POOL[r % n]}"

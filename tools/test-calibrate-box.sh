#!/usr/bin/env bash
#
# test-calibrate-box.sh -- the mechanics of the per-box baseline calibrator.
#
# tools/calibrate-box.sh makes a wiring claim: it takes the GPU lock and runs
# EXACTLY ONE `benchd calibrate-baseline`, carrying every value the band will
# record -- the reference workspace, the staged adapter RELATIVE to it, the
# weights, the fixture's live golden, the passes, the box, the track, the prompt,
# the reference commit and the benchd source commit read from the pinned benchd's
# manifest.
#
# IT BOOTS NO SERVE. benchd owns the residency now: it runs the reference tree's
# own `serve-up.sh --boot --spec serial --draft-len 0` once PER PASS and `--stop`s
# it before the next pass boots. A driver that booted its own serve would certify
# a different arrangement than the scored run uses, so "serve-up was never run" is
# an assertion here, not an omission.
#
# This suite drives the REAL script with STUB collaborators (benchd, flock,
# setsid) that record how they were called, against a THROWAWAY git repository
# standing in for the staged reference tree.
#
# Hermetic: no GPU, no lock, no weights, no resident engine, no network.
#
# Cases:
#   1. --dry-run prints the one calibrate-baseline command with every flag and
#      the per-leg boot/stop benchd will run, and creates no output, takes no
#      lock and boots nothing.
#   2. a full mechanics run runs exactly one calibrate-baseline, with every flag,
#      and boots NO serve of its own.
#   3. a reference workspace at the wrong commit refuses BY NAME before anything
#      boots.
#   4. an unset MLXFAST_BASELINE_WORKSPACE refuses by name.
#   5. --passes 1 refuses: the CV is undefined below two passes.
#   6. a box argument that disagrees with RUNNER_NAME refuses by name.
#   7. a preset SERVE_UP_SPECULATIVE refuses by name: the control leg is serial.
#   8. an inherited DS4_RESIDENT_SOCKET refuses by name.
#   9. a reference workspace with no staged adapter refuses by name.
#  10. an unattributable benchd refuses by name: the band records what measured
#      it.
set -uo pipefail
# A hosted runner exports RUNNER_NAME, and calibrate-box lets it win over --box;
# every case here names its own box, so the ambient runner identity must not leak in.
unset RUNNER_NAME

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRIVER="${REPO_ROOT}/tools/calibrate-box.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }

command -v jq  >/dev/null 2>&1 || { echo "test-calibrate-box.sh: jq is required" >&2; exit 1; }
command -v git >/dev/null 2>&1 || { echo "test-calibrate-box.sh: git is required" >&2; exit 1; }

BIN="${WORK}/bin"
mkdir -p "${BIN}"

cat > "${BIN}/flock" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
cat > "${BIN}/setsid" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
# serve-up TRIPWIRE. The driver must not boot a serve at all -- benchd does. It
# is placed at the path the driver would use if it ever went back to wrapping,
# and it records the call rather than refusing, so the assertion below reads
# "it was never called" instead of "the run died".
cat > "${BIN}/serve-up.sh" <<'EOF'
#!/usr/bin/env bash
printf 'serve-up weights=%s speculative=%s\n' \
  "${SERVE_UP_WEIGHTS_DIR:-}" "${SERVE_UP_SPECULATIVE:-}" >> "${STUB_SERVE_LOG}"
exec "$@"
EOF
# benchd stub: records argv and writes a plausible calibration file.
cat > "${BIN}/benchd" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${STUB_BENCHD_LOG}"
printf 'private=%s\n' "${MLXFAST_PRIVATE_DIR:-}" >> "${STUB_PRIVATE_LOG}"
if [ -n "${MLXFAST_PRIVATE_DIR:-}" ]; then
  printf 'mode=%s\n' "$(stat -c %a "${MLXFAST_PRIVATE_DIR}")" >> "${STUB_PRIVATE_LOG}"
  find "${MLXFAST_PRIVATE_DIR}" -maxdepth 1 -type f -printf 'file=%f\n' \
    | sort >> "${STUB_PRIVATE_LOG}"
fi
[ "${1:-}" = "calibrate-baseline" ] || exit 1
out=""
while [ $# -gt 0 ]; do
  [ "$1" = "--out" ] && out="$2"
  shift
done
[ -n "${out}" ] || exit 1
printf '{"version":3}\n' > "${out}"
exit "${STUB_BENCHD_EXIT:-0}"
EOF
chmod 755 "${BIN}"/*

# --- the throwaway reference tree -------------------------------------------
# A real git repository, so `git rev-parse HEAD` answers, plus the build outputs
# tools/ranked-box-preflight.sh and the driver look for.
WS="${WORK}/reference"
mkdir -p "${WS}"
git -C "${WS}" init -q
git -C "${WS}" config user.email "test@example.com"
git -C "${WS}" config user.name "test"
echo reference > "${WS}/README.md"
git -C "${WS}" add README.md
git -C "${WS}" commit -qm "reference tree"
REF_COMMIT="$(git -C "${WS}" rev-parse HEAD)"
mkdir -p "${WS}/.build/vllm-overlay/vllm" "${WS}/.build/release" "${WS}/tools"
: > "${WS}/.build/vllm-overlay/vllm/__init__.py"
: > "${WS}/.build/release/mlxfast-runtime-worker"
# The reference tree's own serve script. The driver only checks it is there and
# executable -- benchd is what runs it.
printf '#!/usr/bin/env bash\nexit 0\n' > "${WS}/tools/serve-up.sh"
chmod +x "${WS}/.build/release/mlxfast-runtime-worker" \
         "${WS}/tools/serve-up.sh"

# The pinned benchd's manifest: where the band's benchd_source_commit is read
# from. The driver never invents that value.
BENCHD_DIR="${WORK}/benchd-bin"
mkdir -p "${BENCHD_DIR}"
BENCHD_COMMIT="1111111111111111111111111111111111111111"
jq -n --arg c "${BENCHD_COMMIT}" '{sha256: "aa", bytes: 1, source_commit: $c}' \
  > "${BENCHD_DIR}/benchd.manifest.json"

# A fixture whose baseline_reference_commit is that tree's HEAD. Every other
# value is the repository's own, so the golden name and the pool stay real.
FIXTURE="${WORK}/fixture.json"
jq --arg c "${REF_COMMIT}" '.baseline_reference_commit = $c' \
  "${REPO_ROOT}/fixtures/nemotron35_lightning_30b_a3b_track.json" > "${FIXTURE}"
mapfile -t LIVE_GOLDENS < <(jq -r '.live_goldens[]' "${FIXTURE}")
# One --golden/--prompt pair per live golden, in fixture order.
GOLDEN_NEEDLES=()
for name in "${LIVE_GOLDENS[@]}"; do
  GOLDEN_NEEDLES+=("--golden ${WORK}/goldens/${name}.golden.json --prompt ${name}")
done

# The staged golden pool. The track goldens are organizer material published in
# R2 and staged on the box, so there is no copy in this checkout to point at:
# the pool is synthesized here and named through the same variable the box
# exports. The driver resolves the live golden's PATH from it; the bytes are
# benchd's business, and benchd is a stub in this suite.
GOLDEN_DIR="${WORK}/goldens"
mkdir -p "${GOLDEN_DIR}"
for name in "${LIVE_GOLDENS[@]}"; do echo '{}' > "${GOLDEN_DIR}/${name}.golden.json"; done

WEIGHTS="${WORK}/weights"; mkdir -p "${WEIGHTS}"
RUNNER_TEMP_DIR="${WORK}/runner-temp"; mkdir -p "${RUNNER_TEMP_DIR}"
LOCK="${WORK}/never-taken.lock"
BOX="test-box-1"

drive() { # drive OUTFILE [args...]
  local out="$1"; shift
  env PATH="${BIN}:${PATH}" \
      CALIBRATE_BOX_SETSID=1 \
      MLXFAST_BASELINE_WORKSPACE="${WS_OVERRIDE-${WS}}" \
      MLXFAST_TARGET_SNAPSHOT_DIR="${WEIGHTS}" \
      MLXFAST_NEMOTRON35_GOLDEN_DIR="${GOLDEN_DIR_OVERRIDE-${GOLDEN_DIR}}" \
      BENCHD_BIN_DIR="${BENCHD_DIR}" \
      STUB_SERVE_LOG="${WORK}/serve.log" \
      STUB_BENCHD_LOG="${WORK}/benchd.log" \
      STUB_PRIVATE_LOG="${WORK}/private.log" \
      STUB_BENCHD_EXIT="${STUB_BENCHD_EXIT:-0}" \
      RUNNER_TEMP="${RUNNER_TEMP_DIR}" \
      BENCHD="${BIN}/benchd" \
      "${DRIVER}" "$@" --fixture "${FIXTURE_OVERRIDE-${FIXTURE}}" --lock "${LOCK}" > "${out}" 2>&1
  return $?
}

# The benchmark inputs this driver must never touch, as one digest.
inputs_digest() {
  find "${REPO_ROOT}/fixtures" "${REPO_ROOT}/correctness_prompts" -type f -print0 \
    | sort -z | xargs -0 shasum -a 256 \
    | shasum -a 256 | awk '{print $1}'
}
INPUTS_BEFORE="$(inputs_digest)"

# --- case 1: --dry-run ------------------------------------------------------
: > "${WORK}/serve.log"; : > "${WORK}/benchd.log"; : > "${WORK}/private.log"
DRY="${WORK}/dry.out"
drive "${DRY}" "${BOX}" "${WORK}/out1/baseline-calibration.json" --dry-run
[ $? -eq 0 ] || { fail "case 1: --dry-run exited non-zero"; sed 's/^/    /' "${DRY}" >&2; }

DRY_FLAGS=(
  "--baseline-workspace ${WS}"
  "--engine .build/release/mlxfast-runtime-worker"
  "--weights ${WEIGHTS}"
  "${GOLDEN_NEEDLES[@]}"
  "--out ${WORK}/out1/baseline-calibration.json"
  "--passes 3"
  "--box ${BOX}"
  "--track nemotron3.5-lightning-30b-a3b-cuda-v1"
  "--reference-commit ${REF_COMMIT}"
  "--benchd-source-commit ${BENCHD_COMMIT}"
  "--warmup-golden ${REPO_ROOT}/correctness_prompts/public-longcopy-gate-english-496.golden.json"
)
dry_missing=0
for needle in "${DRY_FLAGS[@]}"; do
  grep -qF -- "${needle}" "${DRY}" || { fail "case 1: the plan does not carry '${needle}'"; dry_missing=1; }
done
[ "${dry_missing}" -eq 0 ] && ok "case 1: the plan carries all ${#DRY_FLAGS[@]} calibrate-baseline flags"
# The plan states the per-leg boot/stop benchd will run, so an operator reading
# it sees which tree serves the control leg and how it ends.
grep -qF -- "${WS}/tools/serve-up.sh --boot --spec serial --draft-len 0" "${DRY}" \
  && grep -qF -- "${WS}/tools/serve-up.sh --stop --socket" "${DRY}" \
  && ok "case 1: the plan names the per-leg boot and stop benchd runs on the reference tree" \
  || fail "case 1: the plan does not name the per-leg boot/stop"
[ -e "${LOCK}" ] && fail "case 1: --dry-run created the lock file"
[ -e "${WORK}/out1" ] && fail "case 1: --dry-run created the output directory"
[ -s "${WORK}/serve.log" ] && fail "case 1: --dry-run booted serve-up"
[ -s "${WORK}/benchd.log" ] && fail "case 1: --dry-run ran benchd"
ok "case 1: --dry-run took no lock, booted nothing and wrote nothing"

# --- case 2: a full mechanics run -------------------------------------------
: > "${LOCK}"
: > "${WORK}/serve.log"; : > "${WORK}/benchd.log"; : > "${WORK}/private.log"
OUT="${WORK}/run/baseline-calibration.json"
RUN_OUT="${WORK}/run.out"
drive "${RUN_OUT}" "${BOX}" "${OUT}"
rc=$?
[ "${rc}" -eq 0 ] || { fail "case 2: the run exited ${rc}"; sed 's/^/    /' "${RUN_OUT}" >&2; }

# THE DRIVER BOOTS NO SERVE. benchd owns the residency, and a driver that went
# back to wrapping would certify an arrangement the scored run does not use.
[ -s "${WORK}/serve.log" ] \
  && fail "case 2: the driver booted a serve of its own; benchd boots each leg's resident" \
  || ok "case 2: the driver booted no serve -- benchd owns the residency"

cal_calls="$(grep -c '^calibrate-baseline ' "${WORK}/benchd.log" 2>/dev/null || printf '0')"
[ "${cal_calls}" = "1" ] \
  && ok "case 2: exactly one calibrate-baseline call" \
  || fail "case 2: ${cal_calls} calibrate-baseline call(s)"
cal="$(grep '^calibrate-baseline ' "${WORK}/benchd.log" || true)"
run_missing=0
for needle in "--baseline-workspace ${WS}" \
              "--engine .build/release/mlxfast-runtime-worker" \
              "--weights ${WEIGHTS}" \
              "--out ${OUT}" "--passes 3" "--box ${BOX}" \
              "--track nemotron3.5-lightning-30b-a3b-cuda-v1" \
              "--reference-commit ${REF_COMMIT}" \
              "--benchd-source-commit ${BENCHD_COMMIT}"; do
  case "${cal}" in
    *"${needle}"*) ;;
    *) fail "case 2: calibrate-baseline was called without '${needle}'"; run_missing=1 ;;
  esac
done
[ "${run_missing}" -eq 0 ] \
  && ok "case 2: every value the band records was passed explicitly, none left to a fallback"
# THE ENGINE IS RELATIVE. An absolute path here would point the control leg at
# this checkout's binary instead of the reference tree's.
case "${cal}" in
  *"--engine /"*) fail "case 2: --engine is an absolute path; it must be relative to the reference workspace" ;;
  *) ok "case 2: --engine is relative to the reference workspace" ;;
esac
[ -s "${OUT}" ] && ok "case 2: the calibration file was written" || fail "case 2: no calibration file"

PRIVATE_DIR="$(sed -n 's/^private=//p' "${WORK}/private.log")"
[ -n "${PRIVATE_DIR}" ] \
  && ok "case 2: benchd received MLXFAST_PRIVATE_DIR" \
  || fail "case 2: benchd did not receive MLXFAST_PRIVATE_DIR"
grep -qx 'mode=700' "${WORK}/private.log" \
  && ok "case 2: the private directory had mode 0700" \
  || fail "case 2: the private directory did not have mode 0700"
EXPECTED_PRIVATE_FILES="$(printf '%s\n' "${LIVE_GOLDENS[@]/%/.golden.json}" \
  public-longcopy-gate-english-496.golden.json | sort | sed 's/^/file=/')"
ACTUAL_PRIVATE_FILES="$(grep '^file=' "${WORK}/private.log" || true)"
[ "${ACTUAL_PRIVATE_FILES}" = "${EXPECTED_PRIVATE_FILES}" ] \
  && ok "case 2: the private directory held exactly the passed goldens" \
  || fail "case 2: wrong private files: ${ACTUAL_PRIVATE_FILES}"
private_args_ok=1
for name in "${LIVE_GOLDENS[@]}"; do
  [[ "${cal}" == *"--golden ${PRIVATE_DIR}/${name}.golden.json --prompt ${name}"* ]] \
    || private_args_ok=0
done
[[ "${cal}" == *"--warmup-golden ${PRIVATE_DIR}/public-longcopy-gate-english-496.golden.json"* ]] \
  || private_args_ok=0
[[ "${cal}" != *"--golden ${GOLDEN_DIR}/"* ]] || private_args_ok=0
[ "${private_args_ok}" -eq 1 ] \
  && ok "case 2: benchd received only private golden paths" \
  || fail "case 2: benchd did not receive only private golden paths"
[ ! -e "${PRIVATE_DIR}" ] \
  && ok "case 2: the private directory was removed after success" \
  || fail "case 2: the private directory remained after success"

# The EXIT trap also removes the private directory after a benchd refusal.
: > "${WORK}/private.log"
STUB_BENCHD_EXIT=9 drive "${WORK}/private-fail.out" "${BOX}" "${WORK}/private-fail.json"
[ $? -eq 4 ] || fail "case 2b: a benchd refusal must exit 4"
FAILED_PRIVATE_DIR="$(sed -n 's/^private=//p' "${WORK}/private.log")"
[ -n "${FAILED_PRIVATE_DIR}" ] && [ ! -e "${FAILED_PRIVATE_DIR}" ] \
  && ok "case 2b: the private directory was removed after failure" \
  || fail "case 2b: the private directory remained after failure"

# An old fixture keeps the original paths and does not set the private variable.
OLD_FIXTURE="${WORK}/fixture-old.json"
jq 'del(.warmup_golden, .require_private_dir)' "${FIXTURE}" > "${OLD_FIXTURE}"
: > "${WORK}/benchd.log"; : > "${WORK}/private.log"
FIXTURE_OVERRIDE="${OLD_FIXTURE}" drive "${WORK}/old.out" "${BOX}" "${WORK}/old.json"
old_cal="$(grep '^calibrate-baseline ' "${WORK}/benchd.log" || true)"
if grep -q -- '--warmup-golden' "${WORK}/benchd.log"; then
  fail "case 2c: an old fixture passed --warmup-golden"
elif grep -q '^private=.' "${WORK}/private.log"; then
  fail "case 2c: an old fixture set MLXFAST_PRIVATE_DIR"
elif [[ "${old_cal}" != *"--golden ${GOLDEN_DIR}/pool-9.golden.json"* ]]; then
  fail "case 2c: an old fixture did not keep the staged golden paths"
else
  ok "case 2c: an old fixture keeps the old interface"
fi
# APPLIES NOTHING ELSE: the benchmark inputs are byte-identical after the run.
# Snapshotted rather than compared against git, so the suite is honest on a
# working tree that legitimately carries uncommitted changes.
if [ "$(inputs_digest)" = "${INPUTS_BEFORE}" ]; then
  ok "case 2: the run edited no golden, fixture or manifest"
else
  fail "case 2: the run modified tracked benchmark inputs"
fi

# --- case 3: the reference tree is at the wrong commit ----------------------
: > "${WORK}/serve.log"; : > "${WORK}/benchd.log"
WRONG_FIXTURE="${WORK}/wrong-fixture.json"
jq '.baseline_reference_commit = "0000000000000000000000000000000000000000"' "${FIXTURE}" > "${WRONG_FIXTURE}"
env PATH="${BIN}:${PATH}" CALIBRATE_BOX_SETSID=1 \
    MLXFAST_BASELINE_WORKSPACE="${WS}" MLXFAST_TARGET_SNAPSHOT_DIR="${WEIGHTS}" \
    BENCHD_BIN_DIR="${BENCHD_DIR}" \
    STUB_SERVE_LOG="${WORK}/serve.log" STUB_BENCHD_LOG="${WORK}/benchd.log" \
    BENCHD="${BIN}/benchd" \
    "${DRIVER}" "${BOX}" "${WORK}/out3.json" --fixture "${WRONG_FIXTURE}" --lock "${LOCK}" \
    > "${WORK}/commit.out" 2>&1
[ $? -eq 2 ] || fail "case 3: a reference tree at the wrong commit must refuse with exit 2"
grep -q 'REFUSE reference-commit-mismatch' "${WORK}/commit.out" \
  && ok "case 3: a wrong reference commit refuses by name" \
  || fail "case 3: the refusal did not name reference-commit-mismatch"
[ -s "${WORK}/serve.log" ] && fail "case 3: it booted serve-up before refusing"

# --- case 4: no reference workspace -----------------------------------------
env PATH="${BIN}:${PATH}" CALIBRATE_BOX_SETSID=1 \
    MLXFAST_TARGET_SNAPSHOT_DIR="${WEIGHTS}" \
    BENCHD_BIN_DIR="${BENCHD_DIR}" \
    STUB_SERVE_LOG="${WORK}/serve.log" STUB_BENCHD_LOG="${WORK}/benchd.log" \
    BENCHD="${BIN}/benchd" \
    "${DRIVER}" "${BOX}" "${WORK}/out4.json" --fixture "${FIXTURE}" --lock "${LOCK}" \
    > "${WORK}/nows.out" 2>&1
[ $? -eq 2 ] || fail "case 4: an unset MLXFAST_BASELINE_WORKSPACE must refuse with exit 2"
grep -q 'REFUSE missing-baseline-workspace' "${WORK}/nows.out" \
  && ok "case 4: an unset reference workspace refuses by name" \
  || fail "case 4: the refusal did not name missing-baseline-workspace"

# --- case 5: too few passes -------------------------------------------------
drive "${WORK}/passes.out" "${BOX}" "${WORK}/out5.json" --passes 1 --dry-run
[ $? -eq 2 ] || fail "case 5: --passes 1 must refuse with exit 2"
grep -q 'undefined below two passes' "${WORK}/passes.out" \
  && ok "case 5: one pass is not a calibration" \
  || fail "case 5: the refusal did not name the undefined CV"

# --- case 6: the box argument disagrees with the runner ---------------------
env PATH="${BIN}:${PATH}" CALIBRATE_BOX_SETSID=1 \
    MLXFAST_BASELINE_WORKSPACE="${WS}" MLXFAST_TARGET_SNAPSHOT_DIR="${WEIGHTS}" \
    BENCHD_BIN_DIR="${BENCHD_DIR}" \
    STUB_SERVE_LOG="${WORK}/serve.log" STUB_BENCHD_LOG="${WORK}/benchd.log" \
    BENCHD="${BIN}/benchd" RUNNER_NAME="another-box" \
    "${DRIVER}" "${BOX}" "${WORK}/out6.json" --fixture "${FIXTURE}" --lock "${LOCK}" --dry-run \
    > "${WORK}/box.out" 2>&1
[ $? -eq 2 ] || fail "case 6: a box that disagrees with RUNNER_NAME must refuse with exit 2"
grep -q 'REFUSE box-mismatch' "${WORK}/box.out" \
  && ok "case 6: a band cannot be attributed to another box" \
  || fail "case 6: the refusal did not name box-mismatch"

# --- case 7: a preset serve spec --------------------------------------------
env PATH="${BIN}:${PATH}" CALIBRATE_BOX_SETSID=1 \
    MLXFAST_BASELINE_WORKSPACE="${WS}" MLXFAST_TARGET_SNAPSHOT_DIR="${WEIGHTS}" \
    BENCHD_BIN_DIR="${BENCHD_DIR}" \
    STUB_SERVE_LOG="${WORK}/serve.log" STUB_BENCHD_LOG="${WORK}/benchd.log" \
    BENCHD="${BIN}/benchd" SERVE_UP_SPECULATIVE=1 \
    "${DRIVER}" "${BOX}" "${WORK}/out7.json" --fixture "${FIXTURE}" --lock "${LOCK}" --dry-run \
    > "${WORK}/spec.out" 2>&1
[ $? -eq 2 ] || fail "case 7: a preset SERVE_UP_SPECULATIVE must refuse with exit 2"
grep -q 'REFUSE preset-serve-spec' "${WORK}/spec.out" \
  && ok "case 7: the control leg's serve spec comes from nowhere but the driver" \
  || fail "case 7: the refusal did not name preset-serve-spec"

# --- case 8: an inherited resident socket -----------------------------------
env PATH="${BIN}:${PATH}" CALIBRATE_BOX_SETSID=1 \
    MLXFAST_BASELINE_WORKSPACE="${WS}" MLXFAST_TARGET_SNAPSHOT_DIR="${WEIGHTS}" \
    BENCHD_BIN_DIR="${BENCHD_DIR}" \
    STUB_SERVE_LOG="${WORK}/serve.log" STUB_BENCHD_LOG="${WORK}/benchd.log" \
    BENCHD="${BIN}/benchd" DS4_RESIDENT_SOCKET=/tmp/somebody-elses.sock \
    "${DRIVER}" "${BOX}" "${WORK}/out8.json" --fixture "${FIXTURE}" --lock "${LOCK}" --dry-run \
    > "${WORK}/socket.out" 2>&1
[ $? -eq 2 ] || fail "case 8: an inherited DS4_RESIDENT_SOCKET must refuse with exit 2"
grep -q 'REFUSE inherited-resident-socket' "${WORK}/socket.out" \
  && ok "case 8: an inherited resident socket refuses by name" \
  || fail "case 8: the refusal did not name inherited-resident-socket"

# --- case 9: the reference tree has no staged adapter -----------------------
WS_NOADAPTER="${WORK}/reference-no-adapter"
cp -R "${WS}" "${WS_NOADAPTER}"
rm -f "${WS_NOADAPTER}/.build/release/mlxfast-runtime-worker"
env PATH="${BIN}:${PATH}" CALIBRATE_BOX_SETSID=1 \
    MLXFAST_BASELINE_WORKSPACE="${WS_NOADAPTER}" MLXFAST_TARGET_SNAPSHOT_DIR="${WEIGHTS}" \
    BENCHD_BIN_DIR="${BENCHD_DIR}" \
    STUB_SERVE_LOG="${WORK}/serve.log" STUB_BENCHD_LOG="${WORK}/benchd.log" \
    BENCHD="${BIN}/benchd" \
    "${DRIVER}" "${BOX}" "${WORK}/out9.json" --fixture "${FIXTURE}" --lock "${LOCK}" --dry-run \
    > "${WORK}/adapter.out" 2>&1
[ $? -eq 2 ] || fail "case 9: a reference tree with no staged adapter must refuse with exit 2"
grep -q 'REFUSE baseline-workspace-not-built' "${WORK}/adapter.out" \
  && ok "case 9: a reference tree with no staged adapter refuses by name" \
  || fail "case 9: the refusal did not name baseline-workspace-not-built"

# --- case 10: an unattributable benchd --------------------------------------
BAD_BENCHD_DIR="${WORK}/benchd-bin-bad"
mkdir -p "${BAD_BENCHD_DIR}"
printf '{"sha256":"aa","bytes":1}\n' > "${BAD_BENCHD_DIR}/benchd.manifest.json"
env PATH="${BIN}:${PATH}" CALIBRATE_BOX_SETSID=1 \
    MLXFAST_BASELINE_WORKSPACE="${WS}" MLXFAST_TARGET_SNAPSHOT_DIR="${WEIGHTS}" \
    MLXFAST_NEMOTRON35_GOLDEN_DIR="${GOLDEN_DIR}" \
    BENCHD_BIN_DIR="${BAD_BENCHD_DIR}" \
    STUB_SERVE_LOG="${WORK}/serve.log" STUB_BENCHD_LOG="${WORK}/benchd.log" \
    BENCHD="${BIN}/benchd" \
    "${DRIVER}" "${BOX}" "${WORK}/out10.json" --fixture "${FIXTURE}" --lock "${LOCK}" --dry-run \
    > "${WORK}/benchdcommit.out" 2>&1
[ $? -eq 2 ] || fail "case 10: a manifest with no source_commit must refuse with exit 2"
grep -q 'REFUSE unattributable-benchd' "${WORK}/benchdcommit.out" \
  && ok "case 10: a band cannot be captured by an unattributable benchd" \
  || fail "case 10: the refusal did not name unattributable-benchd"

# --- case 11: the staged golden pool is required ----------------------------
# There is no in-repo copy any more, so an unset variable has nothing to fall
# back on. It must be refused by name, before the lock and before any serve.
GOLDEN_DIR_OVERRIDE="" \
drive "${WORK}/case11.out" "${BOX}" "${WORK}/out11.json" --dry-run
[ $? -ne 0 ] || fail "case 11: the driver ran with no staged golden pool"
grep -q "MLXFAST_NEMOTRON35_GOLDEN_DIR is unset" "${WORK}/case11.out" \
  && ok "case 11: an unstaged golden pool refuses by name" \
  || fail "case 11: the refusal did not name MLXFAST_NEMOTRON35_GOLDEN_DIR"

if [ "${failures}" -eq 0 ]; then
  echo "PASS: test-calibrate-box.sh"
  exit 0
fi
echo "FAIL: ${failures} failure(s)" >&2
exit 1

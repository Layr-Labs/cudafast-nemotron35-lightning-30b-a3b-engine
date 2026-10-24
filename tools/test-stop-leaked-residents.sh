#!/usr/bin/env bash
#
# test-stop-leaked-residents.sh -- prove that cleanup stops only exact resident
# container names and refuses to act without the GPU lock.
#
# This test uses stub Docker and NVIDIA commands. It needs no GPU or network.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STOPPER="${STOPPER_UNDER_TEST:-${REPO_ROOT}/tools/stop-leaked-residents.sh}"
WORK="$(mktemp -d)" || exit 1
trap 'rm -rf "${WORK}"' EXIT

STUB_BIN="${WORK}/bin"
DOCKER_NAMES_FILE="${WORK}/docker-names"
DOCKER_LOG="${WORK}/docker.log"
GPU_FILE="${WORK}/gpu"
GPU_LOG="${WORK}/gpu.log"
LOCK="${WORK}/gpu.lock"
STOP_MARK="${WORK}/stopped"
PS_COUNT_FILE="${WORK}/ps-count"
mkdir -p "${STUB_BIN}"
: > "${LOCK}"

cat > "${STUB_BIN}/docker" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${DOCKER_LOG}"
case "$1" in
  ps)
    if [[ -e "${STOP_MARK}" && "${AUTO_REMOVE_AFTER_PS}" -ge 0 ]]; then
      count="$(cat "${PS_COUNT_FILE}")"
      count=$((count + 1))
      printf '%s\n' "${count}" > "${PS_COUNT_FILE}"
      if [[ "${count}" -gt "${AUTO_REMOVE_AFTER_PS}" ]]; then
        grep -Fxv "${AUTO_REMOVE_NAME}" "${DOCKER_NAMES_FILE}" || true
        exit 0
      fi
    fi
    cat "${DOCKER_NAMES_FILE}"
    ;;
  stop)
    : > "${STOP_MARK}"
    ;;
  rm)
    name="$3"
    case "${DOCKER_RM_MODE}" in
      already-in-progress)
        echo "Error response from daemon: removal of container ${name} is already in progress" >&2
        exit 1
        ;;
      keep)
        exit 0
        ;;
    esac
    grep -Fxv "${name}" "${DOCKER_NAMES_FILE}" > "${DOCKER_NAMES_FILE}.new" || true
    mv "${DOCKER_NAMES_FILE}.new" "${DOCKER_NAMES_FILE}"
    ;;
  *)
    echo "unexpected docker call: $*" >&2
    exit 1
    ;;
esac
EOF
cat > "${STUB_BIN}/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${GPU_LOG}"
[[ "$*" == "--query-compute-apps=pid,process_name --format=csv,noheader" ]] || exit 1
if [[ "${NVIDIA_SMI_HANG:-}" == "1" ]]; then
  while :; do :; done
fi
cat "${GPU_FILE}"
EOF
chmod +x "${STUB_BIN}/docker" "${STUB_BIN}/nvidia-smi"
export PATH="${STUB_BIN}:${PATH}"
[[ "$(command -v docker)" == "${STUB_BIN}/docker" ]] || exit 1
[[ "$(command -v nvidia-smi)" == "${STUB_BIN}/nvidia-smi" ]] || exit 1

failures=0
fail() {
  echo "FAIL: $*" >&2
  failures=$((failures + 1))
}

reset_case() {
  : > "${DOCKER_NAMES_FILE}"
  : > "${DOCKER_LOG}"
  : > "${GPU_FILE}"
  : > "${GPU_LOG}"
  : > "${PS_COUNT_FILE}"
  rm -f "${STOP_MARK}"
  AUTO_REMOVE_AFTER_PS=-1
  AUTO_REMOVE_NAME=""
  DOCKER_RM_MODE=normal
  TEST_GPU_WAIT=2
  TEST_LOCK_WAIT=0
}

run_case() {
  local prefix="$1" mode="$2"
  OUT="$(env \
    PATH="${STUB_BIN}:${PATH}" \
    DOCKER_NAMES_FILE="${DOCKER_NAMES_FILE}" \
    DOCKER_LOG="${DOCKER_LOG}" \
    GPU_FILE="${GPU_FILE}" \
    GPU_LOG="${GPU_LOG}" \
    STOP_MARK="${STOP_MARK}" \
    PS_COUNT_FILE="${PS_COUNT_FILE}" \
    AUTO_REMOVE_AFTER_PS="${AUTO_REMOVE_AFTER_PS}" \
    AUTO_REMOVE_NAME="${AUTO_REMOVE_NAME}" \
    DOCKER_RM_MODE="${DOCKER_RM_MODE}" \
    SERVE_UP_CONTAINER_PREFIX="${prefix}" \
    "${STOPPER}" --mode "${mode}" --lock "${LOCK}" \
      --gpu-wait "${TEST_GPU_WAIT}" --lock-wait "${TEST_LOCK_WAIT}" 2>&1)"
  RC=$?
}

run_without_prefix() {
  OUT="$(env -u SERVE_UP_CONTAINER_PREFIX \
    PATH="${STUB_BIN}:${PATH}" \
    DOCKER_NAMES_FILE="${DOCKER_NAMES_FILE}" \
    DOCKER_LOG="${DOCKER_LOG}" \
    GPU_FILE="${GPU_FILE}" \
    STOP_MARK="${STOP_MARK}" \
    PS_COUNT_FILE="${PS_COUNT_FILE}" \
    AUTO_REMOVE_AFTER_PS="${AUTO_REMOVE_AFTER_PS}" \
    AUTO_REMOVE_NAME="${AUTO_REMOVE_NAME}" \
    DOCKER_RM_MODE="${DOCKER_RM_MODE}" \
    "${STOPPER}" --mode preflight --lock "${LOCK}" --gpu-wait 2 --lock-wait 0 2>&1)"
  RC=$?
}

run_hung_query() {
  OUT="$(env \
    PATH="${STUB_BIN}:${PATH}" \
    DOCKER_NAMES_FILE="${DOCKER_NAMES_FILE}" \
    DOCKER_LOG="${DOCKER_LOG}" \
    GPU_FILE="${GPU_FILE}" \
    GPU_LOG="${GPU_LOG}" \
    NVIDIA_SMI_HANG=1 \
    STOP_MARK="${STOP_MARK}" \
    PS_COUNT_FILE="${PS_COUNT_FILE}" \
    AUTO_REMOVE_AFTER_PS="${AUTO_REMOVE_AFTER_PS}" \
    AUTO_REMOVE_NAME="${AUTO_REMOVE_NAME}" \
    DOCKER_RM_MODE="${DOCKER_RM_MODE}" \
    SERVE_UP_CONTAINER_PREFIX="${PREFIX}" \
    timeout --kill-after=1 --signal=TERM 4 \
      "${STOPPER}" --mode preflight --lock "${LOCK}" --gpu-wait 2 --lock-wait 0 2>&1)"
  RC=$?
}

actions() {
  grep -E '^(stop -t 10|rm -f) ' "${DOCKER_LOG}" || true
}

expect_rc() {
  local name="$1" want="$2"
  [[ "${RC}" -eq "${want}" ]] || fail "${name}: exit ${RC}, expected ${want}; ${OUT}"
}

expect_actions() {
  local name="$1" want="$2" got
  got="$(actions)"
  [[ "${got}" == "${want}" ]] || fail "${name}: Docker actions were '${got}', expected '${want}'"
}

PREFIX="nemotron35prod-serve-up-"
MATCH="${PREFIX}1629628-20261023T233337Z"

# 1. An unset prefix must not select any container.
reset_case
run_without_prefix
expect_rc "unset prefix" 2
expect_actions "unset prefix" ""

# 2. Invalid prefixes must not select any container.
for bad_prefix in serve-up 'a*-'; do
  reset_case
  run_case "${bad_prefix}" preflight
  expect_rc "bad prefix ${bad_prefix}" 2
  expect_actions "bad prefix ${bad_prefix}" ""
done

# 3. An idle box needs no cleanup.
reset_case
run_case "${PREFIX}" preflight
expect_rc "idle box" 0
expect_actions "idle box" ""

# 4. Preflight stops only the full exact match.
reset_case
printf '%s\n' \
  "${MATCH}" \
  'nemotron35-serve-up-12-20261023T000000Z' \
  'other-app' > "${DOCKER_NAMES_FILE}"
run_case "${PREFIX}" preflight
expect_rc "exact preflight match" 0
expect_actions "exact preflight match" "stop -t 10 ${MATCH}
rm -f ${MATCH}"
[[ "${OUT}" == *"::warning::"*"left by an earlier job"* ]] \
  || fail "exact preflight match: warning text is missing; ${OUT}"

# 5. Another GPU lock holder prevents all container actions.
reset_case
printf '%s\n' "${MATCH}" > "${DOCKER_NAMES_FILE}"
READY="${WORK}/lock-ready"
(
  exec 8< "${LOCK}"
  flock -n 8
  : > "${READY}"
  sleep 5
) &
holder=$!
for _ in $(seq 1 100); do
  [[ -e "${READY}" ]] && break
  sleep 0.01
done
run_case "${PREFIX}" preflight
kill "${holder}" 2>/dev/null || true
wait "${holder}" 2>/dev/null || true
expect_rc "held lock" 1
expect_actions "held lock" ""
[[ "${OUT}" == *"${LOCK}"* && "${OUT}" == *"another GPU window holds it"* ]] \
  || fail "held lock: error does not name the lock; ${OUT}"

# 6. A process that stays on the GPU fails after container cleanup.
reset_case
printf '%s\n' "${MATCH}" > "${DOCKER_NAMES_FILE}"
printf '%s\n' '4242, python3' > "${GPU_FILE}"
run_case "${PREFIX}" preflight
expect_rc "busy GPU" 1
expect_actions "busy GPU" "stop -t 10 ${MATCH}
rm -f ${MATCH}"
[[ "${OUT}" == *"4242, python3"* ]] || fail "busy GPU: error does not list the process; ${OUT}"

# 7. Cleanup uses the same exact stop and remove calls.
reset_case
printf '%s\n' \
  "${MATCH}" \
  'nemotron35-serve-up-12-20261023T000000Z' \
  'other-app' > "${DOCKER_NAMES_FILE}"
run_case "${PREFIX}" cleanup
expect_rc "cleanup match" 0
expect_actions "cleanup match" "stop -t 10 ${MATCH}
rm -f ${MATCH}"
[[ "${OUT}" != *"::warning::"* ]] || fail "cleanup match: cleanup used a warning; ${OUT}"

# 8. A prefix-only name is not a resident container name.
reset_case
printf '%s\n' "${PREFIX}1-2026x" > "${DOCKER_NAMES_FILE}"
run_case "${PREFIX}" preflight
expect_rc "prefix-only name" 0
expect_actions "prefix-only name" ""

# 9. A hung GPU query must not exceed the GPU wait.
reset_case
run_hung_query
expect_rc "hung GPU query" 1
expect_actions "hung GPU query" ""
[[ "${OUT}" == *"::error::nvidia-smi query timed out"* ]] \
  || fail "hung GPU query: timeout error is missing; ${OUT}"

# A. Docker can report that automatic removal is still in progress.
reset_case
printf '%s\n' "${MATCH}" > "${DOCKER_NAMES_FILE}"
DOCKER_RM_MODE=already-in-progress
AUTO_REMOVE_AFTER_PS=2
AUTO_REMOVE_NAME="${MATCH}"
run_case "${PREFIX}" preflight
expect_rc "delayed automatic removal" 0
expect_actions "delayed automatic removal" "stop -t 10 ${MATCH}
rm -f ${MATCH}"

# B. A resident that stays past the removal deadline is an error.
reset_case
printf '%s\n' "${MATCH}" > "${DOCKER_NAMES_FILE}"
DOCKER_RM_MODE=keep
run_case "${PREFIX}" preflight
expect_rc "resident remains" 1
[[ "${OUT}" == *"::error::resident containers remain after cleanup: ${MATCH}"* ]] \
  || fail "resident remains: remain error is missing; ${OUT}"

# C. The box graphics contexts do not make the GPU busy.
reset_case
printf '%s\n' '123, /usr/lib/xorg/Xorg' '456, /usr/bin/gnome-shell' > "${GPU_FILE}"
run_case "${PREFIX}" preflight
expect_rc "graphics contexts" 0

# D. A vLLM process is foreign even when Xorg is present.
reset_case
printf '%s\n' '123, /usr/lib/xorg/Xorg' '789, VLLM::EngineCore' > "${GPU_FILE}"
run_case "${PREFIX}" preflight
expect_rc "foreign GPU process" 1
[[ "${OUT}" == *"789, VLLM::EngineCore"* ]] \
  || fail "foreign GPU process: process is missing from the error; ${OUT}"

# E. A bounded lock wait queues until the current GPU window ends.
reset_case
printf '%s\n' "${MATCH}" > "${DOCKER_NAMES_FILE}"
READY="${WORK}/lock-ready-wait"
(
  exec 8< "${LOCK}"
  flock -n 8
  : > "${READY}"
  sleep 2
) &
holder=$!
for _ in $(seq 1 100); do
  [[ -e "${READY}" ]] && break
  sleep 0.01
done
TEST_LOCK_WAIT=5
run_case "${PREFIX}" preflight
wait "${holder}" 2>/dev/null || true
expect_rc "bounded lock wait" 0
expect_actions "bounded lock wait" "stop -t 10 ${MATCH}
rm -f ${MATCH}"

# F. A lock wait deadline stops no container.
reset_case
printf '%s\n' "${MATCH}" > "${DOCKER_NAMES_FILE}"
READY="${WORK}/lock-ready-timeout"
(
  exec 8< "${LOCK}"
  flock -n 8
  : > "${READY}"
  sleep 3
) &
holder=$!
for _ in $(seq 1 100); do
  [[ -e "${READY}" ]] && break
  sleep 0.01
done
TEST_LOCK_WAIT=1
run_case "${PREFIX}" preflight
kill "${holder}" 2>/dev/null || true
wait "${holder}" 2>/dev/null || true
expect_rc "lock wait deadline" 1
expect_actions "lock wait deadline" ""
[[ "${OUT}" == *"${LOCK}"* && "${OUT}" == *"another GPU window holds it"* ]] \
  || fail "lock wait deadline: lock error is missing; ${OUT}"

# G. Cleanup defers to a lock holder without stopping containers or checking the GPU.
reset_case
printf '%s\n' "${MATCH}" > "${DOCKER_NAMES_FILE}"
READY="${WORK}/lock-ready-cleanup-timeout"
(
  exec 8< "${LOCK}"
  flock -n 8
  : > "${READY}"
  sleep 3
) &
holder=$!
for _ in $(seq 1 100); do
  [[ -e "${READY}" ]] && break
  sleep 0.01
done
TEST_LOCK_WAIT=1
run_case "${PREFIX}" cleanup
kill "${holder}" 2>/dev/null || true
wait "${holder}" 2>/dev/null || true
expect_rc "cleanup lock wait deadline" 0
expect_actions "cleanup lock wait deadline" ""
[[ "${OUT}" == "::warning::the GPU lock is still held (a cancelled step may still be tearing down); not stopping containers now; the next job's preflight stops any leftover" ]] \
  || fail "cleanup lock wait deadline: warning is not exact; ${OUT}"
[[ ! -s "${GPU_LOG}" ]] || fail "cleanup lock wait deadline: nvidia-smi was called"

if [[ "${failures}" -ne 0 ]]; then
  echo "test-stop-leaked-residents.sh: ${failures} case(s) failed" >&2
  exit 1
fi
echo "test-stop-leaked-residents.sh: all cases passed"

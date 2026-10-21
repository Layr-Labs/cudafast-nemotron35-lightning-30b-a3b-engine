#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null && pwd -P)"
HELPER="${SCRIPT_DIR}/wait-for-memory.sh"

fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }
[[ -x "${HELPER}" ]] || fail "missing executable ${HELPER}"
command -v timeout >/dev/null || fail "timeout command is required"

# Use the helper's fixture and trusted overhead source. A 100 GiB synthetic
# total makes the current fixture's threshold an exact MemAvailable kB value.
read -r TOTAL_KB REQUIRED_BYTES THRESHOLD_KB <<<"$(python3 - "${ROOT_DIR}" <<'PY'
import json
import math
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
fixture = json.loads((root / "fixtures/nemotron35_lightning_30b_a3b_track.json").read_text())
util = float(fixture["serve_configuration"]["engine_env"]["RESIDENT_GPU_MEMORY_UTILIZATION"])
matches = re.findall(r"^HOST_OVERHEAD_GIB=([0-9]+)$",
                     (root / "tools/serve-up.sh").read_text(), re.MULTILINE)
if len(matches) != 1:
    raise SystemExit("could not read the HOST_OVERHEAD_GIB literal")
total_kb = 100 * 1024 * 1024
required = int(total_kb * 1024 * util) + int(matches[0]) * 1024**3 + 2 * 1024**3
print(total_kb, required, math.ceil(required / 1024))
PY
)"
[[ "${THRESHOLD_KB}" =~ ^[0-9]+$ ]] || fail "could not calculate the memory threshold"
(( THRESHOLD_KB > 0 )) || fail "calculated threshold is not positive"

TEST_DIR="$(mktemp -d "${TMPDIR:-/tmp}/wait-for-memory.XXXXXX")"
MEMINFO="${TEST_DIR}/meminfo"
ACTIVE_PID=""
cleanup() {
  if [[ -n "${ACTIVE_PID}" ]]; then
    kill "${ACTIVE_PID}" 2>/dev/null || true
    wait "${ACTIVE_PID}" 2>/dev/null || true
  fi
  rm -rf -- "${TEST_DIR}"
}
trap cleanup EXIT

write_meminfo() {
  local available_kb="$1"
  printf 'MemTotal: %s kB\nMemAvailable: %s kB\n' \
    "${TOTAL_KB}" "${available_kb}" >"${MEMINFO}.new"
  mv -f -- "${MEMINFO}.new" "${MEMINFO}"
}

# MemAvailable is reported in kB, so THRESHOLD_KB is the lowest representable
# value at or above the calculated byte threshold.
write_meminfo "${THRESHOLD_KB}"
if timeout 3s env SERVE_UP_MEMINFO="${MEMINFO}" \
    WAIT_FOR_MEMORY_TIMEOUT_S=0 WAIT_FOR_MEMORY_POLL_S=0.01 \
    "${HELPER}" >"${TEST_DIR}/ready.log" 2>&1; then
  grep -Fq 'ready after' "${TEST_DIR}/ready.log" || {
    cat "${TEST_DIR}/ready.log" >&2
    fail "threshold memory did not report readiness"
  }
  if grep -Fq 'waiting:' "${TEST_DIR}/ready.log"; then
    cat "${TEST_DIR}/ready.log" >&2
    fail "threshold memory waited instead of passing the initial check"
  fi
else
  status=$?
  cat "${TEST_DIR}/ready.log" >&2
  fail "immediate threshold check exited ${status}"
fi
printf 'PASS: immediate readiness at %s bytes\n' "${REQUIRED_BYTES}"

# Release memory only after two observed insufficient polls. mv replaces the
# fake meminfo atomically, avoiding partial reads by the helper.
write_meminfo "$((THRESHOLD_KB - 1))"
timeout 8s env SERVE_UP_MEMINFO="${MEMINFO}" \
  WAIT_FOR_MEMORY_TIMEOUT_S=4 WAIT_FOR_MEMORY_POLL_S=0.05 \
  "${HELPER}" >"${TEST_DIR}/release.log" 2>&1 &
ACTIVE_PID=$!
waiting_count=0
for ((i = 0; i < 500; i++)); do
  waiting_count="$(grep -F -c 'waiting:' "${TEST_DIR}/release.log" 2>/dev/null || true)"
  if [[ "${waiting_count}" =~ ^[0-9]+$ ]] && (( waiting_count >= 2 )); then break; fi
  if ! kill -0 "${ACTIVE_PID}" 2>/dev/null; then break; fi
  sleep 0.01
done
(( waiting_count >= 2 )) || {
  cat "${TEST_DIR}/release.log" >&2
  fail "did not observe two insufficient polls before releasing memory"
}
write_meminfo "${THRESHOLD_KB}"
if wait "${ACTIVE_PID}"; then status=0; else status=$?; fi
ACTIVE_PID=""
[[ "${status}" -eq 0 ]] || {
  cat "${TEST_DIR}/release.log" >&2
  fail "released memory did not succeed (exit ${status})"
}
grep -Fq 'ready after' "${TEST_DIR}/release.log" || {
  cat "${TEST_DIR}/release.log" >&2
  fail "released memory did not report readiness"
}
printf 'PASS: readiness after %s insufficient polls\n' "${waiting_count}"

# Keep memory below threshold for the whole bounded run; it must time out.
write_meminfo "$((THRESHOLD_KB - 1))"
timeout 3s env SERVE_UP_MEMINFO="${MEMINFO}" \
  WAIT_FOR_MEMORY_TIMEOUT_S=0.15 WAIT_FOR_MEMORY_POLL_S=0.02 \
  "${HELPER}" >"${TEST_DIR}/timeout.log" 2>&1 &
ACTIVE_PID=$!
if wait "${ACTIVE_PID}"; then status=0; else status=$?; fi
ACTIVE_PID=""
[[ "${status}" -eq 1 ]] || {
  cat "${TEST_DIR}/timeout.log" >&2
  fail "insufficient memory should time out with exit 1 (got ${status})"
}
grep -Fq 'timed out after' "${TEST_DIR}/timeout.log" || {
  cat "${TEST_DIR}/timeout.log" >&2
  fail "timeout did not report the bounded wait"
}
printf 'PASS: insufficient memory timed out with exit 1\n'

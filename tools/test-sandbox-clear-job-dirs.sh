#!/usr/bin/env bash
# test-sandbox-clear-job-dirs.sh -- tools/sandbox-clear-job-dirs.sh.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLEAR="${ROOT_DIR}/tools/sandbox-clear-job-dirs.sh"
WORK="$(mktemp -d)"
trap 'rm -rf -- "${WORK}"' EXIT
ME="$(id -un)"
failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok() { echo "ok: $*"; }

mkdir -p "${WORK}/bin"
cat > "${WORK}/bin/sudo" <<'EOF'
#!/usr/bin/env bash
[[ "$1 $2" == "-n -u" && "$4" == "--" ]] || exit 97
shift 4
[[ "$1" == kill ]] && exit 0
exec "$@"
EOF
cat > "${WORK}/bin/pgrep" <<'EOF'
#!/usr/bin/env bash
exit 1
EOF
chmod +x "${WORK}/bin/"*

run_clear() {
  env PATH="${WORK}/bin:${PATH}" SANDBOX_USER="${ME}" "${CLEAR}" "$@"
}

root="${WORK}/remove"
mkdir -p "${root}/41-1/adapter-home" "${root}/42-1/adapter-home"
echo old > "${root}/41-1/adapter-home/state"
echo current > "${root}/42-1/adapter-home/state"
out="$(run_clear "${root}" --except 42-1 2>&1)"; status=$?
[[ "${status}" -eq 0 && ! -e "${root}/41-1" && -f "${root}/42-1/adapter-home/state" ]] \
  && ok "removes old job directories and keeps the current job" \
  || fail "remove and keep: status ${status}, '${out}'"

root="${WORK}/symlink"
mkdir -p "${root}" "${WORK}/outside"
echo keep > "${WORK}/outside/state"
ln -s "${WORK}/outside" "${root}/41-1"
out="$(run_clear "${root}" 2>&1)"; status=$?
[[ "${status}" -ne 0 && -L "${root}/41-1" && -f "${WORK}/outside/state" ]] \
  && ok "refuses a job symlink without following it" \
  || fail "symlink refusal: status ${status}, '${out}'"

root="${WORK}/unsafe"
mkdir -p "${root}/_bad"
out="$(run_clear "${root}" 2>&1)"; status=$?
[[ "${status}" -ne 0 && -d "${root}/_bad" ]] \
  && ok "refuses an unsafe job name" \
  || fail "unsafe name refusal: status ${status}, '${out}'"

if [[ "${failures}" -eq 0 ]]; then
  echo "PASS: test-sandbox-clear-job-dirs.sh"
  exit 0
fi
echo "FAIL: ${failures} failure(s)" >&2
exit 1

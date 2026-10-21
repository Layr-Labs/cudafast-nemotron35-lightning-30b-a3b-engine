#!/usr/bin/env bash
# test-sandbox-wipe.sh -- tools/sandbox-wipe.sh (docs/sandbox-design.md 6a).
#
# Asserts, with sudo, ipcs and ipcrm stubbed (the sandbox user is the current
# user, the wiped directories are scratch ones):
#   1. every file, directory (even mode 000) and FIFO the sandbox user owns is
#      deleted, the wiped directories themselves are kept, and so is a file
#      another user owns; POSIX queues under the mqueue directory go too, and
#      an absent mqueue directory is skipped;
#   2. the sandbox user's SysV shm, semaphores and queues are removed through
#      sudo, another user's are not;
#   3. every sandbox process is killed first;
#   4. something the wipe cannot remove fails it, with a count and no names;
#   5. sudo to the sandbox user failing is a failure;
#   6. the directories come from the wipe list file (every listed one is
#      wiped, e.g. a /run/lock or /var/crash stand-in, plus the home); a
#      missing or empty list fails before anything runs.
#
# Hermetic: no root, no real IPC objects touched.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WIPE="${ROOT_DIR}/tools/sandbox-wipe.sh"
WORK="$(mktemp -d)"
trap 'chmod -R u+rwx "${WORK}" 2>/dev/null; rm -rf "${WORK}"' EXIT
ME="$(id -un)"

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }

mkdir -p "${WORK}/bin" "${WORK}/tmp" "${WORK}/shm" "${WORK}/home" "${WORK}/mqueue" "${WORK}/lock" "${WORK}/crash"
# The wipe list as setup-sandbox-user.sh writes it: one absolute path per line.
printf '%s\n' "${WORK}/tmp" "${WORK}/shm" "${WORK}/lock" "${WORK}/crash" "${WORK}/absent" "${WORK}/mqueue" > "${WORK}/wipe-dirs"
cat > "${WORK}/bin/sudo" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${STUB_LOG}"
[[ "${STUB_SUDO_FAIL:-0}" == 1 ]] && exit 1
[[ "$1 $2" == "-n -u" && "$4" == "--" ]] || exit 97
shift 4
[[ "$1" == kill ]] && exit 0
[[ "$1" == ipcrm ]] && exec "${STUB_BIN}/ipcrm" "${@:2}"
exec "$@"
EOF
# ipcs: a table from ${STUB_IPC}; ipcrm drops a line from it (unless stuck).
cat > "${WORK}/bin/ipcs" <<'EOF'
#!/usr/bin/env bash
echo "------ header --------"
echo "key        id         owner      perms"
grep -- "^$1 " "${STUB_IPC}" | cut -d' ' -f2-
EOF
cat > "${WORK}/bin/ipcrm" <<'EOF'
#!/usr/bin/env bash
echo "ipcrm $*" >> "${STUB_LOG}"
[[ -n "${STUB_IPC_STUCK:-}" ]] && exit 1
grep -v -- "^$1 0x[0-9a-f]* $2 " "${STUB_IPC}" > "${STUB_IPC}.new"; mv "${STUB_IPC}.new" "${STUB_IPC}"
EOF
chmod +x "${WORK}/bin/"*

seed() {
  cat > "${WORK}/ipc" <<EOF
-m 0x00000000 11 ${ME} 600
-m 0x00000001 12 someoneelse 600
-s 0x00000002 21 ${ME} 600
-q 0x00000003 31 ${ME} 600
EOF
  mkdir -p "${WORK}/tmp/deep/er" "${WORK}/shm/locked"
  echo x > "${WORK}/tmp/deep/er/f"; echo y > "${WORK}/tmp/top"; echo z > "${WORK}/home/.cache"
  echo s > "${WORK}/shm/locked/seg"; chmod 000 "${WORK}/shm/locked"
  mkfifo "${WORK}/tmp/fifo"
  echo q > "${WORK}/mqueue/nemosub-queue"
  mkdir -p "${WORK}/lock/persist"; echo l > "${WORK}/lock/persist/f"; echo c > "${WORK}/crash/core"
}
run_wipe() {
  env PATH="${WORK}/bin:${PATH}" SANDBOX_USER="${ME}" STUB_BIN="${WORK}/bin" \
    STUB_LOG="${WORK}/log" STUB_IPC="${WORK}/ipc" \
    SANDBOX_WIPE_LIST="${WORK}/wipe-dirs" SANDBOX_WIPE_HOME="${WORK}/home" "$@" "${WIPE}"
}

# --- case 1-3: a full wipe ----------------------------------------------------
seed; : > "${WORK}/log"
run_wipe > "${WORK}/out1" 2>&1; status=$?
[[ "${status}" -eq 0 ]] && ok "case 1: the wipe succeeds" || fail "case 1: exit ${status}: $(cat "${WORK}/out1")"
[[ -z "$(find "${WORK}/tmp" "${WORK}/shm" "${WORK}/home" "${WORK}/mqueue" "${WORK}/lock" "${WORK}/crash" -mindepth 1 -print -quit)" ]] \
  && ok "case 1: every owned file, dir (even mode 000), FIFO and POSIX queue is gone" \
  || fail "case 1: left: $(find "${WORK}/tmp" "${WORK}/shm" "${WORK}/home" "${WORK}/mqueue" "${WORK}/lock" "${WORK}/crash" -mindepth 1)"
[[ ! -e "${WORK}/lock/persist" && ! -e "${WORK}/crash/core" ]] \
  && ok "case 6: every directory in the wipe list is wiped (run/lock and var/crash stand-ins)" \
  || fail "case 6: a listed directory kept the sandbox user's file"
[[ -d "${WORK}/tmp" && -d "${WORK}/shm" && -d "${WORK}/home" ]] \
  && ok "case 1: the wiped directories themselves are kept" || fail "case 1: a wiped directory was removed"
grep -qxF -- "-n -u ${ME} -- kill -KILL -1" "${WORK}/log" \
  && [[ "$(grep -n -- 'kill -KILL -1' "${WORK}/log" | cut -d: -f1)" -lt "$(grep -n -- ' find ' "${WORK}/log" | head -1 | cut -d: -f1)" ]] \
  && ok "case 3: every sandbox process is killed before anything is deleted" \
  || fail "case 3: no kill before the delete: $(cat "${WORK}/log")"
grep -qxF -- "-n -u ${ME} -- ipcrm -m 11" "${WORK}/log" && grep -qxF -- "-n -u ${ME} -- ipcrm -s 21" "${WORK}/log" \
  && grep -qxF -- "-n -u ${ME} -- ipcrm -q 31" "${WORK}/log" \
  && ok "case 2: owned shm, semaphores and queues are removed as the sandbox user" \
  || fail "case 2: ipcrm calls: $(grep ipcrm "${WORK}/log")"
[[ "$(cat "${WORK}/ipc")" == "-m 0x00000001 12 someoneelse 600" ]] && ! grep -q "ipcrm -m 12" "${WORK}/log" \
  && ok "case 2: another user's segment is left alone" || fail "case 2: ipc left: $(cat "${WORK}/ipc")"

# A file the sandbox user does not own is never deleted (find -user decides).
seed; : > "${WORK}/log"
run_wipe SANDBOX_USER=root > /dev/null 2>&1
[[ -f "${WORK}/tmp/top" ]] && ok "case 1: a file another user owns is kept" || fail "case 1: a non-owned file was deleted"
chmod -R u+rwx "${WORK}/shm"; rm -rf "${WORK:?}/tmp/"* "${WORK:?}/shm/"* "${WORK:?}/home/".??* "${WORK:?}/lock/"* "${WORK:?}/crash/"*

# An absent mqueue directory (not mounted) is skipped, not a failure.
seed; : > "${WORK}/log"
sed "s#^${WORK}/mqueue\$#${WORK}/no-mqueue#" "${WORK}/wipe-dirs" > "${WORK}/wipe-dirs-nomq"
run_wipe SANDBOX_WIPE_LIST="${WORK}/wipe-dirs-nomq" > "${WORK}/out1b" 2>&1; status=$?
[[ "${status}" -eq 0 ]] && ok "case 1: an absent mqueue directory is skipped cleanly" \
  || fail "case 1: absent mqueue: exit ${status}: $(cat "${WORK}/out1b")"
chmod -R u+rwx "${WORK}/shm"; rm -rf "${WORK:?}/tmp/"* "${WORK:?}/shm/"* "${WORK:?}/home/".??* "${WORK:?}/mqueue/"* "${WORK:?}/lock/"* "${WORK:?}/crash/"*

# --- case 6: no list, or an empty one, is a failure before anything runs ------
for list in "${WORK}/no-such-list" "${WORK}/empty-list"; do
  : > "${WORK}/empty-list"; : > "${WORK}/log"
  out="$(run_wipe SANDBOX_WIPE_LIST="${list}" 2>&1)"; status=$?
  [[ "${status}" -ne 0 && ! -s "${WORK}/log" ]] && grep -q "wipe list" <<< "${out}" \
    && ok "case 6: ${list##*/} fails the wipe before sudo runs" || fail "case 6: ${list##*/}: status ${status}, '${out}'"
done

# --- case 4: what cannot be removed fails the wipe ---------------------------
seed; : > "${WORK}/log"
out="$(run_wipe STUB_IPC_STUCK=1 2>&1)"; status=$?
[[ "${status}" -ne 0 ]] && grep -q "3 ${ME}-owned SysV\|1 ${ME}-owned SysV" <<< "${out}" \
  && ok "case 4: SysV objects that survive fail the wipe" || fail "case 4: status ${status}, '${out}'"
chmod -R u+rwx "${WORK}/shm"
mkdir -p "${WORK}/tmp/keep"; echo secret-prompt > "${WORK}/tmp/keep/secret-prompt-name"
chmod 555 "${WORK}/tmp/keep"
# find cannot delete inside a dir it cannot write once chmod is a no-op: stub
# chmod away so the 555 directory keeps its file.
printf '#!/usr/bin/env bash\nexit 0\n' > "${WORK}/bin/chmod"; chmod +x "${WORK}/bin/chmod"
seed_ipc_clean() { : > "${WORK}/ipc"; }
seed_ipc_clean
out="$(run_wipe 2>&1)"; status=$?
rm -f "${WORK}/bin/chmod"
[[ "${status}" -ne 0 ]] && grep -q "path(s) remain under ${WORK}/tmp" <<< "${out}" \
  && ok "case 4: a file that survives fails the wipe" || fail "case 4: status ${status}, '${out}'"
grep -q "secret-prompt" <<< "${out}" && fail "case 4: a surviving file's name was printed" \
  || ok "case 4: only a count is printed, never a name"
chmod -R u+rwx "${WORK}/tmp"

# --- case 5: no sudo to the sandbox user --------------------------------------
out="$(run_wipe STUB_SUDO_FAIL=1 2>&1)"; status=$?
[[ "${status}" -ne 0 ]] && grep -q "cannot wipe" <<< "${out}" \
  && ok "case 5: a failing sudo fails the wipe" || fail "case 5: status ${status}, '${out}'"

if [[ "${failures}" -eq 0 ]]; then
  echo "PASS: test-sandbox-wipe.sh"
  exit 0
fi
echo "FAIL: ${failures} failure(s)" >&2
exit 1

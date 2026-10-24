#!/usr/bin/env bash
# test-sandbox-wipe.sh -- tools/sandbox-wipe.sh (docs/sandbox-design.md 6a).
#
# Asserts, with sudo, ipcs and ipcrm stubbed (the sandbox user is the current
# user, the wiped directories are scratch ones):
#   1. every file, directory (even mode 000) and FIFO the sandbox user owns is
#      deleted, the wiped directories themselves are kept, and so is a file
#      another user owns; POSIX queues under the mqueue directory go too, and
#      an absent mqueue directory is skipped;
#   2. SysV objects whose uid or cuid is the sandbox uid are removed through
#      sudo, another user's are not;
#   3. every sandbox process is killed first;
#   4. something the wipe cannot remove fails it, with a count and no names;
#   5. sudo to the sandbox user failing is a failure;
#   6. the directories come from the wipe list file (every listed one is
#      wiped, e.g. a /run/lock or /var/crash stand-in, plus the home); /dev/shm
#      and /dev/mqueue are added even when absent from the list; a missing or
#      empty list fails before anything runs;
#   7. an unreadable /proc/sysvipc table or failed ipcs fallback fails closed.
#   8. an unreadable, non-owned and non-writable directory is pruned; an owned
#      or writable unreadable directory still makes enumeration fail.
#   9. the passwd home must be root-owned and not sandbox-writable. It must
#      have no default ACL or user.* xattr;
#  10. a writable path that the sandbox user does not own fails the wipe;
#  11. the runner-owned job, adapter-home and sock directories are trusted
#      only with fixed metadata and empty state directories;
#  12. each wipe clears the persistent and user keyrings as the sandbox user.
#
# Hermetic: no root, no real IPC objects touched.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WIPE="${SANDBOX_WIPE_SCRIPT:-${ROOT_DIR}/tools/sandbox-wipe.sh}"
WORK="$(mktemp -d)"
trap 'chmod -R u+rwx "${WORK}" 2>/dev/null; rm -rf "${WORK}"' EXIT
ME="$(id -un)"
ME_UID="$(id -u)"

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }

mkdir -p "${WORK}/bin" "${WORK}/tmp" "${WORK}/shm" "${WORK}/home" "${WORK}/mqueue" "${WORK}/lock" "${WORK}/crash" "${WORK}/sysvipc"
# The wipe list as setup-sandbox-user.sh writes it: one absolute path per line.
printf '%s\n' "${WORK}/tmp" "${WORK}/lock" "${WORK}/crash" "${WORK}/absent" > "${WORK}/wipe-dirs"
cat > "${WORK}/bin/sudo" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${STUB_LOG}"
[[ "${STUB_SUDO_FAIL:-0}" == 1 ]] && exit 1
if [[ "$1" == -n && "$2" == ipcrm ]]; then shift 2; exec "${STUB_BIN}/ipcrm" "$@"; fi
[[ "$1 $2" == "-n -u" && "$4" == "--" ]] || exit 97
shift 4
[[ "$1" == kill ]] && exit 0
[[ "$1" == ipcrm && "${STUB_SANDBOX_IPCRM_FAIL:-0}" == 1 ]] && exit 1
[[ "$1" == ipcrm ]] && exec "${STUB_BIN}/ipcrm" "${@:2}"
if [[ "$1 $2" == "test -w" && "$3" == "${STUB_HOME}" ]]; then
  [[ "${STUB_HOME_WRITABLE:-0}" == 1 ]]
  exit
fi
exec "$@"
EOF
# ipcs: a table from ${STUB_IPC}; ipcrm drops a line from it (unless stuck).
cat > "${WORK}/bin/ipcs" <<'EOF'
#!/usr/bin/env bash
[[ "${STUB_IPCS_FAIL:-0}" == 1 ]] && exit 1
echo "------ header --------"
case "$1" in -m) name=shmid ;; -s) name=semid ;; -q) name=msqid ;; esac
echo "${name} perms cuid cgid uid gid"
grep -- "^$1 " "${STUB_IPC}" | awk '{print $3, 600, $5, 0, $4, 0}'
EOF
cat > "${WORK}/bin/ipcrm" <<'EOF'
#!/usr/bin/env bash
echo "ipcrm $*" >> "${STUB_LOG}"
[[ -n "${STUB_IPC_STUCK:-}" ]] && exit 1
grep -v -- "^$1 0x[0-9a-f]* $2 " "${STUB_IPC}" > "${STUB_IPC}.new"; mv "${STUB_IPC}.new" "${STUB_IPC}"
case "$1" in -m) table=shm; id_col=2 ;; -s) table=sem; id_col=2 ;; -q) table=msg; id_col=2 ;; esac
awk -v id="$2" -v col="${id_col}" 'NR == 1 || $col != id' "${STUB_SYSVIPC}/${table}" > "${STUB_SYSVIPC}/${table}.new"
mv "${STUB_SYSVIPC}/${table}.new" "${STUB_SYSVIPC}/${table}"
EOF
chmod +x "${WORK}/bin/"*
REAL_PYTHON="$(command -v python3)"
cat > "${WORK}/bin/python3" <<EOF
#!/usr/bin/env bash
if [[ "\$1 \${2:-}" == "- keyring-clear" ]]; then
  cat >/dev/null
  case "\${STUB_KEYRING_STATUS:-cleared}" in
    cleared|unsupported) echo "\${STUB_KEYRING_STATUS:-cleared}"; exit 0 ;;
    *) exit 1 ;;
  esac
fi
exec ${REAL_PYTHON@Q} "\$@"
EOF
chmod +x "${WORK}/bin/python3"
REAL_FIND="$(command -v find)"
REAL_STAT="$(command -v stat)"
cat > "${WORK}/bin/stat" <<EOF
#!/usr/bin/env bash
if [[ "\$1" == -Lc && "\$2" == %u && "\${4:-\$3}" == "${WORK}/home" ]]; then echo 0; exit 0; fi
exec ${REAL_STAT@Q} "\$@"
EOF
chmod +x "${WORK}/bin/stat"

seed() {
  cat > "${WORK}/ipc" <<EOF
-m 0x00000000 11 ${ME_UID} ${ME_UID}
-m 0x00000001 12 99998 99998
-m 0x00000004 13 99999 ${ME_UID}
-s 0x00000002 21 ${ME_UID} ${ME_UID}
-q 0x00000003 31 ${ME_UID} ${ME_UID}
EOF
  cat > "${WORK}/sysvipc/shm" <<EOF
key shmid perms size cpid lpid nattch uid gid cuid cgid atime dtime ctime rss swap
0 11 600 1 1 1 0 ${ME_UID} 1 ${ME_UID} 1 0 0 0 0 0
1 12 600 1 1 1 0 99998 1 99998 1 0 0 0 0 0
4 13 600 1 1 1 0 99999 1 ${ME_UID} 1 0 0 0 0 0
EOF
  cat > "${WORK}/sysvipc/sem" <<EOF
key semid perms nsems uid gid cuid cgid otime ctime
2 21 600 1 ${ME_UID} 1 ${ME_UID} 1 0 0
EOF
  cat > "${WORK}/sysvipc/msg" <<EOF
key msqid perms cbytes qnum lspid lrpid uid gid cuid cgid stime rtime ctime
3 31 600 0 0 1 1 ${ME_UID} 1 ${ME_UID} 1 0 0 0
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
    STUB_HOME="${WORK}/home" \
    STUB_SYSVIPC="${WORK}/sysvipc" SANDBOX_SYSVIPC_DIR="${WORK}/sysvipc" \
    SANDBOX_DEV_SHM="${WORK}/shm" SANDBOX_DEV_MQUEUE="${WORK}/mqueue" \
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
  && grep -qxF -- "-n -u ${ME} -- ipcrm -m 13" "${WORK}/log" \
  && grep -qxF -- "-n -u ${ME} -- ipcrm -q 31" "${WORK}/log" \
  && ok "case 2: uid- or cuid-matched shm, semaphores and queues are removed as the sandbox user" \
  || fail "case 2: ipcrm calls: $(grep ipcrm "${WORK}/log")"
[[ "$(cat "${WORK}/ipc")" == "-m 0x00000001 12 99998 99998" ]] && ! grep -q "ipcrm -m 12" "${WORK}/log" \
  && ok "case 2: another user's segment is left alone" || fail "case 2: ipc left: $(cat "${WORK}/ipc")"
if grep -qxF -- "-n -u ${ME} -- python3 - keyring-clear ${ME_UID}" "${WORK}/log" \
    && grep -q "keyrings cleared" "${WORK}/out1"; then
  ok "case 12: the wipe clears both keyrings as the sandbox user"
else
  fail "case 12: the keyring clear did not run through sandbox sudo"
fi

: > "${WORK}/log"
out="$(run_wipe STUB_KEYRING_STATUS=unsupported 2>&1)"; status=$?
if [[ "${status}" -eq 0 && "${out}" == *"keyrings unsupported"* ]]; then
  ok "case 12: EOPNOTSUPP or ENOSYS marks keyrings unsupported"
else
  fail "case 12: unsupported keyrings: status ${status}, '${out}'"
fi
out="$(run_wipe STUB_KEYRING_STATUS=failed 2>&1)"; status=$?
if [[ "${status}" -ne 0 && "${out}" == *"cannot clear sandbox keyrings"* ]]; then
  ok "case 12: a keyring clear error fails closed"
else
  fail "case 12: keyring failure: status ${status}, '${out}'"
fi

# A missing /proc/sysvipc uses ipcs -c. A pre-existing root sudo grant is the
# fallback when the creator cannot remove an object after its owner changes.
seed; : > "${WORK}/log"
run_wipe SANDBOX_SYSVIPC_DIR="${WORK}/no-sysvipc" STUB_SANDBOX_IPCRM_FAIL=1 > "${WORK}/out2b" 2>&1; status=$?
[[ "${status}" -eq 0 ]] && grep -qxF -- "-n ipcrm -m 13" "${WORK}/log" \
  && [[ "$(cat "${WORK}/ipc")" == "-m 0x00000001 12 99998 99998" ]] \
  && ok "case 2: ipcs -c and an existing root sudo grant remove uid- or cuid-matched objects" \
  || fail "case 2: ipcs/root fallback: status ${status}, ipc '$(cat "${WORK}/ipc")'"

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
[[ "${status}" -ne 0 ]] && grep -q "SysV IPC object(s) tied to uid ${ME_UID}" <<< "${out}" \
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

# --- case 9: the passwd home is fixed metadata, not writable state ----------
seed_ipc_clean
out="$(run_wipe STUB_HOME_WRITABLE=1 2>&1)"; status=$?
[[ "${status}" -ne 0 ]] && grep -q "sandbox home is writable" <<<"${out}" \
  && ok "case 9: a sandbox-writable passwd home fails the wipe" \
  || fail "case 9: writable home: status ${status}, '${out}'"

setfacl -d -m u::rwx,g::r-x,o::--- "${WORK}/home"
out="$(run_wipe 2>&1)"; status=$?
setfacl -k "${WORK}/home"
[[ "${status}" -ne 0 ]] && grep -q "sandbox home has a default ACL" <<<"${out}" \
  && ok "case 9: a default ACL on the passwd home fails the wipe" \
  || fail "case 9: home default ACL: status ${status}, '${out}'"

python3 - "${WORK}/home" <<'PY'
import os
import sys
os.setxattr(sys.argv[1], "user.test", b"state")
PY
out="$(run_wipe 2>&1)"; status=$?
python3 - "${WORK}/home" <<'PY'
import os
import sys
try:
    os.removexattr(sys.argv[1], "user.test")
except OSError:
    pass
PY
[[ "${status}" -ne 0 ]] && grep -q "sandbox home has user.\* xattrs" <<<"${out}" \
  && ok "case 9: a user.* xattr on the passwd home fails the wipe" \
  || fail "case 9: home xattr: status ${status}, '${out}'"

# --- case 10: writable paths owned by another user fail closed -------------
rm -rf "${WORK:?}/tmp/"* "${WORK:?}/shm/"* "${WORK:?}/mqueue/"* "${WORK:?}/lock/"* "${WORK:?}/crash/"*
echo secret-prompt > "${WORK}/tmp/foreign-writable-secret"
chmod 0666 "${WORK}/tmp/foreign-writable-secret"
out="$(run_wipe SANDBOX_USER=root 2>&1)"; status=$?
[[ "${status}" -ne 0 ]] && grep -q "foreign-owned writable path(s)" <<<"${out}" \
  && ok "case 10: a foreign-owned sandbox-writable file fails the wipe" \
  || fail "case 10: foreign writable file: status ${status}, '${out}'"
grep -q "foreign-writable-secret" <<<"${out}" \
  && fail "case 10: a writable file's name was printed" \
  || ok "case 10: the failure prints only a count"
rm -f "${WORK}/tmp/foreign-writable-secret"

# --- case 11: only the fixed runner-owned job directories are trusted -------
JOB_ROOT="${WORK}/jobs"
JOB="${JOB_ROOT}/42-1"
reset_job() {
  rm -rf -- "${JOB_ROOT}"
  # only the deepest directory (the job dir) needs mode 2750; its parent is test scratch
  # shellcheck disable=SC2174
  mkdir -p -m 2750 "${JOB}"
  mkdir -m 2770 "${JOB}/adapter-home" "${JOB}/sock"
}
run_job_wipe() {
  run_wipe SANDBOX_USER=nobody SANDBOX_ROOT="${JOB_ROOT}" SANDBOX_JOB=42-1 "$@"
}

reset_job
out="$(run_job_wipe 2>&1)"; status=$?
[[ "${status}" -eq 0 ]] \
  && ok "case 11: empty runner-owned adapter-home and sock directories are trusted" \
  || fail "case 11: empty trusted job directories: status ${status}, '${out}'"

reset_job; echo state > "${JOB}/adapter-home/state"
out="$(run_job_wipe 2>&1)"; status=$?
[[ "${status}" -ne 0 ]] \
  && ok "case 11: a file in adapter-home fails closed" \
  || fail "case 11: adapter-home file: status ${status}, '${out}'"

reset_job
if python3 - "${JOB}/adapter-home" <<'PY'
import os
import sys
os.setxattr(sys.argv[1], "user.test", b"state")
PY
then
  out="$(run_job_wipe 2>&1)"; status=$?
  [[ "${status}" -ne 0 ]] \
    && ok "case 11: a user.* xattr on adapter-home fails closed" \
    || fail "case 11: adapter-home xattr: status ${status}, '${out}'"
else
  ok "case 11: user.* xattr check skipped because the file system has no xattr support"
fi

reset_job; mkdir -m 2770 "${JOB}/other"
out="$(run_job_wipe 2>&1)"; status=$?
[[ "${status}" -ne 0 ]] \
  && ok "case 11: any other writable job directory fails closed" \
  || fail "case 11: other writable directory: status ${status}, '${out}'"

reset_job; mkdir -m 2770 "${JOB_ROOT}/41-1"
out="$(run_job_wipe 2>&1)"; status=$?
[[ "${status}" -ne 0 ]] \
  && ok "case 11: an old job directory under SANDBOX_ROOT fails closed" \
  || fail "case 11: old job directory: status ${status}, '${out}'"

# A wipe-list dir below another one (/tmp/.X11-unix below /tmp, root 1777 on
# the boxes) is checked as its own root, not counted as a hit under /tmp.
reset_job; mkdir -m 1777 "${WORK}/tmp/.X11-unix"
{ cat "${WORK}/wipe-dirs"; echo "${WORK}/tmp/.X11-unix"; } > "${WORK}/wipe-dirs-nested"
out="$(run_job_wipe SANDBOX_WIPE_LIST="${WORK}/wipe-dirs-nested" 2>&1)"; status=$?
[[ "${status}" -eq 0 ]] \
  && ok "case 11: a nested wipe-list directory is not a foreign-owned hit" \
  || fail "case 11: nested wipe-list directory: status ${status}, '${out}'"

mkdir -m 0777 "${WORK}/tmp/nested-wipe"
{ cat "${WORK}/wipe-dirs"; echo "${WORK}/tmp/nested-wipe"; } > "${WORK}/wipe-dirs-xattr"
if python3 - "${WORK}/tmp/nested-wipe" <<'PY'
import os
import sys
os.setxattr(sys.argv[1], "user.test", b"state")
PY
then
  out="$(run_job_wipe SANDBOX_WIPE_LIST="${WORK}/wipe-dirs-xattr" 2>&1)"; status=$?
  [[ "${status}" -ne 0 ]] && grep -q "1 wipe directories have user\.\* xattrs" <<<"${out}" \
    && ok "case 11: a user.* xattr on a nested wipe-list directory fails closed" \
    || fail "case 11: nested wipe-list xattr: status ${status}, '${out}'"
else
  ok "case 11: nested wipe-list xattr check skipped because the file system has no xattr support"
fi
rmdir "${WORK}/tmp/nested-wipe"

out="$(run_job_wipe 2>&1)"; status=$?
[[ "${status}" -ne 0 ]] && grep -q "foreign-owned writable path(s)" <<<"${out}" \
  && ok "case 11: the same directory off the wipe list fails closed" \
  || fail "case 11: unlisted nested directory: status ${status}, '${out}'"
rmdir "${WORK}/tmp/.X11-unix"

# Only X11 and ICE session sockets are trusted; a foreign FIFO still is not.
reset_job
python3 -c 'import socket,sys; s=socket.socket(socket.AF_UNIX); s.bind(sys.argv[1])' "${WORK}/tmp/session-socket"
chmod 0777 "${WORK}/tmp/session-socket"
out="$(run_job_wipe 2>&1)"; status=$?
[[ "${status}" -ne 0 ]] && grep -q "foreign-owned writable path(s)" <<<"${out}" \
  && ok "case 11: a foreign-owned writable socket fails closed" \
  || fail "case 11: foreign socket: status ${status}, '${out}'"
rm -f "${WORK}/tmp/session-socket"

mkdir -m 1777 "${WORK}/tmp/.ICE-unix"
{ cat "${WORK}/wipe-dirs"; echo "${WORK}/tmp/.ICE-unix"; } > "${WORK}/wipe-dirs-ice"
python3 -c 'import socket,sys; s=socket.socket(socket.AF_UNIX); s.bind(sys.argv[1])' "${WORK}/tmp/.ICE-unix/session-socket"
chmod 0777 "${WORK}/tmp/.ICE-unix/session-socket"
out="$(run_job_wipe SANDBOX_WIPE_LIST="${WORK}/wipe-dirs-ice" 2>&1)"; status=$?
[[ "${status}" -eq 0 ]] \
  && ok "case 11: a foreign-owned writable ICE socket is not a hit" \
  || fail "case 11: foreign ICE socket: status ${status}, '${out}'"
rm -f "${WORK}/tmp/.ICE-unix/session-socket"
rmdir "${WORK}/tmp/.ICE-unix"

ln -s "${WORK}/lock" "${WORK}/tmp/foreign-link"
out="$(run_job_wipe 2>&1)"; status=$?
[[ "${status}" -eq 0 ]] \
  && ok "case 11: a foreign-owned symlink to a writable directory is not a hit" \
  || fail "case 11: foreign symlink: status ${status}, '${out}'"
rm -f "${WORK}/tmp/foreign-link"
mkfifo -m 0666 "${WORK}/tmp/foreign-fifo"
out="$(run_job_wipe 2>&1)"; status=$?
[[ "${status}" -ne 0 ]] && grep -q "foreign-owned writable path(s)" <<<"${out}" \
  && ok "case 11: a foreign-owned writable FIFO fails closed" \
  || fail "case 11: foreign FIFO: status ${status}, '${out}'"
rm -f "${WORK}/tmp/foreign-fifo"
rm -rf -- "${JOB_ROOT}"

# --- case 5: no sudo to the sandbox user --------------------------------------
out="$(run_wipe STUB_SUDO_FAIL=1 2>&1)"; status=$?
[[ "${status}" -ne 0 ]] && grep -q "cannot wipe" <<< "${out}" \
  && ok "case 5: a failing sudo fails the wipe" || fail "case 5: status ${status}, '${out}'"

# --- case 7: enumeration errors fail closed -----------------------------------
seed; chmod 000 "${WORK}/sysvipc/sem"
out="$(run_wipe 2>&1)"; status=$?
chmod 600 "${WORK}/sysvipc/sem"
[[ "${status}" -ne 0 ]] && grep -q "cannot enumerate SysV IPC" <<< "${out}" \
  && ok "case 7: an unreadable /proc/sysvipc table fails closed" \
  || fail "case 7: unreadable table: status ${status}, '${out}'"

out="$(run_wipe SANDBOX_SYSVIPC_DIR="${WORK}/no-sysvipc" STUB_IPCS_FAIL=1 2>&1)"; status=$?
[[ "${status}" -ne 0 ]] && grep -q "cannot enumerate SysV IPC" <<< "${out}" \
  && ok "case 7: a failed ipcs -c fallback fails closed" \
  || fail "case 7: ipcs failure: status ${status}, '${out}'"

# --- case 8: prune only directories that cannot hold sandbox state -----------
cat > "${WORK}/bin/find" <<EOF
#!/usr/bin/env bash
if [[ -n "\${STUB_FIND_CASE:-}" && " \$* " == *" \${STUB_FIND_DIR:-} "* ]]; then
  case "\${STUB_FIND_CASE}" in
    foreign)
      [[ " \$* " == *" -readable "* && " \$* " == *" -writable "* \
         && " \$* " == *" -prune "* ]] && exit 0
      ;;
    owned|writable) ;;
  esac
  echo "find: permission denied" >&2
  exit 1
fi
exec ${REAL_FIND@Q} "\$@"
EOF
chmod +x "${WORK}/bin/find"
seed_ipc_clean
out="$(run_wipe STUB_FIND_CASE=foreign STUB_FIND_DIR="${WORK}/tmp" 2>&1)"; status=$?
[[ "${status}" -eq 0 ]] \
  && ok "case 8: an unreadable, non-owned and non-writable directory is pruned" \
  || fail "case 8: safe prune: status ${status}, '${out}'"
for access in owned writable; do
  out="$(run_wipe STUB_FIND_CASE="${access}" STUB_FIND_DIR="${WORK}/tmp" 2>&1)"; status=$?
  [[ "${status}" -ne 0 ]] && grep -q "cannot enumerate ${WORK}/tmp" <<<"${out}" \
    && ok "case 8: an unreadable ${access} directory fails closed" \
    || fail "case 8: unreadable ${access}: status ${status}, '${out}'"
done
rm -f "${WORK}/bin/find"

if [[ "${failures}" -eq 0 ]]; then
  echo "PASS: test-sandbox-wipe.sh"
  exit 0
fi
echo "FAIL: ${failures} failure(s)" >&2
exit 1

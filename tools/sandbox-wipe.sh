#!/usr/bin/env bash
#
# sandbox-wipe.sh -- remove everything the sandbox user left behind outside the
# per-job directory, so nothing survives from one job into the next
# (docs/sandbox-design.md section 5 item 6 and section 6a). TRUSTED: not in
# editablePaths. The workflow runs it at job start and in the cleanup step.
#
#   1. kills every sandbox process (nothing may recreate what is deleted);
#   2. verifies that the passwd home is root-owned and not sandbox-writable.
#      It also rejects a default ACL or user.* xattr on that directory;
#   3. as the sandbox user, deletes every file it owns under each directory in
#      the box's wipe list and below its passwd home (never crossing a mount;
#      races ignored), after giving itself access to its own directories. The list
#      is every directory on the box the sandbox user can write (/tmp,
#      /var/tmp, /dev/shm, /run/lock, /var/crash, /dev/mqueue, ...), recorded
#      once by tools/box-runner/setup-sandbox-user.sh in a root-owned file. A
#      POSIX message queue is a file under /dev/mqueue, so it goes the same way;
#      a listed directory that does not exist (mqueue not mounted) is skipped;
#   4. rejects any foreign-owned path the sandbox user can write below a wipe
#      directory or the current per-job directory. The runner-owned job,
#      adapter-home and sock directories are trusted only with fixed metadata;
#   5. removes every SysV object whose owner uid or creator uid is the sandbox
#      uid;
#   6. verifies nothing tied to the sandbox uid is left in any of those places,
#      and exits non-zero if something is. Only counts are printed: a name
#      could carry prompt text.
#
# A missing or empty wipe list is a failure: the box was never set up.
#
# Env: SANDBOX_USER (default nemosub) and SANDBOX_ETC (default
# /etc/nemotron35-sandbox), the names tools/box-runner/setup-sandbox-user.sh
# uses. SANDBOX_WIPE_LIST (default $SANDBOX_ETC/wipe-dirs) and
# SANDBOX_WIPE_HOME (default /var/lib/$SANDBOX_USER), SANDBOX_ROOT,
# SANDBOX_JOB, SANDBOX_SYSVIPC_DIR
# (default /proc/sysvipc), SANDBOX_DEV_SHM (default /dev/shm) and
# SANDBOX_DEV_MQUEUE (default /dev/mqueue) are for tests.
# No -e: every step's status is checked explicitly; the delete passes are
# best-effort (races) and the verification pass decides the exit status.
set -uo pipefail

SANDBOX_USER="${SANDBOX_USER:-nemosub}"
WIPE_LIST="${SANDBOX_WIPE_LIST:-${SANDBOX_ETC:-/etc/nemotron35-sandbox}/wipe-dirs}"
[[ -f "${WIPE_LIST}" && -r "${WIPE_LIST}" ]] \
  || { echo "sandbox-wipe.sh: no wipe list ${WIPE_LIST}; run tools/box-runner/setup-sandbox-user.sh as root" >&2; exit 1; }
# One absolute path per line; anything else in the list is ignored.
mapfile -t DIRS < <(grep '^/' "${WIPE_LIST}")
[[ "${#DIRS[@]}" -gt 0 ]] || { echo "sandbox-wipe.sh: the wipe list ${WIPE_LIST} names no directory" >&2; exit 1; }
add_dir() {
  local add="$1" current
  for current in "${DIRS[@]}"; do [[ "${current}" == "${add}" ]] && return; done
  DIRS+=("${add}")
}
add_dir "${SANDBOX_DEV_SHM:-/dev/shm}"
add_dir "${SANDBOX_DEV_MQUEUE:-/dev/mqueue}"
WIPE_HOME="${SANDBOX_WIPE_HOME:-/var/lib/${SANDBOX_USER}}"
JOB_DIR=""
if [[ -n "${SANDBOX_ROOT:-}" || -n "${SANDBOX_JOB:-}" ]]; then
  [[ "${SANDBOX_ROOT:-}" == /* && "${SANDBOX_JOB:-}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] \
    || { echo "sandbox-wipe.sh: SANDBOX_ROOT and SANDBOX_JOB do not name a safe per-job directory" >&2; exit 1; }
  JOB_DIR="${SANDBOX_ROOT%/}/${SANDBOX_JOB}"
fi
as_sandbox() { sudo -n -u "${SANDBOX_USER}" -- "$@"; }

as_sandbox true || { echo "sandbox-wipe.sh: 'sudo -n -u ${SANDBOX_USER}' fails; cannot wipe" >&2; exit 1; }
SANDBOX_UID="$(id -u "${SANDBOX_USER}" 2>/dev/null)" \
  || { echo "sandbox-wipe.sh: cannot resolve uid for ${SANDBOX_USER}" >&2; exit 1; }
as_sandbox kill -KILL -1 >/dev/null 2>&1

if [[ -n "${JOB_DIR}" ]] && ! python3 - "${SANDBOX_ROOT%/}" "${SANDBOX_JOB}" "$(id -u)" <<'PY'
import os
import stat
import sys

root, job, runner_uid = sys.argv[1], sys.argv[2], int(sys.argv[3])

def trusted_dir(path, empty=False):
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != runner_uid:
        raise SystemExit(1)
    if any(name.startswith("user.") for name in os.listxattr(path, follow_symlinks=False)):
        raise SystemExit(1)
    if empty and next(os.scandir(path), None) is not None:
        raise SystemExit(1)

root_st = os.lstat(root)
if not stat.S_ISDIR(root_st.st_mode):
    raise SystemExit(1)
entries = list(os.scandir(root))
if any(entry.name != job for entry in entries):
    raise SystemExit(1)
job_dir = os.path.join(root, job)
if os.path.lexists(job_dir):
    trusted_dir(job_dir)
    for name in ("adapter-home", "sock"):
        path = os.path.join(job_dir, name)
        if os.path.lexists(path):
            trusted_dir(path, empty=True)
PY
then
  echo "sandbox-wipe.sh: the per-job directory has unsafe metadata, state, or an old job entry" >&2
  exit 1
fi

[[ -d "${WIPE_HOME}" && ! -L "${WIPE_HOME}" ]] \
  || { echo "sandbox-wipe.sh: the sandbox home is missing or is a symlink" >&2; exit 1; }
[[ "$(stat -Lc %u -- "${WIPE_HOME}" 2>/dev/null)" == 0 ]] \
  || { echo "sandbox-wipe.sh: the sandbox home is not root-owned" >&2; exit 1; }
if as_sandbox test -w "${WIPE_HOME}"; then
  echo "sandbox-wipe.sh: the sandbox home is writable by ${SANDBOX_USER}" >&2
  exit 1
fi
HOME_ACL="$(getfacl -cp -- "${WIPE_HOME}" 2>/dev/null)" \
  || { echo "sandbox-wipe.sh: cannot read the sandbox home ACL" >&2; exit 1; }
if grep -q '^default:' <<<"${HOME_ACL}"; then
  echo "sandbox-wipe.sh: the sandbox home has a default ACL" >&2
  exit 1
fi
if ! python3 - "${WIPE_HOME}" <<'PY'
import os
import sys

names = os.listxattr(sys.argv[1], follow_symlinks=False)
raise SystemExit(any(name.startswith("user.") for name in names))
PY
then
  echo "sandbox-wipe.sh: the sandbox home has user.* xattrs" >&2
  exit 1
fi

# ipc_ids FLAG -- ids of SysV objects whose numeric uid or cuid is the sandbox
# uid. /proc gives numeric ids. Old kernels without /proc/sysvipc use ipcs -c.
ipc_ids() {
  local kind="$1" table id_name output
  case "${kind}" in
    -m) table=shm; id_name=shmid ;;
    -s) table=sem; id_name=semid ;;
    -q) table=msg; id_name=msqid ;;
    *) return 1 ;;
  esac
  if [[ -d "${SANDBOX_SYSVIPC_DIR:-/proc/sysvipc}" ]]; then
    table="${SANDBOX_SYSVIPC_DIR:-/proc/sysvipc}/${table}"
    [[ -r "${table}" ]] || return 1
    awk -v want="${SANDBOX_UID}" -v id_name="${id_name}" '
      NR == 1 {
        for (i = 1; i <= NF; i++) col[$i] = i
        if (!col[id_name] || !col["uid"] || !col["cuid"]) exit 2
        next
      }
      NF {
        id = $(col[id_name]); uid = $(col["uid"]); cuid = $(col["cuid"])
        if (id !~ /^[0-9]+$/ || uid !~ /^[0-9]+$/ || cuid !~ /^[0-9]+$/) exit 3
        if (uid == want || cuid == want) print id
      }
      END { if (NR == 0) exit 4 }
    ' "${table}"
    return
  fi
  output="$(LC_ALL=C ipcs "${kind}" -c 2>/dev/null)" || return 1
  awk -v want_uid="${SANDBOX_UID}" -v want_user="${SANDBOX_USER}" -v id_name="${id_name}" '
    $1 == id_name {
      for (i = 1; i <= NF; i++) col[$i] = i
      if (!col[id_name] || !col["uid"] || !col["cuid"]) exit 2
      header = 1
      next
    }
    header && NF {
      id = $(col[id_name]); uid = $(col["uid"]); cuid = $(col["cuid"])
      if (id !~ /^[0-9]+$/) exit 3
      if (uid == want_uid || cuid == want_uid || uid == want_user || cuid == want_user) print id
    }
    END { if (!header) exit 4 }
  ' <<< "${output}"
}
enumerate_ipc() {
  local kind="$1" ids
  if ! ids="$(ipc_ids "${kind}")"; then
    echo "sandbox-wipe.sh: cannot enumerate SysV IPC objects (${kind})" >&2
    return 1
  fi
  printf '%s' "${ids}"
}

# Do not enter a directory that the sandbox user cannot read, own or write.
# Such a directory cannot hold state that user created. Every other find error
# remains a wipe failure.
sandbox_find() {
  local root="$1"
  shift
  as_sandbox find -P "${root}" -xdev \
    \( -type d ! -readable ! -user "${SANDBOX_USER}" ! -writable -prune \) -o \
    \( -mindepth 1 "$@" \)
}

OWNED_PATHS=()
collect_owned_paths() {
  local root="$1" list
  list="$(mktemp)" || return 1
  if ! sandbox_find "${root}" -user "${SANDBOX_USER}" -print0 > "${list}" 2>/dev/null; then
    rm -f -- "${list}"
    return 1
  fi
  mapfile -d '' -t OWNED_PATHS < "${list}"
  rm -f -- "${list}"
}
delete_owned_paths() {
  local root="$1"
  if ! sandbox_find "${root}" -type d -user "${SANDBOX_USER}" -exec chmod u+rwx {} \; >/dev/null 2>&1 \
      || ! collect_owned_paths "${root}"; then
    return 1
  fi
  for ((i=${#OWNED_PATHS[@]} - 1; i >= 0; i--)); do
    as_sandbox rm -f -- "${OWNED_PATHS[i]}" >/dev/null 2>&1 \
      || as_sandbox rmdir -- "${OWNED_PATHS[i]}" >/dev/null 2>&1 \
      || true
  done
}
count_foreign_writable() {
  local root="$1" list
  list="$(mktemp)" || return 1
  if [[ -n "${JOB_DIR}" && "${root}" == "${JOB_DIR}" ]]; then
    sandbox_find "${root}" ! -path "${JOB_DIR}/adapter-home" ! -path "${JOB_DIR}/sock" \
      ! -user "${SANDBOX_USER}" -writable -print0 >"${list}" 2>/dev/null
  else
    sandbox_find "${root}" ! -user "${SANDBOX_USER}" -writable -print0 >"${list}" 2>/dev/null
  fi
  if [[ "$?" -ne 0 ]]; then
    rm -f -- "${list}"
    return 1
  fi
  COUNT="$(tr -cd '\0' <"${list}" | wc -c)"
  rm -f -- "${list}"
}

left=0
for d in "${DIRS[@]}"; do
  [[ -d "${d}" ]] || continue
  if ! delete_owned_paths "${d}"; then
    echo "sandbox-wipe.sh: cannot enumerate ${d}" >&2
    left=1
  fi
done
if ! delete_owned_paths "${WIPE_HOME}"; then
  echo "sandbox-wipe.sh: cannot enumerate the sandbox home" >&2
  left=1
fi
for kind in -m -s -q; do
  ids="$(enumerate_ipc "${kind}")" || exit 1
  while IFS= read -r id; do
    [[ -n "${id}" ]] || continue
    as_sandbox ipcrm "${kind}" "${id}" >/dev/null 2>&1 \
      || sudo -n ipcrm "${kind}" "${id}" >/dev/null 2>&1 \
      || true
  done <<< "${ids}"
done

for d in "${DIRS[@]}"; do
  [[ -d "${d}" ]] || continue
  if ! collect_owned_paths "${d}"; then
    echo "sandbox-wipe.sh: cannot verify ${d}" >&2
    left=1
    continue
  fi
  n="${#OWNED_PATHS[@]}"
  [[ "${n}" -eq 0 ]] || { echo "sandbox-wipe.sh: ${n} ${SANDBOX_USER}-owned path(s) remain under ${d}" >&2; left=1; }
done
if ! collect_owned_paths "${WIPE_HOME}"; then
  echo "sandbox-wipe.sh: cannot verify the sandbox home" >&2
  left=1
else
  n="${#OWNED_PATHS[@]}"
  [[ "${n}" -eq 0 ]] || { echo "sandbox-wipe.sh: ${n} ${SANDBOX_USER}-owned path(s) remain under the sandbox home" >&2; left=1; }
fi
SCAN_DIRS=("${DIRS[@]}")
[[ -z "${JOB_DIR}" || ! -d "${JOB_DIR}" ]] || SCAN_DIRS+=("${JOB_DIR}")
for d in "${SCAN_DIRS[@]}"; do
  [[ -d "${d}" ]] || continue
  if ! count_foreign_writable "${d}"; then
    echo "sandbox-wipe.sh: cannot check foreign-owned writable paths under ${d}" >&2
    left=1
    continue
  fi
  [[ "${COUNT}" -eq 0 ]] \
    || { echo "sandbox-wipe.sh: ${COUNT} foreign-owned writable path(s) remain under a checked directory" >&2; left=1; }
done
for kind in -m -s -q; do
  if ! ids="$(enumerate_ipc "${kind}")"; then left=1; continue; fi
  n=0
  [[ -z "${ids}" ]] || n="$(wc -l <<< "${ids}")"
  [[ "${n}" -eq 0 ]] || { echo "sandbox-wipe.sh: ${n} SysV IPC object(s) tied to uid ${SANDBOX_UID} (${kind}) remain" >&2; left=1; }
done
[[ "${left}" -eq 0 ]] || exit 1
echo "sandbox-wipe.sh: no sandbox state remains in ${#DIRS[@]} wipe directories, the passwd home, the job directory, or SysV IPC"

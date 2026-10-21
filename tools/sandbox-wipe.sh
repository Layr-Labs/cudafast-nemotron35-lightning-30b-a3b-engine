#!/usr/bin/env bash
#
# sandbox-wipe.sh -- remove everything the sandbox user left behind outside the
# per-job directory, so nothing survives from one job into the next
# (docs/sandbox-design.md section 5 item 6 and section 6a). TRUSTED: not in
# editablePaths. The workflow runs it at job start and in the cleanup step.
#
#   1. kills every sandbox process (nothing may recreate what is deleted);
#   2. as the sandbox user, deletes every file it owns under each directory in
#      the box's wipe list and under its home (never crossing a mount; races
#      ignored), after giving itself access to its own directories. The list
#      is every directory on the box the sandbox user can write (/tmp,
#      /var/tmp, /dev/shm, /run/lock, /var/crash, /dev/mqueue, ...), recorded
#      once by tools/box-runner/setup-sandbox-user.sh in a root-owned file. A
#      POSIX message queue is a file under /dev/mqueue, so it goes the same way;
#      a listed directory that does not exist (mqueue not mounted) is skipped;
#   3. as the sandbox user, removes every SysV shared memory segment, semaphore
#      set and message queue it owns;
#   4. verifies nothing the sandbox user owns is left in any of those places,
#      and exits non-zero if something is. Only counts are printed: a name
#      could carry prompt text.
#
# A missing or empty wipe list is a failure: the box was never set up.
#
# Env: SANDBOX_USER (default nemosub) and SANDBOX_ETC (default
# /etc/nemotron35-sandbox), the names tools/box-runner/setup-sandbox-user.sh
# uses. SANDBOX_WIPE_LIST (default $SANDBOX_ETC/wipe-dirs) and
# SANDBOX_WIPE_HOME (default /var/lib/$SANDBOX_USER) are for tests.
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
DIRS+=("${SANDBOX_WIPE_HOME:-/var/lib/${SANDBOX_USER}}")
as_sandbox() { sudo -n -u "${SANDBOX_USER}" -- "$@"; }

as_sandbox true || { echo "sandbox-wipe.sh: 'sudo -n -u ${SANDBOX_USER}' fails; cannot wipe" >&2; exit 1; }
as_sandbox kill -KILL -1 >/dev/null 2>&1

# ipc_ids FLAG -- ids of the SysV objects of one kind (-m, -s, -q) the sandbox
# user owns. ipcs prints: key id owner perms ...
ipc_ids() { ipcs "$1" 2>/dev/null | awk -v u="${SANDBOX_USER}" '$1 ~ /^0x/ && $3 == u { print $2 }'; }

for d in "${DIRS[@]}"; do
  [[ -d "${d}" ]] || continue
  as_sandbox find "${d}" -xdev -mindepth 1 -user "${SANDBOX_USER}" -type d -exec chmod u+rwx {} \; >/dev/null 2>&1
  as_sandbox find "${d}" -xdev -mindepth 1 -depth -user "${SANDBOX_USER}" -delete >/dev/null 2>&1
done
for kind in -m -s -q; do
  for id in $(ipc_ids "${kind}"); do as_sandbox ipcrm "${kind}" "${id}" >/dev/null 2>&1; done
done

left=0
for d in "${DIRS[@]}"; do
  [[ -d "${d}" ]] || continue
  n="$(as_sandbox find "${d}" -xdev -mindepth 1 -user "${SANDBOX_USER}" -print 2>/dev/null | wc -l)"
  [[ "${n}" -eq 0 ]] || { echo "sandbox-wipe.sh: ${n} ${SANDBOX_USER}-owned path(s) remain under ${d}" >&2; left=1; }
done
for kind in -m -s -q; do
  n="$(ipc_ids "${kind}" | wc -l)"
  [[ "${n}" -eq 0 ]] || { echo "sandbox-wipe.sh: ${n} ${SANDBOX_USER}-owned SysV IPC object(s) (ipcs ${kind}) remain" >&2; left=1; }
done
[[ "${left}" -eq 0 ]] || exit 1
echo "sandbox-wipe.sh: nothing ${SANDBOX_USER}-owned left in ${#DIRS[@]} directories or SysV IPC"

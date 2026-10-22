#!/usr/bin/env bash
# Remove old per-job sandbox state without following symlinks.
set -uo pipefail

die() { echo "sandbox-clear-job-dirs.sh: $*" >&2; exit 1; }
safe_name() { [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; }

[[ "$#" -eq 1 || "$#" -eq 3 ]] \
  || die "usage: sandbox-clear-job-dirs.sh SANDBOX_ROOT [--except SANDBOX_JOB]"
root="${1%/}"
except=""
if [[ "$#" -eq 3 ]]; then
  [[ "$2" == --except ]] || die "the second argument must be --except"
  safe_name "$3" || die "the excepted SANDBOX_JOB is not a safe name"
  except="$3"
fi
[[ "${root}" == /* && "${root}" != / && -d "${root}" && ! -L "${root}" ]] \
  || die "SANDBOX_ROOT is not a safe directory"

shopt -s dotglob nullglob
entries=("${root}"/*)
for path in "${entries[@]}"; do
  name="${path##*/}"
  safe_name "${name}" || die "SANDBOX_ROOT contains an unsafe job name"
  [[ ! -L "${path}" ]] || die "SANDBOX_ROOT contains a job symlink"
done

user="${SANDBOX_USER:-nemosub}"
sudo -n -u "${user}" -- true \
  || die "'sudo -n -u ${user}' fails; cannot clear job directories"

left="x"
for _ in 1 2 3 4 5 6 7 8 9 10; do
  sudo -n -u "${user}" -- kill -KILL -1 >/dev/null 2>&1 || true
  left="$(pgrep -u "${user}" 2>/dev/null | tr '\n' ' ' || true)"
  [[ -z "${left// /}" ]] && break
  sleep 0.5
done
[[ -z "${left// /}" ]] \
  || die "${user} processes survived kill -KILL -1: ${left}"

for path in "${entries[@]}"; do
  name="${path##*/}"
  [[ "${name}" != "${except}" ]] || continue
  if [[ -d "${path}" ]]; then
    sudo -n -u "${user}" -- chmod -R u+rwX "${path}" >/dev/null 2>&1 || true
    sudo -n -u "${user}" -- find -P "${path}" -mindepth 1 -delete >/dev/null 2>&1 || true
  fi
  rm -rf -- "${path}" || die "cannot remove a sandbox job entry"
done

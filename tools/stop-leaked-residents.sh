#!/usr/bin/env bash
#
# stop-leaked-residents.sh -- remove exact resident containers that a cancelled
# job could not stop, then wait until no compute process uses the GPU.
set -euo pipefail

usage() {
  echo "Usage: tools/stop-leaked-residents.sh --mode preflight|cleanup [--lock PATH] [--gpu-wait SECONDS] [--lock-wait SECONDS]" >&2
}

error() {
  echo "::error::$*" >&2
}

invalid() {
  error "$*"
  usage
  exit 2
}

mode=""
lock_path="/tmp/mtplx-gpu-exclusive.lock"
gpu_wait="60"
lock_wait="0"
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ "$#" -ge 2 ]] || invalid "--mode needs a value"
      mode="$2"
      shift 2
      ;;
    --lock)
      [[ "$#" -ge 2 && -n "$2" ]] || invalid "--lock needs a path"
      lock_path="$2"
      shift 2
      ;;
    --gpu-wait)
      [[ "$#" -ge 2 ]] || invalid "--gpu-wait needs a value"
      gpu_wait="$2"
      shift 2
      ;;
    --lock-wait)
      [[ "$#" -ge 2 ]] || invalid "--lock-wait needs a value"
      lock_wait="$2"
      shift 2
      ;;
    *)
      invalid "unknown argument: $1"
      ;;
  esac
done

[[ "${mode}" == "preflight" || "${mode}" == "cleanup" ]] \
  || invalid "--mode must be preflight or cleanup"
[[ "${gpu_wait}" =~ ^[0-9]+$ ]] || invalid "--gpu-wait must be a non-negative integer"
[[ "${lock_wait}" =~ ^[0-9]+$ ]] || invalid "--lock-wait must be a non-negative integer"
gpu_wait=$((10#${gpu_wait}))
lock_wait=$((10#${lock_wait}))

prefix="${SERVE_UP_CONTAINER_PREFIX:-}"
[[ -n "${prefix}" && "${prefix}" =~ ^[a-z0-9][a-z0-9-]*-$ ]] || {
  error "SERVE_UP_CONTAINER_PREFIX is unset or invalid; refusing to stop any container"
  exit 2
}

command -v flock >/dev/null 2>&1 || { error "flock is required for the GPU lock"; exit 1; }
if ! exec {lock_fd}< "${lock_path}"; then
  error "cannot open the GPU lock for reading: ${lock_path}"
  exit 1
fi
if [[ "${lock_wait}" -eq 0 ]]; then
  flock_args=(-n)
else
  flock_args=(-w "${lock_wait}")
fi
if flock "${flock_args[@]}" "${lock_fd}"; then
  :
else
  lock_rc=$?
  if [[ "${mode}" == "cleanup" && "${lock_rc}" -eq 1 ]]; then
    echo "::warning::the GPU lock is still held (a cancelled step may still be tearing down); not stopping containers now; the next job's preflight stops any leftover" >&2
    exit 0
  fi
  error "another GPU window holds it: ${lock_path}"
  exit 1
fi

list_containers() {
  local output
  if ! output="$(docker ps -a --format '{{.Names}}' 2>&1)"; then
    error "docker ps failed: ${output//$'\n'/; }"
    return 1
  fi
  printf '%s\n' "${output}"
}

find_matches() {
  local name
  while IFS= read -r name; do
    if [[ "${name}" =~ ^${prefix}[0-9]+-[0-9TZ]+$ ]]; then
      printf '%s\n' "${name}"
    fi
  done
}

containers="$(list_containers)" || exit 1
mapfile -t matches < <(find_matches <<< "${containers}")

if [[ "${#matches[@]}" -gt 0 ]]; then
  for name in "${matches[@]}"; do
    if [[ "${mode}" == "preflight" ]]; then
      echo "::warning::stopping resident container left by an earlier job: ${name}" >&2
    else
      echo "notice: stopping resident container during cleanup: ${name}" >&2
    fi
    if ! stop_output="$(docker stop -t 10 "${name}" 2>&1)"; then
      error "docker stop failed for ${name}: ${stop_output//$'\n'/; }"
      exit 1
    fi
    if ! rm_output="$(docker rm -f "${name}" 2>&1)"; then
      if [[ "${rm_output}" != *"No such container"* \
          && "${rm_output}" != *"already in progress"* ]]; then
        error "docker rm failed for ${name}: ${rm_output//$'\n'/; }"
        exit 1
      fi
    fi
  done
fi

container_deadline=$((SECONDS + gpu_wait))
while :; do
  containers="$(list_containers)" || exit 1
  mapfile -t remaining_matches < <(find_matches <<< "${containers}")
  [[ "${#remaining_matches[@]}" -eq 0 ]] && break
  if [[ "${SECONDS}" -ge "${container_deadline}" ]]; then
    error "resident containers remain after cleanup: ${remaining_matches[*]}"
    exit 1
  fi
  sleep 1
done

command -v nvidia-smi >/dev/null 2>&1 || { error "nvidia-smi is required to check GPU compute processes"; exit 1; }
command -v timeout >/dev/null 2>&1 || { error "timeout is required to bound the nvidia-smi query"; exit 1; }
deadline=$((SECONDS + gpu_wait))
while :; do
  remaining=$((deadline - SECONDS))
  [[ "${remaining}" -gt 0 ]] || remaining=1
  if gpu_apps="$(timeout --signal=KILL "${remaining}" \
      nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>&1)"; then
    :
  else
    query_rc=$?
    if [[ "${query_rc}" -eq 124 || "${query_rc}" -eq 137 ]]; then
      error "nvidia-smi query timed out after ${remaining} seconds"
      exit 1
    fi
    error "nvidia-smi failed: ${gpu_apps//$'\n'/; }"
    exit 1
  fi
  foreign="$(printf '%s\n' "${gpu_apps}" | grep -vE '^\s*$' | grep -vE 'Xorg|gnome-shell' || true)"
  [[ -z "${foreign}" ]] && break
  if [[ "${SECONDS}" -ge "${deadline}" ]]; then
    error "GPU compute processes remain after ${gpu_wait} seconds: ${foreign//$'\n'/; }"
    exit 1
  fi
  delay=$((deadline - SECONDS))
  [[ "${delay}" -gt 2 ]] && delay=2
  sleep "${delay}"
done

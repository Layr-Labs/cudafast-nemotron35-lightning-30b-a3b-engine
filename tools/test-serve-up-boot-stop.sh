#!/usr/bin/env bash
# test-serve-up-boot-stop.sh -- the per-leg boot/stop verbs in tools/serve-up.sh.
#
# The paired ranked path needs a resident that OUTLIVES the command that booted
# it: benchd boots each leg's resident from that leg's workspace, measures the
# leg through its own worker spawns, and stops it before the other leg boots.
# That is `--boot ... --socket-out FILE` and `--stop --socket PATH`, and the
# properties that make the pair usable are:
#
#   * the socket path is the FIRST LINE of the file, so one `head -1` reads it;
#   * --boot exits 0 with the resident STILL SERVING;
#   * --stop ends it and cleans up after it;
#   * --stop is IDEMPOTENT, because the caller runs it on success and on failure
#     alike and a second call must not turn a clean failure into two.
#
# This suite drives the REAL tools/serve-up.sh and the REAL resident
# (harness/vllm_resident/resident.py) on its fake backend, through the host
# launcher. The memory plan, the health probe, the identity file and the
# teardown are the real ones.
#
# Hermetic: no GPU, no container, no checkpoint, no network.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVE_UP="${ROOT_DIR}/tools/serve-up.sh"
# shellcheck source=tools/testlib/synthetic-window.sh
. "${ROOT_DIR}/tools/testlib/synthetic-window.sh"
synthetic_window_env

WORK="$(mktemp -d)"
ME="$(id -un)"
# A short socket directory. A Unix socket address holds 103 bytes on macOS, and
# the default mktemp directory plus serve-up.sh's run-tagged leaf is already
# close to it.
SOCKDIR="$(mktemp -d /tmp/suptest.XXXXXX)"
# The sandbox root for cases 7-9 (docs/sandbox-design.md 6a), short for the same reason.
SBXROOT="$(mktemp -d /tmp/supsbx.XXXXXX)"
DECOY_PID=""
cleanup() {
  # Never leave a stub resident behind, whatever the suite did.
  for f in "${SOCKDIR}"/*.pid "${WORK}"/stub/container.*; do
    [[ -e "${f}" ]] || continue
    pid="$(sed -n 1p "${f}" | tr -d '[:space:]')"
    [[ -n "${pid}" ]] && kill -KILL "${pid}" 2>/dev/null
  done
  [[ -z "${DECOY_PID}" ]] || kill -KILL "${DECOY_PID}" 2>/dev/null
  rm -rf "${WORK}" "${SOCKDIR}" "${SBXROOT}"
}
trap cleanup EXIT

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }

command -v python3 >/dev/null 2>&1 || { echo "test-serve-up-boot-stop.sh: python3 is required" >&2; exit 1; }
command -v jq      >/dev/null 2>&1 || { echo "test-serve-up-boot-stop.sh: jq is required" >&2; exit 1; }

# --- the synthetic window ---------------------------------------------------
WEIGHTS="${WORK}/weights"
synthetic_window_weights "${WEIGHTS}" || { echo "cannot write the synthetic window" >&2; exit 1; }
synthetic_window_meminfo "${WORK}/meminfo"
LOGDIR="${WORK}/logs"

serve() {
  env SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" \
      SERVE_UP_LOG_DIR="${LOGDIR}" \
      SERVE_UP_SOCKET_DIR="${SOCKDIR}" \
      SERVE_UP_HEALTH_TIMEOUT_S=60 \
      SERVE_UP_MEMINFO="${WORK}/meminfo" \
      "$@"
}

hello_ok() { # hello_ok SOCKET -- the probe a worker sends
  python3 - "$1" <<'PY'
import json, socket, sys
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(10)
s.connect(sys.argv[1])
s.sendall(b'{"op":"hello"}\n')
line = b""
while not line.endswith(b"\n"):
    chunk = s.recv(65536)
    if not chunk:
        break
    line += chunk
s.close()
sys.exit(0 if json.loads(line.decode()).get("ok") is True else 1)
PY
}

# --- case 1: --boot publishes the socket and leaves the resident serving -----
SOCKET_OUT="${WORK}/leg1/socket.txt"
BOOT_OUT="${WORK}/boot.out"
serve "${SERVE_UP}" --boot --spec serial --draft-len 0 --socket-out "${SOCKET_OUT}" \
  > "${BOOT_OUT}" 2>&1
rc=$?
[[ "${rc}" -eq 0 ]] || { fail "case 1: --boot exited ${rc}"; sed 's/^/    /' "${BOOT_OUT}" >&2; }

SOCKET="$(head -1 "${SOCKET_OUT}" 2>/dev/null)"
if [[ -n "${SOCKET}" && -S "${SOCKET}" ]]; then
  ok "case 1: the socket path is the first line of the socket-out file"
else
  fail "case 1: the socket-out file does not name a bound socket (got '${SOCKET:-}')"
fi
[[ "$(wc -l < "${SOCKET_OUT}" | tr -d ' ')" == "1" ]] \
  && ok "case 1: the socket-out file is exactly the one line" \
  || fail "case 1: the socket-out file is not one line"

RESIDENT_PID="$(sed -n 1p "${SOCKET}.pid" 2>/dev/null | tr -d '[:space:]')"
if [[ -n "${RESIDENT_PID}" ]] && kill -0 "${RESIDENT_PID}" 2>/dev/null; then
  ok "case 1: --boot exited 0 with the resident still running (pid ${RESIDENT_PID})"
else
  fail "case 1: the resident did not survive --boot"
fi
hello_ok "${SOCKET}" \
  && ok "case 1: the handed-over resident answers the hello a worker sends" \
  || fail "case 1: the handed-over resident does not answer a hello"

# The control leg is SERIAL: no drafter armed, depth 0.
if [[ -f "${LOGDIR}/serve-identity.json" ]]; then
  spec="$(jq -r '.spec_config' "${LOGDIR}/serve-identity.json")"
  armed="$(jq -r '.hello.mtp_armed' "${LOGDIR}/serve-identity.json")"
  drafts="$(jq -r '.hello.draft_tokens' "${LOGDIR}/serve-identity.json")"
  [[ "${spec}" == "serial" && "${armed}" == "false" && "${drafts}" == "1" ]] \
    && ok "case 1: --spec serial armed no drafter (spec_config=serial, mtp_armed=false, draft_tokens=1)" \
    || fail "case 1: --spec serial booted spec_config=${spec} mtp_armed=${armed} draft_tokens=${drafts}"
else
  fail "case 1: no serve-identity.json was written"
fi

# The resident log and identity file are never other-readable (a later job's
# sandbox user must not read what a candidate resident printed).
logs1=("${LOGDIR}"/vllm-resident.*.log)
[[ "$(stat -c %a "${logs1[0]}" 2>/dev/null)" == 600 && "$(stat -c %a "${LOGDIR}/serve-identity.json" 2>/dev/null)" == 640 ]] \
  && ok "case 1: the resident log is 0600 and the identity file 0640" \
  || fail "case 1: resident log mode $(stat -c %a "${logs1[0]}" 2>/dev/null), identity mode $(stat -c %a "${LOGDIR}/serve-identity.json" 2>/dev/null)"

# --- case 2: --stop ends it and cleans up -----------------------------------
STOP_OUT="${WORK}/stop.out"
serve "${SERVE_UP}" --stop --socket "${SOCKET}" > "${STOP_OUT}" 2>&1
rc=$?
[[ "${rc}" -eq 0 ]] || { fail "case 2: --stop exited ${rc}"; sed 's/^/    /' "${STOP_OUT}" >&2; }
for _ in $(seq 1 25); do
  kill -0 "${RESIDENT_PID}" 2>/dev/null || break
  sleep 0.2
done
kill -0 "${RESIDENT_PID}" 2>/dev/null \
  && fail "case 2: the resident survived --stop" \
  || ok "case 2: --stop ended the resident"
[[ ! -e "${SOCKET}" ]] && ok "case 2: --stop removed the socket" || fail "case 2: the socket survived --stop"
[[ ! -e "${SOCKET}.pid" ]] && ok "case 2: --stop removed the pid sidecar" || fail "case 2: the sidecar survived --stop"
[[ -z "$(find "${LOGDIR}" -maxdepth 1 -name '*.ready' -print -quit)" ]] \
  && ok "case 2: --stop removed the ready file" \
  || fail "case 2: the ready file survived --stop"

# --- case 3: --stop is idempotent -------------------------------------------
serve "${SERVE_UP}" --stop --socket "${SOCKET}" > "${WORK}/stop2.out" 2>&1
[[ $? -eq 0 ]] \
  && ok "case 3: a second --stop is a no-op that exits 0" \
  || { fail "case 3: the second --stop did not exit 0"; sed 's/^/    /' "${WORK}/stop2.out" >&2; }
grep -q 'nothing to stop' "${WORK}/stop2.out" \
  && ok "case 3: the second --stop says there was nothing to stop" \
  || fail "case 3: the second --stop did not say what it found"

serve "${SERVE_UP}" --stop --socket "${SOCKDIR}/never-booted.sock" > "${WORK}/stop3.out" 2>&1
[[ $? -eq 0 ]] \
  && ok "case 3: --stop on a socket that was never booted exits 0" \
  || fail "case 3: --stop on an unknown socket did not exit 0"

# --- case 4: --spec mtp arms the drafter at the declared depth ---------------
rm -f "${LOGDIR}/serve-identity.json"
SOCKET_OUT2="${WORK}/leg2/socket.txt"
serve "${SERVE_UP}" --boot --spec mtp --draft-len 2 --socket-out "${SOCKET_OUT2}" \
  > "${WORK}/boot2.out" 2>&1
rc=$?
[[ "${rc}" -eq 0 ]] || { fail "case 4: --boot --spec mtp exited ${rc}"; sed 's/^/    /' "${WORK}/boot2.out" >&2; }
SOCKET2="$(head -1 "${SOCKET_OUT2}" 2>/dev/null)"
if [[ -f "${LOGDIR}/serve-identity.json" ]]; then
  spec="$(jq -r '.spec_config' "${LOGDIR}/serve-identity.json")"
  drafts="$(jq -r '.hello.draft_tokens' "${LOGDIR}/serve-identity.json")"
  head="$(jq -r '.hello.mtp_armed' "${LOGDIR}/serve-identity.json")"
  [[ "${spec}" == "mtp2" && "${drafts}" == "3" && "${head}" == "true" ]] \
    && ok "case 4: --spec mtp --draft-len 2 armed the drafter (spec_config=mtp2, draft_tokens=3)" \
    || fail "case 4: booted spec_config=${spec} draft_tokens=${drafts} mtp_armed='${head}'"
else
  fail "case 4: no serve-identity.json was written for the mtp boot"
fi
[[ -n "${SOCKET2}" ]] && serve "${SERVE_UP}" --stop --socket "${SOCKET2}" >/dev/null 2>&1

# --- case 5: the refusals ---------------------------------------------------
refuse() { # refuse LABEL NEEDLE [extra env...] -- everything after is argv
  local label="$1" needle="$2"; shift 2
  local out rc
  out="$("$@" 2>&1)"
  rc=$?
  if [[ "${rc}" -eq 0 ]]; then
    fail "case 5 / ${label}: exited 0"
    return
  fi
  grep -qF -- "${needle}" <<<"${out}" \
    && ok "case 5 / ${label}: refused by name" \
    || fail "case 5 / ${label}: the refusal does not say '${needle}'; got: $(tail -1 <<<"${out}")"
}

refuse "serial with a draft length" "the serial control leg arms no drafter" \
  env SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" SERVE_UP_SOCKET_DIR="${SOCKDIR}" \
      "${SERVE_UP}" --boot --spec serial --draft-len 1 --socket-out "${WORK}/x.txt"
refuse "mtp with no draft length" "requires --draft-len N with N >= 1" \
  env SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" SERVE_UP_SOCKET_DIR="${SOCKDIR}" \
      "${SERVE_UP}" --boot --spec mtp --socket-out "${WORK}/x.txt"
refuse "no spec" "--boot requires --spec serial|mtp" \
  env SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" SERVE_UP_SOCKET_DIR="${SOCKDIR}" \
      "${SERVE_UP}" --boot --socket-out "${WORK}/x.txt"
refuse "no socket-out" "--boot requires --socket-out FILE" \
  env SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" SERVE_UP_SOCKET_DIR="${SOCKDIR}" \
      "${SERVE_UP}" --boot --spec serial --draft-len 0
refuse "stop with no socket" "--stop requires --socket PATH" \
  env "${SERVE_UP}" --stop
# A FLAG AT THE END OF THE ARGV. `--spec` with nothing after it used to take the
# empty string and then `shift 2` ran off the end, which under `set -u` ended the
# script with a bare exit 1 and no message: the caller saw a failed boot and
# nothing saying why. One case per flag that takes a value.
refuse "--spec with no value" "--spec requires a value" \
  env SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" SERVE_UP_SOCKET_DIR="${SOCKDIR}" \
      "${SERVE_UP}" --boot --spec
refuse "--draft-len with no value" "--draft-len requires a value" \
  env SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" SERVE_UP_SOCKET_DIR="${SOCKDIR}" \
      "${SERVE_UP}" --boot --spec serial --draft-len
refuse "--socket-out with no value" "--socket-out requires a value" \
  env SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" SERVE_UP_SOCKET_DIR="${SOCKDIR}" \
      "${SERVE_UP}" --boot --spec serial --draft-len 0 --socket-out
refuse "--socket with no value" "--socket requires a value" \
  env "${SERVE_UP}" --stop --socket
# THE ONE THAT PROTECTS THE CONTROL LEG: an inherited speculative value cannot
# turn a serial boot into a speculative one.
refuse "environment disagrees with the flag" "in --boot mode the flag is the authority" \
  env SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" SERVE_UP_SOCKET_DIR="${SOCKDIR}" \
      SERVE_UP_SPECULATIVE=1 \
      "${SERVE_UP}" --boot --spec serial --draft-len 0 --socket-out "${WORK}/x.txt"

# --- case 5b: a HANDOVER that fails leaves no stale pid sidecar --------------
# --boot writes <socket>.pid FIRST and publishes the socket second, so the caller
# never holds a socket path it cannot stop. That ordering creates the one window
# this case covers: a failure BETWEEN the sidecar write and the handover. The
# teardown then runs, and a sidecar left behind would name a dead pid and outlive
# every run that could remove it -- nothing later looks for it, because no caller
# ever learned the socket path.
#
# The failure is an unwritable --socket-out. /dev/null is not a directory, so the
# mkdir of its "subdirectory" fails with ENOTDIR after the sidecar exists. A boot
# that dies EARLIER (a resident that never comes up) would not exercise this at
# all: the sidecar would not have been written yet, and the case would pass on a
# teardown that removes nothing.
sidecars_before="$(find "${SOCKDIR}" -name '*.sock.pid' | wc -l | tr -d ' ')"
serve "${SERVE_UP}" --boot --spec serial --draft-len 0 \
  --socket-out /dev/null/not-a-directory/socket.txt \
  > "${WORK}/badhandover.out" 2>&1
[ $? -ne 0 ] \
  && ok "case 5b: a boot whose handover cannot be published fails" \
  || fail "case 5b: --boot exited 0 without publishing its socket"
sidecars_after="$(find "${SOCKDIR}" -name '*.sock.pid' | wc -l | tr -d ' ')"
[ "${sidecars_after}" = "${sidecars_before}" ] \
  && ok "case 5b: the failed handover left no stale pid sidecar behind" \
  || fail "case 5b: the failed handover left $(( sidecars_after - sidecars_before )) stale sidecar(s)"
leftover_socks="$(find "${SOCKDIR}" -name '*.sock' | wc -l | tr -d ' ')"
[ "${leftover_socks}" = "0" ] \
  && ok "case 5b: the failed handover left no socket behind" \
  || fail "case 5b: the failed handover left ${leftover_socks} socket(s)"

# --- case 6: the wrapper form still works -----------------------------------
# Local drivers (tools/benchmark.sh) still use it, so the per-leg verbs must be
# additive.
rm -f "${LOGDIR}/serve-identity.json"
WRAP_OUT="${WORK}/wrap.out"
serve "${SERVE_UP}" sh -c 'test -S "${DS4_RESIDENT_SOCKET}" && echo WRAPPED_OK' \
  > "${WRAP_OUT}" 2>&1
rc=$?
[[ "${rc}" -eq 0 ]] && grep -q 'WRAPPED_OK' "${WRAP_OUT}" \
  && ok "case 6: the wrapper form still boots, runs the command and exits with its code" \
  || { fail "case 6: the wrapper form regressed (exit ${rc})"; sed 's/^/    /' "${WRAP_OUT}" >&2; }
# And it still tears its own resident down: nothing is left listening.
leftover=0
for f in "${SOCKDIR}"/*.sock; do
  [[ -e "${f}" ]] && leftover=$((leftover + 1))
done
[[ "${leftover}" -eq 0 ]] \
  && ok "case 6: the wrapper form left no resident behind" \
  || fail "case 6: ${leftover} socket(s) survived the wrapper run"

# --- cases 7-9: the DOCKER launcher, unsandboxed and as the sandbox user -----
# docker, sudo, id and getent are stubs. The stub docker runs the "container" as
# a host process (the same real resident on its fake backend) and records its
# argv; the stub sudo records its argv and runs the command as this user. So the
# cases check WHAT serve-up asks for (--user, mounts, sudo argv, stop by name),
# not that the kernel enforces it. nemosub is uid 4242, nemoshare gid 4343.
#
# The docker launcher boots from the tree serve-up.sh lives in and needs a
# staged overlay, so the cases run a copy of the tree.
TREE="${WORK}/tree"
mkdir -p "${TREE}/tools" "${TREE}/fixtures" "${TREE}/harness" "${TREE}/.build/vllm-overlay/vllm"
cp "${SERVE_UP}" "${TREE}/tools/"
cp "${ROOT_DIR}/tools/sandbox-wipe.sh" "${ROOT_DIR}/tools/wait-for-memory.sh" "${TREE}/tools/"
cp "${ROOT_DIR}/fixtures/nemotron35_lightning_30b_a3b_track.json" "${TREE}/fixtures/"
cp -R "${ROOT_DIR}/harness/vllm_resident" "${TREE}/harness/"
STUB="${WORK}/stub"
mkdir -p "${STUB}/bin" "${WORK}/sbx-wipe/shm" "${WORK}/sbx-home"
printf '%s\n' "${WORK}/sbx-wipe/shm" > "${WORK}/sbx-wipe-dirs"
: > "${WORK}/sbx-ipc"
cat > "${STUB}/bin/ipcs" <<'STUBEOF'
#!/usr/bin/env bash
echo "------ header --------"
echo "key        id         owner      perms"
grep -- "^$1 " "${STUB_IPC}" 2>/dev/null | cut -d' ' -f2- || true
STUBEOF
cat > "${STUB}/bin/ipcrm" <<'STUBEOF'
#!/usr/bin/env bash
[[ -n "${STUB_IPC_STUCK:-}" ]] && exit 1
grep -v -- "^$1 0x[0-9a-f]* $2 " "${STUB_IPC}" > "${STUB_IPC}.new"
mv "${STUB_IPC}.new" "${STUB_IPC}"
STUBEOF
chmod +x "${STUB}/bin/ipcs" "${STUB}/bin/ipcrm"
cat > "${STUB}/bin/docker" <<STUBEOF
#!/usr/bin/env bash
printf '%s\n' "\$*" >> "${STUB}/docker.log"
case "\$1" in
  image) exit 0 ;;
  run)
    printf '%s\n' "\$@" > "${STUB}/docker-run.argv"
    shift; name=""; envs=()
    while [[ \$# -gt 0 ]]; do
      case "\$1" in
        --name) name="\$2"; shift 2 ;;
        --env) envs+=("\$2"); shift 2 ;;
        --entrypoint) shift 3; break ;;   # --entrypoint python3 IMAGE
        --rm|--init|--read-only) shift ;;
        *) shift 2 ;;
      esac
    done
    printf '%s\n' "\$\$" > "${STUB}/container.\${name}"
    exec env "\${envs[@]}" RESIDENT_BACKEND=fake python3 "\$@" ;;
  stop|rm)
    f="${STUB}/container.\${!#}"
    [[ -f "\${f}" ]] || { [[ "\$1" == rm ]]; exit; }
    pid="\$(cat "\${f}")"; kill -TERM "\${pid}" 2>/dev/null
    for _ in \$(seq 1 50); do kill -0 "\${pid}" 2>/dev/null || break; sleep 0.2; done
    rm -f "\${f}"; exit 0 ;;
esac
exit 1
STUBEOF
cat > "${STUB}/bin/sudo" <<STUBEOF
#!/usr/bin/env bash
printf '%s\n' "\$*" >> "${STUB}/sudo.log"
[[ "\${STUB_SUDO_FAIL:-0}" == 1 ]] && exit 1
[[ "\$1" == -n && "\$2" == -u && "\$4" == -- ]] || { echo "stub sudo: unexpected argv: \$*" >&2; exit 1; }
shift 4
[[ "\$1" == kill || "\$1" == true ]] && exit 0
[[ "\$1" == ipcrm ]] && exec "${STUB}/bin/ipcrm" "\${@:2}"
exec "\$@"
STUBEOF
cat > "${STUB}/bin/pgrep" <<'STUBEOF'
#!/usr/bin/env bash
[[ "$1" == -u && -n "${STUB_PGREP_USER:-}" && "$2" == "${STUB_PGREP_USER}" ]] && exit 1
exec pgrep "$@"
STUBEOF
chmod +x "${STUB}/bin/pgrep"
cat > "${STUB}/bin/id" <<STUBEOF
#!/usr/bin/env bash
[[ "\$*" == "-u nemosub" || "\$*" == "-u ${ME}" ]] && { echo 4242; exit 0; }
exec $(command -v id) "\$@"
STUBEOF
cat > "${STUB}/bin/getent" <<STUBEOF
#!/usr/bin/env bash
[[ "\$*" == "group nemoshare" ]] && { echo 'nemoshare:x:4343:runner,nemosub'; exit 0; }
[[ "\$*" == "group layrshare" ]] && { echo 'layrshare:x:5353:layrbench,layrsub'; exit 0; }
exec $(command -v getent 2>/dev/null || echo false) "\$@"
STUBEOF
chmod +x "${STUB}/bin/"*

dserve() { # dserve [VAR=VALUE...] ARGS... -- serve-up from the copied tree via the stub docker
  env PATH="${STUB}/bin:${PATH}" SERVE_UP_LAUNCHER=docker \
      SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" SERVE_UP_LOG_DIR="${WORK}/dlogs" \
      SERVE_UP_SOCKET_DIR="${SOCKDIR}" SERVE_UP_HEALTH_TIMEOUT_S=60 \
      SERVE_UP_MEMINFO="${WORK}/meminfo" "$@"
}
run_arg_after() { # run_arg_after FLAG -- every value that follows FLAG in the last docker run
  awk -v flag="$1" 'prev == flag { print } { prev = $0 }' "${STUB}/docker-run.argv"
}

# --- case 7: unsandboxed, the docker path is exactly today's -----------------
dserve "${TREE}/tools/serve-up.sh" --boot --spec serial --draft-len 0 \
  --socket-out "${WORK}/leg7/socket.txt" > "${WORK}/boot7.out" 2>&1 \
  || { fail "case 7: --boot exited non-zero"; sed 's/^/    /' "${WORK}/boot7.out" >&2; }
SOCKET7="$(head -1 "${WORK}/leg7/socket.txt" 2>/dev/null)"
[[ "${SOCKET7}" == "${SOCKDIR}"/vllm-resident.*.sock && -S "${SOCKET7}" ]] \
  && ok "case 7: the socket is bound in SERVE_UP_SOCKET_DIR under today's name" \
  || fail "case 7: unexpected socket '${SOCKET7}'"
[[ "$(run_arg_after --user)" == "$(id -u):$(id -g)" ]] \
  && ok "case 7: the container runs as the invoking user" \
  || fail "case 7: docker run --user is '$(run_arg_after --user)'"
[[ -z "$(run_arg_after --security-opt)$(run_arg_after --cap-drop)" ]] \
  && ok "case 7: no hardening flags are added unsandboxed" \
  || fail "case 7: unsandboxed run got hardening flags"
[[ "$(run_arg_after --mount | grep -vc ',readonly$')" == "2" ]] \
  && run_arg_after --mount | grep -qxF "type=bind,src=${SOCKDIR},dst=${SOCKDIR}" \
  && run_arg_after --mount | grep -qxF "type=bind,src=${WORK}/dlogs,dst=${WORK}/dlogs" \
  && ok "case 7: the writable mounts are the socket dir and the log dir, as today" \
  || fail "case 7: the writable mounts changed: $(run_arg_after --mount | tr '\n' ' ')"
[[ "$(sed -n 3p "${SOCKET7}.pid" 2>/dev/null)" == serve-up-* ]] \
  && ok "case 7: the pid sidecar beside the socket names the container" \
  || fail "case 7: no sidecar with the container name beside the socket"
dserve "${TREE}/tools/serve-up.sh" --stop --socket "${SOCKET7}" > "${WORK}/stop7.out" 2>&1
[[ ! -e "${SOCKET7}" && ! -e "${SOCKET7}.pid" && -z "$(ls "${STUB}"/container.* 2>/dev/null)" ]] \
  && ok "case 7: --stop ended the container and removed socket and sidecar" \
  || fail "case 7: --stop left the socket, the sidecar or the container"
[[ ! -e "${STUB}/sudo.log" ]] \
  && ok "case 7: nothing went through sudo" \
  || fail "case 7: sudo was called: $(cat "${STUB}/sudo.log")"
grep -q 'sandbox-wipe.sh' "${WORK}/stop7.out" 2>/dev/null \
  && fail "case 7: unsandboxed --stop must not run sandbox-wipe" \
  || ok "case 7: unsandboxed --stop does not run sandbox-wipe"

# --- case 8: SERVE_UP_CONTAINER_USER=nemosub -------------------------------
mkdir -p "${SBXROOT}/42-1" && mkdir -m 2750 "${SBXROOT}/42-1/sock"
sbx() {
  dserve SERVE_UP_CONTAINER_USER="${ME}" SANDBOX_ROOT="${SBXROOT}" SANDBOX_JOB=42-1 \
    SANDBOX_WIPE_LIST="${WORK}/sbx-wipe-dirs" SANDBOX_WIPE_HOME="${WORK}/sbx-home" \
    STUB_IPC="${WORK}/sbx-ipc" STUB_PGREP_USER="${ME}" "$@"
}
sbx "${TREE}/tools/serve-up.sh" --boot --spec mtp --draft-len 2 \
  --socket-out "${WORK}/leg8/socket.txt" > "${WORK}/boot8.out" 2>&1 \
  || { fail "case 8: --boot exited non-zero"; sed 's/^/    /' "${WORK}/boot8.out" >&2; }
SOCKET8="$(head -1 "${WORK}/leg8/socket.txt" 2>/dev/null)"
LEGDIR8="$(dirname -- "${SOCKET8}")"
TAG8="$(basename -- "${LEGDIR8}")"
[[ "${SOCKET8}" == "${SBXROOT}/42-1/sock/${TAG8}/vllm-resident.sock" && "${TAG8}" =~ ^[0-9]+-[0-9]{8}T[0-9]{6}Z$ && -S "${SOCKET8}" ]] \
  && ok "case 8: --socket-out names the socket in \$SANDBOX_ROOT/\$SANDBOX_JOB/sock/<run tag>/" \
  || fail "case 8: unexpected socket '${SOCKET8}'"
[[ "$(stat -c %a "${LEGDIR8}" 2>/dev/null)" == "2770" ]] \
  && ok "case 8: the per-leg socket dir is mode 2770" \
  || fail "case 8: the per-leg socket dir is mode $(stat -c %a "${LEGDIR8}" 2>/dev/null)"
[[ "$(run_arg_after --user)" == "4242:4343" ]] \
  && ok "case 8: the container runs as nemosub's uid and nemoshare's gid" \
  || fail "case 8: docker run --user is '$(run_arg_after --user)'"
run_arg_after --security-opt | grep -qxF no-new-privileges && run_arg_after --cap-drop | grep -qxF ALL \
  && ok "case 8: the container gets no-new-privileges and --cap-drop ALL" \
  || fail "case 8: hardening flags missing: $(tr '\n' ' ' < "${STUB}/docker-run.argv")"
[[ "$(run_arg_after --mount | grep -v ',readonly$')" == "type=bind,src=${LEGDIR8},dst=${LEGDIR8}" ]] \
  && ok "case 8: the per-leg dir is the container's only writable mount" \
  || fail "case 8: the writable mounts are: $(run_arg_after --mount | grep -v ',readonly$' | tr '\n' ' ')"
run_arg_after --env | grep -qxF "HOME=${LEGDIR8}/run/home" \
  && run_arg_after --env | grep -qxF "RESIDENT_READY_FILE=" \
  && ok "case 8: HOME is in the per-leg dir and no ready file is asked for" \
  || fail "case 8: the resident env is: $(run_arg_after --env | grep -E '^(HOME|RESIDENT_READY_FILE)=' | tr '\n' ' ')"
[[ -z "$(find "${SBXROOT}" "${WORK}/dlogs" \( -name '*.pid' -o -name '*.ready' \) -print -quit)" ]] \
  && [[ "$(jq -r .resident_socket "${WORK}/dlogs/serve-identity.json" 2>/dev/null)" == "${SOCKET8}" ]] \
  && ok "case 8: no sidecar or ready file is written; the identity file is in the log dir" \
  || fail "case 8: a sidecar or ready file was written, or the identity file is missing"
logs8=("${WORK}/dlogs/vllm-resident.${TAG8}.log")
[[ "$(stat -c %a "${logs8[0]}" 2>/dev/null)" == 600 && "$(stat -c %a "${WORK}/dlogs/serve-identity.json" 2>/dev/null)" == 640 ]] \
  && ok "case 8: the candidate resident's log is 0600, the identity file 0640" \
  || fail "case 8: resident log mode $(stat -c %a "${logs8[0]}" 2>/dev/null), identity mode $(stat -c %a "${WORK}/dlogs/serve-identity.json" 2>/dev/null)"
grep -qxF -- "-n -u ${ME} -- python3 - ${SOCKET8}" "${STUB}/sudo.log" 2>/dev/null \
  && ok "case 8: the health probe ran as the sandbox user through sudo -n -u ${ME} --" \
  || fail "case 8: the probe did not go through sudo: $(cat "${STUB}/sudo.log" 2>/dev/null)"

# A sidecar the sandbox user plants beside its socket, naming a process it does
# not own, must not be signalled: --stop goes by the container name only.
echo candidate-shm-leftover > "${WORK}/sbx-wipe/shm/candidate-shm-leftover"
sleep 300 & DECOY_PID=$!
printf '%s\n\n\n' "${DECOY_PID}" > "${SOCKET8}.pid"
sbx WAIT_FOR_MEMORY_TIMEOUT_S=0.1 WAIT_FOR_MEMORY_POLL_S=0.01 \
  "${TREE}/tools/serve-up.sh" --stop --socket "${SOCKET8}" > "${WORK}/stop8.out" 2>&1 \
  || fail "case 8: --stop exited non-zero"
grep -qxF "stop -t 10 serve-up-${TAG8}" "${STUB}/docker.log" \
  && [[ ! -e "${STUB}/container.serve-up-${TAG8}" && ! -S "${SOCKET8}" ]] \
  && ok "case 8: --stop stopped the container by name and removed the socket" \
  || { fail "case 8: --stop did not stop serve-up-${TAG8} by name"; sed 's/^/    /' "${WORK}/stop8.out" >&2; }
kill -0 "${DECOY_PID}" 2>/dev/null \
  && ok "case 8: --stop did not signal the pid in a planted sidecar" \
  || fail "case 8: --stop signalled the planted pid ${DECOY_PID}"
grep -Fxq -- "-n -u ${ME} -- kill -KILL -1" "${STUB}/sudo.log" 2>/dev/null \
  && ok "case 8: --stop kills remaining sandbox processes before the wipe" \
  || fail "case 8: --stop did not run kill -KILL -1 as ${ME}"
[[ ! -f "${WORK}/sbx-wipe/shm/candidate-shm-leftover" ]] \
  && grep -q "sandbox-wipe.sh: nothing ${ME}-owned left" "${WORK}/stop8.out" \
  && grep -q 'wait-for-memory:' "${WORK}/stop8.out" \
  && ok "case 8: --stop runs sandbox-wipe after a sandboxed resident" \
  || fail "case 8: --stop did not wipe shared dirs (leftover=$(test -f "${WORK}/sbx-wipe/shm/candidate-shm-leftover" && echo yes || echo no))"
# Without SERVE_UP_CONTAINER_USER or SANDBOX_* the socket shape alone keeps --stop
# off the sidecar and skips the wipe.
dserve "${TREE}/tools/serve-up.sh" --stop --socket "${SOCKET8}" > "${WORK}/stop8b.out" 2>&1 \
  && grep -q 'nothing to stop' "${WORK}/stop8b.out" && kill -0 "${DECOY_PID}" 2>/dev/null \
  && ok "case 8: a second --stop without sandbox env is a no-op and still ignores the sidecar" \
  || fail "case 8: a second --stop without sandbox env misbehaved"

# --- case 8d: sandbox socket shape + SANDBOX_USER, no SERVE_UP_CONTAINER_USER ---
sbx "${TREE}/tools/serve-up.sh" --boot --spec serial --draft-len 0 \
  --socket-out "${WORK}/leg8d/socket.txt" > "${WORK}/boot8d.out" 2>&1 \
  || { fail "case 8d: --boot exited non-zero"; sed 's/^/    /' "${WORK}/boot8d.out" >&2; }
SOCKET8D="$(head -1 "${WORK}/leg8d/socket.txt" 2>/dev/null)"
echo socket-shape-wipe-leftover > "${WORK}/sbx-wipe/shm/socket-shape-wipe-leftover"
: > "${STUB}/sudo.log"
dserve SANDBOX_USER="${ME}" SANDBOX_ROOT="${SBXROOT}" SANDBOX_JOB=42-1 \
  SANDBOX_WIPE_LIST="${WORK}/sbx-wipe-dirs" SANDBOX_WIPE_HOME="${WORK}/sbx-home" \
  STUB_IPC="${WORK}/sbx-ipc" STUB_PGREP_USER="${ME}" \
  SERVE_UP_MEMINFO="${WORK}/meminfo" WAIT_FOR_MEMORY_TIMEOUT_S=0.1 WAIT_FOR_MEMORY_POLL_S=0.01 \
  "${TREE}/tools/serve-up.sh" --stop --socket "${SOCKET8D}" > "${WORK}/stop8d.out" 2>&1 \
  || fail "case 8d: --stop exited non-zero"
[[ ! -f "${WORK}/sbx-wipe/shm/socket-shape-wipe-leftover" ]] \
  && grep -q "sandbox-wipe.sh: nothing ${ME}-owned left" "${WORK}/stop8d.out" \
  && grep -q 'wait-for-memory:' "${WORK}/stop8d.out" \
  && ok "case 8d: --stop wipes (and waits for memory) from socket shape + SANDBOX_USER alone" \
  || fail "case 8d: socket-shape-only stop did not wipe (leftover=$(test -f "${WORK}/sbx-wipe/shm/socket-shape-wipe-leftover" && echo yes || echo no))"

# The two off switches drop the hardening flags, each on its own.
sbx SERVE_UP_NO_CAP_DROP=1 SERVE_UP_ALLOW_NEW_PRIVS=1 "${TREE}/tools/serve-up.sh" --boot --spec serial --draft-len 0 \
  --socket-out "${WORK}/leg8c/socket.txt" > "${WORK}/boot8c.out" 2>&1 \
  || { fail "case 8: --boot with the off switches exited non-zero"; sed 's/^/    /' "${WORK}/boot8c.out" >&2; }
[[ -z "$(run_arg_after --security-opt)$(run_arg_after --cap-drop)" ]] \
  && ok "case 8: SERVE_UP_NO_CAP_DROP=1 and SERVE_UP_ALLOW_NEW_PRIVS=1 turn the hardening off" \
  || fail "case 8: the off switches left: $(run_arg_after --security-opt) $(run_arg_after --cap-drop)"
sbx "${TREE}/tools/serve-up.sh" --stop --socket "$(head -1 "${WORK}/leg8c/socket.txt" 2>/dev/null)" > /dev/null 2>&1 || true

# --- case 8e: SERVE_UP_SANDBOX_TREE sandboxes the named tree only ----------
# benchd boots both legs of a paired run with one environment. The tree the
# workflow names (the candidate checkout) runs as SANDBOX_USER; any other tree
# (the reference tree) runs as the invoking user, so benchd's own control-leg
# worker can open its socket.
named() {
  dserve SANDBOX_USER="${ME}" SANDBOX_ROOT="${SBXROOT}" SANDBOX_JOB=42-1 \
    SANDBOX_WIPE_LIST="${WORK}/sbx-wipe-dirs" SANDBOX_WIPE_HOME="${WORK}/sbx-home" \
    STUB_IPC="${WORK}/sbx-ipc" STUB_PGREP_USER="${ME}" \
    WAIT_FOR_MEMORY_TIMEOUT_S=0.1 WAIT_FOR_MEMORY_POLL_S=0.01 "$@"
}
named SERVE_UP_SANDBOX_TREE="${TREE}" "${TREE}/tools/serve-up.sh" --boot --spec mtp --draft-len 3 \
  --socket-out "${WORK}/leg8e/socket.txt" > "${WORK}/boot8e.out" 2>&1 \
  || { fail "case 8e: --boot of the named tree exited non-zero"; sed 's/^/    /' "${WORK}/boot8e.out" >&2; }
SOCKET8E="$(head -1 "${WORK}/leg8e/socket.txt" 2>/dev/null)"
[[ "${SOCKET8E}" == "${SBXROOT}/42-1/sock/"*/vllm-resident.sock && "$(run_arg_after --user)" == "4242:4343" ]] \
  && ok "case 8e: the tree SERVE_UP_SANDBOX_TREE names boots sandboxed, as SANDBOX_USER" \
  || fail "case 8e: the named tree booted at '${SOCKET8E}' as '$(run_arg_after --user)'"
named SERVE_UP_SANDBOX_TREE="${TREE}" "${TREE}/tools/serve-up.sh" --stop --socket "${SOCKET8E}" > /dev/null 2>&1 \
  || fail "case 8e: --stop of the named tree exited non-zero"
named SERVE_UP_SANDBOX_TREE="${WORK}" "${TREE}/tools/serve-up.sh" --boot --spec serial --draft-len 0 \
  --socket-out "${WORK}/leg8f/socket.txt" > "${WORK}/boot8f.out" 2>&1 \
  || { fail "case 8e: --boot of another tree exited non-zero"; sed 's/^/    /' "${WORK}/boot8f.out" >&2; }
SOCKET8F="$(head -1 "${WORK}/leg8f/socket.txt" 2>/dev/null)"
[[ "${SOCKET8F}" == "${SOCKDIR}"/vllm-resident.*.sock && "$(run_arg_after --user)" == "$(id -u):$(id -g)" ]] \
  && ok "case 8e: a tree SERVE_UP_SANDBOX_TREE does not name boots unsandboxed, as the invoking user" \
  || fail "case 8e: the other tree booted at '${SOCKET8F}' as '$(run_arg_after --user)'"
named SERVE_UP_SANDBOX_TREE="${WORK}" "${TREE}/tools/serve-up.sh" --stop --socket "${SOCKET8F}" > "${WORK}/stop8f.out" 2>&1
[[ ! -e "${SOCKET8F}" ]] && ! grep -q 'sandbox-wipe.sh' "${WORK}/stop8f.out" \
  && ok "case 8e: the other tree's --stop is the unsandboxed stop" \
  || fail "case 8e: the other tree's --stop misbehaved"
refuse "a named tree without SANDBOX_USER" "SANDBOX_USER is unset" \
  named SANDBOX_USER= SERVE_UP_SANDBOX_TREE="${TREE}" "${TREE}/tools/serve-up.sh" --boot --spec serial --draft-len 0 --socket-out "${WORK}/x.txt"

# --- case 9: the sandbox refusals -------------------------------------------
refuse "sandbox on the host launcher" "cannot be combined with SERVE_UP_LAUNCHER=host" \
  sbx SERVE_UP_LAUNCHER=host "${TREE}/tools/serve-up.sh" --boot --spec serial --draft-len 0 --socket-out "${WORK}/x.txt"
refuse "sandbox without SANDBOX_JOB" "needs SANDBOX_ROOT and SANDBOX_JOB" \
  sbx SANDBOX_JOB= "${TREE}/tools/serve-up.sh" --boot --spec serial --draft-len 0 --socket-out "${WORK}/x.txt"
refuse "sandbox with no per-job socket dir" "per-job socket directory" \
  sbx SANDBOX_JOB=nope "${TREE}/tools/serve-up.sh" --boot --spec serial --draft-len 0 --socket-out "${WORK}/x.txt"

# --- case 10: a failing sandbox wipe fails sandboxed --stop ------------------
sbx "${TREE}/tools/serve-up.sh" --boot --spec serial --draft-len 0 \
  --socket-out "${WORK}/leg10/socket.txt" > "${WORK}/boot10.out" 2>&1 \
  || { fail "case 10: --boot exited non-zero"; sed 's/^/    /' "${WORK}/boot10.out" >&2; }
SOCKET10="$(head -1 "${WORK}/leg10/socket.txt" 2>/dev/null)"
sbx SANDBOX_WIPE_LIST="${WORK}/no-such-wipe-list" "${TREE}/tools/serve-up.sh" \
  --stop --socket "${SOCKET10}" > "${WORK}/stop10.out" 2>&1
rc=$?
[[ "${rc}" -ne 0 ]] \
  && grep -q 'sandbox wipe after candidate-leg --stop failed' "${WORK}/stop10.out" \
  && ok "case 10: a failing sandbox wipe makes sandboxed --stop exit non-zero" \
  || { fail "case 10: expected wipe failure on --stop (rc=${rc})"; sed 's/^/    /' "${WORK}/stop10.out" >&2; }

# --- case 11: a second runner's names (docs/sandbox-design.md 6b) -----------
# SHARE_GROUP, SANDBOX_ETC and SERVE_UP_CONTAINER_PREFIX set: the container runs
# with that group's gid under the prefixed name, --stop stops it by that name,
# and the wipe reads $SANDBOX_ETC/wipe-dirs.
mkdir -p "${WORK}/sbx-etc"
cp "${WORK}/sbx-wipe-dirs" "${WORK}/sbx-etc/wipe-dirs"
second() {
  dserve SERVE_UP_CONTAINER_USER="${ME}" SANDBOX_ROOT="${SBXROOT}" SANDBOX_JOB=42-1 \
    SHARE_GROUP=layrshare SANDBOX_ETC="${WORK}/sbx-etc" SERVE_UP_CONTAINER_PREFIX=layr-serve-up- \
    STUB_IPC="${WORK}/sbx-ipc" STUB_PGREP_USER="${ME}" "$@"
}
second "${TREE}/tools/serve-up.sh" --boot --spec serial --draft-len 0 \
  --socket-out "${WORK}/leg11/socket.txt" > "${WORK}/boot11.out" 2>&1 \
  || { fail "case 11: --boot exited non-zero"; sed 's/^/    /' "${WORK}/boot11.out" >&2; }
SOCKET11="$(head -1 "${WORK}/leg11/socket.txt" 2>/dev/null)"
TAG11="$(basename -- "$(dirname -- "${SOCKET11}")")"
[[ "$(run_arg_after --name)" == "layr-serve-up-${TAG11}" && "$(run_arg_after --user)" == "4242:5353" ]] \
  && ok "case 11: the container is layr-serve-up-<tag> and runs with layrshare's gid" \
  || fail "case 11: docker run --name '$(run_arg_after --name)' --user '$(run_arg_after --user)'"
echo second-leftover > "${WORK}/sbx-wipe/shm/second-leftover"
second WAIT_FOR_MEMORY_TIMEOUT_S=0.1 WAIT_FOR_MEMORY_POLL_S=0.01 \
  "${TREE}/tools/serve-up.sh" --stop --socket "${SOCKET11}" > "${WORK}/stop11.out" 2>&1 \
  || fail "case 11: --stop exited non-zero"
grep -qxF "stop -t 10 layr-serve-up-${TAG11}" "${STUB}/docker.log" \
  && [[ ! -e "${STUB}/container.layr-serve-up-${TAG11}" ]] \
  && ok "case 11: --stop stopped layr-serve-up-<tag> by name" \
  || { fail "case 11: --stop did not stop layr-serve-up-${TAG11}"; sed 's/^/    /' "${WORK}/stop11.out" >&2; }
[[ ! -f "${WORK}/sbx-wipe/shm/second-leftover" ]] \
  && ok "case 11: the wipe read \$SANDBOX_ETC/wipe-dirs" \
  || fail "case 11: the wipe did not use \$SANDBOX_ETC/wipe-dirs"
# Unsandboxed: the sidecar names the prefixed container and --stop removes it.
dserve SERVE_UP_CONTAINER_PREFIX=layr-serve-up- "${TREE}/tools/serve-up.sh" --boot --spec serial --draft-len 0 \
  --socket-out "${WORK}/leg11b/socket.txt" > "${WORK}/boot11b.out" 2>&1 \
  || { fail "case 11: unsandboxed --boot exited non-zero"; sed 's/^/    /' "${WORK}/boot11b.out" >&2; }
SOCKET11B="$(head -1 "${WORK}/leg11b/socket.txt" 2>/dev/null)"
NAME11B="$(sed -n 3p "${SOCKET11B}.pid" 2>/dev/null)"
[[ "${NAME11B}" == layr-serve-up-* ]] \
  && ok "case 11: the unsandboxed sidecar names layr-serve-up-<tag>" \
  || fail "case 11: the unsandboxed sidecar names '${NAME11B}'"
dserve SERVE_UP_CONTAINER_PREFIX=layr-serve-up- "${TREE}/tools/serve-up.sh" --stop --socket "${SOCKET11B}" > "${WORK}/stop11b.out" 2>&1
grep -qxF "rm -f ${NAME11B}" "${STUB}/docker.log" && [[ ! -e "${STUB}/container.${NAME11B}" ]] \
  && ok "case 11: unsandboxed --stop removed the prefixed container" \
  || fail "case 11: unsandboxed --stop did not remove ${NAME11B}"

if [[ "${failures}" -eq 0 ]]; then
  echo "PASS: test-serve-up-boot-stop.sh"
  exit 0
fi
echo "FAIL: ${failures} failure(s)" >&2
exit 1

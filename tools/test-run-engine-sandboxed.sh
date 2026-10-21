#!/usr/bin/env bash
# test-run-engine-sandboxed.sh -- tools/run-engine-sandboxed.sh, the candidate
# leg's engine wrapper (docs/sandbox-design.md section 5 item 4).
#
# The properties it must keep:
#
#   * stdin/stdout are passed straight through: a message round-trips before the
#     next is sent, so nothing buffers;
#   * the adapter runs through `sudo -n -u nemosub --preserve-env=...` with the
#     two socket variables, and its arguments unchanged;
#   * the adapter's stderr lands in a 0600 file, never on the wrapper's stderr;
#   * the adapter's exit status is the wrapper's;
#   * every sandbox process is killed after the adapter exits, and one that
#     survives fails the wrapper;
#   * SIGTERM reaches the adapter;
#   * HOME and TMPDIR are the per-job adapter-home, and a missing one is refused;
#   * the shim tools/stage-cuda-engine.sh stages at the benchd path execs the
#     wrapper on <staged>.sandboxed with stdio, exit status and the kill-all
#     intact, under a cleared environment (benchd spawns it with env -i);
#   * the stderr log dir comes only from SANDBOX_ENGINE_LOG_DIR.
#   * the adapter runs under prlimit --fsize=64 MiB, --nproc=4096, --as=16 GiB:
#     a stderr flood stops at the fsize cap (the writer gets SIGXFSZ) and stdio
#     keeps flowing.
#
# Hermetic: sudo and pgrep are stubs on PATH. The stub sudo runs the command as
# the current user with a scrubbed environment, and its `kill -KILL -1` kills
# only the pids the fake adapter recorded -- never this user's processes.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRAPPER="${ROOT_DIR}/tools/run-engine-sandboxed.sh"
WORK="$(mktemp -d)"
PIDS="${WORK}/sandbox.pids"
cleanup() {
  [[ -f "${PIDS}" ]] && while read -r p; do kill -KILL "${p}" 2>/dev/null; done < "${PIDS}"
  rm -rf "${WORK}"
}
trap cleanup EXIT

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }

mkdir -p "${WORK}/bin" "${WORK}/logs" "${WORK}/sbx/7-1/adapter-home"
: > "${PIDS}"

cat > "${WORK}/bin/sudo" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${STUB_SUDO_LOG}"
user=""; keep=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) shift ;;
    -u) user="$2"; shift 2 ;;
    --preserve-env=*) keep="${1#*=}"; shift ;;
    --) shift; break ;;
    *) echo "stub sudo: unexpected option $1" >&2; exit 97 ;;
  esac
done
[[ "${user}" == "${STUB_USER:-nemosub}" ]] || { echo "stub sudo: not the sandbox user: ${user}" >&2; exit 98; }
if [[ "$1" == kill ]]; then
  [[ "$2 $3" == "-KILL -1" ]] || exit 96
  while read -r p; do kill -KILL "${p}" 2>/dev/null; done < "${STUB_SANDBOX_PIDS}"
  exit 0
fi
envs=("PATH=${PATH}")
for v in ${keep//,/ }; do
  [[ -n "${!v:-}" ]] && envs+=("${v}=${!v}")
done
exec env -i "${envs[@]}" "$@"
EOF

cat > "${WORK}/bin/pgrep" <<'EOF'
#!/usr/bin/env bash
[[ "$1" == -u && "$2" == "${STUB_USER:-nemosub}" ]] || exit 3
if [[ -n "${STUB_PGREP_STUCK:-}" ]]; then echo 99999; exit 0; fi
found=1
while read -r p; do kill -0 "${p}" 2>/dev/null && { echo "${p}"; found=0; }; done < "${STUB_SANDBOX_PIDS}"
exit "${found}"
EOF

# The fake adapter: an Engine-Protocol-shaped stdio loop. It leaves a process
# behind (the wrapper must kill it), writes "prompt" text to stderr (it must not
# reach the job log), and exits with the status it is told.
cat > "${WORK}/adapter" <<'EOF'
#!/usr/bin/env bash
pids=""; code=0; flood=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pids) pids="$2"; shift 2 ;;
    --exit) code="$2"; shift 2 ;;
    --trap) trap 'echo "adapter got TERM" >&2; exit 42' TERM; shift ;;
    --flood) flood="$2"; shift 2 ;;
    *) shift ;;
  esac
done
sleep 300 </dev/null >/dev/null 2>&1 &
echo "$!" >> "${pids}"
echo "socket=${DS4_RESIDENT_SOCKET:-} bench=${BENCH_WORKER_RESIDENT_SOCKET:-} home=${HOME:-} tmp=${TMPDIR:-} leak=${STUB_SUDO_LOG:-}"
echo "HIDDEN PROMPT TOKENS" >&2
# Past the fsize cap, any write to stderr kills its writer, so the flood runs in
# a child and bash's own "File size limit exceeded" notice goes to /dev/null.
if [[ -n "${flood}" ]]; then
  exec 3>&2; { head -c 70000000 /dev/zero >&3; echo "flood=$?" >> "${flood}"; } 2>/dev/null; exec 3>&-
fi
while IFS= read -r line; do
  [[ "${line}" == quit ]] && break
  echo "ack:${line}"
done
exit "${code}"
EOF
chmod +x "${WORK}/bin/sudo" "${WORK}/bin/pgrep" "${WORK}/adapter"

# Always called in a subshell (coproc, pipeline, background), so `exec` makes the
# wrapper that subshell's own pid -- the pid case 4 signals.
run_wrapper() {
  exec env PATH="${WORK}/bin:${PATH}" \
      SANDBOX_USER=nemosub SANDBOX_ENGINE_BIN="${WORK}/adapter" \
      SANDBOX_ENGINE_LOG_DIR="${WORK}/logs" SANDBOX_ROOT="${WORK}/sbx" SANDBOX_JOB=7-1 \
      DS4_RESIDENT_SOCKET=/sock/resident BENCH_WORKER_RESIDENT_SOCKET=/sock/bench \
      STUB_SUDO_LOG="${WORK}/sudo.log" STUB_SANDBOX_PIDS="${PIDS}" \
      "${WRAPPER}" "$@"
}

all_dead() {
  while read -r p; do kill -0 "${p}" 2>/dev/null && return 1; done < "${PIDS}"
  return 0
}

# --- case 1: interactive stdio round trip, env, stderr, status, kill-all ---
coproc W { run_wrapper --pids "${PIDS}" --exit 7 2> "${WORK}/case1.err"; }
out_fd="${W[0]}"; in_fd="${W[1]}"
IFS= read -r -t 10 first <&"${out_fd}" || first=""
[[ "${first}" == "socket=/sock/resident bench=/sock/bench home=${WORK}/sbx/7-1/adapter-home tmp=${WORK}/sbx/7-1/adapter-home leak=" ]] \
  && ok "case 1: both socket variables and the per-job HOME/TMPDIR reach the adapter, nothing else does" \
  || fail "case 1: adapter saw '${first}'"
echo one >&"${in_fd}"
IFS= read -r -t 10 reply <&"${out_fd}" || reply=""
echo two >&"${in_fd}"
IFS= read -r -t 10 reply2 <&"${out_fd}" || reply2=""
[[ "${reply}" == ack:one && "${reply2}" == ack:two ]] \
  && ok "case 1: each message round-trips before the next is sent (no buffering)" \
  || fail "case 1: round trip got '${reply}' / '${reply2}'"
echo quit >&"${in_fd}"
wait "${W_PID}"; status=$?
[[ "${status}" -eq 7 ]] && ok "case 1: the adapter's exit status 7 is the wrapper's" \
  || fail "case 1: wrapper exited ${status}, expected 7"
grep -Eq -- "^-n -u nemosub --preserve-env=DS4_RESIDENT_SOCKET,BENCH_WORKER_RESIDENT_SOCKET -- prlimit --fsize=67108864 --nproc=4096 --as=17179869184 -- env HOME=[^ ]+/adapter-home TMPDIR=[^ ]+/adapter-home ${WORK}/adapter --pids ${PIDS} --exit 7$" "${WORK}/sudo.log" \
  && ok "case 1: the adapter runs through sudo -n -u nemosub under prlimit fsize/nproc/as, arguments unchanged" \
  || fail "case 1: sudo was not called as expected: $(cat "${WORK}/sudo.log")"
grep -Fxq -- "-n -u nemosub -- kill -KILL -1" "${WORK}/sudo.log" \
  && ok "case 1: every sandbox process is killed after the adapter exits" \
  || fail "case 1: no kill -KILL -1 as nemosub"
all_dead && ok "case 1: the process the adapter left behind is gone" \
  || fail "case 1: the adapter's leftover survived"
if grep -q "HIDDEN PROMPT" "${WORK}/case1.err"; then
  fail "case 1: the adapter's stderr reached the wrapper's stderr (the job log)"
else
  ok "case 1: the adapter's stderr stays off the job log"
fi
logs=("${WORK}"/logs/engine-stderr.*)
if [[ "${#logs[@]}" -eq 1 && -f "${logs[0]}" ]] && grep -q "HIDDEN PROMPT" "${logs[0]}" \
   && [[ "$(stat -c %a "${logs[0]}")" == 600 ]]; then
  ok "case 1: the adapter's stderr is in a 0600 file under SANDBOX_ENGINE_LOG_DIR"
else
  fail "case 1: no 0600 stderr file with the adapter's output: ${logs[*]}"
fi

# --- case 2: piped stdin, clean exit ---------------------------------------
out="$(printf 'a\nquit\n' | run_wrapper --pids "${PIDS}" --exit 0 2>/dev/null)"; status=$?
[[ "${status}" -eq 0 && "$(sed -n 2p <<< "${out}")" == ack:a ]] \
  && ok "case 2: piped stdin reaches the adapter and exit 0 passes through" \
  || fail "case 2: status ${status}, output '${out}'"

# --- case 3: a sandbox process that survives fails the wrapper -------------
err="$(printf 'quit\n' | STUB_PGREP_STUCK=1 run_wrapper --pids "${PIDS}" --exit 0 2>&1 >/dev/null)"; status=$?
[[ "${status}" -ne 0 ]] && grep -q "survived" <<< "${err}" \
  && ok "case 3: a surviving sandbox process fails the wrapper loudly (exit ${status})" \
  || fail "case 3: status ${status}, stderr '${err}'"

# --- case 4: SIGTERM is forwarded to the adapter ---------------------------
mkfifo "${WORK}/in"
run_wrapper --pids "${PIDS}" --trap < "${WORK}/in" > "${WORK}/case4.out" 2>/dev/null &
wpid=$!
exec 8> "${WORK}/in"
for _ in $(seq 50); do [[ -s "${WORK}/case4.out" ]] && break; sleep 0.1; done
kill -TERM "${wpid}"
wait "${wpid}"; status=$?
exec 8>&-
[[ "${status}" -eq 42 ]] && grep -qh "adapter got TERM" "${WORK}"/logs/engine-stderr.* \
  && ok "case 4: SIGTERM reaches the adapter and its status comes back" \
  || fail "case 4: wrapper exited ${status}"
all_dead && ok "case 4: nothing left behind after a signalled run" \
  || fail "case 4: a sandbox process survived the signalled run"

# --- case 5: no adapter, no run --------------------------------------------
: > "${WORK}/sudo.log"
env PATH="${WORK}/bin:${PATH}" SANDBOX_USER=nemosub \
  SANDBOX_ENGINE_BIN="${WORK}/nope" STUB_SUDO_LOG="${WORK}/sudo.log" \
  "${WRAPPER}" </dev/null >/dev/null 2>&1; status=$?
[[ "${status}" -ne 0 && ! -s "${WORK}/sudo.log" ]] \
  && ok "case 5: a missing adapter is refused before sudo runs" \
  || fail "case 5: status ${status}, sudo log '$(cat "${WORK}/sudo.log")'"

# --- case 6: no per-job adapter home, no run --------------------------------
: > "${WORK}/sudo.log"
env PATH="${WORK}/bin:${PATH}" SANDBOX_USER=nemosub SANDBOX_ENGINE_BIN="${WORK}/adapter" \
  SANDBOX_ROOT="${WORK}/sbx" SANDBOX_JOB=nope STUB_SUDO_LOG="${WORK}/sudo.log" \
  "${WRAPPER}" </dev/null >/dev/null 2>&1; status=$?
[[ "${status}" -ne 0 && ! -s "${WORK}/sudo.log" ]] \
  && ok "case 6: a missing adapter-home is refused before sudo runs" \
  || fail "case 6: status ${status}, sudo log '$(cat "${WORK}/sudo.log")'"

# --- case 7: the staged shim -> wrapper -> adapter chain, cleared env ---------
# tools/stage-cuda-engine.sh with SANDBOX_USER set puts the adapter at
# <staged>.sandboxed and a shim at the staged path that bakes in every variable
# the wrapper reads. benchd spawns the shim with a CLEARED environment, so it
# runs under `env -i` here. bin7 holds stubs with their own settings baked in,
# since env -i clears STUB_*; the sandbox user is this user (staging checks
# the adapter is owned by it).
ME="$(id -un)"
REPO="$(cd "${WORK}" && pwd -P)/repo"
mkdir -p "${REPO}/tools" "${WORK}/bin7"
cp "${ROOT_DIR}/tools/stage-cuda-engine.sh" "${WRAPPER}" "${REPO}/tools/"
cp "${WORK}/adapter" "${WORK}/built-adapter"
mkdir -p "${WORK}/sbx/7-1/adapter-home"
for t in sudo pgrep; do
  printf '#!/usr/bin/env bash\nexport STUB_USER=%q STUB_SUDO_LOG=%q STUB_SANDBOX_PIDS=%q\nexec %q "$@"\n' \
    "${ME}" "${WORK}/sudo.log" "${PIDS}" "${WORK}/bin/${t}" > "${WORK}/bin7/${t}"
done
chmod +x "${WORK}/bin7/sudo" "${WORK}/bin7/pgrep"
SANDBOX_USER="${ME}" SANDBOX_ROOT="${WORK}/sbx" SANDBOX_JOB=7-1 RUNNER_TEMP="${WORK}/logs" \
  SANDBOX_ENGINE_PATH="${WORK}/bin7:/usr/bin:/bin" CUDA_ENGINE_EXECUTABLE="${WORK}/built-adapter" \
  "${REPO}/tools/stage-cuda-engine.sh" >/dev/null || fail "case 7: staging failed"
SHIM="${REPO}/.build/release/mlxfast-runtime-worker"
: > "${WORK}/sudo.log"
rm -f "${WORK}"/logs/engine-stderr.*
out="$(printf 'x\ny\nquit\n' | env -i DS4_RESIDENT_SOCKET=/sock/resident \
  "${SHIM}" --pids "${PIDS}" --exit 5 2>/dev/null)"; status=$?
[[ "${status}" -eq 5 && "$(sed -n 2,3p <<< "${out}")" == $'ack:x\nack:y' ]] \
  && ok "case 7: under env -i the shim passes stdio through the wrapper and returns the adapter's status" \
  || fail "case 7: status ${status}, output '${out}'"
[[ "$(sed -n 1p <<< "${out}")" == "socket=/sock/resident bench= home=${WORK}/sbx/7-1/adapter-home tmp=${WORK}/sbx/7-1/adapter-home leak=" ]] \
  && ok "case 7: the baked SANDBOX_ROOT/JOB give the adapter its per-job home; benchd's socket still reaches it" \
  || fail "case 7: adapter saw '$(sed -n 1p <<< "${out}")'"
grep -Eq -- "^-n -u ${ME} --preserve-env=[^ ]+ -- prlimit --fsize=67108864 --nproc=4096 --as=17179869184 -- env [^ ]+ [^ ]+ ${SHIM}\\.sandboxed --pids ${PIDS} --exit 5$" "${WORK}/sudo.log" \
  && ok "case 7: the shim runs the .sandboxed adapter as the baked SANDBOX_USER" \
  || fail "case 7: sudo log '$(cat "${WORK}/sudo.log")'"
grep -Fxq -- "-n -u ${ME} -- kill -KILL -1" "${WORK}/sudo.log" && all_dead \
  && ok "case 7: every sandbox process is killed and none is left" \
  || fail "case 7: no kill-all, or a sandbox process survived"
logs=("${WORK}"/logs/engine-stderr.*)
[[ -f "${logs[0]}" ]] && grep -q "HIDDEN PROMPT" "${logs[0]}" \
  && ok "case 7: the adapter's stderr lands in the baked log dir" \
  || fail "case 7: no stderr file in ${WORK}/logs"

# --- case 9: the fsize cap bounds the stderr file; stdio still flows -----------
rm -f "${WORK}"/logs/engine-stderr.*
out="$(printf 'z\nquit\n' | run_wrapper --pids "${PIDS}" --flood "${WORK}/flood" --exit 0 2>/dev/null)"; status=$?
logs=("${WORK}"/logs/engine-stderr.*)
size="$(stat -c %s "${logs[0]}" 2>/dev/null || echo missing)"
[[ "${size}" == 67108864 && "$(cat "${WORK}/flood")" == flood=153 ]] \
  && ok "case 9: a 70 MB stderr flood stops at 64 MiB (SIGXFSZ to the writer)" \
  || fail "case 9: stderr file size ${size}, flood '$(cat "${WORK}/flood" 2>/dev/null)'"
[[ "${status}" -eq 0 && "$(sed -n 2p <<< "${out}")" == ack:z ]] \
  && ok "case 9: the adapter itself keeps serving stdio after the flood" \
  || fail "case 9: status ${status}, output '${out}'"
rm -f "${WORK}"/logs/engine-stderr.*

# --- case 8: no log dir, no run ---------------------------------------------
: > "${WORK}/sudo.log"
env PATH="${WORK}/bin:${PATH}" SANDBOX_USER=nemosub SANDBOX_ENGINE_BIN="${WORK}/adapter" \
  SANDBOX_ROOT="${WORK}/sbx" SANDBOX_JOB=7-1 STUB_SUDO_LOG="${WORK}/sudo.log" \
  "${WRAPPER}" </dev/null >/dev/null 2>&1; status=$?
[[ "${status}" -ne 0 && ! -s "${WORK}/sudo.log" ]] \
  && ok "case 8: an unset SANDBOX_ENGINE_LOG_DIR is refused before sudo runs" \
  || fail "case 8: status ${status}, sudo log '$(cat "${WORK}/sudo.log")'"

if [[ "${failures}" -eq 0 ]]; then
  echo "PASS: test-run-engine-sandboxed.sh"
  exit 0
fi
echo "FAIL: ${failures} failure(s)" >&2
exit 1

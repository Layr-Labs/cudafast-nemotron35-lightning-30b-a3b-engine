#!/usr/bin/env bash
# test-box-runner.sh -- dry-run test suite for tools/box-runner/.
#
# Exercises supervisor.sh's spawn/stop/status logic and lib/pidtree.sh's
# kill-tree primitive against MOCK processes only (mock-run.sh,
# mock-orphan-child.sh, mock-gh-app-mint.sh, all in this directory) -- no
# GitHub network access, no GitHub App credential, no root, no real
# actions-runner binary. RUNNER_SU_ENABLED=0 throughout (see supervisor.sh's
# header: that is the ONLY thing that differs from production here -- the
# spawn/track/kill mechanism under test is identical either way, since it
# never depends on which user the job ran as).
#
# REQUIRED regression coverage: section 1b proves the kill-tree mechanism actually reaps a
# DETACHED grandchild of the tracked job root -- the orphaned-session bug that
# lib/pidtree.sh and supervisor.sh's `stop` entrypoint exist to fix. Section
# 1c is the negative control: it proves the bug is real by showing that
# killing ONLY the tracked root pid (the naive/original behavior) does NOT
# reap the grandchild, so 1b is not a vacuous pass. Section 2 repeats the
# same proof end-to-end through supervisor.sh's actual `run`/`stop`/`status`
# entrypoints, including calling `stop` from a SEPARATE process invocation
# than the one running the job loop -- the recovery scenario itself
# (an operator, or a restarted daemon, reaping a tree a now-dead process was
# tracking).
#
# Usage: tools/box-runner/test/test-box-runner.sh [-v]
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)"
BOX_RUNNER_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null && pwd -P)"
VERBOSE=0
[[ "${1:-}" == "-v" ]] && VERBOSE=1

# shellcheck source=tools/box-runner/lib/pidtree.sh
. "${BOX_RUNNER_DIR}/lib/pidtree.sh"

# This floor is the only thing that notices a section silently stopping (an
# early `exit`, a deleted block) under `set -uo pipefail` without `-e`, where a
# truncated suite otherwise reports "0 failed" and looks green.
EXPECTED_MIN_ASSERTIONS=126

WORK="$(mktemp -d "${TMPDIR:-/tmp}/box-runner-test.XXXXXX")"
# shellcheck disable=SC2329  # invoked indirectly via `trap cleanup EXIT` below
cleanup() {
  # Belt and suspenders: every test below is expected to clean up its own
  # processes: this just makes sure nothing outlives the suite even if one
  # of them fails partway through.
  local pf
  for pf in "${WORK}"/*/job.pid "${WORK}"/*/mock.pid "${WORK}"/*/orphan.pid "${WORK}"/sup-state/job.pid; do
    [ -f "${pf}" ] || continue
    kill -KILL "$(cat "${pf}" 2>/dev/null)" 2>/dev/null || true
  done
  rm -rf "${WORK}"
}
trap cleanup EXIT

PASSED=0
FAILED=0
FAILURES=()

pass() { PASSED=$((PASSED + 1)); [ "${VERBOSE}" -eq 1 ] && printf 'PASS  %s\n' "$1"; }
fail_case() {
  FAILED=$((FAILED + 1))
  FAILURES+=("$1")
  printf 'FAIL  %s\n' "$1"
}

assert_true() {
  local name="$1"; shift
  if "$@"; then pass "${name}"; else fail_case "${name} (expected success: $*)"; fi
}

assert_false() {
  local name="$1"; shift
  if "$@"; then fail_case "${name} (expected failure: $*)"; else pass "${name}"; fi
}

assert_eq() {
  local name="$1" want="$2" got="$3"
  if [ "${got}" = "${want}" ]; then
    pass "${name}"
  else
    fail_case "${name} (want '${want}', got '${got}')"
  fi
}

# assert_exit NAME WANT_STATUS WANT_SUBSTRING CMD...
assert_exit() {
  local name="$1" want="$2" pattern="$3"
  shift 3
  local out status
  out="$("$@" 2>&1)"
  status=$?
  [ "${VERBOSE}" -eq 1 ] && printf -- '--- %s (exit %d)\n%s\n' "${name}" "${status}" "${out}"
  if [ "${status}" != "${want}" ]; then
    fail_case "${name} (exit ${status}, wanted ${want}) :: ${out}"
    return
  fi
  if [ -n "${pattern}" ]; then
    case "${out}" in
      *"${pattern}"*) : ;;
      *) fail_case "${name} (missing '${pattern}' in output) :: ${out}"; return ;;
    esac
  fi
  pass "${name}"
}

# The four helpers below are only ever invoked indirectly, as the CMD... of
# assert_true/assert_false ("$@" further down each), so shellcheck cannot
# see the call site; SC2329 ("never invoked") is a false positive on all of
# them for that reason.
# shellcheck disable=SC2329
pid_alive() { kill -0 "$1" 2>/dev/null; }
# shellcheck disable=SC2329
pid_dead() { ! kill -0 "$1" 2>/dev/null; }
# shellcheck disable=SC2329
contains() { case "$1" in *"$2"*) return 0 ;; *) return 1 ;; esac; }
# list_has LIST NEEDLE -- LIST is newline-separated; true iff one line
# equals NEEDLE exactly (used for pid-list membership, where a substring
# match could false-positive on e.g. pid 12 matching pid 123).
# shellcheck disable=SC2329
list_has() { printf '%s\n' "$1" | grep -qx -- "$2"; }

wait_for_file() {
  local path="$1" timeout="${2:-5}" waited=0
  while [ ! -s "${path}" ]; do
    waited=$((waited + 1))
    [ "${waited}" -gt $((timeout * 10)) ] && return 1
    sleep 0.1
  done
  return 0
}

# =========================================================================
# Section 1: lib/pidtree.sh primitives, directly, against mock-run.sh
# =========================================================================

# start_mock_tree STATE_DIR -- starts mock-run.sh (which spawns a detached
# mock-orphan-child.sh grandchild) and blocks until both have recorded their
# own pids, printing mock-run.sh's pid on stdout.
start_mock_tree() {
  local state_dir="$1"
  mkdir -p "${state_dir}"
  "${SCRIPT_DIR}/mock-run.sh" "${state_dir}" \
    > "${state_dir}/mock-run.log" 2>&1 &
  wait_for_file "${state_dir}/mock.pid" 5 || { echo "mock.pid never appeared" >&2; return 1; }
  wait_for_file "${state_dir}/orphan.pid" 5 || { echo "orphan.pid never appeared" >&2; return 1; }
  cat "${state_dir}/mock.pid"
}

# --- 1a. pidtree_collect finds root + orphan while alive -------------------
t1_dir="${WORK}/t1"
root_pid="$(start_mock_tree "${t1_dir}")"
orphan_pid="$(cat "${t1_dir}/orphan.pid")"
collected="$(pidtree_collect "${root_pid}")"
assert_true "pidtree_collect includes root pid" list_has "${collected}" "${root_pid}"
assert_true "pidtree_collect includes orphan grandchild pid" list_has "${collected}" "${orphan_pid}"
pidtree_kill_tree "${root_pid}" 5 >/dev/null 2>&1 || true
nonexistent_collected="$(pidtree_collect 999999)"
assert_eq "pidtree_collect: nonexistent pid returns nothing" "" "${nonexistent_collected}"

# --- 1b. THE REQUIRED REGRESSION TEST: kill-tree reaps a detached orphan ---
t2_dir="${WORK}/t2"
root_pid="$(start_mock_tree "${t2_dir}")"
orphan_pid="$(cat "${t2_dir}/orphan.pid")"
assert_true "orphan-kill setup: root alive before stop" pid_alive "${root_pid}"
assert_true "orphan-kill setup: orphan alive before stop" pid_alive "${orphan_pid}"
assert_false "orphan-kill setup: orphan is NOT a direct child of the test shell (pgrep -P misses it from here)" \
  bash -c "pgrep -P \$\$ | grep -qx '${orphan_pid}'"

pidtree_kill_tree "${root_pid}" 5
kill_tree_status=$?
assert_eq "orphan-kill: pidtree_kill_tree reports success" "0" "${kill_tree_status}"
sleep 0.3
assert_true "orphan-kill: root pid dead after kill_tree" pid_dead "${root_pid}"
assert_true "orphan-kill: DETACHED ORPHAN pid dead after kill_tree" pid_dead "${orphan_pid}"

# --- 1c. Negative control: naive single-pid kill leaves the orphan alive --
#     (proves 1b is not vacuous -- this is what the orphan bug looks like:
#     killing only the tracked pid, or relying on a process-group signal
#     that never reaches a re-sessioned descendant, leaves it running.)
t3_dir="${WORK}/t3"
root_pid="$(start_mock_tree "${t3_dir}")"
orphan_pid="$(cat "${t3_dir}/orphan.pid")"
kill -KILL "${root_pid}" 2>/dev/null
sleep 0.3
assert_true "negative control: naive kill removes only the root" pid_dead "${root_pid}"
assert_true "negative control: naive kill LEAVES the detached orphan alive (the bug)" pid_alive "${orphan_pid}"
kill -KILL "${orphan_pid}" 2>/dev/null || true # clean up what the negative control deliberately left running

# =========================================================================
# Section 2: supervisor.sh end-to-end (spawn / status / stop), test mode
# =========================================================================

run_dir="${WORK}/runner-dir"
mkdir -p "${run_dir}"
# Real copies, not symlinks: mock-run.sh locates mock-orphan-child.sh next
# to itself (mirroring how a real actions-runner install keeps its helper
# scripts beside run.sh), so both must live in RUNNER_DIR together.
cp "${SCRIPT_DIR}/mock-run.sh" "${run_dir}/run.sh"
cp "${SCRIPT_DIR}/mock-orphan-child.sh" "${run_dir}/mock-orphan-child.sh"
chmod +x "${run_dir}/run.sh" "${run_dir}/mock-orphan-child.sh"

e2e_state="${WORK}/e2e-state"
sup_log_dir="${WORK}/sup-log"
sup_state_dir="${WORK}/sup-state"
mkdir -p "${e2e_state}"

run_supervisor() {
  env \
    RUNNER_SU_ENABLED=0 \
    RUNNER_DIR="${run_dir}" \
    RUNNER_LABELS="nemotron35-box" \
    GH_APP_MINT="${SCRIPT_DIR}/mock-gh-app-mint.sh" \
    LOG_DIR="${sup_log_dir}" \
    STATE_DIR="${sup_state_dir}" \
    MOCK_STATE_DIR="${e2e_state}" \
    PAUSE_BETWEEN_JOBS=1 \
    BOX_RUNNER_MAX_ITERATIONS=1 \
    BOX_RUNNER_CONFIG=/nonexistent \
    "${BOX_RUNNER_DIR}/supervisor.sh" "$@"
}

# --- 2a. status is idle before anything has run -----------------------------
out="$(run_supervisor status)"
assert_eq "status: idle before any job" "idle" "${out}"

# --- 2b. spawn a job (background, single-iteration loop) --------------------
run_supervisor run > "${WORK}/supervisor-run.log" 2>&1 &
sup_pid=$!
wait_for_file "${sup_state_dir}/job.pid" 5
assert_true "supervisor: job.pid appears" test -s "${sup_state_dir}/job.pid"
wait_for_file "${e2e_state}/mock.pid" 5
wait_for_file "${e2e_state}/orphan.pid" 5
job_pid="$(cat "${sup_state_dir}/job.pid")"
mock_pid="$(cat "${e2e_state}/mock.pid")"
orphan_pid="$(cat "${e2e_state}/orphan.pid")"
# The tracked root pid is whatever `exec` ultimately replaced spawn_job's
# background subshell with -- mock-run.sh itself when no timeout(1)/
# gtimeout(1) is on PATH, or a `timeout` WRAPPER process (a distinct pid)
# ahead of it when one is (as it is on this dev machine, via Homebrew
# coreutils). Both are correct: what matters is that the tracked pid's tree
# includes mock-run.sh, which is what the fix actually depends on.
job_tree="$(pidtree_collect "${job_pid}")"
assert_true "supervisor: tracked root's tree includes mock-run.sh's pid" list_has "${job_tree}" "${mock_pid}"
assert_true "supervisor: mock-run.sh alive while job in progress" pid_alive "${mock_pid}"
assert_true "supervisor: detached orphan alive while job in progress" pid_alive "${orphan_pid}"

# --- 2c. status while running lists the tree --------------------------------
status_out="$(run_supervisor status)"
# shellcheck disable=SC2016  # deliberate: $0 must expand inside the nested
# bash -c (bound there to status_out, passed as its NAME argument), not here.
assert_true "status while running: reports 'running'" bash -c 'case "$0" in running:*) exit 0;; *) exit 1;; esac' "${status_out}"
assert_true "status while running: lists the mock-run.sh pid" contains "${status_out}" "${mock_pid}"

# --- 2d. stop, called as a SEPARATE process invocation (the standalone
#     entrypoint -- the recovery case: an operator, or a restarted
#     `run`, reaping a tree a DIFFERENT process instance is tracking) -------
stop_out="$(run_supervisor stop)"
assert_true "stop: reports stopped" contains "${stop_out}" "stopped"
sleep 0.3
assert_true "stop: mock-run.sh pid dead" pid_dead "${mock_pid}"
assert_true "stop: DETACHED ORPHAN pid dead (kill-tree via standalone stop)" pid_dead "${orphan_pid}"
assert_false "stop: job.pid file removed" test -f "${sup_state_dir}/job.pid"

# --- 2e. status is idle again after stop ------------------------------------
final_status="$(run_supervisor status)"
assert_eq "status: idle again after stop" "idle" "${final_status}"

# Reap the backgrounded `run` loop itself. Its single job was already killed
# by `stop` above; BOX_RUNNER_MAX_ITERATIONS=1 makes the loop exit on its
# own within PAUSE_BETWEEN_JOBS=1s, so this is a bounded wait, not a hang.
kill "${sup_pid}" 2>/dev/null || true
wait "${sup_pid}" 2>/dev/null || true

# =========================================================================
# Section 3: fail-closed configuration (supervisor.sh)
# =========================================================================

assert_exit "supervisor: refuses with no RUNNER_DIR" 78 "RUNNER_DIR is required" \
  env RUNNER_SU_ENABLED=0 RUNNER_LABELS=x GH_APP_MINT="${SCRIPT_DIR}/mock-gh-app-mint.sh" \
      LOG_DIR="${WORK}/x1-log" STATE_DIR="${WORK}/x1-state" BOX_RUNNER_MAX_ITERATIONS=1 \
      BOX_RUNNER_CONFIG=/nonexistent "${BOX_RUNNER_DIR}/supervisor.sh" run

assert_exit "supervisor: refuses with missing run.sh" 78 "run.sh" \
  env RUNNER_SU_ENABLED=0 RUNNER_DIR="${WORK}/x2-empty" RUNNER_LABELS=x \
      GH_APP_MINT="${SCRIPT_DIR}/mock-gh-app-mint.sh" LOG_DIR="${WORK}/x2-log" \
      STATE_DIR="${WORK}/x2-state" BOX_RUNNER_MAX_ITERATIONS=1 BOX_RUNNER_CONFIG=/nonexistent \
      "${BOX_RUNNER_DIR}/supervisor.sh" run

assert_exit "supervisor: refuses with no RUNNER_LABELS" 78 "RUNNER_LABELS is required" \
  env RUNNER_SU_ENABLED=0 RUNNER_DIR="${run_dir}" \
      GH_APP_MINT="${SCRIPT_DIR}/mock-gh-app-mint.sh" LOG_DIR="${WORK}/x3-log" \
      STATE_DIR="${WORK}/x3-state" BOX_RUNNER_MAX_ITERATIONS=1 BOX_RUNNER_CONFIG=/nonexistent \
      "${BOX_RUNNER_DIR}/supervisor.sh" run

assert_exit "supervisor: refuses with missing gh-app-mint helper" 78 "gh-app mint helper missing" \
  env RUNNER_SU_ENABLED=0 RUNNER_DIR="${run_dir}" RUNNER_LABELS=x \
      GH_APP_MINT="${WORK}/does-not-exist.sh" LOG_DIR="${WORK}/x4-log" \
      STATE_DIR="${WORK}/x4-state" BOX_RUNNER_MAX_ITERATIONS=1 BOX_RUNNER_CONFIG=/nonexistent \
      "${BOX_RUNNER_DIR}/supervisor.sh" run

# =========================================================================
# Section 4: fail-closed configuration (gh-app-mint.sh)
# =========================================================================

assert_exit "gh-app-mint: refuses with no GH_APP_ID" 78 "GH_APP_ID is required" \
  env GH_APP_KEY="${WORK}/fake-key.pem" GH_RUNNER_REPO="owner/repo" \
      "${BOX_RUNNER_DIR}/gh-app-mint.sh" installation-id

echo "fake" > "${WORK}/fake-key.pem"
assert_exit "gh-app-mint: refuses with no GH_RUNNER_REPO" 78 "GH_RUNNER_REPO is required" \
  env GH_APP_ID=1 GH_APP_KEY="${WORK}/fake-key.pem" \
      "${BOX_RUNNER_DIR}/gh-app-mint.sh" installation-id

assert_exit "gh-app-mint: refuses with missing GH_APP_KEY file" 78 "GH_APP_KEY does not exist" \
  env GH_APP_ID=1 GH_APP_KEY="${WORK}/no-such-key.pem" GH_RUNNER_REPO="owner/repo" \
      "${BOX_RUNNER_DIR}/gh-app-mint.sh" installation-id

assert_exit "gh-app-mint: usage error on unknown subcommand" 64 "usage" \
  env GH_APP_ID=1 GH_APP_KEY="${WORK}/fake-key.pem" GH_RUNNER_REPO="owner/repo" \
      "${BOX_RUNNER_DIR}/gh-app-mint.sh" bogus-subcommand

# =========================================================================
# Section 5: install.sh pure helpers (isolated subshell: install.sh's own
# `set -euo pipefail` must never leak into this suite's shell, so each call
# below runs install.sh in a fresh `bash -c`, not via `source` here).
# =========================================================================

install_call() {
  bash -c '
    set -uo pipefail
    BOX_RUNNER_INSTALL_SOURCE_ONLY=1
    # shellcheck disable=SC1091
    . "$1/install.sh"
    shift
    "$@"
  ' _ "${BOX_RUNNER_DIR}" "$@"
}

assert_eq "install.sh: pinned sha256 for linux-arm64 is 64 hex chars" "64" \
  "$(install_call pinned_sha256 linux-arm64 | wc -c | tr -d ' ')"
assert_eq "install.sh: pinned sha256 for linux-x64 is 64 hex chars" "64" \
  "$(install_call pinned_sha256 linux-x64 | wc -c | tr -d ' ')"
assert_false "install.sh: pinned_sha256 refuses an unknown platform tag" \
  install_call pinned_sha256 osx-arm64

echo -n "box-runner" > "${WORK}/checksum-fixture.txt"
known_sha="$(shasum -a 256 "${WORK}/checksum-fixture.txt" 2>/dev/null | awk '{print $1}')"
assert_eq "install.sh: compute_sha256 matches shasum" "${known_sha}" \
  "$(install_call compute_sha256 "${WORK}/checksum-fixture.txt")"

# =========================================================================
# Section 6: install.sh FIRST INSTALL, end to end through main()
#
# Section 5 above only reaches install.sh's pure helpers: it sources the
# script with BOX_RUNNER_INSTALL_SOURCE_ONLY=1 and never runs main(), so
# nothing there can see a failure in the install path itself. That blind
# spot shipped a real one. install_actions_runner armed
# `trap 'rm -rf "${tmp_dir}"' RETURN` against its own `local tmp_dir`; bash
# RETURN traps are global, not function-scoped, so the trap outlived the
# function and fired a second time when main() returned, where `tmp_dir` is
# unbound -- fatal under install.sh's `set -euo pipefail`. Every first
# install on a fresh box therefore exited 1 with "tmp_dir: unbound variable"
# AFTER all four install steps had succeeded. Re-running took the idempotent
# early `return 0` that sits above the trap, so the second run exited 0 and
# made the first look like a fluke.
#
# So this section drives the WHOLE first-install path through main(), with
# the network and the tarball stubbed out (the same source-then-override
# pattern section 5 uses), and asserts the thing the bug got wrong: a
# successful first install exits 0. It runs against a STAGED COPY of
# install.sh so that the files main() writes -- box-runner.env, the rendered
# systemd unit -- land in this suite's temp dir instead of in the repo.
# =========================================================================

install_stage="${WORK}/install-stage"
mkdir -p "${install_stage}"
cp "${BOX_RUNNER_DIR}/install.sh" \
   "${BOX_RUNNER_DIR}/box-runner.env.example" \
   "${BOX_RUNNER_DIR}/nemotron35-box-runner.service.template" \
   "${install_stage}/"

install_runner_dir="${WORK}/install-runner-dir"
install_tmpdir="${WORK}/install-tmp"
mkdir -p "${install_tmpdir}"

# run_install_main -- install.sh's main(), start to finish, as a first
# install on a box that has none: platform forced to the ranked linux-arm64 so
# the result does not depend on the test host, curl(1) replaced by a stub
# that just materializes the tarball path, tar(1) replaced by one that lays
# down the single file the idempotency check looks for, and compute_sha256
# answering with the script's OWN pin so the verify step passes (a stub
# cannot forge bytes hashing to the real pin, and the pin itself is already
# under test in section 5). Everything else -- the trap, the directory
# creation, the env seeding, the unit render, main()'s return -- is the
# real code path.
# shellcheck disable=SC2329  # invoked indirectly, as assert_exit's CMD...
run_install_main() {
  # shellcheck disable=SC2016  # deliberate: every expansion in the nested
  # bash -c body below must happen THERE, against install.sh's own
  # variables (RUNNER_DIR, the stubs' arguments), not out here.
  env \
    RUNNER_DIR="${install_runner_dir}" \
    LOG_DIR="${WORK}/install-log" \
    STATE_DIR="${WORK}/install-state" \
    TMPDIR="${install_tmpdir}" \
    bash -c '
      set -uo pipefail
      BOX_RUNNER_INSTALL_SOURCE_ONLY=1
      # shellcheck disable=SC1091
      . "$1/install.sh"
      platform_tag() { printf "linux-arm64"; }
      curl() {
        local arg out="" prev=""
        for arg in "$@"; do
          [ "${prev}" = "-o" ] && out="${arg}"
          prev="${arg}"
        done
        [ -n "${out}" ] || return 1
        printf "stub-tarball" > "${out}"
      }
      compute_sha256() { pinned_sha256 linux-arm64; }
      tar() {
        mkdir -p "${RUNNER_DIR}"
        printf "#!/bin/sh\n" > "${RUNNER_DIR}/run.sh"
        chmod +x "${RUNNER_DIR}/run.sh"
      }
      main
    ' _ "${install_stage}"
}

# --- 6a. THE REGRESSION ASSERTION: a successful first install exits 0 ------
assert_exit "install.sh: first install through main() exits 0" 0 "done." run_install_main

# --- 6b. ...and it exited 0 because it did the work, not because it bailed -
# shellcheck disable=SC2016  # deliberate: ACTIONS_RUNNER_VERSION must expand
# inside install_call's shell, where install.sh has been sourced, not here.
assert_eq "install.sh: first install records the pinned runner version" \
  "$(install_call eval 'printf %s "${ACTIONS_RUNNER_VERSION}"')" \
  "$(cat "${install_runner_dir}/.box-runner-installed-version" 2>/dev/null)"
assert_true "install.sh: first install seeds box-runner.env" \
  test -f "${install_stage}/box-runner.env"
assert_true "install.sh: first install renders the systemd unit" \
  test -f "${install_stage}/nemotron35-box-runner.service"
install_svc="${install_stage}/nemotron35-box-runner.service"
assert_true "install: box-runner unit requires the egress block" \
  grep -qx 'Requires=nemotron35-sandbox-egress.service' "${install_svc}"
assert_true "install: box-runner unit starts after egress block" \
  grep -q '^After=.*nemotron35-sandbox-egress.service' "${install_svc}"
assert_true "install: box-runner refuses start without iptables egress" \
  grep -q '^ExecStartPre=/usr/sbin/iptables -C OUTPUT -m owner --uid-owner nemosub -j REJECT' "${install_svc}"
assert_true "install: box-runner refuses start without ip6tables egress" \
  grep -q '^ExecStartPre=/usr/sbin/ip6tables -C OUTPUT -m owner --uid-owner nemosub -j REJECT' "${install_svc}"

# --- 6c. the cleanup the trap exists for still happened, exactly once ------
leftover_tmp="$(find "${install_tmpdir}" -maxdepth 1 -name 'box-runner-install.*' 2>/dev/null | wc -l | tr -d ' ')"
assert_eq "install.sh: first install leaves no temp download dir behind" "0" "${leftover_tmp}"

# --- 6d. the re-run path, which is what masked the bug ---------------------
#     It returns before the trap is ever armed, so it exited 0 even when the
#     first install did not: green here is NOT evidence 6a is green.
assert_exit "install.sh: idempotent re-run is a no-op and exits 0" 0 "idempotent no-op" run_install_main

# --- BOX_RUNNER_APP_ENV reaches the mint helper -----------------------------
# box-runner.env is SOURCED, so a plain `BOX_RUNNER_APP_ENV=...` line is not
# exported, and gh-app-mint.sh (a child process) never saw it: every mint failed
# with "GH_APP_ID is required" on the first real box. The supervisor exports it.
app_cfg="${WORK}/app-cfg.env"
printf 'BOX_RUNNER_APP_ENV=/etc/example/app.env\nRUNNER_LABELS=x\n' > "${app_cfg}"
assert_true "supervisor: exports BOX_RUNNER_APP_ENV from a sourced config" \
  bash -c "BOX_RUNNER_CONFIG='${app_cfg}' RUNNER_SU_ENABLED=0 LOG_DIR='${WORK}/app-log' STATE_DIR='${WORK}/app-state' \
    bash -x '${BOX_RUNNER_DIR}/supervisor.sh' status 2>&1 | grep -q 'export BOX_RUNNER_APP_ENV'"

# --- idle recycle kills only a runner with no job ---------------------------
# Regression: `timeout 3600 ./run.sh` once killed every job at 60 minutes.
recycle_case() {  # $1 = MOCK_JOB_STARTED value; prints the logged exit status
  local st="${WORK}/recycle-$1"; rm -rf "${st}"; mkdir -p "${st}"
  env RUNNER_SU_ENABLED=0 RUNNER_DIR="${run_dir}" RUNNER_LABELS=nemotron35-box \
    GH_APP_MINT="${SCRIPT_DIR}/mock-gh-app-mint.sh" LOG_DIR="${st}/log" STATE_DIR="${st}/state" \
    MOCK_STATE_DIR="${st}/mock" MOCK_JOB_STARTED="$1" IDLE_RECYCLE_SECONDS=5 \
    PAUSE_BETWEEN_JOBS=1 BOX_RUNNER_MAX_ITERATIONS=1 BOX_RUNNER_CONFIG=/nonexistent \
    "${BOX_RUNNER_DIR}/supervisor.sh" run > "${st}/run.log" 2>&1 &
  local sp=$!
  sleep 16
  grep -o 'runner exited status=[0-9]*' "${st}/log/supervisor.log" 2>/dev/null | tail -1
  env RUNNER_SU_ENABLED=0 RUNNER_DIR="${run_dir}" LOG_DIR="${st}/log" STATE_DIR="${st}/state" \
    RUNNER_LABELS=nemotron35-box GH_APP_MINT="${SCRIPT_DIR}/mock-gh-app-mint.sh" \
    BOX_RUNNER_CONFIG=/nonexistent "${BOX_RUNNER_DIR}/supervisor.sh" stop >/dev/null 2>&1
  kill "${sp}" 2>/dev/null; wait "${sp}" 2>/dev/null
  [ -f "${st}/mock/orphan.pid" ] && kill "$(cat "${st}/mock/orphan.pid")" 2>/dev/null
  return 0
}
assert_eq "idle recycle: no job -> killed with 124" "runner exited status=124" "$(recycle_case '')"
assert_eq "idle recycle: running job is NOT killed" "" "$(recycle_case 1)"

# --- stop removes stray serve-up containers, even when idle -----------------
fake_bin="${WORK}/fake-docker-bin"; mkdir -p "${fake_bin}"
cat > "${fake_bin}/docker" <<'FAKE'
#!/bin/sh
echo "$*" >> "${FAKE_DOCKER_LOG}"
case "$1" in ps) echo abc123 ;; esac
FAKE
chmod +x "${fake_bin}/docker"
: > "${WORK}/fake-docker.log"
env PATH="${fake_bin}:${PATH}" FAKE_DOCKER_LOG="${WORK}/fake-docker.log" RUNNER_SU_ENABLED=0 \
  RUNNER_DIR="${run_dir}" RUNNER_LABELS=nemotron35-box GH_APP_MINT="${SCRIPT_DIR}/mock-gh-app-mint.sh" \
  LOG_DIR="${WORK}/stray-log" STATE_DIR="${WORK}/stray-state" BOX_RUNNER_CONFIG=/nonexistent \
  "${BOX_RUNNER_DIR}/supervisor.sh" stop >/dev/null 2>&1
assert_true "stop (idle): removes stray serve-up container" grep -q '^rm -f abc123' "${WORK}/fake-docker.log"

# --- stable per-box runner name ---------------------------------------------
# The ranked preflight requires calibration.box == RUNNER_NAME, so the minted
# name must be the SAME on every job: <prefix>-<host>, never a timestamped name.
stable_name="nemotron35-box-$(hostname -s)"
assert_true "supervisor: minted the stable per-box name ${stable_name}" \
  grep -q "minted JIT config for ${stable_name} " "${sup_log_dir}/supervisor.log"
assert_true "supervisor: no timestamped runner name is minted" \
  bash -c "! grep -qE 'minted JIT config for nemotron35-box-[0-9]{14}-' '${sup_log_dir}/supervisor.log'"

# =========================================================================
# 7. setup-sandbox-user.sh: egress migration, resolver ACL unit, cron/at deny,
#    traverse ACLs, and the new --check items (docs/sandbox-design.md 6a).
# =========================================================================
# The script runs from a copy whose absolute /etc, /var, /opt/rust and /run
# paths point into a scratch tree; id, getent, sudo, runuser, iptables and the
# other root tools are stubs, so nothing on this machine is touched.
sbx="${WORK}/sandbox-setup"
FS="${sbx}/fs"; SBIN="${sbx}/bin"
mkdir -p "${SBIN}" "${FS}/etc/sudoers.d" "${FS}/etc/systemd/system" "${FS}/var/spool/cron/crontabs" \
  "${FS}/var/lib/systemd/linger" "${FS}/opt/rust/cargo/bin" "${FS}/opt/rust/rustup" \
  "${FS}/home/runner/actions-runner/_work" "${FS}/run/lock" "${FS}/var/crash" "${FS}/var/tmp" "${FS}/tmp" \
  "${FS}/dev/shm" "${FS}/var/lib/docker/overlay2/l1/diff/tmp" "${FS}/var/lib/nemosub" \
  "${FS}/home/runner/.cache" "${FS}/home/runner/actions-runner/_diag" "${FS}/home/runner/actions-runner/bin" \
  "${FS}/home/runner/actions-runner/_work/_actions" "${FS}/home/runner/actions-runner/_work/_temp" \
  "${FS}/home/runner/actions-runner/_work/cudafast/cudafast" "${FS}/var/lib/nemotron35/serve-log-env/old"
# World-writable, as on a Spark (a docker layer's tmp is one too, but pruned).
chmod 1777 "${FS}/run/lock" "${FS}/var/tmp" "${FS}/tmp" "${FS}/dev/shm" "${FS}/var/lib/docker/overlay2/l1/diff/tmp"
chmod 3777 "${FS}/var/crash"
# runner's .env names the resident log dir; an older world-readable log is in it.
printf 'LANG=C.UTF-8\nSERVE_UP_LOG_DIR=%s\n' "${FS}/var/lib/nemotron35/serve-log-env" > "${FS}/home/runner/actions-runner/.env"
echo prompt > "${FS}/var/lib/nemotron35/serve-log-env/old/vllm-resident.x.log"
chmod 0755 "${FS}/var/lib/nemotron35/serve-log-env"; chmod 0644 "${FS}/var/lib/nemotron35/serve-log-env/old/vllm-resident.x.log"
sed -e "s#/etc/#${FS}/etc/#g" -e "s#/var/#${FS}/var/#g" -e "s#/opt/rust#${FS}/opt/rust#g" -e "s#/run/#${FS}/run/#g" \
    -e "s#^WIPE_ALWAYS=.*#WIPE_ALWAYS=(${FS}/dev/shm ${FS}/dev/mqueue)#" \
  "${BOX_RUNNER_DIR}/setup-sandbox-user.sh" > "${sbx}/setup-sandbox-user.sh"
chmod +x "${sbx}/setup-sandbox-user.sh"
export SBX_LOG="${sbx}/calls.log" SBX_IPT="${sbx}/iptables.rules" SBX_FS="${FS}" SBX_ACL="${sbx}/acl.log" SBX_DEFAULT_ACL="${sbx}/default-acl.log"
: > "${SBX_ACL}"
: > "${SBX_DEFAULT_ACL}"
cat > "${SBIN}/id" <<'EOF'
#!/usr/bin/env bash
case "$*" in
  -u) echo "${SBX_UID:-0}" ;;
  -un) echo "${SBX_UNAME:-root}" ;;
  "-u nemosub") echo "${SBX_SANDBOX_UID:-999}" ;;
  "-nG nemosub") echo "nemosub nemoshare" ;;
  "-nG runner") echo "runner docker nemoshare" ;;
  nemosub|runner) echo "uid=999" ;;
  "-u layrsub") echo 998 ;;
  "-nG layrsub") echo "layrsub layrshare" ;;
  "-nG layrbench") echo "layrbench docker layrshare" ;;
  layrsub|layrbench) echo "uid=998" ;;
  *) exit 1 ;;
esac
EOF
cat > "${SBIN}/getent" <<'EOF'
#!/usr/bin/env bash
case "$*" in
  "passwd nemosub") echo "nemosub:x:999:999::${SBX_FS}/var/lib/nemosub:/usr/sbin/nologin" ;;
  "passwd runner") echo "runner:x:1001:1001::${SBX_FS}/home/runner:/bin/bash" ;;
  "group nemoshare") echo "nemoshare:x:4343:runner,nemosub" ;;
  "passwd layrsub") echo "layrsub:x:998:998::${SBX_FS}/var/lib/layrsub:/usr/sbin/nologin" ;;
  "passwd layrbench") echo "layrbench:x:1002:1002::${SBX_FS}/home/layrbench:/bin/bash" ;;
  "group layrshare") echo "layrshare:x:4444:layrbench,layrsub" ;;
  *) exit 2 ;;
esac
EOF
# iptables/ip6tables: a rules file, one "<tool> <rule>" line per rule, in order.
cat > "${SBIN}/iptables" <<'EOF'
#!/usr/bin/env bash
tool="$(basename "$0")"; op="$1"; shift
[[ "${op}" == -I ]] && { chain="$1"; shift 2; set -- "${chain}" "$@"; }
line="${tool} $*"
echo "${tool} ${op} $*" >> "${SBX_LOG}"
case "${op}" in
  -C) grep -qxF -- "${line}" "${SBX_IPT}" ;;
  -D) grep -qxF -- "${line}" "${SBX_IPT}" || exit 1
      awk -v l="${line}" '!done && $0 == l { done = 1; next } 1' "${SBX_IPT}" > "${SBX_IPT}.n"; mv "${SBX_IPT}.n" "${SBX_IPT}" ;;
  -I) { echo "${line}"; cat "${SBX_IPT}"; } > "${SBX_IPT}.n"; mv "${SBX_IPT}.n" "${SBX_IPT}" ;;
esac
EOF
cp "${SBIN}/iptables" "${SBIN}/ip6tables"
# sudo / runuser: run the command as this user; STUB network denial for the
# socket probes the real firewall would refuse.
cat > "${SBIN}/sudo" <<'EOF'
#!/usr/bin/env bash
echo "sudo $*" >> "${SBX_LOG}"
while [[ "$1" == -* && "$1" != -- ]]; do [[ "$1" == -u ]] && shift; shift; done
[[ "$1" == -- ]] && shift
# The public addresses are always "denied": the test never reaches the network.
[[ "$*" == *1.1.1.1* || "$*" == *2606:4700* ]] && exit 1
if [[ "${SBX_NET_DENY:-1}" == 1 && "$*" == *"/dev/"[tu][cd]p* ]]; then exit 1; fi
# `test` as nemosub: a path under the scratch tree is out of reach when it or an
# ancestor carries a u:nemosub:--- ACL (the setfacl stub's record) or its mode
# gives "other" nothing -- the sandbox user's view, simulated.
if [[ "$1" == test ]]; then
  for a in "${@:2}"; do
    [[ "${a}" == "${SBX_FS}"/* && -e "${a}" ]] || continue
    [[ "$(stat -c %a "${a}")" == *0 ]] && exit 1
    p="${a}"
    while [[ "${p}" == "${SBX_FS}"/* ]]; do
      grep -qxF -- "${p}" "${SBX_ACL}" && exit 1
      p="${p%/*}"
    done
  done
fi
exec "$@"
EOF
cat > "${SBIN}/runuser" <<'EOF'
#!/usr/bin/env bash
shift 3; exec "$@"
EOF
cat > "${SBIN}/crontab" <<'EOF'
#!/usr/bin/env bash
echo "crontab $*" >> "${SBX_LOG}"
[[ "$*" == "-l" ]] || exit 0
if grep -qx nemosub "${SBX_FS}/etc/cron.deny" 2>/dev/null; then
  echo "You (nemosub) are not allowed to use this program (crontab)" >&2; exit 1
fi
echo "no crontab for nemosub" >&2; exit 1
EOF
cat > "${SBIN}/install" <<'EOF'
#!/usr/bin/env bash
args=(); while [[ $# -gt 0 ]]; do case "$1" in -o|-g) shift 2 ;; *) args+=("$1"); shift ;; esac; done
exec /usr/bin/install "${args[@]}"
EOF
cat > "${SBIN}/systemctl" <<'EOF'
#!/usr/bin/env bash
echo "systemctl $*" >> "${SBX_LOG}"
case "$1" in
  is-enabled)
    u="${@: -1}"
    grep -qF "enable ${u}" "${SBX_LOG}" || exit 1
    ;;
esac
exit 0
EOF
for t in loginctl atq useradd usermod groupadd gpasswd visudo curl chown su netfilter-persistent; do
  printf '#!/usr/bin/env bash\necho "%s $*" >> "${SBX_LOG}"\n' "${t}" > "${SBIN}/${t}"
done
# setfacl: logged; a u:nemosub:--- target is also recorded for the sudo stub.
# Default ACL (-d -m u:…:---) is recorded for the getfacl stub and --check.
cat > "${SBIN}/setfacl" <<'EOF'
#!/usr/bin/env bash
echo "setfacl $*" >> "${SBX_LOG}"
if [[ "$1" == -n && "$2" == -d && "$3" == -m && "$4" == u:*:--- ]]; then
  echo "$5 $4" >> "${SBX_DEFAULT_ACL}"
elif [[ "$1" == -m && "$2" == u:*:--- ]]; then
  echo "$3" >> "${SBX_ACL}"
fi
exit 0
EOF
cp "${SBIN}/setfacl" "${sbx}/setfacl.stub"
cat > "${SBIN}/getfacl" <<'EOF'
#!/usr/bin/env bash
default_only=0 path=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--default) default_only=1; shift ;;
    -*) shift ;;
    *) path="$1"; shift ;;
  esac
done
[[ -n "${path}" && -f "${SBX_DEFAULT_ACL:-}" ]] || exit 0
while IFS= read -r rec; do
  set -- ${rec}
  [[ "$1" == "${path}" ]] || continue
  user="${2#u:}"; user="${user%:---}"
  # Like real getfacl: -d prints the default entries without the default: prefix.
  if [[ "${default_only}" == 1 ]]; then echo "user:${user}:---"; else echo "default:user:${user}:---"; fi
done < "${SBX_DEFAULT_ACL}"
exit 0
EOF
# findmnt: the scratch tree is the one local filesystem.
printf '#!/usr/bin/env bash\necho "%s"\n' "${FS}" > "${SBIN}/findmnt"
printf '#!/usr/bin/env bash\necho "rustup $*" >> "${SBX_LOG}"\n' > "${FS}/opt/rust/cargo/bin/rustup"
chmod +x "${SBIN}"/* "${FS}/opt/rust/cargo/bin/rustup"
rm -f "${SBIN}/netfilter-persistent"   # not installed: the persistence hint path
old_rule="OUTPUT -m owner --uid-owner nemosub ! -o lo -j REJECT"
new_rule="OUTPUT -m owner --uid-owner nemosub -j REJECT"
printf 'iptables OUTPUT -j ACCEPT\niptables %s\nip6tables %s\niptables %s\n' "${old_rule}" "${old_rule}" "${old_rule}" > "${SBX_IPT}"
: > "${SBX_LOG}"
sbx_run() { env PATH="${SBIN}:${PATH}" "$@"; }

sbx_run "${sbx}/setup-sandbox-user.sh" > "${sbx}/setup1.out" 2>&1
sbx_run "${sbx}/setup-sandbox-user.sh" > "${sbx}/setup2.out" 2>&1   # idempotence
assert_eq "sandbox setup: the old loopback-exempt rule is gone (every copy)" 0 "$(grep -cF -- "${old_rule}" "${SBX_IPT}")"
assert_eq "sandbox setup: iptables REJECT (no loopback exemption) is first" "iptables ${new_rule}" "$(grep '^iptables ' "${SBX_IPT}" | head -1)"
assert_eq "sandbox setup: ip6tables REJECT (no loopback exemption) is first" "ip6tables ${new_rule}" "$(grep '^ip6tables ' "${SBX_IPT}" | head -1)"
assert_eq "sandbox setup: a second run adds no duplicate rule" 2 "$(grep -cF -- "${new_rule}" "${SBX_IPT}")"
for f in cron.deny at.deny; do
  assert_eq "sandbox setup: nemosub listed once in ${f}" 1 "$(grep -cx nemosub "${FS}/etc/${f}" 2>/dev/null)"
done
unit="${FS}/etc/systemd/system/nemotron35-sandbox-acl.service"
assert_true "sandbox setup: ACL unit re-applies the deny on the resolved varlink socket" \
  grep -qE "^ExecStart=-.*setfacl -m u:nemosub:--- ${FS}/run/systemd/resolve/io.systemd.Resolve$" "${unit}"
assert_true "sandbox setup: ACL unit covers nscd's socket" grep -qE "^ExecStart=-.*setfacl -m u:nemosub:--- ${FS}/run/nscd/socket$" "${unit}"
assert_true "sandbox setup: ACL unit runs after resolved, at boot" \
  bash -c "grep -qx 'After=systemd-resolved.service nscd.service' '${unit}' && grep -qx 'WantedBy=multi-user.target' '${unit}'"
assert_true "sandbox setup: ACL unit enabled and run" \
  bash -c "grep -qx 'systemctl enable nemotron35-sandbox-acl.service' '${SBX_LOG}' && grep -qx 'systemctl restart nemotron35-sandbox-acl.service' '${SBX_LOG}'"
egress_unit="${FS}/etc/systemd/system/nemotron35-sandbox-egress.service"
assert_true "sandbox setup: egress unit runs after network-pre and ufw, before box-runner" \
  bash -c "grep -qx 'After=network-pre.target ufw.service' '${egress_unit}' && grep -qx 'Before=nemotron35-box-runner.service' '${egress_unit}'"
egress_script="${FS}/etc/nemotron35-sandbox/egress.sh"
assert_true "sandbox setup: egress unit runs the installed script by path (no shell text in the unit)" \
  grep -qx "ExecStart=${egress_script}" "${egress_unit}"
assert_true "sandbox setup: egress script checks then inserts the iptables rule idempotently" \
  bash -c "grep -qE '/iptables -C OUTPUT -m owner --uid-owner nemosub -j REJECT 2>/dev/null; do .*/iptables -D OUTPUT' \"\$0\" && grep -qE '/iptables -I OUTPUT 1 -m owner --uid-owner nemosub -j REJECT$' \"\$0\"" "${egress_script}"
assert_true "sandbox setup: egress script covers ip6tables" \
  grep -qE "/ip6tables -I OUTPUT 1 -m owner --uid-owner nemosub -j REJECT$" "${egress_script}"
assert_true "sandbox setup: egress script is valid sh" sh -n "${egress_script}"
assert_true "sandbox setup: egress unit enabled and run" \
  bash -c "grep -qx 'systemctl enable nemotron35-sandbox-egress.service' '${SBX_LOG}' && grep -qx 'systemctl restart nemotron35-sandbox-egress.service' '${SBX_LOG}'"
for d in "" /actions-runner /actions-runner/_work; do
  assert_true "sandbox setup: traverse-only ACL on runner home${d}" grep -qxF "setfacl -m u:nemosub:--x ${FS}/home/runner${d}" "${SBX_LOG}"
done
assert_true "sandbox setup: default deny ACL on runner home (new entries inherit)" \
  grep -qxF "setfacl -n -d -m u:nemosub:--- ${FS}/home/runner" "${SBX_LOG}"
# F4: every entry beside the path to the checkout is denied; the path itself is not.
for d in /.cache /actions-runner/.env /actions-runner/_diag /actions-runner/bin /actions-runner/_work/_actions /actions-runner/_work/_temp; do
  assert_true "sandbox setup: deny ACL on runner home${d}" grep -qxF "setfacl -m u:nemosub:--- ${FS}/home/runner${d}" "${SBX_LOG}"
done
for d in /actions-runner /actions-runner/_work /actions-runner/_work/cudafast; do
  assert_eq "sandbox setup: no deny on the path to the checkout (runner home${d})" "" \
    "$(grep -xF "setfacl -m u:nemosub:--- ${FS}/home/runner${d}" "${SBX_LOG}")"
done
# F3: the wipe list holds each world-writable dir nemosub can reach, and /dev/shm.
wl="${FS}/etc/nemotron35-sandbox/wipe-dirs"
for d in /tmp /var/tmp /run/lock /var/crash /dev/shm; do
  assert_true "sandbox setup: wipe list has ${d}" grep -qxF "${FS}${d}" "${wl}"
done
assert_eq "sandbox setup: wipe list skips docker's layers and an unmounted mqueue" "" "$(grep -E 'docker|mqueue' "${wl}")"
assert_eq "sandbox setup: wipe list is mode 0644" 644 "$(stat -c %a "${wl}")"
# serve-log: the .env's dir, 0750, runner:runner, nothing in it readable by others.
sl="${FS}/var/lib/nemotron35/serve-log-env"
assert_eq "sandbox setup: the .env's resident log dir is 0750" 750 "$(stat -c %a "${sl}")"
assert_true "sandbox setup: the resident log dir is chowned runner:runner" grep -qxF "chown runner:runner ${sl}" "${SBX_LOG}"
assert_eq "sandbox setup: an older log in it loses other-read" 640 "$(stat -c %a "${sl}/old/vllm-resident.x.log")"
assert_true "sandbox setup: crontab removed and linger disabled" \
  bash -c "grep -qx 'crontab -r -u nemosub' '${SBX_LOG}' && grep -qx 'loginctl disable-linger nemosub' '${SBX_LOG}'"

# No setfacl: refused by name, before anything is changed. PATH holds only
# the stubs and links to every other real tool.
mkdir -p "${sbx}/nofacl"
for b in /usr/bin/* /bin/*; do
  n="$(basename "${b}")"; [[ "${n}" == setfacl || -e "${sbx}/nofacl/${n}" ]] || ln -s "${b}" "${sbx}/nofacl/${n}" 2>/dev/null
done
rm -f "${SBIN}/setfacl"
: > "${SBX_LOG}"
assert_exit "sandbox setup: refused without setfacl (acl package)" 78 "install the acl package" \
  env PATH="${SBIN}:${sbx}/nofacl" "${sbx}/setup-sandbox-user.sh"
assert_eq "sandbox setup: nothing ran before the setfacl refusal" "" "$(grep -E '^(useradd|iptables|ip6tables) ' "${SBX_LOG}")"
cp "${sbx}/setfacl.stub" "${SBIN}/setfacl"; chmod +x "${SBIN}/setfacl"

# --check as runner: the new items pass on a denied sandbox and fail open ones.
sbx_check() { env PATH="${SBIN}:${PATH}" SBX_UID=1001 SBX_UNAME=runner "$@" "${sbx}/setup-sandbox-user.sh" --check 2>&1; }
out="$(sbx_check)"
assert_true "sandbox --check: loopback TCP to a live listener is denied" grep -q '^PASS  nemosub cannot open a TCP connection to a live listener on 127.0.0.1' <<<"${out}"
assert_true "sandbox --check: UDP to 127.0.0.53:53 is denied" grep -q '^PASS  nemosub cannot send UDP to 127.0.0.53:53' <<<"${out}"
assert_true "sandbox --check: name resolution fails" grep -q '^PASS  nemosub cannot resolve a name' <<<"${out}"
# The systemctl stub reports a unit enabled only if the call log records its enable.
out_root="$(env PATH="${SBIN}:${PATH}" "${sbx}/setup-sandbox-user.sh" --check 2>&1)"
assert_true "sandbox --check (root): a disabled egress unit fails" \
  grep -q '^FAIL  nemotron35-sandbox-egress.service is enabled' <<<"${out_root}"
echo "systemctl enable nemotron35-sandbox-egress.service" >> "${SBX_LOG}"
out_root="$(env PATH="${SBIN}:${PATH}" "${sbx}/setup-sandbox-user.sh" --check 2>&1)"
assert_true "sandbox --check (root): egress unit enabled" \
  grep -q '^PASS  nemotron35-sandbox-egress.service is enabled' <<<"${out_root}"
assert_true "sandbox --check: cron refuses nemosub" grep -q '^PASS  cron refuses nemosub' <<<"${out}"
assert_true "sandbox --check: cron.deny and at.deny list nemosub" \
  bash -c "grep -q '^PASS  nemosub is listed in .*/cron.deny' <<<\"\$0\" && grep -q '^PASS  nemosub is listed in .*/at.deny' <<<\"\$0\"" "${out}"
assert_true "sandbox --check: no linger" grep -q '^PASS  nemosub does not linger' <<<"${out}"
assert_true "sandbox --check: nemosub can traverse _work" grep -q '^PASS  nemosub can traverse .*/_work' <<<"${out}"
assert_true "sandbox --check: runner home carries default deny ACL for nemosub" \
  grep -q "^PASS  runner home ${FS}/home/runner has default ACL default:user:nemosub:---" <<<"${out}"
for p in actions-runner/.env actions-runner/_diag .cache; do
  assert_true "sandbox --check: nemosub cannot read ~runner/${p}" grep -q "^PASS  nemosub cannot read ${FS}/home/runner/${p}\$" <<<"${out}"
done
assert_true "sandbox --check: the wipe list exists" grep -q '^PASS  wipe list .* exists' <<<"${out}"
assert_true "sandbox --check: nothing nemosub-owned in the wipe-list dirs" grep -q '^PASS  nothing nemosub-owned in any wipe-list directory' <<<"${out}"
assert_true "sandbox --check: nemosub cannot read the resident log dir" grep -q "^PASS  nemosub cannot read the resident log dir ${FS}/var/lib/nemotron35/serve-log-env\$" <<<"${out}"
# Negative: a file "nemosub" (here: this user) left in /run/lock fails the check;
# lifting the home denies and opening the log dir fail theirs.
touch "${FS}/run/lock/leftover"
cp "${SBX_ACL}" "${sbx}/acl.saved"; : > "${SBX_ACL}"; chmod 0755 "${sl}"
out="$(sbx_check SBX_SANDBOX_UID="$(id -u)")"
assert_true "sandbox --check: a nemosub-owned file left in a wipe-list dir fails" grep -q '^FAIL  nothing nemosub-owned in any wipe-list directory' <<<"${out}"
assert_true "sandbox --check: a readable actions-runner/.env fails" grep -q "^FAIL  nemosub cannot read ${FS}/home/runner/actions-runner/.env\$" <<<"${out}"
: > "${SBX_DEFAULT_ACL}"
out="$(sbx_check)"
assert_true "sandbox --check: missing runner home default deny ACL fails" \
  grep -q "^FAIL  runner home ${FS}/home/runner has default ACL default:user:nemosub:---" <<<"${out}"
echo "${FS}/home/runner u:nemosub:---" >> "${SBX_DEFAULT_ACL}"
assert_true "sandbox --check: a readable resident log dir fails" grep -q '^FAIL  nemosub cannot read the resident log dir' <<<"${out}"
rm -f "${FS}/run/lock/leftover"; cp "${sbx}/acl.saved" "${SBX_ACL}"; chmod 0750 "${sl}"
mv "${wl}" "${wl}.off"
out="$(sbx_check)"
assert_true "sandbox --check: a missing wipe list fails" grep -q '^FAIL  wipe list .* exists' <<<"${out}"
mv "${wl}.off" "${wl}"
out="$(sbx_check SBX_NET_DENY=0)"
assert_true "sandbox --check: an OPEN loopback fails (the listener proves it was reachable)" \
  grep -q '^FAIL  nemosub cannot open a TCP connection to a live listener on 127.0.0.1' <<<"${out}"
assert_true "sandbox --check: an OPEN UDP path fails" grep -q '^FAIL  nemosub cannot send UDP to 127.0.0.53:53' <<<"${out}"
sed -i '/^nemosub$/d' "${FS}/etc/cron.deny"
touch "${FS}/var/lib/systemd/linger/nemosub"
out="$(sbx_check)"
assert_true "sandbox --check: 'no crontab' is not enough, cron must refuse" grep -q '^FAIL  cron refuses nemosub' <<<"${out}"
assert_true "sandbox --check: a linger file fails" grep -q '^FAIL  nemosub does not linger' <<<"${out}"
rm -rf "${FS}/home/runner/actions-runner/_work"
out="$(sbx_check)"
assert_true "sandbox --check: no _work yet is a warning, not a failure" \
  bash -c "grep -q '^WARN  .*_work does not exist yet' <<<\"\$0\" && ! grep -q '^FAIL  nemosub can traverse .*/_work' <<<\"\$0\"" "${out}"

# =========================================================================
# 8. setup-sandbox-user.sh for a SECOND runner on the same box
#    (docs/sandbox-design.md 6b): every name set to the second runner's value.
#    The setup adds its own user, rules, units and files, and changes nothing
#    of the first runner's: not its files, not its iptables rules (an older
#    loopback-exempt nemosub rule included), not its ACLs.
# =========================================================================
mkdir -p "${FS}/home/layrbench/actions-runner/_work" "${FS}/opt/layrsub-rust/cargo/bin" "${FS}/opt/layrsub-rust/rustup"
printf 'SERVE_UP_LOG_DIR=%s\n' "${FS}/var/lib/layrbench/serve-log" > "${FS}/home/layrbench/actions-runner/.env"
printf '#!/usr/bin/env bash\necho "rustup $*" >> "${SBX_LOG}"\n' > "${FS}/opt/layrsub-rust/cargo/bin/rustup"
chmod +x "${FS}/opt/layrsub-rust/cargo/bin/rustup"
echo "iptables ${old_rule}" >> "${SBX_IPT}"
first_files=("${FS}/etc/nemotron35-sandbox/egress.sh" "${FS}/etc/nemotron35-sandbox/wipe-dirs"
  "${FS}/etc/systemd/system/nemotron35-sandbox-egress.service" "${FS}/etc/systemd/system/nemotron35-sandbox-acl.service"
  "${FS}/etc/sudoers.d/nemotron35-sandbox")
first_sums="$(sha256sum "${first_files[@]}" 2>&1)"
first_rules="$(grep -F -- '--uid-owner nemosub' "${SBX_IPT}")"
: > "${SBX_LOG}"
sbx_run env RUNNER_USER=layrbench SANDBOX_USER=layrsub SHARE_GROUP=layrshare \
  SANDBOX_ROOT="${FS}/var/lib/layrbench/sandbox" SANDBOX_ETC="${FS}/etc/layrbench-sandbox" \
  SUDOERS_NAME=layrbench-sandbox EGRESS_UNIT=layrbench-sandbox-egress.service \
  ACL_UNIT=layrbench-sandbox-acl.service RUST_DIR="${FS}/opt/layrsub-rust" \
  "${sbx}/setup-sandbox-user.sh" > "${sbx}/setup-second.out" 2>&1
assert_true "second runner: setup ran to its closing --check" grep -q '^setup-sandbox-user.sh --check' "${sbx}/setup-second.out"
assert_eq "second runner: the first runner's files are unchanged" "${first_sums}" "$(sha256sum "${first_files[@]}" 2>&1)"
assert_eq "second runner: the first runner's iptables rules are unchanged" "${first_rules}" "$(grep -F -- '--uid-owner nemosub' "${SBX_IPT}")"
assert_eq "second runner: no call names nemosub, runner or nemoshare" "" "$(grep -E 'nemosub|nemoshare|/home/runner|runner:' "${SBX_LOG}")"
assert_eq "second runner: its iptables REJECT is first" "iptables OUTPUT -m owner --uid-owner layrsub -j REJECT" "$(grep '^iptables ' "${SBX_IPT}" | head -1)"
assert_true "second runner: its own egress script and unit" \
  bash -c "grep -qE '/ip6tables -I OUTPUT 1 -m owner --uid-owner layrsub -j REJECT$' '${FS}/etc/layrbench-sandbox/egress.sh' && grep -qx 'ExecStart=${FS}/etc/layrbench-sandbox/egress.sh' '${FS}/etc/systemd/system/layrbench-sandbox-egress.service'"
assert_true "second runner: its own resolver ACL unit" \
  grep -qE "^ExecStart=-.*setfacl -m u:layrsub:--- ${FS}/run/nscd/socket$" "${FS}/etc/systemd/system/layrbench-sandbox-acl.service"
assert_true "second runner: its own sudoers file" grep -qxF 'layrbench ALL=(layrsub) NOPASSWD:SETENV: ALL' "${FS}/etc/sudoers.d/layrbench-sandbox"
assert_true "second runner: its own wipe list" grep -qxF "${FS}/tmp" "${FS}/etc/layrbench-sandbox/wipe-dirs"
assert_true "second runner: its sandbox root is layrbench:layrshare" grep -qxF "chown layrbench:layrshare ${FS}/var/lib/layrbench/sandbox" "${SBX_LOG}"
assert_true "second runner: its own toolchain dir" grep -q '^rustup default ' "${SBX_LOG}"
assert_true "second runner: traverse ACL on its own runner home" grep -qxF "setfacl -m u:layrsub:--x ${FS}/home/layrbench" "${SBX_LOG}"
assert_true "second runner: default deny ACL on its runner home" \
  grep -qxF "setfacl -n -d -m u:layrsub:--- ${FS}/home/layrbench" "${SBX_LOG}"
assert_eq "second runner: at.deny lists both sandbox users" "nemosub layrsub" "$(grep -xE 'nemosub|layrsub' "${FS}/etc/at.deny" | paste -sd' ' -)"

# =========================================================================
printf '\n%d passed, %d failed (floor: %d assertions)\n' "${PASSED}" "${FAILED}" "${EXPECTED_MIN_ASSERTIONS}"
total=$((PASSED + FAILED))
if [ "${total}" -lt "${EXPECTED_MIN_ASSERTIONS}" ]; then
  printf 'FAIL  suite ran only %d assertions, floor is %d (a section stopped running silently)\n' "${total}" "${EXPECTED_MIN_ASSERTIONS}"
  exit 1
fi
if [ "${FAILED}" -gt 0 ]; then
  printf '\nFailures:\n'
  printf '  - %s\n' "${FAILURES[@]}"
  exit 1
fi
exit 0

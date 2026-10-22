#!/usr/bin/env bash
# Unit test for the sandboxed build in tools/engine/build.sh and
# tools/stage-cuda-engine.sh (docs/sandbox-design.md sections 5 and 6a).
#
# Asserts, with sudo, docker, cargo, pgrep and getent stubbed:
#   1. SANDBOX_USER unset: cargo, docker and staging run exactly as before.
#   2. SANDBOX_USER set: cargo vendors as the caller, builds via sudo as the
#      sandbox user from the repo root with --offline and a job-private
#      CARGO_HOME/target; the kernels container gets --user, a read-only
#      checkout mount, one writable kernels-out mount and the limits; the
#      per-job dirs have the contract modes; the outputs are staged, the
#      adapter as <staged>.sandboxed and a shim at the staged path that execs
#      tools/run-engine-sandboxed.sh on it.
#   2b. a second runner's SHARE_GROUP and RUST_DIR (docs/sandbox-design.md
#      6b): the build uses that toolchain and the kernels container that gid.
#   3. SANDBOX_USER set but sudo to it fails: refused before anything builds.
#   4. stage-cuda-engine.sh refuses a symlink or a binary the sandbox user
#      does not own, and re-checks its COPY (a source swapped for a symlink
#      after the check is refused).
#   5. vendoring is refused for a non-crates.io Cargo.lock source, a path
#      outside harness/, a [patch]/[replace] section, another registry, and a
#      vendored symlink; a kernels output holding a symlink is refused and
#      deleted.
#
# Hermetic: no GPU, no docker, no cargo, no root.
#
# Usage: tools/test-engine-build-sandbox.sh [-v]
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
VERBOSE=0
[[ "${1:-}" == "-v" ]] && VERBOSE=1

EXPECTED_MIN_ASSERTIONS=52
PASSED=0; FAILED=0; FAILURES=()
pass() { PASSED=$((PASSED+1)); [[ "${VERBOSE}" == "1" ]] && echo "ok    $1"; return 0; }
fail() { FAILED=$((FAILED+1)); FAILURES+=("$1"); echo "FAIL  $1" >&2; }
check() { local label="$1"; shift; if "$@"; then pass "${label}"; else fail "${label}"; fi; }
logged() { grep -qxF -- "$1" "${LOG}"; }
not_logged() { ! grep -qF -- "$1" "${LOG}"; }

WORK="$(cd "$(mktemp -d "${TMPDIR:-/tmp}/engine-build-sandbox.XXXXXX")" && pwd -P)"
trap 'rm -rf "${WORK}"' EXIT
REPO="${WORK}/repo"; STUBS="${WORK}/bin"; LOG="${WORK}/log"
export LOG STUBS
# The job's private temp dir (Actions sets it); the shim bakes it in as the
# adapter's stderr dir.
RUNNER_TEMP="${WORK}/runner-temp"; mkdir -p "${RUNNER_TEMP}"; export RUNNER_TEMP

# A minimal tree: the two scripts under test, the real fixture, and a stubbed
# overlay stage (tested elsewhere).
mkdir -p "${REPO}/tools/engine" "${REPO}/fixtures" "${REPO}/harness/protocol-adapter" \
  "${REPO}/harness/kernels" "${STUBS}"
cp "${REPO_ROOT}/tools/engine/build.sh" "${REPO}/tools/engine/build.sh"
cp "${REPO_ROOT}/tools/stage-cuda-engine.sh" "${REPO}/tools/stage-cuda-engine.sh"
cp "${REPO_ROOT}/fixtures/nemotron35_lightning_30b_a3b_track.json" "${REPO}/fixtures/"
printf '#!/usr/bin/env bash\nexit 0\n' > "${REPO}/tools/stage-vllm-overlay.sh"; chmod +x "${REPO}/tools/stage-vllm-overlay.sh"
touch "${REPO}/harness/kernels/build.sh"
GOOD_TOML='[package]
name = "adapter"
version = "0.1.0"

[[bin]]
name = "cuda-engine"
path = "src/bin/cuda-engine.rs"

[dependencies]
serde = "1"
'
GOOD_LOCK='version = 3

[[package]]
name = "adapter"
version = "0.1.0"

[[package]]
name = "serde"
version = "1.0.0"
source = "registry+https://github.com/rust-lang/crates.io-index"
'
good_inputs() {
  printf '%s' "${GOOD_TOML}" > "${REPO}/harness/protocol-adapter/Cargo.toml"
  printf '%s' "${GOOD_LOCK}" > "${REPO}/harness/protocol-adapter/Cargo.lock"
  rm -rf "${REPO}/harness/extra"
}
good_inputs

cat > "${STUBS}/cargo" <<'EOF'
#!/usr/bin/env bash
echo "cargo cwd=${PWD} CARGO_HOME=${CARGO_HOME:-} CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-} ENGINE_CUDA_ARCH=${ENGINE_CUDA_ARCH:-} $*" >> "${LOG}"
manifest=""; prev=""
for a in "$@"; do [[ "${prev}" == "--manifest-path" ]] && manifest="${a}"; prev="${a}"; done
case "$1" in
  vendor) mkdir -p "${@: -1}/some-crate"
          [[ -n "${STUB_VENDOR_SYMLINK:-}" ]] && ln -s /etc/passwd "${@: -1}/some-crate/leak"
          printf '[source.crates-io]\nreplace-with = "vendored-sources"\n' ;;
  build)  t="${CARGO_TARGET_DIR:-$(dirname "${manifest}")/target}/release"
          mkdir -p "${t}"; printf '#!/bin/sh\n' > "${t}/cuda-engine"; chmod +x "${t}/cuda-engine" ;;
esac
EOF
# sudo: log, then run `true`, skip `kill`, and run `[prlimit --X=N...] env
# K=V... cargo ...` against the cargo stub (the real env would search /opt/rust).
cat > "${STUBS}/sudo" <<'EOF'
#!/usr/bin/env bash
echo "sudo cwd=${PWD} $*" >> "${LOG}"
[[ "${STUB_SUDO_FAIL:-0}" == "1" ]] && exit 1
shift 3; [[ "$1" == "--" ]] && shift
if [[ "$1" == prlimit ]]; then shift; while [[ "$1" == --* ]]; do shift; done; fi
case "$1" in
  true|kill) exit 0 ;;
  env) shift; while [[ "$1" == *=* ]]; do export "$1"; shift; done
       exec "${STUBS}/$1" "${@:2}" ;;
esac
exit 1
EOF
cat > "${STUBS}/docker" <<'EOF'
#!/usr/bin/env bash
echo "docker $*" >> "${LOG}"
[[ "$1" == "run" ]] && for a in "$@"; do
  [[ "${a}" == KERNELS_OUT=* ]] || continue
  touch "${a#KERNELS_OUT=}/k.so"
  [[ -n "${STUB_KERNELS_SYMLINK:-}" ]] && ln -s /etc/passwd "${a#KERNELS_OUT=}/leak.so"
  [[ -n "${STUB_KERNELS_SUID:-}" ]] && touch "${a#KERNELS_OUT=}/suid.so" && chmod 4755 "${a#KERNELS_OUT=}/suid.so"
  [[ -n "${STUB_KERNELS_WRITABLE:-}" ]] && chmod 0666 "${a#KERNELS_OUT=}/k.so"
done
exit 0
EOF
printf '#!/usr/bin/env bash\nexit 1\n' > "${STUBS}/pgrep"
printf '#!/usr/bin/env bash\n[[ "$2" == layrshare ]] && { echo layrshare:x:5353:; exit 0; }\necho nemoshare:x:4242:\n' > "${STUBS}/getent"
chmod +x "${STUBS}"/*
export PATH="${STUBS}:${PATH}"

ME="$(id -un)"
IMAGE_REF="docker.io/vllm/vllm-openai@$(jq -r '.serve_configuration.container_digest' "${REPO}/fixtures/nemotron35_lightning_30b_a3b_track.json")"
MANIFEST="${REPO}/harness/protocol-adapter/Cargo.toml"
TAIL="--env TORCH_CUDA_ARCH_LIST=12.1a --env PYTHONPATH=${REPO}/.build/vllm-overlay --workdir ${REPO}/harness/kernels --entrypoint bash ${IMAGE_REF} ./build.sh"
STAGED="${REPO}/.build/release/mlxfast-runtime-worker"

# ---- 1. SANDBOX_USER unset: today's commands --------------------------------
: > "${LOG}"
if env -u SANDBOX_USER "${REPO}/tools/engine/build.sh" > "${WORK}/out1" 2>&1; then
  pass "unsandboxed build succeeds"
else
  fail "unsandboxed build succeeds ($(tail -3 "${WORK}/out1"))"
fi
check "unsandboxed cargo build runs directly, as before" \
  logged "cargo cwd=${REPO} CARGO_HOME= CARGO_TARGET_DIR= ENGINE_CUDA_ARCH=sm_121 build --release --locked --manifest-path ${MANIFEST}"
check "unsandboxed kernels container is today's command" \
  logged "docker run --rm --gpus all --network none --user $(id -u):$(id -g) --mount type=bind,src=${REPO},dst=${REPO} --env HOME=/tmp --env KERNELS_OUT=${REPO}/.build/kernels ${TAIL}"
check "unsandboxed build never calls sudo" not_logged "sudo "
check "unsandboxed build never vendors" not_logged " vendor "
check "unsandboxed adapter staged" test -x "${STAGED}"
check "unsandboxed staging writes no shim and no .sandboxed binary" \
  test ! -e "${STAGED}.sandboxed" -a -z "$(grep -F run-engine-sandboxed "${STAGED}")"
check "unsandboxed kernels output in .build/kernels" test -f "${REPO}/.build/kernels/k.so"

# ---- 2. SANDBOX_USER set -----------------------------------------------------
rm -rf "${REPO}/.build" "${REPO}/harness/protocol-adapter/target"
mkdir -p "${WORK}/sandbox"
JOB="${WORK}/sandbox/42-1"
: > "${LOG}"
if SANDBOX_USER="${ME}" SANDBOX_ROOT="${WORK}/sandbox" SANDBOX_JOB=42-1 KERNELS_MEMORY=32g \
     "${REPO}/tools/engine/build.sh" > "${WORK}/out2" 2>&1; then
  pass "sandboxed build succeeds"
else
  fail "sandboxed build succeeds ($(tail -3 "${WORK}/out2"))"
fi
check "sandbox user checked with sudo -n true" logged "sudo cwd=${REPO} -n -u ${ME} true"
check "vendor runs as the caller from the repo root, own CARGO_HOME" \
  logged "cargo cwd=${REPO} CARGO_HOME= CARGO_TARGET_DIR= ENGINE_CUDA_ARCH= vendor --locked --manifest-path ${MANIFEST} ${JOB}/vendor"
check "build runs via sudo as the sandbox user, offline, from the repo root" \
  logged "sudo cwd=${REPO} -n -u ${ME} -- prlimit --nproc=4096 --as=68719476736 env PATH=/opt/rust/cargo/bin:/usr/bin:/bin RUSTUP_HOME=/opt/rust/rustup CARGO_HOME=${JOB}/cargo-home CARGO_TARGET_DIR=${JOB}/target ENGINE_CUDA_ARCH=sm_121 cargo build --release --locked --offline --manifest-path ${MANIFEST}"
check "cargo build sees only the job-private CARGO_HOME and target" \
  logged "cargo cwd=${REPO} CARGO_HOME=${JOB}/cargo-home CARGO_TARGET_DIR=${JOB}/target ENGINE_CUDA_ARCH=sm_121 build --release --locked --offline --manifest-path ${MANIFEST}"
check "no cargo build runs outside sudo" \
  not_logged "CARGO_TARGET_DIR= ENGINE_CUDA_ARCH=sm_121 build"
check "source replacement written to the sandbox CARGO_HOME" \
  grep -q 'replace-with = "vendored-sources"' "${JOB}/cargo-home/config.toml"
check "nothing written to a repo .cargo" test ! -e "${REPO}/.cargo"
check "every sandbox process killed after the build" logged "sudo cwd=${REPO} -n -u ${ME} -- kill -KILL -1"
for d in vendor:2750 sock:2750 cargo-home:2770 target:2770 kernels-out:2770 adapter-home:2770; do
  check "job dir ${d%%:*} mode ${d##*:}" test "$(stat -c %a "${JOB}/${d%%:*}")" == "${d##*:}"
done
check "kernels container: sandbox uid, nemoshare gid, read-only checkout, one writable mount, limits, no caps, no new privileges" \
  logged "docker run --rm --gpus all --network none --user $(id -u):4242 --mount type=bind,src=${REPO},dst=${REPO},readonly --mount type=bind,src=${JOB}/kernels-out,dst=${JOB}/kernels-out --cpus 16 --memory 32g --pids-limit 4096 --cap-drop ALL --security-opt no-new-privileges --env HOME=/tmp --env KERNELS_OUT=${JOB}/kernels-out ${TAIL}"
check "sandboxed adapter staged next to the shim as .sandboxed" \
  cmp -s "${JOB}/target/release/cuda-engine" "${STAGED}.sandboxed"
check "the .sandboxed adapter is a regular file, mode 755, ours" \
  test ! -L "${STAGED}.sandboxed" -a -f "${STAGED}.sandboxed" \
    -a "$(stat -c '%a %U' "${STAGED}.sandboxed")" == "755 ${ME}"
check "the staged path is the shim, mode 755, ours" \
  test ! -L "${STAGED}" -a "$(stat -c '%a %U' "${STAGED}")" == "755 ${ME}"
EXPECTED_SHIM="#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
export SANDBOX_USER=${ME}
export SANDBOX_ROOT=${WORK}/sandbox
export SANDBOX_JOB=42-1
export SANDBOX_ENGINE_LOG_DIR=${RUNNER_TEMP}
export SANDBOX_ENGINE_BIN=${STAGED}.sandboxed
exec ${REPO}/tools/run-engine-sandboxed.sh \"\$@\""
check "the shim bakes in PATH, the SANDBOX_* values and the log dir, then execs the wrapper" \
  test "$(cat "${STAGED}")" == "${EXPECTED_SHIM}"
check "kernels-out copied into .build/kernels" test -f "${REPO}/.build/kernels/k.so"

# ---- 2b. a second runner's names: SHARE_GROUP and RUST_DIR -------------------
rm -rf "${REPO}/.build"
: > "${LOG}"
if SANDBOX_USER="${ME}" SANDBOX_ROOT="${WORK}/sandbox" SANDBOX_JOB=42-2 \
     SHARE_GROUP=layrshare RUST_DIR=/opt/layrsub-rust \
     "${REPO}/tools/engine/build.sh" > "${WORK}/out2b" 2>&1; then
  pass "second-runner sandboxed build succeeds"
else
  fail "second-runner sandboxed build succeeds ($(tail -3 "${WORK}/out2b"))"
fi
JOB2="${WORK}/sandbox/42-2"
check "second runner: cargo uses RUST_DIR's toolchain" \
  logged "sudo cwd=${REPO} -n -u ${ME} -- prlimit --nproc=4096 --as=68719476736 env PATH=/opt/layrsub-rust/cargo/bin:/usr/bin:/bin RUSTUP_HOME=/opt/layrsub-rust/rustup CARGO_HOME=${JOB2}/cargo-home CARGO_TARGET_DIR=${JOB2}/target ENGINE_CUDA_ARCH=sm_121 cargo build --release --locked --offline --manifest-path ${MANIFEST}"
check "second runner: the kernels container runs with SHARE_GROUP's gid" \
  grep -qF -- "docker run --rm --gpus all --network none --user $(id -u):5353 " "${LOG}"

# ---- 3. sudo to the sandbox user fails: refuse --------------------------------
: > "${LOG}"
err="$(STUB_SUDO_FAIL=1 SANDBOX_USER="${ME}" SANDBOX_ROOT="${WORK}/sandbox" SANDBOX_JOB=43-1 \
  "${REPO}/tools/engine/build.sh" 2>&1 >/dev/null)"; rc=$?
check "refused when sudo -n -u SANDBOX_USER fails" \
  test "${rc}" -ne 0 -a -n "$(grep -F "sudo -n -u ${ME} true' fails" <<<"${err}")"
check "refusal happens before any cargo run or job dir" \
  test ! -e "${WORK}/sandbox/43-1" -a -z "$(grep '^cargo' "${LOG}")"

# ---- 4. stage-cuda-engine.sh ownership guard ---------------------------------
export SANDBOX_ROOT="${WORK}/sandbox" SANDBOX_JOB=44-1
printf '#!/bin/sh\n' > "${WORK}/real"; chmod +x "${WORK}/real"
ln -s "${WORK}/real" "${WORK}/link"
SANDBOX_USER="${ME}" CUDA_ENGINE_EXECUTABLE="${WORK}/link" "${REPO}/tools/stage-cuda-engine.sh" >/dev/null 2>&1
check "stage refuses a symlinked source" test $? -ne 0
SANDBOX_USER=root CUDA_ENGINE_EXECUTABLE="${WORK}/real" "${REPO}/tools/stage-cuda-engine.sh" >/dev/null 2>&1
check "stage refuses a source the sandbox user does not own" test $? -ne 0
SANDBOX_USER="${ME}" CUDA_ENGINE_EXECUTABLE="${WORK}/real" "${REPO}/tools/stage-cuda-engine.sh" >/dev/null 2>&1
check "stage accepts a regular file the sandbox user owns" test $? -eq 0

# A source swapped for a symlink between the check and the copy (the cp stub
# does the swap): the staged copy is re-checked, refused and removed.
mkdir -p "${WORK}/cpstub"
REAL_CP="$(command -v cp)"
cat > "${WORK}/cpstub/cp" <<STUBEOF
#!/usr/bin/env bash
src="\${*: -2:1}"; rm -f "\${src}"; ln -s /etc/passwd "\${src}"
exec "${REAL_CP}" "\$@"
STUBEOF
chmod +x "${WORK}/cpstub/cp"
printf '#!/bin/sh\n' > "${WORK}/swap"; chmod +x "${WORK}/swap"
err="$(PATH="${WORK}/cpstub:${PATH}" SANDBOX_USER="${ME}" CUDA_ENGINE_EXECUTABLE="${WORK}/swap" \
  "${REPO}/tools/stage-cuda-engine.sh" 2>&1 >/dev/null)"; rc=$?
check "stage refuses a copy that is not a regular file (swap after the check)" \
  test "${rc}" -ne 0 -a -n "$(grep -F 'refusing the staged copy' <<<"${err}")"
check "the refused staged copy is removed" test ! -e "${STAGED}.sandboxed" -a ! -L "${STAGED}.sandboxed"

# The values baked into the shim are validated first; nothing is staged.
printf '#!/bin/sh\n' > "${WORK}/real2"; chmod +x "${WORK}/real2"
shim_refused() { # shim_refused LABEL MESSAGE VAR=VALUE...
  local label="$1" msg="$2"; shift 2
  rm -rf "${REPO}/.build"
  err="$(env "$@" CUDA_ENGINE_EXECUTABLE="${WORK}/real2" "${REPO}/tools/stage-cuda-engine.sh" 2>&1 >/dev/null)"; rc=$?
  check "stage refuses ${label}" \
    test "${rc}" -ne 0 -a -n "$(grep -F -- "${msg}" <<<"${err}")" -a ! -e "${STAGED}" -a ! -e "${STAGED}.sandboxed"
}
shim_refused "a SANDBOX_JOB outside [A-Za-z0-9._-]" "SANDBOX_JOB" SANDBOX_USER="${ME}" 'SANDBOX_JOB=1;rm -rf ~'
err="$(SANDBOX_USER='a$(id)' "${REPO}/tools/stage-cuda-engine.sh" --print-shim 2>&1 >/dev/null)"; rc=$?
check "the shim refuses a SANDBOX_USER that is not a plain name" \
  test "${rc}" -ne 0 -a -n "$(grep -F 'is not a plain user name' <<<"${err}")"
shim_refused "a relative SANDBOX_ROOT" "SANDBOX_ROOT" SANDBOX_USER="${ME}" SANDBOX_ROOT=rel
shim_refused "an unset RUNNER_TEMP" "RUNNER_TEMP" SANDBOX_USER="${ME}" RUNNER_TEMP=
ln -s "${RUNNER_TEMP}" "${WORK}/rt-link"
shim_refused "a symlinked RUNNER_TEMP" "RUNNER_TEMP" SANDBOX_USER="${ME}" RUNNER_TEMP="${WORK}/rt-link"
shim_refused "a RUNNER_TEMP owned by someone else" "RUNNER_TEMP" SANDBOX_USER="${ME}" RUNNER_TEMP=/
check "--print-shim prints what staging writes" \
  test "$(SANDBOX_USER="${ME}" CUDA_ENGINE_EXECUTABLE="${WORK}/real2" "${REPO}/tools/stage-cuda-engine.sh" >/dev/null 2>&1; cat "${STAGED}")" \
    == "$(SANDBOX_USER="${ME}" "${REPO}/tools/stage-cuda-engine.sh" --print-shim)"
unset SANDBOX_ROOT SANDBOX_JOB

# ---- 5. refusals before and after vendoring ----------------------------------
n=50
refuse() { # refuse LABEL MESSAGE [VAR=VALUE...]
  local label="$1" msg="$2"; shift 2
  n=$((n + 1))
  : > "${LOG}"
  err="$(env "$@" SANDBOX_USER="${ME}" SANDBOX_ROOT="${WORK}/sandbox" SANDBOX_JOB="${n}-1" \
    "${REPO}/tools/engine/build.sh" 2>&1 >/dev/null)"; rc=$?
  check "refused: ${label}" test "${rc}" -ne 0 -a -n "$(grep -F -- "${msg}" <<<"${err}")"
  [[ "${rc}" -ne 0 ]] || printf '    %s\n' "${err}" >&2
  check "no cargo build after refusal: ${label}" test -z "$(grep ' build ' "${LOG}")"
  good_inputs
}
sed -i 's|^source = .*|source = "git+https://example.com/evil.git#abc"|' "${REPO}/harness/protocol-adapter/Cargo.lock"
refuse "git source in Cargo.lock" "not crates.io"
check "a refused lock is never vendored" test -z "$(grep ' vendor ' "${LOG}")"
printf "\n[dependencies.evil]\npath = '../../../outside'\n" >> "${REPO}/harness/protocol-adapter/Cargo.toml"
refuse "path dependency outside harness/ (single-quoted)" "resolves outside harness/"
mkdir -p "${REPO}/harness/extra"
printf '[package]\nname = "x"\nversion = "0.1.0"\n[dependencies]\ny = { path = "\\u002e\\u002e/\\u002e\\u002e/tools" }\n' > "${REPO}/harness/extra/Cargo.toml"
refuse "escaped path in another harness Cargo.toml" "resolves outside harness/"
printf '\n[patch.crates-io]\nserde = { version = "1" }\n' >> "${REPO}/harness/protocol-adapter/Cargo.toml"
refuse "[patch] section" "has a [patch] section"
printf '\n[replace]\n"serde:1.0.0" = { version = "1" }\n' >> "${REPO}/harness/protocol-adapter/Cargo.toml"
refuse "[replace] section" "has a [replace] section"
printf '\n[dependencies.z]\nversion = "1"\nregistry = "evil"\n' >> "${REPO}/harness/protocol-adapter/Cargo.toml"
refuse "a non-crates.io registry" "is not crates-io"
refuse "a symlink in the vendored tree" "vendored tree holds a symlink" STUB_VENDOR_SYMLINK=1

: > "${LOG}"
err="$(STUB_KERNELS_SYMLINK=1 SANDBOX_USER="${ME}" SANDBOX_ROOT="${WORK}/sandbox" SANDBOX_JOB=70-1 \
  "${REPO}/tools/engine/build.sh" 2>&1 >/dev/null)"; rc=$?
check "kernels output with a symlink is refused" \
  test "${rc}" -ne 0 -a -n "$(grep -F 'kernels refused' <<<"${err}")"
check "the refused kernels copy is deleted" test ! -e "${REPO}/.build/kernels"
err="$(STUB_KERNELS_SUID=1 SANDBOX_USER="${ME}" SANDBOX_ROOT="${WORK}/sandbox" SANDBOX_JOB=71-1 \
  "${REPO}/tools/engine/build.sh" 2>&1 >/dev/null)"; rc=$?
check "kernels output with a setuid file is refused" \
  test "${rc}" -ne 0 -a -n "$(grep -F 'kernels refused' <<<"${err}")"
STUB_KERNELS_WRITABLE=1 SANDBOX_USER="${ME}" SANDBOX_ROOT="${WORK}/sandbox" SANDBOX_JOB=72-1 \
  "${REPO}/tools/engine/build.sh" >/dev/null 2>&1; rc=$?
check "a world-writable kernels output still builds" test "${rc}" -eq 0
check "the staged kernels copy is not group- or other-writable" test -z "$(find "${REPO}/.build/kernels" -perm /022)"

echo "engine-build-sandbox: ${PASSED} passed, ${FAILED} failed"
if [[ "${FAILED}" -ne 0 ]]; then
  printf '  - %s\n' "${FAILURES[@]}" >&2
  exit 1
fi
if [[ "${PASSED}" -lt "${EXPECTED_MIN_ASSERTIONS}" ]]; then
  echo "engine-build-sandbox: ran only ${PASSED} assertions, expected at least ${EXPECTED_MIN_ASSERTIONS} -- the suite shrank" >&2
  exit 1
fi
exit 0

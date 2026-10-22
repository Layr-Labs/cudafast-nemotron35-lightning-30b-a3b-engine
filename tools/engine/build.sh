#!/usr/bin/env bash
# tools/engine/build.sh -- build everything the scored engine needs, from THIS
# tree, and stage it where benchd and tools/serve-up.sh look.
#
#   1. the Engine Protocol v1 adapter (harness/protocol-adapter, cargo), staged
#      as .build/release/mlxfast-runtime-worker by tools/stage-cuda-engine.sh;
#   2. the pinned engine container, present by DIGEST (pulled when absent,
#      unless ENGINE_BUILD_OFFLINE=1, which the ranked box sets);
#   3. the vLLM overlay: the container's own install of the pinned vLLM tag
#      (compiled extensions and all) with this tree's vllm/vllm/ layered on top,
#      at .build/vllm-overlay (tools/stage-vllm-overlay.sh);
#   4. the participant's optional harness/kernels/: when harness/kernels/build.sh
#      exists it runs INSIDE the pinned container with no network and writes
#      .build/kernels (first on the resident's PYTHONPATH after the overlay).
#      It lives under harness/ so it is part of the editable surface AND of
#      benchd's harness-hash roster.
#
# SANDBOX (docs/sandbox-design.md sections 5 and 6a). With SANDBOX_USER set
# (the ranked box: SANDBOX_USER, SANDBOX_ROOT, SANDBOX_JOB from the workflow),
# participant code never runs as the caller:
#   - the caller vendors the crates (`cargo vendor`, no build scripts run) into
#     <job>/vendor, then cargo builds OFFLINE as SANDBOX_USER with its own
#     CARGO_HOME and target dir under <job>, still from the repo root so no
#     harness/**/.cargo/config.toml is read; every sandbox process is then
#     killed and the caller copies the binary into .build (which SANDBOX_USER
#     cannot write). Vendoring is refused unless every Cargo.lock source is
#     crates.io, every harness/**/Cargo.toml path stays inside harness/, and no
#     [patch]/[replace] or other registry is named; the vendored tree may hold
#     no symlink or special file. The build runs under prlimit
#     (CARGO_NPROC_LIMIT, CARGO_AS_LIMIT bytes; default 4096 and 64 GiB);
#   - the kernels container runs as SANDBOX_USER:SHARE_GROUP (default
#     nemoshare) with the checkout
#     mounted read-only, writes only <job>/kernels-out, and is CPU/memory/pid
#     limited (KERNELS_CPUS, KERNELS_MEMORY, KERNELS_PIDS_LIMIT).
# cargo uses the sandbox toolchain in RUST_DIR (default /opt/rust). SHARE_GROUP
# and RUST_DIR are tools/box-runner/setup-sandbox-user.sh's names.
# With SANDBOX_USER unset, everything runs as the caller, as before.
#
# Needs cargo, docker (with the NVIDIA runtime) and jq.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null && pwd -P)"
cd "${ROOT_DIR}"
log() { printf 'engine/build.sh: %s\n' "$*"; }
die() { printf 'engine/build.sh: %s\n' "$*" >&2; exit 1; }

FIXTURE="${ROOT_DIR}/fixtures/nemotron35_lightning_30b_a3b_track.json"
for tool in cargo docker jq; do command -v "${tool}" >/dev/null 2>&1 || die "${tool} is required"; done
IMAGE="$(jq -r '.serve_configuration.container_image' "${FIXTURE}")"
DIGEST="$(jq -r '.serve_configuration.container_digest' "${FIXTURE}")"
IMAGE_REF="${IMAGE%%:*}@${DIGEST}"
ARCH="$(jq -r '.serve_configuration.toolchain.cuda_arch' "${FIXTURE}")"

MANIFEST="${ROOT_DIR}/harness/protocol-adapter/Cargo.toml"
SANDBOX_USER="${SANDBOX_USER:-}"
if [[ -n "${SANDBOX_USER}" ]]; then
  [[ -n "${SANDBOX_ROOT:-}" && -n "${SANDBOX_JOB:-}" ]] \
    || die "SANDBOX_USER is set but SANDBOX_ROOT or SANDBOX_JOB is not"
  sudo -n -u "${SANDBOX_USER}" true \
    || die "SANDBOX_USER=${SANDBOX_USER} is set but 'sudo -n -u ${SANDBOX_USER} true' fails; run tools/box-runner/setup-sandbox-user.sh"
  command -v python3 >/dev/null 2>&1 || die "python3 is required to check the Cargo inputs before vendoring"
  JOB="${SANDBOX_ROOT}/${SANDBOX_JOB}"
  [[ ! -e "${JOB}" ]] || die "sandbox job directory ${JOB} already exists"
  # sock is 2750: serve-up creates each leg's own 2770 <tag>/ inside it, so the
  # sandbox user can never plant a name beside another leg's socket.
  mkdir -m 2750 "${JOB}" "${JOB}/vendor" "${JOB}/sock"
  mkdir -m 2770 "${JOB}/cargo-home" "${JOB}/target" "${JOB}/kernels-out" "${JOB}/adapter-home"
fi

# `cargo vendor` runs as the caller over participant Cargo.toml/Cargo.lock, so
# it may only fetch from crates.io and read inside harness/. Refuse anything
# else before it runs.
vendor_inputs_ok() {
  # A real TOML parser, not grep: quoting and escapes cannot hide a key.
  python3 - "${ROOT_DIR}" <<'PY' || die "vendor refused (see above); nothing was vendored"
import os, sys, tomllib
root = sys.argv[1]
harness = os.path.join(root, "harness")
CRATES_IO = "registry+https://github.com/rust-lang/crates.io-index"
bad = [] if not os.path.islink(harness) else ["harness/ is a symlink"]
lock = os.path.join(harness, "protocol-adapter", "Cargo.lock")
try:
    with open(lock, "rb") as f:
        pkgs = tomllib.load(f).get("package", [])
except (OSError, tomllib.TOMLDecodeError) as e:
    pkgs = []; bad.append(f"Cargo.lock unreadable: {e}")
for p in pkgs:
    if "source" in p and p["source"] != CRATES_IO:
        bad.append(f"Cargo.lock: {p.get('name')} has source {p['source']!r}, not crates.io")
def walk(node, rel, where, top):
    if isinstance(node, list):
        for v in node: walk(v, rel, where, False)
        return
    if not isinstance(node, dict):
        return
    for k, v in node.items():
        if top and k in ("patch", "replace"):
            bad.append(f"{where}: has a [{k}] section")
        if k in ("registry", "registry-index") and v != "crates-io":
            bad.append(f"{where}: {k} = {v!r} is not crates-io")
        paths = [v] if k in ("path", "workspace", "build") and isinstance(v, str) else \
                [x for x in v if isinstance(x, str)] if k == "members" and isinstance(v, list) else []
        for p in paths:
            real = os.path.realpath(os.path.join(rel, p))
            if not real.startswith(harness + os.sep):
                bad.append(f"{where}: {k} = {p!r} resolves outside harness/ ({real})")
        walk(v, rel, where, False)
for d, dirs, files in os.walk(harness):
    dirs[:] = [x for x in dirs if x != "target"]
    if "Cargo.toml" in files:
        t = os.path.join(d, "Cargo.toml")
        where = os.path.relpath(t, root)
        try:
            with open(t, "rb") as f:
                walk(tomllib.load(f), d, where, True)
        except (OSError, tomllib.TOMLDecodeError) as e:
            bad.append(f"{where}: unreadable: {e}")
for b in bad:
    print(f"engine/build.sh: vendor refused: {b}", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
}

# Kill every sandbox process and prove none is left, so nothing the build
# started can race the copies below or outlive the build.
sandbox_quiet() {
  sudo -n -u "${SANDBOX_USER}" -- kill -KILL -1 2>/dev/null || true
  local rc=0
  pgrep -u "${SANDBOX_USER}" >/dev/null || rc=$?
  [[ ${rc} -eq 1 ]] || die "sandbox user ${SANDBOX_USER} still has processes after the build (pgrep rc=${rc})"
}

# 1. adapter
log "building the protocol adapter"
if [[ -z "${SANDBOX_USER}" ]]; then
  ENGINE_CUDA_ARCH="${ARCH}" cargo build --release --locked \
    --manifest-path "${MANIFEST}"
  "${ROOT_DIR}/tools/stage-cuda-engine.sh"
else
  vendor_inputs_ok
  log "vendoring crates into ${JOB}/vendor"
  # cargo vendor prints the source-replacement config on stdout; it goes into
  # the sandbox CARGO_HOME, never into the repo.
  cargo vendor --locked --manifest-path "${MANIFEST}" "${JOB}/vendor" \
    > "${JOB}/cargo-home/config.toml"
  odd="$(find "${JOB}/vendor" -type l -o ! -type f ! -type d | head -5)"
  [[ -z "${odd}" ]] || die "vendor refused: the vendored tree holds a symlink or special file: ${odd//$'\n'/; }"
  chmod -R g+rX,o-rwx "${JOB}/vendor"
  log "building as ${SANDBOX_USER}, offline"
  # prlimit runs INSIDE the sudo, so the limits bind the sandbox user's build.
  sudo -n -u "${SANDBOX_USER}" -- prlimit --nproc="${CARGO_NPROC_LIMIT:-4096}" \
    --as="${CARGO_AS_LIMIT:-68719476736}" env PATH="${RUST_DIR:-/opt/rust}/cargo/bin:/usr/bin:/bin" \
    HOME="${JOB}/cargo-home" RUSTUP_HOME="${RUST_DIR:-/opt/rust}/rustup" CARGO_HOME="${JOB}/cargo-home" \
    CARGO_TARGET_DIR="${JOB}/target" ENGINE_CUDA_ARCH="${ARCH}" \
    cargo build --release --locked --offline --manifest-path "${MANIFEST}"
  sandbox_quiet
  CUDA_ENGINE_EXECUTABLE="${JOB}/target/release/cuda-engine" \
    "${ROOT_DIR}/tools/stage-cuda-engine.sh"
fi

# 2. container by digest
if ! docker image inspect "${IMAGE_REF}" >/dev/null 2>&1; then
  [[ "${ENGINE_BUILD_OFFLINE:-0}" != "1" ]] \
    || die "the pinned engine image ${IMAGE_REF} is not on this box and ENGINE_BUILD_OFFLINE=1 forbids pulling it"
  log "pulling ${IMAGE_REF}"
  docker pull "${IMAGE_REF}"
fi
log "engine image present: ${IMAGE_REF}"

# 3. overlay
"${ROOT_DIR}/tools/stage-vllm-overlay.sh"

# 4. optional participant kernels
KERNELS_SRC="${ROOT_DIR}/harness/kernels"
rm -rf "${ROOT_DIR}/.build/kernels"
mkdir -p "${ROOT_DIR}/.build/kernels"
if [[ -f "${KERNELS_SRC}/build.sh" ]]; then
  log "building harness/kernels/ inside the pinned container (no network)"
  if [[ -z "${SANDBOX_USER}" ]]; then
    KERNELS_OUT="${ROOT_DIR}/.build/kernels"
    RUN_ARGS=(--user "$(id -u):$(id -g)" --mount "type=bind,src=${ROOT_DIR},dst=${ROOT_DIR}")
  else
    KERNELS_OUT="${JOB}/kernels-out"
    SHARE_GID="$(getent group "${SHARE_GROUP:-nemoshare}" | cut -d: -f3)"
    [[ -n "${SHARE_GID}" ]] || die "group ${SHARE_GROUP:-nemoshare} does not exist"
    RUN_ARGS=(--user "$(id -u "${SANDBOX_USER}"):${SHARE_GID}"
      --mount "type=bind,src=${ROOT_DIR},dst=${ROOT_DIR},readonly"
      --mount "type=bind,src=${KERNELS_OUT},dst=${KERNELS_OUT}"
      # GB10: 20 cores, 128 GB unified memory.
      --cpus "${KERNELS_CPUS:-16}" --memory "${KERNELS_MEMORY:-64g}"
      --pids-limit "${KERNELS_PIDS_LIMIT:-4096}"
      # As serve-up.sh's sandboxed resident, with no off switch: a compile needs
      # no capability and no setuid gain.
      --cap-drop ALL --security-opt no-new-privileges)
  fi
  docker run --rm --gpus all --network none "${RUN_ARGS[@]}" \
    --env HOME=/tmp --env "KERNELS_OUT=${KERNELS_OUT}" \
    --env "TORCH_CUDA_ARCH_LIST=$(printf '%s' "${ARCH#sm_}" | sed 's/^\(.*\)\(.\)$/\1.\2/')a" \
    --env "PYTHONPATH=${ROOT_DIR}/.build/vllm-overlay" \
    --workdir "${KERNELS_SRC}" --entrypoint bash "${IMAGE_REF}" ./build.sh
  if [[ -n "${SANDBOX_USER}" ]]; then
    sandbox_quiet
    # -rP copies a symlink as a symlink and never follows it; the COPY is then
    # checked, so a swap between any check and the copy cannot slip through.
    # --preserve=mode keeps a setuid/setgid bit, and the copy is runner-owned, so
    # such a regular file is refused too (a setgid directory is harmless).
    cp -rP --preserve=mode "${KERNELS_OUT}/." "${ROOT_DIR}/.build/kernels/" \
      || { rm -rf "${ROOT_DIR}/.build/kernels"; die "kernels refused: copying ${KERNELS_OUT} failed"; }
    odd="$(find "${ROOT_DIR}/.build/kernels" \( -type l -o ! -type f ! -type d -o -type f -perm /6000 \) -print -quit)"
    if [[ -n "${odd}" ]]; then
      rm -rf "${ROOT_DIR}/.build/kernels"
      die "kernels refused: harness/kernels/build.sh produced a symlink, special file or setuid/setgid file: ${odd//$'\n'/; }"
    fi
    # The copy keeps the build's modes and ACLs; the sandbox user must not be able to write it.
    if command -v setfacl >/dev/null; then setfacl -bR "${ROOT_DIR}/.build/kernels"; fi
    chmod -R u=rwX,go=rX "${ROOT_DIR}/.build/kernels"
  fi
else
  log "no harness/kernels/build.sh; nothing to build"
fi

if [[ -n "${SANDBOX_USER}" ]]; then
  # Measurement reads only the runner-owned copies under .build. Remove the
  # sandbox-owned build trees, and keep their parent dirs read-only to the
  # sandbox group. This leaves no build directory for cross-leg state.
  for sub in cargo-home target kernels-out; do
    path="${JOB}/${sub}"
    sudo -n -u "${SANDBOX_USER}" -- chmod -R u+rwX "${path}" >/dev/null 2>&1 || true
    sudo -n -u "${SANDBOX_USER}" -- find -P "${path}" -mindepth 1 -delete >/dev/null 2>&1 || true
    find -P "${path}" -mindepth 1 -delete >/dev/null 2>&1 || true
    count="$(find -P "${path}" -mindepth 1 -printf . 2>/dev/null | wc -c)" \
      || die "cannot verify the empty sandbox build directory ${sub}"
    (( count == 0 )) || die "sandbox build directory ${sub} is not empty after staging (${count} entries remain)"
    chmod 2750 "${path}"
  done

  # These are the only build outputs used during measurement. Strip inherited
  # ACLs and write bits that participant files can carry into the staged copy.
  outputs=("${ROOT_DIR}/.build/release" "${ROOT_DIR}/.build/vllm-overlay" "${ROOT_DIR}/.build/kernels")
  for path in "${outputs[@]}"; do [[ -d "${path}" ]] || die "staged output is missing: ${path}"; done
  if command -v setfacl >/dev/null; then setfacl -bR "${outputs[@]}"; fi
  chmod -R u=rwX,go=rX "${outputs[@]}"
  odd="$(find -P "${outputs[@]}" ! -user "$(id -u)" -print -quit)"
  [[ -z "${odd}" ]] || die "a staged measurement output is not owned by $(id -un): ${odd}"
  odd="$(find -P "${outputs[@]}" -perm /022 -print -quit)"
  [[ -z "${odd}" ]] || die "a staged measurement output remains group- or other-writable: ${odd}"
fi
log "engine built and staged"

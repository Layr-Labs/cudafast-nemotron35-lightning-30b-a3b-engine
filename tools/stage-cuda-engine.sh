#!/usr/bin/env bash
# Stage the scored CUDA engine into the location the benchmarker (benchd)
# resolves.
#
# WHY THIS EXISTS. benchd resolves the scored engine at a FIXED workspace-
# relative path -- <workspace>/.build/release/mlxfast-runtime-worker -- and
# spawns it, speaking Engine Protocol v1 over stdio. That path is a CONTRACT
# with the benchmarker, not a preference: the resolution rule lives benchd-side
# (the pinned prebuilt / dist channel), so this repository honours the path
# rather than renaming it.
#
# WHAT SITS THERE NOW. The Engine Protocol v1 adapter -- the `cuda-engine`
# binary built from harness/protocol-adapter. On the scored path it attaches to
# the leg's vLLM resident over the socket benchd injects; it loads no weights
# and links no CUDA itself. The name is retained from the MLX era on purpose:
# benchd's resolver spawns exactly this path.
#
# This step copies the FINISHED adapter binary into .build/release under the
# name benchd resolves. It runs AFTER the cargo build (tools/engine/build.sh).
#
# Usage:
#   tools/stage-cuda-engine.sh                # stage the built adapter
#   tools/stage-cuda-engine.sh --print-shim   # print the sandbox shim (needs the SANDBOX_* env)
# Env: CUDA_ENGINE_CARGO_PROFILE (release), CUDA_ENGINE_EXECUTABLE (a pre-built
# binary), SANDBOX_USER / SANDBOX_ROOT / SANDBOX_JOB with RUNNER_TEMP (stage
# behind the sandbox shim; see below).
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
cd "${ROOT_DIR}"

repository_path() {
  local candidate="$1"
  if [[ "${candidate}" == /* ]]; then
    printf '%s\n' "${candidate}"
  else
    printf '%s/%s\n' "${ROOT_DIR}" "${candidate#./}"
  fi
}

BUILD_CONFIGURATION="${CUDA_ENGINE_CARGO_PROFILE:-release}"

# SOURCE -- the adapter's cargo target directory. Resolved the same way an
# operator override is honoured, so a pre-built binary can be pointed at
# directly.
CUDA_ENGINE_BIN="$(repository_path \
  "${CUDA_ENGINE_EXECUTABLE:-harness/protocol-adapter/target/${BUILD_CONFIGURATION}/cuda-engine}")"

# DESTINATION -- the FIXED location benchd resolves. Not env-overridable: it is
# a contract with the benchmarker, not a preference. The name is retained from
# the MLX era on purpose -- benchd's resolver still spawns exactly this path.
STAGED_ENGINE_BIN="$(repository_path ".build/${BUILD_CONFIGURATION}/mlxfast-runtime-worker")"

# THE SANDBOX SHIM (see below). benchd spawns the engine with a CLEARED
# environment, so everything the wrapper needs is baked in as literal exports,
# validated here first. PATH is fixed too (SANDBOX_ENGINE_PATH is only a test
# seam for stubbed sudo/pgrep). `--print-shim` prints the expected content so
# tools/nemotron35-measure-and-score.sh can byte-check the staged shim.
sandbox_shim() {
  local user="${SANDBOX_USER:-}" root="${SANDBOX_ROOT:-}" job="${SANDBOX_JOB:-}" log_dir="${RUNNER_TEMP:-}"
  [[ "${user}" =~ ^[a-z_][a-z0-9_-]{0,31}$ ]] \
    || { echo "stage-cuda-engine.sh: SANDBOX_USER '${user}' is not a plain user name" >&2; return 1; }
  [[ "${job}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] \
    || { echo "stage-cuda-engine.sh: SANDBOX_JOB '${job}' is not [A-Za-z0-9._-]" >&2; return 1; }
  [[ "${root}" == /* ]] \
    || { echo "stage-cuda-engine.sh: SANDBOX_ROOT '${root}' is not an absolute path" >&2; return 1; }
  if [[ -z "${log_dir}" || -L "${log_dir}" || ! -d "${log_dir}" \
        || "$(stat -c %U -- "${log_dir}")" != "$(id -un)" ]]; then
    echo "stage-cuda-engine.sh: RUNNER_TEMP '${log_dir}' is not a directory owned by $(id -un); the adapter's stderr file needs one" >&2
    return 1
  fi
  printf '#!/usr/bin/env bash\nset -euo pipefail\n'
  printf 'export %s=%q\n' \
    PATH "${SANDBOX_ENGINE_PATH:-/usr/sbin:/usr/bin:/sbin:/bin}" \
    SANDBOX_USER "${user}" SANDBOX_ROOT "${root}" SANDBOX_JOB "${job}" \
    SANDBOX_ENGINE_LOG_DIR "${log_dir}" \
    SANDBOX_ENGINE_BIN "${STAGED_ENGINE_BIN}.sandboxed"
  printf 'exec %q "$@"\n' "${ROOT_DIR}/tools/run-engine-sandboxed.sh"
}
if [[ "${1:-}" == "--print-shim" ]]; then
  sandbox_shim
  exit
fi

if [[ ! -x "${CUDA_ENGINE_BIN}" ]]; then
  echo "stage-cuda-engine.sh: cuda-engine binary missing or not executable: ${CUDA_ENGINE_BIN}" >&2
  echo "stage-cuda-engine.sh: build it first:" >&2
  echo "  tools/engine/build.sh   (or: cargo build --release --manifest-path harness/protocol-adapter/Cargo.toml)" >&2
  exit 1
fi

# SANDBOX (docs/sandbox-design.md): with SANDBOX_USER set the source was built
# by that user, and tools/engine/build.sh has killed every one of its
# processes first. Copy it only if it is a regular file that user owns, so a
# symlink or hard link cannot make this copy expose a file only we can read.
if [[ -n "${SANDBOX_USER:-}" ]]; then
  if [[ -L "${CUDA_ENGINE_BIN}" || ! -f "${CUDA_ENGINE_BIN}" \
        || "$(stat -c %U -- "${CUDA_ENGINE_BIN}")" != "${SANDBOX_USER}" ]]; then
    echo "stage-cuda-engine.sh: refusing ${CUDA_ENGINE_BIN}: not a regular file owned by ${SANDBOX_USER}" >&2
    exit 1
  fi
  SHIM="$(sandbox_shim)" || exit 1
fi

# SANDBOX SHIM. benchd resolves the serial-control leg's engine at the SAME
# root-relative path inside the reference workspace, so --engine must stay the
# normal staged path. With SANDBOX_USER set the real binary goes next to it as
# <staged>.sandboxed, and the staged path holds a trusted shim that hands it to
# tools/run-engine-sandboxed.sh (which runs it as SANDBOX_USER). exec keeps
# benchd's stdin/stdout -- the Engine Protocol pipes -- and the exit status.
REAL_ENGINE_BIN="${STAGED_ENGINE_BIN}"
if [[ -n "${SANDBOX_USER:-}" ]]; then
  REAL_ENGINE_BIN="${STAGED_ENGINE_BIN}.sandboxed"
fi

mkdir -p "$(dirname "${STAGED_ENGINE_BIN}")"

# `-ef` is true only when both paths already resolve to the same file, which
# happens when an operator override points the source straight at the benchd
# path -- then the copy is a no-op (and cp would error copying a file onto
# itself).
if [[ ! "${CUDA_ENGINE_BIN}" -ef "${REAL_ENGINE_BIN}" ]]; then
  if [[ -n "${SANDBOX_USER:-}" ]]; then
    # --no-dereference: a symlink swapped in after the check above is copied as
    # a symlink, and the destination check below refuses it.
    rm -f "${REAL_ENGINE_BIN}"
    cp --no-dereference --preserve=mode "${CUDA_ENGINE_BIN}" "${REAL_ENGINE_BIN}"
  else
    cp -f "${CUDA_ENGINE_BIN}" "${REAL_ENGINE_BIN}"
  fi
fi
if [[ -n "${SANDBOX_USER:-}" ]] && [[ -L "${REAL_ENGINE_BIN}" || ! -f "${REAL_ENGINE_BIN}" ]]; then
  rm -f "${REAL_ENGINE_BIN}"
  echo "stage-cuda-engine.sh: refusing the staged copy ${REAL_ENGINE_BIN}: not a regular file (the source changed after it was checked)" >&2
  exit 1
fi

if [[ -n "${SANDBOX_USER:-}" ]]; then
  chmod 0755 "${REAL_ENGINE_BIN}"
  rm -f "${STAGED_ENGINE_BIN}"
  printf '%s\n' "${SHIM}" > "${STAGED_ENGINE_BIN}"
  chmod 0755 "${STAGED_ENGINE_BIN}"
  echo "stage-cuda-engine.sh: staged ${REAL_ENGINE_BIN} (cuda-engine) and the sandbox shim ${STAGED_ENGINE_BIN} for benchd (runs it as ${SANDBOX_USER})"
  exit 0
fi

# benchd checks the execute bit on the resolved binary; guarantee it survives
# the copy regardless of the caller's umask.
chmod +x "${STAGED_ENGINE_BIN}"

echo "stage-cuda-engine.sh: staged ${STAGED_ENGINE_BIN} (cuda-engine, Engine Protocol v1 adapter for the vLLM resident) for benchd"

#!/usr/bin/env bash
# The vendoring proof: a participant may EDIT THE ENGINE, and the gates agree.
#
# The engine is vLLM's Python package, VENDORED at vllm/vllm/ and declared in
# benchmark.json editablePaths, plus optional CUDA/C++ kernels under
# harness/kernels/. This suite keeps that true. It asserts:
#
#   1. THE ENGINE IS IN THE SURFACE. benchmark.json editablePaths carries vllm
#      and harness, and nothing else but the MTP declaration.
#   2. THE HARNESS HASH RECORDS ENGINE EDITS. benchd seals a hash over a FIXED
#      root set that includes vllm/ and harness/, so an engine edit or a kernel
#      edit is visible in the sealed provenance; a docs-only edit outside the
#      roster is not.
#   3. AN ENGINE EDIT REACHES THE STAGED OVERLAY. tools/stage-vllm-overlay.sh
#      layers vllm/vllm/ over the pinned build: the participant's source wins,
#      and the base's compiled extensions survive.
#
# THE ROSTER BELOW IS A HAND-COPY of benchd's HARNESS_HASH_ROOTS
# (crates/bench-core/src/harness_hash.rs). benchd owns the real one; if the two
# drift, this suite is testing a hash nobody computes.
#
# Hermetic: no weights, no GPU, no docker, no network. The overlay case uses
# the stage script's STAGE_VLLM_OVERLAY_BASE seam in place of the container.
#
# Usage: tools/test-participant-engine-edit.sh [-v]
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
VERBOSE=0
[[ "${1:-}" == "-v" ]] && VERBOSE=1

EXPECTED_MIN_ASSERTIONS=9
PASSED=0; FAILED=0; FAILURES=()
pass() { PASSED=$((PASSED+1)); [[ "${VERBOSE}" == "1" ]] && echo "ok    $1"; return 0; }
fail() { FAILED=$((FAILED+1)); FAILURES+=("$1"); echo "FAIL  $1" >&2; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/participant-edit.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

# benchd's roster, transcribed.
harness_hash() {
  python3 - "$1" <<'PY'
import hashlib, os, sys
root = sys.argv[1]
ROOTS = ["Package.swift", "Sources", "Tests", "benchmark.json", "benchmark.sh",
         "setup.sh", "tools", "README.md", "TASK.md", "harness", "vllm"]
paths = []
for r in ROOTS:
    p = os.path.join(root, r)
    if not os.path.exists(p):
        continue
    if os.path.isdir(p):
        for dirpath, dirnames, filenames in os.walk(p):
            # hidden entries are skipped, subtree and all; symlinks are excluded
            dirnames[:] = [d for d in dirnames
                           if not d.startswith('.')
                           and not os.path.islink(os.path.join(dirpath, d))]
            for fn in filenames:
                if fn.startswith('.'):
                    continue
                fp = os.path.join(dirpath, fn)
                if os.path.islink(fp) or not os.path.isfile(fp):
                    continue
                paths.append(fp)
    else:
        paths.append(p)
h = hashlib.sha256()
for p in sorted(paths):
    try:
        with open(p, 'rb') as fh:
            data = fh.read()
    except OSError:
        continue
    h.update(p.encode()); h.update(data)
print(h.hexdigest())
PY
}

# ---------------------------------------------------------------------------
# A sandbox repository: the real manifest, a miniature engine.
# ---------------------------------------------------------------------------
SB="${WORK}/repo"
mkdir -p "${SB}/vllm/vllm" "${SB}/harness/kernels" "${SB}/tools" "${SB}/docs"
# The REAL manifest, so editablePaths is the shipping list and not a fixture.
cp "${REPO_ROOT}/benchmark.json" "${SB}/benchmark.json"
printf 'def forward():\n    return 1\n' > "${SB}/vllm/vllm/model.py"
printf 'kernel\n' > "${SB}/harness/kernels/k.cu"
printf 'tool\n' > "${SB}/tools/thing.sh"
printf 'notes\n' > "${SB}/docs/notes.md"
git -C "${SB}" init --quiet
git -C "${SB}" add -A
git -C "${SB}" -c user.email=t@t -c user.name=t commit --quiet -m base

# ---------------------------------------------------------------------------
# 1. The engine is in the surface.
# ---------------------------------------------------------------------------
if jq -e '.editablePaths == ["vllm","harness","mtp-head.manifest.json"]' "${SB}/benchmark.json" >/dev/null; then
  pass "benchmark.json editablePaths is exactly vllm, harness, mtp-head.manifest.json"
else
  fail "benchmark.json editablePaths is exactly vllm, harness, mtp-head.manifest.json (got $(jq -c .editablePaths "${SB}/benchmark.json"))"
fi

# ---------------------------------------------------------------------------
# 2. The harness hash records engine and kernel edits.
# ---------------------------------------------------------------------------
H0="$(harness_hash "${SB}")"
if [[ "${H0}" =~ ^[0-9a-f]{64}$ ]]; then
  pass "the harness hash is a sha256 over the roster"
else
  fail "the harness hash is a sha256 over the roster (got '${H0}')"
fi
for probe in vllm/vllm/model.py harness/kernels/k.cu tools/thing.sh; do
  printf '# participant tuning\n' >> "${SB}/${probe}"
  if [[ "$(harness_hash "${SB}")" != "${H0}" ]]; then
    pass "an edit to ${probe} CHANGES the harness hash"
  else
    fail "an edit to ${probe} CHANGES the harness hash"
  fi
  git -C "${SB}" checkout --quiet -- "${probe}"
done
printf 'more notes\n' >> "${SB}/docs/notes.md"
if [[ "$(harness_hash "${SB}")" == "${H0}" ]]; then
  pass "an edit outside the roster (docs/) leaves the harness hash unchanged"
else
  fail "an edit outside the roster (docs/) leaves the harness hash unchanged"
fi
git -C "${SB}" checkout --quiet -- docs/notes.md
if [[ "$(harness_hash "${SB}")" == "${H0}" ]]; then
  pass "reverting every edit returns the harness hash to its original value"
else
  fail "reverting every edit returns the harness hash to its original value"
fi

# THE TRANSCRIPTION AGAINST THE REAL BINARY. `benchd harness-hash` prints the
# hash benchd itself seals for its CWD; when a benchd is at hand (BENCHD, or
# the pair tools/fetch-benchd.sh installs) it must equal the transcription.
benchd_bin="${BENCHD:-${REPO_ROOT}/benchd-bin/benchd}"
if [[ -x "${benchd_bin}" ]]; then
  H_BENCHD="$(cd "${SB}" && "${benchd_bin}" harness-hash 2>/dev/null | head -1)"
  if [[ "${H_BENCHD}" == "${H0}" ]]; then
    pass "benchd harness-hash equals the transcribed roster's hash"
  else
    fail "benchd harness-hash equals the transcribed roster's hash (benchd ${H_BENCHD:0:12}, transcription ${H0:0:12})"
  fi
else
  echo "test-participant-engine-edit: NOTE -- no benchd binary at ${benchd_bin}; the transcription was not cross-checked against it" >&2
fi

# The transcription must match benchd's roster ORDER AND MEMBERSHIP if a benchd
# checkout is reachable. Absent one, the hand-copy stands on its comment.
BENCHD_ROSTER_SRC="${BENCHD_SRC_DIR:-}/crates/bench-core/src/harness_hash.rs"
if [[ -n "${BENCHD_SRC_DIR:-}" && -f "${BENCHD_ROSTER_SRC}" ]]; then
  want="$(sed -n '/pub const HARNESS_HASH_ROOTS/,/\];/p' "${BENCHD_ROSTER_SRC}" \
    | grep -oE '"[^"]+"' | tr -d '"' | tr '\n' ' ')"
  have="Package.swift Sources Tests benchmark.json benchmark.sh setup.sh tools README.md TASK.md harness vllm "
  if [[ "${want}" == "${have}" ]]; then
    pass "the transcribed roster matches benchd's HARNESS_HASH_ROOTS"
  else
    fail "the transcribed roster matches benchd's HARNESS_HASH_ROOTS (benchd: ${want})"
  fi
  if [[ "${want}" == *"vllm"* && "${want}" == *"harness"* ]]; then
    pass "benchd's roster includes vllm and harness (engine edits are in the sealed provenance)"
  else
    fail "benchd's roster no longer includes vllm and harness; this suite's premise has changed"
  fi
else
  echo "test-participant-engine-edit: NOTE -- BENCHD_SRC_DIR unset, the roster transcription was not cross-checked" >&2
fi

# ---------------------------------------------------------------------------
# 3. In THIS repository: an engine edit reaches the staged overlay, and the
#    base's compiled extensions survive the layering.
# ---------------------------------------------------------------------------
OV="${WORK}/overlay-repo"
mkdir -p "${OV}/tools" "${OV}/fixtures" "${OV}/vllm/vllm/model_executor"
cp "${REPO_ROOT}/tools/stage-vllm-overlay.sh" "${OV}/tools/"
cp "${REPO_ROOT}/fixtures/nemotron35_lightning_30b_a3b_track.json" "${OV}/fixtures/"
printf 'EDITED = True\n' > "${OV}/vllm/vllm/model_executor/nemotron_probe.py"
printf '__version__ = "tree"\n' > "${OV}/vllm/vllm/__init__.py"
BASE="${WORK}/base"
mkdir -p "${BASE}/vllm/model_executor"
printf 'ELF' > "${BASE}/vllm/_C.abi3.so"
printf 'EDITED = False\n' > "${BASE}/vllm/model_executor/nemotron_probe.py"
printf 'generated\n' > "${BASE}/vllm/_version.py"
if STAGE_VLLM_OVERLAY_BASE="${BASE}" "${OV}/tools/stage-vllm-overlay.sh" > "${WORK}/overlay.out" 2>&1; then
  staged="${OV}/.build/vllm-overlay/vllm"
  if grep -q 'EDITED = True' "${staged}/model_executor/nemotron_probe.py"; then
    pass "a vllm/vllm/ edit is what the staged overlay carries"
  else
    fail "a vllm/vllm/ edit is what the staged overlay carries"
  fi
  if [[ -f "${staged}/_C.abi3.so" && -f "${staged}/_version.py" ]]; then
    pass "the base's compiled extensions and generated files survive the layering"
  else
    fail "the base's compiled extensions and generated files survive the layering"
  fi
else
  fail "tools/stage-vllm-overlay.sh stages from a local base ($(tail -2 "${WORK}/overlay.out"))"
fi

echo "participant-engine-edit: ${PASSED} passed, ${FAILED} failed"
if [[ "${FAILED}" -ne 0 ]]; then
  printf '  - %s\n' "${FAILURES[@]}" >&2
  exit 1
fi
if [[ "${PASSED}" -lt "${EXPECTED_MIN_ASSERTIONS}" ]]; then
  echo "participant-engine-edit: ran only ${PASSED} assertions, expected at least ${EXPECTED_MIN_ASSERTIONS} -- the suite shrank" >&2
  exit 1
fi
exit 0

#!/usr/bin/env bash
#
# test-ranked-box-preflight-env.sh -- the ranked preflight's environment gate.
#
# tools/ranked-box-preflight.sh section 1 refuses a CREDENTIAL in the ranked
# job's environment, and section 2 refuses a MEASUREMENT-WEAKENING OVERRIDE.
# Both lists are just names in a `for` loop, and a name that falls off one is
# invisible: the run proceeds, measures something else, and seals a normal-
# looking artifact. So the lists are pinned here, one case per name, by running
# the REAL preflight with that one variable set and requiring a refusal that
# names it.
#
# The control case runs with none of them set and requires the preflight to get
# PAST section 2. Whatever it does after that is not this suite's business --
# the later sections need staged box assets.
#
# Hermetic: no toolchain, weights, GPU, benchd binary, staged golden or network.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFLIGHT="${REPO_ROOT}/tools/ranked-box-preflight.sh"

failures=0
fail() {
  echo "FAIL: $*" >&2
  failures=$((failures + 1))
}

# Every name the two gates refuse. The environment is cleared of all of them for
# each case, so one case sets exactly one.
CREDENTIALS=(R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY MLXFAST_NEMOTRON35_R2_DOWNLOADER BENCHD_DIST_TOKEN)
OVERRIDES=(BENCHD MLXFAST_SKIP_WEIGHTS_DOWNLOAD SKIP_MODEL_DOWNLOAD MLXFAST_LOCAL_COOL_GATE
           MLXFAST_LOCAL_ALLOW_GOLDEN_DRIFT MLXFAST_SKIP_ENGINE_BUILD MLXFAST_SKIP_WEIGHTS_SHA256
           SERVE_UP_LAUNCHER SERVE_UP_NO_CAP_DROP SERVE_UP_ALLOW_NEW_PRIVS
           STAGE_VLLM_OVERLAY_BASE ENGINE_EOS_IDS
           DS4_RESIDENT_SOCKET BENCH_WORKER_RESIDENT_SOCKET)
# Engine-configuration prefixes the preflight refuses as a family; one sample
# name per prefix.
ENGINE_PREFIXED=(RESIDENT_SPEC_DEPTH VLLM_ATTENTION_BACKEND HF_TOKEN TRANSFORMERS_CACHE
                 TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR)

clear_args=()
for var in "${CREDENTIALS[@]}" "${OVERRIDES[@]}" "${ENGINE_PREFIXED[@]}"; do
  clear_args+=(-u "${var}")
done
# The runner's own environment may carry engine-prefixed names (a developer
# shell with HF_HOME set); clear every one so the control sees a clean job.
while IFS= read -r name; do
  clear_args+=(-u "${name}")
done < <(env | cut -d= -f1 | grep -E '^(RESIDENT_|VLLM_|HF_|TRANSFORMERS_|TORCHINDUCTOR_|TRITON_)' || true)

# The control: with a clean environment the preflight reaches section 3.
control_out="$(env "${clear_args[@]}" "${PREFLIGHT}" 2>&1)"
if ! grep -q 'no R2 credential, signer, or dist token in the job environment' <<<"${control_out}"; then
  fail "control: the preflight did not pass the credential gate on a clean environment; got: ${control_out}"
fi
if ! grep -q 'no measurement-weakening override in the job environment' <<<"${control_out}"; then
  fail "control: the preflight did not pass the override gate on a clean environment; got: ${control_out}"
fi

# One case per name: set it, and require a refusal that names it.
check_refusal() {
  local var="$1" expected="$2" out rc
  out="$(env "${clear_args[@]}" "${var}=set-by-the-test" "${PREFLIGHT}" 2>&1)"
  rc=$?
  if [[ "${rc}" -eq 0 ]]; then
    fail "${var}: the preflight exited 0 with ${var} set in the environment"
    return
  fi
  if ! grep -q "REFUSING -- ${var} is set" <<<"${out}"; then
    fail "${var}: the refusal does not name it; got: $(tail -2 <<<"${out}")"
    return
  fi
  if ! grep -q "${expected}" <<<"${out}"; then
    fail "${var}: refused for the wrong reason (expected '${expected}'); got: $(tail -2 <<<"${out}")"
  fi
}

for var in "${CREDENTIALS[@]}"; do
  check_refusal "${var}" 'this job must hold no credential'
done
for var in "${OVERRIDES[@]}"; do
  check_refusal "${var}" 'it weakens or bypasses what the ranked run measures'
done
for var in "${ENGINE_PREFIXED[@]}"; do
  out="$(env "${clear_args[@]}" "${var}=set-by-the-test" "${PREFLIGHT}" 2>&1)"
  if [[ $? -eq 0 ]] || ! grep -q "engine configuration preset in the ranked job's environment: .*${var}" <<<"${out}"; then
    fail "${var}: an engine-prefixed variable was not refused by name; got: $(tail -2 <<<"${out}")"
  fi
done

# ---------------------------------------------------------------------------
# THE PAIRED, PER-BOX BASELINE GATE (section 8).
#
# Section 8 refuses a ranked box whose REFERENCE TREE is not the pinned one, or
# whose CALIBRATION FILE does not describe this box against that tree. Every one
# of those checks is a line that can silently stop refusing, and the consequence
# is not a failed run: it is a scored run whose serial control ran on the wrong
# engine, or was gated by a band from another machine.
#
# Sections 1 and 2 refuse on an environment variable alone, so the cases above
# need no box. Section 8 sits behind sections 2b to 7, which want a temperature
# reader, a staged benchd pair and a toolchain. So this half builds the SMALLEST
# scaffold that lets the REAL preflight reach section 8 -- stub docker/nvidia-smi,
# a fake benchd beside a manifest that describes it, and a clone of this
# repository at the fixture's own baseline_reference_commit -- and then makes ONE
# thing wrong per case.
#
# Still hermetic: no GPU, no weights, no engine, no network. The clone is a
# --shared local clone of this repository.
scaffold_failures_before="${failures}"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

FIXTURE="${REPO_ROOT}/fixtures/nemotron35_lightning_30b_a3b_track.json"
TRACK_ID="$(jq -r '.track_id' "${FIXTURE}")"
REF_COMMIT="$(jq -r '.baseline_reference_commit' "${FIXTURE}")"
# The shipped fixture declares live_goldens. Most cases below run the scaffold
# in the per-job rotation shape over the same prompts (live_golden = the first
# one); the live_goldens cases near the end run the shipped shape.
mapfile -t LIVE_NAMES < <(jq -r '.live_goldens[]' "${FIXTURE}")
LIVE_GOLDEN="${LIVE_NAMES[0]}"
SECOND="${LIVE_NAMES[1]}"
IMAGE_REF="$(jq -r '.serve_configuration.container_image | split(":")[0]' "${FIXTURE}")@$(jq -r '.serve_configuration.container_digest' "${FIXTURE}")"
DRIVER_PIN="$(jq -r '.serve_configuration.toolchain.driver_min' "${FIXTURE}")"
RUNNER="test-box-nemotron35-cuda"

# The scaffold is only buildable where the pinned reference commit is reachable
# from this checkout. That is the normal case (it is a commit of this
# repository); when it is not, the suite says so rather than passing quietly.
if ! git -C "${REPO_ROOT}" cat-file -e "${REF_COMMIT}^{commit}" 2>/dev/null; then
  echo "SKIP: the fixture's baseline_reference_commit ${REF_COMMIT} is not in this checkout, so the section 8 scaffold cannot be built here" >&2
  scaffold_ready=0
else
  scaffold_ready=1
fi

if [[ "${scaffold_ready}" == "1" ]]; then
  # --- the submission root ---------------------------------------------------
  # THE PINNED GOLDENS ARE NOT IN ANY TREE. They are organizer material
  # published in R2 and staged on the box, so this suite stages a SYNTHETIC pool
  # and re-pins a COPY of the contract to it. Everything else in the root is
  # this checkout, entry by entry, so the preflight under test is the real one
  # reading the real serve script, the real spec declaration and the real
  # engine tree.
  #
  # tools/ must be a real directory holding symlinks rather than a symlink to
  # tools/: the preflight derives its repository root from its own PHYSICAL
  # location, and a symlinked tools/ would resolve back to this checkout and
  # read the shipped contract instead of the re-pinned copy.
  ROOT="${WORK}/root"
  mkdir -p "${ROOT}/tools" "${ROOT}/fixtures"
  for entry in "${REPO_ROOT}"/tools/*; do
    [[ "$(basename "${entry}")" == box-runner ]] && continue
    ln -s "${entry}" "${ROOT}/tools/$(basename "${entry}")"
  done
  # Section 9 runs tools/box-runner/setup-sandbox-user.sh --check, which needs
  # the real nemosub user, sudoers and firewall of a set-up box. The scaffold
  # stands in a stub that passes, or fails the way the real one does when
  # SANDBOX_CHECK_STUB=fail.
  mkdir -p "${ROOT}/tools/box-runner"
  cat > "${ROOT}/tools/box-runner/setup-sandbox-user.sh" <<'EOF'
#!/usr/bin/env bash
[[ "$1" == --check ]] || exit 78
if [[ "${SANDBOX_CHECK_STUB:-}" == fail ]]; then
  echo "PASS  user nemosub exists"
  echo "FAIL  nemosub cannot open a TCP connection to 1.1.1.1:443"
  echo "setup-sandbox-user.sh --check: 1 check(s) FAILED" >&2
  exit 1
fi
if [[ "${SANDBOX_CHECK_STUB:-}" == open ]]; then
  echo "FAIL  nemosub cannot open a TCP connection to a live listener on 127.0.0.1"
  echo "FAIL  cron refuses nemosub ('crontab -l' is not allowed, not merely empty)"
  echo "setup-sandbox-user.sh --check: 2 check(s) FAILED" >&2
  exit 1
fi
echo "setup-sandbox-user.sh --check: all checks passed"
EOF
  chmod 755 "${ROOT}/tools/box-runner/setup-sandbox-user.sh"
  for entry in "${REPO_ROOT}"/*; do
    case "$(basename "${entry}")" in
      tools|fixtures|correctness_prompts) continue ;;
    esac
    ln -s "${entry}" "${ROOT}/$(basename "${entry}")"
  done
  ln -s "${REPO_ROOT}/correctness_prompts" "${ROOT}/correctness_prompts"
  PREFLIGHT8="${ROOT}/tools/ranked-box-preflight.sh"

  # The staged pool, and the contract re-pinned to it. Only sha256 and bytes
  # move; every other field is the shipped fixture's, so the arm gate, the
  # depth envelope and baseline_reference_commit stay under test.
  GOLDEN_DIR="${WORK}/goldens"
  mkdir -p "${GOLDEN_DIR}"
  python3 - "${FIXTURE}" "${GOLDEN_DIR}" "${ROOT}/fixtures/nemotron35_lightning_30b_a3b_track.json" <<'REPINEOF'
import hashlib, json, os, sys

src, golden_dir, dst = sys.argv[1:4]
contract = json.load(open(src, encoding="utf-8"))


def stage(r2_path):
    base = r2_path.rsplit("/", 1)[-1]
    path = os.path.join(golden_dir, base)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"synthetic_golden": base}) + "\n")
    data = open(path, "rb").read()
    return hashlib.sha256(data).hexdigest(), len(data)


for entry in contract.get("timed_prompt_pool", []):
    entry["sha256"], entry["bytes"] = stage(entry["r2_path"])
for oracles in (contract.get("speculative_oracles") or {}).values():
    for entry in oracles.values():
        entry["sha256"], entry["bytes"] = stage(entry["r2_path"])
names = contract["live_goldens"]
live = next(e for e in contract["timed_prompt_pool"]
            if e["r2_path"].endswith("/" + names[0] + ".golden.json"))
contract["hidden_correctness_golden"] = {"sha256": live["sha256"], "bytes": live["bytes"]}
# The copy is ARMED: section 5 refuses an unarmed fixture, and the suite is
# about what comes after it.
contract["official_scoring_enabled"] = True
# The shipped live_goldens shape, kept for the live_goldens cases.
with open(dst + ".live-goldens", "w", encoding="utf-8") as fh:
    json.dump(contract, fh, indent=2)
    fh.write("\n")
# The per-job rotation shape over the same prompts, for every other case.
del contract["live_goldens"]
contract["live_golden"] = names[0]
contract["live_golden_rotation"] = {"mode": "per_job_random", "pool": names, "pending": []}
with open(dst, "w", encoding="utf-8") as fh:
    json.dump(contract, fh, indent=2)
    fh.write("\n")
REPINEOF
  [[ -s "${ROOT}/fixtures/nemotron35_lightning_30b_a3b_track.json" ]] \
    || { echo "FAIL: could not re-pin the contract onto the synthetic golden pool" >&2; exit 1; }

  STUB="${WORK}/bin"
  mkdir -p "${STUB}"
  # docker: the pinned image is present by digest and the nvidia runtime is
  # installed; anything else is an error, so a preflight that starts running
  # containers fails here loudly.
  cat > "${STUB}/docker" <<EOF
#!/usr/bin/env bash
case "\$1 \$2" in
  "image inspect") [[ "\$3" == "${IMAGE_REF}" ]] && exit 0; exit 1 ;;
  "info --format") printf '{"nvidia":{},"runc":{}}\n'; exit 0 ;;
esac
echo "docker stub: unexpected \$*" >&2; exit 1
EOF
  cat > "${STUB}/nvidia-smi" <<EOF
#!/usr/bin/env bash
printf '%s\n' "${DRIVER_PIN}"
EOF
  chmod 755 "${STUB}"/*

  # A benchd that matches the manifest beside it. Section 6 verifies exactly
  # that pair, and nothing here ever runs the binary.
  BENCHD_DIR="${WORK}/benchd-bin"
  mkdir -p "${BENCHD_DIR}"
  printf 'not a real benchd\n' > "${BENCHD_DIR}/benchd"
  chmod 755 "${BENCHD_DIR}/benchd"
  jq -n --arg s "$(shasum -a 256 "${BENCHD_DIR}/benchd" | awk '{print $1}')" \
        --argjson b "$(wc -c < "${BENCHD_DIR}/benchd" | tr -d '[:space:]')" \
        '{sha256: $s, bytes: $b, source_commit: "0000000000000000000000000000000000000000"}' \
    > "${BENCHD_DIR}/benchd.manifest.json"

  # The reference tree: this repository at the pinned commit, plus the build
  # outputs tools/engine/build.sh stages.
  WS="${WORK}/reference"
  git clone --quiet --shared --no-checkout "${REPO_ROOT}" "${WS}" 2>/dev/null
  git -C "${WS}" checkout --quiet --detach "${REF_COMMIT}" 2>/dev/null
  mkdir -p "${WS}/.build/vllm-overlay/vllm" "${WS}/.build/release"
  : > "${WS}/.build/vllm-overlay/vllm/__init__.py"
  : > "${WS}/.build/release/mlxfast-runtime-worker"
  chmod +x "${WS}/.build/release/mlxfast-runtime-worker"

  # A healthy calibration for this box, against that tree, captured now (after
  # the reference commit and before the clock).
  calibration_band_fields() {
    jq -n '
      {
        passes: 3,
        prefill_seconds_per_token_mean: 0.0006282488193359375,
        decode_seconds_per_token_mean: 0.0329116748046875,
        prefill_cv: 0.004,
        decode_cv: 0.002,
        prefill_band_low: 0.95,
        prefill_band_high: 1.05,
        decode_band_low: 0.98,
        decode_band_high: 1.02
      }'
  }

  # Version 3: a header and one prompts[] entry per prompt. Its decode mean is
  # the decode window per token (decode-run time / N, no seed prefill).
  write_calibration() { # write_calibration PATH [jq filter]
    local path="$1" filter="${2:-.}"
    jq -n --arg track "${TRACK_ID}" --arg box "${RUNNER}" --arg ref "${REF_COMMIT}" \
          --arg prompt "${LIVE_GOLDEN}" --arg at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
          --argjson band "$(calibration_band_fields)" '
      {
        version: 3,
        track_id: $track,
        box: $box,
        reference_commit: $ref,
        captured_at: $at,
        benchd_source_commit: "1111111111111111111111111111111111111111",
        gates: [],
        prompts: [({prompt: $prompt} + $band)]
      }' | jq "${filter}" > "${path}"
  }

  CAL_OK="${WORK}/baseline-calibration.json"
  write_calibration "${CAL_OK}"

  # Run the REAL preflight with the scaffold, overriding one thing at a time.
  # KEY=VALUE arguments are set; KEY= clears one.
  run_preflight() {
    env "${clear_args[@]}" \
        PATH="${STUB}:${PATH}" \
        MLXFAST_GPU_TEMP_CMD="echo 42" \
        BENCHD_BIN_DIR="${BENCHD_DIR}" \
        RUNNER_NAME="${RUNNER}" \
        MLXFAST_NEMOTRON35_GOLDEN_DIR="${GOLDEN_DIR}" \
        MLXFAST_BASELINE_WORKSPACE="${WS}" \
        MLXFAST_BASELINE_CALIBRATION="${CAL_OK}" \
        "$@" "${PREFLIGHT8}" 2>&1
  }

  expect_refusal() { # expect_refusal LABEL NEEDLE [env overrides...]
    local label="$1" needle="$2"; shift 2
    local out rc
    out="$(run_preflight "$@")"
    rc=$?
    if [[ "${rc}" -eq 0 ]]; then
      fail "section 8 / ${label}: the preflight exited 0"
      return
    fi
    if ! grep -qF "${needle}" <<<"${out}"; then
      fail "section 8 / ${label}: the refusal does not say '${needle}'; got: $(tail -2 <<<"${out}")"
      return
    fi
    echo "ok    section 8 / ${label}: refused by name"
  }

  # The control: the whole scaffold is right, so the preflight passes EVERY
  # check. Without this, a section 8 that refused unconditionally would pass
  # every case above it.
  control8="$(run_preflight)"
  if grep -q 'ranked-box-preflight: all checks passed' <<<"${control8}" \
    && grep -q 'warm-up golden matches its contract pin' <<<"${control8}"; then
    echo "ok    section 8 / control: a correctly staged box passes every check"
  else
    fail "section 8 / control: a correctly staged box did not pass; got: $(tail -3 <<<"${control8}")"
  fi

  expect_refusal "workspace unset" \
    "MLXFAST_BASELINE_WORKSPACE is not set" MLXFAST_BASELINE_WORKSPACE=
  expect_refusal "calibration unset" \
    "MLXFAST_BASELINE_CALIBRATION is not set" MLXFAST_BASELINE_CALIBRATION=
  expect_refusal "workspace absent" \
    "is not a directory on this box" "MLXFAST_BASELINE_WORKSPACE=${WORK}/absent"
  # The candidate tree is the root the preflight runs FROM, which is ${ROOT}
  # here, so that is the value the refusal has to recognise.
  expect_refusal "workspace is the candidate" \
    "points at this checkout" "MLXFAST_BASELINE_WORKSPACE=${ROOT}"
  # The staged goldens. They are not in any checkout, so an unset name has
  # nothing to fall back on and an extra *.json changes the measured cohort.
  expect_refusal "goldens unstaged" \
    "MLXFAST_NEMOTRON35_GOLDEN_DIR is unset" MLXFAST_NEMOTRON35_GOLDEN_DIR=
  expect_refusal "goldens directory absent" \
    "does not exist or is not a directory" "MLXFAST_NEMOTRON35_GOLDEN_DIR=${WORK}/no-such-goldens"
  expect_refusal "runner unnamed" \
    "has no RUNNER_NAME" RUNNER_NAME=

  # A reference tree at the wrong commit.
  WS_OLD="${WORK}/reference-old"
  git clone --quiet --shared --no-checkout "${REPO_ROOT}" "${WS_OLD}" 2>/dev/null
  if git -C "${WS_OLD}" checkout --quiet --detach HEAD 2>/dev/null; then
    mkdir -p "${WS_OLD}/.build/vllm-overlay/vllm" "${WS_OLD}/.build/release"
    : > "${WS_OLD}/.build/vllm-overlay/vllm/__init__.py"
    : > "${WS_OLD}/.build/release/mlxfast-runtime-worker"
    chmod +x "${WS_OLD}/.build/release/mlxfast-runtime-worker"
    expect_refusal "workspace at the wrong commit" \
      "but the fixture pins ${REF_COMMIT}" "MLXFAST_BASELINE_WORKSPACE=${WS_OLD}"
  else
    fail "section 8 / workspace at the wrong commit: cannot check out HEAD"
  fi

  # A reference tree that was staged but never built.
  WS_UNBUILT="${WORK}/reference-unbuilt"
  git clone --quiet --shared --no-checkout "${REPO_ROOT}" "${WS_UNBUILT}" 2>/dev/null
  git -C "${WS_UNBUILT}" checkout --quiet --detach "${REF_COMMIT}" 2>/dev/null
  expect_refusal "workspace not built" \
    "its engine build is not staged" "MLXFAST_BASELINE_WORKSPACE=${WS_UNBUILT}"

  # A reference tree with an edit in it.
  WS_DIRTY="${WORK}/reference-dirty"
  cp -R "${WS}" "${WS_DIRTY}"
  printf 'edited\n' >> "${WS_DIRTY}/README.md"
  expect_refusal "workspace edited" \
    "has uncommitted changes" "MLXFAST_BASELINE_WORKSPACE=${WS_DIRTY}"

  # One bad field per calibration file.
  cal_case() { # cal_case LABEL JQ_FILTER NEEDLE
    local label="$1" filter="$2" needle="$3"
    local path
    path="${WORK}/cal-$(printf '%s' "${label}" | tr -c 'a-z0-9' '-').json"
    write_calibration "${path}" "${filter}"
    expect_refusal "${label}" "${needle}" "MLXFAST_BASELINE_CALIBRATION=${path}"
  }

  cal_case "calibration version" '.version = 4' "reads version 3 only"
  cal_case "calibration version 2 recorded the whole window as decode" '.version = 2' \
    "decode is now the decode window only (decode-run time / N)"
  cal_case "calibration version 1 recorded the whole window as decode" \
    '.version = 1 | . + .prompts[0] | del(.prompts)' "Recalibrate this box"
  cal_case "calibration prompt missing" '.prompts[0].prompt = "other-prompt"' \
    "has no prompts[] entry for prompt"
  cal_case "calibration prompt duplicated" '.prompts = [.prompts[0], .prompts[0]]' \
    "more than one prompts[] entry for prompt"
  cal_case "calibration track" '.track_id = "another-track-v1"' "names track 'another-track-v1'"
  cal_case "calibration box" '.box = "another-box"' "was captured on box 'another-box'"
  cal_case "calibration reference commit" \
    '.reference_commit = "0000000000000000000000000000000000000000"' \
    "was captured against reference commit"
  cal_case "calibration passes" '.prompts[0].passes = 1' "a band needs at least two"
  cal_case "calibration mean" '.prompts[0].decode_seconds_per_token_mean = 0' \
    "is not a plausible per-token time"
  cal_case "calibration cv" '.prompts[0].decode_cv = 0.05' \
    "above the 2.0 % stability gate (the track contract's calibration_max_cv_percent)"
  CAL_CV_PASS="${WORK}/cal-cv-1-5-percent.json"
  write_calibration "${CAL_CV_PASS}" '.prompts[0].decode_cv = 0.015'
  cv_pass="$(run_preflight "MLXFAST_BASELINE_CALIBRATION=${CAL_CV_PASS}")"
  if grep -q 'ranked-box-preflight: all checks passed' <<<"${cv_pass}"; then
    echo "ok    section 8 / calibration cv 1.5 percent: passes the 2 percent gate"
  else
    fail "section 8 / calibration cv 1.5 percent: did not pass; got: $(tail -3 <<<"${cv_pass}")"
  fi
  cal_case "calibration cv 2.5 percent" '.prompts[0].decode_cv = 0.025' \
    "above the 2.0 % stability gate (the track contract's calibration_max_cv_percent)"
  cal_case "calibration band does not bracket 1" '.prompts[0].decode_band_high = 0.99' \
    "is not a usable band"
  cal_case "calibration band admits a half-speed box" '.prompts[0].prefill_band_high = 2.5' \
    "is not a usable band"
  cal_case "calibration benchd" '.benchd_source_commit = "unknown"' \
    "cannot be traced to the code that measured it"
  cal_case "calibration timestamp unreadable" '.captured_at = "yesterday"' \
    "is not an RFC 3339 UTC timestamp"
  cal_case "calibration older than the reference" '.captured_at = "2000-01-01T00:00:00Z"' \
    "describes a different engine"
  cal_case "calibration dated in the future" '.captured_at = "2999-01-01T00:00:00Z"' \
    "the box clock is wrong"

  # THE ROTATION POOL (section 4b). Every prompt the workflow can draw must
  # carry its serial golden and one oracle per permitted depth, pinned and
  # staged. Each case rewrites the scaffold's contract, runs, and restores it.
  SCAFFOLD_CONTRACT="${ROOT}/fixtures/nemotron35_lightning_30b_a3b_track.json"
  cp "${SCAFFOLD_CONTRACT}" "${WORK}/contract.orig.json"
  rotation_case() { # rotation_case LABEL JQ_FILTER NEEDLE [env overrides...]
    local label="$1" filter="$2" needle="$3"; shift 3
    jq "${filter}" "${WORK}/contract.orig.json" > "${SCAFFOLD_CONTRACT}"
    expect_refusal "${label}" "${needle}" "$@"
    cp "${WORK}/contract.orig.json" "${SCAFFOLD_CONTRACT}"
  }
  rotation_case "calibration CV limit is boolean" '.calibration_max_cv_percent = false' \
    "calibration_max_cv_percent is 'false', not a usable percentage"
  rotation_case "calibration CV limit is a string" '.calibration_max_cv_percent = "2.0"' \
    "calibration_max_cv_percent is '2.0', not a usable percentage"
  rotation_case "warm-up golden byte pin" '.warmup_golden.bytes += 1' \
    "warm-up golden: byte-count mismatch"
  rotation_case "warm-up golden sha256 pin" '.warmup_golden.sha256 = ("0" * 64)' \
    "warm-up golden: sha256 mismatch"
  # SECOND is in the pool with its oracles pinned. The two cases below
  # construct the broken state explicitly: SECOND stays in the pool, its own
  # oracle pins are removed or replaced, and its oracle files are not staged
  # (a staged file with no pin is refused on its own, before these checks).
  GOLDEN_DIR_NO_SECOND="${WORK}/goldens-no-second-oracles"
  cp -R "${GOLDEN_DIR}" "${GOLDEN_DIR_NO_SECOND}"
  rm -f "${GOLDEN_DIR_NO_SECOND}/${SECOND}".mtp*.golden.json
  SECOND_IN_POOL=".live_golden_rotation.pool |= ((. - [\"${SECOND}\"]) + [\"${SECOND}\"])"
  rotation_case "rotation prompt without per-depth oracles" \
    "${SECOND_IN_POOL} | del(.speculative_oracles[\"${SECOND}\"])" \
    "has no pinned per-depth oracle speculative_oracles[${SECOND}][mtp1]" \
    "MLXFAST_NEMOTRON35_GOLDEN_DIR=${GOLDEN_DIR_NO_SECOND}"
  rotation_case "rotation prompt outside timed_prompt_pool" \
    '.live_golden_rotation.pool += ["nosuch"]' \
    "rotation pool prompt 'nosuch' names no timed_prompt_pool entry"
  rotation_case "rotation prompt borrowing another prompt's oracles" \
    "${SECOND_IN_POOL} | .speculative_oracles[\"${SECOND}\"] = (.speculative_oracles[\"${LIVE_GOLDEN}\"] | del(.mtp6))" \
    "speculative_oracles[${SECOND}][mtp1] points at ${LIVE_GOLDEN}.mtp1.golden.json" \
    "MLXFAST_NEMOTRON35_GOLDEN_DIR=${GOLDEN_DIR_NO_SECOND}"
  rotation_case "rotation pool empty" '.live_golden_rotation.pool = []' \
    "live_golden_rotation.pool is empty"
  rotation_case "calibration prompt is not the band's" ".calibration_prompt = \"${SECOND}\"" \
    "has no prompts[] entry for prompt '${SECOND}'"
  # The control: a second prompt fully pinned and staged joins the pool.
  GOLDEN_DIR2="${WORK}/goldens2"
  cp -R "${GOLDEN_DIR}" "${GOLDEN_DIR2}"
  python3 - "${WORK}/contract.orig.json" "${GOLDEN_DIR2}" "${SCAFFOLD_CONTRACT}" "${LIVE_GOLDEN}" "${SECOND}" <<'ROTEOF'
import hashlib, json, os, sys
src, golden_dir, dst, first, second = sys.argv[1:6]
c = json.load(open(src, encoding="utf-8"))
c["live_golden_rotation"]["pool"] = [first, second]
prefix = c["timed_prompt_pool"][0]["r2_path"].rsplit("/", 1)[0]
c["speculative_oracles"][second] = {}
for d in c["mtp_head"]["permitted_draft_depths"]:
    base = f"{second}.mtp{d}.golden.json"
    data = (json.dumps({"synthetic_golden": base, "restaged": True}) + "\n").encode()
    open(os.path.join(golden_dir, base), "wb").write(data)
    c["speculative_oracles"][second][f"mtp{d}"] = {
        "r2_path": f"{prefix}/{base}", "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
json.dump(c, open(dst, "w", encoding="utf-8"), indent=2)
ROTEOF
  control_rot="$(run_preflight "MLXFAST_NEMOTRON35_GOLDEN_DIR=${GOLDEN_DIR2}")"
  if grep -q 'ranked-box-preflight: all checks passed' <<<"${control_rot}" \
    && grep -q "every rotation pool prompt (${LIVE_GOLDEN} ${SECOND})" <<<"${control_rot}"; then
    echo "ok    section 4b / control: a two-prompt pool, fully pinned and staged, passes"
  else
    fail "section 4b / control: a fully pinned two-prompt pool did not pass; got: $(tail -3 <<<"${control_rot}")"
  fi
  # ...and the same pool with one oracle left unstaged is refused by name.
  rm "${GOLDEN_DIR2}/${SECOND}.mtp4.golden.json"
  expect_refusal "rotation oracle not staged" "${SECOND}.mtp4.golden.json" \
    "MLXFAST_NEMOTRON35_GOLDEN_DIR=${GOLDEN_DIR2}"
  cp "${WORK}/contract.orig.json" "${SCAFFOLD_CONTRACT}"

  # A refresh generation staged ahead of its pin passes (tools/pool-refresh.sh
  # step 3); a held-out final-* golden does not.
  GOLDEN_DIR3="${WORK}/goldens3"
  cp -R "${GOLDEN_DIR}" "${GOLDEN_DIR3}"
  echo '{}' > "${GOLDEN_DIR3}/g2-5.golden.json"
  echo '{}' > "${GOLDEN_DIR3}/g2-5.mtp3.golden.json"
  if run_preflight "MLXFAST_NEMOTRON35_GOLDEN_DIR=${GOLDEN_DIR3}" | grep -q 'ranked-box-preflight: all checks passed'; then
    echo "ok    section 4 / a refresh generation staged ahead of its pin passes"
  else
    fail "section 4 / a refresh generation staged ahead of its pin was refused"
  fi
  echo '{}' > "${GOLDEN_DIR3}/final-1.golden.json"
  expect_refusal "held-out final golden staged" "final-1.golden.json" \
    "MLXFAST_NEMOTRON35_GOLDEN_DIR=${GOLDEN_DIR3}"

  # THE SHIPPED SHAPE (sections 4b and 8b): live_goldens. Every job measures
  # every live golden, so every one needs its own band in the calibration file.
  cp "${SCAFFOLD_CONTRACT}.live-goldens" "${SCAFFOLD_CONTRACT}"
  CAL_ALL="${WORK}/baseline-calibration-all.json"
  write_calibration "${CAL_ALL}" "$(jq -nc --args '$ARGS.positional' "${LIVE_NAMES[@]}") as \$names | .prompts = [\$names[] as \$n | .prompts[0] + {prompt: \$n}]"
  control_lg="$(run_preflight "MLXFAST_BASELINE_CALIBRATION=${CAL_ALL}")"
  if grep -q 'ranked-box-preflight: all checks passed' <<<"${control_lg}" \
    && grep -q "every rotation pool prompt (${LIVE_NAMES[*]})" <<<"${control_lg}"; then
    echo "ok    section 4b+8b / control live_goldens: every live golden staged with its own band passes"
  else
    fail "section 4b+8b / control live_goldens: did not pass; got: $(tail -3 <<<"${control_lg}")"
  fi
  CAL_MISSING="${WORK}/baseline-calibration-missing-one.json"
  jq --arg n "${LIVE_NAMES[2]}" '.prompts |= map(select(.prompt != $n))' "${CAL_ALL}" > "${CAL_MISSING}"
  expect_refusal "live_goldens: one prompt has no band" \
    "has no prompts[] entry for prompt '${LIVE_NAMES[2]}'" "MLXFAST_BASELINE_CALIBRATION=${CAL_MISSING}"
  expect_refusal "live_goldens: a band for one prompt only" \
    "has no prompts[] entry for prompt '${SECOND}'"
  jq '.live_golden_rotation = {mode: "per_job_random", pool: .live_goldens}' \
    "${SCAFFOLD_CONTRACT}.live-goldens" > "${SCAFFOLD_CONTRACT}"
  expect_refusal "live_goldens with a rotation block" \
    "declares both live_goldens and live_golden_rotation" "MLXFAST_BASELINE_CALIBRATION=${CAL_ALL}"
  cp "${WORK}/contract.orig.json" "${SCAFFOLD_CONTRACT}"

  # Section 9: a box without the sandbox user is refused, naming the failed check.
  expect_refusal "sandbox user not set up" \
    "cannot open a TCP connection to 1.1.1.1:443" SANDBOX_CHECK_STUB=fail
  # ...and a box with open loopback or a usable cron (F1/F3) names both.
  expect_refusal "sandbox loopback and cron open" \
    "live listener on 127.0.0.1;FAIL  cron refuses nemosub" SANDBOX_CHECK_STUB=open

  if [[ "${failures}" -eq "${scaffold_failures_before}" ]]; then
    echo "test-ranked-box-preflight-env.sh: the section 8 paired-baseline cases and the control passed"
  fi
fi

total=$(( ${#CREDENTIALS[@]} + ${#OVERRIDES[@]} ))
if [[ "${failures}" -eq 0 ]]; then
  echo "test-ranked-box-preflight-env.sh: all ${total} environment refusals, the section 8 paired-baseline cases and both controls passed"
  exit 0
fi
echo "test-ranked-box-preflight-env.sh: ${failures} case(s) failed" >&2
exit 1

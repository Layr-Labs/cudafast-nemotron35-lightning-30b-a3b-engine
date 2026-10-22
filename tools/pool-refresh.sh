#!/usr/bin/env bash
# pool-refresh.sh -- replace the hidden rotation pool with a fresh generation.
#
# ORGANIZER TOOL, run from the organizer workstation
# (docs/pool-refresh-and-final-rescore.md).
# Steps, each idempotent and recorded in <gen>/state so a crash resumes where
# it stopped (<gen> = .mlxfast-private/refresh/<label>, or .mlxfast-private/final):
#
#   author   stitched OR --texts chat prompts <label>-1..N
#   record   serial + MTP-1..6 goldens for every prompt (N x 7), each on TWO
#            boxes with tools/record-goldens.sh --only in the box's checkout,
#            under the box GPU lock; the two copies must be byte-identical
#   stage    install the N x 7 goldens next to the current ones on every box
#            (root:bench 0640), verified by sha256; every install is logged to
#            .mlxfast-private/refresh/staging.log first
#   pin      edit the fixture: pin the generation and make it live_goldens (and
#            hidden_correctness_golden). official_pairs must already be a multiple
#            of the generation size, because every job measures every live golden.
#            Prints the commit and push commands; pushes nothing
#   verify   once the pin is on origin/main: every box holds every pinned golden
#            with matching sha256 and nothing unpinned (REFRESH_PREFLIGHT_CMD, if
#            set, is also run on each box from its checkout at origin/main)
#   retire   (only by --step) for an OLD label: first unpin it from the fixture
#            (prints commit/push), then, re-run after every job dispatched before
#            that push has finished, remove its goldens from the boxes. The copies
#            in <gen>/goldens are the private archive.
#
# `final` (the held-out re-score pool) runs author and record only: it never
# goes to a box and is never pinned by this script. At the close, after
# `router.py freeze` and the drain, `--step stage --final-after-freeze <FROZEN>`
# stages it (router.py rescore pins it on rescore/base).
#
# Usage:
#   tools/pool-refresh.sh --label g2 [--step STEP] [--dry-run]
#                         [--count N] [--texts DIR] [--corpus DIR] [--snapshot DIR]
#                         [--final-after-freeze FILE]
#   tools/pool-refresh.sh pin-fixture FIXTURE LABEL PINS_JSON   (the fixture edit alone)
#
# Env: REFRESH_BOXES (required: ssh host names of the fleet, at least two),
#      REFRESH_PRIVATE (<repo>/.mlxfast-private),
#      REFRESH_BOX_REPO (nemotron35/repo, relative to the box home),
#      REFRESH_BOX_GOLDENS (/srv/nemotron35/goldens), REFRESH_BOX_SNAPSHOT (the box's
#      verified snapshot; else the box login env's MLXFAST_TARGET_SNAPSHOT_DIR),
#      REFRESH_GPU_LOCK (/tmp/mtplx-gpu-exclusive.lock), REFRESH_LOCK_WAIT (14400 s),
#      REFRESH_PYTHON (python3 with `tokenizers`, for author), REFRESH_ARMED_BASE
#      (loadtest/base), REFRESH_PREFLIGHT_CMD (optional).
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
FIXTURE="${ROOT}/fixtures/nemotron35_lightning_30b_a3b_track.json"
log() { printf 'pool-refresh: %s\n' "$*" >&2; }
die() { printf 'pool-refresh: %s\n' "$*" >&2; exit 1; }

# pin_fixture FIXTURE LABEL PINS_JSON -- pin <label>-<n>.golden.json and
# <label>-<n>.mtp<d>.golden.json from PINS_JSON ({file: {sha256, bytes}}) and
# make that generation the live_goldens and the hidden correctness golden.
# Refuses a prompt missing its serial golden or any depth, and an official_pairs
# that is not a multiple of the generation size.
pin_fixture() {
  local fx="$1" label="$2" pins="$3" out
  out="$(jq --slurpfile p "${pins}" --arg label "${label}" '
    $p[0] as $p
    | ([.timed_prompt_pool[0].r2_path | split("/")[:-1] | join("/")] | first) as $prefix
    | ([$p | keys[] | select(test("^" + $label + "-[0-9]+\\.golden\\.json$")) | sub("\\.golden\\.json$"; "")]
       | sort_by(split("-")[-1] | tonumber)) as $names
    | if ($names | length) == 0 then error("no \($label) serial golden in the pins") else . end
    | ([ $names[] as $n | .mtp_head.permitted_draft_depths[] as $d
         | "\($n).mtp\($d).golden.json" | select($p[.] == null) ]) as $missing
    | if ($missing | length) > 0 then error("missing oracle pins: \($missing | join(", "))") else . end
    | .timed_prompt_pool = ([.timed_prompt_pool[] | select($p[.r2_path | split("/")[-1]] == null)]
        + [$names[] | {r2_path: "\($prefix)/\(.).golden.json"} + $p["\(.).golden.json"]])
    | reduce ($p | to_entries[] | select(.key | test("^" + $label + "-[0-9]+\\.mtp[0-9]+\\.golden\\.json$"))) as $o (.;
        ($o.key | split(".")) as $k
        | .speculative_oracles[$k[0]][$k[1]] = ({r2_path: "\($prefix)/\($o.key)"} + $o.value))
    | if (.official_pairs % ($names | length)) != 0
      then error("official_pairs \(.official_pairs) is not a multiple of the \($names | length) prompts of \($label); every job measures every live golden, so set official_pairs first")
      else . end
    | .live_goldens = $names
    | del(.live_golden, .live_golden_rotation, .calibration_prompt)
    | .hidden_correctness_golden = $p["\($names[0]).golden.json"]' "${fx}")" \
    || die "pin-fixture: refused (above)"
  printf '%s\n' "${out}" > "${fx}.tmp" && mv "${fx}.tmp" "${fx}"
}

if [[ "${1:-}" == pin-fixture ]]; then
  [[ $# -eq 4 ]] || die "usage: pin-fixture FIXTURE LABEL PINS_JSON"
  pin_fixture "$2" "$3" "$4"
  exit 0
fi

LABEL="" STEP="" DRY=0 COUNT="" COUNT_SET=0 TEXTS="" CORPUS="" SNAPSHOT="" FROZEN=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --label) LABEL="$2"; shift 2 ;;
    --step) STEP="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --count) COUNT="$2"; COUNT_SET=1; shift 2 ;;
    --texts) TEXTS="$2"; shift 2 ;;
    --corpus) CORPUS="$2"; shift 2 ;;
    --snapshot) SNAPSHOT="$2"; shift 2 ;;
    --final-after-freeze) FROZEN="$2"; shift 2 ;;
    *) die "unknown argument '$1'" ;;
  esac
done
[[ "${LABEL}" =~ ^(g[0-9]+|final)$ ]] || die "--label must be g<N> or final"
case "${STEP}" in ""|author|record|stage|pin|verify|retire) ;; *) die "--step ${STEP}: not author|record|stage|pin|verify|retire" ;; esac

normalize_count() {
  local supplied="$1" normalized="$1"
  if [[ ! "${normalized}" =~ ^[0-9]+$ ]]; then
    [[ -n "${supplied}" ]] || supplied="''"
    die "--count ${supplied}: use an integer from 1 through 64"
  fi
  while [[ "${normalized}" == 0* && "${#normalized}" -gt 1 ]]; do normalized="${normalized#0}"; done
  [[ "${#normalized}" -le 2 ]] || die "--count ${supplied}: use an integer from 1 through 64"
  (( 10#${normalized} >= 1 && 10#${normalized} <= 64 )) \
    || die "--count ${supplied}: use an integer from 1 through 64"
  printf '%d\n' "$((10#${normalized}))"
}
[[ "${COUNT_SET}" == 0 ]] || COUNT="$(normalize_count "${COUNT}")"

PRIV="${REFRESH_PRIVATE:-${ROOT}/.mlxfast-private}"
[[ -n "${REFRESH_BOXES:-}" ]] || die "REFRESH_BOXES is required (space-separated ssh host names; at least two boxes — every golden is recorded twice)"
read -ra BOXES <<<"${REFRESH_BOXES}"
[[ "${#BOXES[@]}" -ge 2 ]] || die "REFRESH_BOXES needs at least two boxes (every golden is recorded twice)"
BOX_REPO="${REFRESH_BOX_REPO:-nemotron35/repo}"
BOX_GOLDENS="${REFRESH_BOX_GOLDENS:-/srv/nemotron35/goldens}"
LOCK="${REFRESH_GPU_LOCK:-/tmp/mtplx-gpu-exclusive.lock}"
LOCK_WAIT="${REFRESH_LOCK_WAIT:-14400}"
STAGING_LOG="${PRIV}/refresh/staging.log"
if [[ "${LABEL}" == final ]]; then GEN="${PRIV}/final"; else GEN="${PRIV}/refresh/${LABEL}"; fi
STATE="${GEN}/state"
REMOTE="nemotron35-refresh/${LABEL}"   # relative to the box home
if [[ -f "${GEN}/count" ]]; then
  STORED_COUNT="$(normalize_count "$(<"${GEN}/count")")"
  if [[ "${COUNT_SET}" == 1 && "${COUNT}" != "${STORED_COUNT}" ]]; then
    die "count conflict: stored count ${STORED_COUNT}; supplied count ${COUNT}"
  fi
  COUNT="${STORED_COUNT}"
elif [[ "${COUNT_SET}" == 0 ]]; then
  COUNT="$(normalize_count "$(jq -r 'if has("live_goldens") then (.live_goldens | length) else 8 end' "${FIXTURE}")")"
fi
mapfile -t DEPTHS < <(jq -r '.mtp_head.permitted_draft_depths[]' "${FIXTURE}")
NAMES=()
for n in $(seq 1 "${COUNT}"); do NAMES+=("${LABEL}-${n}"); done
mkdir -p "${GEN}"

is_done() { grep -qxF "done $1" "${STATE}" 2>/dev/null; }
mark() { [[ "${DRY}" == 1 ]] || echo "done $1" >> "${STATE}"; }
run() { if [[ "${DRY}" == 1 ]]; then printf '+ %q' "$1" >&2; shift; printf ' %q' "$@" >&2; echo >&2; else "$@"; fi; }
rsh() { local box="$1"; shift; run ssh -n -o BatchMode=yes "${box}" "$*"; }
sha() { sha256sum "$1" | cut -d' ' -f1; }

# ---- 1. author ---------------------------------------------------------------
step_author() {
  is_done author && { log "author: done"; return; }
  if [[ -n "${TEXTS}" ]]; then
    [[ -d "${SNAPSHOT}" ]] || die "author with --texts needs --snapshot DIR (the verified target snapshot)"
    [[ -d "${TEXTS}" ]] || die "author: no --texts directory ${TEXTS}"
    local chats=() n text
    for n in "${NAMES[@]}"; do
      text="${TEXTS}/${n}.txt"
      [[ -f "${text}" ]] || die "author: no ${text}"
      chats+=("${n}=${text}")
    done
    run "${REFRESH_PYTHON:-python3}" "${ROOT}/tools/author-prompts.py" --snapshot "${SNAPSHOT}" \
      --out "${GEN}/prompts" --chat "${chats[@]}"
  else
    [[ -d "${CORPUS}" && -d "${SNAPSHOT}" ]] || die "author needs --corpus DIR and --snapshot DIR (the verified target snapshot)"
    run "${REFRESH_PYTHON:-python3}" "${ROOT}/tools/stitch-prompts.py" --corpus "${CORPUS}" --snapshot "${SNAPSHOT}" \
      --label "${LABEL}" --count "${COUNT}" --out "${GEN}/prompts" --ledger "${PRIV}/refresh/ledger.json"
  fi
  [[ "${DRY}" == 1 ]] && return
  for n in "${NAMES[@]}"; do [[ -f "${GEN}/prompts/${n}.tokens.json" ]] || die "author: no ${n}.tokens.json"; done
  if [[ -n "${TEXTS}" ]]; then
    for n in "${NAMES[@]}"; do cp "${TEXTS}/${n}.txt" "${GEN}/prompts/"; done
  fi
  printf '%s\n' "${COUNT}" > "${GEN}/count"
  mark author
}

# ---- 2. record ---------------------------------------------------------------
# One box job: copy the prompts (and, for an oracle job, the agreed serial
# goldens) to the box, run record-goldens.sh --only under the GPU lock, fetch
# the goldens back into <gen>/rec/<key>/.
box_job() { # box_job BOX KEY STEPS FILE_SUFFIX NAME...
  local box="$1" key="$2" steps="$3" suffix="$4"; shift 4
  local names=("$@") dir="${REMOTE}/${key}" only=() n
  is_done "job ${key}" && { log "record: ${key} done"; return 0; }
  for n in "${names[@]}"; do only+=(--only "${n}"); done
  rsh "${box}" "rm -rf ${dir} && mkdir -p ${dir}/prompts ${dir}/out"
  local src=() back=()
  for n in "${names[@]}"; do src+=("${GEN}/prompts/${n}.tokens.json"); back+=("${box}:${dir}/out/${n}${suffix}"); done
  run scp -q "${src[@]}" "${box}:${dir}/prompts/"
  if [[ "${steps}" != serial ]]; then
    src=()
    for n in "${names[@]}"; do src+=("${GEN}/goldens/${n}.golden.json"); done
    run scp -q "${src[@]}" "${box}:${dir}/out/"
  fi
  local snap="" logf="${GEN}/rec-${key}.log"
  [[ "${DRY}" != 1 ]] || logf=/dev/stderr
  [[ -z "${REFRESH_BOX_SNAPSHOT:-}" ]] || snap="MLXFAST_TARGET_SNAPSHOT_DIR=${REFRESH_BOX_SNAPSHOT}"
  # serve-up refuses a /tmp socket under docker, and the operator's shell has no
  # SERVE_UP_* or BENCHD_BIN_DIR: give the recording private dirs and the staged benchd.
  rsh "${box}" "mkdir -p \$HOME/rsock \$HOME/${dir}/log && cd ${BOX_REPO} && [[ -f ${LOCK} && ! -L ${LOCK} ]] && exec 9<${LOCK} && flock -w ${LOCK_WAIT} 9 && ${snap} SERVE_UP_SOCKET_DIR=\$HOME/rsock SERVE_UP_LOG_DIR=\$HOME/${dir}/log BENCHD_BIN_DIR=${REFRESH_BOX_BENCHD:-/srv/nemotron35/benchd-bin} tools/record-goldens.sh" \
    "--prompts \$HOME/${dir}/prompts --out \$HOME/${dir}/out ${only[*]} --steps ${steps}" \
    >"${logf}" 2>&1 || { log "record: ${key} on ${box} failed; see ${GEN}/rec-${key}.log"; return 1; }
  run mkdir -p "${GEN}/rec/${key}"
  run scp -q "${back[@]}" "${GEN}/rec/${key}/"
  mark "job ${key}"
}

run_jobs() { # run_jobs JOBFILE -- lines "BOX KEY STEPS SUFFIX NAMES..."; boxes in parallel, a box's jobs in turn
  local jobs="$1" box pids=() rc=0 p
  for box in "${BOXES[@]}"; do
    (
      while read -r b key steps suffix names; do
        [[ "${b}" == "${box}" ]] || continue
        # shellcheck disable=SC2086
        box_job "${b}" "${key}" "${steps}" "${suffix}" ${names} || exit 1
      done < "${jobs}"
    ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "${p}" || rc=1; done
  return "${rc}"
}

# agree FILE KEY_A KEY_B -- the two recordings must be byte-identical.
agree() {
  local f="$1" a="${GEN}/rec/$2/$1" b="${GEN}/rec/$3/$1"
  [[ -f "${a}" && -f "${b}" ]] || die "record: ${f} missing from $2 or $3"
  [[ "$(sha "${a}")" == "$(sha "${b}")" ]] \
    || die "record: ${f} differs between $2 ($(sha "${a}")) and $3 ($(sha "${b}")); a broken box or engine -- nothing is pinned. Remove the bad job's 'done job' line from ${STATE} to re-record it"
  cp "${a}" "${GEN}/goldens/${f}"
}

step_record() {
  is_done record && { log "record: done"; return; }
  [[ "${DRY}" == 1 ]] || for n in "${NAMES[@]}"; do [[ -f "${GEN}/prompts/${n}.tokens.json" ]] || die "record: run --step author first (no ${n}.tokens.json)"; done
  mkdir -p "${GEN}/goldens"
  local nb="${#BOXES[@]}" i d a b jobs="${GEN}/jobs"
  # serial: prompt i on boxes (i-1)%nb and i%nb, one boot per box
  : > "${jobs}"
  for i in "${!BOXES[@]}"; do
    local mine=()
    for n in $(seq 1 "${COUNT}"); do
      if (( (n - 1) % nb == i || n % nb == i )); then mine+=("${LABEL}-${n}"); fi
    done
    [[ "${#mine[@]}" -eq 0 ]] || echo "${BOXES[i]} serial@${BOXES[i]} serial .golden.json ${mine[*]}" >> "${jobs}"
  done
  run_jobs "${jobs}" || die "record: a serial job failed; re-run to resume"
  [[ "${DRY}" == 1 ]] || for n in $(seq 1 "${COUNT}"); do
    agree "${LABEL}-${n}.golden.json" "serial@${BOXES[(n - 1) % nb]}" "serial@${BOXES[n % nb]}"
  done
  # oracles: depth d, all prompts, on boxes 2(d-1)%nb and (2(d-1)+1)%nb
  : > "${jobs}"
  for d in "${DEPTHS[@]}"; do
    a=$(( (2 * (d - 1)) % nb )); b=$(( (2 * (d - 1) + 1) % nb ))
    for i in "${a}" "${b}"; do echo "${BOXES[i]} mtp${d}@${BOXES[i]} ${d} .mtp${d}.golden.json ${NAMES[*]}" >> "${jobs}"; done
  done
  run_jobs "${jobs}" || die "record: an oracle job failed; re-run to resume"
  [[ "${DRY}" == 1 ]] && return
  for d in "${DEPTHS[@]}"; do
    a=$(( (2 * (d - 1)) % nb )); b=$(( (2 * (d - 1) + 1) % nb ))
    for n in "${NAMES[@]}"; do agree "${n}.mtp${d}.golden.json" "mtp${d}@${BOXES[a]}" "mtp${d}@${BOXES[b]}"; done
  done
  local pins="{}" f
  for f in "${GEN}"/goldens/*.golden.json; do
    pins="$(jq -c --arg k "${f##*/}" --arg s "$(sha "${f}")" --argjson b "$(wc -c < "${f}" | tr -d ' ')" \
      '. + {($k): {sha256: $s, bytes: $b}}' <<<"${pins}")"
  done
  jq . <<<"${pins}" > "${GEN}/pins.json"
  [[ "$(jq length "${GEN}/pins.json")" -eq $(( COUNT * (1 + ${#DEPTHS[@]}) )) ]] || die "record: pins.json is incomplete"
  mark record
  log "record: ${COUNT} x $(( 1 + ${#DEPTHS[@]} )) goldens agreed across two boxes each -> ${GEN}/pins.json"
}

# ---- 3. stage ----------------------------------------------------------------
# box_sums BOX -> "sha256  name" for every *.json staged on BOX
box_sums() { rsh "$1" "sudo -n sh -c 'cd \"\$0\" && sha256sum -- *.json' \"${BOX_GOLDENS}\""; }

step_stage() {
  if [[ "${LABEL}" == final ]]; then
    [[ -n "${FROZEN}" && -f "${FROZEN}" ]] \
      || die "REFUSING to stage the held-out final generation: it stays off the boxes until the close (router.py freeze, drain, then --final-after-freeze <state-dir>/FROZEN)"
  fi
  is_done record || [[ "${DRY}" == 1 ]] || die "stage: run --step record first"
  local box f files=()
  for f in "${GEN}"/goldens/*.golden.json; do [[ -e "${f}" ]] && files+=("${f}"); done
  for box in "${BOXES[@]}"; do
    is_done "stage@${box}" && { log "stage: ${box} done"; continue; }
    if [[ "${DRY}" != 1 ]]; then
      mkdir -p "$(dirname "${STAGING_LOG}")"
      for f in "${files[@]}"; do echo "$(date +%s) ${LABEL} ${box} ${f##*/} $(sha "${f}")" >> "${STAGING_LOG}"; done
    fi
    rsh "${box}" "rm -rf ${REMOTE}/stage && mkdir -p ${REMOTE}/stage"
    run scp -q "${files[@]}" "${box}:${REMOTE}/stage/"
    rsh "${box}" "sudo -n install -m 0640 -o root -g bench \$HOME/${REMOTE}/stage/*.golden.json \"${BOX_GOLDENS}/\" && rm -rf \$HOME/${REMOTE}/stage"
    [[ "${DRY}" == 1 ]] && continue
    local sums
    sums="$(box_sums "${box}")" || die "stage: cannot list ${BOX_GOLDENS} on ${box}"
    for f in "${files[@]}"; do
      grep -qxF "$(sha "${f}")  ${f##*/}" <<<"${sums}" || die "stage: ${f##*/} on ${box} does not match its sha256"
    done
    mark "stage@${box}"
    log "stage: ${box}: ${#files[@]} goldens installed and verified"
  done
}

# ---- 4. pin ------------------------------------------------------------------
step_pin() {
  [[ "${LABEL}" != final ]] || die "REFUSING to pin the held-out final generation (router.py rescore pins it on rescore/base only)"
  is_done pin && { log "pin: done"; return; }
  local box
  [[ "${DRY}" == 1 ]] && { log "would pin ${GEN}/pins.json into ${FIXTURE}"; return; }
  for box in "${BOXES[@]}"; do is_done "stage@${box}" || die "pin: ${box} is not staged; run --step stage first"; done
  pin_fixture "${FIXTURE}" "${LABEL}" "${GEN}/pins.json"
  mark pin
  cat <<EOF
pool-refresh: pinned ${LABEL} in ${FIXTURE}; review the diff, then:
  git -C ${ROOT} add fixtures/nemotron35_lightning_30b_a3b_track.json
  git -C ${ROOT} commit -m "pool refresh: pin ${LABEL} and switch the rotation pool"
  git -C ${ROOT} push origin HEAD:main
  git -C ${ROOT} push origin HEAD:${REFRESH_ARMED_BASE:-loadtest/base}
then: tools/pool-refresh.sh --label ${LABEL} --step verify
EOF
}

# ---- 5. verify ---------------------------------------------------------------
# check_box_against FIXTURE BOX -- every pin staged with its sha256; nothing
# else except a g<N> generation staged ahead of its pin.
check_box_against() {
  local fx="$1" box="$2" sums want
  sums="$(box_sums "${box}")" || die "verify: cannot list ${BOX_GOLDENS} on ${box}"
  want="$(jq -r '[.timed_prompt_pool[], (.speculative_oracles // {} | .[] | .[])] | unique_by(.r2_path)[]
                 | "\(.sha256)  \(.r2_path | split("/")[-1])"' "${fx}")"
  local missing extra
  missing="$(comm -23 <(sort <<<"${want}") <(sort <<<"${sums}"))"
  [[ -z "${missing}" ]] || die "verify: ${box} lacks or mismatches: $(awk '{print $2}' <<<"${missing}" | tr '\n' ' ')"
  extra="$(comm -13 <(sort <<<"${want}") <(sort <<<"${sums}") | awk '{print $2}' | grep -Ev '^g[0-9]+-[0-9]+(\.mtp[0-9]+)?\.golden\.json$' || true)"
  [[ -z "${extra}" ]] || die "verify: ${box} holds unpinned goldens: $(tr '\n' ' ' <<<"${extra}")"
}

origin_fixture() { # the fixture on origin/main, into a temp file
  git -C "${ROOT}" fetch -q origin main || die "cannot fetch origin main"
  git -C "${ROOT}" show origin/main:fixtures/nemotron35_lightning_30b_a3b_track.json > "$1"
}

step_verify() {
  [[ "${LABEL}" != final ]] || die "final is verified by router.py rescore, not here"
  is_done verify && { log "verify: done"; return; }
  [[ "${DRY}" == 1 ]] && { log "would verify every box against origin/main's fixture"; return; }
  local fx="${GEN}/origin-fixture.json" box
  origin_fixture "${fx}"
  jq -e --argjson n "$(printf '%s\n' "${NAMES[@]}" | jq -R . | jq -s .)" '.live_goldens == $n' "${fx}" >/dev/null \
    || die "verify: origin/main's live_goldens are not ${LABEL} yet; push the pin commit first"
  for box in "${BOXES[@]}"; do
    check_box_against "${fx}" "${box}"
    if [[ -n "${REFRESH_PREFLIGHT_CMD:-}" ]]; then
      rsh "${box}" "cd ${BOX_REPO} && git fetch -q origin main && git checkout -q --detach origin/main && ${REFRESH_PREFLIGHT_CMD}" \
        || die "verify: the ranked preflight failed on ${box}"
    fi
    log "verify: ${box} ok"
  done
  mark verify
  log "verify: ${LABEL} is live on all ${#BOXES[@]} boxes"
}

# ---- 6. retire ---------------------------------------------------------------
step_retire() {
  [[ "${LABEL}" != final ]] || die "final is not a contest generation; nothing to retire"
  is_done retired && { log "retire: done"; return; }
  local re="^${LABEL}-[0-9]+(\\.mtp[0-9]+)?\\.golden\\.json\$"
  jq -e --arg l "${LABEL}-" 'any(.live_goldens[]?; startswith($l))' "${FIXTURE}" >/dev/null \
    && die "retire: ${LABEL} is still the live goldens; refresh to the next generation first"
  if jq -e --arg re "${re}" 'any(.timed_prompt_pool[]; .r2_path | split("/")[-1] | test($re))' "${FIXTURE}" >/dev/null; then
    [[ "${DRY}" == 1 ]] && { log "would unpin ${LABEL} from ${FIXTURE}"; return; }
    jq --arg re "${re}" --arg l "${LABEL}-" '
      .timed_prompt_pool |= map(select(.r2_path | split("/")[-1] | test($re) | not))
      | .speculative_oracles |= with_entries(select(.key | startswith($l) | not))' "${FIXTURE}" > "${FIXTURE}.tmp"
    mv "${FIXTURE}.tmp" "${FIXTURE}"
    cat <<EOF
pool-refresh: unpinned ${LABEL} in ${FIXTURE}; review, then:
  git -C ${ROOT} commit -am "pool refresh: retire ${LABEL}"
  git -C ${ROOT} push origin HEAD:main
  git -C ${ROOT} push origin HEAD:${REFRESH_ARMED_BASE:-loadtest/base}
When every job dispatched before that push has finished, re-run --step retire
to remove ${LABEL}'s goldens from the boxes (the archive is ${GEN}/goldens).
EOF
    return
  fi
  local fx="${GEN}/origin-fixture.json" box
  if [[ "${DRY}" != 1 ]]; then
    origin_fixture "${fx}"
    jq -e --arg re "${re}" 'any(.timed_prompt_pool[]; .r2_path | split("/")[-1] | test($re))' "${fx}" >/dev/null \
      && die "retire: origin/main still pins ${LABEL}; push the retire commit first"
  fi
  for box in "${BOXES[@]}"; do
    is_done "retire@${box}" && continue
    rsh "${box}" "sudo -n sh -c 'rm -f -- \"\$0\"/${LABEL}-*.golden.json' \"${BOX_GOLDENS}\""
    mark "retire@${box}"
  done
  mark retired
  log "retire: ${LABEL} removed from every box; archive kept at ${GEN}/goldens"
}

if [[ -n "${STEP}" ]]; then
  "step_${STEP}"
elif [[ "${LABEL}" == final ]]; then
  step_author; step_record
  log "final: authored and recorded in ${GEN}; it stays off the boxes until the close"
else
  step_author; step_record; step_stage; step_pin
  log "next: commit and push the fixture (above), then --step verify; retire the previous generation once its jobs are done"
fi

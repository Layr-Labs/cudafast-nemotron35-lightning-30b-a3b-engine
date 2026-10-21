#!/usr/bin/env bash
# record-goldens.sh -- author and pin this track's goldens ON A SPARK.
#
# ORGANIZER TOOL. Run it on a ranked-class box, from a clean checkout at the
# commit that will become baseline_reference_commit, after ./setup.sh has built
# the engine and verified the snapshot. It records every golden the fixture
# names, through the SAME adapter -> resident path a ranked leg runs:
#
#   1. one SERIAL resident: for every timed_prompt_pool entry, a golden with the
#      teacher-forced cases[] and the free-run benchmark block (the serial
#      timed oracle). The first live golden's serial tape is also the hidden
#      correctness golden. A case is the ANSWER: it ends at the first
#      end-of-turn token (the eos_token_id list of the snapshot's pinned
#      generation_config.json), and it is at most ANSWER_MAX_TOKENS long.
#   2. one resident per permitted MTP depth d: each oracle prompt's free-run
#      benchmark block recorded UNDER THAT SPEC (author-with-what-you-replay),
#      with the SERIAL cases[] spliced in, so every per-depth golden carries
#      the same token-fidelity oracle. The oracle prompts are the live_goldens
#      by default, or the --oracles-for names (every live golden needs all six
#      before a run may declare a depth).
#   3. every golden re-validated by `benchd validate-golden --contract` (model
#      identity + provenance), then its {sha256, bytes} printed -- and, with
#      --apply, written into the fixture.
#
# Upload to R2 at each entry's r2_path is the organizer's own step; this tool
# holds no credential and prints the list.
#
# Usage:
#   tools/record-goldens.sh --prompts <DIR> --out <DIR> [--apply]
#     <DIR>/<name>.tokens.json for every pool name (tools/author-prompts.py),
#     and optionally <DIR>/public.tokens.json for the committed public golden
#   tools/record-goldens.sh --prompts <DIR> --out <DIR> --oracles-for <name>... [--apply]
#     ORACLES ONLY, for already-pinned serial goldens: step 1 is skipped, and
#     each <name> (repeatable, or `all` for every timed_prompt_pool name) must
#     already have its PINNED serial golden in <out> (tools/fetch-goldens.sh
#     --all --out <out> stages them) -- its cases[] are what every per-depth
#     oracle carries, so a re-recorded serial tape is never spliced in. Then
#     one resident per depth records <name>.mtp<d>.golden.json for every name.
#     --apply writes speculative_oracles.<name>.mtp<d>; adding <name> to
#     live_goldens stays a reviewed edit.
#   tools/record-goldens.sh --prompts <DIR> --out <DIR> --only <name>... [--steps serial,1,...]
#     A NEW GENERATION not yet in the fixture (tools/pool-refresh.sh): records
#     only the --only names (repeatable), and only the --steps asked for
#     (default serial plus every permitted depth). An oracle step splices the
#     serial <out>/<name>.golden.json already there -- recorded by this run's
#     serial step, or the cross-box-agreed one pool-refresh.sh put there. No
#     public golden, and no --apply: pool-refresh.sh pins the generation.
#
# Env: MLXFAST_TARGET_SNAPSHOT_DIR (required), BENCHD_BIN_DIR (default
#      benchd-bin), SERVE_UP_* as for tools/serve-up.sh.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
FIXTURE="${ROOT}/fixtures/nemotron35_lightning_30b_a3b_track.json"
log() { printf 'record-goldens: %s\n' "$*" >&2; }
die() { printf 'record-goldens: %s\n' "$*" >&2; exit 1; }

PROMPTS="" OUT="" APPLY=0 ORACLES_FOR=() ONLY=() STEPS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompts) PROMPTS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --oracles-for) ORACLES_FOR+=("$2"); shift 2 ;;
    --apply) APPLY=1; shift ;;
    --only) ONLY+=("$2"); shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    *) die "unknown argument '$1'" ;;
  esac
done
[[ -d "${PROMPTS}" && -n "${OUT}" ]] || die "usage: --prompts <DIR> --out <DIR> [--apply]"
SNAP="${MLXFAST_TARGET_SNAPSHOT_DIR:-}"
[[ -d "${SNAP}" ]] || die "MLXFAST_TARGET_SNAPSHOT_DIR must name the verified snapshot"
BIN="${BENCHD_BIN_DIR:-${ROOT}/benchd-bin}"
RECORDER="${BIN}/record-correctness-golden"
BENCHD="${BIN}/benchd"
[[ -x "${RECORDER}" && -x "${BENCHD}" ]] || die "no benchd pair in ${BIN} (run tools/fetch-benchd.sh)"
mkdir -p "${OUT}"
OUT="$(cd -- "${OUT}" && pwd -P)"

TRACK="$(jq -r .trackId "${ROOT}/benchmark.json")"
REPO="$(jq -r .target.upstream_model_id "${FIXTURE}")"
REV="$(jq -r .target.upstream_revision "${FIXTURE}")"
mapfile -t LIVE_NAMES < <(jq -r '(.live_goldens // [])[]' "${FIXTURE}")
[[ "${#LIVE_NAMES[@]}" -gt 0 ]] || die "the fixture declares no live_goldens"
LIVE="${LIVE_NAMES[0]}"
# The answer: at most ANSWER_MAX_TOKENS, ended by the first end-of-turn token.
ANSWER_MAX_TOKENS=1024
GENCFG="${SNAP}/generation_config.json"
want_gencfg="$(jq -r '.target.files[] | select(.path == "generation_config.json") | .sha256' "${FIXTURE}")"
[[ -f "${GENCFG}" && "$(sha256sum "${GENCFG}" | cut -d' ' -f1)" == "${want_gencfg}" ]] \
  || die "${GENCFG} does not match its fixture pin; the end-of-turn tokens come from the pinned file only"
mapfile -t STOP_IDS < <(jq -r '.eos_token_id | if type == "array" then .[] else . end' "${GENCFG}")
[[ "${#STOP_IDS[@]}" -gt 0 ]] || die "${GENCFG} declares no eos_token_id"
STOP_ARGS=()
for t in "${STOP_IDS[@]}"; do
  [[ "${t}" =~ ^[0-9]+$ ]] || die "${GENCFG}: eos_token_id ${t} is not a token id"
  STOP_ARGS+=(--stop-token "${t}")
done
mapfile -t NAMES < <(jq -r '.timed_prompt_pool[].r2_path | split("/")[-1] | sub("\\.golden\\.json$"; "")' "${FIXTURE}")
mapfile -t DEPTHS < <(jq -r '.mtp_head.permitted_draft_depths[]' "${FIXTURE}")
ENGINE_REL=".build/release/mlxfast-runtime-worker"
DO_SERIAL=1
if [[ "${#ONLY[@]}" -gt 0 ]]; then
  [[ "${#ORACLES_FOR[@]}" -eq 0 && "${APPLY}" == 0 ]] || die "--only takes neither --oracles-for nor --apply (tools/pool-refresh.sh pins the generation)"
  for n in "${ONLY[@]}"; do
    [[ "${n}" =~ ^[a-z0-9_-]+$ ]] || die "--only ${n}: not [a-z0-9_-]+"
    [[ -f "${PROMPTS}/${n}.tokens.json" ]] || die "no ${PROMPTS}/${n}.tokens.json"
  done
  NAMES=("${ONLY[@]}") ORACLE_NAMES=("${ONLY[@]}")
  if [[ -n "${STEPS}" ]]; then
    DO_SERIAL=0 WANT=()
    IFS=, read -ra steps <<<"${STEPS}"
    for st in "${steps[@]}"; do
      if [[ "${st}" == serial ]]; then DO_SERIAL=1
      elif [[ " ${DEPTHS[*]} " == *" ${st} "* ]]; then WANT+=("${st}")
      else die "--steps ${st}: not serial or a permitted depth (${DEPTHS[*]})"; fi
    done
    DEPTHS=(${WANT[@]+"${WANT[@]}"})
  fi
  if [[ "${#DEPTHS[@]}" -gt 0 && "${DO_SERIAL}" == 0 ]]; then
    for n in "${ONLY[@]}"; do
      [[ -f "${OUT}/${n}.golden.json" ]] || die "--only ${n}: an oracle step needs its serial golden in ${OUT} (record --steps serial first)"
    done
  fi
elif [[ -n "${STEPS}" ]]; then
  die "--steps needs --only"
elif [[ "${#ORACLES_FOR[@]}" -eq 0 ]]; then
  ORACLE_NAMES=("${LIVE_NAMES[@]}")
  for n in "${NAMES[@]}"; do [[ -f "${PROMPTS}/${n}.tokens.json" ]] || die "no ${PROMPTS}/${n}.tokens.json"; done
else
  if [[ " ${ORACLES_FOR[*]} " == *" all "* ]]; then ORACLE_NAMES=("${NAMES[@]}"); else ORACLE_NAMES=("${ORACLES_FOR[@]}"); fi
  for n in "${ORACLE_NAMES[@]}"; do
    [[ " ${NAMES[*]} " == *" ${n} "* ]] || die "--oracles-for ${n}: not a timed_prompt_pool name (${NAMES[*]})"
    [[ -f "${PROMPTS}/${n}.tokens.json" ]] || die "no ${PROMPTS}/${n}.tokens.json"
    serial="${OUT}/${n}.golden.json"
    [[ -f "${serial}" ]] || die "--oracles-for ${n}: its pinned serial golden is not in ${OUT} (stage it: tools/fetch-goldens.sh --all --out ${OUT})"
    want="$(jq -r --arg f "/${n}.golden.json" 'first(.timed_prompt_pool[] | select(.r2_path | endswith($f))) | "\(.sha256) \(.bytes)"' "${FIXTURE}")"
    [[ "$(sha256sum "${serial}" | cut -d' ' -f1) $(wc -c < "${serial}" | tr -d ' ')" == "${want}" ]] \
      || die "--oracles-for ${n}: ${serial} does not match its timed_prompt_pool pin (${want}); every oracle must splice the PINNED serial cases[]"
  done
fi

SOCKET=""
stop_resident() { [[ -z "${SOCKET}" ]] || "${ROOT}/tools/serve-up.sh" --stop --socket "${SOCKET}" >&2 || true; SOCKET=""; }
trap stop_resident EXIT
boot() { # boot SPEC DEPTH
  "${ROOT}/tools/serve-up.sh" --boot --spec "$1" --draft-len "$2" --socket-out "${OUT}/.socket" >&2
  SOCKET="$(head -1 "${OUT}/.socket")"
}
record() { # record NAME OUTFILE
  (cd "${ROOT}" && BENCH_WORKER_RESIDENT_SOCKET="${SOCKET}" "${RECORDER}" \
     --worker-bin "${ENGINE_REL}" --weights "${SNAP}" \
     --prompt-tokens "${PROMPTS}/$1.tokens.json" --case-name "$1" \
     --steps "${ANSWER_MAX_TOKENS}" "${STOP_ARGS[@]}" \
     --benchmark-free-run --track "${TRACK}" --contract "${FIXTURE}" \
     --model-provenance-repo "${REPO}" --model-provenance-rev "${REV}" --out "$2") >&2
}
validate() { "${BENCHD}" validate-golden --golden "$1" --track "${TRACK}" --contract "${FIXTURE}" >&2; }

# 1. serial tapes (and the public golden, when its tokens are given; it is
#    committed to correctness_prompts/, never pinned in the fixture). Skipped
#    under --oracles-for, which splices the already-pinned serial tapes.
if [[ "${#ORACLES_FOR[@]}" -eq 0 && "${DO_SERIAL}" == 1 ]]; then
  log "serial resident: ${#NAMES[@]} pool golden(s)"
  boot serial 0
  for n in "${NAMES[@]}"; do record "${n}" "${OUT}/${n}.golden.json"; done
  if [[ "${#ONLY[@]}" -eq 0 && -f "${PROMPTS}/public.tokens.json" ]]; then
    mkdir -p "${OUT}/public"
    record public "${OUT}/public/public-longcopy-gate-english-496.golden.json"
    validate "${OUT}/public/public-longcopy-gate-english-496.golden.json" || die "benchd refused the public golden"
  fi
  stop_resident
fi

# 2. per-depth timed oracles for every oracle prompt: one resident per depth,
#    every prompt recorded under it.
for d in ${DEPTHS[@]+"${DEPTHS[@]}"}; do
  log "mtp${d} resident: ${ORACLE_NAMES[*]}"
  boot mtp "${d}"
  for n in "${ORACLE_NAMES[@]}"; do
    record "${n}" "${OUT}/${n}.mtp${d}.raw.json"
    jq --slurpfile serial "${OUT}/${n}.golden.json" '.cases = $serial[0].cases' \
      "${OUT}/${n}.mtp${d}.raw.json" > "${OUT}/${n}.mtp${d}.golden.json"
    rm -f "${OUT}/${n}.mtp${d}.raw.json"
  done
  stop_resident
done

# 3. validate and pin
pins="{}"
for f in "${OUT}"/*.golden.json; do
  validate "${f}" || die "benchd refused ${f}"
  pins="$(jq -c --arg k "$(basename "${f}")" --arg s "$(sha256sum "${f}" | cut -d' ' -f1)" \
          --argjson b "$(wc -c < "${f}" | tr -d ' ')" '. + {($k): {sha256: $s, bytes: $b}}' <<<"${pins}")"
done
jq . <<<"${pins}"
log "upload each file to R2 at its fixture r2_path (correctness_prompts/${TRACK}/<file>)"

if [[ "${APPLY}" == "1" ]]; then
  jq --argjson p "${pins}" --arg live "${LIVE}.golden.json" --arg prefix "correctness_prompts/${TRACK}" '
    def pin: . as $e | ($e.r2_path | split("/")[-1]) as $f | $e + ($p[$f] // {});
    .timed_prompt_pool |= map(pin)
    | reduce ($p | to_entries[] | select(.key | test("^[a-z0-9_-]+\\.mtp[0-9]+\\.golden\\.json$"))) as $o (.;
        ($o.key | split(".")) as $k
        | .speculative_oracles[$k[0]][$k[1]] = ({r2_path: ($prefix + "/" + $o.key)} + $o.value))
    | if $p[$live] then .hidden_correctness_golden = $p[$live] else . end' "${FIXTURE}" > "${FIXTURE}.tmp"
  mv "${FIXTURE}.tmp" "${FIXTURE}"
  log "pinned ${FIXTURE}; review the diff, then commit"
  log "a prompt is scored only once it is in live_goldens: add each fully pinned name there"
fi

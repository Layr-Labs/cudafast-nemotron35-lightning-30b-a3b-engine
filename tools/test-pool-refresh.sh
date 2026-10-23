#!/usr/bin/env bash
# test-pool-refresh.sh -- tools/pool-refresh.sh against stubbed boxes.
#
# ssh/scp/sudo/install/flock are stubs: each "box" is a directory with a home
# (holding a stub tools/record-goldens.sh in nemotron35/repo) and a goldens
# directory. The script runs from a throwaway repo whose fixture it may edit,
# with a local bare origin. No network, no GPU, no real box.
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok() { echo "ok    $*"; }

FAKE="${WORK}/boxes" STUB="${WORK}/stub" REMOTE_STUB="${WORK}/remote-stub"
ROOT="${WORK}/root" PRIV="${WORK}/private"
mkdir -p "${FAKE}" "${STUB}" "${REMOTE_STUB}" "${ROOT}/tools" "${ROOT}/fixtures" "${PRIV}"
export FAKE REMOTE_STUB

cat > "${STUB}/ssh" <<'EOF'
#!/usr/bin/env bash
while [[ "$1" == -* ]]; do [[ "$1" == -o ]] && shift; shift; done
box="$1"; shift
echo "ssh ${box} $*" >> "${FAKE}/calls.log"
host="${box%%/*}"
home="${FAKE}/${host}/home"
case "${FAKE_REMOTE_FAULT:-}" in
  home-unset) unset home ;;
  home-empty) home="" ;;
  home-root) home=/ ;;
  home-dotdot) home=/tmp/.. ;;
  home-symlink) home="${FAKE}/${host}/home-link" ;;
esac
cd "${FAKE}/${host}/home" || exit 1
if [[ -v home ]]; then
  HOME="${home}" BOX="${host}" PATH="${REMOTE_STUB}:${PATH}" bash -c "$*"
else
  env -u HOME BOX="${host}" PATH="${REMOTE_STUB}:${PATH}" bash -c "$*"
fi
EOF
cat > "${STUB}/scp" <<'EOF'
#!/usr/bin/env bash
[[ "$1" == -q ]] && shift
echo "scp $*" >> "${FAKE}/calls.log"
echo copy >> "${FAKE}/scp.log"
last="${!#}"
if [[ "${last}" =~ ^([a-z0-9-]+):(.*/(prompts|out)/)$ ]]; then
  box="${BASH_REMATCH[1]}"
  path="${BASH_REMATCH[2]%/}"
  job="${path%/*}"
  [[ "${job}" == /* ]] || job="${FAKE}/${box}/home/${job}"
  mode="$(stat -c %a "${job}" 2>/dev/null || echo missing)"
  owner="$(stat -c %u "${job}" 2>/dev/null || echo missing)"
  echo "copy-mode ${box} ${path} ${mode} ${owner} $(id -u)" >> "${FAKE}/calls.log"
fi
args=()
for a in "$@"; do
  if [[ "${a}" =~ ^([a-z0-9-]+):(.*)$ ]]; then
    box="${BASH_REMATCH[1]}" path="${BASH_REMATCH[2]}"
    [[ "${path}" == /* ]] || path="${FAKE}/${box}/home/${path}"
    a="${path}"
  fi
  args+=("${a}")
done
cp "${args[@]}"
EOF
cat > "${REMOTE_STUB}/stat" <<'EOF'
#!/usr/bin/env bash
path="${!#}"
if [[ "${path}" == */nemotron35-refresh/* ]]; then
  case "${FAKE_REMOTE_FAULT:-}:$*" in
    wrong-mode:*"%a"*) echo 755; exit 0 ;;
    wrong-owner:*"%u"*) echo $(( $(id -u) + 1 )); exit 0 ;;
    stat-failure:*"%a"*) echo 700; exit 1 ;;
    stat-failure:*"%u"*) echo "$(id -u)"; exit 1 ;;
  esac
fi
exec /usr/bin/stat "$@"
EOF
cat > "${REMOTE_STUB}/rm" <<'EOF'
#!/usr/bin/env bash
path="${!#}"
[[ "${path}" == /* ]] || path="$(pwd -P)/${path}"
if [[ "${FAKE_REMOTE_FAULT:-}" == removal-failure && "${path}" == */nemotron35-refresh/* ]]; then
  exit 0
fi
exec /usr/bin/rm "$@"
EOF
cat > "${STUB}/sudo" <<'EOF'
#!/usr/bin/env bash
[[ "$1" == -n ]] && shift
exec "$@"
EOF
cat > "${STUB}/install" <<'EOF'
#!/usr/bin/env bash
files=()
while [[ $# -gt 0 ]]; do case "$1" in -m|-o|-g) shift 2 ;; *) files+=("$1"); shift ;; esac; done
cp "${files[@]}"
EOF
cat > "${STUB}/flock" <<'EOF'
#!/usr/bin/env bash
shift 3
[[ "$#" -eq 0 ]] && exit 0
exec "$@"
EOF
# record-goldens stub: deterministic content per (name, step); a box named in
# ${FAKE}/mismatch writes different bytes; ${FAKE}/fail-<box>-<steps> makes one job fail.
RECORD_STUB="${WORK}/record-goldens.sh"
cat > "${RECORD_STUB}" <<'EOF'
#!/usr/bin/env bash
set -e
only=() steps=""
while [[ $# -gt 0 ]]; do
  case "$1" in --prompts) P="$2";; --out) O="$2";; --only) only+=("$2");; --steps) steps="$2";; esac; shift 2
done
echo "record ${BOX} ${steps} ${only[*]}" >> "${FAKE}/calls.log"
[[ ! -e "${FAKE}/fail-${BOX}-${steps}" ]] || { echo "boom" >&2; exit 1; }
extra=""; grep -qx "${BOX}" "${FAKE}/mismatch" 2>/dev/null && extra="${BOX}"
for n in "${only[@]}"; do
  [[ -f "${P}/${n}.tokens.json" ]]
  if [[ "${steps}" == serial ]]; then echo "{\"serial\":\"${n}${extra}\"}" > "${O}/${n}.golden.json"
  else [[ -f "${O}/${n}.golden.json" ]]; echo "{\"mtp${steps}\":\"${n}${extra}\"}" > "${O}/${n}.mtp${steps}.golden.json"; fi
done
EOF
cat > "${WORK}/fake-stitch" <<'EOF'
#!/usr/bin/env bash
count=8 chats=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) out="$2"; shift 2 ;;
    --label) label="$2"; shift 2 ;;
    --count) count="$2"; shift 2 ;;
    --chat) shift; chats=("$@"); break ;;
    *) shift ;;
  esac
done
mkdir -p "${out}"
if [[ "${#chats[@]}" -gt 0 ]]; then
  skip="$(cat "${FAKE}/skip-author-token" 2>/dev/null || true)"
  for chat in "${chats[@]}"; do
    name="${chat%%=*}"
    [[ "${name}" == "${skip}" ]] || echo "[${name}]" > "${out}/${name}.tokens.json"
  done
else
  for n in $(seq 1 "${count}"); do echo "[${n}]" > "${out}/${label}-${n}.tokens.json"; done
fi
EOF
chmod +x "${STUB}"/* "${REMOTE_STUB}"/* "${RECORD_STUB}" "${WORK}/fake-stitch"

BOXES="b1 b2 b3"
for b in ${BOXES}; do
  mkdir -p "${FAKE}/${b}/home/nemotron35/repo/tools" "${FAKE}/${b}/home/goldens"
  ln -s home "${FAKE}/${b}/home-link"
  cp "${RECORD_STUB}" "${FAKE}/${b}/home/nemotron35/repo/tools/record-goldens.sh"
  cp /dev/null "${FAKE}/${b}/home/goldens/.keep"
done

ln -s "${REPO_ROOT}/tools/pool-refresh.sh" "${ROOT}/tools/pool-refresh.sh"
cp "${REPO_ROOT}/fixtures/nemotron35_lightning_30b_a3b_track.json" "${ROOT}/fixtures/"
git -C "${ROOT}" init -q -b main
git -C "${ROOT}" add -A
git -C "${ROOT}" -c user.name=t -c user.email=t@t commit -qm base
git clone -q --bare "${ROOT}" "${WORK}/origin.git"
git -C "${ROOT}" remote add origin "${WORK}/origin.git"
git -C "${ROOT}" fetch -q origin
# The legacy pins must be "staged" on each box for verify.
for b in ${BOXES}; do
  jq -r '[.timed_prompt_pool[], (.speculative_oracles[] | .[])] | .[].r2_path | split("/")[-1]' \
    "${ROOT}/fixtures/nemotron35_lightning_30b_a3b_track.json" | sort -u | while read -r f; do echo legacy > "${FAKE}/${b}/home/goldens/${f}"; done
done
# ...with the sha256 the verify step compares: rewrite the fixture pins to match.
LEGACY_SHA="$(echo legacy | sha256sum | cut -d' ' -f1)"
jq --arg s "${LEGACY_SHA}" '.timed_prompt_pool[].sha256 = $s | .speculative_oracles[][].sha256 = $s
    | .hidden_correctness_golden.sha256 = $s' "${ROOT}/fixtures/nemotron35_lightning_30b_a3b_track.json" > "${WORK}/fx"
mv "${WORK}/fx" "${ROOT}/fixtures/nemotron35_lightning_30b_a3b_track.json"
git -C "${ROOT}" -c user.name=t -c user.email=t@t commit -qam "legacy pins" && git -C "${ROOT}" push -q origin HEAD:main

FX="${ROOT}/fixtures/nemotron35_lightning_30b_a3b_track.json"
# This scaffold has three live goldens. The default refresh count must match it.
jq '.official_pairs = 3 | .live_goldens = ["pool-9", "pool-10", "pool-11"]' "${FX}" > "${WORK}/fx" && mv "${WORK}/fx" "${FX}"
SHIPPED_POOL="$(jq '.timed_prompt_pool | length' "${FX}")"
SHIPPED_LIVE="$(jq -c '.live_goldens' "${FX}")"
GPU_LOCK="${WORK}/gpu.lock"
: > "${GPU_LOCK}"
refresh_raw() {
  env PATH="${STUB}:${PATH}" FAKE_REMOTE_FAULT="${FAKE_REMOTE_FAULT:-}" \
    REFRESH_BOXES="${TEST_BOXES:-${BOXES}}" REFRESH_PRIVATE="${PRIV}" \
    REFRESH_BOX_GOLDENS='$HOME/goldens' REFRESH_GPU_LOCK="${GPU_LOCK}" REFRESH_PYTHON="${WORK}/fake-stitch" \
    "${ROOT}/tools/pool-refresh.sh" "$@" 2>&1
}
refresh() {
  refresh_raw --corpus "${WORK}" --snapshot "${WORK}" "$@"
}
nf() { local c=0 f; for f in $1; do [[ -e "${f}" ]] && c=$((c + 1)); done; echo "${c}"; }  # files matching a glob
records() { grep -c "^record " "${FAKE}/calls.log" 2>/dev/null || echo 0; }

# --- count and text inputs -------------------------------------------------------
cp "${FX}" "${WORK}/fx-with-live"
jq 'del(.live_goldens)' "${FX}" > "${WORK}/fx" && mv "${WORK}/fx" "${FX}"
out="$(refresh --label g7 --dry-run)"
grep -q 'stitch-prompts.py.*--label g7 --count 8' <<<"${out}" \
  && grep -q 'g7-1.*g7-8' <<<"${out}" && ! grep -q 'g7-9' <<<"${out}" \
  && ok "a fixture without live_goldens defaults to eight prompts" \
  || fail "missing live_goldens did not default to eight prompts: $(tail -3 <<<"${out}")"
mv "${WORK}/fx-with-live" "${FX}"

out="$(refresh --label g1 --dry-run)"
if grep -q 'stitch-prompts.py.*--label g1 --count 3' <<<"${out}" \
  && grep -q 'g1-1.*g1-2.*g1-3' <<<"${out}" && ! grep -q 'g1-4' <<<"${out}" \
  && [[ ! -e "${PRIV}/refresh/g1/count" ]] \
  && [[ ! -s "${PRIV}/refresh/g1/state" ]] && [[ "$(records)" == 0 ]]; then
  ok "default count plans author and record for the three live goldens only"
else
  fail "default count plan is wrong: $(tail -3 <<<"${out}")"
fi

out="$(refresh --label g8 --dry-run --count 2)"
grep -q 'stitch-prompts.py.*--label g8 --count 2' <<<"${out}" \
  && grep -q 'g8-1.*g8-2' <<<"${out}" && ! grep -q 'g8-3' <<<"${out}" \
  && ok "--count 2 plans two prompts" || fail "--count 2 plan is wrong: $(tail -3 <<<"${out}")"

out="$(refresh --label g8 --dry-run --count 0002)"
grep -q 'stitch-prompts.py.*--label g8 --count 2' <<<"${out}" \
  && ok "--count strips leading zeros" || fail "--count did not strip leading zeros: $(tail -3 <<<"${out}")"

for bad in 0 x 65 18446744073709551617; do
  out="$(refresh --label g8 --dry-run --count "${bad}")" && fail "--count ${bad} was accepted"
  grep -q -- "--count ${bad}" <<<"${out}" && ok "--count ${bad} is refused by name" \
    || fail "--count ${bad} error does not name the value: ${out}"
done
out="$(refresh --label g8 --step author --dry-run --count '')" && fail "an empty --count was accepted"
grep -q -- "--count '':" <<<"${out}" && ok "an empty --count is refused by name" \
  || fail "empty --count error does not name the value: ${out}"

out="$(refresh --label g5 --step author --count 2)" || fail "author --count 2 failed: ${out}"
[[ "$(cat "${PRIV}/refresh/g5/count" 2>/dev/null)" == 2 ]] \
  && ok "author stores its count" || fail "author did not store count 2"
out="$(refresh --label g5 --step record --dry-run)"
grep -q 'g5-1.*g5-2' <<<"${out}" && ! grep -q 'g5-3' <<<"${out}" \
  && ok "record resumes with the stored count" || fail "record did not use stored count 2: $(tail -3 <<<"${out}")"
out="$(refresh --label g5 --step record --dry-run --count 3)" && fail "a conflicting resumed --count was accepted"
grep -q 'stored count 2' <<<"${out}" && grep -q 'supplied count 3' <<<"${out}" \
  && ok "a conflicting resumed --count names both values" \
  || fail "conflicting resumed --count error is wrong: ${out}"

TEXTS="${WORK}/texts"
mkdir -p "${TEXTS}"
for n in 1 2 3; do echo "chat ${n}" > "${TEXTS}/g3-${n}.txt"; done
out="$(refresh_raw --label g3 --step author --dry-run --snapshot "${WORK}" --texts "${TEXTS}")"
author_cmd="$(grep 'author-prompts.py' <<<"${out}" || true)"
if [[ "$(grep -c 'author-prompts.py' <<<"${out}")" == 1 ]] && ! grep -q 'stitch-prompts.py' <<<"${out}" \
  && grep -q -- "--chat g3-1=${TEXTS}/g3-1.txt g3-2=${TEXTS}/g3-2.txt g3-3=${TEXTS}/g3-3.txt" <<<"${author_cmd}"; then
  ok "--texts plans one chat author command and does not stitch"
else
  fail "--texts author command is wrong: ${out}"
fi

rm "${TEXTS}/g3-2.txt"
out="$(refresh_raw --label g3 --step author --dry-run --snapshot "${WORK}" --texts "${TEXTS}")" \
  && fail "--texts accepted a missing input"
grep -q "${TEXTS}/g3-2.txt" <<<"${out}" && ok "--texts names a missing input" \
  || fail "--texts missing-input error is wrong: ${out}"

echo "chat 2" > "${TEXTS}/g3-2.txt"
out="$(refresh_raw --label g3 --step author --snapshot "${WORK}" --texts "${TEXTS}")" \
  || fail "--texts author failed: ${out}"
copied=1
for n in 1 2 3; do cmp -s "${TEXTS}/g3-${n}.txt" "${PRIV}/refresh/g3/prompts/g3-${n}.txt" || copied=0; done
[[ "${copied}" == 1 ]] && ok "--texts keeps exact private copies" || fail "--texts changed or omitted a private copy"

for n in 1 2 3; do echo "chat ${n}" > "${TEXTS}/g4-${n}.txt"; done
echo g4-2 > "${FAKE}/skip-author-token"
out="$(refresh_raw --label g4 --step author --snapshot "${WORK}" --texts "${TEXTS}")" \
  && fail "--texts accepted a missing generated token file"
grep -q 'author: no g4-2.tokens.json' <<<"${out}" && ok "--texts names a missing generated token file" \
  || fail "--texts generated-token error is wrong: ${out}"
rm "${FAKE}/skip-author-token"

FINAL_TEXTS="${WORK}/final-texts"
mkdir -p "${FINAL_TEXTS}"
for n in 1 2; do echo "final chat ${n}" > "${FINAL_TEXTS}/final-${n}.txt"; done
out="$(refresh_raw --label final --step author --dry-run --count 2 --snapshot "${WORK}" --texts "${FINAL_TEXTS}")"
grep -q -- "--chat final-1=${FINAL_TEXTS}/final-1.txt final-2=${FINAL_TEXTS}/final-2.txt" <<<"${out}" \
  && ! grep -q 'final-3' <<<"${out}" && [[ ! -e "${PRIV}/final/count" ]] \
  && ok "--texts plans final-1 through final-N" || fail "final --texts plan is wrong: ${out}"
out="$(refresh_raw --label final --step author --count 2 --snapshot "${WORK}" --texts "${FINAL_TEXTS}")" \
  || fail "final --texts author failed: ${out}"
[[ "$(cat "${PRIV}/final/count" 2>/dev/null)" == 2 ]] \
  && ok "final --texts stores its count" || fail "final --texts did not store count 2"

# --- a crash mid-record resumes without re-recording finished jobs ---------------
touch "${FAKE}/fail-b2-3"
out="$(refresh --label g1)" && fail "a failed box job did not fail the run"
grep -q "mtp3@b2 on b2 failed" <<<"${out}" && ok "a failed box job is named" || fail "no failed-job message: $(tail -3 <<<"${out}")"
[[ ! -e "${FAKE}/b2/home/nemotron35-refresh/g1/mtp3@b2" ]] \
  && ok "a failed recording job removes its remote tree" \
  || fail "a failed recording job left its remote tree"
before="$(records)"
rm "${FAKE}/fail-b2-3"
out="$(refresh --label g1)" || fail "resume failed: $(tail -5 <<<"${out}")"
after="$(records)"
# 3 serial + 12 oracle jobs total; the resume re-runs only the failed one
# (plus any job its box had not reached yet).
[[ $((after - before)) -lt 12 ]] && ok "resume re-ran $((after - before)) job(s), not the whole record step" \
  || fail "resume re-ran $((after - before)) jobs"
[[ "$(jq length "${PRIV}/refresh/g1/pins.json")" == 21 ]] && ok "21 goldens pinned (3 prompts x serial+mtp1..6)" || fail "pins.json incomplete"
for b in ${BOXES}; do
  [[ "$(nf "${FAKE}/${b}/home/goldens/g1-*")" == 21 ]] || fail "${b}: g1 not staged"
done
ok "all boxes staged"
grep -c " g1 " "${PRIV}/refresh/staging.log" | grep -qx 63 && ok "staging log records every install (21 x 3 boxes)" || fail "staging log count"
if [[ "$(jq -c '.live_goldens' "${FX}")" == '["g1-1","g1-2","g1-3"]' ]] \
  && [[ "$(jq -c '[has("live_golden_rotation", "live_golden", "calibration_prompt")]' "${FX}")" == '[false,false,false]' ]] \
  && [[ "$(jq '.timed_prompt_pool | length' "${FX}")" == $((SHIPPED_POOL + 3)) ]] \
  && [[ "$(jq '.speculative_oracles["g1-3"] | length' "${FX}")" == 6 ]] \
  && [[ "$(jq -r '.hidden_correctness_golden.sha256' "${FX}")" == "$(jq -r '.["g1-1.golden.json"].sha256' "${PRIV}/refresh/g1/pins.json")" ]]; then
  ok "pin: live_goldens, hidden golden and oracles point at g1; no rotation keys"
else
  fail "pin: fixture not as expected"
fi
grep -q "git -C .* push origin HEAD:main" <<<"${out}" && [[ "$(git -C "${ROOT}" rev-list --count HEAD)" == 2 ]] \
  && ok "pin prints the commit/push commands and commits nothing" || fail "pin committed or printed nothing"
# every golden agreed across two boxes: the serial of g1-1 came from b1 and b2
grep -q "^record b1 serial .*g1-1" "${FAKE}/calls.log" && grep -q "^record b2 serial .*g1-1" "${FAKE}/calls.log" \
  && ok "each serial golden recorded on two boxes" || fail "serial not recorded twice"
if ! grep -q '^copy-mode ' "${FAKE}/calls.log"; then
  fail "no recording copy mode was checked"
elif awk '$1 == "copy-mode" && ($4 != "700" || $5 != $6) { bad = 1 } END { exit !bad }' "${FAKE}/calls.log"; then
  fail "a recording copy started before its remote job directory was private"
else
  ok "each remote recording job is mode 0700 and owned by its user before copy"
fi
clean=1
for b in ${BOXES}; do
  [[ "$(nf "${FAKE}/${b}/home/nemotron35-refresh/g1/*")" == 0 ]] \
    || { fail "${b}: a successful recording job left its remote tree"; clean=0; }
done
[[ "${clean}" == 0 ]] || ok "successful recording jobs remove their remote trees"

# --- remote recording guard failures -------------------------------------------
record_guard_case() {
  local label="$1" fault="$2" message="$3" out
  out="$(refresh --label "${label}" --step author --count 1)" \
    || { fail "${fault}: author failed: ${out}"; return; }
  if [[ "${fault}" == removal-failure ]]; then
    mkdir -p "${FAKE}/b1/home/nemotron35-refresh/${label}/serial@b1"
    mkdir -p "${FAKE}/b2/home/nemotron35-refresh/${label}/serial@b2"
  fi
  : > "${FAKE}/scp.log"
  out="$(FAKE_REMOTE_FAULT="${fault}" TEST_BOXES='b1 b2' \
    refresh --label "${label}" --step record --count 1)" \
    && fail "${fault}: the recording guard accepted the fault"
  if grep -qF "${message}" <<<"${out}" \
    && [[ ! -s "${FAKE}/scp.log" ]] \
    && ! grep -q '^done job ' "${PRIV}/refresh/${label}/state"; then
    ok "${fault}: the run fails before copy and no job is complete"
  else
    fail "${fault}: guard result is wrong: $(tail -3 <<<"${out}")"
  fi
}

record_guard_case g20 wrong-mode "cannot create a private recording directory"
record_guard_case g21 wrong-owner "cannot create a private recording directory"
record_guard_case g22 stat-failure "cannot create a private recording directory"
record_guard_case g23 removal-failure "cleanup failed"
record_guard_case g24 home-unset "REFUSING unsafe remote home"
record_guard_case g25 home-empty "REFUSING unsafe remote home"
record_guard_case g26 home-root "REFUSING unsafe remote home"
record_guard_case g27 home-dotdot "REFUSING unsafe remote home"
record_guard_case g28 home-symlink "REFUSING unsafe remote home"

# --- dangerous cleanup path ----------------------------------------------------
out="$(refresh --label g6 --step author --count 1)" || fail "g6 author failed: ${out}"
mkdir -p "${WORK}/do-not-remove"
echo keep > "${WORK}/do-not-remove/sentinel"
mkdir -p "${FAKE}/b1/home/nemotron35-refresh/g6"
ln -s "${WORK}/do-not-remove" "${FAKE}/b1/home/nemotron35-refresh/g6/serial@b1"
out="$(refresh --label g6 --step record)" && fail "a symlink recording job path was accepted"
[[ -L "${FAKE}/b1/home/nemotron35-refresh/g6/serial@b1" && -f "${WORK}/do-not-remove/sentinel" ]] \
  && ok "cleanup refuses a dangerous job path and does not follow its symlink" \
  || fail "cleanup followed or removed a dangerous job path"

# A box name reaches the job key. A slash must not let cleanup leave the generation.
out="$(refresh --label g7 --step author --count 1)" || fail "g7 author failed: ${out}"
mkdir -p "${FAKE}/b1/home/nemotron35-refresh/g7/serial@b1"
mkdir -p "${FAKE}/b1/home/nemotron35-refresh/g8/victim"
echo keep > "${FAKE}/b1/home/nemotron35-refresh/g8/victim/sentinel"
out="$(TEST_BOXES='b1/../../g8/victim b2' refresh --label g7 --step record)" \
  && fail "a traversal recording job path was accepted"
[[ -f "${FAKE}/b1/home/nemotron35-refresh/g8/victim/sentinel" ]] \
  && ! grep -Fq 'ssh b1/../../g8/victim ' "${FAKE}/calls.log" \
  && ok "cleanup refuses traversal before it reaches a sibling generation" \
  || fail "cleanup reached or removed a sibling generation through traversal"

# --- verify ---------------------------------------------------------------------
out="$(refresh --label g1 --step verify)" && fail "verify passed before the push" \
  || { grep -q "push the pin commit first" <<<"${out}" && ok "verify refuses before the pin is on origin/main" || fail "verify: $(tail -2 <<<"${out}")"; }
git -C "${ROOT}" -c user.name=t -c user.email=t@t commit -qam "pin g1" && git -C "${ROOT}" push -q origin HEAD:main
echo x > "${FAKE}/b3/home/goldens/stray.golden.json"
out="$(refresh --label g1 --step verify)" && fail "verify passed with a stray golden" \
  || { grep -q "b3 holds unpinned goldens: stray.golden.json" <<<"${out}" && ok "verify refuses an unpinned golden by name" || fail "verify: $(tail -2 <<<"${out}")"; }
rm "${FAKE}/b3/home/goldens/stray.golden.json"
echo x > "${FAKE}/b3/home/goldens/g9-1.golden.json"
out="$(refresh --label g1 --step verify)" && ok "verify passes (a later generation staged ahead is tolerated)" || fail "verify: $(tail -3 <<<"${out}")"
rm "${FAKE}/b3/home/goldens/g9-1.golden.json"

# --- byte-identical check ---------------------------------------------------------
echo b2 > "${FAKE}/mismatch"
out="$(refresh --label g2)" && fail "a mismatching box was accepted"
grep -q "g2-1.golden.json differs between serial@b1" <<<"${out}" && [[ ! -f "${PRIV}/refresh/g2/pins.json" ]] \
  && [[ "$(nf "${FAKE}/b1/home/goldens/g2-*")" == 0 ]] \
  && ok "a byte mismatch between two boxes refuses before anything is pinned or staged" \
  || fail "mismatch: $(tail -3 <<<"${out}")"
rm "${FAKE}/mismatch"

# --- final stays off the boxes -------------------------------------------------------
out="$(refresh --label final)" || fail "final author+record failed: $(tail -3 <<<"${out}")"
[[ -f "${PRIV}/final/pins.json" ]] && [[ "$(nf "${FAKE}/*/home/goldens/final-*")" == 0 ]] \
  && ok "final: authored and recorded, nothing staged" || fail "final touched a box"
out="$(refresh --label final --step stage)" && fail "final was staged"
grep -q "REFUSING to stage the held-out final" <<<"${out}" && ok "final stage is refused" || fail "final stage: ${out}"
out="$(refresh --label final --step pin)" && fail "final was pinned"
grep -q "REFUSING to pin the held-out final" <<<"${out}" && ok "final pin is refused" || fail "final pin: ${out}"
out="$(refresh --label final --step stage --final-after-freeze "${WORK}/nope")" && fail "final staged without a freeze"
touch "${WORK}/FROZEN"
out="$(refresh --label final --step stage --final-after-freeze "${WORK}/FROZEN")" || fail "final stage after freeze: $(tail -3 <<<"${out}")"
grep -q " final b1 final-1.golden.json " "${PRIV}/refresh/staging.log" && ok "final staged after the freeze, and logged" || fail "final not logged"

# --- retire ------------------------------------------------------------------------
out="$(refresh --label g1 --step retire)" && fail "retired the live generation"
grep -q "still the live goldens" <<<"${out}" && ok "retire refuses the live generation" || fail "retire: ${out}"
jq --argjson l "${SHIPPED_LIVE}" '.live_goldens = $l' "${FX}" > "${WORK}/fx" && mv "${WORK}/fx" "${FX}"
out="$(refresh --label g1 --step retire)" || fail "retire unpin: $(tail -3 <<<"${out}")"
[[ "$(jq '.timed_prompt_pool | length' "${FX}")" == "${SHIPPED_POOL}" && "$(jq 'has("speculative_oracles") and (.speculative_oracles | has("g1-1") | not)' "${FX}")" == true ]] \
  && [[ "$(nf "${FAKE}/b1/home/goldens/g1-*")" -gt 0 ]] && ok "retire phase 1 unpins g1 and leaves the boxes alone" || fail "retire phase 1"
out="$(refresh --label g1 --step retire)" && fail "retire removed goldens before the unpin was pushed"
git -C "${ROOT}" -c user.name=t -c user.email=t@t commit -qam "retire g1" && git -C "${ROOT}" push -q origin HEAD:main
out="$(refresh --label g1 --step retire)" || fail "retire phase 2: $(tail -3 <<<"${out}")"
[[ "$(nf "${FAKE}/*/home/goldens/g1-*")" == 0 && "$(nf "${PRIV}/refresh/g1/goldens/*")" == 21 ]] \
  && ok "retire phase 2 removes g1 from every box; the private archive stays" || fail "retire phase 2 left files"

if [[ "${failures}" -eq 0 ]]; then
  echo "test-pool-refresh.sh: all cases passed"
  exit 0
fi
echo "test-pool-refresh.sh: ${failures} case(s) failed" >&2
exit 1

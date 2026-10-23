# Hidden-pool refresh and final re-score

Status: current design, implemented by `tools/pool-refresh.sh` and
`tools/router/router.py`. Contest length two weeks, unlimited submissions per
team.

## Why

A candidate engine necessarily sees the scored prompt at run time. The sealed
record names a prompt only by its sha256, but it can carry engine-chosen
values. Timing is a slow side channel too, and teams may submit without limit.
A prompt learned that way could be hard-coded into an engine for a fake
speed-up. Three layers limit the damage:

1. **Every live golden per job**: each ranked job measures every prompt of
   the fixture's `live_goldens`, one pair per prompt, and scores the mean of
   the per-pair composites.
2. **Pool refresh every 3-4 days** (4 refreshes in two weeks): the live pool is
   replaced by a new generation of the same size (now 3 prompts) that no
   submission has seen.
3. **Final re-score**: at the close, the top submissions are re-measured on a
   held-out pool that was never staged on a box during the contest. The final
   ranking comes only from this. Anything leaked during the contest is worth
   nothing at the end, and day-to-day prompt drift does not decide the winner.

## Prompt generation (both the refresh and the held-out pool)

**Current form (since 2026-10-19): organizer-written chat prompts.** The live
pool (`pool-9`, `pool-10`, `pool-11`) and the held-out `final` pool are coding
tasks, each written as one user message and rendered through the snapshot's
pinned chat template with thinking off. `tools/author-prompts.py --chat`
renders each text and refuses any rendering that is not exactly `seed_tokens`
(496) ids; a chat prompt is never trimmed, so the organizer edits the text
until it fits. `tools/pool-refresh.sh --texts DIR` runs this step for a
generation (`DIR/<label>-<n>.txt`). Each new text must be a task not used in
any earlier generation; the texts, their token files and their hashes go to
the organizer archive ([reproduce.md, section 2](reproduce.md#2-re-record-the-hidden-goldens)).

**Earlier form: stitched book passages** (the 4096-token pools of September,
and the default of `tools/pool-refresh.sh` without `--texts`). The first live pool's
prompts are single passages of public-domain books, so a few leaked tokens
identify the book and therefore the whole prompt. New prompts are **stitched**:

- A corpus directory of public-domain works (plain text), kept in
  `.mlxfast-private/corpus/` (never in git), large enough for ~40 prompts.
- Each prompt is built from paragraphs drawn at random from several different
  works (at least 4 works per prompt), in random order, until it reaches the
  fixture's `seed_tokens` length with the target tokenizer
  (`tools/author-prompts.py` does the final trim). A leaked fragment identifies
  one paragraph, not the prompt.
- Randomness from `/dev/urandom`. The generator records, privately, which work
  and paragraph each piece came from, so a prompt can be audited.
- Every generation gets a label (`g1`, `g2`, …, and `final` for the held-out
  pool); prompt names are `<label>-<n>` (for example `g2-5`).

## Refresh (`tools/pool-refresh.sh`, run from the organizer workstation)

Idempotent steps, each refusing on any mismatch:

1. **Author**: N prompts for the next generation (above). N is `--count`,
   by default the number of `live_goldens` in the fixture (3 now); use
   `--texts DIR` for chat prompts.
2. **Record**: serial golden plus MTP-1…6 oracles for every prompt
   (N × 7 goldens, 21 for 3 prompts), spread across the fleet with `tools/record-goldens.sh`,
   each one on **two** boxes and required to be byte-identical (the engine is
   deterministic; a mismatch means a broken box or engine).

   Each recording job resolves and checks the remote home once.
   The home must be an absolute, canonical path that the remote user owns.
   It must not be `/`, contain a `..` component, or have a symlink ancestor.
   The script builds all recording paths from this checked value.
   Each job uses `<checked-home>/nemotron35-refresh/<label>/<key>`.
   The remote shell creates this tree with `umask 077` and mode `0700`.
   Before the first copy, the script checks each directory's mode and owner.
   After retrieval, the script checks that each expected output exists.
   It then deletes only the job directory and verifies that the directory is absent.
   An `EXIT` trap tries the same cleanup after each job failure.
   Cleanup accepts exactly one nonempty job-name component below the generation.
   The component can contain letters, digits, `_`, `.`, `@`, `:`, `%`, `+`, and `-`.
   Cleanup refuses the generation root, `/`, `.`, `..`, names with `/`, outside paths, and symlink paths.
   Cleanup does not follow symlinks.

3. **Stage**: install the new goldens next to the current ones on every box of `REFRESH_BOXES`
   (`/srv/nemotron35/goldens`, root:bench 0750), verified by sha256. The old
   generation stays in place, so in-flight jobs are unaffected.
4. **Pin and switch**: one commit to the trusted fixture that pins the new
   goldens and sets `live_goldens` to the new generation, pushed to `main`
   (and to the armed base branch the dispatcher overlays onto). Jobs
   dispatched after the push use the new pool; jobs already running keep the
   pool of the commit they were dispatched at.
5. **Verify**: the ranked preflight on every box passes with the new fixture
   (pins staged and matching) before the script reports success.
6. **Retire**: after all jobs on the old generation have finished, remove its
   goldens from the boxes; keep a private archive (for disputes).

Timing: fleet time is spent in step 2 (and a short drain). The script
records its progress so a crash resumes where it stopped. The script itself
never calibrates; the organizer does, between stage and pin (below).

Calibration: the box calibration file carries one band per live golden
(`tools/calibrate-box.sh`), and benchd gates each control leg with the band of
the prompt it measured. A new generation has no bands yet, so after the stage
step every box must be calibrated again on the new live goldens
([reproduce.md, section 3](reproduce.md#3-recalibrate-a-box)), and the new
files installed, before the pin commit is pushed; otherwise the ranked
preflight refuses every job. The pin step also refuses an
`official_pairs` that is not a multiple of the generation size, because every
job measures every live golden.

## Final re-score (`router.py rescore`, operator-run at the close)

1. **Held-out pool**: generation `final` is authored and recorded: 3
   organizer-written 496-token coding prompts × 7 goldens = 21 goldens,
   agreed byte for byte by two boxes (Spark box 4 and Spark box 2, 2026-10-20). It
   replaces the September pool of 8 stitched 4096-token prompts, which does
   not fit the 3-pair, 496-token fixture.
   It is kept off the boxes: goldens and prompts live only in
   `.mlxfast-private/final/` on the organizer workstation and in the organizer
   archive (never in git, never staged, never pinned in a pushed fixture)
   until the close. The script deletes each box-side job after it verifies
   retrieval. It compares the two local copies later. The pool must have as many
   prompts as `official_pairs` (or a divisor of it), because the re-score fixture
   makes it the `live_goldens`.
2. **Freeze**: at the deadline the router stops accepting submissions.
3. **Select**: for each team, its best submission by mean contest score over
   runs that scored or failed because of the submission (a team failure counts
   as 0.0; a post-setup `control_failed` or `band_refused` run also counts as
   0.0; other infrastructure runs are excluded). Ties go to the most recent
   submission; then the top N teams (N set by the organizer; default 10).
4. **Stage and pin**: stage the `final` goldens on all boxes; build a
   `rescore/base` branch = the contest's final base plus one fixture commit
   whose `live_goldens` are the `final` generation.
5. **Re-measure**: each selected submission's editable paths are overlaid onto
   `rescore/base` (the same REPLACE overlay as normal submission) and dispatched
   K times (default K = 5), one run per slot, spread across different boxes, so
   every submission is scored on several `final` prompts and boxes.
6. **Slots and retries**: each team has exactly K slots. A slot's value is its
   first run that ends with a score or a team failure (replay rejected, floor or
   correctness failure, refusal, unsealed, `watchdog_timeout`, or a watchdog
   cancellation because the engine hung). A cancelled run without a sealed
   score is also a team failure and is not re-dispatched. A run whose scored
   prompt is not in its fixture's `live_goldens` counts as infrastructure.
   The router checks team failures and watchdog outcomes before the prompt.
   It marks a run off-pool only with a sealed, finite numeric score and a
   present scored prompt. Booleans, text, NaN, and infinite values are not
   score evidence. A missing score or prompt is not off-pool evidence.
   Infrastructure outcomes do not fill a slot. `router.py rescore-retry`
   re-dispatches those slots. Per slot, total
   infrastructure attempts = `1 + --max-infra-retries` (default 2 extra attempts
   after the first, so 3 tries). The first candidate resident boot failure in a
   slot may be re-dispatched once (separate from the infra budget); a second
   boot failure in that slot counts as 0.0. After setup, a control-leg failure
   (`control_failed`) or band refusal (`band_refused`) that still fails after
   those attempts counts as **0.0** for that slot (status `penalized`), because
   the submission already ran and the fault is induced by the engine afterward.
   Pre-setup infrastructure (preflight refusal, setup failure, or off-pool)
   leaves the slot **unresolved** when retries are exhausted.
7. **Rank**: the mean is the average of the K slot values (the score, 0.0 for a
   team failure, or 0.0 for a penalized post-setup infrastructure slot). A team
   with any slot still pending or unresolved has no mean and status `incomplete`;
   incomplete teams are listed after every complete team and are never ranked
   above them. `rescore-report` lists every run and, per team, each slot's run
   ids, outcomes, and value so the ranking can be audited.

## Checks

- A stitched prompt never contains a paragraph that appears in any earlier
  generation or in the held-out pool (the generator refuses reuse). A chat
  prompt is a new task, checked by the organizer against the archive.
- The refresh never stages the `final` generation; the rescore refuses to run
  if any `final` golden was ever staged before the freeze (the staging log is
  the evidence).
- Both scripts have `--dry-run`, and a test suite with stubbed boxes.

## Where the goldens live

| Copy | Holds | Purpose |
|---|---|---|
| Each Spark, `/srv/nemotron35/goldens` (root:bench, unreadable by the sandbox user) | the live pool (and the next generation once staged) | the only copy scoring reads; a ranked job never fetches goldens |
| Organizer staging host (a private path, 0700) | the live pool, with `SHA256SUMS` from the fixture pins | the source for restaging a rebuilt box |
| Organizer archive (a private repository, organizer only) | the live pool, every retired generation, the held-out `final` pool, the paragraph ledger | backup and dispute evidence |

The held-out `final` pool never goes to the jump box or a Spark before the
freeze. Goldens are never committed to the challenge repository, because the
sandboxed participant code can read the ranked checkout. After each refresh,
copy the new generation to the jump box and the archive. Before restaging a box
from the jump box, verify with `sha256sum -c SHA256SUMS`.

## Runbook (organizer commands, from the organizer workstation's checkout)

Tools: `tools/author-prompts.py --chat` or `tools/stitch-prompts.py` (authoring), `tools/pool-refresh.sh` (refresh),
`tools/router/router.py freeze|rescore|rescore-retry|rescore-report`. Private state lives
under `.mlxfast-private/`: `refresh/<label>/` per generation (state file,
prompts, provenance, per-box recordings, agreed goldens, `pins.json`),
`final/` for the held-out pool, `refresh/ledger.json` (every paragraph ever
used) and `refresh/staging.log` (every golden installed on a box, with time).

Box-side assumptions (override by env): the boxes named in `REFRESH_BOXES`
(`REFRESH_BOXES`, required), checkout `~/nemotron35/repo` (`REFRESH_BOX_REPO`, must be at a
commit that has `record-goldens.sh --only`), goldens `/srv/nemotron35/goldens`
(`REFRESH_BOX_GOLDENS`), snapshot `REFRESH_BOX_SNAPSHOT`, GPU lock
`/tmp/mtplx-gpu-exclusive.lock`. Stitching needs the `tokenizers` package:
`REFRESH_PYTHON` may name a python that has it (the pinned image does).

**Before the contest: author and record the held-out pool** (done for this
contest; kept for a replacement pool or the next contest).

```bash
tools/pool-refresh.sh --label final --dry-run --texts <DIR with final-1..3.txt> --snapshot <SNAPSHOT>
tools/pool-refresh.sh --label final --texts <DIR with final-1..3.txt> --snapshot <SNAPSHOT>
# -> .mlxfast-private/final/{prompts,goldens,pins.json}; nothing is staged or pinned
```

**Each refresh (every 3-4 days), generation g<N>:**

```bash
tools/pool-refresh.sh --label g2 --dry-run --texts <DIR with g2-1..3.txt> --snapshot <SNAPSHOT>
tools/pool-refresh.sh --label g2 --step author --texts <DIR> --snapshot <SNAPSHOT>
tools/pool-refresh.sh --label g2 --step record
tools/pool-refresh.sh --label g2 --step stage
#   bands for the NEW prompts: calibrate against a proposed fixture, not the live one
cp fixtures/nemotron35_lightning_30b_a3b_track.json /tmp/next-fixture.json
tools/pool-refresh.sh pin-fixture /tmp/next-fixture.json g2 .mlxfast-private/refresh/g2/pins.json
#   copy /tmp/next-fixture.json to every box, then on each box:
#   tools/calibrate-box.sh <runner name> <new file> --fixture <next-fixture> --passes 3
#   keep the old calibration in place until the switch: the live fixture needs its bands
tools/pool-refresh.sh --label g2 --step pin
#   at the switch: drain (no job queued or running), install every box's new
#   calibration file, then push the pin commit
#   record: 21 goldens x 2 boxes, byte-identical
#   a crash or a failed box: fix it and re-run the same command; finished work is skipped
#   one step alone: --step author|record|stage|pin|verify|retire
# review the fixture diff, then run the printed commands:
git add fixtures/nemotron35_lightning_30b_a3b_track.json
git commit -m "pool refresh: pin g2 and switch the live goldens"
git push origin HEAD:main && git push origin HEAD:<armed-base>
tools/pool-refresh.sh --label g2 --step verify      # every box matches origin/main's pins
# once no job dispatched before the switch is still queued or running:
tools/pool-refresh.sh --label g1 --step retire      # unpins g1: commit and push as printed
tools/pool-refresh.sh --label g1 --step retire      # after that push drains: removes g1 from the boxes
```

The pin commit sets `live_goldens` to the new prompts and
`hidden_correctness_golden` to the serial tape of `<label>-1`. The old
generation stays pinned and staged until it is retired, so
jobs queued at the old commit keep passing. The ranked preflight tolerates
unpinned `g<N>-*` goldens (staged ahead of their pin) and nothing else, so a
`final-*` golden on a box before the close fails every job.

**On Yukon** this track is a scalar row with a GitHub Actions runner, so every
new submission is built on the tip of the row's branch (Yukon
`src/execution/github-actions.ts`: `pinSource` is set only for Pareto rows).
A pin commit pushed to `main` therefore reaches the next Yukon submission
without any change to the row; the submission test of 2026-10-21 was built
on the then-current `main`. A pin push does not recompute the baseline
or past submission scores. Winning submissions can update the current best.
To recompute the baseline and replay eligible submissions, pause, drain, and
call `POST /api/benchmarks/<id>/reorg`.

The armed base branch is the branch the dispatcher overlays submissions onto.
Its name is configurable: `router.py --base` for the router and
`REFRESH_ARMED_BASE` for the refresh script (both carry a default in the code;
set them to the branch you arm). `<armed-base>` above stands for that name.

A byte mismatch between the two boxes that recorded a golden stops the
record step before anything is pinned. Find the bad box (`rec-<job>.log`,
determinism probe), delete that job's `done job <key>` line from
`.mlxfast-private/refresh/<label>/state` and re-run.

**At the close:**

```bash
tools/router/router.py freeze
tools/router/router.py status        # wait until nothing is queued, dispatched or running
tools/pool-refresh.sh --label final --step stage --final-after-freeze ~/nemotron35-router/FROZEN
#   bands for the final prompts: a private proposed fixture, then every box
cp fixtures/nemotron35_lightning_30b_a3b_track.json /tmp/final-fixture.json
tools/pool-refresh.sh pin-fixture /tmp/final-fixture.json final .mlxfast-private/final/pins.json
#   on each box: tools/calibrate-box.sh <runner name> <file> --fixture <final-fixture> --passes 3,
#   install the file at the runner's MLXFAST_BASELINE_CALIBRATION
tools/router/router.py rescore --base-ref origin/main --top 10 --runs 5
git -C ~/nemotron35-router/clone push origin rescore/base   # as printed; the sub/ branches carry it anyway
# repeat until rescore-report is complete (no incomplete teams):
tools/router/router.py rescore-retry --tag final --max-infra-retries 2   # infra + one boot retry per slot
tools/router/router.py rescore-report --wait    # ~/nemotron35-router/rescore-final/rescore.{json,html}
```

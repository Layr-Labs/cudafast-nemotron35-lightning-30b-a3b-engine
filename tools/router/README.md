# tools/router/

Organizer tooling, trusted side, outside `editablePaths`. It covers three jobs:

- a **dispatcher stand-in** for Yukon's submission service
  (`docs/submission-restriction-spec.md` sections 1 and 4) that drives
  `.github/workflows/benchmark.yml` on the box runners (`tools/box-runner/`);
- the **acceptance load test** (`loadgen.py`): seeded synthetic submissions
  with declared expectations;
- the contest close: **freeze and the final re-score**.

| File | Purpose |
|---|---|
| `router.py` | `submit` (keep only `editablePaths`, REPLACE overlay onto the base, push `sub/<id>`), `run` (dispatch one waiting run per online runner, watch, 90-minute watchdog, download `score.json`, classify), `status`, `report` (JSON + HTML). State: SQLite in `~/nemotron35-router`. |
| `loadgen.py` | Seeded synthetic submissions for the acceptance load test: null, MTP depths, numerics variants, garbage (replay must reject) and negatives, each with a declared expectation. |
| `test_router.py` | Overlay semantics, classification, every variant applying to the real tree, teams, freeze and the re-score. No network. |

### `--expect` outcomes

`submit --expect` names which terminal `outcome` values count as meeting the
submission's goal (`router.py` `EXPECT`):

| `--expect` | Counts as meeting |
|---|---|
| `pass` | `pass` |
| `replay_rejected` | `replay_rejected` |
| `boot_failed` | `boot_failed` (the candidate resident did not boot; load-test variant `neg_boot_fail`) |
| `refused` | `surface_refused`, `preflight_refused`, `refused`, `unsealed`, `watchdog_timeout` |
| `measure` | any terminal outcome (acceptance load test) |

Team-side failures after participant code ran include **`unsealed`** (no valid
sealed score after Setup) and **`watchdog_timeout`** (router watchdog cancelled
the run). Manual **`cancelled`** is infrastructure. So is **`control_failed`**: the reference
(serial-control) leg diverged twice, or failed to boot twice in pair 1.

Requirements: the `gh` CLI logged in with `repo` scope, SSH push access to the
repository, and a base branch whose fixture is armed.

| Env | Default |
|---|---|
| `ROUTER_REPO` | owner/repo; else `gh repo view` for this checkout (`--repo` overrides) |
| `ROUTER_BASE` | base branch, `loadtest/base` (`--base` overrides) |
| `WATCH_HOSTS` | **required**: space-separated ssh hosts `watch.sh` probes |

```bash
python3 -m unittest discover -s tools/router
tools/router/loadgen.py --only null,garbage       # smoke test
tools/router/loadgen.py --hours 8                 # the acceptance load test
tools/router/router.py report                     # ~/nemotron35-router/report/
```

## Teams, freeze and the final re-score

Every submission belongs to a **team** (`submit --team T`, `loadgen.py --team T`;
default: the `--variant` label). Older databases gain the `team` column on
start, filled from the label.

`router.py freeze` writes `~/nemotron35-router/FROZEN` (the deadline, epoch
seconds); from then on `submit` refuses. The daemon still finishes what was
already queued. `router.py unfreeze` removes the flag.

`router.py rescore --base-ref origin/main [--top 10] [--runs 5]` runs the final
re-score (`docs/pool-refresh-and-final-rescore.md`). It refuses unless the
router is frozen, no contest submission is still queued or running, and the
staging log (`.mlxfast-private/refresh/staging.log`) shows the `final` goldens
staged, none of them before the freeze. Then it:

1. picks each team's best submission: for each editable tree, the mean contest
   score counts runs that scored or failed because of the submission (failure =
   0.0; post-setup control or band failure = 0.0; other infrastructure runs are
   excluded); the best mean wins, ties go to the most recent; then the top N
   teams;
2. builds the local branch `rescore/base` = `--base-ref` plus one fixture commit
   (`tools/pool-refresh.sh pin-fixture ... final .mlxfast-private/final/pins.json`)
   whose rotation pool is the `final` generation, and prints its push command;
3. overlays each selected submission's editable paths onto `rescore/base` exactly
   as `submit` does (REPLACE) and queues K runs (one per slot) as its own
   `sub/<id>` branches, tagged `rescore=final`, with a `rescore_slot` index on
   each row. The running `run` daemon dispatches them; GitHub picks the box, so
   K runs normally land on several boxes and draw several prompts.

`router.py rescore-retry [--tag final] [--max-infra-retries 2]` re-dispatches
every slot whose latest attempt ended in infrastructure and still has retries
left (same overlay as `rescore`). Each slot gets `1 + max-infra-retries`
infrastructure attempts (default 3), plus at most one re-dispatch after a
candidate `boot_failed` outcome (a second boot failure in the slot counts as 0.0).
Post-setup `control_failed` / `band_refused`
that exhaust retries count as 0.0 (`penalized`); pre-setup infra leaves the slot
`unresolved`. It refuses while any rescore run is still queued or running.
Idempotent.

`router.py rescore-report [--wait] [--max-infra-retries 2]` writes
`~/nemotron35-router/rescore-final/` (`rescore.json` + `rescore.html`): every
run (team, box, prompt, score, outcome, `off_pool`) and a ranking by mean over
K slots (score, 0.0 on team failure, or 0.0 when post-setup infra exhausts
retries). Each slot records its attempts; a team with any slot pending or
unresolved has status `incomplete`, no mean, and ranks below every complete team.
A run whose prompt is not `final-*` is `off_pool` and counts as infrastructure
(re-run via `rescore-retry`, never as a score).

## Where a score came from

Each report row carries `score_source`: `artifact` (the uploaded
`score.json`), `job-log` (benchd's one-line `wrote score.json` marker, read
when the artifact is missing, for example over the storage quota) or empty (no
score). The summary counts them under `by_score_source`.

Neither source is fit for ranking yet. The job runs untrusted submission code
(`build.rs`, the engine) as the runner user, which can rewrite the job's own
tools and so forge an artifact or a log line alike. Ranking needs the build
and the engine sandboxed away from the measuring tools first (port notes
§7.9). Until then the numbers are acceptance-test data only.


## Durable user services

Install or update the router and the 30-minute fleet watcher from this checkout:

```bash
tools/router/install-service.sh
```

This renders the templates in `systemd/` into `~/.config/systemd/user` (or
`$XDG_CONFIG_HOME/systemd/user`), replaces same-name transient units, reloads
systemd, and enables and starts both services. Re-running it updates and
restarts them. Stop any separately backgrounded router or old
`~/nemotron35-router/watch.sh` process first to avoid duplicate daemons.
`loadgen.py` remains manual.

The installer records the absolute checkout path and the installing shell's
`PATH`, including tools such as mise-managed `gh`. Run it as your regular user,
with authenticated `gh`, Git, SSH, and the watcher's existing fleet tools
available. Re-run it after moving the checkout or changing tool paths. Services
use the account's existing credentials; no credentials are copied into units.

Both services retain the sleep/idle inhibitor and restart after failures with a
30-second delay. Router logs and watcher snapshots go to the journal; the
copied watcher no longer appends to `~/nemotron35-router/watch.log`.

```bash
systemctl --user status nemotron35-router.service nemotron35-watch.service
journalctl --user -u nemotron35-router.service -f
journalctl --user -u nemotron35-watch.service -f
systemctl --user stop nemotron35-router.service nemotron35-watch.service
systemctl --user start nemotron35-router.service nemotron35-watch.service
```

The router reopens `~/nemotron35-router/state.db` on startup, watches stored
`dispatched`/`running` jobs and dispatches queued submissions. Stop/start,
updates, and uninstall preserve the database and reports.

To start at boot without an interactive login and keep running after logout,
enable lingering. The installer only prints this command; an administrator may
need to authorize it:

```bash
loginctl enable-linger "$USER"
```

Remove the services, or render and test without a systemd session:

```bash
tools/router/install-service.sh --uninstall
tools/router/install-service.sh --render /tmp/nemotron35-router-units
python3 tools/router/test_router.py
python3 tools/router/test_services.py
```

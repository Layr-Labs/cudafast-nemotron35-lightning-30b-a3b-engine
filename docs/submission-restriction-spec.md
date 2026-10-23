# Submission-restriction enforcement: extracted spec and parity notes

This document records what an untrusted submission is allowed to CONTAIN, and
how much of it, rule by rule: what the upstream reference implementation
enforces (with a citation for each rule), what this repository enforces, what
the intake service (the dispatcher that accepts a participant's archive and
starts the ranked workflow) must enforce, and where this repository
deliberately differs from the reference.

The byte caps have one live enforcer here: the pre-dispatch static review
(`.github/scripts/submission-static-review-checks.sh`, §3), which runs in the
ranked workflow's `surface-check` job on a box runner before the measurement
job starts. `tools/lint-benchmark-manifest.py` check 3b holds its fallback
constants to the manifest. The upstream launch-time byte budget (R1) is cited
below as reference only; it does not run here.

## 0. Provenance of every citation

The reference is the challenger repository, read-only:

```text
Layr-Labs/qwen-3.8-mtp-challenge   main @ bfab0de58d43453e506523707e1720a3485570f4
```

Citations are written `<path>@bfab0de:<line>`; every one is reproducible with:

```bash
git -C <engine-checkout> fetch upstream
git -C <engine-checkout> show upstream/main:<path> | sed -n '<line>p'
```

The upstream files this document extracts rule by rule are:

| Upstream file | Role upstream | Here |
|---|---|---|
| `Sources/MLXFastTrustedHarness/EditableSurfaceByteBudget.swift` | launch-time byte budget (R1) | not run; the static review enforces the caps |
| `.github/scripts/run-submission-static-review.sh` | pre-dispatch deterministic caps (R2) + LLM bypass judge | deterministic half: `.github/scripts/submission-static-review-checks.sh`; judge not ported (§9) |
| `overlay` (overlay of editable paths, R3) | REPLACE overlay of the archive | the intake service (§4); `tools/router/router.py submit` implements the same rules for the organizer's own runs |
| `enforce` (modifiable-surface gate, R4) | refuses changes outside `editablePaths` | the intake service (§5); `tools/router/router.py` keeps only `editablePaths` |

`.github/scripts/hardened-git.sh` supports the static review.

**These files are not the whole upstream submission-restriction surface.**
Three further upstream gates matter against a hostile submission. None exists
as a script here; §9 says what covers each on this track:

| Upstream file | Role upstream | Here |
|---|---|---|
| `.github/scripts/verify-trusted-source-scope.sh` | re-verifies the trusted source scope before the trusted build; refuses any `editablePaths` entry overlapping it | the intake service reads `editablePaths` from the trusted base only; the linter refuses an editable entry over the scorer's pin (A1); participant code runs as a user that cannot write the checkout (`docs/sandbox-design.md`) |
| `.github/scripts/pin-trusted-harness.sh` | content pins over the built trusted and worker artifacts | benchd is fetched and sha256-verified by `tools/fetch-benchd.sh`; the staged engine shim is byte-checked by `tools/nemotron35-measure-and-score.sh`; participant builds cannot write trusted paths (`docs/sandbox-design.md`) |
| `.github/scripts/deny-private-artifacts.sh` | refuses artifact/cache uploads that could leak hidden data | the ranked workflow uploads only the sealed `score.json` after sandbox cleanup and uses no cache; the score is read from that artifact and the job log |

## 1. Layer map

An untrusted submission passes these gates, each fail-closed on its own:

```text
archive ──► intake overlay                 keeps and REPLACES only editablePaths,
            (the intake service)           read from the trusted contract; an
                                           absent OPTIONAL path keeps the trusted
                                           copy; refuses symlinks/special files
   │
   └──────► static-review checks           per-file / total / growth byte caps,
            (ranked workflow,              path validation, exempt handling
             surface-check job)
   │
   └──────► sandbox                        every participant build and process
            (ranked workflow)              runs as an unprivileged user
                                           (docs/sandbox-design.md)
```

The byte caps this repository enforces are `benchmark.json`
`editableSurfaceByteBudget` (participant contract §3.2); the upstream numbers
in the tables below are the reference's.

## 2. R1 — launch-time byte budget (upstream reference)

Source: `Sources/MLXFastTrustedHarness/EditableSurfaceByteBudget.swift@bfab0de:1-157`,
called from `Sources/MLXFastCLI/main.swift@bfab0de:2286` inside the runtime-worker
spawn preflight, implemented at `main.swift@bfab0de:2361-2397`. It does not run
on this track; the rules are listed because the static review carries the same
caps.

| # | Rule | Original |
|---|---|---|
| R1.1 | Total editable surface ≤ 3,000,000 bytes | `bfab0de:18` |
| R1.2 | Any single editable file ≤ 524,288 bytes | `bfab0de:19` |
| R1.3 | Contract is `benchmark.json` next to the cwd | `bfab0de:20`, `main.swift@bfab0de:2362-2364` |
| R1.4 | Absent contract = `.skipped`; official runs treat that as fatal, local runs proceed | `bfab0de:57-59`, `main.swift@bfab0de:2380-2386` |
| R1.5 | Contract that does not parse, or carries no non-empty `editablePaths`, = `.exceeded` (fail closed) | `bfab0de:60-68` |
| R1.6 | Walk is rooted at the contract's own directory | `bfab0de:75` |
| R1.7 | Upstream: only regular files count; symlinks and other non-regular entries are SKIPPED, not rejected. Here a non-regular entry fails closed (D8) | `bfab0de:37-39`, `bfab0de:118-124`, `bfab0de:138-144` |
| R1.8 | A missing editable path is skipped silently. Here a surface that walks to zero files fails closed (D8) | `bfab0de:115-117` |
| R1.9 | A directory that cannot be enumerated = `.exceeded` | `bfab0de:133-137` |
| R1.10 | `editableSurfaceByteBudget.exemptPaths` are held out of the CODE budget and charged to their own cap. This track declares no `exemptPaths`, so the exempt arm is unreachable here; the mechanism is retained in the static review and tested against a synthetic contract | `bfab0de:41-50`, `bfab0de:70-73`, `bfab0de:98-109` |
| R1.11 | Exempt AGGREGATE cap. Here `exemptPathMaxBytes` = 512,000,000: declared and drift-checked, but it cannot bind while `exemptPaths` is absent | `bfab0de:22-25` |
| R1.11a | Exempt PER-FILE cap (`exemptPathMaxFileBytes` = 100,000,000 here), checked before the aggregate so an oversize blob names itself. Same status as R1.11 | — |
| R1.12 | The per-file cap is checked BEFORE the running total, so an oversized file names itself | `bfab0de:81-86` |
| R1.13 | The total is checked after each file, so the message says "at least N bytes" | `bfab0de:89-94` |
| R1.14 | Official run: `.exceeded` refuses the worker spawn; local run: warn on stderr and continue | `main.swift@bfab0de:2387-2396` |
| R1.15 | Env overrides `MLXFAST_SUBMISSION_STATIC_REVIEW_MAX_BYTES` / `_MAX_FILE_BYTES` — not ported (§8 Q9) | `main.swift@bfab0de:2365-2372` |

Upstream, the exemption let a submission carry MTP head weights in-branch. On
this track no submission carries head weights: the head is embedded in the
pinned checkpoint, `mtp-head.manifest.json` accepts `"source": "pinned"` only,
and no head directory is an `editablePaths` entry (participant contract §4).
There is nothing to exempt. The declaration's `max_bytes`, `bytes` and
`sha256` are recorded, not enforced, and nothing loads from a declaration
(participant contract §4.1, §4.3).

## 3. R2 — static review, deterministic half

Source: `.github/scripts/run-submission-static-review.sh@bfab0de:1-412`. Lines
414-698 of that file are the LLM bypass judge and are NOT ported (§9).

| # | Rule | Original |
|---|---|---|
| R2.1 | Caps: total 3,000,000 / per-file 524,288 / growth 262,144, each overridable by env, each required to be a positive integer | `bfab0de:36-38`, `bfab0de:98-109` |
| R2.2 | The growth cap exists because the total cap alone lets a small diff hide a large table | `bfab0de:348-353` |
| R2.3 | Contract must exist and be non-empty | `bfab0de:94-97` |
| R2.4 | Track id set-but-empty is fatal; an unknown track id is fatal; reviewing under another track's policy inverts every verdict | `bfab0de:40-88` |
| R2.5 | Editable path validation: non-empty, not absolute, no `\`, no leading `:` (git pathspec magic), no `/../` or `/./` component | `bfab0de:136-149` |
| R2.6 | Exempt paths are validated as paths too — they also end up in a git pathspec | `bfab0de:277-281` |
| R2.7 | In diff mode the allowlist AND the exemptions are read from the BASE commit, never from the submission work tree | `bfab0de:249-260` |
| R2.8 | `MLXFAST_SUBMISSION_REVIEW_BASE_SHA` set-but-empty is fatal — never degrade to whole-surface review | `bfab0de:221-227` |
| R2.9 | Base and head must resolve to commits, and the work tree must BE the review head, or the collected bytes would not match the reviewed diff | `bfab0de:235-248` |
| R2.10 | An empty allowlist is fatal, never an accidental clean pass (a jq failure inside a process substitution is invisible to `set -e`) | `bfab0de:270-275` |
| R2.11 | Exempt paths keep editable status but never enter the payload, the diff, or the growth arithmetic | `bfab0de:283-293` |
| R2.12 | A surface where every entry is exempt is fatal | `bfab0de:294-297` |
| R2.13 | Zero changed paths = clean pass, recorded as such | `bfab0de:342-346` |
| R2.14 | Growth is computed from COMMITTED blob sizes on both sides so the work tree cannot skew it | `bfab0de:354-369` |
| R2.15 | Editable files deleted vs the base are reported as the stale-clone signature and the hint is appended to any later oversize refusal | `bfab0de:319-340` |
| R2.16 | Every changed non-deleted path must be a regular file (not a symlink) in the checkout | `bfab0de:371-378` |
| R2.17 | The unified diff counts against the same total cap | `bfab0de:380-395` |
| R2.18 | Whole-surface mode (no base): every reviewed path must EXIST, and selecting zero files is fatal. Upstream selects this mode by the ABSENCE of a base sha; here it is an explicit opt-in (D7). Here whole-surface also rejects a symlink or non-regular entry under a reviewed path (D8) | `bfab0de:396-412` |
| R2.19 | Every git read goes through `hardened-git.sh`, resolved next to the script (the trusted copy), never one inside the submission worktree | `bfab0de:5-14` |

`hardened-git.sh@bfab0de:1-46`: `env -i` with only `PATH` restored,
`GIT_CONFIG_GLOBAL`/`_SYSTEM` to `/dev/null`, `GIT_CONFIG_NOSYSTEM=1`,
`HOME=/var/empty`, and `-c core.fsmonitor=false -c core.hooksPath=/dev/null -c
core.pager=cat -c protocol.ext.allow=never -c safe.directory='*'`. The threat is
a planted repo-local `.git/config` running arbitrary programs as the invoking
uid.

**Implementation here:** `.github/scripts/submission-static-review-checks.sh`
and `.github/scripts/hardened-git.sh`. R2.1–R2.19 hold except as noted in §7
(D1, D5, D7, D8).

## 4. R3 — overlay, REPLACE semantics (intake service)

The intake service performs the overlay of a submission archive onto the
trusted base; there is no overlay script in this repository's ranked path.
`tools/router/router.py submit` implements the same rules for the organizer's
own load tests and re-scores, and is a working reference for them. R3.6 is the
rule the participant contract states (§3.1).

Source: `overlay@bfab0de:1-186` (the original).

| # | Rule | Original |
|---|---|---|
| R3.1 | `SUBMISSION_WORKTREE` required; missing worktree, missing trusted contract, or missing `jq` are all fatal | `bfab0de:5-20` |
| R3.2 | Editable path validation: non-empty, not absolute, no `\`, no `/../` or `/./` | `bfab0de:22-34` |
| R3.3 | The allowlist and the optional set come from the TRUSTED contract in the cwd, never from the submission worktree | `bfab0de:7`, `bfab0de:39`, `bfab0de:158-160`, `bfab0de:184` |
| R3.4 | REPLACE, not merge: `rm -rf` the target, then copy the submission's copy wholesale | `bfab0de:173-180` |
| R3.5 | A missing REQUIRED editable path is a loud refusal (it means a stale clone) | `bfab0de:165-166` |
| R3.6 | A missing OPTIONAL editable path is SKIPPED and the trusted copy is kept — absence means "use the trusted default", never "delete it" | `bfab0de:145-164` |
| R3.7 | Optional matching is exact string equality against `optionalEditablePaths` | `bfab0de:41-47` |
| R3.8 | Any symlink anywhere under a submitted editable path is a refusal, checked BEFORE the copy | `bfab0de:168-171` |
| R3.9 | After the copy the target tree must contain no symlink, no non-regular entry, no hardlinked file, no setuid/setgid file | `bfab0de:49-67`, `bfab0de:181` |
| R3.10 | Stale-editable-file detection is REPORT-ONLY and silently skipped without `TRUSTED_MAIN_SHA` or a resolvable merge base | `bfab0de:69-137` |

R3.6 is the rule the whole optional mechanism exists for (`bfab0de:148-160`):
archives have REPLACE semantics, the declaration is a small opt-in file most
archives will not carry, and failing the overlay would turn the default case
into a refusal. The optional set is read from the trusted contract so a
submission cannot make its own missing source files optional. On this track an
absent `mtp-head.manifest.json` keeps the trusted declaration, which runs
serial.

## 5. R4 — modifiable-surface gate (intake service)

What bounds the surface is the intake service: it archives and overlays only
`editablePaths`, so a file outside them never reaches the measured tree.
`tools/router/router.py` does the same for the organizer's own runs (it drops
and logs everything else). The rules below describe the original.

Source: `enforce@bfab0de:1-77` (the original).

| # | Rule | Original |
|---|---|---|
| R4.1 | `BASE_SHA` and `HEAD_SHA` required | `bfab0de:7-8` |
| R4.2 | The allowlist is read from the BASE commit so a PR cannot grant itself access | `bfab0de:2-3`, `bfab0de:34-47` |
| R4.3 | `CONTRACT_PATH` is deliberately the SAME variable the overlay and the review read, so the three gates cannot disagree about which contract binds | `bfab0de:10-16` |
| R4.4 | No fallback to `benchmark.json` on an unreadable contract — judging a submission against another track's surface would reject it naming a contract it never used | `bfab0de:38-43` |
| R4.5 | A contract with no, or an empty, `editablePaths` is fatal | `bfab0de:48-57` |
| R4.6 | A changed file matches the surface by exact equality OR by allowed-directory prefix | `bfab0de:61-76` |
| R4.7 | Every git read goes through `hardened-git.sh` from the trusted checkout | `bfab0de:18-27` |

## 6. R5 — sandbox profile (not applicable)

Upstream generates a macOS Seatbelt profile for the runtime worker in two
places (`benchmark.sh@bfab0de:1266-1307` and
`Sources/MLXFastCLI/main.swift@bfab0de:2605-2652`), and only the Swift one
denies the mDNS resolver, which leaves a DNS exfiltration channel in the bash
one. This track runs on Linux and does not use either. Its sandbox is the
unprivileged `nemosub` user (`docs/sandbox-design.md`), whose egress block
covers loopback and whose access to the host resolver sockets is denied, so
the DNS channel that divergence was about is closed.

## 7. Divergences in this repository

Each is deliberate.

**D1 — the caps come from the manifest, not from constants.**
`benchmark.json` `editableSurfaceByteBudget` carries machine-readable
`maxTotalBytes` / `maxFileBytes` / `maxGrowthBytes` alongside the exempt caps.
The static review reads them; its shell defaults survive only as the fallback
for a contract that declares none, and `tools/lint-benchmark-manifest.py`
check 3b asserts they still equal the manifest. Upstream, the numbers lived in
a Swift constant (`bfab0de:18-19`) and a shell default (`review@bfab0de:36-38`)
kept in sync by a comment (`review@bfab0de:32-35`).

**D2 — a present-but-malformed cap fails closed.** Upstream coerced a
malformed `exemptPathMaxBytes` to the default (`bfab0de:72-73`). Here the
static review type-gates on the JSON type instead of `jq -r ... // empty`
(which read `false` and `null` as absent, and stringified `"2000000"` into a
value that passed the integer regex), and rejects a non-positive value.

**D3 — `maxFileBytes > maxTotalBytes` is rejected.** A per-file cap that can
never bind is a configuration error that reads as a working defence.

**D5 — the static-review track default is the contract's.** Upstream defaults
`TRACK_ID` to `serial` when unset (`review@bfab0de:51`), which would review a
legal MTP submission under a speculation ban. Here the default is the
manifest's `staticReviewTrackId`, and any explicitly requested track that is
not the one the contract declares is rejected.

**D6 — the LLM judge half is not ported.** See §9.

**D7 — whole-surface mode is an explicit opt-in, not the fallback.** Upstream
selects the review mode by the ABSENCE of `MLXFAST_SUBMISSION_REVIEW_BASE_SHA`.
Whole-surface mode reads the contract from the WORK TREE, which is only safe in
the trusted checkout after the overlay; as a fallback, a checkout carrying a
contract with huge caps and no base sha would review itself against its own
caps. Here the mode must be named:

```text
BASE_SHA set-but-empty                    -> fatal (base computation failed)
BASE_SHA non-empty, MODE unset or "diff"  -> diff mode
BASE_SHA non-empty, MODE anything else    -> fatal (contradictory request)
BASE_SHA unset, MODE "whole-surface"      -> whole-surface mode
BASE_SHA unset, MODE anything else        -> fatal (no mode selected)
```

**A1 (addition) — the scorer pin is untouchable by construction.**
`tools/lint-benchmark-manifest.py` refuses any editable entry equal to, or
inside, `benchd`, `.gitmodules`, `benchd.pin` or `benchd-bin`
(`FORBIDDEN_EDITABLE`), case-folded and checked by filesystem identity, so an
entry spelled in another case is refused too. An editable entry over the
scorer's pin would let a submission repoint its own scorer.

**A2 (addition) — one path-validation rule, not two.** Upstream's overlay
validator (`overlay@bfab0de:22-34`) omits the leading-`:` test that its static
review (`review@bfab0de:136-149`) carries. This repository applies the stricter
rule in the static review and the linter; the intake service should apply it
too.

**D8 — fail-closed hardening.** Upstream behaviours that read as clean passes
are refusals here: a non-regular entry (symlink, FIFO, socket, device) at or
under an editable path is rejected rather than skipped, in diff and
whole-surface mode alike (diverges from R1.7, extends R2.16 to R2.18); a
surface that walks to zero files fails closed (diverges from R1.8); and an
exempt-cap overflow names the aggregate, not a single path. None widens the
surface — each turns a silent pass into a refusal.

## 8. Ambiguities in the original, and how they are resolved here

**Q1 — two definitions of a legal editable path.** Resolved by A2: the stricter
rule applies.

**Q2 — the byte budget SKIPS symlinks rather than rejecting them.** Resolved:
reject (D8). Skipping is only safe where a separate overlay rejects them; a
whole-surface review is blind otherwise, since `find -type f` skips a link.

**Q3 — a missing editable path is treated differently by two gates.**
Resolved: a surface with zero files fails closed (D8), matching whole-surface
review's `file_count == 0` guard.

**Q4 — the exempt cap is an aggregate charged with a per-path message.**
Resolved: the message names the aggregate and the number of exempt paths.

**Q5 — the growth cap has no launch-time counterpart.** `maxGrowthBytes`
exists only in the static review, because a launch has no base commit. On this
track every ranked job passes the static review first (`needs: surface-check`),
so there is no dispatch path that skips it.

**Q6 — the setuid check runs on the copy, after the copy stripped it.**
Upstream's `validate_overlay_tree` (`bfab0de:63-66`) inspects the target, but an
unprivileged `tar`/`cp` already drops setuid/setgid. The guarantee that holds is
"the bit never lands". The intake service should refuse setuid/setgid in the
source archive outright.

**Q7 — the byte caps' authority.** Resolved by D1: the manifest wins
(participant contract §1, §3.2).

**Q8 — the overlay reports success after overlaying NOTHING.** Upstream read
`.editablePaths[]` in the loop header, so an empty or unparseable allowlist
produced zero iterations and a success trailer. The intake service must fail
closed on an empty or unparseable allowlist; `tools/router/router.py` does.

**Q9 — may an environment variable re-state a run's caps?** Resolved: no. The
static review reads none of `MLXFAST_SUBMISSION_STATIC_REVIEW_MAX_BYTES`,
`_MAX_FILE_BYTES` or `_MAX_GROWTH_BYTES`, and resolves each cap as
`manifest > constant`. A cap that can be re-stated out of band is not a single
source, and the reviewed run could differ from the contract with nothing in
the record saying so.

## 9. Responsibilities of the intake service, and what is not ported

**The intake service** (the dispatcher that accepts an archive and starts the
ranked workflow) is responsible for:

- **The overlay (R3) and the surface bound (R4).** Keep only the trusted base's
  `editablePaths`, REPLACE each one present, keep the trusted copy of an absent
  optional path, refuse symlinks, special files, hardlinks and setuid/setgid,
  and fail closed on an empty or unparseable allowlist. Read `editablePaths`
  from the trusted base, never from the archive: this is also what stands in
  for upstream's trusted-source-scope check (`verify-trusted-source-scope.sh`).
- **Archive validation.** Validate all members before the first write. Refuse
  hardlinks, symlinks, special files, duplicate names, absolute paths, parent
  paths, NUL paths, sparse members, and paths outside `editablePaths`. Refuse
  each negative or non-integer member size before byte accounting. Read the
  byte caps from the trusted `benchmark.json`. The router permits at most
  10,000 archive members. After
  standard decompression, it limits expanded bytes to
  `maxTotalBytes + max(64 KiB, maxTotalBytes / 4)`. After validation, create
  approved directories and write approved regular files. A refusal leaves
  the archive staging directory empty.
- **Dispatching only on the trusted base.** The workflow, tools and fixture
  come from the base commit; a submission changes only its editable paths.
- **Returning only the score.** Job logs are never shown to participants
  (`docs/sandbox-design.md` §5 item 6).

**Covered on the box, not by a separate script:**

- **Trusted-binary pins** (upstream `pin-trusted-harness.sh`). benchd is
  resolved offline from the staged pair and verified by
  `tools/fetch-benchd.sh`; the staged engine shim is byte-checked by
  `tools/nemotron35-measure-and-score.sh`; participant builds run as `nemosub`
  and cannot write any trusted path.
- **Artifact exfiltration** (upstream `deny-private-artifacts.sh`). The ranked
  workflow uploads exactly one artifact, the sealed `score.json` after sandbox
  cleanup, and uses no cache. `tools/ci-workflow-egress-scan.sh` fails any
  other upload.

**Not ported:**

- **The LLM bypass judge** — `review@bfab0de:414-698`: the track-selected
  decode rules, the system prompt, the model call, retry handling, and the
  verdict/severity reconciliation. It is policy, it needs a credential and a
  network, and it cannot run on an offline box. The deterministic half fails
  closed on its own. If a judge is added, `bfab0de:684-694` is the rule that
  matters most: a schema-valid but self-contradictory verdict — `passed=true`
  with `severity` `high` or `critical`, the shape a partially successful prompt
  injection produces — is forced to FAILED.
- **The launch-time byte budget** (R1). The static review runs before every
  ranked job, so the caps are enforced before anything reaches the GPU leg.

## 10. Reading the score from the job log (intake service)

Participant code can print arbitrary text into the ranked job log (`cargo:warning=`
lines, build output, resident boot diagnostics). The ranked workflow therefore
does not treat benchd's own log lines or a bare JSON blob as the score channel.

**Reveal on the box.** Before sandbox start, the workflow draws 32 bytes from
`/dev/urandom` as 64 lowercase hex into runner-private storage (`umask 077`);
the nonce is not printed and does not appear in `GITHUB_ENV` or
`GITHUB_OUTPUT`. **Sandbox cleanup** (with `if: always()`) kills every
`nemosub` process, wipes participant-writable temp state, and removes the
per-job sandbox directory; only when every check in that step passes does it
write `${RUNNER_TEMP}/cleanup-ok` (`umask 077`) as its last action. After
cleanup, **Reveal the sealed score** prints, in order:

1. `NEMOTRON35-SCORE-BEGIN <nonce>`
2. when `cleanup-ok` exists: the exact bytes of `score.json` if the file exists
   (the step may append one trailing newline when the file has no final
   newline), then `NEMOTRON35-SCORE-END <nonce> sha256=<sha256 of score.json>
   bytes=<n>`, or `NEMOTRON35-SCORE-END <nonce> none` when `score.json` is
   absent
3. when `cleanup-ok` is missing (cleanup did not fully succeed): only
   `NEMOTRON35-SCORE-END <nonce> none` after the `BEGIN` line — even if
   `score.json` is still on disk

The nonce is unknown until this step, so earlier log lines cannot forge a valid
block. Surviving sandbox processes after cleanup therefore yield an explicit
empty reveal (`{}`), not a passing score.

**Parsing (the intake service MUST implement this).** Strip the GitHub Actions
timestamp prefix from each log line (everything after the first space on the
line). Then:

1. Find the **last** line matching
   `^NEMOTRON35-SCORE-END ([0-9a-f]{64}) (sha256=([0-9a-f]{64}) bytes=(\d+)|none)$`.
   If none, there is no sealed score (`None`).
2. Let `<nonce>` be the first capture group. If the line ends with `none`, the
   reveal explicitly sealed no file: treat as an empty score (`{}`), not a
   parse failure. If there is no matching `END` line at all, parsing yields
   `None` (no valid block — forged or truncated text must not count).
3. Otherwise take every line **strictly between** the last
   `NEMOTRON35-SCORE-BEGIN <nonce>` before that end line and the end line;
   join with `\n`. Require a JSON object with `passed` and `metrics`. Require
   its sha256 and byte length to match the end line, allowing for at most one
   trailing-newline difference either way (the reveal step may add a newline
   when the file lacks one; a log line break may omit a file-final newline).
   Any mismatch → no sealed score (`None`). Ignore earlier `BEGIN`/`END` pairs,
   benchd marker lines, and other JSON printed elsewhere in the log.

**Outcome when nothing valid is sealed.** Classify using the ranked job's step
conclusions and the parse result:

- A **passing** score from the log counts only if the step whose name starts
  with **GPU-exclusive measurement** concluded **success**, same as
  `tools/router/router.py` `finish()` — a syntactically valid block cannot
  upgrade a failed or skipped measurement step.
- **`{}`** (last `END` line is `none`): explicit empty reveal; after Setup ran,
  **`unsealed`** (team failure).
- **`None`** (no valid `BEGIN`/`END` block): no sealed score; after Setup ran,
  **`unsealed`** (team failure).
- **`watchdog_timeout`**: the router cancelled the run after the watchdog;
  incident starts with `watchdog:` — team failure (same expectation bucket as
  `unsealed` for load tests).
- Failures before **Setup** runs: **`surface_refused`**, **`preflight_refused`**
  (and related refused outcomes), not `unsealed`.
- **`cancelled`** without a sealed score: team failure. Do not re-dispatch it.

The reference parser in this repository is `tools/router/router.py`
`score_from_log` (organizer tooling, outside participant `editablePaths`).

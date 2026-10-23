# Reproduce the track's results

Organizer document. Each section gives the commands that re-create one input
of a ranked score, and the recorded result to compare against. The values are
for dev `main` after PR #37 (fixture unchanged since `c1ef8de`), recorded
2026-10-19/20.

| Input | Section | Recorded result |
|---|---|---|
| benchd binaries | [1](#1-rebuild-benchd-from-source) | dist `4d7aa43` is the published build of source `b0a79f2`; an independent no-cache rebuild is not yet recorded (the previous pin's rebuild matched) |
| Hidden goldens | [2](#2-re-record-the-hidden-goldens) | 21 files, byte-identical on two Sparks, all pinned in the fixture |
| Box calibration | [3](#3-recalibrate-a-box) | 4 Sparks, every CV under 1% (limit 2%) |
| Ranked score | [4](#4-reproduce-a-ranked-score) | dev run: `passed: true`, score 1.310 |

## 1. Rebuild benchd from source

benchd is pinned in `tools/fetch-benchd.sh`:

| Pin | Value |
|---|---|
| Repository | `Layr-Labs/benchd` (public dist channel and public source snapshot `45bfca2`, the same source tree as development commit `b0a79f2`) |
| `PINNED_COMMIT` (dist) | `4d7aa439ec82b280c62233944eaa57695c59fda7` |
| `PINNED_SOURCE_COMMIT` | `b0a79f281821b46b86f0049e3fbf232546898626` |
| `PINNED_PUBLIC_SOURCE_COMMIT` | `45bfca2cd2ab1bcf572cfd9133e900f7dc7def11` |
| `dist/linux-aarch64/benchd` | sha256 `af4ab9537469842392232a81b3482d37069884b81b56b2a07191cf7aadfb2379`, 4,062,608 bytes |
| `dist/linux-aarch64/record-correctness-golden` | sha256 `bcc2582bd2cc240602e7200f76e0605ef0d81147cb5ad156ad513d261d365408`, 1,716,472 bytes |
| `dist/linux-x86_64/benchd` | sha256 `23ababbd7d3e55b356d261d84f4a6c4d571d6c182adea6f944b0ee7c364e59c8`, 4,275,992 bytes |
| `dist/linux-x86_64/record-correctness-golden` | sha256 `9bfd8160c6d44a6efe7388b2fad193029e0c1e5bdd47247bbabd84ea74347a9b`, 1,749,544 bytes |

The bench repository builds the Linux binaries in a pinned container
(`deploy/Dockerfile.dist-linux`), which fetches the source commit by sha from
`BENCHD_DIST_REPO`. To rebuild with no layer cache and no token, from a clone of
the public `Layr-Labs/benchd` repository at the dist commit, on a host with
Docker:

```bash
git checkout 4d7aa439ec82b280c62233944eaa57695c59fda7
BENCHD_DIST_NO_CACHE=1 BENCHD_DIST_BRANCH=main \
  BENCHD_DIST_REPO=https://github.com/Layr-Labs/benchd.git \
  ./scripts/build-dist-linux.sh 45bfca2cd2ab1bcf572cfd9133e900f7dc7def11
git diff --stat -- dist/linux-aarch64   # no output: the rebuilt bytes equal the published ones
```

Public output carries fixed markers, never engine-chosen values (#433).
Adapter stdout lines are capped and the channel bounded (#431).

The script prints each binary's sha256 and bytes. Compare them with the table.

**Recorded result:** for the previous pin (2026-10-20, an x86_64 Linux host,
Docker, no layer cache) both binaries matched its table byte for byte. For
this pin (2026-10-23) the table is the published dist, built with this same
pinned Docker build; an independent no-cache rebuild of it is not yet recorded.

## 2. Re-record the hidden goldens

The live pool is `pool-9`, `pool-10` and `pool-11`: three coding prompts, each
one user message rendered through the snapshot's pinned chat template with
thinking off, exactly 496 tokens. Each prompt has a serial golden and six
per-depth oracles (MTP 1 to 6), 21 files in all. The fixture pins every file
by sha256 and bytes: `timed_prompt_pool[]` pins the three serial goldens,
`speculative_oracles` pins the 18 per-depth oracles, and
`hidden_correctness_golden` repeats the `pool-9` serial pin.

**Where the inputs live.** The prompt texts and token files are organizer
material and are never committed here, because sandboxed participant code can
read the ranked checkout. The organizer archive (a private repository; access
by the organizer) holds the texts, the token files, the goldens and
`SHA256SUMS` (see [pool refresh, "Where the goldens live"](pool-refresh-and-final-rescore.md#where-the-goldens-live)).
The input hashes:

| File | sha256 |
|---|---|
| `pool-9.txt` | `037a676d462577477562277a677d7f7fea16736faefd030f93951499e705d0d9` |
| `pool-10.txt` | `aedb5f3d561e58829dc6df56bfa4c74bc31fa85e508b26c5dfe29ba2aae4567a` |
| `pool-11.txt` | `b65a9862eb5cebb42cdb1522a07b5da0c0ff7862fbe5517bb16d2368113e681e` |
| `pool-9.tokens.json` | `36f5d8256728c4ced104eccb5aa73f228f8c0a780a9b83f0444f0caae2e2448a` |
| `pool-10.tokens.json` | `a4e2b5b55d26fb61d78bee1eff0f3dfbf2e6ecfb5bf078fa355d35e59fe72141` |
| `pool-11.tokens.json` | `daeea88e15b9de0a6bcf1ca2e92648278b3e65d074fa3f57b82512b30cf43b2f` |

**Steps**, on two Sparks, each with this repository checked out, the snapshot
staged, the pinned benchd pair in `benchd-bin/`, and the engine built
(`./setup.sh`, which also stages the vLLM overlay). Use absolute paths:

```bash
export MLXFAST_TARGET_SNAPSHOT_DIR=<snapshot> BENCHD_BIN_DIR="$PWD/benchd-bin"
export SERVE_UP_SOCKET_DIR="$HOME/rsock"; mkdir -p "$SERVE_UP_SOCKET_DIR"   # serve-up.sh refuses /tmp
TOKENS="$HOME/rec/tokens"; OUT="$HOME/rec/goldens"; mkdir -p "$TOKENS" "$OUT"
```

1. Tokenize. The pinned image carries `tokenizers` and `jinja2`; the snapshot
   must be mounted at the same path:

   ```bash
   S=$MLXFAST_TARGET_SNAPSHOT_DIR
   docker run --rm --network none -v "$HOME:$HOME" -v "$S:$S:ro" -w "$PWD" \
     --entrypoint python3 docker.io/vllm/vllm-openai@sha256:1c8e60a0841b333c700488cb029d3664807249da0c071e862191b00fe34b228c \
     tools/author-prompts.py --snapshot "$S" --out "$TOKENS" --chat \
       pool-9=pool-9.txt pool-10=pool-10.txt pool-11=pool-11.txt
   sha256sum "$TOKENS"/*.tokens.json   # must equal the table
   ```

   `author-prompts.py --chat` refuses a rendering that is not exactly
   `seed_tokens` (496) ids; it never trims a chat prompt.

2. Record, under the box's GPU lock:

   ```bash
   flock /tmp/mtplx-gpu-exclusive.lock \
     tools/record-goldens.sh --prompts "$TOKENS" --out "$OUT" \
       --only pool-9 --only pool-10 --only pool-11
   ```

   One serial resident records the three serial goldens; then one resident per
   depth records that depth's oracles, with the serial `cases[]` spliced in.
   Every file is re-validated with `benchd validate-golden --contract`.

3. Compare. `(cd "$OUT" && sha256sum *.json)` on both boxes must give the same
   21 lines, and each sha256 and byte count must equal the fixture's pin.

**Recorded result (2026-10-19):** recorded on Spark box 3 and Spark box 4 with the
pinned image and snapshot, and with the benchd pair staged before PR #35 (so
not yet `b65af98`); the 21 files were byte-identical on both boxes and are the
fixture's pins (PR #33). The pinned benchd `b65af98` validates and scores
against them (section 4). The staged copies on Spark box 1 to Spark box 4 and the
jump-box copy were re-checked on 2026-10-20: the same 21 files on all five.

**Re-recorded with the pinned benchd (2026-10-20, 17:01-17:34 UTC):** the same
token files, recorded again on Spark box 4 and Spark box 2 at `c1ef8de` with benchd
`b65af98`, gave 21 files that are byte-identical on both boxes and equal to
the fixture's pins.

## 3. Recalibrate a box

A calibration file is the box's health band for the serial-control leg.

1. Stage the reference tree at the fixture's `baseline_reference_commit`
   (`f68318514e80b910597b1e00146ed64ddbbe2ec0`), from a local bundle or mirror
   (the ranked job holds no credential):

   ```bash
   REF="$HOME/nemotron35/reference"     # absolute path
   tools/stage-baseline-workspace.sh "$REF" --source <bundle-or-mirror>
   ```

2. Calibrate, with the box's runner name (the ranked job passes it as
   `--box`; a band under another name is refused):

   ```bash
   export MLXFAST_TARGET_SNAPSHOT_DIR=<snapshot> \
          MLXFAST_NEMOTRON35_GOLDEN_DIR=<staged goldens> \
          MLXFAST_BASELINE_WORKSPACE="$REF" BENCHD_BIN_DIR=<benchd pair> \
          SERVE_UP_SOCKET_DIR="$HOME/rsock"
   NEW="$HOME/nemotron35/calibration.new.json"   # absolute path
   tools/calibrate-box.sh <runner name> "$NEW" --passes 3
   # for a pool that is not yet live (refresh, final re-score), add
   # --fixture <proposed fixture> so the bands are for the new prompts
   ```

   `calibrate-box.sh` takes the GPU lock itself; do not wrap it in `flock` of
   the same lock.

3. Check the file: `version` is 3, `reference_commit` and
   `benchd_source_commit` are the pins, there is one `prompts[]` entry per live
   golden, and every `prefill_cv` and `decode_cv` is at most
   `calibration_max_cv_percent` / 100 (0.02).
4. Install it at the path the runner's `MLXFAST_BASELINE_CALIBRATION` names,
   keep the old file under a dated name, and run
   `tools/ranked-box-preflight.sh` with the runner environment.

Recalibrate after any change to the live pool, the reference commit or benchd.

**Recorded result:**

| Box | Runner name | Captured (UTC) | File sha256 | CV % prefill / decode: pool-9, pool-10, pool-11 |
|---|---|---|---|---|
| Spark box 1 | (organizer runner) | 2026-10-20 15:21 | `6fd0d46f…4383b3` | 0.32/0.27, 0.80/0.69, 0.27/0.25 |
| Spark box 2 | (box label) | 2026-10-20 15:21 | `ea5e5da9…79a2c25` | 0.57/0.91, 0.14/0.53, 0.62/0.19 |
| Spark box 3 | (organizer runner) | 2026-10-19 22:07 | `c9a37374…6657371` | 0.65/0.36, 0.47/0.23, 0.36/0.36 |
| Spark box 4 | (organizer runner) | 2026-10-19 22:08 | `f9c0374f…fb77b51` | 0.64/0.08, 0.56/0.19, 0.43/0.43 |

All four: version 3, reference `5be35aa`, benchd source `b65af98`.

## 4. Reproduce a ranked score

Never re-type the benchd command: the workflow adds the GPU lock, the sandbox,
the cleanup and the nonce-bound reveal.

1. Dispatch `.github/workflows/benchmark.yml` on a fixed ref and record the
   run id and its `headSha`:

   ```bash
   gh workflow run benchmark.yml --repo <repo> --ref <ref>
   gh run list --repo <repo> --workflow benchmark.yml --branch <ref> --commit <sha> \
     --event workflow_dispatch --limit 1 --json databaseId,headSha,createdAt
   # check that createdAt is your dispatch time and headSha is <sha>
   ```

2. Wait for the run. Both jobs, the measurement step and **Sandbox cleanup**
   must succeed.
3. Read the score from the reveal block, which binds a per-job nonce to the
   file's sha256 and length:

   ```bash
   gh run view <run_id> --repo <repo> --log | cut -f3- > job.log
   python3 -c 'import sys, json; sys.path.insert(0, "tools/router"); import router; \
     print(json.dumps(router.score_from_log(open("job.log").read()), indent=1))'
   ```

   `cut -f3-` drops the job and step columns, leaving the timestamped lines
   the parser expects. `score_from_log` returns `None` when the block is
   missing or its bytes do not match the end line.
4. Download the artifact and check that its `score.json` has the sha256 and
   byte count of the reveal block's end line:

   ```bash
   gh run download <run_id> --repo <repo> -n benchmark-results-<run_id> -D art
   sha256sum art/score.json; wc -c < art/score.json
   grep -o 'NEMOTRON35-SCORE-END .*' job.log | tail -1
   ```

   For the dev repository run above both give sha256
   `533217192a497b4bf92d41f286c79054cab5c754f81f6038a08d6d30e560efa6` and
   8,018 bytes, and the parser returns `passed: true`, score
   1.3098707780322463.

**Recorded results** (the unchanged engine of dev `main`, which declares MTP
depth 3):

| Run | Box | Result |
|---|---|---|
| dev repository run (`c1ef8de`) | Spark box 2 | every step succeeded; `passed: true`, score 1.3099; reveal sha256 `53321719…`, 8,018 bytes; artifact uploaded |
| organizer test repository run (same tree) | Spark box 1 | `passed: true`, score 1.3095 (only the artifact upload failed, on that account's storage quota) |

A participant submission through Yukon dev (2026-10-21, `yukon submit` to
the dev row, MTP depth 2) was built by Yukon on `main` `c39f90c` and scored
`passed: true`, 1.0970 in a dev repository run;
its artifact equals the reveal line (sha256 `d858782f…`, 8,013 bytes).

The two boxes agree to 0.03%. Depths 1 to 6 on the same tree scored 1.164,
1.114, 1.316, 1.189, 1.131 and 1.104, all passing correctness, and an engine
that returns wrong tokens was refused with no score (organizer test repo,
2026-10-19/20).

**Why 1.31 and not 1.44.** The same unchanged engine scored 1.44 on dev
`main` before 2026-10-19 (four runs, 1.4376 to 1.4427). The score is measured
on the live pool, and PR #33 replaced the three prompts behind `pool-9`,
`pool-10` and `pool-11` with the organizer's own coding prompts (same names,
same 496-token length, new goldens). The decode gain of MTP depends on how well the head drafts the text, so
a new pool moves the reference point; the band and the floors are unchanged.

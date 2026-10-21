#!/usr/bin/env python3
"""router.py -- the submission router for the Nemotron 3.5 Lightning track.

ORGANIZER TOOL, trusted side (outside editablePaths). It does what Yukon's
service does for a ranked submission (docs/submission-restriction-spec.md
sections 1 and 4), then drives GitHub Actions the way Yukon does:

  submit   take a directory or tarball holding a participant's editable paths,
           keep ONLY the base's editablePaths (everything else is dropped and
           logged), overlay them onto the trusted base with REPLACE semantics
           (an editable path absent from the archive keeps the trusted copy,
           R3.6), refuse symlinks and special files (R3.8/R3.9), commit, push
           `sub/<id>`, and enqueue.
  run      the dispatch daemon: keep one dispatched-but-not-started run per
           online runner (GitHub picks the box), resolve each run, watch it,
           cancel a ranked job that runs past the watchdog, download the sealed
           score.json, and record the outcome.
  status   one-line counts.
  report   JSON + a static HTML summary of every submission.
  freeze   stop accepting submissions (the contest deadline); `unfreeze` undoes it.
  rescore  the final re-score (docs/pool-refresh-and-final-rescore.md): each
           team's best submission by mean score, top N, overlaid onto a
           `rescore/base` whose live_goldens are the held-out `final`
           generation, queued K times for the `run` daemon; `rescore-report`
           ranks them by mean score once they are done.

State lives in SQLite (--state-dir, default ~/nemotron35-router), so a restart
resumes where it stopped. The `gh` CLI must be logged in with repo scope.

    tools/router/router.py submit <dir|tar.gz> --variant null --expect pass
    tools/router/router.py run
    tools/router/router.py status
    tools/router/router.py report
    tools/router/router.py freeze
    tools/router/router.py rescore --base-ref origin/main [--top 10 --runs 5]
    tools/router/router.py rescore-report [--wait]
    tools/router/router.py rescore-retry [--tag final] [--max-infra-retries 2]
           # per slot: 1 + max-infra-retries infrastructure attempts (default 3)
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import html
import json
import re
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

DEFAULT_BASE = os.environ.get("ROUTER_BASE", "loadtest/base")
WORKFLOW = "benchmark.yml"
RANKED_JOB_PREFIX = "ranked measurement"
ROOT = Path(__file__).resolve().parent.parent.parent


def default_repo() -> str:
    """owner/repo to drive: ROUTER_REPO, else what `gh` resolves for this checkout."""
    repo = os.environ.get("ROUTER_REPO")
    if repo:
        return repo
    return subprocess.run(["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
                          cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
FIXTURE_REL = "fixtures/nemotron35_lightning_30b_a3b_track.json"
TRACK_LABEL = "nemotron3.5-lightning-30b-a3b-cuda-v1"
WATCHDOG_SECONDS = 90 * 60
POLL_SECONDS = 60

SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
  id TEXT PRIMARY KEY, variant TEXT, expect TEXT, note TEXT,
  created REAL, branch TEXT, commit_sha TEXT, dropped TEXT,
  state TEXT, run_id INTEGER, dispatched_at REAL, started_at REAL,
  finished_at REAL, conclusion TEXT, box TEXT, passed INTEGER, score REAL,
  error TEXT, outcome TEXT, incident TEXT, result TEXT
);
CREATE INDEX IF NOT EXISTS by_state ON submissions(state);
"""
# Columns added after the first schema: (name, SQL type, backfill expression).
MIGRATIONS = [
    ("team", "TEXT", "variant"),   # a team's submissions are ranked together; old rows: the label
    ("rescore", "TEXT", None),     # NULL for a contest submission, the rescore tag for a re-score run
    ("source", "TEXT", None),      # a re-score run: the contest submission it re-measures
    ("rescore_slot", "INTEGER", None),  # 0-based slot index within a team's K rescore runs
]
# States: queued -> dispatched -> running -> done. A dispatch that cannot be
# resolved to a run becomes `dispatch_failed`.
OPEN_STATES = ("dispatched", "running")


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} router: {msg}", flush=True)


# This box's link to GitHub resets now and then; a network call that fails with
# one of these (and none of the deterministic markers) is worth another try.
TRANSIENT = re.compile(r"connection reset|broken pipe|timed? ?out|expected flush|could not read from remote"
                       r"|\bEOF\b|early eof|hung up|HTTP 5\d\d|\b50[234]\b|connection refused"
                       r"|could not resolve host|temporary failure", re.I)
DETERMINISTIC = re.compile(r"permission denied|authentication|\b40[134]\b|not found|could not find"
                           r"|conflict|rejected|non-fast-forward|invalid|validation", re.I)
RETRY_DELAYS = (2, 5, 10)  # 4 attempts


def transient(err: str) -> bool:
    return bool(TRANSIENT.search(err)) and not DETERMINISTIC.search(err)


def sh(args: list[str], cwd: str | None = None, check: bool = True, retry: bool = False, **kw) -> str:
    """Run a command. `retry=True` (network calls only) retries a transient failure."""
    for delay in (*RETRY_DELAYS, None) if retry else (None,):
        r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, **kw)
        if r.returncode == 0 or delay is None or not transient(r.stderr):
            break
        log(f"{' '.join(args[:3])}: transient failure, retry in {delay}s: {r.stderr.strip()[-200:]}")
        time.sleep(delay)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:4])}... failed ({r.returncode}): {r.stderr.strip()[-400:]}")
    return r.stdout


class Router:
    def __init__(self, state_dir: Path, repo: str, base: str, gh: str = "gh"):
        self.state_dir = state_dir
        self.repo = repo
        self.base = base
        self.gh = gh
        state_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(state_dir / "state.db")
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        have = {r[1] for r in self.db.execute("PRAGMA table_info(submissions)")}
        for col, typ, backfill in MIGRATIONS:
            if col not in have:
                self.db.execute(f"ALTER TABLE submissions ADD COLUMN {col} {typ}")
                if backfill:
                    self.db.execute(f"UPDATE submissions SET {col} = {backfill}")
        self.db.commit()
        self.frozen_flag = state_dir / "FROZEN"
        self.clone = state_dir / "clone"
        self.remote = os.environ.get("ROUTER_REMOTE", f"git@github.com:{repo}.git")

    # ---- git side -----------------------------------------------------------------
    def ensure_clone(self) -> None:
        if not (self.clone / ".git").exists():
            sh(["git", "clone", "-q", self.remote, str(self.clone)], retry=True)
        sh(["git", "fetch", "-q", "origin", self.base], cwd=str(self.clone), retry=True)

    def editable_paths(self, ref: str | None = None) -> list[str]:
        raw = sh(["git", "show", f"{ref or 'origin/' + self.base}:benchmark.json"], cwd=str(self.clone))
        paths = json.loads(raw).get("editablePaths") or []
        if not paths:
            raise RuntimeError("the base benchmark.json declares no editablePaths; refusing")
        for p in paths:  # R3.2
            if not p or p.startswith("/") or "\\" in p or "/../" in f"/{p}/" or "/./" in f"/{p}/":
                raise RuntimeError(f"invalid editable path in the base contract: {p!r}")
        return paths

    @staticmethod
    def _check_tree(root: Path) -> None:
        """R3.8/R3.9: no symlink, no special file, no hardlink, no setuid/setgid."""
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            for name in dirnames + filenames:
                p = Path(dirpath) / name
                st = p.lstat()
                if stat.S_ISLNK(st.st_mode):
                    raise RuntimeError(f"symlink in the submission: {p.relative_to(root)}")
                if stat.S_ISDIR(st.st_mode):
                    continue
                if not stat.S_ISREG(st.st_mode):
                    raise RuntimeError(f"non-regular entry in the submission: {p.relative_to(root)}")
                if st.st_nlink > 1:
                    raise RuntimeError(f"hardlinked file in the submission: {p.relative_to(root)}")
                if st.st_mode & (stat.S_ISUID | stat.S_ISGID):
                    raise RuntimeError(f"setuid/setgid file in the submission: {p.relative_to(root)}")

    def submit(self, source: Path, variant: str, expect: str, note: str = "", team: str | None = None) -> str:
        if self.frozen_flag.exists():
            raise RuntimeError("the router is frozen (contest closed); no new submissions")
        self.ensure_clone()
        return self._overlay(source, f"origin/{self.base}", variant, expect, note, team or variant)

    def _overlay(self, source: Path, base_ref: str, variant: str, expect: str, note: str, team: str,
                 rescore: str | None = None, source_id: str | None = None, rescore_slot: int | None = None) -> str:
        editable = self.editable_paths(base_ref)
        sid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        with tempfile.TemporaryDirectory(prefix="router-sub-") as tmp:
            tmp = Path(tmp)
            src = tmp / "archive"
            if source.is_dir():
                shutil.copytree(source, src, symlinks=True)
            else:
                src.mkdir()
                with tarfile.open(source) as tf:
                    tf.extractall(src, filter="tar")
            present = {p.name for p in src.iterdir()}
            # Keep only editablePaths (top-level entry match, as the contract lists them).
            dropped = sorted(n for n in present if n not in editable)
            for n in dropped:
                target = src / n
                shutil.rmtree(target) if target.is_dir() and not target.is_symlink() else target.unlink()
            self._check_tree(src)
            wt = tmp / "worktree"
            sh(["git", "worktree", "add", "-q", "--detach", str(wt), base_ref], cwd=str(self.clone))
            try:
                for p in editable:  # REPLACE (R3.4); absent keeps the trusted copy (R3.6)
                    s = src / p
                    if not s.exists():
                        continue
                    t = wt / p
                    if t.is_dir():
                        shutil.rmtree(t)
                    elif t.exists():
                        t.unlink()
                    shutil.copytree(s, t) if s.is_dir() else shutil.copy2(s, t)
                sh(["git", "add", "-A"], cwd=str(wt))
                sh(["git", "-c", "user.name=nemotron35-router", "-c", "user.email=router@localhost",
                    "commit", "-q", "--allow-empty", "-m", f"submission {sid} ({variant})"], cwd=str(wt))
                commit = sh(["git", "rev-parse", "HEAD"], cwd=str(wt)).strip()
                branch = f"sub/{sid}"
                sh(["git", "push", "-q", "origin", f"HEAD:refs/heads/{branch}"], cwd=str(wt), retry=True)
            finally:
                sh(["git", "worktree", "remove", "--force", str(wt)], cwd=str(self.clone), check=False)
        self.db.execute(
            "INSERT INTO submissions(id, variant, expect, note, created, branch, commit_sha, dropped, state,"
            " team, rescore, source, rescore_slot) VALUES (?,?,?,?,?,?,?,?, 'queued', ?,?,?,?)",
            (sid, variant, expect, note, time.time(), branch, commit, json.dumps(dropped), team, rescore, source_id,
             rescore_slot))
        self.db.commit()
        log(f"queued {sid} team={team} variant={variant} commit={commit[:8]} dropped={dropped}")
        return sid

    # ---- GitHub side --------------------------------------------------------------
    def gh_json(self, args: list[str]):
        return json.loads(sh([self.gh, *args], retry=True) or "null")

    def ranked_runner(self, run_id) -> str:
        # `gh run view --json jobs` carries no runner; the REST jobs API does.
        data = self.gh_json(["api", f"repos/{self.repo}/actions/runs/{run_id}/jobs"])
        return next((j.get("runner_name") or "" for j in data.get("jobs", [])
                     if j.get("name", "").startswith(RANKED_JOB_PREFIX)), "")

    def ranked_log(self, run_id) -> str:
        data = self.gh_json(["api", f"repos/{self.repo}/actions/runs/{run_id}/jobs"])
        job = next((j for j in data.get("jobs", []) if j.get("name", "").startswith(RANKED_JOB_PREFIX)), None)
        if not job:
            return ""
        try:
            return sh([self.gh, "api", "--allow-escape-sequences",
                       f"repos/{self.repo}/actions/jobs/{job['id']}/logs"], retry=True)
        except RuntimeError:
            return ""

    def online_runners(self) -> int:
        data = self.gh_json(["api", f"repos/{self.repo}/actions/runners?per_page=100"])
        return sum(1 for r in data.get("runners", [])
                   if r.get("status") == "online" and any(l.get("name") == TRACK_LABEL for l in r.get("labels", [])))

    def find_run(self, branch: str, polls: int):
        for _ in range(polls):  # the run appears a few seconds after the dispatch
            time.sleep(5)
            runs = self.gh_json(["run", "list", "--repo", self.repo, "--workflow", WORKFLOW,
                                 "--branch", branch, "--json", "databaseId", "--limit", "1"])
            if runs:
                return runs[0]["databaseId"]
        return None

    def dispatch(self, row) -> None:
        # Not sh(retry=True): a reset can land after GitHub accepted the dispatch.
        # Each branch is a fresh sub/<id>, so any run on it means the dispatch
        # landed; only re-dispatch when none has appeared.
        for delay in (*RETRY_DELAYS, None):
            try:
                sh([self.gh, "workflow", "run", WORKFLOW, "--repo", self.repo, "--ref", row["branch"]])
                break
            except RuntimeError as e:
                if delay is None or not transient(str(e)):
                    raise
                log(f"{row['id']}: dispatch transient failure: {e}")
                time.sleep(delay)
                # ponytail: a run that shows up >15 s after a reset still double-dispatches.
                if self.find_run(row["branch"], polls=3):
                    break
        run_id = self.find_run(row["branch"], polls=12)
        if run_id is None:
            self._update(row["id"], state="dispatch_failed", incident="dispatched but no run appeared")
            log(f"{row['id']}: dispatched but no run appeared")
            return
        self._update(row["id"], state="dispatched", run_id=run_id, dispatched_at=time.time())
        log(f"dispatched {row['id']} -> run {run_id}")

    def _update(self, sid: str, **fields) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE submissions SET {cols} WHERE id=?", (*fields.values(), sid))
        self.db.commit()

    @staticmethod
    def _ts(s: str | None) -> float | None:
        if not s or s.startswith("0001"):
            return None
        return float(calendar.timegm(time.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")))

    def watch(self, row) -> None:
        run = self.gh_json(["run", "view", str(row["run_id"]), "--repo", self.repo,
                            "--json", "status,conclusion,jobs"])
        ranked = next((j for j in run.get("jobs", []) if j.get("name", "").startswith(RANKED_JOB_PREFIX)), None)
        if ranked and row["state"] == "dispatched" and self._ts(ranked.get("startedAt")) and ranked.get("status") != "queued":
            box = self.ranked_runner(row["run_id"])
            self._update(row["id"], state="running", started_at=self._ts(ranked.get("startedAt")), box=box)
            log(f"{row['id']} running on {box}")
            row = self.db.execute("SELECT * FROM submissions WHERE id=?", (row["id"],)).fetchone()
        if run.get("status") != "completed":
            if row["state"] == "running" and row["started_at"] and time.time() - row["started_at"] > WATCHDOG_SECONDS:
                sh([self.gh, "run", "cancel", str(row["run_id"]), "--repo", self.repo], check=False)
                self._update(row["id"], incident=f"watchdog: cancelled after {WATCHDOG_SECONDS // 60} min")
                log(f"{row['id']}: WATCHDOG cancelled run {row['run_id']}")
            return
        row = self.db.execute("SELECT * FROM submissions WHERE id=?", (row["id"],)).fetchone()
        self.finish(row, run, ranked)

    def finish(self, row, run, ranked) -> None:
        result, passed, score, error = None, None, None, ""
        with tempfile.TemporaryDirectory(prefix="router-art-") as tmp:
            try:
                sh([self.gh, "run", "download", str(row["run_id"]), "--repo", self.repo,
                    "-n", f"benchmark-results-{row['run_id']}", "-D", tmp], retry=True)
                ok = True
            except RuntimeError:
                ok = False
            p = Path(tmp) / "score.json"
            if ok and p.exists():
                result = json.loads(p.read_text())
            else:
                # ponytail: artifact storage can be over quota (billing); the job
                # log prints the same score.json, so read it from there instead.
                result = score_from_log(self.ranked_log(row["run_id"]))
                if result:  # an empty {} stays empty: no score, not a refusal
                    result["score_source"] = "job-log"
                    # A log-derived PASS is trusted only if benchd's own step
                    # succeeded; a forged block cannot make that step green.
                    measured = next((st for st in (ranked or {}).get("steps", [])
                                     if st.get("name", "").startswith(MEASURE_STEP_PREFIX)), None)
                    if result.get("passed") and (measured or {}).get("conclusion") != "success":
                        result = None
            if result:
                passed = 1 if result.get("passed") else 0
                score = result.get("score")
                error = (result.get("metrics") or {}).get("error") or ""
        failed_step = ""
        for j in run.get("jobs", []):
            for st in j.get("steps", []):
                if st.get("conclusion") == "failure":
                    failed_step = f"{j.get('name', '')[:40]} / {st.get('name', '')}"
                    break
            if failed_step:
                break
        paired_legs = None
        if result:
            pl = (result.get("metrics") or {}).get("paired_legs")
            if pl is not None:
                paired_legs = pl
        outcome = classify(run.get("conclusion"), passed, error, failed_step,
                           ranked is not None and bool(result), setup_ran(ranked),
                           incident=row["incident"] or "", paired_legs=paired_legs)
        self._update(row["id"], state="done", finished_at=time.time(), conclusion=run.get("conclusion"),
                     passed=passed, score=score, error=error or failed_step, outcome=outcome,
                     result=json.dumps(result) if result else None,
                     box=row["box"] or self.ranked_runner(row["run_id"]))
        log(f"done {row['id']} variant={row['variant']} outcome={outcome} score={score} {error[:120]}")

    def run(self, once: bool = False) -> None:
        self.ensure_clone()
        while True:
            try:
                for row in self.db.execute("SELECT * FROM submissions WHERE state IN (?,?)", OPEN_STATES).fetchall():
                    self.watch(row)
                waiting = self.db.execute("SELECT COUNT(*) FROM submissions WHERE state='dispatched'").fetchone()[0]
                online = self.online_runners()
                for row in self.db.execute("SELECT * FROM submissions WHERE state='queued' ORDER BY created "
                                           "LIMIT ?", (max(0, online - waiting),)).fetchall():
                    self.dispatch(row)
            except Exception as e:  # never let one bad poll kill the daemon
                log(f"poll error: {e}")
            if once:
                return
            time.sleep(POLL_SECONDS)

    # ---- reporting ---------------------------------------------------------------
    def status(self) -> dict:
        rows = self.db.execute("SELECT state, COUNT(*) FROM submissions GROUP BY state").fetchall()
        return {r[0]: r[1] for r in rows}

    def report(self, out: Path) -> dict:
        rows = [dict(r) for r in self.db.execute("SELECT * FROM submissions ORDER BY created").fetchall()]
        for r in rows:
            res = json.loads(r.pop("result", None) or "null") or {}
            # Where the score came from -- a fact, not a trust level: the job
            # runs untrusted code as the runner user, so neither source is fit
            # for ranking until the build is sandboxed (README).
            r["score_source"] = (res.get("score_source") or "artifact") if res else ""
            r["meets_expectation"] = meets(r.get("expect"), r.get("outcome"))
        summary = {
            "total": len(rows), "by_state": self.status(),
            "by_outcome": _count(rows, "outcome"), "by_variant": _count(rows, "variant"),
            "by_box": _count([r for r in rows if r.get("box")], "box"),
            "by_score_source": _count([r for r in rows if r["score_source"]], "score_source"),
            "unexpected": [r["id"] for r in rows if r["state"] == "done" and r["meets_expectation"] is False],
        }
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps({"summary": summary, "submissions": rows}, indent=1))
        (out / "report.html").write_text(render_html(summary, rows))
        return summary


    # ---- freeze and the final re-score ----------------------------------------------
    def freeze(self) -> float:
        if not self.frozen_flag.exists():
            self.frozen_flag.write_text(f"{time.time()}\n")
        return self.frozen_at()

    def unfreeze(self) -> None:
        self.frozen_flag.unlink(missing_ok=True)

    def frozen_at(self) -> float | None:
        return float(self.frozen_flag.read_text().split()[0]) if self.frozen_flag.exists() else None

    def select_best(self, top: int) -> list[dict]:
        """Each team's best submission by mean contest score over runs that scored
        or failed as the team (team failures and post-setup control/band runs count
        as 0.0; pre-setup infrastructure runs are excluded).
        Runs of the same editable tree count together; ties -> most recent; top N teams."""
        rows = self.db.execute("SELECT * FROM submissions WHERE rescore IS NULL AND state='done' "
                               "ORDER BY created").fetchall()
        editable = self.editable_paths()
        groups: dict = {}
        for r in rows:
            value = contest_run_value(dict(r))
            if value is None:
                continue
            tree = sh(["git", "ls-tree", "-r", r["commit_sha"], "--", *editable], cwd=str(self.clone))
            g = groups.setdefault((r["team"], tree), {"team": r["team"], "scores": [], "latest": None})
            g["scores"].append(value)
            g["latest"] = dict(r)  # rows are in created order: the last is the most recent
        best: dict = {}
        for g in groups.values():
            if not g["scores"]:
                continue
            g["mean"] = sum(g["scores"]) / len(g["scores"])
            key = (g["mean"], g["latest"]["created"])
            if g["team"] not in best or key > (best[g["team"]]["mean"], best[g["team"]]["latest"]["created"]):
                best[g["team"]] = g
        ranked = sorted(best.values(), key=lambda g: (g["mean"], g["latest"]["created"]), reverse=True)
        return [{"team": g["team"], "submission": g["latest"]["id"], "commit": g["latest"]["commit_sha"],
                 "contest_mean": g["mean"], "contest_runs": len(g["scores"])} for g in ranked[:top]]

    def check_final_staging(self, staging_log: Path) -> None:
        """Refuse unless the `final` goldens were staged, and never before the freeze."""
        frozen = self.frozen_at()
        if frozen is None:
            raise RuntimeError("not frozen: run `router.py freeze` at the deadline first")
        lines = [ln.split() for ln in (staging_log.read_text().splitlines() if staging_log.exists() else [])]
        final = [ln for ln in lines if len(ln) >= 3 and ln[1] == "final"]
        early = [ln for ln in final if float(ln[0]) < frozen]
        if early:
            raise RuntimeError(f"{staging_log}: {len(early)} `final` golden(s) were staged BEFORE the freeze "
                               f"(first: box {early[0][2]} at {early[0][0]}); the held-out pool is burnt")
        if not final:
            raise RuntimeError(f"{staging_log}: the `final` goldens were never staged; run "
                               f"tools/pool-refresh.sh --label final --step stage --final-after-freeze {self.frozen_flag}")

    def build_rescore_base(self, base_ref: str, final_pins: Path) -> str:
        """rescore/base = base_ref + one fixture commit pointing live_goldens at `final`."""
        sh(["git", "fetch", "-q", "origin"], cwd=str(self.clone), retry=True)
        with tempfile.TemporaryDirectory(prefix="router-rescore-") as tmp:
            wt = Path(tmp) / "wt"
            sh(["git", "worktree", "add", "-q", "--detach", str(wt), base_ref], cwd=str(self.clone))
            try:
                sh([str(ROOT / "tools/pool-refresh.sh"), "pin-fixture", str(wt / FIXTURE_REL), "final", str(final_pins)])
                sh(["git", "-c", "user.name=nemotron35-router", "-c", "user.email=router@localhost",
                    "commit", "-q", "-am", "rescore: live_goldens = the held-out final generation"], cwd=str(wt))
                commit = sh(["git", "rev-parse", "HEAD"], cwd=str(wt)).strip()
            finally:
                sh(["git", "worktree", "remove", "--force", str(wt)], cwd=str(self.clone), check=False)
        sh(["git", "branch", "-f", "rescore/base", commit], cwd=str(self.clone))
        return commit

    def rescore(self, base_ref: str, final_pins: Path, staging_log: Path, top: int = 10, runs: int = 5,
                tag: str = "final") -> dict:
        self.ensure_clone()
        self.check_final_staging(staging_log)
        busy = self.db.execute("SELECT COUNT(*) FROM submissions WHERE rescore IS NULL AND state IN "
                               "('queued','dispatched','running')").fetchone()[0]
        if busy:
            raise RuntimeError(f"{busy} contest submission(s) are still queued or running; drain first")
        if self.db.execute("SELECT COUNT(*) FROM submissions WHERE rescore=?", (tag,)).fetchone()[0]:
            raise RuntimeError(f"rescore '{tag}' already queued; see `rescore-report --tag {tag}`")
        selected = self.select_best(top)
        if not selected:
            raise RuntimeError("no team has a contest run that scored or failed on the submission")
        base_commit = self.build_rescore_base(base_ref, final_pins)
        for sel in selected:
            sel["runs"] = [self._queue_rescore_run(sel["team"], sel["submission"], sel["commit"], k, runs, tag)
                           for k in range(runs)]
        plan = {"tag": tag, "runs": runs, "base_ref": base_ref, "rescore_base": base_commit, "selected": selected,
                "push": f"git -C {self.clone} push origin rescore/base"}
        (self.state_dir / f"rescore-{tag}.json").write_text(json.dumps(plan, indent=1))
        return plan

    def _queue_rescore_run(self, team: str, submission_id: str, commit: str, slot_i: int, runs_k: int,
                           tag: str) -> str:
        editable = self.editable_paths("rescore/base")
        with tempfile.TemporaryDirectory(prefix="router-src-") as tmp:
            tar = Path(tmp) / "s.tar"
            sh(["git", "archive", "-o", str(tar), commit, "--", *editable], cwd=str(self.clone))
            src = Path(tmp) / "src"
            src.mkdir()
            with tarfile.open(tar) as tf:
                tf.extractall(src, filter="tar")
            return self._overlay(src, "rescore/base", f"rescore:{submission_id}", "measure",
                                 f"rescore {tag} run {slot_i + 1}/{runs_k}", team, tag, submission_id,
                                 rescore_slot=slot_i)

    def _rescore_rows_with_prompts(self, tag: str) -> list[dict]:
        rows = [dict(r) for r in self.db.execute("SELECT * FROM submissions WHERE rescore=? ORDER BY created",
                                                 (tag,)).fetchall()]
        names: dict = {}
        for r in rows:
            res = json.loads(r.get("result") or "null") or {}
            prompt = self._scored_prompt(res.get("metrics") or {}, r.get("commit_sha"), names)
            r["_off_pool"] = not prompt.startswith("final-")
        return rows

    def rescore_retry(self, tag: str = "final", max_infra_retries: int = 2) -> dict:
        self.ensure_clone()
        busy = self.db.execute("SELECT COUNT(*) FROM submissions WHERE rescore=? AND state IN "
                               "('queued','dispatched','running')", (tag,)).fetchone()[0]
        if busy:
            raise RuntimeError(f"{busy} rescore '{tag}' run(s) still queued or running")
        plan_path = self.state_dir / f"rescore-{tag}.json"
        if not plan_path.exists():
            raise RuntimeError(f"no rescore plan for '{tag}'; run `rescore` first")
        plan = json.loads(plan_path.read_text())
        runs_k = plan["runs"]
        redispatched = []
        for sel in plan["selected"]:
            team = sel["team"]
            by_slot: dict[int, list] = {}
            for r in self._rescore_rows_with_prompts(tag):
                if r["team"] != team:
                    continue
                slot_i = rescore_slot_index(r)
                if slot_i is None:
                    continue
                by_slot.setdefault(slot_i, []).append(r)
            for slot_i in range(runs_k):
                slot_rows = by_slot.get(slot_i, [])
                if any(r["state"] in ("queued", "dispatched", "running") for r in slot_rows):
                    continue
                resolved = resolve_rescore_slot(slot_rows, max_infra_retries)
                if resolved["status"] != "pending":
                    continue
                if not any(a.get("kind") in ("infra", "boot_failed") for a in resolved.get("attempts", [])):
                    continue
                run_id = self._queue_rescore_run(team, sel["submission"], sel["commit"], slot_i, runs_k, tag)
                redispatched.append({"team": team, "slot": slot_i, "run_id": run_id})
        return {"tag": tag, "max_infra_retries": max_infra_retries, "redispatched": redispatched}

    def _scored_prompt(self, metrics: dict, commit: str | None, cache: dict) -> str:
        """The prompt a run measured. benchd seals no prompt name, only the serial
        golden's sha256 (metrics.baseline_golden_sha256); map it back through the
        timed_prompt_pool pins of the fixture at the run's own commit. An unknown
        digest maps to "" and so counts as off-pool."""
        sha = metrics.get("baseline_golden_sha256")
        if not sha:
            return metrics.get("live_prompt") or ""  # records sealed before benchd stopped sealing names
        if commit not in cache:
            try:
                fx = json.loads(sh(["git", "show", f"{commit}:{FIXTURE_REL}"], cwd=str(self.clone)))
            except (RuntimeError, ValueError):
                fx = {}
            cache[commit] = {e["sha256"]: e["r2_path"].rsplit("/", 1)[-1].removesuffix(".golden.json")
                             for e in fx.get("timed_prompt_pool", []) if e.get("sha256") and e.get("r2_path")}
        return cache[commit].get(sha, "")

    def rescore_report(self, out: Path, tag: str = "final", max_infra_retries: int = 2) -> dict:
        rows = [dict(r) for r in self.db.execute("SELECT * FROM submissions WHERE rescore=? ORDER BY created",
                                                 (tag,)).fetchall()]
        if not rows:
            raise RuntimeError(f"no rescore '{tag}' runs")
        plan_path = self.state_dir / f"rescore-{tag}.json"
        runs_k = json.loads(plan_path.read_text()).get("runs") if plan_path.exists() else None
        if runs_k is None:
            parsed = [parse_rescore_slot(r.get("note") or "") for r in rows]
            runs_k = max((k for _, k in parsed if k), default=0)
        runs, teams, names = [], {}, {}
        for r in rows:
            res = json.loads(r.get("result") or "null") or {}
            prompt = self._scored_prompt(res.get("metrics") or {}, r.get("commit_sha"), names)
            off_pool = not prompt.startswith("final-")
            r["_off_pool"] = off_pool
            run = {"team": r["team"], "submission": r["source"], "id": r["id"], "state": r["state"],
                   "outcome": r["outcome"], "box": r["box"], "prompt": prompt, "score": r["score"],
                   "passed": r["passed"], "off_pool": off_pool, "note": r.get("note")}
            runs.append(run)
            slot_i = rescore_slot_index(r)
            if slot_i is None:
                continue
            by_slot = teams.setdefault(r["team"], {"team": r["team"], "submission": r["source"],
                                                   "by_slot": {}})["by_slot"]
            by_slot.setdefault(slot_i, []).append(r)
        for t in teams.values():
            slots = []
            values = []
            complete = True
            for i in range(runs_k):
                resolved = resolve_rescore_slot(t["by_slot"].get(i, []), max_infra_retries)
                slots.append(resolved)
                if resolved["status"] == "pending":
                    complete = False
                elif resolved["status"] == "unresolved":
                    complete = False
                else:
                    values.append(resolved["value"])
            t["slots"] = slots
            t["status"] = "complete" if complete and len(values) == runs_k else "incomplete"
            t["mean"] = sum(values) / runs_k if t["status"] == "complete" else None
            del t["by_slot"]
        ranking = sorted(teams.values(),
                         key=lambda t: (0 if t["status"] == "complete" else 1, -(t["mean"] or 0), t["team"]))
        rank = 0
        for t in ranking:
            if t["status"] == "complete":
                rank += 1
                t["rank"] = rank
            else:
                t["rank"] = None
        summary = {"tag": tag, "runs": len(runs), "runs_per_team": runs_k,
                   "pending": sum(r["state"] != "done" for r in runs),
                   "off_pool_runs": [r["id"] for r in runs if r["off_pool"]],
                   "ranking": [{k: t[k] for k in ("rank", "team", "submission", "status", "mean", "slots")} for t in ranking]}
        out.mkdir(parents=True, exist_ok=True)
        (out / "rescore.json").write_text(json.dumps({"summary": summary, "runs": runs}, indent=1))
        (out / "rescore.html").write_text(render_html(summary, runs, head=[
            "team", "submission", "id", "box", "prompt", "score", "passed", "outcome", "off_pool"]))
        return summary

MEASURE_STEP_PREFIX = "GPU-exclusive measurement"

RESCORE_NOTE_RE = re.compile(r"rescore \S+ run (\d+)/(\d+)")
TEAM_FAILURE_OUTCOMES = frozenset({
    "replay_rejected", "floor_failed", "correctness_failed", "refused", "surface_refused", "unsealed",
    "watchdog_timeout",
})
POST_SETUP_INFRA_OUTCOMES = frozenset({"control_failed", "band_refused"})
INFRA_OUTCOMES = frozenset({
    "control_failed", "band_refused", "infra_failed", "cancelled", "preflight_refused",
})


def parse_rescore_slot(note: str) -> tuple[int, int] | None:
    m = RESCORE_NOTE_RE.search(note)
    if not m:
        return None
    return int(m.group(1)) - 1, int(m.group(2))


def rescore_slot_index(row: dict) -> int | None:
    if row.get("rescore_slot") is not None:
        return int(row["rescore_slot"])
    parsed = parse_rescore_slot(row.get("note") or "")
    return parsed[0] if parsed else None


def rescore_run_kind(row: dict, off_pool: bool) -> str:
    if row["state"] != "done":
        return "pending"
    if off_pool:
        return "infra"
    incident = row.get("incident") or ""
    if incident.startswith("watchdog:"):
        return "team_failure"
    outcome = row.get("outcome") or ""
    if outcome == "pass":
        return "scored"
    if outcome == "boot_failed":
        return "boot_failed"
    if outcome in TEAM_FAILURE_OUTCOMES:
        return "team_failure"
    if outcome in INFRA_OUTCOMES:
        return "infra"
    if row.get("passed") == 1 and row.get("score") is not None:
        return "scored"
    return "infra"


def resolve_rescore_slot(run_rows: list[dict], max_infra_retries: int) -> dict:
    """First terminating scored or team-failure run wins; infra runs may be retried."""
    infra = 0
    boot_failed = 0
    saw_post_setup_infra = False
    attempts: list[dict] = []
    for r in sorted(run_rows, key=lambda x: x["created"]):
        kind = rescore_run_kind(r, r.get("_off_pool", False))
        outcome = r.get("outcome") or ""
        attempts.append({"id": r["id"], "outcome": r.get("outcome"), "state": r["state"],
                         "score": r.get("score"), "kind": kind})
        if kind == "pending":
            return {"status": "pending", "value": None, "run_id": r["id"], "outcome": r.get("outcome"),
                    "attempts": attempts}
        if kind == "scored":
            return {"status": "scored", "value": r["score"], "run_id": r["id"], "outcome": r.get("outcome"),
                    "attempts": attempts}
        if kind == "team_failure":
            return {"status": "team_failure", "value": 0.0, "run_id": r["id"], "outcome": r.get("outcome"),
                    "attempts": attempts}
        if kind == "boot_failed":
            boot_failed += 1
            if boot_failed >= 2:
                return {"status": "team_failure", "value": 0.0, "run_id": r["id"], "outcome": outcome,
                        "attempts": attempts}
            continue
        if outcome in POST_SETUP_INFRA_OUTCOMES:
            saw_post_setup_infra = True
        infra += 1
        if infra > max_infra_retries:
            if saw_post_setup_infra:
                return {"status": "penalized", "value": 0.0, "run_id": r["id"], "outcome": outcome,
                        "attempts": attempts}
            return {"status": "unresolved", "value": None, "run_id": r["id"], "outcome": outcome,
                    "attempts": attempts}
    if infra or boot_failed:
        return {"status": "pending", "value": None, "run_id": None, "outcome": None, "attempts": attempts}
    return {"status": "pending", "value": None, "run_id": None, "outcome": None, "attempts": attempts}


def contest_run_value(row: dict) -> float | None:
    """Contest run contribution to select_best: score, 0.0 on team failure, None if infra."""
    if row["state"] != "done":
        return None
    incident = row.get("incident") or ""
    if incident.startswith("watchdog:"):
        return 0.0
    outcome = row.get("outcome") or ""
    if outcome == "pass" and row.get("score") is not None:
        return row["score"]
    if outcome == "boot_failed":
        return 0.0
    if outcome in TEAM_FAILURE_OUTCOMES:
        return 0.0
    if outcome in POST_SETUP_INFRA_OUTCOMES:
        return 0.0
    if outcome in INFRA_OUTCOMES:
        return None
    if row.get("passed") == 1 and row.get("score") is not None:
        return row["score"]
    return None


_SCORE_END_RE = re.compile(
    r"^NEMOTRON35-SCORE-END ([0-9a-f]{64}) (sha256=([0-9a-f]{64}) bytes=(\d+)|none)$"
)


def _log_text_lines(log: str) -> list[str]:
    return [ln.split(" ", 1)[1] if " " in ln else ln for ln in log.splitlines()]


def _seal_body_matches(body: str, expected_hash: str, expected_bytes: int) -> bool:
    """END sha256/bytes are of score.json on disk. The reveal step may add one
    trailing newline when the file lacks one; the log block may also omit a
    file-final newline because each line is a separate log record."""
    raw = body.encode()
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) == expected_bytes and digest == expected_hash:
        return True
    if raw.endswith(b"\n") and len(raw) - 1 == expected_bytes:
        if hashlib.sha256(raw[:-1]).hexdigest() == expected_hash:
            return True
    if len(raw) + 1 == expected_bytes and hashlib.sha256(raw + b"\n").hexdigest() == expected_hash:
        return True
    return False


def score_from_log(log: str) -> dict | None:
    """Parse the post-cleanup commit-reveal score block from a ranked job log."""
    lines = _log_text_lines(log)
    end_idx = end_m = None
    for i in range(len(lines) - 1, -1, -1):
        m = _SCORE_END_RE.match(lines[i])
        if m:
            end_idx, end_m = i, m
            break
    if end_m is None:
        return None
    nonce = end_m.group(1)
    if end_m.group(2) == "none":
        return {}
    expected_hash, expected_bytes = end_m.group(3), int(end_m.group(4))
    begin_re = re.compile(rf"^NEMOTRON35-SCORE-BEGIN {re.escape(nonce)}$")
    begin_idx = None
    for i in range(end_idx - 1, -1, -1):
        if begin_re.match(lines[i]):
            begin_idx = i
            break
    if begin_idx is None:
        return None
    body = "\n".join(lines[begin_idx + 1 : end_idx])
    if not _seal_body_matches(body, expected_hash, expected_bytes):
        return None
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "passed" not in data or "metrics" not in data:
        return None
    return data


def setup_ran(ranked) -> bool:
    if not ranked:
        return False
    for st in ranked.get("steps", []):
        if st.get("name", "").startswith("Setup") and st.get("conclusion") != "skipped":
            return True
    return False


def classify(conclusion, passed, error: str, failed_step: str, have_score: bool,
             setup_ran_flag: bool = False, incident: str = "",
             paired_legs=None) -> str:
    e = error or ""
    if passed:
        return "pass"
    if e.startswith("SERIAL-CONTROL-LEG-FAILED"):
        return "control_failed"  # reference leg failed twice (measure-and-score retries once): infra
    if "OUTSIDE-BAND" in e:
        return "band_refused"  # box health (control leg drifted), not the submission
    # benchd boots this leg with the label "serial-control" (main.rs boot_leg).
    if e.startswith("LEG-SERVE-BOOT-FAILED: the serial-control leg"):
        # benchd omits paired_legs while it is empty and always seals the completed
        # pairs on a boot refusal, so a pair-2 control boot always carries pair 1
        # and stays final; absent or [] is pair 1, the case measure-and-score retries.
        if paired_legs is not None and len(paired_legs) > 0:
            pass  # fall through to have_score -> refused
        else:
            return "control_failed"  # pair 1 or unknown audit trail: infra
    # benchd boots this leg with the label "candidate" (main.rs boot_leg).
    if e.startswith("LEG-SERVE-BOOT-FAILED: the candidate leg"):
        return "boot_failed"
    if "TIMED-REPLAY-REJECTED" in e:
        return "replay_rejected"
    if "performance floor failed" in e:
        return "floor_failed"
    if "token mismatch" in e.lower() or "correctness" in e.lower():
        return "correctness_failed"
    if conclusion == "cancelled":
        if incident.startswith("watchdog:"):
            return "watchdog_timeout"
        return "cancelled"
    if have_score:
        return "refused"
    if "surface-check" in failed_step or "static review" in failed_step.lower() or "lint" in failed_step.lower():
        return "surface_refused"
    if "preflight" in failed_step.lower():
        return "preflight_refused"
    if setup_ran_flag and not have_score:
        return "unsealed"
    return "infra_failed"


EXPECT = {
    "pass": {"pass"},
    "replay_rejected": {"replay_rejected"},
    "boot_failed": {"boot_failed"},
    "refused": {"surface_refused", "preflight_refused", "refused", "unsealed", "watchdog_timeout"},
    "measure": None,  # any terminal outcome is acceptable; the number is the point
}


def meets(expect, outcome):
    if outcome is None:
        return None
    allowed = EXPECT.get(expect or "measure")
    return True if allowed is None else outcome in allowed


def _count(rows, key):
    out: dict = {}
    for r in rows:
        out[r.get(key) or "-"] = out.get(r.get(key) or "-", 0) + 1
    return out


def render_html(summary, rows, head=None) -> str:
    def cell(v):
        return html.escape("" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v)))
    head = head or ["id", "team", "variant", "expect", "outcome", "meets_expectation", "box", "score",
                    "score_source", "error"]
    body = "".join("<tr>" + "".join(f"<td>{cell(r.get(k))}</td>" for k in head) + "</tr>" for r in rows)
    return (f"<!doctype html><meta charset=utf-8><title>Router report</title>"
            f"<pre>{html.escape(json.dumps(summary, indent=1))}</pre>"
            f"<table border=1 cellpadding=3><tr>{''.join(f'<th>{h}</th>' for h in head)}</tr>{body}</table>")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state-dir", default=os.path.expanduser("~/nemotron35-router"))
    ap.add_argument("--repo", default=None, help="owner/repo (default: ROUTER_REPO, else `gh repo view`)")
    ap.add_argument("--base", default=DEFAULT_BASE, help="base branch (default: ROUTER_BASE, else loadtest/base)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("source")
    s.add_argument("--variant", required=True)
    s.add_argument("--expect", default="measure", choices=sorted(EXPECT))
    s.add_argument("--note", default="")
    s.add_argument("--team", default=None, help="the team this submission is ranked under (default: --variant)")
    r = sub.add_parser("run")
    r.add_argument("--once", action="store_true")
    sub.add_parser("status")
    rp = sub.add_parser("report")
    rp.add_argument("--out", default=None)
    sub.add_parser("freeze")
    sub.add_parser("unfreeze")
    rs = sub.add_parser("rescore")
    rs.add_argument("--base-ref", required=True, help="the contest's final base, e.g. origin/main")
    rs.add_argument("--top", type=int, default=10)
    rs.add_argument("--runs", type=int, default=5)
    rs.add_argument("--tag", default="final")
    rs.add_argument("--final-pins", default=str(ROOT / ".mlxfast-private/final/pins.json"))
    rs.add_argument("--staging-log", default=str(ROOT / ".mlxfast-private/refresh/staging.log"))
    rr = sub.add_parser("rescore-report")
    rr.add_argument("--tag", default="final")
    rr.add_argument("--out", default=None)
    rr.add_argument("--wait", action="store_true", help="poll until every run is done (the `run` daemon dispatches)")
    rr.add_argument("--max-infra-retries", type=int, default=2,
                    help="extra infra re-dispatches per slot after the first attempt (total attempts = 1 + this)")
    rt = sub.add_parser("rescore-retry")
    rt.add_argument("--tag", default="final")
    rt.add_argument("--max-infra-retries", type=int, default=2,
                    help="extra infra re-dispatches per slot after the first attempt (total attempts = 1 + this)")
    a = ap.parse_args()
    router = Router(Path(a.state_dir), a.repo or default_repo(), a.base)
    if a.cmd == "submit":
        print(router.submit(Path(a.source), a.variant, a.expect, a.note, a.team))
    elif a.cmd == "run":
        router.run(once=a.once)
    elif a.cmd == "status":
        print(json.dumps(router.status()))
    elif a.cmd == "report":
        print(json.dumps(router.report(Path(a.out or Path(a.state_dir) / "report")), indent=1))
    elif a.cmd == "freeze":
        print(f"frozen at {router.freeze()}; submissions are refused")
    elif a.cmd == "unfreeze":
        router.unfreeze()
    elif a.cmd == "rescore":
        plan = router.rescore(a.base_ref, Path(a.final_pins), Path(a.staging_log), a.top, a.runs, a.tag)
        print(json.dumps(plan, indent=1))
        print(f"push the base (the sub/ branches already carry it): {plan['push']}", file=sys.stderr)
    elif a.cmd == "rescore-report":
        while a.wait and router.db.execute("SELECT COUNT(*) FROM submissions WHERE rescore=? AND state IN "
                                           "('queued','dispatched','running')", (a.tag,)).fetchone()[0]:
            time.sleep(POLL_SECONDS)
        print(json.dumps(router.rescore_report(Path(a.out or Path(a.state_dir) / f"rescore-{a.tag}"), a.tag,
                                               a.max_infra_retries), indent=1))
    elif a.cmd == "rescore-retry":
        print(json.dumps(router.rescore_retry(a.tag, a.max_infra_retries), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

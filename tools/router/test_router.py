#!/usr/bin/env python3
"""Tests for router.py and loadgen.py. No network, no `gh`: a local bare repo
stands in for GitHub (ROUTER_REMOTE); retries and dispatch use a fake `gh`."""
from __future__ import annotations

import hashlib
import json
import os
import py_compile
import random
import shutil
import subprocess
import sys
import sqlite3
import tarfile
import time
import tempfile
import unittest
from unittest import mock
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import loadgen  # noqa: E402
import router  # noqa: E402

REPO_ROOT = HERE.parent.parent


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


class RouterSubmitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        src = t / "src"
        (src / "harness").mkdir(parents=True)
        (src / "vllm").mkdir()
        (src / "tools").mkdir()
        (src / "benchmark.json").write_text(json.dumps(
            {"editablePaths": ["vllm", "harness", "mtp-head.manifest.json"]}))
        (src / "harness/a.txt").write_text("trusted harness\n")
        (src / "vllm/b.txt").write_text("trusted vllm\n")
        (src / "tools/x.sh").write_text("trusted tool\n")
        (src / "mtp-head.manifest.json").write_text("{}\n")
        git("init", "-q", "-b", "loadtest/base", cwd=src)
        git("add", "-A", cwd=src)
        git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base", cwd=src)
        self.bare = t / "remote.git"
        git("clone", "-q", "--bare", str(src), str(self.bare), cwd=t)
        os.environ["ROUTER_REMOTE"] = str(self.bare)
        self.r = router.Router(t / "state", "o/r", "loadtest/base")

    def tearDown(self):
        self.r.db.close()
        os.environ.pop("ROUTER_REMOTE", None)
        self.tmp.cleanup()

    def tree(self, branch):
        return set(git("ls-tree", "-r", "--name-only", branch, cwd=self.bare).split())

    def test_drops_trusted_paths_and_replaces_editable_ones(self):
        sub = Path(self.tmp.name) / "sub"
        (sub / "harness").mkdir(parents=True)
        (sub / "harness/new.txt").write_text("participant\n")
        (sub / "tools").mkdir()
        (sub / "tools/x.sh").write_text("pwned\n")
        sid = self.r.submit(sub, "t", "pass")
        row = self.r.db.execute("SELECT * FROM submissions WHERE id=?", (sid,)).fetchone()
        self.assertEqual(row["state"], "queued")
        self.assertEqual(json.loads(row["dropped"]), ["tools"])
        files = self.tree(row["branch"])
        self.assertIn("harness/new.txt", files)
        self.assertNotIn("harness/a.txt", files, "REPLACE, not merge (R3.4)")
        self.assertIn("vllm/b.txt", files, "an absent editable path keeps the trusted copy (R3.6)")
        tool = git("show", f"{row['branch']}:tools/x.sh", cwd=self.bare)
        self.assertEqual(tool, "trusted tool\n", "a trusted path never changes")

    def test_tarball_source_and_symlink_refusal(self):
        t = Path(self.tmp.name)
        sub = t / "tsub"
        (sub / "vllm").mkdir(parents=True)
        (sub / "vllm/c.txt").write_text("v\n")
        tar = t / "s.tar.gz"
        with tarfile.open(tar, "w:gz") as tf:
            tf.add(sub / "vllm", arcname="vllm")
        sid = self.r.submit(tar, "t", "pass")
        branch = self.r.db.execute("SELECT branch FROM submissions WHERE id=?", (sid,)).fetchone()[0]
        self.assertIn("vllm/c.txt", self.tree(branch))
        (sub / "vllm/link").symlink_to("/etc/passwd")
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            self.r.submit(sub, "t", "pass")


class TeamFreezeRescoreTest(unittest.TestCase):
    """team column + migration, freeze, and the final re-score, against a local bare repo."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = self.t = Path(self.tmp.name)
        src = t / "src"
        (src / "harness").mkdir(parents=True)
        (src / "vllm").mkdir()
        (src / "fixtures").mkdir()
        (src / "benchmark.json").write_text(json.dumps(
            {"editablePaths": ["vllm", "harness", "mtp-head.manifest.json"]}))
        (src / "harness/a.txt").write_text("trusted\n")
        (src / "vllm/b.txt").write_text("trusted\n")
        (src / "mtp-head.manifest.json").write_text("{}\n")
        (src / router.FIXTURE_REL).write_text(json.dumps({
            "timed_prompt_pool": [{"r2_path": "cp/t/botany.golden.json", "sha256": "a" * 64, "bytes": 1}],
            "mtp_head": {"permitted_draft_depths": [1, 2]}, "official_pairs": 2,
            "speculative_oracles": {}, "live_goldens": ["botany"],
            "hidden_correctness_golden": {"sha256": "a" * 64, "bytes": 1}}, indent=1))
        git("init", "-q", "-b", "main", cwd=src)
        git("add", "-A", cwd=src)
        git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base", cwd=src)
        self.bare = t / "remote.git"
        git("clone", "-q", "--bare", str(src), str(self.bare), cwd=t)
        os.environ["ROUTER_REMOTE"] = str(self.bare)
        self.r = router.Router(t / "state", "o/r", "main")
        self.pins = t / "pins.json"
        self.pins.write_text(json.dumps({f: {"sha256": __import__("hashlib").sha256(f.encode()).hexdigest(), "bytes": 2} for f in [
            "final-1.golden.json", "final-2.golden.json", "final-1.mtp1.golden.json", "final-1.mtp2.golden.json",
            "final-2.mtp1.golden.json", "final-2.mtp2.golden.json"]}))
        self.log = t / "staging.log"

    def tearDown(self):
        self.r.db.close()
        os.environ.pop("ROUTER_REMOTE", None)
        self.tmp.cleanup()

    def sub(self, team, content, score, passed=1):
        d = self.t / f"s-{uuid_hex()}"
        (d / "harness").mkdir(parents=True)
        (d / "harness/engine.txt").write_text(content)
        sid = self.r.submit(d, "v", "measure", team=team)
        self.r._update(sid, state="done", passed=passed, score=score)
        return sid

    def test_old_database_gets_a_team_column_from_the_label(self):
        state = self.t / "old"
        state.mkdir()
        db = sqlite3.connect(state / "state.db")
        db.execute("CREATE TABLE submissions (id TEXT PRIMARY KEY, variant TEXT, expect TEXT, note TEXT, created REAL,"
                   " branch TEXT, commit_sha TEXT, dropped TEXT, state TEXT, run_id INTEGER, dispatched_at REAL,"
                   " started_at REAL, finished_at REAL, conclusion TEXT, box TEXT, passed INTEGER, score REAL,"
                   " error TEXT, outcome TEXT, incident TEXT, result TEXT)")
        db.execute("INSERT INTO submissions(id, variant, state) VALUES ('x', 'null', 'done')")
        db.commit()
        db.close()
        r = router.Router(state, "o/r", "main")
        self.addCleanup(r.db.close)
        row = r.db.execute("SELECT team, rescore FROM submissions").fetchone()
        self.assertEqual((row["team"], row["rescore"]), ("null", None))

    def test_submit_defaults_team_to_label_and_freeze_refuses(self):
        d = self.t / "s"
        (d / "vllm").mkdir(parents=True)
        sid = self.r.submit(d, "mtp", "measure")
        self.assertEqual(self.r.db.execute("SELECT team FROM submissions WHERE id=?", (sid,)).fetchone()[0], "mtp")
        self.r.freeze()
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            self.r.submit(d, "mtp", "measure")
        self.r.unfreeze()
        self.r.submit(d, "mtp", "measure")

    def test_select_best_groups_resubmissions_by_tree(self):
        self.sub("A", "x", 1.0)
        self.sub("A", "x", 1.0)
        a_best = self.sub("A", "y", 1.3)
        self.sub("C", "p", 2.0)
        self.sub("C", "p", 0.0)            # tree p means 1.0 although one run scored 2.0
        c_best = self.sub("C", "q", 1.5)
        self.sub("B", "z", 9.0, passed=0)  # a failed run never counts
        b_best = self.sub("B", "w", 0.5)
        sel = self.r.select_best(10)
        self.assertEqual([(s["team"], s["submission"]) for s in sel], [("C", c_best), ("A", a_best), ("B", b_best)])
        self.assertEqual(len(self.r.select_best(2)), 2)

    def test_rescore_refuses_without_freeze_or_with_early_final_staging(self):
        self.sub("A", "x", 1.0)
        with self.assertRaisesRegex(RuntimeError, "not frozen"):
            self.r.rescore("origin/main", self.pins, self.log)
        self.log.write_text(f"{time.time() - 100} final box-1 final-1.golden.json {'b' * 64}\n")
        self.r.freeze()
        with self.assertRaisesRegex(RuntimeError, "BEFORE the freeze"):
            self.r.rescore("origin/main", self.pins, self.log)
        self.log.write_text(f"{time.time() - 100} g1 box-1 g1-1.golden.json {'b' * 64}\n")
        with self.assertRaisesRegex(RuntimeError, "never staged"):
            self.r.rescore("origin/main", self.pins, self.log)

    def test_rescore_overlays_the_best_onto_a_final_base_and_ranks(self):
        a = self.sub("A", "a-engine", 1.2)
        b = self.sub("B", "b-engine", 1.1)
        self.sub("C", "c-engine", 0.9)
        busy = self.sub("C", "late", 0.1)
        self.r._update(busy, state="running", score=None)
        self.r.freeze()
        self.log.write_text(f"{time.time() + 1} final box-1 final-1.golden.json {'b' * 64}\n")
        with self.assertRaisesRegex(RuntimeError, "drain first"):
            self.r.rescore("origin/main", self.pins, self.log, top=2, runs=2)
        self.r._update(busy, state="done", passed=0)
        plan = self.r.rescore("origin/main", self.pins, self.log, top=2, runs=2)
        self.assertEqual([s["submission"] for s in plan["selected"]], [a, b])
        fx = json.loads(git("show", f"rescore/base:{router.FIXTURE_REL}", cwd=self.r.clone))
        self.assertEqual(fx["live_goldens"], ["final-1", "final-2"])
        self.assertNotIn("live_golden_rotation", fx)
        self.assertIn("push origin rescore/base", plan["push"])
        rows = self.r.db.execute("SELECT * FROM submissions WHERE rescore='final' ORDER BY created").fetchall()
        self.assertEqual([(r["team"], r["source"], r["state"]) for r in rows],
                         [("A", a, "queued")] * 2 + [("B", b, "queued")] * 2)
        for r in rows:
            engine = git("show", f"{r['branch']}:harness/engine.txt", cwd=self.bare)
            self.assertEqual(engine, {"A": "a-engine", "B": "b-engine"}[r["team"]])
            pool = json.loads(git("show", f"{r['branch']}:{router.FIXTURE_REL}", cwd=self.bare))["live_goldens"]
            self.assertEqual(pool, ["final-1", "final-2"])
        with self.assertRaisesRegex(RuntimeError, "already queued"):
            self.r.rescore("origin/main", self.pins, self.log, top=2, runs=2)
        # B completes both in-pool slots; A's second slot is off-pool (infra) so A stays incomplete.
        sha = lambda name: __import__("hashlib").sha256(f"{name}.golden.json".encode()).hexdigest()
        results = [("A", 1.0, "final-1"), ("A", 3.0, "botany"), ("B", 1.5, "final-2"), ("B", 1.7, "final-1")]
        for r, (team, score, prompt) in zip(rows, results):
            self.r._update(r["id"], state="done", passed=1, score=score, box=f"box-{score}",
                           outcome="pass",
                           result=json.dumps({"passed": True, "score": score,
                                              "metrics": {"baseline_golden_sha256": sha(prompt)}}))
        out = self.t / "rep"
        summary = self.r.rescore_report(out)
        self.assertEqual(summary["ranking"][0]["team"], "B")
        self.assertEqual(summary["ranking"][0]["mean"], 1.6)
        self.assertEqual(summary["ranking"][0]["status"], "complete")
        self.assertEqual(summary["ranking"][1]["team"], "A")
        self.assertIsNone(summary["ranking"][1]["rank"])
        self.assertEqual(summary["ranking"][1]["status"], "incomplete")
        self.assertIsNone(summary["ranking"][1]["mean"])
        self.assertEqual(len(summary["off_pool_runs"]), 1)
        rep = json.loads((out / "rescore.json").read_text())
        self.assertEqual({r["prompt"] for r in rep["runs"]}, {"final-1", "final-2", ""})
        self.assertIn("box-1.7", (out / "rescore.html").read_text())

    def test_rescore_report_slot_mean_counts_team_failures_as_zero(self):
        self.sub("slow", "x", 1.0)
        self.sub("fast", "y", 1.0)
        self.r.freeze()
        self.log.write_text(f"{time.time() + 1} final box-1 final-1.golden.json {'b' * 64}\n")
        self.r.rescore("origin/main", self.pins, self.log, top=2, runs=5)
        rows = self.r.db.execute("SELECT * FROM submissions WHERE rescore='final' ORDER BY created").fetchall()
        sha = __import__("hashlib").sha256(b"final-1.golden.json").hexdigest()
        by_team = {}
        for r in rows:
            by_team.setdefault(r["team"], []).append(r)
        self.r._update(by_team["slow"][0]["id"], state="done", passed=1, score=1.4, outcome="pass",
                       result=json.dumps({"metrics": {"baseline_golden_sha256": sha}}))
        for r in by_team["slow"][1:]:
            self.r._update(r["id"], state="done", passed=0, outcome="replay_rejected",
                           result=json.dumps({"metrics": {"baseline_golden_sha256": sha}}))
        for r in by_team["fast"]:
            self.r._update(r["id"], state="done", passed=1, score=1.05, outcome="pass",
                           result=json.dumps({"metrics": {"baseline_golden_sha256": sha}}))
        summary = self.r.rescore_report(self.t / "slot-mean")
        by_team = {t["team"]: t for t in summary["ranking"]}
        self.assertAlmostEqual(by_team["slow"]["mean"], 0.28)
        self.assertEqual(by_team["fast"]["mean"], 1.05)
        self.assertEqual(by_team["slow"]["rank"], 2)
        self.assertEqual(by_team["fast"]["rank"], 1)

    def test_select_best_counts_team_failures_as_zero(self):
        self.sub("A", "a", 1.0)
        sid = self.sub("A", "a", 9.0, passed=0)
        self.r._update(sid, state="done", passed=0, score=None, outcome="replay_rejected")
        sel = self.r.select_best(10)
        self.assertEqual(len(sel), 1)
        self.assertEqual(sel[0]["team"], "A")
        self.assertAlmostEqual(sel[0]["contest_mean"], 0.5)

    def test_rescore_retry_redispatches_infra_and_off_pool(self):
        a = self.sub("A", "a-engine", 1.2)
        self.r.freeze()
        self.log.write_text(f"{time.time() + 1} final box-1 final-1.golden.json {'b' * 64}\n")
        self.r.rescore("origin/main", self.pins, self.log, top=1, runs=2)
        rows = self.r.db.execute("SELECT * FROM submissions WHERE rescore='final' ORDER BY created").fetchall()
        sha_final = __import__("hashlib").sha256(b"final-1.golden.json").hexdigest()
        sha_botany = __import__("hashlib").sha256(b"botany.golden.json").hexdigest()
        by_slot = {router.rescore_slot_index(dict(r)): r for r in rows}
        self.r._update(by_slot[0]["id"], state="done", passed=1, score=1.0, outcome="pass",
                       result=json.dumps({"metrics": {"baseline_golden_sha256": sha_final}}))
        self.r._update(by_slot[1]["id"], state="done", passed=0, outcome="control_failed",
                       result=json.dumps({"metrics": {"baseline_golden_sha256": sha_final}}))
        summary = self.r.rescore_report(self.t / "pre-retry")
        self.assertEqual(summary["ranking"][0]["status"], "incomplete")
        plan = self.r.rescore_retry("final")
        self.assertEqual(len(plan["redispatched"]), 1)
        self.assertEqual(plan["redispatched"][0]["slot"], 1)
        retry_row = self.r.db.execute("SELECT * FROM submissions WHERE id=?", (plan["redispatched"][0]["run_id"],)).fetchone()
        self.assertEqual(retry_row["rescore_slot"], 1)
        with self.assertRaisesRegex(RuntimeError, "still queued or running"):
            self.r.rescore_retry("final")
        self.r._update(retry_row["id"], state="done", passed=0, outcome="pass",
                       result=json.dumps({"metrics": {"baseline_golden_sha256": sha_botany}}))
        summary = self.r.rescore_report(self.t / "off-pool")
        self.assertEqual(summary["ranking"][0]["status"], "incomplete")
        self.assertEqual(len(summary["off_pool_runs"]), 1)
        plan2 = self.r.rescore_retry("final")
        self.assertEqual(len(plan2["redispatched"]), 1)
        self.r._update(plan2["redispatched"][0]["run_id"], state="done", passed=1, score=1.5, outcome="pass",
                       result=json.dumps({"metrics": {"baseline_golden_sha256": sha_final}}))
        summary = self.r.rescore_report(self.t / "done")
        self.assertEqual(summary["ranking"][0]["status"], "complete")
        self.assertAlmostEqual(summary["ranking"][0]["mean"], 1.25)
        self.assertEqual(self.r.rescore_retry("final")["redispatched"], [])

    def test_rescore_retry_cap_penalizes_post_setup_infra(self):
        self.sub("A", "a", 1.0)
        self.r.freeze()
        self.log.write_text(f"{time.time() + 1} final box-1 final-1.golden.json {'b' * 64}\n")
        self.r.rescore("origin/main", self.pins, self.log, top=1, runs=1)
        sha = __import__("hashlib").sha256(b"final-1.golden.json").hexdigest()
        infra = lambda: {"metrics": {"baseline_golden_sha256": sha}}

        def fail_slot(outcome):
            row = self.r.db.execute("SELECT id FROM submissions WHERE rescore='final' AND team='A' "
                                    "AND state='queued' ORDER BY created DESC LIMIT 1").fetchone()
            self.r._update(row["id"], state="done", passed=0, outcome=outcome, result=json.dumps(infra()))

        row0 = self.r.db.execute("SELECT id FROM submissions WHERE rescore='final'").fetchone()
        self.r._update(row0["id"], state="done", passed=0, outcome="control_failed", result=json.dumps(infra()))
        self.r.rescore_retry("final", max_infra_retries=2)
        fail_slot("band_refused")
        self.r.rescore_retry("final", max_infra_retries=2)
        fail_slot("control_failed")
        summary = self.r.rescore_report(self.t / "cap", max_infra_retries=2)
        slot = summary["ranking"][0]["slots"][0]
        self.assertEqual(slot["status"], "penalized")
        self.assertEqual(slot["value"], 0.0)
        self.assertEqual(summary["ranking"][0]["status"], "complete")
        self.assertEqual(summary["ranking"][0]["mean"], 0.0)
        self.assertEqual(self.r.rescore_retry("final", max_infra_retries=2)["redispatched"], [])

    def test_rescore_retry_cap_leaves_pre_setup_infra_unresolved(self):
        self.sub("A", "a", 1.0)
        self.r.freeze()
        self.log.write_text(f"{time.time() + 1} final box-1 final-1.golden.json {'b' * 64}\n")
        self.r.rescore("origin/main", self.pins, self.log, top=1, runs=1)
        sha = __import__("hashlib").sha256(b"final-1.golden.json").hexdigest()
        infra = lambda: {"metrics": {"baseline_golden_sha256": sha}}

        def fail_slot():
            row = self.r.db.execute("SELECT id FROM submissions WHERE rescore='final' AND team='A' "
                                    "AND state='queued' ORDER BY created DESC LIMIT 1").fetchone()
            self.r._update(row["id"], state="done", passed=0, outcome="preflight_refused", result=json.dumps(infra()))

        row0 = self.r.db.execute("SELECT id FROM submissions WHERE rescore='final'").fetchone()
        self.r._update(row0["id"], state="done", passed=0, outcome="infra_failed", result=json.dumps(infra()))
        self.r.rescore_retry("final", max_infra_retries=2)
        fail_slot()
        self.r.rescore_retry("final", max_infra_retries=2)
        fail_slot()
        summary = self.r.rescore_report(self.t / "cap-pre", max_infra_retries=2)
        slot = summary["ranking"][0]["slots"][0]
        self.assertEqual(slot["status"], "unresolved")
        self.assertEqual(summary["ranking"][0]["status"], "incomplete")

    def test_select_best_counts_boot_failed_as_zero(self):
        self.sub("A", "a", 1.0)
        sid = self.sub("A", "a", 9.0, passed=0)
        self.r._update(sid, state="done", passed=0, score=None, outcome="boot_failed")
        sel = self.r.select_best(10)
        self.assertAlmostEqual(sel[0]["contest_mean"], 0.5)

    def test_select_best_counts_post_setup_infra_as_zero(self):
        self.sub("A", "a", 1.0)
        sid = self.sub("A", "a", 9.0, passed=0)
        self.r._update(sid, state="done", passed=0, score=None, outcome="control_failed")
        sel = self.r.select_best(10)
        self.assertAlmostEqual(sel[0]["contest_mean"], 0.5)
        sid2 = self.sub("B", "b", 1.0)
        self.r._update(sid2, state="done", passed=0, score=None, outcome="band_refused")
        sel = self.r.select_best(10)
        by_team = {s["team"]: s for s in sel}
        self.assertAlmostEqual(by_team["A"]["contest_mean"], 0.5)
        self.assertAlmostEqual(by_team["B"]["contest_mean"], 0.0)

    def test_contest_and_rescore_treat_watchdog_timeout_as_team_failure(self):
        self.assertEqual(router.contest_run_value({"state": "done", "outcome": "watchdog_timeout"}), 0.0)
        row = {"id": "w1", "state": "done", "outcome": "cancelled",
               "incident": "watchdog: cancelled after 90 min", "_off_pool": False, "created": 1.0}
        self.assertEqual(router.rescore_run_kind(row, False), "team_failure")
        resolved = router.resolve_rescore_slot([row], 2)
        self.assertEqual(resolved["status"], "team_failure")
        self.assertEqual(resolved["value"], 0.0)


class ResolveRescoreSlotTest(unittest.TestCase):
    @staticmethod
    def _done(outcome: str, created: float, row_id: str) -> dict:
        return {"id": row_id, "state": "done", "outcome": outcome, "_off_pool": False, "created": created}

    def test_cap_penalized_if_any_attempt_post_setup_not_only_last(self):
        rows = [
            self._done("control_failed", 1.0, "a"),
            self._done("infra_failed", 2.0, "b"),
            self._done("infra_failed", 3.0, "c"),
        ]
        resolved = router.resolve_rescore_slot(rows, 2)
        self.assertEqual(resolved["status"], "penalized")
        self.assertEqual(resolved["value"], 0.0)
        self.assertEqual(resolved["outcome"], "infra_failed")

    def test_cap_unresolved_pre_setup_infra_only(self):
        rows = [
            self._done("preflight_refused", 1.0, "a"),
            self._done("infra_failed", 2.0, "b"),
            self._done("infra_failed", 3.0, "c"),
        ]
        resolved = router.resolve_rescore_slot(rows, 2)
        self.assertEqual(resolved["status"], "unresolved")
        self.assertIsNone(resolved["value"])

    def test_boot_failed_once_then_pass_scores(self):
        rows = [
            self._done("boot_failed", 1.0, "a"),
            {**self._done("pass", 2.0, "b"), "score": 1.7, "passed": 1},
        ]
        resolved = router.resolve_rescore_slot(rows, 2)
        self.assertEqual(resolved["status"], "scored")
        self.assertEqual(resolved["value"], 1.7)
        self.assertEqual(len(resolved["attempts"]), 2)

    def test_two_boot_failed_team_failure(self):
        rows = [
            self._done("boot_failed", 1.0, "a"),
            self._done("boot_failed", 2.0, "b"),
        ]
        resolved = router.resolve_rescore_slot(rows, 2)
        self.assertEqual(resolved["status"], "team_failure")
        self.assertEqual(resolved["value"], 0.0)

    def test_late_boot_failed_team_failure_without_retry(self):
        row = self._done("boot_failed", 1.0, "a")
        row["result"] = json.dumps({"metrics": {"paired_legs": [{"pair": 1}]}})
        row["_off_pool"] = True
        resolved = router.resolve_rescore_slot([row], 2)
        self.assertEqual(resolved["status"], "team_failure")
        self.assertEqual(resolved["value"], 0.0)
        self.assertEqual(resolved["run_id"], "a")

    def test_boot_failed_then_replay_rejected_team_failure(self):
        rows = [
            self._done("boot_failed", 1.0, "a"),
            self._done("replay_rejected", 2.0, "b"),
        ]
        resolved = router.resolve_rescore_slot(rows, 2)
        self.assertEqual(resolved["status"], "team_failure")
        self.assertEqual(resolved["value"], 0.0)
        self.assertEqual(resolved["outcome"], "replay_rejected")

    def test_teardown_failed_is_not_redispatched(self):
        row = self._done("teardown_failed", 1.0, "a")
        self.assertEqual(router.rescore_run_kind(row, False), "team_failure")
        self.assertEqual(router.rescore_run_kind(row, True), "team_failure")
        resolved = router.resolve_rescore_slot([row], 2)
        self.assertEqual(resolved["status"], "team_failure")
        self.assertEqual(resolved["value"], 0.0)


def uuid_hex():
    return __import__("uuid").uuid4().hex[:8]

class ClassifyTest(unittest.TestCase):
    def test_outcomes_and_expectations(self):
        c = router.classify
        self.assertEqual(c("success", 1, "", "", True), "pass")
        self.assertEqual(c("failure", 0, "pair 1 of 2: SERIAL-CONTROL-LEG-OUTSIDE-BAND: x", "", True), "band_refused")
        self.assertEqual(c("failure", 0, "SERIAL-CONTROL-LEG-FAILED: benchmark free-run decode token mismatch at step 47",
                           "", True), "control_failed")
        boot = "LEG-SERVE-BOOT-FAILED: the candidate leg's resident did not boot"
        self.assertEqual(c("failure", 0, boot, "", True), "boot_failed")
        ctl_boot = "LEG-SERVE-BOOT-FAILED: the serial-control leg's resident did not boot (1): abcd"
        self.assertEqual(c("failure", 0, ctl_boot, "", True), "control_failed")
        self.assertEqual(c("failure", 0, ctl_boot, "", True, paired_legs=[]), "control_failed")
        self.assertEqual(
            c("failure", 0, ctl_boot, "", True, paired_legs=[{"pair": 1}]),
            "refused",
        )
        self.assertEqual(c("failure", 0, "LEG-SERVE-BOOT-FAILED: the control leg's resident did not boot", "", True),
                         "refused")
        self.assertEqual(c("failure", 0, "candidate: SERIAL-CONTROL-LEG-FAILED token mismatch", "", True),
                         "correctness_failed")
        self.assertEqual(c("failure", 0, "TIMED-REPLAY-REJECTED: pair 1 ...", "", True), "replay_rejected")
        self.assertEqual(c("failure", 0, "performance floor failed: decode_speedup=0.9", "", True), "floor_failed")
        self.assertEqual(c("failure", 0, "LEG-TEARDOWN-FAILED: the candidate leg stop failed", "", True),
                         "teardown_failed")
        self.assertEqual(c("failure", None, "", "editable-surface + static review / Submission static review", False),
                         "surface_refused")
        self.assertEqual(c("failure", None, "", "ranked measurement / Ranked box preflight (staged assets)", False),
                         "preflight_refused")
        self.assertEqual(c("cancelled", None, "", "", False), "cancelled")
        self.assertEqual(c("cancelled", None, "", "", False, False, "watchdog: cancelled after 90 min"),
                         "watchdog_timeout")
        self.assertEqual(c("cancelled", 0, "", "", True, False, "watchdog: cancelled after 90 min"),
                         "watchdog_timeout")
        self.assertEqual(c("failure", None, "", "ranked measurement / Setup", False), "infra_failed")
        self.assertEqual(c("failure", None, "", "ranked measurement / Setup", False, True), "unsealed")
        self.assertTrue(router.meets("refused", "preflight_refused"))
        self.assertTrue(router.meets("refused", "unsealed"))
        self.assertTrue(router.meets("refused", "watchdog_timeout"))
        self.assertTrue(router.meets("refused", "teardown_failed"))
        self.assertFalse(router.meets("pass", "floor_failed"))
        self.assertTrue(router.meets("measure", "floor_failed"))
        self.assertIsNone(router.meets("pass", None))


class ScoreFromLogTest(unittest.TestCase):
    TS = "2026-09-27T07:21:57.1Z"

    @classmethod
    def _gh(cls, text: str) -> str:
        return f"{cls.TS} {text}"

    @classmethod
    def _sealed(cls, nonce: str, raw: bytes, *, extra_newline: bool = False) -> str:
        h = hashlib.sha256(raw).hexdigest()
        n = len(raw)
        lines = [cls._gh(f"NEMOTRON35-SCORE-BEGIN {nonce}")]
        if raw:
            text = raw.decode()
            for ln in text.splitlines():
                lines.append(cls._gh(ln))
            if text.endswith("\n"):
                pass
            elif extra_newline:
                lines.append(cls._gh(""))
        end = cls._gh(f"NEMOTRON35-SCORE-END {nonce} sha256={h} bytes={n}")
        return "\n".join(lines + [end])

    def test_accepts_real_reveal(self):
        nonce = "a" * 64
        raw = b'{\n  "metrics": {"error": ""},\n  "passed": true,\n  "score": 0.99\n}'
        log = self._sealed(nonce, raw)
        self.assertEqual(router.score_from_log(log)["score"], 0.99)

    def test_score_without_trailing_newline(self):
        nonce = "b" * 64
        raw = b'{"metrics": {"error": ""}, "passed": true, "score": 0.42}'
        log = self._sealed(nonce, raw, extra_newline=True)
        self.assertEqual(router.score_from_log(log)["score"], 0.42)

    def test_ignores_planted_marker_and_json(self):
        nonce = "c" * 64
        raw = b'{"metrics": {"error": ""}, "passed": true, "score": 0.7}'
        real = self._sealed(nonce, raw)
        planted = "\n".join([
            self._gh("benchd iterate: wrote score.json (passed=true, score=9.9)"),
            self._gh("{"),
            self._gh('  "metrics": {"error": ""}, "passed": true, "score": 9.9'),
            self._gh("}"),
            real,
        ])
        self.assertEqual(router.score_from_log(planted)["score"], 0.7)

    def test_ignores_earlier_fake_nonce_block(self):
        good = "d" * 64
        bad = "e" * 64
        raw = b'{"metrics": {"error": ""}, "passed": true, "score": 0.5}'
        fake_raw = b'{"metrics": {"error": ""}, "passed": true, "score": 9.9}'
        log = self._sealed(bad, fake_raw) + "\n" + self._sealed(good, raw)
        self.assertEqual(router.score_from_log(log)["score"], 0.5)

    def test_last_end_wins_over_earlier_valid_sealed_block(self):
        real_nonce = "a" * 64
        plant_nonce = "b" * 64
        real_raw = b'{"metrics": {"error": ""}, "passed": true, "score": 0.5}'
        plant_raw = b'{"metrics": {"error": ""}, "passed": true, "score": 9.9}'
        log = self._sealed(plant_nonce, plant_raw) + "\n" + self._sealed(real_nonce, real_raw)
        self.assertEqual(router.score_from_log(log)["score"], 0.5)

    def test_bad_sha256_returns_none(self):
        nonce = "f" * 64
        raw = b'{"metrics": {"error": ""}, "passed": true, "score": 0.1}'
        lines = [self._gh(f"NEMOTRON35-SCORE-BEGIN {nonce}")]
        for ln in raw.decode().splitlines():
            lines.append(self._gh(ln))
        lines.append(self._gh(f"NEMOTRON35-SCORE-END {nonce} sha256={'0' * 64} bytes={len(raw)}"))
        self.assertIsNone(router.score_from_log("\n".join(lines)))

    def test_none_end_returns_empty_dict(self):
        nonce = "0" * 64
        log = "\n".join([
            self._gh(f"NEMOTRON35-SCORE-BEGIN {nonce}"),
            self._gh(f"NEMOTRON35-SCORE-END {nonce} none"),
        ])
        self.assertEqual(router.score_from_log(log), {})

    def test_teardown_failed_with_matching_nonce(self):
        nonce = "1" * 64
        log = "\n".join([
            self._gh(f"NEMOTRON35-SCORE-BEGIN {nonce}"),
            self._gh(f"NEMOTRON35-TEARDOWN-FAILED {nonce}"),
            self._gh(f"NEMOTRON35-SCORE-END {nonce} none"),
        ])
        self.assertEqual(router.score_from_log(log), "teardown_failed")

    def test_teardown_failed_with_wrong_nonce_is_ignored(self):
        nonce = "2" * 64
        log = "\n".join([
            self._gh(f"NEMOTRON35-SCORE-BEGIN {nonce}"),
            self._gh(f"NEMOTRON35-TEARDOWN-FAILED {'3' * 64}"),
            self._gh(f"NEMOTRON35-SCORE-END {nonce} none"),
        ])
        self.assertEqual(router.score_from_log(log), {})

    def test_valid_sealed_block_then_final_end_none_parses_empty(self):
        nonce = "a" * 64
        other = "b" * 64
        raw = b'{"metrics": {"error": ""}, "passed": true, "score": 0.5}'
        for end_nonce in (nonce, other):
            with self.subTest(end_nonce=end_nonce):
                log = self._sealed(nonce, raw) + "\n" + self._gh(
                    f"NEMOTRON35-SCORE-END {end_nonce} none")
                self.assertEqual(router.score_from_log(log), {})

    def test_no_block_returns_none(self):
        self.assertIsNone(router.score_from_log(self._gh("nothing here")))


class NegBootFailVariantTest(unittest.TestCase):
    def test_apply_variant_and_expect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resident = root / "harness/vllm_resident/resident.py"
            resident.parent.mkdir(parents=True)
            shutil.copy(REPO_ROOT / "harness/vllm_resident/resident.py", resident)
            note = loadgen.apply_variant(root, "neg_boot_fail", random.Random(1))
            text = resident.read_text()
            self.assertIn("acceptance-test variant: refusing to boot", text)
            self.assertIn("return 1", text)
            self.assertLess(text.index("return 1", text.index("acceptance-test variant: refusing to boot")),
                            text.index("loading the model"))
            py_compile.compile(str(resident), doraise=True)
            self.assertEqual(
                note, "candidate resident exits at boot (exercises the pair-1 boot retry and teardown)")
        self.assertEqual(router.EXPECT["boot_failed"], {"boot_failed"})
        self.assertTrue(router.meets("boot_failed", "boot_failed"))


class VariantsApplyTest(unittest.TestCase):
    """Every load-test variant applies cleanly to the REAL editable tree at HEAD."""

    def test_all_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            tar = Path(tmp) / "t.tar"
            subprocess.run(["git", "archive", "-o", str(tar), "HEAD", "vllm", "harness", "mtp-head.manifest.json"],
                           cwd=REPO_ROOT, check=True)
            for _, name, _ in loadgen.MIX:
                root = Path(tmp) / name
                root.mkdir()
                with tarfile.open(tar) as tf:
                    tf.extractall(root, filter="tar")
                note = loadgen.apply_variant(root, name, random.Random(1))
                self.assertTrue(note, name)


def fake_cmd(dir: Path, body: str) -> str:
    """A fake gh/git: a shell script that logs each call to calls.log."""
    exe = dir / "fake"
    exe.write_text(f'#!/bin/sh\necho "$*" >> "{dir}/calls.log"\n' + body)
    exe.chmod(0o755)
    return str(exe)


def calls(dir: Path) -> list[str]:
    p = dir / "calls.log"
    return p.read_text().splitlines() if p.exists() else []


@mock.patch("router.time.sleep", lambda _s: None)
class RetryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def flaky(self, fails: int, msg: str) -> str:
        return fake_cmd(self.d, f'''n=$(wc -l < "{self.d}/calls.log")
if [ "$n" -le {fails} ]; then echo "{msg}" >&2; exit 128; fi
echo ok
''')

    def test_transient_reset_is_retried_until_success(self):
        exe = self.flaky(3, "fatal: read: connection reset by peer")
        self.assertEqual(router.sh([exe, "fetch"], retry=True), "ok\n")
        self.assertEqual(len(calls(self.d)), 4)

    def test_gives_up_after_four_attempts(self):
        exe = self.flaky(9, "fatal: expected flush after ref listing")
        with self.assertRaisesRegex(RuntimeError, "expected flush"):
            router.sh([exe, "fetch"], retry=True)
        self.assertEqual(len(calls(self.d)), 4)

    def test_deterministic_failure_is_not_retried(self):
        for msg in ("HTTP 404: Not Found (https://api.github.com/x)",
                    "git@github.com: Permission denied (publickey). fatal: Could not read from remote repository."):
            (self.d / "calls.log").unlink(missing_ok=True)
            with self.assertRaises(RuntimeError):
                router.sh([self.flaky(9, msg), "api"], retry=True)
            self.assertEqual(len(calls(self.d)), 1, msg)

    def test_no_retry_without_flag(self):
        with self.assertRaises(RuntimeError):
            router.sh([self.flaky(1, "Connection reset by peer"), "x"])
        self.assertEqual(len(calls(self.d)), 1)

    def gh_router(self, dispatch_body: str):
        # `run list` reports a run once the dispatch has landed (landed file exists).
        gh = fake_cmd(self.d, f'''case "$1 $2" in
"workflow run") {dispatch_body} ;;
"run list") if [ -e "{self.d}/landed" ]; then echo '[{{"databaseId": 7}}]'; else echo '[]'; fi ;;
esac
''')
        os.environ["ROUTER_REMOTE"] = str(self.d / "none.git")
        self.addCleanup(os.environ.pop, "ROUTER_REMOTE", None)
        r = router.Router(self.d / "state", "o/r", "b", gh=gh)
        self.addCleanup(r.db.close)
        r.db.execute("INSERT INTO submissions(id, branch, state, created) VALUES ('s1', 'sub/s1', 'queued', 0)")
        return r

    def dispatches(self):
        return [c for c in calls(self.d) if c.startswith("workflow run")]

    def test_dispatch_reset_after_landing_is_not_redispatched(self):
        r = self.gh_router(f'touch "{self.d}/landed"; echo "Connection reset by peer" >&2; exit 1')
        r.dispatch(r.db.execute("SELECT * FROM submissions").fetchone())
        self.assertEqual(len(self.dispatches()), 1)
        row = r.db.execute("SELECT state, run_id FROM submissions").fetchone()
        self.assertEqual((row["state"], row["run_id"]), ("dispatched", 7))

    def test_dispatch_reset_before_landing_is_redispatched_once(self):
        r = self.gh_router(f'''if [ -e "{self.d}/tried" ]; then touch "{self.d}/landed"; exit 0; fi
touch "{self.d}/tried"; echo "write: broken pipe" >&2; exit 1''')
        r.dispatch(r.db.execute("SELECT * FROM submissions").fetchone())
        self.assertEqual(len(self.dispatches()), 2)
        self.assertEqual(r.db.execute("SELECT run_id FROM submissions").fetchone()[0], 7)

    def test_dispatch_deterministic_failure_raises(self):
        r = self.gh_router('echo "HTTP 422: Workflow does not have workflow_dispatch trigger" >&2; exit 1')
        with self.assertRaisesRegex(RuntimeError, "422"):
            r.dispatch(r.db.execute("SELECT * FROM submissions").fetchone())
        self.assertEqual(len(self.dispatches()), 1)


if __name__ == "__main__":
    unittest.main()

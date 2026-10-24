"""End to end with no GPU: the real `cuda-engine` binary attached to the real
resident process (fake backend), driven over Engine Protocol v1 the way benchd
drives a leg: hello, free_decode_begin, free_decode_run, then a correctness
phase. It proves the Rust client and the Python resident agree on the wire and
that every depth commits exactly the serial stream.

Needs `cargo build --release` in harness/protocol-adapter first; skips if the
binary is absent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from backends import FakeBackend
from resident import Session
from test_resident import PROMPT, serial_stream  # noqa: E402

ENGINE = os.path.join(HERE, "..", "protocol-adapter", "target", "release", "cuda-engine")


@unittest.skipUnless(os.path.exists(ENGINE), "build harness/protocol-adapter --release first")
class AdapterEndToEnd(unittest.TestCase):
    def boot(self, depth: int) -> tuple[subprocess.Popen, str]:
        d = tempfile.mkdtemp()
        sock, ready = os.path.join(d, "r.sock"), os.path.join(d, "ready")
        self.log_path = os.path.join(d, "resident.log")
        self.log_file = open(self.log_path, "w")
        self.addCleanup(self.log_file.close)
        env = dict(
            os.environ,
            RESIDENT_BACKEND="fake",
            RESIDENT_SPEC_DEPTH=str(depth),
            RESIDENT_SOCKET=sock,
            RESIDENT_READY_FILE=ready,
        )
        proc = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "resident.py")], env=env, stderr=self.log_file
        )
        deadline = time.time() + 20
        while not os.path.exists(ready):
            if time.time() > deadline or proc.poll() is not None:
                self.fail("the resident never became ready")
            time.sleep(0.05)
        self.addCleanup(lambda: (proc.terminate(), proc.wait(timeout=10)))
        return proc, sock

    def phase(self, sock: str, requests: list[dict]) -> list[dict]:
        env = dict(os.environ, BENCH_WORKER_RESIDENT_SOCKET=sock)
        env.pop("DS4_RESIDENT_SOCKET", None)
        stdin = "".join(json.dumps(r) + "\n" for r in requests)
        out = subprocess.run(
            [ENGINE], input=stdin, capture_output=True, text=True, env=env, timeout=60
        )
        lines = [json.loads(l) for l in out.stdout.splitlines() if l.strip()]
        self.assertTrue(lines, out.stderr)
        return lines

    def test_every_depth_commits_the_serial_stream(self):
        want = serial_stream(128)
        for depth in range(0, 7):
            with self.subTest(depth=depth):
                _, sock = self.boot(depth)
                spec = {"mode": "serial"} if depth == 0 else {"mode": "mtp", "mtp": {"depth": depth}}
                replies = self.phase(
                    sock,
                    [
                        {"id": 1, "kind": "free_decode_begin", "seed_tokens": PROMPT, "spec": spec},
                        {"id": 2, "kind": "free_decode_run", "count": 128},
                    ],
                )
                hello, begin, run = replies
                self.assertTrue(hello["backend"].startswith("vllm-resident load_epoch="))
                self.assertTrue(begin["ok"], begin)
                self.assertTrue(run["ok"], run)
                self.assertEqual(run["tokens"], want)
                self.assertEqual(run["committed_total"], 128)
                self.assertEqual(sum(run["acceptance_lengths"]), 128)
                if depth:
                    self.assertEqual(begin["effective_spec"]["mtp"]["depth"], depth)
                    self.assertGreater(run["drafted_total"], 0)

    def resident_log(self) -> list[str]:
        self.log_file.flush()
        with open(self.log_path) as fp:
            return [l.strip() for l in fp]

    def test_nothing_runs_on_after_a_phase(self):
        """benchd keeps ONE worker for a whole leg, so the resident sees no new
        connection between phases. The phase close must stop the run-on decode
        itself, inside the same connection: a timed prefill leaves nothing
        running, and a free run is stopped at its phase_diagnostics."""
        _, sock = self.boot(3)
        spec = {"mode": "mtp", "mtp": {"depth": 3}}
        replies = self.phase(
            sock,
            [
                {"id": 1, "kind": "prefill", "prompt_tokens": PROMPT},
                {"id": 2, "kind": "prefill", "prompt_tokens": PROMPT},
                {"id": 3, "kind": "phase_diagnostics"},
                {"id": 4, "kind": "free_decode_begin", "seed_tokens": PROMPT, "spec": spec},
                {"id": 5, "kind": "free_decode_run", "count": 32},
                {"id": 6, "kind": "phase_diagnostics"},
            ],
        )
        self.assertTrue(all(r.get("ok", True) for r in replies[1:]), replies)
        self.assertEqual(replies[5]["tokens"], serial_stream(32))
        time.sleep(0.2)
        log = self.resident_log()
        stops = [i for i, l in enumerate(log) if "stopped a request" in l]
        closed = next(i for i, l in enumerate(log) if "phase 1 closed" in l)
        self.assertEqual(len(stops), 1, log)
        self.assertLess(stops[0], closed, "the free run is stopped inside its own connection")

    def test_correctness_phase_reports_top8(self):
        _, sock = self.boot(0)
        want = serial_stream(3)
        replies = self.phase(
            sock,
            [
                {"id": 1, "kind": "correctness_begin", "prompt_tokens": PROMPT},
                {"id": 2, "kind": "correctness_step", "token": None},
            ],
        )
        begin = replies[1]
        self.assertTrue(begin["ok"], begin)
        self.assertEqual(len(begin["top_logits"]), 8)
        step = self.phase(
            sock,
            [
                {"id": 1, "kind": "correctness_begin", "prompt_tokens": PROMPT},
                {"id": 2, "kind": "correctness_step", "token": begin["token"]},
            ],
        )[2]
        self.assertTrue(step["ok"], step)
        self.assertEqual(step["token"], want[0])

    def test_a_whole_leg_on_one_connection_keeps_the_gate_top8(self):
        """The leg's order on ONE worker: the timed prefill and the free run
        sync argmax-only (no top-k computed), then the correctness gate syncs
        with top-k and reports top-8 on every step."""
        _, sock = self.boot(3)
        spec = {"mode": "mtp", "mtp": {"depth": 3}}
        want = serial_stream(33)
        seed = Session(FakeBackend(depth=0)).sync(PROMPT)
        replies = self.phase(
            sock,
            [
                {"id": 1, "kind": "prefill", "prompt_tokens": PROMPT},
                {"id": 2, "kind": "prefill", "prompt_tokens": PROMPT},
                {"id": 3, "kind": "phase_diagnostics"},
                {"id": 4, "kind": "free_decode_begin", "seed_tokens": PROMPT, "spec": spec},
                {"id": 5, "kind": "free_decode_run", "count": 32},
                {"id": 6, "kind": "phase_diagnostics"},
                {"id": 7, "kind": "correctness", "prompt_tokens": PROMPT, "steps": 33},
                {"id": 8, "kind": "phase_diagnostics"},
                {"id": 9, "kind": "correctness_begin", "prompt_tokens": PROMPT},
                {"id": 10, "kind": "correctness_step", "token": seed},
                {"id": 11, "kind": "phase_diagnostics"},
            ],
        )
        self.assertTrue(all(r.get("ok", True) for r in replies[1:]), replies)
        self.assertEqual(replies[2]["token"], seed)
        self.assertEqual(replies[4]["seed_token"], seed)
        self.assertEqual(replies[5]["tokens"], want[:32])
        self.assertEqual(replies[7]["tokens"], [seed] + want[:32])
        self.assertEqual(len(replies[9]["top_logits"]), 8)
        self.assertEqual(replies[9]["top_logits"][0]["token"], seed)
        self.assertEqual(replies[10]["token"], want[0])
        self.assertEqual(len(replies[10]["top_logits"]), 8)


if __name__ == "__main__":
    unittest.main()

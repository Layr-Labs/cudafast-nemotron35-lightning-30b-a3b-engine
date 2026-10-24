"""GPU-free tests of the resident's bookkeeping and wire (FakeBackend).

Run: python3 -m unittest discover -s harness/vllm_resident -v
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backends import FakeBackend  # noqa: E402
from resident import IMPLEMENTED_DEPTH, RefusedError, Session  # noqa: E402

PROMPT = list(range(100, 164))


def serial_stream(n: int) -> list[int]:
    s = Session(FakeBackend(depth=0))
    out, t = [], s.sync(PROMPT)
    for _ in range(n):
        t = s.eval(t)
        out.append(t)
    return out


class SessionTests(unittest.TestCase):
    def test_serial_eval_advances_one_token(self):
        s = Session(FakeBackend(depth=0))
        first = s.sync(PROMPT)
        self.assertEqual(s.eval(first), s.frontier.token)
        self.assertEqual(s.ctx, PROMPT + [first])

    def test_spec_run_equals_the_serial_stream_at_every_depth(self):
        want = serial_stream(128)
        for depth in range(1, IMPLEMENTED_DEPTH + 1):
            s = Session(FakeBackend(depth=depth))
            seed = s.sync(PROMPT)
            rounds, frontiers = s.spec_run(seed, 128)
            got = []
            for r, f in zip(rounds, frontiers):
                got.extend(r[1:])
                got.append(f)
            self.assertEqual(got, want, f"depth {depth}")
            self.assertEqual(sum(len(r) for r in rounds), 128)
            self.assertTrue(any(len(r) > 1 for r in rounds), f"depth {depth} never accepted")
            self.assertGreater(s.drafts, 0)

    def test_a_free_run_sync_drops_top_logits_and_keeps_the_stream(self):
        want = serial_stream(64)
        for depth in (0, 3):
            backend = FakeBackend(depth=depth)
            s = Session(backend)
            seed = s.sync(PROMPT, top_logits=False)
            self.assertEqual(backend.logprob_starts, [False])
            if depth:
                rounds, frontiers = s.spec_run(seed, 64)
                got = []
                for r, f in zip(rounds, frontiers):
                    got.extend(r[1:])
                    got.append(f)
            else:
                got, t = [], seed
                for _ in range(64):
                    t = s.eval(t)
                    got.append(t)
            self.assertEqual(got, want, f"depth {depth}")
            with self.assertRaises(RefusedError):
                s.top_logits(8)
            # The next sequence asks for them again by default.
            s.invalidate()
            s.sync(PROMPT)
            self.assertEqual(backend.logprob_starts, [False, True])
            self.assertEqual(len(s.top_logits(8)[0]), 8)

    def test_resync_preserves_output_mode_for_eval_and_cycle(self):
        for mode in (False, True):
            for verb in ("eval", "cycle"):
                for bounded in (False, True):
                    with self.subTest(mode=mode, verb=verb, bounded=bounded):
                        backend = FakeBackend(depth=3)
                        s = Session(backend)
                        frontier = s.sync(PROMPT, max_tokens=1 if bounded else None,
                                          top_logits=mode)
                        fed = frontier if bounded else (frontier + 1) % backend.vocab_size
                        if verb == "eval":
                            s.eval(fed)
                        else:
                            self.assertEqual(s.cycle(fed, 4), [fed])
                        self.assertEqual(backend.logprob_starts, [mode, mode])
                        ref = Session(FakeBackend(depth=0))
                        self.assertEqual(s.frontier.token, ref.sync(PROMPT + [fed]))
                        if mode:
                            self.assertEqual(len(s.top_logits(8)[0]), 8)
                        else:
                            with self.assertRaises(RefusedError):
                                s.top_logits(8)
                        s.invalidate()
                        s.sync(PROMPT)
                        self.assertEqual(len(s.top_logits(8)[0]), 8)

    def test_the_skip_logprobs_switch_keeps_top_logits(self):
        import resident

        backend = FakeBackend(depth=3)
        s = Session(backend)
        old, resident.SKIP_LOGPROBS = resident.SKIP_LOGPROBS, False
        try:
            s.sync(PROMPT, top_logits=False)
        finally:
            resident.SKIP_LOGPROBS = old
        self.assertEqual(backend.logprob_starts, [True])
        self.assertEqual(len(s.top_logits(8)[0]), 8)

    def test_spec_run_never_overshoots_the_count(self):
        for count in (1, 2, 3, 7, 129):
            s = Session(FakeBackend(depth=6))
            seed = s.sync(PROMPT)
            rounds, _ = s.spec_run(seed, count)
            self.assertEqual(sum(len(r) for r in rounds), count)

    def test_overhang_is_consumed_by_the_next_run(self):
        want = serial_stream(20)
        s = Session(FakeBackend(depth=6))
        seed = s.sync(PROMPT)
        got = []
        pending = seed
        for count in (3, 5, 12):
            rounds, frontiers = s.spec_run(pending, count)
            for r, f in zip(rounds, frontiers):
                got.extend(r[1:])
                got.append(f)
            pending = frontiers[-1]
        self.assertEqual(got, want)

    def test_teacher_forcing_reprefills(self):
        backend = FakeBackend(depth=0)
        s = Session(backend)
        first = s.sync(PROMPT)
        forced = (first + 1) % 1000 + 20
        s.eval(forced)
        self.assertEqual(s.ctx, PROMPT + [forced])
        self.assertEqual(backend.starts, 2)

    def test_a_bounded_sync_leaves_the_engine_idle(self):
        backend = FakeBackend(depth=0)
        s = Session(backend)
        s.sync(PROMPT, max_tokens=1)
        self.assertFalse(backend.abort(), "a 1-token request must have finished on its own")

    def test_an_unbounded_sync_is_stopped_by_invalidate(self):
        s = Session(FakeBackend(depth=3))
        seed = s.sync(PROMPT)
        s.spec_run(seed, 16)
        s.invalidate()
        self.assertEqual(s.stopped, 1, "the run-on request is stopped and counted")
        s.sync(PROMPT, max_tokens=1)
        s.invalidate()
        self.assertEqual(s.stopped, 1, "a finished request needs no stop")

    def test_a_bounded_sequence_that_goes_on_is_still_the_serial_stream(self):
        want = serial_stream(40)
        s = Session(FakeBackend(depth=0))
        t, got = s.sync(PROMPT, max_tokens=1), []
        for _ in range(40):
            t = s.eval(t)
            got.append(t)
        self.assertEqual(got, want)
        for depth in range(1, IMPLEMENTED_DEPTH + 1):
            s = Session(FakeBackend(depth=depth))
            seed = s.sync(PROMPT, max_tokens=1)
            rounds, frontiers = s.spec_run(seed, 40)
            got = []
            for r, f in zip(rounds, frontiers):
                got.extend(r[1:])
                got.append(f)
            self.assertEqual(got, want, f"depth {depth}")

    def test_a_non_positive_bound_is_refused(self):
        s = Session(FakeBackend(depth=0))
        with self.assertRaises(RefusedError):
            s.sync(PROMPT, max_tokens=0)

    def test_top_logits_are_the_frontiers(self):
        s = Session(FakeBackend(depth=0))
        first = s.sync(PROMPT)
        ids, logits = s.top_logits(8)
        self.assertEqual(ids[0], first)
        self.assertEqual(len(ids), 8)
        self.assertEqual(logits, sorted(logits, reverse=True))

    def test_verbs_refuse_without_a_sequence(self):
        s = Session(FakeBackend(depth=0))
        with self.assertRaises(RefusedError):
            s.eval(5)
        with self.assertRaises(RefusedError):
            s.top_logits(8)

    def test_out_of_vocab_is_refused(self):
        s = Session(FakeBackend(depth=0))
        with self.assertRaises(RefusedError):
            s.sync([1, 2, 131072])

    def test_hello_reports_depth_in_the_reference_convention(self):
        self.assertEqual(Session(FakeBackend(depth=0)).hello()["draft_tokens"], 1)
        h = Session(FakeBackend(depth=3)).hello()
        self.assertEqual(h["draft_tokens"], 4)
        self.assertTrue(h["mtp_armed"])

    def test_implemented_depth_agrees_with_the_adapter(self):
        src = os.path.join(
            os.path.dirname(__file__), "..", "protocol-adapter", "src", "session_backend.rs"
        )
        with open(src) as fp:
            line = next(l for l in fp if l.startswith("pub const ENGINE_IMPLEMENTED_DEPTH"))
        self.assertEqual(int(line.split("=")[1].strip().rstrip(";")), IMPLEMENTED_DEPTH)


class WireTests(unittest.TestCase):
    """Boots the real resident process on the fake backend and talks NDJSON."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.sock = os.path.join(self.dir, "r.sock")
        self.ready = os.path.join(self.dir, "ready")
        env = dict(
            os.environ,
            RESIDENT_BACKEND="fake",
            RESIDENT_SPEC_DEPTH="2",
            RESIDENT_SOCKET=self.sock,
            RESIDENT_READY_FILE=self.ready,
        )
        self.proc = subprocess.Popen(
            [sys.executable, os.path.join(os.path.dirname(__file__), "resident.py")],
            env=env,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 20
        while not os.path.exists(self.ready):
            if time.time() > deadline or self.proc.poll() is not None:
                self.fail("the resident never became ready")
            time.sleep(0.05)

    def tearDown(self):
        self.proc.terminate()
        self.proc.wait(timeout=10)

    def call(self, conn, reader, obj):
        conn.sendall((json.dumps(obj) + "\n").encode())
        return json.loads(reader.readline())

    def test_phase_round_trip_and_reset(self):
        for _phase in range(2):
            conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            conn.connect(self.sock)
            reader = conn.makefile("rb")
            hello = self.call(conn, reader, {"op": "hello"})
            self.assertTrue(hello["ok"])
            self.assertEqual(hello["draft_tokens"], 3)
            # A fresh phase has no live sequence.
            self.assertFalse(self.call(conn, reader, {"op": "argmax"})["ok"])
            seed = self.call(conn, reader, {"op": "sync", "tokens": PROMPT})["token"]
            run = self.call(conn, reader, {"op": "spec_run", "first_token": seed, "count": 32})
            self.assertTrue(run["ok"])
            self.assertEqual(sum(len(r) for r in run["rounds"]), 32)
            self.assertEqual(run["frontiers"][-1], run["token"])
            top = self.call(conn, reader, {"op": "top_logits", "k": 8})
            self.assertEqual(len(top["ids"]), 8)
            free = self.call(conn, reader, {"op": "sync", "tokens": PROMPT, "top_logits": False})
            self.assertEqual(free["token"], seed)
            self.assertFalse(self.call(conn, reader, {"op": "top_logits", "k": 8})["ok"])
            self.assertFalse(
                self.call(conn, reader, {"op": "sync", "tokens": PROMPT, "top_logits": 0})["ok"]
            )
            bounded = self.call(conn, reader, {"op": "sync", "tokens": PROMPT, "max_tokens": 1})
            self.assertEqual(bounded["token"], seed)
            self.assertFalse(
                self.call(conn, reader, {"op": "sync", "tokens": PROMPT, "max_tokens": 0})["ok"]
            )
            bad = self.call(conn, reader, {"op": "nope"})
            self.assertFalse(bad["ok"])
            self.assertTrue(self.call(conn, reader, {"op": "bye"})["ok"])
            reader.close()
            conn.close()


if __name__ == "__main__":
    unittest.main()

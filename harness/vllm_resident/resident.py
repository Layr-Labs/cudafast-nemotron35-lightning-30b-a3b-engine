"""The resident: the ONE process that holds the model for a benchmark window.

benchd spawns a fresh `cuda-engine` worker per phase (warmup, timed prefill,
timed decode, correctness). Loading the checkpoint in each would put a model
load between every phase, so the weights live here instead, loaded once, and
each phase's worker connects over a Unix socket (`src/resident.rs`).

THE WIRE is the reference track's resident wire, verb for verb, so the Rust
client is unchanged: NDJSON over AF_UNIX, one request per line, one reply per
line, every reply carrying "ok" (and "error" when false).

  hello                          identity, vocab, EOS, depth, load epoch (ms)
  invalidate                     drop the live request
  sync {tokens[,max_tokens][,argmax_only]}
                                 prefill; reply carries the frontier argmax.
                                 max_tokens bounds what the engine generates
                                 for this sequence (a timed prefill asks for 1).
                                 argmax_only: true says the caller reads only
                                 the greedy tokens of this sequence (timed
                                 prefill, free run), so no top-k logits are
                                 computed for it and top_logits is refused on
                                 it. Absent or false: every position carries
                                 its top-k, as before
  eval {token}                   append one token; reply carries the argmax
  argmax                         the frontier token
  top_logits {k}                 the frontier position's top-k raw logits
                                 (refused on an argmax_only sequence)
  eval_speculative {first_token,budget}
                                 one verify cycle; committed tokens + argmax
  spec_run {first_token,count}   cycles until `count` tokens are committed,
                                 in ONE reply (rounds + frontiers)
  spec_counters                  drafts / hits / quenches / disagreements
  bye                            acknowledge, then close

THE BOOKKEEPING. The engine emits tokens a step at a time and a speculative
step can emit several. The session keeps `ctx` (tokens the model has
consumed), `frontier` (the greedy next token, not yet consumed) and `ahead`
(tokens the engine already emitted past the frontier). A token in `ahead` is
the target's own greedy choice, so feeding the frontier consumes it without a
forward pass. A fed token that is NOT the frontier (teacher forcing) re-prefills
`ctx + [token]`; that path is correct and deliberately not fast.

PHASE RESET. Every accepted connection starts from an invalidated session, so a
phase can never inherit the previous phase's state.

TOP-K ONLY WHERE IT IS READ. The adapter asks for `top_logits` in the
teacher-forced gates only (`correctness_begin` / `correctness_step`, and
`decode_step`, which shares their forward). The timed prefill, the free run
and the free-run correctness reference read nothing but the argmax, so their
syncs say `argmax_only` and vLLM skips the top-k over the vocabulary on every
step of them. A sequence keeps its mode until the next sync or invalidate: a
teacher-forced re-prefill inside it inherits the mode, so a gate sequence
never loses its top-k. The flag changes no token: greedy sampling never reads
the logprobs.

NOTHING RUNS ON AFTER A PHASE. vLLM's engine core steps a live request until it
finishes, whether or not anyone reads the tokens. benchd holds ONE connection
for a whole leg, so an accept never comes between its phases. Two things end
the request when the phase has what it asked for: the adapter sends
`invalidate` when it closes each phase, and a timed prefill syncs with
`max_tokens: 1`, so its request finishes at the frontier token. Without them the
last request of a phase decoded on to the context limit (about 7,700 tokens,
about 100 s on GB10) and kept the GPU busy through benchd's quiescence gate.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import time
from collections import deque

try:
    from .backends import Backend, Emit, FakeBackend, VllmBackend, VllmConfig
except ImportError:  # run as a script
    from backends import Backend, Emit, FakeBackend, VllmBackend, VllmConfig  # type: ignore

# The depth the resident implements. MUST equal ENGINE_IMPLEMENTED_DEPTH in
# harness/protocol-adapter/src/session_backend.rs.
IMPLEMENTED_DEPTH = 6
MAX_TOP_K = 64
MAX_RUN = 65536
# The load epoch: milliseconds since the Unix epoch when this process loaded
# the model. Every phase of one leg reports the same value, and two legs never
# share one. (A pid would not do: inside the container every resident is pid 7.)
LOAD_EPOCH = int(time.time() * 1000)


def log(msg: str) -> None:
    print(f"resident: {msg}", file=sys.stderr, flush=True)


class RefusedError(Exception):
    """A request the resident refuses; becomes an ok:false reply."""


class Session:
    def __init__(self, backend: Backend, spec_run: bool = True):
        self.b = backend
        self.spec_run_offered = spec_run
        self.drafts = 0
        self.hits = 0
        # Tokens the live request may still emit; None = unbounded.
        self.left: int | None = None
        # Invalidates that found the engine still generating.
        self.stopped = 0
        self.invalidate()

    # ---- state ---------------------------------------------------------- #

    def invalidate(self) -> None:
        if self.b.abort():
            self.stopped += 1
            log(f"stopped a request that was still generating ({self.stopped} so far)")
        self.left = None
        self.ctx: list[int] = []
        self.frontier: Emit | None = None
        self.ahead: deque[Emit] = deque()
        # The live sequence computes no top-k logits (sync argmax_only).
        self.argmax_only = False

    def _need_frontier(self) -> Emit:
        if self.frontier is None:
            raise RefusedError("no live sequence; sync first")
        return self.frontier

    def _engine_step(self) -> list[Emit]:
        emitted = self.b.step()
        if not emitted:
            raise RefusedError("the engine step emitted nothing")
        if self.b.depth > 0:
            self.drafts += self.b.depth
            self.hits += len(emitted) - 1
        if self.left is not None:
            self.left -= len(emitted)
        return emitted

    def _spent(self) -> bool:
        """The bounded request emitted all it may; the engine has stopped."""
        return not self.ahead and self.left is not None and self.left <= 0

    def sync(
        self, tokens: list[int], max_tokens: int | None = None, argmax_only: bool = False
    ) -> int:
        if not tokens:
            raise RefusedError("sync needs a non-empty token list")
        if max_tokens is not None and max_tokens < 1:
            raise RefusedError("sync max_tokens must be positive")
        self._check_ids(tokens)
        emitted = self.b.start(tokens, max_tokens, top_logits=not argmax_only)
        self.argmax_only = argmax_only
        self.left = None if max_tokens is None else max_tokens - len(emitted)
        self.ctx = list(tokens)
        self.frontier = emitted[0]
        self.ahead = deque(emitted[1:])
        return self.frontier.token

    def _feed(self, token: int) -> None:
        """Consume `token` at the frontier and advance to the next one."""
        frontier = self._need_frontier()
        if token != frontier.token or self._spent():
            # Teacher forcing, or a bounded sequence asked to go on: prefill
            # again from the context. Correct, and deliberately not fast. The
            # sequence keeps its top-k mode.
            self.sync(self.ctx + [token], argmax_only=self.argmax_only)
            return
        self.ctx.append(token)
        if not self.ahead:
            self.ahead.extend(self._engine_step())
        self.frontier = self.ahead.popleft()

    def eval(self, token: int) -> int:
        self._check_ids([token])
        self._feed(token)
        return self.frontier.token  # type: ignore[union-attr]

    def cycle(self, fed: int, budget: int) -> list[int]:
        """One verify cycle on `fed` with `budget` tokens still wanted.

        Returns the committed tokens, `fed` first. The frontier it leaves is
        the token the next cycle is fed. `len(committed)` tokens are produced:
        every committed token after `fed`, plus the new frontier.
        """
        if budget <= 0:
            raise RefusedError("a cycle needs a positive budget")
        self._check_ids([fed])
        frontier = self._need_frontier()
        if fed != frontier.token or self._spent():
            self.sync(self.ctx + [fed], argmax_only=self.argmax_only)
            return [fed]
        if not self.ahead:
            self.ahead.extend(self._engine_step())
        m = min(len(self.ahead), budget)
        taken = [self.ahead.popleft() for _ in range(m)]
        committed = [fed] + [e.token for e in taken[:-1]]
        self.ctx.extend(committed)
        self.frontier = taken[-1]
        return committed

    def spec_run(self, first: int, count: int) -> tuple[list[list[int]], list[int]]:
        rounds: list[list[int]] = []
        frontiers: list[int] = []
        pending, produced = first, 0
        while produced < count:
            committed = self.cycle(pending, count - produced)
            rounds.append(committed)
            pending = self.frontier.token  # type: ignore[union-attr]
            frontiers.append(pending)
            produced += len(committed)
        return rounds, frontiers

    def top_logits(self, k: int) -> tuple[list[int], list[float]]:
        top = self._need_frontier().top
        if self.argmax_only:
            raise RefusedError(
                "the live sequence was synced argmax_only and computed no top logits; "
                "sync it without argmax_only to read them"
            )
        if len(top) < k:
            raise RefusedError(f"the engine returned {len(top)} top logits, {k} were asked")
        return [t for t, _ in top[:k]], [v for _, v in top[:k]]

    def _check_ids(self, tokens: list[int]) -> None:
        for t in tokens:
            if not isinstance(t, int) or isinstance(t, bool) or not 0 <= t < self.b.vocab_size:
                raise RefusedError(f"token {t!r} is outside the vocabulary")

    # ---- wire ----------------------------------------------------------- #

    def hello(self) -> dict:
        return {
            "vocab_size": self.b.vocab_size,
            "eos_token": self.b.eos_token,
            "mtp_armed": self.b.depth > 0,
            # The reference wire's convention: 1 = serial, N + 1 = depth N.
            "draft_tokens": self.b.depth + 1,
            "ctx_size": self.b.ctx_size,
            "load_epoch": LOAD_EPOCH,
            "spec_run": self.spec_run_offered,
            "ident": self.b.ident,
            "model_path": self.b.model_path,
            # The native MTP layer is embedded in the pinned checkpoint.
            "mtp_head_path": f"{self.b.model_path}#mtp.*" if self.b.depth > 0 else "",
        }

    def handle(self, req: dict) -> tuple[dict, bool]:
        """Serve one request. Returns (reply, keep_connection)."""
        op = req.get("op")
        if not isinstance(op, str):
            raise RefusedError('request carries no "op" string')

        def int_field(name: str) -> int:
            v = req.get(name)
            if not isinstance(v, int) or isinstance(v, bool):
                raise RefusedError(f'{op} needs a "{name}" integer')
            return v

        if op == "hello":
            return {"ok": True, **self.hello()}, True
        if op == "invalidate":
            self.invalidate()
            return {"ok": True}, True
        if op == "sync":
            tokens = req.get("tokens")
            if not isinstance(tokens, list):
                raise RefusedError('sync needs a "tokens" integer array')
            bound = None
            if "max_tokens" in req:
                bound = int_field("max_tokens")
            argmax_only = req.get("argmax_only", False)
            if not isinstance(argmax_only, bool):
                raise RefusedError('sync "argmax_only" must be a boolean')
            return {"ok": True, "token": self.sync(tokens, bound, argmax_only)}, True
        if op == "eval":
            return {"ok": True, "token": self.eval(int_field("token"))}, True
        if op == "argmax":
            return {"ok": True, "token": self._need_frontier().token}, True
        if op == "top_logits":
            k = int_field("k")
            if not 1 <= k <= MAX_TOP_K:
                raise RefusedError(f'top_logits needs a "k" integer in 1..{MAX_TOP_K}')
            ids, logits = self.top_logits(k)
            return {"ok": True, "ids": ids, "logits": logits}, True
        if op == "eval_speculative":
            committed = self.cycle(int_field("first_token"), int_field("budget"))
            return {"ok": True, "tokens": committed, "token": self.frontier.token}, True  # type: ignore[union-attr]
        if op == "spec_run":
            if not self.spec_run_offered:
                raise RefusedError("spec_run is not offered (RESIDENT_NO_SPEC_RUN)")
            count = int_field("count")
            if not 1 <= count <= MAX_RUN:
                raise RefusedError(f'spec_run needs a "count" in 1..{MAX_RUN}')
            rounds, frontiers = self.spec_run(int_field("first_token"), count)
            return {
                "ok": True,
                "rounds": rounds,
                "frontiers": frontiers,
                "token": self.frontier.token,  # type: ignore[union-attr]
            }, True
        if op == "spec_counters":
            return {
                "ok": True,
                "drafts": self.drafts,
                "hits": self.hits,
                "quenches": 0,
                "disagreements": 0,
            }, True
        if op == "bye":
            return {"ok": True}, False
        raise RefusedError(f'unknown op "{op}"')


def serve_connection(session: Session, conn: socket.socket, timeout_s: float) -> None:
    conn.settimeout(timeout_s)
    reader = conn.makefile("rb")
    try:
        for raw in reader:
            try:
                req = json.loads(raw)
                if not isinstance(req, dict):
                    raise RefusedError("a request must be a JSON object")
                reply, keep = session.handle(req)
            except (RefusedError, ValueError, RuntimeError) as err:
                reply, keep = {"ok": False, "error": str(err)}, True
            conn.sendall((json.dumps(reply, separators=(",", ":")) + "\n").encode())
            if not keep:
                return
    except (socket.timeout, ConnectionError, OSError) as err:
        log(f"phase dropped: {err}")
    finally:
        reader.close()


def make_backend() -> Backend:
    kind = os.environ.get("RESIDENT_BACKEND", "vllm")
    if kind == "vllm":
        return VllmBackend(VllmConfig.from_env())
    if kind == "fake":
        return FakeBackend(depth=int(os.environ.get("RESIDENT_SPEC_DEPTH", "0")))
    raise RuntimeError(f"RESIDENT_BACKEND={kind!r} is neither vllm nor fake")


def main() -> int:
    socket_path = os.environ.get("RESIDENT_SOCKET", "")
    if not socket_path:
        log("RESIDENT_SOCKET is unset")
        return 1
    ready_file = os.environ.get("RESIDENT_READY_FILE", "")
    timeout_s = float(os.environ.get("RESIDENT_PHASE_TIMEOUT_S", "1800"))
    spec_run = os.environ.get("RESIDENT_NO_SPEC_RUN", "0") != "1"

    stop = {"flag": False}

    def on_signal(_sig, _frame):
        stop["flag"] = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_signal)

    # THE ONE LOAD. Everything after this is a socket connect.
    log("loading the model")
    backend = make_backend()
    log(f"model loaded once for this window ({backend.ident}, depth={backend.depth})")
    session = Session(backend, spec_run=spec_run)

    # Bind only after the load, so a connect that succeeds means the weights
    # are resident and a phase never waits on the loader.
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(socket_path)
    os.chmod(socket_path, 0o600)
    listener.listen(8)
    if ready_file:
        with open(ready_file, "w") as fp:
            fp.write(socket_path + "\n")
    log(f"listening on {socket_path}; spec_run {'on' if spec_run else 'off'}")

    phase = 0
    try:
        while not stop["flag"]:
            conn, _ = listener.accept()
            phase += 1
            session.invalidate()
            log(f"phase {phase} connected")
            with conn:
                serve_connection(session, conn, timeout_s)
            log(f"phase {phase} closed")
    except KeyboardInterrupt:
        pass
    finally:
        log(f"shutting down after {phase} phase(s) on one load")
        listener.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        if ready_file and os.path.exists(ready_file):
            os.unlink(ready_file)
        backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

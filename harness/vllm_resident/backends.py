"""Generation backends the resident drives.

A backend owns ONE live greedy request at a time and exposes it as a stream of
engine steps. `start(tokens)` prefills `tokens` and returns the first step's
emitted tokens; `step()` runs one more engine step (one verify cycle when
speculation is on) and returns what it emitted; `abort()` drops the request.
`start(tokens, max_tokens)` bounds the request: the engine emits at most
`max_tokens` tokens and then stops on its own, so nothing decodes on after the
caller has what it asked for.

Every emitted token carries the top-k RAW LOGITS of the position that produced
it (highest first), so the resident can answer `top_logits` for whatever
frontier it holds without another forward pass. `start(tokens, max_tokens,
logprobs=False)` opens a request whose tokens carry none (`top` is empty): a
free run reads only tokens, so the engine skips the top-k work for it.

`VllmBackend` is the scored engine. `FakeBackend` is a deterministic stand-in
with the same step semantics (including multi-token speculative steps), so the
resident's wire and bookkeeping are tested with no GPU and no vLLM install.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

# DETERMINISM (measured on GB10, docs/nemotron35-port-notes.md "Findings").
# With vLLM's default kernels the same prompt gives different logits run to
# run: the Marlin NVFP4 MoE is nondeterministic even on a fixed input, and this
# checkpoint's FP8-quantized activations amplify a last-bit difference into a
# changed greedy token within a few steps. The serial control leg must be
# reproducible, so the reference engine runs the humming W4A16 MoE with its
# batch-invariant kernels, and the Mamba FP8 projections on CUTLASS rather than
# FlashInfer's autotuned FP8 GEMM. Set before vLLM is imported; the engine-core
# process inherits them.
os.environ.setdefault("VLLM_HUMMING_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_DISABLED_KERNELS", "FlashInferFP8ScaledMMLinearKernel")
from typing import Protocol

TOP_K = 8


@dataclass(frozen=True)
class Emit:
    """One emitted token and the top-k (token, raw logit) of its position."""

    token: int
    top: tuple[tuple[int, float], ...]


class Backend(Protocol):
    vocab_size: int
    eos_token: int
    depth: int  # draft tokens per cycle; 0 = serial
    ctx_size: int
    ident: str
    model_path: str

    def start(
        self, tokens: list[int], max_tokens: int | None = None, logprobs: bool = True
    ) -> list[Emit]: ...

    def step(self) -> list[Emit]: ...

    def abort(self) -> bool:
        """Drop the request. True when it was still generating."""
        ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# vLLM (the scored engine)
# --------------------------------------------------------------------------- #


@dataclass
class VllmConfig:
    """Everything the fixture pins for the engine. `tools/serve-up.sh` exports
    each value; nothing here has a silent default the fixture does not name."""

    model_dir: str
    depth: int
    ctx_size: int
    gpu_memory_utilization: float
    max_num_batched_tokens: int
    kv_cache_memory_bytes: int
    ident: str
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "VllmConfig":
        def need(name: str) -> str:
            value = os.environ.get(name, "")
            if not value:
                raise RuntimeError(f"{name} is unset; tools/serve-up.sh exports it")
            return value

        depth = int(need("RESIDENT_SPEC_DEPTH"))
        if not 0 <= depth <= 6:
            raise RuntimeError(f"RESIDENT_SPEC_DEPTH={depth} is outside 0..6")
        return cls(
            model_dir=need("RESIDENT_MODEL_DIR"),
            depth=depth,
            ctx_size=int(need("RESIDENT_CTX_SIZE")),
            gpu_memory_utilization=float(need("RESIDENT_GPU_MEMORY_UTILIZATION")),
            max_num_batched_tokens=int(need("RESIDENT_MAX_NUM_BATCHED_TOKENS")),
            kv_cache_memory_bytes=int(need("RESIDENT_KV_CACHE_MEMORY_BYTES")),
            ident=os.environ.get("RESIDENT_IDENT", "vllm"),
        )


class VllmBackend:
    """One `LLMEngine`, one greedy request at a time.

    THE PINNED ENGINE SHAPE (fixture `engine_env`): batch 1, no prefix cache
    (a phase must never reuse another phase's prefill), raw-logit top-k,
    mamba state in fp32 and the checkpoint's own quantization as declared by
    its `hf_quant_config.json`. The only thing that differs between legs is
    `speculative_config`, and only through `RESIDENT_SPEC_DEPTH`.
    """

    def __init__(self, cfg: VllmConfig):
        from vllm import EngineArgs, LLMEngine, SamplingParams
        from vllm.sampling_params import RequestOutputKind
        from vllm.usage.usage_lib import UsageContext
        from vllm.v1.executor.abstract import Executor

        spec = None
        if cfg.depth > 0:
            spec = {"method": "mtp", "num_speculative_tokens": cfg.depth}
        args = EngineArgs(
            model=cfg.model_dir,
            tokenizer=cfg.model_dir,
            skip_tokenizer_init=True,
            trust_remote_code=False,
            dtype="auto",
            seed=0,
            max_model_len=cfg.ctx_size,
            max_num_seqs=1,
            max_num_batched_tokens=cfg.max_num_batched_tokens,
            enable_prefix_caching=False,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            # PINNED, not profiled: the profiled KV size differs box to box
            # (CUDA-graph memory estimates vary), which changes the cache
            # geometry and, with it, this model's numerics. A fixed size makes
            # every box identical (port notes section 7.2).
            kv_cache_memory_bytes=cfg.kv_cache_memory_bytes,
            mamba_ssm_cache_dtype="float32",
            logprobs_mode="raw_logits",
            max_logprobs=TOP_K,
            speculative_config=spec,
            moe_backend="humming",
            # Read each shard whole before parsing. The default mmap path
            # loads the 52 shards in about 135 s on GB10; eager loads them in
            # about 12 s (measured on a Spark, warm page cache). Same bytes,
            # same tensors: it changes load time only, not numerics.
            safetensors_load_strategy="eager",
            disable_log_stats=True,
        )
        # Drive the single-request engine directly, preserving the factory's
        # context and executor. Its default-on multiprocessing environment
        # flag would override from_engine_args(enable_multiprocessing=False).
        usage_context = UsageContext.ENGINE_CONTEXT
        engine_config = args.create_engine_config(usage_context)
        self._engine = LLMEngine(
            vllm_config=engine_config,
            executor_class=Executor.get_class(engine_config),
            log_stats=not args.disable_log_stats,
            usage_context=usage_context,
            multiprocess_mode=False,
        )
        # Startup is done: keep this process's threads on the fastest cores
        # too (the engine core pins its own the same way). VLLM_HOSTFAST_PIN=0
        # disables; it changes no computation.
        from vllm.v1.worker.hostfast import pin_to_fast_cpus

        pinned = pin_to_fast_cpus()
        if pinned:
            print(f"resident: pinned to CPUs {pinned}", file=sys.stderr, flush=True)
        self._SamplingParams = SamplingParams
        self._delta = RequestOutputKind.DELTA
        hf = self._engine.model_config.hf_config
        self.vocab_size = int(hf.vocab_size)
        eos = hf.eos_token_id
        self.eos_token = int(eos[0] if isinstance(eos, (list, tuple)) else eos)
        self.depth = cfg.depth
        self.ctx_size = cfg.ctx_size
        self.ident = cfg.ident
        self.model_path = cfg.model_dir
        self._rid = 0
        self._live: str | None = None
        self._warm_up()

    # BOOT WARM-UP. The first prefill and the first decode steps after the
    # engine is built run cold (lazy kernel JIT, autotune, allocator growth):
    # measured on GB10, the first seed-length prefill is ~2.8x slower than every
    # later one. benchd times the FIRST prefill a leg's resident serves, so the
    # resident pays that cost here, before it reports ready, on a synthetic
    # input-independent prompt -- never the benchmark's tokens. It is the same
    # class of work as CUDA graph capture at boot. The prompt is the fixture's
    # `seed_tokens` long, so the warm-up JITs the exact shapes the timed
    # prefill runs.
    WARMUP_PROMPT_TOKENS = 4096
    WARMUP_DECODE_STEPS = 64
    WARMUP_ROUNDS = 3

    def _warm_up(self) -> None:
        prompt = [(i * 7919) % 30000 + 1000 for i in range(self.WARMUP_PROMPT_TOKENS)]
        for _ in range(self.WARMUP_ROUNDS):
            self.start(prompt)
            for _ in range(self.WARMUP_DECODE_STEPS):
                self.step()
            self.abort()

    def _params(self, prompt_len: int, max_tokens: int | None = None, logprobs: bool = True):
        room = max(1, self.ctx_size - prompt_len - self.depth - 1)
        return self._SamplingParams(
            temperature=0.0,
            max_tokens=room if max_tokens is None else min(room, max_tokens),
            ignore_eos=True,
            # None: no top-k logprob work on the GPU, in the D2H copy or in
            # either process's output path. The sampled tokens do not depend
            # on it (the logprobs are computed from copies of the logits).
            logprobs=TOP_K if logprobs else None,
            detokenize=False,
            output_kind=self._delta,
            # A fresh object per request that nothing else holds: the engine's
            # copy of it may be shallow (no deepcopy on the timed prefill path).
            skip_clone=True,
        )

    def _collect(self) -> list[Emit]:
        emitted: list[Emit] = []
        for out in self._engine.step():
            if out.request_id != self._live:
                continue
            comp = out.outputs[0]
            if comp.logprobs:
                for token, lp in zip(comp.token_ids, comp.logprobs):
                    top: tuple[tuple[int, float], ...] = ()
                    if lp:
                        ranked = sorted(lp.items(), key=lambda kv: kv[1].logprob, reverse=True)
                        top = tuple((int(t), float(v.logprob)) for t, v in ranked[:TOP_K])
                    emitted.append(Emit(int(token), top))
            else:
                emitted.extend(Emit(int(token), ()) for token in comp.token_ids)
        return emitted

    def start(
        self, tokens: list[int], max_tokens: int | None = None, logprobs: bool = True
    ) -> list[Emit]:
        self.abort()
        self.set_emit_target(None)
        if len(tokens) + self.depth + 2 > self.ctx_size:
            raise RuntimeError(f"{len(tokens)} tokens do not fit the {self.ctx_size}-token context")
        self._rid += 1
        self._live = f"r{self._rid}"
        self._engine.add_request(
            self._live,
            {"prompt_token_ids": list(tokens)},
            self._params(len(tokens), max_tokens, logprobs),
        )
        emitted: list[Emit] = []
        while not emitted:
            if not self._engine.has_unfinished_requests():
                raise RuntimeError("the engine finished the request without emitting a token")
            emitted = self._collect()
        return emitted

    def set_emit_target(self, target: int | None) -> None:
        """Prompt + emitted tokens at which the running free run is done (None:
        unbounded). The MTP drafter skips draft passes that could only extend a
        verify past that point (llm_base_proposer._emit_budget_drafts)."""
        try:
            runner = self._engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
            drafter = getattr(runner, "drafter", None)
        except AttributeError:
            return
        if drafter is not None:
            drafter.emit_target = target

    def step(self) -> list[Emit]:
        if self._live is None:
            raise RuntimeError("step with no live request")
        emitted: list[Emit] = []
        while not emitted:
            if not self._engine.has_unfinished_requests():
                raise RuntimeError("the request reached the context limit")
            emitted = self._collect()
        return emitted

    def step_with_budget(self, budget: int) -> list[Emit]:
        """One step for a caller that still wants `budget` tokens.

        Near the end of the budget the engine core may return a queued
        batch's output without first speculating another round, when that
        output already covers the budget. The budget applies to this call
        alone; the generation limit and scheduler arithmetic are unchanged.
        """
        if os.environ.get("C57_LASTDRAFT", "1") == "0" or self.depth <= 0:
            return self.step()
        core = getattr(self._engine.engine_core, "engine_core", None)
        if core is None:
            return self.step()
        core._c57_decode_budget = budget
        core._c57_drained = False
        try:
            emitted = self.step()
        finally:
            core._c57_decode_budget = None
        if getattr(core, "_c57_drained", False):
            core._c57_drained = False
            self._c57_tail_drained = True
        return emitted

    def tail_drained(self) -> bool:
        """The live request's queue was drained at a budget's end, so no batch
        of it is in flight. The engine never continues such a request ahead;
        the session restarts it from its context instead."""
        return self._live is not None and getattr(self, "_c57_tail_drained", False)

    def abort(self) -> bool:
        running = False
        self._c57_tail_drained = False
        if self._live is not None:
            running = self._engine.has_unfinished_requests()
            self._engine.abort_request([self._live])
            # Drain the abort so the next request starts on an idle engine.
            while self._engine.has_unfinished_requests():
                self._engine.step()
            self._live = None
        return running

    def close(self) -> None:
        self.abort()


# --------------------------------------------------------------------------- #
# Fake (tests only)
# --------------------------------------------------------------------------- #


class FakeBackend:
    """Deterministic greedy "model": the next token is a fixed hash of the last
    two context tokens. With depth > 0 a step emits a variable number of tokens
    (1..depth+1), every one of them the same greedy continuation, which is the
    contract a correct speculative engine keeps."""

    def __init__(self, depth: int = 0, vocab_size: int = 131072, ctx_size: int = 8192):
        self.vocab_size = vocab_size
        self.eos_token = 2
        self.depth = depth
        self.ctx_size = ctx_size
        self.ident = f"fake depth={depth}"
        self.model_path = "/fake/model"
        self._ctx: list[int] | None = None
        self._cycle = 0
        self._left: int | None = None
        self._logprobs = True
        self.starts = 0
        # The `logprobs` argument of every start, in order (tests read it).
        self.logprob_starts: list[bool] = []

    def _next(self, ctx: list[int]) -> int:
        a = ctx[-1] if ctx else 0
        b = ctx[-2] if len(ctx) > 1 else 0
        return (a * 31 + b * 17 + 7) % 1000 + 20

    def _emit(self, ctx: list[int]) -> Emit:
        token = self._next(ctx)
        if not self._logprobs:
            return Emit(token, ())
        top = tuple(
            (token if i == 0 else (token + i) % self.vocab_size, 10.0 - i) for i in range(TOP_K)
        )
        return Emit(token, top)

    def start(
        self, tokens: list[int], max_tokens: int | None = None, logprobs: bool = True
    ) -> list[Emit]:
        if len(tokens) + self.depth + 2 > self.ctx_size:
            raise RuntimeError("context full")
        self.starts += 1
        self.logprob_starts.append(logprobs)
        self._logprobs = logprobs
        self._ctx = list(tokens)
        self._cycle = 0
        self._left = None if max_tokens is None else max_tokens - 1
        first = self._emit(self._ctx)
        self._ctx.append(first.token)
        return [first]

    def step(self) -> list[Emit]:
        if self._ctx is None:
            raise RuntimeError("step with no live request")
        # vLLM finishes a request at its max_tokens; so does the fake.
        if self._left == 0:
            raise RuntimeError("the request reached its max_tokens")
        self._cycle += 1
        n = 1 + (self._cycle % (self.depth + 1)) if self.depth else 1
        if self._left is not None:
            n = min(n, self._left)
            self._left -= n
        out = []
        for _ in range(n):
            e = self._emit(self._ctx)
            self._ctx.append(e.token)
            out.append(e)
        return out

    def abort(self) -> bool:
        running = self._ctx is not None and self._left != 0
        self._ctx = None
        return running

    def close(self) -> None:
        self.abort()

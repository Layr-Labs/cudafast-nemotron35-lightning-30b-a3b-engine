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
frontier it holds without another forward pass.

`VllmBackend` is the scored engine. `FakeBackend` is a deterministic stand-in
with the same step semantics (including multi-token speculative steps), so the
resident's wire and bookkeeping are tested with no GPU and no vLLM install.
"""

from __future__ import annotations

import os
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

    def start(self, tokens: list[int], max_tokens: int | None = None) -> list[Emit]: ...

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
            disable_log_stats=True,
        )
        self._engine = LLMEngine.from_engine_args(args)
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

    def _params(self, prompt_len: int, max_tokens: int | None = None):
        room = max(1, self.ctx_size - prompt_len - self.depth - 1)
        return self._SamplingParams(
            temperature=0.0,
            max_tokens=room if max_tokens is None else min(room, max_tokens),
            ignore_eos=True,
            logprobs=TOP_K,
            detokenize=False,
            output_kind=self._delta,
        )

    def _collect(self) -> list[Emit]:
        emitted: list[Emit] = []
        for out in self._engine.step():
            if out.request_id != self._live:
                continue
            comp = out.outputs[0]
            lps = comp.logprobs or [None] * len(comp.token_ids)
            for token, lp in zip(comp.token_ids, lps):
                top: tuple[tuple[int, float], ...] = ()
                if lp:
                    ranked = sorted(lp.items(), key=lambda kv: kv[1].logprob, reverse=True)
                    top = tuple((int(t), float(v.logprob)) for t, v in ranked[:TOP_K])
                emitted.append(Emit(int(token), top))
        return emitted

    def start(self, tokens: list[int], max_tokens: int | None = None) -> list[Emit]:
        self.abort()
        if len(tokens) + self.depth + 2 > self.ctx_size:
            raise RuntimeError(f"{len(tokens)} tokens do not fit the {self.ctx_size}-token context")
        self._rid += 1
        self._live = f"r{self._rid}"
        self._engine.add_request(
            self._live, {"prompt_token_ids": list(tokens)}, self._params(len(tokens), max_tokens)
        )
        emitted: list[Emit] = []
        while not emitted:
            if not self._engine.has_unfinished_requests():
                raise RuntimeError("the engine finished the request without emitting a token")
            emitted = self._collect()
        return emitted

    def step(self) -> list[Emit]:
        if self._live is None:
            raise RuntimeError("step with no live request")
        emitted: list[Emit] = []
        while not emitted:
            if not self._engine.has_unfinished_requests():
                raise RuntimeError("the request reached the context limit")
            emitted = self._collect()
        return emitted

    def abort(self) -> bool:
        running = False
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
        self.starts = 0

    def _next(self, ctx: list[int]) -> int:
        a = ctx[-1] if ctx else 0
        b = ctx[-2] if len(ctx) > 1 else 0
        return (a * 31 + b * 17 + 7) % 1000 + 20

    def _emit(self, ctx: list[int]) -> Emit:
        token = self._next(ctx)
        top = tuple(
            (token if i == 0 else (token + i) % self.vocab_size, 10.0 - i) for i in range(TOP_K)
        )
        return Emit(token, top)

    def start(self, tokens: list[int], max_tokens: int | None = None) -> list[Emit]:
        if len(tokens) + self.depth + 2 > self.ctx_size:
            raise RuntimeError("context full")
        self.starts += 1
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

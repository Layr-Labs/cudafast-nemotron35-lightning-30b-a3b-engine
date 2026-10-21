# The resident: one weight load per leg

## 1. The problem

benchd runs every phase of a leg in a FRESH worker process: warmup, timed
prefill, timed decode and correctness. A worker that loaded the model itself
would load about 21.6 GB, then compile and capture CUDA graphs, once for every
phase.

## 2. The decision

The weights have ONE owner per leg: the vLLM resident
(`harness/vllm_resident/resident.py`). `tools/serve-up.sh --boot` starts it
inside the pinned image and waits for a healthy `hello`. It then writes the
socket path as the first line of `--socket-out` and exits, leaving the resident
running. benchd injects that socket into every worker it spawns for the leg.

The worker is the adapter binary `cuda-engine` (built from
`harness/protocol-adapter`), staged as `.build/release/mlxfast-runtime-worker`,
the path benchd spawns. Each per-phase worker connects to the socket and loads
nothing.
`tools/serve-up.sh --stop` tears the resident down, and a second stop is a
harmless no-op.

## 3. Why not `vllm serve`

The OpenAI-compatible server carries text or tokens in and sampled tokens out.
It cannot carry the four things the adapter's contract needs:

- a teacher-forced `eval` of a caller-chosen token;
- the raw top-8 logits of the current frontier;
- per-cycle speculative accounting (`acceptance_lengths`, drafted and accepted
  counts);
- a hard phase boundary: nothing may survive from one benchd phase into the
  next.

So the resident drives `vllm.LLMEngine` directly: one request, stepped one
engine step at a time.

## 4. The wire

The transport is NDJSON over `AF_UNIX`/`SOCK_STREAM`:

- One request object per line and one response object per line, in strict
  order.
- Each response carries `ok`; a failed one also carries `error`.
- One client at a time: a second connection waits in the backlog.

| Verb | Request | Response |
| --- | --- | --- |
| `hello` | — | `vocab_size`, `eos_token`, `mtp_armed`, `draft_tokens` (depth + 1), `ctx_size`, `load_epoch` (ms timestamp of the load), `spec_run`, `ident`, `model_path`, `mtp_head_path` |
| `invalidate` | — | — |
| `sync` | `tokens`, optional `max_tokens` | `token` (the frontier argmax) |
| `eval` | `token` | `token` |
| `argmax` | — | `token` |
| `top_logits` | `k` | `ids`, `logits` (raw logits, highest first) |
| `eval_speculative` | `first_token`, `budget` | `tokens` (committed, fed token first), `token` |
| `spec_run` | `first_token`, `count` | `rounds` (each cycle's committed tokens), `frontiers`, `token` |
| `spec_counters` | — | `drafts`, `hits`, `quenches`, `disagreements` |
| `bye` | — | — |

The Rust client is `harness/protocol-adapter/src/resident.rs`; it and
`resident.py` must agree verb for verb.

- **The frontier argmax comes back eagerly.** Every state-advancing verb
  returns it, so `argmax` costs no round trip.
- **One request per free run.** `spec_run` runs a whole speculative free run in
  one request.

### Frontier bookkeeping

vLLM emits tokens one engine step at a time. A speculative step can emit up to
depth + 1 tokens. The resident keeps three things:

- `ctx`: the tokens the model has consumed;
- `frontier`: the greedy next token, not yet consumed, with its top-k logits;
- `ahead`: tokens the engine already emitted past the frontier.

Feeding the frontier consumes `ahead` before stepping the engine again. Every
token in `ahead` is the target's own greedy choice, so this changes nothing.
Feeding a token that is NOT the frontier (teacher forcing) re-prefills
`ctx + [token]`. That path is correct and deliberately not fast.

A cycle commits the fed token plus the tokens it takes from `ahead`, capped at
the budget still wanted. `acceptance_lengths` are the committed counts per
cycle. For every engine step with speculation on, the counters add `depth` to
`drafts` and `len(emitted) - 1` to `hits`.

### The disagreement counter, and the order it must ship in

`spec_counters.disagreements` is audit-only. It counts verify/replay
disagreements, and the vLLM resident reports 0. The adapter forwards the leg's
delta as `verify_replay_disagreements` on `free_decode_run`.

An absent field means NOT REPORTED, and the serial route sends none. A benchd
that predates the field parses the response with `deny_unknown_fields` and
would reject the line, so the box's benchd pair must come from the pinned
benchd channel (`tools/fetch-benchd.sh`), which carries it. The worker cannot
detect the benchd version, so this is a deployment order, not a runtime gate.

## 5. The phase boundary

The resident invalidates the session when it accepts each connection, before
the client's first byte. `invalidate` aborts the live request and drains the
engine. vLLM's prefix cache is off in the scored configuration, so no prefill
survives into the next phase. Reusing work across phases is prohibited
(contract section 8), whatever mechanism does it.

benchd keeps ONE worker, so one connection, for a whole leg. vLLM's engine
core steps a live request until it finishes, whether or not anyone reads the
tokens. Two rules stop the engine when a phase has what it asked for:

- The adapter sends `invalidate` when it closes each phase (the
  `phase_diagnostics` barrier, or a fail-closed discard).
- A timed prefill sends `sync` with `max_tokens: 1`. Its request ends at the
  frontier token, so the next timed pass does not start behind a decode step.
  A bounded sequence that is fed on is prefilled again from its context.

Without these rules the last request of a phase decoded on to the context
limit (about 7,700 tokens, about 100 s on GB10). The GPU stayed busy through
benchd's quiescence gate before the next phase.

## 6. The memory plan

GB10 memory is unified, so vLLM's `gpu_memory_utilization` is a fraction of the
box's whole memory. `serve-up.sh` refuses BEFORE the load in two cases:

- the fixture's fraction of `MemTotal`, plus 8 GiB of host overhead, is not
  free in `MemAvailable`;
- the weights plus 4 GiB of cache do not fit that reservation.

`tools/test-serve-up-plan-memory.sh` proves both boundaries to the kilobyte.

## 7. Fail-closed

- **The client poisons itself.** A resident failure seen by an infallible
  client call poisons the client session, and the next fallible verb returns
  the error. A broken resident can never read as a fast phase.
- **The adapter checks the arithmetic.** It checks
  `committed == count == sum(acceptance_lengths)`. It refuses a cycle that
  does not start with the fed token, or that overshoots the count.
- **`serve-up.sh` checks the resident against the leg.** It refuses a resident
  whose `hello` depth or context disagrees with the leg it booted.

## 8. The identity strings

The worker's hello announces `vllm-resident load_epoch=<ms> <ident>`, and
benchd seals it as `engine_backend`. The `ident` carries the vLLM pin, the
image digest, the driver and the spec label. Every phase of a leg reports the
same `load_epoch`.

## 9. What proves it

| Test | What it proves |
|---|---|
| `harness/vllm_resident/test_resident.py` | the bookkeeping: spec runs equal the serial stream at every depth, overhang, teacher forcing, refusals, and the wire over a real socket |
| `harness/vllm_resident/test_adapter_e2e.py` | the real `cuda-engine` against the real resident at depths 0 to 6 |
| `tools/test-serve-up-boot-stop.sh` | `--boot`/`--stop`, idempotence, handover failure, the wrapper form |
| `tools/three-flows-dry-run.sh` | the real benchd binaries over the resident: golden recording, local legs, the paired lifecycle |

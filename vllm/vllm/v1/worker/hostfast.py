# SPDX-License-Identifier: Apache-2.0
"""LOCAL PATCH (cudafast-nemotron35): host-side trims for the batch-1 async
speculative decode loop.

Nothing here changes a kernel, a shape, a batch or a value the GPU reads. Each
piece removes a host wait or host work from the engine core's critical path:

* S1 and S2. In async spec decode the attention builders asked for
  ``CommonAttentionMetadata.seq_lens_cpu`` with no CPU copy cached, so the
  lazy property read ``seq_lens`` back from the GPU. That readback waits for
  every kernel queued on the default stream, and the GPU then idles while the
  CPU plans and launches the next forward. S1 is the verify's build (it waited
  for the previous cycle's whole draft chain); S2 is the first single-token
  draft step's build (it waited for the verify, the rejection and draft step
  0). The integers they read back are computed here on the CPU from values
  already on the host (``exact_verify_seq_lens``, ``draft_seq_lens``), so the
  FlashInfer plan receives the same integers without the wait.
* Once the CPU no longer waits at S2, it can reach the next draft step's build
  while the previous step's host-to-device copies are still queued behind GPU
  work. ``StagingGuard`` waits for those copies before a builder rewrites its
  pinned staging buffers (the builder's ``paged_kv_*`` host buffers and the
  FlashInfer wrappers' pinned plan workspace).
* Draft staging sets. ``StagingGuard`` (and the proposer's per-step plan
  event) made every draft step's build wait for the previous step's copies,
  so the CPU never got more than one draft step ahead of the GPU and reached
  the next verify's launch only after the last draft step had nearly
  finished. ``DraftStagingSets`` gives each draft step of a cycle its own set
  of those pinned host buffers instead; a set is rewritten only after the
  event recorded at its previous use (one cycle earlier) has completed, so no
  draft step waits and the next verify is queued while the drafts still run.
  The device buffers the copies fill are unchanged and still shared: stream
  order serializes them exactly as before.
* GC. Automatic collection is off in the engine core while it has work; young
  generations are collected while the CPU waits for the GPU anyway
  (``gc_in_shadow``) and automatic collection is back on when it goes idle.
* CPU placement. ``pin_to_fast_cpus`` restricts the calling process (and the
  processes it starts later) to the highest-capacity cores (the Cortex-X925
  cluster on GB10).

Switches, read once at import in every process (all default on):
  VLLM_HOSTFAST_S1=0     the verify reads seq_lens back from the GPU again
  VLLM_HOSTFAST_S2=0     the first draft step reads seq_lens back again
  VLLM_HOSTFAST_SETS=0   draft steps share one staging set (StagingGuard)
  VLLM_HOSTFAST_GC=0     automatic GC stays on in the engine core
  VLLM_HOSTFAST_PIN=0    no CPU affinity change
  VLLM_HOSTFAST_CHECK=1  debug, default off: also read seq_lens back from the
                         GPU at every replaced site and raise on a mismatch
"""

from __future__ import annotations

import gc
import os
import sys
from collections.abc import Sequence

import numpy as np
import torch


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip() not in ("0", "", "false", "no")


S1 = _flag("VLLM_HOSTFAST_S1", "1")
S2 = _flag("VLLM_HOSTFAST_S2", "1")
SETS = _flag("VLLM_HOSTFAST_SETS", "1")
GC = _flag("VLLM_HOSTFAST_GC", "1")
PIN = _flag("VLLM_HOSTFAST_PIN", "1")
CHECK = _flag("VLLM_HOSTFAST_CHECK", "0")

# Counters for the debug check: builds compared, and builds that kept the GPU
# readback because the CPU value was not available. Logged to stderr.
checked = {"verify": 0, "draft": 0}
fallbacks = {"verify": 0, "draft": 0}


def _report() -> None:
    print(f"hostfast check: matched {checked} fell back {fallbacks}", file=sys.stderr, flush=True)


if CHECK:
    import atexit

    atexit.register(_report)


def note_fallback(site: str) -> None:
    """Debug (VLLM_HOSTFAST_CHECK=1): count a site that kept the readback."""
    if CHECK:
        fallbacks[site] += 1
        if fallbacks[site] in (1, 10, 100) or fallbacks[site] % 1000 == 0:
            _report()


# --------------------------------------------------------------------------- #
# Exact sequence lengths
# --------------------------------------------------------------------------- #


def exact_verify_seq_lens(
    optimistic_seq_lens: torch.Tensor,
    num_reqs: int,
    prev_positions: np.ndarray,
    prev_num_draft_tokens: np.ndarray,
    valid_sampled_token_count: Sequence[int],
) -> torch.Tensor | None:
    """The seq_lens ``_prepare_inputs`` left on the GPU for this step, on the CPU.

    ``optimistic_seq_lens`` is ``num_computed_tokens_cpu + num_scheduled``, the
    scheduler's view, which counts every draft of the previous step as accepted.
    ``update_num_computed_tokens_for_batch_change`` corrects it on the GPU for
    each request i with ``p = prev_positions[i] >= 0`` and
    ``prev_num_draft_tokens[p] > 0``: the request's previous step scheduled
    ``1 + prev_num_draft_tokens[p]`` tokens of which
    ``valid_sampled_token_count[p]`` were valid, so the scheduler is ahead by
    ``prev_num_draft_tokens[p] + 1 - valid_sampled_token_count[p]``. That is the
    correction the runner itself applies to its CPU counts after the launch
    (``correct_spec_decode_token_counts``); it is subtracted here, from the
    same host values the GPU kernel was given. Rows past ``num_reqs`` (padding)
    keep their value (0, as on the GPU).

    Returns a fresh int32 tensor, or None when a previous position has no valid
    count (the caller then keeps the GPU readback).
    """
    out = optimistic_seq_lens.clone()
    for i in range(num_reqs):
        p = int(prev_positions[i])
        if p < 0:
            continue
        drafts = int(prev_num_draft_tokens[p])
        if drafts <= 0:
            continue
        if p >= len(valid_sampled_token_count):
            return None
        out[i] -= drafts + 1 - int(valid_sampled_token_count[p])
    return out


def num_rejected_tokens(
    num_draft_tokens: Sequence[int], valid_sampled_token_count: Sequence[int]
) -> torch.Tensor | None:
    """``eagle_prepare_inputs_padded_kernel``'s ``num_rejected_tokens`` on the
    CPU: ``num_draft + 1 - valid_count`` for a request that drafted, else 0."""
    n = len(num_draft_tokens)
    if len(valid_sampled_token_count) < n:
        return None
    rejected = [
        (int(d) + 1 - int(v)) if int(d) > 0 else 0
        for d, v in zip(num_draft_tokens, valid_sampled_token_count)
    ]
    return torch.tensor(rejected, dtype=torch.int32)


def draft_seq_lens(
    verify_seq_lens: torch.Tensor | None,
    num_rejected: torch.Tensor | None,
    batch_size: int,
    max_model_len: int,
    num_speculative_tokens: int,
) -> torch.Tensor | None:
    """The first single-token draft step's seq_lens before its position update:
    the verify's seq_lens minus the rejected tokens, as the proposer computes on
    the GPU (``seq_lens -= num_rejected_tokens_gpu``).

    None (keep the GPU readback) when an input is missing or a draft position
    could reach ``max_model_len``, where the GPU step kernel clamps and the
    CPU's ``+= 1`` would not.
    """
    if verify_seq_lens is None or num_rejected is None:
        return None
    if verify_seq_lens.shape[0] != batch_size or num_rejected.shape[0] != batch_size:
        return None
    out = verify_seq_lens.to(torch.int32) - num_rejected
    if batch_size and int(out.max()) + num_speculative_tokens >= max_model_len:
        return None
    return out


def check_seq_lens(site: str, fast: torch.Tensor, gpu_seq_lens: torch.Tensor) -> None:
    """Debug (VLLM_HOSTFAST_CHECK=1): read the GPU value back and compare."""
    ref = gpu_seq_lens.to("cpu")
    if not torch.equal(fast.to(torch.int32), ref.to(torch.int32)):
        raise RuntimeError(
            f"hostfast {site}: CPU seq_lens {fast.tolist()} != GPU {ref.tolist()} "
            f"after {checked[site]} matching builds"
        )
    checked[site] += 1
    if checked[site] in (1, 10, 100) or checked[site] % 1000 == 0:
        _report()


# --------------------------------------------------------------------------- #
# Staging guard
# --------------------------------------------------------------------------- #


class StagingGuard:
    """Orders a metadata builder's host staging writes after the previous
    build's host-to-device copies.

    ``before()`` waits for the event ``after()`` recorded on the current stream
    at the end of the previous build. The stock loop got this ordering from the
    S1/S2 readbacks; without them the CPU can run one draft step ahead of the
    GPU. When the CPU waits here, the previous step's forward is already
    queued, so the GPU does not idle.
    """

    def __init__(self) -> None:
        self._event: torch.cuda.Event | None = None

    def before(self) -> None:
        if self._event is not None:
            self._event.synchronize()

    def after(self) -> None:
        if self._event is None:
            self._event = torch.cuda.Event()
        self._event.record()


class _StagingSet:
    """One draft step's pinned host staging for one FlashInfer builder."""

    def __init__(self, builder) -> None:
        self.indptr_cpu = torch.zeros_like(builder.paged_kv_indptr.cpu).pin_memory()
        self.indptr_np = self.indptr_cpu.numpy()
        self.indptr_buf = torch.zeros_like(builder.paged_kv_indptr_cpu_buffer).pin_memory()
        self.last_page_len_cpu = torch.zeros_like(
            builder.paged_kv_last_page_len.cpu
        ).pin_memory()
        self.last_page_len_np = self.last_page_len_cpu.numpy()
        self.workspaces: dict[int, tuple[object, torch.Tensor]] = {}

    def install(self, builder) -> None:
        builder.paged_kv_indptr.cpu = self.indptr_cpu
        builder.paged_kv_indptr.np = self.indptr_np
        builder.paged_kv_indptr_cpu_buffer = self.indptr_buf
        builder.paged_kv_last_page_len.cpu = self.last_page_len_cpu
        builder.paged_kv_last_page_len.np = self.last_page_len_np
        for wrapper in DraftStagingSets.wrappers(builder):
            entry = self.workspaces.get(id(wrapper))
            if entry is None or entry[0] is not wrapper:
                buf = torch.empty_like(wrapper._pin_memory_int_workspace_buffer).pin_memory()
                entry = (wrapper, buf)
                self.workspaces[id(wrapper)] = entry
            wrapper._pin_memory_int_workspace_buffer = entry[1]


class DraftStagingSets:
    """Per-draft-step pinned staging for the drafter's FlashInfer builders.

    A FlashInfer build writes, on the host, the builder's ``paged_kv_indptr``
    and ``paged_kv_last_page_len`` buffers and its indptr staging buffer, and
    its plan writes the wrapper's pinned plan workspace; host-to-device copies
    queued behind the GPU read all of them later. Draft step ``i`` of every
    cycle uses set ``i`` of these buffers (installed on the builder and its
    wrappers before the build), and set ``i`` is rewritten only once the event
    recorded after its previous build has completed. The values written, the
    device buffers they are copied to and the order of those copies on the
    stream are the same as with one shared set.
    """

    def __init__(self, builders: list, num_sets: int) -> None:
        self.builders = builders
        self.original = [
            (b.paged_kv_indptr.cpu, b.paged_kv_indptr_cpu_buffer, b.paged_kv_last_page_len.cpu)
            for b in builders
        ]
        self.original_workspaces: dict[int, tuple[object, torch.Tensor]] = {}
        self.sets = [[_StagingSet(b) for b in builders] for _ in range(num_sets)]
        self.events: list[torch.cuda.Event | None] = [None] * num_sets
        self._current = 0

    @staticmethod
    def wrappers(builder) -> list:
        found = []
        for name in ("_decode_wrappers_cudagraph", "_spec_prefill_wrappers_cudagraph"):
            found.extend((getattr(builder, name, None) or {}).values())
        for name in ("_prefill_wrapper", "_decode_wrapper", "_noncausal_prefill_wrapper"):
            w = getattr(builder, name, None)
            if w is not None:
                found.append(w)
        return [w for w in found if hasattr(w, "_pin_memory_int_workspace_buffer")]

    @staticmethod
    def supports(builder) -> bool:
        return all(
            hasattr(builder, name)
            for name in (
                "paged_kv_indptr",
                "paged_kv_indptr_cpu_buffer",
                "paged_kv_last_page_len",
            )
        ) and getattr(builder, "pin_memory", False)

    def before(self, draft_index: int) -> None:
        i = draft_index % len(self.sets)
        event = self.events[i]
        if event is not None and not event.query():
            event.synchronize()
        for builder in self.builders:
            for wrapper in self.wrappers(builder):
                if id(wrapper) not in self.original_workspaces:
                    self.original_workspaces[id(wrapper)] = (
                        wrapper,
                        wrapper._pin_memory_int_workspace_buffer,
                    )
        for staging, builder in zip(self.sets[i], self.builders):
            staging.install(builder)
        self._current = i

    def after(self) -> None:
        i = self._current
        if self.events[i] is None:
            self.events[i] = torch.cuda.Event()
        self.events[i].record()


# --------------------------------------------------------------------------- #
# GC
# --------------------------------------------------------------------------- #

_gc_paused = False


def gc_busy() -> None:
    """The engine core has work: stop automatic collection (it fires inside
    the scheduling and launch path)."""
    global _gc_paused
    if GC and not _gc_paused and gc.isenabled():
        gc.disable()
        _gc_paused = True


def gc_idle() -> None:
    """The engine core is idle: give collection back to the interpreter."""
    global _gc_paused
    if _gc_paused:
        gc.enable()
        _gc_paused = False


def gc_in_shadow() -> None:
    """The CPU is about to wait for the GPU: collect the young generations
    now, so the paused GC does not let cyclic garbage pile up."""
    if _gc_paused:
        gc.collect(1)


# --------------------------------------------------------------------------- #
# CPU placement
# --------------------------------------------------------------------------- #


def _read_int(path: str) -> int | None:
    try:
        with open(path) as fp:
            return int(fp.read().strip())
    except (OSError, ValueError):
        return None


def fast_cpus(
    allowed: set[int],
    sysfs: str = "/sys/devices/system/cpu",
    ratio: float = 0.85,
) -> set[int]:
    """The allowed CPUs whose ``cpu_capacity`` (else ``cpuinfo_max_freq``) is
    within ``ratio`` of the largest. Empty when the CPUs do not differ (or
    nothing is readable), so the caller leaves the affinity alone.

    On GB10 the capacities are 997-1024 for the ten Cortex-X925 cores and
    718-731 for the ten Cortex-A725 cores.
    """
    for leaf in ("cpu_capacity", "cpufreq/cpuinfo_max_freq"):
        values = {c: _read_int(f"{sysfs}/cpu{c}/{leaf}") for c in allowed}
        known = {c: v for c, v in values.items() if v is not None and v > 0}
        if len(known) != len(allowed) or not known:
            continue
        top = max(known.values())
        fast = {c for c, v in known.items() if v >= ratio * top}
        if fast and fast != allowed:
            return fast
    return set()


def pin_to_fast_cpus() -> list[int]:
    """Restrict every thread of this process (and so the threads and processes
    it starts afterwards) to the fast cores. Returns the CPU list applied,
    empty when nothing changed. Call it after startup: compile workers and
    other startup parallelism keep every core."""
    if not PIN or not hasattr(os, "sched_setaffinity"):
        return []
    try:
        allowed = set(os.sched_getaffinity(0))
        fast = fast_cpus(allowed)
        if not fast:
            return []
        try:
            tids = [int(t) for t in os.listdir("/proc/self/task")]
        except OSError:
            tids = [0]
        for tid in tids:
            try:
                os.sched_setaffinity(tid, fast)
            except OSError:
                pass  # a thread that exited meanwhile
        os.sched_setaffinity(0, fast)
        return sorted(fast)
    except OSError:
        return []

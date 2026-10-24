# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LOCAL PATCH (cudafast-nemotron35): host-side trims of the prefill step.

Nothing here changes a computed value: the first two switches remove host
latency from the critical path of a step, the third moves the same bytes in
one kernel instead of two.
Each switch is on by default and is read at call time, so a test can flip it
in a live engine (harness/bench/prefill_lab.py ab).

  VLLM_PREFILL_SPIN_OUTPUT_WAIT=0  wait for a step's sampled-token copy with a
                                   blocking (sleeping) CUDA event again.
  VLLM_PREFILL_MAMBA_META_REUSE=0  build the Mamba-2 attention metadata of
                                   every KV-cache group of a prefill again.
  VLLM_PREFILL_STATE_COPY=0        store a prefill's final SSM states with the
                                   stock gather + index_put pair again
                                   (model_executor/layers/mamba/ops/
                                   prefill_state_copy.py).
  VLLM_PREFILL_FAST_CHECK=1        (debug) also build the reused metadata and
                                   compare every field; raise on a mismatch.
"""

import os
from dataclasses import fields, is_dataclass

import torch


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) != "0"


# The engine core waits for each step's sampled tokens on an event recorded
# after the device-to-host copy. A blocking event puts the thread to sleep and
# on GB10 it woke 1.0-1.6 ms after the copy had finished (measured: GPU end of
# sampling by CUDA event vs. the CPU return of the wait); every other wait in
# the step loop uses torch's default (spinning) event. A non-blocking event
# releases the GIL like the blocking one; it only stops the thread sleeping.
SPIN_OUTPUT_WAIT = _flag("VLLM_PREFILL_SPIN_OUTPUT_WAIT")

# With speculative decoding on, Mamba builders do not reuse metadata across
# KV-cache groups (supports_update_block_table is False), so a prefill builds
# the same Mamba-2 metadata once per group (4 groups here). For a batch with
# no decode rows the metadata depends on the group only through the state
# indices, which update_block_table recomputes from the group's block table.
MAMBA_META_REUSE = _flag("VLLM_PREFILL_MAMBA_META_REUSE")

# One copy kernel instead of the gather + scatter of each prefill's final state.
STATE_COPY = _flag("VLLM_PREFILL_STATE_COPY")

CHECK = _flag("VLLM_PREFILL_FAST_CHECK", "0")


def output_event() -> torch.cuda.Event:
    return torch.cuda.Event(blocking=not SPIN_OUTPUT_WAIT)


def mamba_reusable(metadata) -> bool:
    """A cached Mamba metadata may serve another KV-cache group: prefill rows
    only (decode rows carry per-step spec-decode state)."""
    return (
        MAMBA_META_REUSE
        and getattr(metadata, "num_decodes", None) == 0
        and getattr(metadata, "num_prefills", 0) > 0
    )


def _equal(a, b, path: str, diffs: list[str]) -> None:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and a.device == b.device
            and torch.equal(a, b)
        ):
            diffs.append(path)
    elif is_dataclass(a) and is_dataclass(b):
        if type(a) is not type(b):
            diffs.append(path)
            return
        for f in fields(a):
            _equal(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}", diffs)
    elif isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            diffs.append(path)
            return
        for k in a:
            _equal(a[k], b[k], f"{path}[{k!r}]", diffs)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            diffs.append(path)
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _equal(x, y, f"{path}[{i}]", diffs)
    elif a != b:
        diffs.append(path)


CHECKS = {"compared": 0}


def check_same(reused, built, what: str) -> None:
    """Debug check: the reused metadata equals a fresh build, field by field."""
    diffs: list[str] = []
    _equal(reused, built, what, diffs)
    CHECKS["compared"] += 1
    if diffs:
        raise RuntimeError(f"prefill_fast: reused metadata differs: {diffs}")

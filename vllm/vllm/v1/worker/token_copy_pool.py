# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded leases for single-request asynchronous token-result copies.

Buffers hold transport bytes only. A lease is returned after its copy event
has completed and the consumer has materialized independent Python output.
Abandoned or failed copies retain their lease; exhaustion uses the old path.
"""

import torch


class _TokenCopySlot:
    def __init__(self, dtype: torch.dtype):
        self.buffer = torch.empty((1, 7), device="cpu", dtype=dtype, pin_memory=True)
        self.event = torch.cuda.Event(blocking=True)
        self.views: dict[int, torch.Tensor] = {}
        self.in_use = False

    def view(self, width: int) -> torch.Tensor:
        view = self.views.get(width)
        if view is None:
            view = self.buffer[:, :width]
            self.views[width] = view
        return view


class TokenCopyPool:
    """One producer per model runner; at most two slots per integer dtype."""

    def __init__(self):
        self._device = None
        self._slots: dict[torch.dtype, list[_TokenCopySlot]] = {}

    def acquire(self, source: torch.Tensor) -> _TokenCopySlot | None:
        if (
            not source.is_cuda
            or source.ndim != 2
            or source.shape[0] != 1
            or not 1 <= source.shape[1] <= 7
            or source.dtype not in (torch.int32, torch.int64)
            or (self._device is not None and self._device != source.device)
        ):
            return None
        self._device = source.device
        slots = self._slots.setdefault(source.dtype, [])
        for slot in slots:
            if not slot.in_use:
                slot.in_use = True
                return slot
        if len(slots) == 2:
            return None
        slot = _TokenCopySlot(source.dtype)
        slot.in_use = True
        slots.append(slot)
        return slot

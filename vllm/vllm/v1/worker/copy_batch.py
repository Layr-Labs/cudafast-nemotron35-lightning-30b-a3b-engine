# SPDX-License-Identifier: Apache-2.0
"""LOCAL PATCH (cudafast-nemotron35): one H2D + one kernel for the target prep's
remaining small copies.

After the fused decode prep (decode_prep.py) the steady one-request spec step
still issues ~10 small copies between it and the target graph: the
``num_decode_draft_tokens`` upload, each Mamba2 group's state-index copy and
the shared accepted-count copy (mamba2_attn.py), FlashInfer's
``paged_kv_indptr`` / ``paged_kv_indices`` / ``paged_kv_last_page_len`` and
the spec-prefill plan's ``kv_lens`` upload (flashinfer.py). Each is a
``Tensor.copy_`` (pinned-allocator event, driver call: ~10-30 us of host time)
and ~3-5 us of serialized GPU time.

While a batch is open, ``copy(dst, src)`` records the copy instead: a host
source is snapshotted into a pinned staging buffer (the value the stock
non-blocking copy uploads: every such source is a buffer its owner does not
touch again before the copy runs), a device source is recorded by address.
``flush()`` (right before the target forward) sends the descriptors and the
staged bytes with ONE H2D and runs ONE kernel that performs the copies in
their original order (a barrier between copies), so later copies see earlier
ones exactly as stream order would. Nothing issued between ``begin`` and
``flush`` reads a recorded destination or writes a recorded device source
(the FlashInfer plan reads host tensors only). Byte moves only.

``VLLM_PREP_COPY_BATCH=0`` disables it; ``VLLM_PREP_COPY_BATCH_CHECK=1`` also
issues every copy immediately and compares the kernel's bytes at flush.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from vllm.triton_utils import tl, triton

ENABLED = os.environ.get("VLLM_PREP_COPY_BATCH", "1") != "0"
CHECK = os.environ.get("VLLM_PREP_COPY_BATCH_CHECK", "0") == "1"
# LOCAL PATCH: the kernel reads the descriptors and staged bytes straight from
# the pinned buffer (mapped host memory, coherent on GB10) instead of after an
# H2D copy; the guard event then follows the kernel. VLLM_PREP_ZERO_COPY=0
# keeps the H2D.
ZERO_COPY = os.environ.get("VLLM_PREP_ZERO_COPY", "1") != "0"

_MAXD = 32
_HDR = 3 * _MAXD * 8
_CAP = 1 << 16
_DBLOCK = triton.next_power_of_2(3 * _MAXD)


@triton.jit(do_not_specialize=["desc", "N"])
def _copy_batch_kernel(desc, N, BLOCK: tl.constexpr, DBLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    od = tl.arange(0, DBLOCK)
    table = tl.load(desc + od, mask=od < 3 * N, other=0)  # one fetch of all descriptors
    for i in range(N):
        dst = tl.sum(tl.where(od == 3 * i, table, 0)).to(tl.pointer_type(tl.uint8))
        src = tl.sum(tl.where(od == 3 * i + 1, table, 0)).to(tl.pointer_type(tl.uint8))
        n = tl.sum(tl.where(od == 3 * i + 2, table, 0))
        for off in range(0, n, BLOCK):
            m = off + o < n
            tl.store(dst + off + o, tl.load(src + off + o, mask=m), mask=m)
        tl.debug_barrier()


class _Set:
    def __init__(self, device) -> None:
        self.cpu = torch.zeros(_CAP, dtype=torch.uint8, pin_memory=True)
        self.np = self.cpu.numpy()
        self.desc = self.np[:_HDR].view(np.int64)
        self.gpu = torch.zeros(_CAP, dtype=torch.uint8, device=device)
        self.gdesc = self.gpu.view(torch.int64)
        self.cdesc = self.cpu.view(torch.int64)
        self.src_base = self.cpu.data_ptr() if ZERO_COPY else self.gpu.data_ptr()
        self.event = torch.cuda.Event()
        self.recorded = False
        # Deferred device pointers retain their tensor owners until the
        # kernel completion event, even when the caller passed a temporary.
        self.owners: list[torch.Tensor] = []


class CopyBatch:
    def __init__(self, device) -> None:
        self.sets = [_Set(device), _Set(device)]
        self.cur = 1
        self.active = False
        self.n = 0
        self.off = _HDR
        self.checks: list = []
        self.flushes = 0
        self.check_steps = 0
        self.check_diffs = 0
        self.pending_owners: list[torch.Tensor] = []
        s = self.sets[0]
        # compile at boot (a zero-copy launch), never inside a timed step
        _copy_batch_kernel[(1,)](s.cdesc if ZERO_COPY else s.gdesc, 0,
                                 BLOCK=1024, DBLOCK=_DBLOCK, num_warps=4)
        torch.cuda.synchronize()

    def begin(self) -> None:
        self.cur ^= 1
        s = self.sets[self.cur]
        if s.recorded:
            s.event.synchronize()  # its copy ran two steps ago
        s.owners.clear()
        self.active = True
        self.n = 0
        self.off = _HDR
        self.checks = []
        self.pending_owners = []

    def _immediate(self, dst, src) -> None:
        dst.copy_(src, non_blocking=True)

    def copy(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        s = self.sets[self.cur]
        nb = dst.numel() * dst.element_size()
        ok = (
            dst.is_cuda and dst.is_contiguous() and src.is_contiguous()
            and dst.dtype == src.dtype and dst.shape == src.shape
            and self.n < _MAXD
        )
        if ok and not src.is_cuda:
            ok = self.off + nb <= _CAP
        if not ok:
            # keep stream order: earlier recorded copies first
            self.flush()
            self.begin()
            self._immediate(dst, src)
            return
        i = self.n
        self.pending_owners.extend((dst, src))
        if src.is_cuda:
            s.desc[3 * i + 1] = src.data_ptr()
        else:
            s.np[self.off:self.off + nb] = src.numpy().view(np.uint8).reshape(-1)
            s.desc[3 * i + 1] = s.src_base + self.off
            self.off += (nb + 15) & ~15
        s.desc[3 * i] = dst.data_ptr()
        s.desc[3 * i + 2] = nb
        self.n += 1
        if CHECK:
            self._immediate(dst, src)
            self.checks.append(dst)

    def flush(self) -> None:
        if not self.active:
            return
        self.active = False
        if self.n == 0:
            return
        s = self.sets[self.cur]
        s.owners = self.pending_owners
        self.pending_owners = []
        if not ZERO_COPY:
            s.gpu[:self.off].copy_(s.cpu[:self.off], non_blocking=True)
        ref = [d.clone() for d in self.checks] if CHECK else None
        _copy_batch_kernel[(1,)](s.cdesc if ZERO_COPY else s.gdesc, self.n,
                                 BLOCK=1024, DBLOCK=_DBLOCK, num_warps=4)
        # Covers both the descriptor/staging read and every recorded device
        # source read/destination write, with or without the preceding H2D.
        s.event.record()
        s.recorded = True
        self.flushes += 1
        if CHECK:
            torch.cuda.synchronize()
            self.check_steps += 1
            bad = [k for k, (a, b) in enumerate(zip(ref, self.checks)) if not torch.equal(a, b)]
            if bad:
                self.check_diffs += 1
                print(f"copy_batch CHECK MISMATCH copies {bad}", flush=True)
            if self.check_steps % 200 == 0 or bad:
                print(f"copy_batch CHECK flushes {self.check_steps} diffs {self.check_diffs} "
                      f"copies/flush {self.n}", flush=True)


_BATCH: CopyBatch | None = None


def get(device) -> CopyBatch:
    global _BATCH
    if _BATCH is None:
        _BATCH = CopyBatch(device)
    return _BATCH


def copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    """dst.copy_(src, non_blocking=True), recorded while a batch is open."""
    b = _BATCH
    if b is not None and b.active:
        b.copy(dst, src)
    else:
        dst.copy_(src, non_blocking=True)


def materialize() -> None:
    """Publish recorded writes before an opaque external planning call.

    Reopen the batch so later metadata copies can still be coalesced. This
    preserves stream order even if a fallback plan consumes a device buffer.
    """
    b = _BATCH
    if b is not None and b.active:
        b.flush()
        b.begin()

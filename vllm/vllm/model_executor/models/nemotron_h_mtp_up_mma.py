"""Optional packed native-MTP MoE up projection with replicated-row MMA.

Every checkpoint BF16 value and every dot-product term is retained. Sixteen
identical private accumulator rows use tensor cores; only row zero is stored.
The FP32 accumulation tree differs from the drafter GEMV. This is not an exact
draft transformation. All qualification runs only in the official resident.
"""
import os
import statistics

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as tl
from triton.experimental.gluon.language import (
    BlockedLayout, DotOperandLayout, NVMMADistributedLayout, SliceLayout,
)
from triton.experimental.gluon.language.nvidia.ampere import mma_v2

from .nemotron_h_mtp_packed import load_packed, packed_moe_up, pdl_prologue

_ENABLED = os.environ.get("MTP_UP_ROW_MMA", "1") != "0"
_ADMITTED = {}


@gluon.jit
def _up(x, RSM, REX, RBASE, RRI, RRAW, SSM, SEX, SBASE, SRI, SRAW,
        ids, h, gate, TOPK: tl.constexpr, I: tl.constexpr = 1856,
        K: tl.constexpr = 2688, BN: tl.constexpr = 64,
        BK: tl.constexpr = 128, BM: tl.constexpr = 16,
        WORD: tl.constexpr = False, GATED: tl.constexpr = False,
        launch_pdl: tl.constexpr = False):
    if launch_pdl:
        pdl_prologue()
    ld: tl.constexpr = BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0])
    mma: tl.constexpr = NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[1, 4], instr_shape=[16, 8])
    slot = tl.program_id(1)
    nn = tl.program_id(0) * BN + tl.arange(0, BN, layout=SliceLayout(1, ld))
    mm = tl.arange(0, BM, layout=SliceLayout(1, ld))
    kk0 = tl.arange(0, BK, layout=SliceLayout(0, ld))
    if GATED:
        if tl.load(gate) != 0:
            tl.store(h + slot * I + nn, 0.0, mask=nn < I)
            return
    if slot < TOPK:
        base = tl.load(ids + slot).to(tl.int64) * I * K
    else:
        base = (slot - TOPK).to(tl.int64) * I * K
    acc = tl.full((BM, BN), 0.0, tl.float32, mma)
    # K=2688 is exactly21 K128 tiles. No zero-padded tail MMAs are added.
    for k0 in range(0, K, BK):
        kk = k0 + kk0
        index = base + nn[:, None].to(tl.int64) * K + kk[None, :]
        mask = (nn[:, None] < I) & (kk[None, :] < K)
        if slot < TOPK:
            w = load_packed(RSM, REX, RBASE, RRI, RRAW, index, mask, WORD=WORD)
        else:
            w = load_packed(SSM, SEX, SBASE, SRI, SRAW, index, mask, WORD=WORD)
        # All16 rows use the same current x, rather than request-derived cache.
        a = tl.load(x + mm[:, None] * 0 + kk[None, :], mask=kk[None, :] < K, other=0.0)
        a = tl.convert_layout(a, DotOperandLayout(0, mma, 2))
        b = tl.convert_layout(tl.permute(w, (1, 0)), DotOperandLayout(1, mma, 2))
        acc = mma_v2(a, b, acc)
    out: tl.constexpr = BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    acc = tl.convert_layout(acc, out)
    om = tl.arange(0, BM, layout=SliceLayout(1, out))
    on = tl.program_id(0) * BN + tl.arange(0, BN, layout=SliceLayout(0, out))
    value = tl.maximum(acc, 0.0)
    value = value * value
    tl.store(h + slot * I + on[None, :], value.to(h.dtype.element_ty),
             mask=(om[:, None] == 0) & (on[None, :] < I))


def _identity(p, shared):
    return (p.planes[0].device.index, p.planes[0].data_ptr(), shared.planes[0].data_ptr())


def _launch(p, shared, x, ids, h, gate, gated, word, variant, pdl=False):
    kernel_gate = gate if gated else x
    if variant:
        _up[(29, ids.numel() + 2)](
            x, *p.planes, *shared.planes, ids, h, kernel_gate,
            TOPK=ids.numel(), WORD=word, GATED=gated, launch_pdl=pdl,
            num_warps=4, num_stages=3)
    else:
        packed_moe_up[(928, ids.numel() + 2)](
            x, *p.planes, *shared.planes, ids, h, 1856, 2688,
            1856 * 2688, 2688, 2688, kernel_gate,
            TOPK=ids.numel(), BN=2, BK=1024, EVICT=True, GATED=gated,
            WN=2, WORD=word, launch_pdl=pdl, num_warps=8, num_stages=3)


@torch.no_grad()
def prepare(p, shared):
    """Private actual-expert probes and cold kernel timing, official remote only.

    This optional admission bounds finite private output differences; it is not
    proof of all-input bit equality, draft quality or final output correctness.
    Any compiler, comparison or speed failure keeps the existing packed GEMV.
    """
    if (not _ENABLED or tuple(p.shape) != (128, 1856, 2688)
            or tuple(shared.shape) != (3712, 2688)
            or torch.cuda.is_current_stream_capturing()):
        return
    key = _identity(p, shared)
    if key in _ADMITTED:
        return
    flags = {}
    _ADMITTED[key] = (p, shared, flags, False)
    try:
        from .nemotron_h_mtp_packed import word_load_enabled
        from .nemotron_h_mtp_kernels import mtp_pdl
        pdl = mtp_pdl()
        _ADMITTED[key] = (p, shared, flags, pdl)
        device = p.planes[0].device
        words = (False, True) if word_load_enabled(device) else (False,)
        gen = torch.Generator(device=device).manual_seed(26881856)
        gate = torch.zeros(3, dtype=torch.int32, device=device)
        probes = [torch.zeros((1, 2688), device=device, dtype=torch.bfloat16)]
        for scale in (0.001, 1.0, 32.0):
            probes.append((torch.randn((1, 2688), device=device, generator=gen) * scale).to(torch.bfloat16))
        for word in words:
            valid = True
            for topk in (3, 6):
                old = torch.empty((topk + 2, 1856), dtype=torch.bfloat16, device=device)
                new = torch.empty_like(old)
                for first in range(0, 128, topk):
                    ids = ((torch.arange(topk, device=device, dtype=torch.int32) + first) % 128).contiguous()
                    for x in probes:
                        _launch(p, shared, x, ids, old, gate, False, word, False, pdl)
                        _launch(p, shared, x, ids, new, gate, False, word, True, pdl)
                        # Same complete actual-expert output surface; no prompt,
                        # target token or measured acceptance chooses admission.
                        peak = old.float().abs().amax().item()
                        if (not bool(torch.isfinite(old).all() & torch.isfinite(new).all())
                                or not torch.allclose(old.float(), new.float(), rtol=0.015625,
                                                      atol=max(1.0, peak) * 1e-5)):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                # Compile both active/stopped gate specializations and check
                # stopped output bits without ever loading checkpoint weights.
                for stop in (0, 1):
                    gate.fill_(stop)
                    _launch(p, shared, probes[2], ids, old, gate, True, word, False, pdl)
                    _launch(p, shared, probes[2], ids, new, gate, True, word, True, pdl)
                    if stop and not torch.equal(old.view(torch.int16), new.view(torch.int16)):
                        valid = False
                    if not stop:
                        peak = old.float().abs().amax().item()
                        if (not bool(torch.isfinite(old).all() & torch.isfinite(new).all())
                                or not torch.allclose(old.float(), new.float(), rtol=0.015625,
                                                      atol=max(1.0, peak) * 1e-5)):
                            valid = False
                gate.zero_()
            if not valid:
                continue
            flush = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for topk in (3, 6):
                ids = torch.arange(topk, device=device, dtype=torch.int32)
                h = torch.empty((topk + 2, 1856), dtype=torch.bfloat16, device=device)
                times = {False: [], True: []}
                for trial in range(5):
                    for variant in ((False, True) if trial % 2 == 0 else (True, False)):
                        flush.zero_()
                        start.record()
                        _launch(p, shared, probes[2], ids, h, gate, True, word, variant, pdl)
                        end.record()
                        end.synchronize()
                        times[variant].append(start.elapsed_time(end))
                if statistics.median(times[True]) < 0.96 * statistics.median(times[False]):
                    flags[(topk, word)] = True
        print(f"MTP_UP_ROW_MMA admitted={flags} draft_fp32_tree_changed=True target_unchanged=True", flush=True)
    except Exception as exc:
        flags.clear()
        print(f"MTP_UP_ROW_MMA original GEMV retained: {exc!r}", flush=True)


def maybe_up(p, shared, x, ids, h, gate, gated, pdl):
    if not _ENABLED:
        return False
    entry = _ADMITTED.get(_identity(p, shared))
    if entry is None or entry[0] is not p or entry[1] is not shared or entry[3] != pdl:
        return False
    from .nemotron_h_mtp_packed import word_load_enabled
    word = word_load_enabled(x.device)
    if (not entry[2].get((ids.numel(), word), False)
            or x.shape != (1, 2688) or x.dtype != torch.bfloat16 or not x.is_cuda
            or h.shape != (ids.numel() + 2, 1856) or h.dtype != torch.bfloat16
            or ids.dtype not in (torch.int32, torch.int64)
            or not all(t.is_contiguous() and t.device == x.device for t in (x, ids, h))):
        return False
    _launch(p, shared, x, ids, h, gate, gated, word, True, pdl)
    return True

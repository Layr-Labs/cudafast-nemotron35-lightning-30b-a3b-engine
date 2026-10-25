"""Exact hidden conversion and screen gathers in one graph-internal CTA.

Private startup admission compares every output byte with the original
copy_ followed by the screen's actual gather kernel. This does not change
any head dot product, accumulation, candidate selection or depth gate.
"""
from __future__ import annotations

import logging
import os

import torch

from vllm.triton_utils import tl, triton

logger = logging.getLogger(__name__)


@triton.jit
def _stage_head(src, full, ia, oa, ib, ob, ic, oc, raw_out,
                H: tl.constexpr, NA: tl.constexpr, NB: tl.constexpr,
                NC: tl.constexpr, THREE: tl.constexpr,
                BH: tl.constexpr, BA: tl.constexpr, BB: tl.constexpr,
                BC: tl.constexpr, SAVE_RAW: tl.constexpr = False,
                launch_pdl: tl.constexpr = False):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    h = tl.arange(0, BH)
    value = tl.load(src + h, mask=h < H, other=0)
    if SAVE_RAW:
        tl.store(raw_out.to(tl.pointer_type(tl.uint16)) + h,
                 value.to(tl.uint16, bitcast=True), mask=h < H)
    tl.store(full + h, value.to(tl.float32), mask=h < H)
    a = tl.arange(0, BA)
    ai = tl.load(ia + a, mask=a < NA, other=0)
    av = tl.load(src + ai, mask=a < NA, other=0).to(tl.float32)
    tl.store(oa + a, av, mask=a < NA)
    b = tl.arange(0, BB)
    bi = tl.load(ib + b, mask=b < NB, other=0)
    bv = tl.load(src + bi, mask=b < NB, other=0).to(tl.float32)
    tl.store(ob + b, bv, mask=b < NB)
    if THREE:
        c = tl.arange(0, BC)
        ci = tl.load(ic + c, mask=c < NC, other=0)
        cv = tl.load(src + ci, mask=c < NC, other=0).to(tl.float32)
        tl.store(oc + c, cv, mask=c < NC)


def _same(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


class HeadStage:
    def __init__(self, screen):
        self.screen = screen
        self.h = screen._h
        self.three = bool(getattr(screen, "pre", False))
        if self.three:
            self.cols = (screen.pre_cols, screen.mid_cols, screen.rest_cols)
            self.outs = (screen._hp, screen._hm, screen._hr)
        else:
            self.cols = (screen.coarse_cols, screen.rest_cols)
            self.outs = (screen._hc, screen._hr)
        self.width = self.h.numel()
        self.owners = (self.h, *self.cols, *self.outs)

    def run(self, src, full=None, outs=None, pdl=False, raw_out=None):
        full = self.h if full is None else full
        outs = self.outs if outs is None else outs
        if (src.ndim != 1 or src.numel() != self.width or src.stride(0) != 1
                or src.dtype not in (torch.bfloat16, torch.float16)
                or src.device != self.h.device):
            return False
        if raw_out is not None and (
            raw_out.shape != src.shape or raw_out.dtype != src.dtype
            or raw_out.device != src.device or raw_out.stride(0) != 1
        ):
            return False
        a, b = self.cols[:2]
        c = self.cols[2] if self.three else None
        _stage_head[(1,)](
            src, full, a, outs[0], b, outs[1], c,
            outs[2] if self.three else None, raw_out,
            H=self.width, NA=a.numel(), NB=b.numel(),
            NC=c.numel() if self.three else 1, THREE=self.three,
            BH=triton.next_power_of_2(self.width),
            BA=triton.next_power_of_2(a.numel()),
            BB=triton.next_power_of_2(b.numel()),
            BC=triton.next_power_of_2(c.numel()) if self.three else 1,
            SAVE_RAW=raw_out is not None, launch_pdl=pdl, num_warps=4,
        )
        return True


@torch.no_grad()
def prepare_head_stage(screen):
    """Startup only, actual immutable gather columns, wholly private outputs."""
    if os.environ.get("NEMO_MTP_TAIL_HEAD_FUSION", "1") == "0":
        return None
    try:
        if (torch.cuda.is_current_stream_capturing() or not screen.cascade
                or not screen.fused_topk or not screen.reuse):
            return None
        stage = HeadStage(screen)
        if (not 0 < stage.width <= 4096 or stage.h.dtype != torch.float32
                or not stage.h.is_cuda or not stage.h.is_contiguous()
                or any(c.dtype != torch.int64 or not c.is_contiguous()
                       or c.device != stage.h.device for c in stage.cols)
                or any(o.dtype != torch.float32 or not o.is_contiguous()
                       or o.numel() != c.numel() or o.device != stage.h.device
                       for c, o in zip(stage.cols, stage.outs))):
            return None
        from vllm.model_executor.models.nemotron_h_mtp_kernels import mtp_pdl
        from vllm.model_executor.models.nemotron_h_mtp_screen import _gather_two

        if stage.three:
            from vllm.model_executor.models.nemotron_h_mtp_screen import _gather_three
        dev = stage.h.device
        # The signed-int16 view enumerates all 65536 raw bit patterns, including
        # signed zero, NaN payloads, infinities and subnormals. No RNG or request
        # input. A compiler-dependent conversion mismatch disables only this
        # optional fusion; the original head path remains available.
        for dtype in (torch.bfloat16, torch.float16):
            for pdl in ((False, True) if mtp_pdl() else (False,)):
                for start in range(0, 65536, stage.width):
                    raw = ((torch.arange(stage.width, device=dev, dtype=torch.int32)
                            + start) & 65535).to(torch.int16)
                    src = raw.view(dtype)
                    sizes = (stage.width,) + tuple(c.numel() for c in stage.cols)
                    ref = [torch.full((n + 16,), 0.03125, dtype=torch.float32,
                                      device=dev) for n in sizes]
                    got = [t.clone() for t in ref]
                    rv = [t[8:8 + n] for t, n in zip(ref, sizes)]
                    gv = [t[8:8 + n] for t, n in zip(got, sizes)]
                    ref_raw = torch.full((stage.width + 16,), 12345,
                                         dtype=torch.int16, device=dev)
                    got_raw = ref_raw.clone()
                    ref_raw.view(dtype)[8:8 + stage.width].copy_(src)
                    rv[0].copy_(src)
                    a, b = stage.cols[:2]
                    if stage.three:
                        c = stage.cols[2]
                        _gather_three[(1,)](
                            rv[0], a, rv[1], b, rv[2], c, rv[3],
                            NA=a.numel(), NB=b.numel(), NC=c.numel(),
                            launch_pdl=pdl,
                        )
                    else:
                        _gather_two[(1,)](
                            rv[0], a, rv[1], b, rv[2],
                            NA=a.numel(), NB=b.numel(), launch_pdl=pdl,
                        )
                    if not stage.run(src, gv[0], tuple(gv[1:]), pdl=pdl,
                                     raw_out=got_raw.view(dtype)[8:8 + stage.width]):
                        return None
                    torch.cuda.synchronize(dev)
                    if (not all(_same(a, b) for a, b in zip(ref, got))
                            or not _same(ref_raw, got_raw)):
                        return None
                    plain = [torch.full_like(t, 0.03125) for t in ref]
                    pv = [t[8:8 + n] for t, n in zip(plain, sizes)]
                    if not stage.run(src, pv[0], tuple(pv[1:]), pdl=pdl):
                        return None
                    torch.cuda.synchronize(dev)
                    if not all(_same(a, b) for a, b in zip(ref, plain)):
                        return None
        return stage
    except Exception as exc:
        logger.warning("Native MTP fused head staging kept original path: %r", exc)
        return None

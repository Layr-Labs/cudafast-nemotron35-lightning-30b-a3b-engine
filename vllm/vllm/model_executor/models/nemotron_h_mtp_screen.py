# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy draft-token head for the Nemotron-H MTP drafter: cascaded partial-K
screens plus a full-K rescore of the survivors.

This experiment adds an 8-group/512 survivor stage before a original32-group/256
stage and the original64-group/32/full-K stages. New RMODE3 folds three index
levels into the dot and final selector. Fold admission compares every stage.
VLLM_MTP_SCREEN_ULTRA=0 retains the original32/64/full-K cascade.

DRAFTER ONLY. The target's lm_head, logits and verification are untouched;
the drafter's lm_head is the target's shared NVFP4 head (131072 x 2688, about
198 MB), and the drafter reads all of it three times per depth-3 cycle only to
take one argmax per draft. This replaces that argmax:

1. screen every vocabulary row on eight important whole NVFP4 groups;
2. keep512 rows per2048-row chunk, rescore on32 groups and keep256;
3. rescore on the original64 importance groups and keep32 per chunk;
4. rescore those rows on the full K in fp32, round to bf16 as the stock head
   does, and take the maximum with the lowest token id among equals (the rule
   torch.argmax applies to the stock bf16 logits).

Importance is sum over k of final_norm_gamma[k]^2 * ||W[:, k]||^2. Both group
sets are selected by importance rank before being sorted back into K order.
The screen and the full copy are built once from the pinned checkpoint's own
lm_head tensors (raw NVFP4 bytes, fp8 group scales, fp32 global scale): an
input-independent load-time repack, no format change, no new weights. The
proposal can differ when either screen removes a candidate or the rescore's
different fp32 accumulation changes rounded logits. The target's verification decides
every committed token, and the timed window is still checked against its oracle.

VLLM_MTP_SCREEN_CASCADE=0 restores the original 64-group single screen.
VLLM_MTP_SCREEN=0 restores the stock head.
"""

import json
import os

import torch

# l2pf (harness/kernels/l2pf); absent, nothing changes.
try:
    import l2pf as _l2pf
except ImportError:
    _l2pf = None

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

from .nemotron_h_mtp_kernels import DEPTH_THETA, _gate_args, depth_gate_init, mtp_pdl

logger = init_logger(__name__)

SCREEN_FRAC_GROUPS = 64  # of 168 groups: 1024 of 2688 hidden dims (512 bytes per row)
SCREEN_CHUNK = 2048
SCREEN_PER = 32
CASCADE_FRAC_GROUPS = 32
CASCADE_PER = 256
ULTRA_FRAC_GROUPS = 8
ULTRA_PER = 512


def screen_enabled() -> bool:
    return os.environ.get("VLLM_MTP_SCREEN", "1") != "0"


def screen_graph_enabled() -> bool:
    """Kill switch for the per-draft-row capture: VLLM_MTP_SCREEN_GRAPH=0."""
    return os.environ.get("VLLM_MTP_SCREEN_GRAPH", "1") != "0"


@triton.jit
def _e2m1(nib):
    # fp4 e2m1: s ee m -> +-{0, .5, 1, 1.5, 2, 3, 4, 6}
    e = (nib >> 1) & 3
    m = (nib & 1).to(tl.float32)
    p2 = tl.where(e == 1, 1.0, tl.where(e == 2, 2.0, 4.0))
    mag = tl.where(e == 0, m * 0.5, (1.0 + m * 0.5) * p2)
    return tl.where((nib & 8) != 0, -mag, mag)


@triton.jit
def _e2m1_pair(byte):
    # Exact hardware decode of both nibbles; FP4 values all fit in FP16/FP32.
    lo, hi = tl.inline_asm_elementwise(
        """{
        .reg .b8 a, b, c, d;
        .reg .b32 pair;
        .reg .b16 lo, hi;
        mov.b32 {a, b, c, d}, $2;
        cvt.rn.f16x2.e2m1x2 pair, a;
        mov.b32 {lo, hi}, pair;
        cvt.f32.f16 $0, lo;
        cvt.f32.f16 $1, hi;
        }""",
        constraints="=f,=f,r", args=[byte],
        dtype=(tl.float32, tl.float32), is_pure=True, pack=1,
    )
    # The software decoder canonicalizes E2M1 -0 to +0 in FP32. Hardware
    # preserves -0; match the reference even for the two zero encodings.
    return tl.where(lo == 0.0, 0.0, lo), tl.where(hi == 0.0, 0.0, hi)


@triton.jit
def _nvfp4_rows_dot(
    h_ptr,  # [2 * NBYTES] fp32
    w_ptr,  # [N, NBYTES] uint8, element 2j in the low nibble of byte j
    s_ptr,  # [N, NBYTES // 8] fp8 e4m3 group scales
    rows_ptr,  # [R] int64 row ids, or unused when GATHER is False
    out_ptr,  # [R] fp32
    R,
    gscale,
    NBYTES: tl.constexpr,
    GATHER: tl.constexpr,
    BR: tl.constexpr,
    BB: tl.constexpr,
    INDEX32: tl.constexpr = False,
    gate_ptr=None,
    GATED: tl.constexpr = False,
    launch_pdl: tl.constexpr = False,
    hcols_ptr=None,  # [2 * NBYTES] int64 columns of h to read, when HCOLS
    HCOLS: tl.constexpr = False,
    ref2_ptr=None,
    MPER: tl.constexpr = 1,
    ref_ptr=None,  # [R] int64 second-stage top-k indices, when RMODE == 2
    RMODE: tl.constexpr = 0,
    CHUNK: tl.constexpr = 1,
    CPER: tl.constexpr = 1,
    SPER: tl.constexpr = 1,
):
    """RMODE 0: rows from GATHER as before. 1: rows = rows_ptr[r] + (r // CPER) * CHUNK
    (first-stage top-k indices within their chunk). 2: rows = rows_ptr[c * CPER + ref_ptr[r]]
    + c * CHUNK with c = r // SPER (the second stage's picks among them).
    RMODE3 composes ref_ptr into ref2_ptr before looking up rows_ptr. HCOLS reads
    h[hcols_ptr[j]] where h[j] was read: the same elements a gathered copy holds."""
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    rmask = r < R
    # weight/scale-load mask: off once the depth gate stopped the draft loop (zeros out)
    wr = rmask
    if GATED:
        wr = rmask & (tl.load(gate_ptr) == 0)
    if RMODE == 1:
        rows = tl.load(rows_ptr + r, mask=rmask, other=0) + (r // CPER) * CHUNK
        if INDEX32:
            rows = rows.to(tl.int32)
        else:
            rows = rows.to(tl.int64)
    elif RMODE == 2:
        c = r // SPER
        sel = tl.load(ref_ptr + r, mask=rmask, other=0)
        rows = tl.load(rows_ptr + c * CPER + sel, mask=rmask, other=0) + c * CHUNK
        if INDEX32:
            rows = rows.to(tl.int32)
        else:
            rows = rows.to(tl.int64)
    elif RMODE == 3:
        c = r // SPER
        sel = tl.load(ref_ptr + r, mask=rmask, other=0)
        mid = tl.load(ref2_ptr + c * MPER + sel, mask=rmask, other=0)
        rows = tl.load(rows_ptr + c * CPER + mid, mask=rmask, other=0) + c * CHUNK
        if INDEX32:
            rows = rows.to(tl.int32)
        else:
            rows = rows.to(tl.int64)
    elif GATHER:
        if INDEX32:
            rows = tl.load(rows_ptr + r, mask=rmask, other=0).to(tl.int32)
        else:
            rows = tl.load(rows_ptr + r, mask=rmask, other=0).to(tl.int64)
    else:
        if INDEX32:
            rows = r.to(tl.int32)
        else:
            rows = r.to(tl.int64)
    acc = tl.zeros((BR,), dtype=tl.float32)
    for b0 in range(0, NBYTES, BB):
        ob = b0 + tl.arange(0, BB)
        wb = tl.load(
            w_ptr + rows[:, None] * NBYTES + ob[None, :],
            mask=wr[:, None],
            other=0,
            eviction_policy="evict_first",
        ).to(tl.int32)
        og = (b0 // 8) + tl.arange(0, BB // 8)
        sc = tl.load(
            s_ptr + rows[:, None] * (NBYTES // 8) + og[None, :],
            mask=wr[:, None],
            other=0.0,
        ).to(tl.float32)
        if HCOLS:
            hl = tl.load(h_ptr + tl.load(hcols_ptr + 2 * ob))
            hh = tl.load(h_ptr + tl.load(hcols_ptr + 2 * ob + 1))
        else:
            hl = tl.load(h_ptr + 2 * ob)
            hh = tl.load(h_ptr + 2 * ob + 1)
        lo, hi = _e2m1_pair(wb)
        part = lo * hl[None, :] + hi * hh[None, :]
        part = tl.reshape(part, (BR, BB // 8, 8))
        acc += tl.sum(tl.sum(part, axis=2) * sc, axis=1)
    tl.store(out_ptr + r, acc * gscale, mask=rmask)


def _rows_dot(h, w, s, rows, gscale, out, gather, gated=False, hcols=None, ref=None,
              rmode=0, cper=1, sper=1, ref2=None, mper=1):
    n_out = out.numel()
    nbytes = w.shape[1]
    gather = gather or rmode != 0
    br = 64 if not gather else 16
    max_rows = max(w.shape[0], n_out) + br - 1
    index32 = (
        max_rows * nbytes < 2**31
        and max_rows * (nbytes // 8) < 2**31
        and n_out + br - 1 < 2**31
    )
    gate, ok = _gate_args(out)
    gated = gated and ok
    _nvfp4_rows_dot[(triton.cdiv(n_out, br),)](
        h, w, s, rows if gather else out, out, n_out, gscale,
        NBYTES=nbytes, GATHER=gather, BR=br, BB=64, INDEX32=index32,
        gate_ptr=gate, GATED=gated,
        num_warps=4, num_stages=2,
        launch_pdl=mtp_pdl(),
        hcols_ptr=hcols, HCOLS=hcols is not None,
        ref_ptr=ref, RMODE=rmode, CHUNK=SCREEN_CHUNK, CPER=cper, SPER=sper,
        ref2_ptr=ref2, MPER=mper,
    )


@triton.jit
def _select_token(
    logits, ids, out, N: tl.constexpr, BLOCK: tl.constexpr,
    gate_ptr=None,
    GATED: tl.constexpr = False,
    THETA: tl.constexpr = 0.0,
    launch_pdl: tl.constexpr = False,
    ref=None,
    TWO_LEVEL: tl.constexpr = False,
    THREE_LEVEL: tl.constexpr = False,
    ref2=None,
    MPER: tl.constexpr = 1,
    CHUNK: tl.constexpr = 1,
    CPER: tl.constexpr = 1,
    SPER: tl.constexpr = 1,
):
    """Stock BF16 maximum, then minimum token id among ties, in one launch.
    TWO_LEVEL: the token ids are ids[c * CPER + ref[i]] + c * CHUNK, c = i // SPER."""
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    i = tl.arange(0, BLOCK)
    # Round before reduction: FP32 logit differences can disappear in BF16.
    v = tl.load(logits + i, mask=i < N, other=-float("inf"))
    v = v.to(tl.bfloat16).to(tl.float32)
    if THREE_LEVEL:
        c = i // SPER
        sel = tl.load(ref + i, mask=i < N, other=0)
        mid = tl.load(ref2 + c * MPER + sel, mask=i < N, other=0)
        token = tl.load(ids + c * CPER + mid, mask=i < N, other=0) + c * CHUNK
        token = tl.where(i < N, token, 1 << 30)
    elif TWO_LEVEL:
        c = i // SPER
        sel = tl.load(ref + i, mask=i < N, other=0)
        token = tl.load(ids + c * CPER + sel, mask=i < N, other=0) + c * CHUNK
        token = tl.where(i < N, token, 1 << 30)
    else:
        token = tl.load(ids + i, mask=i < N, other=1 << 30)
    best = tl.max(v, axis=0)
    # torch.max propagates NaNs; its subsequent equality tests all fail.
    any_nan = tl.max(((i < N) & (v != v)).to(tl.int32), axis=0)
    chosen = tl.min(tl.where((i < N) & (v == best), token, 1 << 30), axis=0)
    tl.store(out, tl.where(any_nan != 0, 1 << 30, chosen))
    if GATED:
        # Depth gate (nemotron_h_mtp_kernels.py): while a draft loop is active and
        # not stopped, count this draft as kept, then stop the loop if its top-1
        # probability over the candidate logits is below THETA (NaN never stops it).
        if (tl.load(gate_ptr + 2) != 0) & (tl.load(gate_ptr) == 0):
            tl.store(gate_ptr + 1, tl.load(gate_ptr + 1) + 1)
            denom = tl.sum(tl.where(i < N, tl.exp(v - best), 0.0), axis=0)
            if 1.0 / denom < THETA:
                tl.store(gate_ptr, 1)


try:
    import screen_radix as _screen_radix
except ImportError:
    _screen_radix = None

_NATIVE_TOPK = os.environ.get("MTP_SCREEN_RADIX", "1") != "0"


def _screen_topk(scores, width, keep, offset=0):
    if _NATIVE_TOPK and _screen_radix is not None:
        selected = _screen_radix.maybe_topk(scores, width, keep, offset)
        if selected is not None:
            return selected
    selected = scores.view(-1, width).topk(keep, dim=-1, sorted=False).indices
    if offset:
        rows = torch.arange(selected.shape[0], device=scores.device,
                            dtype=selected.dtype).view(-1, 1)
        selected = selected + rows * offset
    return selected


class DraftHeadScreen:
    """Load-time buffers plus the greedy screen for one shared NVFP4 head."""

    def __init__(self, snapshot_dir: str, device: torch.device):
        from safetensors import safe_open

        idx = json.load(open(os.path.join(snapshot_dir, "model.safetensors.index.json")))
        wmap = idx["weight_map"]

        def get(name):
            with safe_open(os.path.join(snapshot_dir, wmap[name]), "pt", device="cpu") as f:
                return f.get_tensor(name)

        w = get("lm_head.weight")  # [N, K/2] uint8
        s = get("lm_head.weight_scale")  # [N, K/16] fp8 e4m3
        g = float(get("lm_head.weight_scale_2").float().reshape(-1)[0])
        gamma = None
        for name in wmap:
            if name.startswith("mtp.") and name.endswith("final_layernorm.weight"):
                gamma = get(name).float()
        n, kb = w.shape
        k = kb * 2
        ng = k // 16
        assert w.dtype == torch.uint8 and s.shape == (n, ng), (w.shape, s.shape)
        assert n % SCREEN_CHUNK == 0
        self.cascade = os.environ.get("VLLM_MTP_SCREEN_CASCADE", "1") != "0"
        self.ultra = self.cascade and os.environ.get("VLLM_MTP_SCREEN_ULTRA", "1") != "0"
        if self.ultra:
            assert ULTRA_FRAC_GROUPS == 8 and CASCADE_FRAC_GROUPS == 32
            assert SCREEN_PER <= CASCADE_PER <= ULTRA_PER <= SCREEN_CHUNK
        # VLLM_MTP_SCREEN_FOLD=0 keeps the cascade's separate gather kernels.
        self.fold = os.environ.get("VLLM_MTP_SCREEN_FOLD", "1") != "0"
        if self.cascade:
            assert 0 < CASCADE_FRAC_GROUPS < SCREEN_FRAC_GROUPS <= ng
            assert SCREEN_PER <= CASCADE_PER <= SCREEN_CHUNK
        # Weight-only importance of each 16-element group.
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
        )
        colsq = torch.zeros(k, dtype=torch.float64)
        for r0 in range(0, n, 16384):
            wb = w[r0:r0 + 16384].long()
            vals = torch.stack([lut[wb & 15], lut[wb >> 4]], -1).reshape(wb.shape[0], k)
            vals = vals * s[r0:r0 + 16384].float().repeat_interleave(16, 1)
            colsq += (vals.double() ** 2).sum(0)
        imp = colsq if gamma is None else colsq * gamma.double() ** 2
        ranked_groups = imp.view(ng, 16).sum(1).argsort(descending=True)
        groups = ranked_groups[:SCREEN_FRAC_GROUPS].sort().values
        cols = (groups[:, None] * 16 + torch.arange(16)[None, :]).reshape(-1)
        self.gscale = g
        self.n = n
        self.w_full = w.to(device).contiguous()
        self.s_full = s.to(device).contiguous()
        self.w_scr = w.view(n, ng, 8)[:, groups, :].reshape(n, -1).to(device).contiguous()
        self.s_scr = s[:, groups].to(device).contiguous()
        self.cols = cols.to(device)
        if self.cascade:
            # Select by importance rank, not the first half of K-sorted groups.
            coarse_groups = ranked_groups[:CASCADE_FRAC_GROUPS].sort().values
            coarse_cols = (
                coarse_groups[:, None] * 16 + torch.arange(16)[None, :]
            ).reshape(-1)
            self.w_coarse = (
                w.view(n, ng, 8)[:, coarse_groups, :]
                .reshape(n, -1).to(device).contiguous()
            )
            self.s_coarse = s[:, coarse_groups].to(device).contiguous()
            self.coarse_cols = coarse_cols.to(device)
            self.refine_scores = torch.empty(
                (n // SCREEN_CHUNK) * CASCADE_PER,
                dtype=torch.float32, device=device,
            )
        if self.ultra:
            ultra_groups = ranked_groups[:ULTRA_FRAC_GROUPS].sort().values
            ultra_cols = (ultra_groups[:, None] * 16 + torch.arange(16)[None, :]).reshape(-1)
            self.w_ultra = w.view(n, ng, 8)[:, ultra_groups, :].reshape(n, -1).to(device).contiguous()
            self.s_ultra = s[:, ultra_groups].to(device).contiguous()
            self.ultra_cols = ultra_cols.to(device)
            self.coarse_refine_scores = torch.empty(
                (n // SCREEN_CHUNK) * ULTRA_PER, dtype=torch.float32, device=device)
        self.chunk_off = (
            torch.arange(n // SCREEN_CHUNK, device=device) * SCREEN_CHUNK
        )[:, None]
        self.scores = torch.empty(n, dtype=torch.float32, device=device)
        self.cand_logits = torch.empty(
            (n // SCREEN_CHUNK) * SCREEN_PER, dtype=torch.float32, device=device
        )
        self.big = torch.tensor(1 << 30, dtype=torch.int32, device=device)
        # l2pf: the coarse matrix is registered for the MTP MoE's down
        # projection window (nemotron_h_mtp_kernels.py), which prefetches its head.
        if _l2pf is not None and self.cascade:
            _l2pf.DRAFT_SCREEN[:] = [self.w_ultra if self.ultra else self.w_coarse]
        if device.type == "cuda" and _NATIVE_TOPK and _screen_radix is not None:
            _screen_radix.available(device)
        self.graph = None
        self._h = torch.empty(k, dtype=torch.float32, device=device)
        self._out = torch.empty(1, dtype=torch.int32, device=device)
        if device.type == "cuda" and self.cascade and self.fold:
            self.fold = self._fold_matches(k, device)
        if device.type == "cuda" and screen_graph_enabled():
            # The captured one-row screen is the gated one (depth gate): allocate
            # the gate first so the graph captures its address.
            gate = depth_gate_init(device)
            try:
                self._h.zero_()
                for _ in range(2):
                    self._greedy_row(self._h, gated=True, out=self._out)
                torch.cuda.synchronize(device)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    self.graph, pool=torch.cuda.graph_pool_handle()
                ):
                    self._greedy_row(self._h, gated=True, out=self._out)
                torch.cuda.synchronize(device)
            except Exception as exc:
                self.graph = None
                logger.warning("MTP draft-head screen graph disabled: %s", exc)
            if gate is not None:
                gate.zero_()  # the warm-up rows above may have counted or stopped
        logger.info(
            "MTP draft-head screen: %d of %d groups, %d candidates, cascade=%s, graph=%s",
            SCREEN_FRAC_GROUPS, ng, self.cand_logits.numel(), self.cascade,
            self.graph is not None,
        )

    def _fold_matches(self, k: int, device: torch.device) -> bool:
        """Load time, before capture: the folded cascade against the unfolded one on
        random rows, bitwise on every stage's scores and on the token."""
        try:
            gen = torch.Generator(device=device)
            gen.manual_seed(1234)
            for probe in range(6):
                if probe == 0:
                    h = torch.zeros(k, dtype=torch.float32, device=device)
                else:
                    h = torch.randn(k, generator=gen, device=device) * 2.0
                    if probe <= 3:
                        h = h.to(torch.bfloat16).float()
                    if probe == 4:
                        idx = torch.arange(k, device=device)
                        h = torch.where(idx % 7 < 4, -torch.zeros_like(h), h)
                outs = []
                for fold in (True, False):
                    self.fold = fold
                    tok = self._greedy_row(h)
                    stages = [tok.clone(), self.scores.clone()]
                    if self.ultra:
                        stages.append(self.coarse_refine_scores.clone())
                    stages.extend([self.refine_scores.clone(), self.cand_logits.clone()])
                    outs.append(stages)
                for a, b in zip(*outs):
                    if not torch.equal(a.view(torch.int32), b.view(torch.int32)):
                        logger.warning("MTP screen: folded gathers differ; unfolded path kept")
                        return False
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("MTP screen: folded gathers unavailable (%r)", exc)
            return False
        finally:
            self.fold = True

    def _greedy_row(
        self, h: torch.Tensor, gated: bool = False, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """One row h [K] float32: the cascade (or the single screen) and the
        argmax tie rule, written to out (a fresh int32 scalar when None). Every
        buffer it touches is fixed-address, so the body is capturable as-is."""
        if out is None:
            out = torch.empty((), dtype=torch.int32, device=h.device)
        if self.cascade and self.fold:
            if self.ultra:
                _rows_dot(h, self.w_ultra, self.s_ultra, None,
                          self.gscale, self.scores, False, gated, hcols=self.ultra_cols)
                ultra_idx = _screen_topk(self.scores, SCREEN_CHUNK, ULTRA_PER)
                _rows_dot(h, self.w_coarse, self.s_coarse, ultra_idx,
                          self.gscale, self.coarse_refine_scores, True, gated,
                          hcols=self.coarse_cols, rmode=1, cper=ULTRA_PER)
                middle = _screen_topk(self.coarse_refine_scores, ULTRA_PER, CASCADE_PER)
                _rows_dot(h, self.w_scr, self.s_scr, ultra_idx,
                          self.gscale, self.refine_scores, True, gated, hcols=self.cols,
                          ref=middle, rmode=2, cper=ULTRA_PER, sper=CASCADE_PER)
                refined = _screen_topk(self.refine_scores, CASCADE_PER, SCREEN_PER)
                _rows_dot(h, self.w_full, self.s_full, ultra_idx,
                          self.gscale, self.cand_logits, True, gated,
                          ref=refined, ref2=middle, rmode=3, cper=ULTRA_PER,
                          mper=CASCADE_PER, sper=SCREEN_PER)
                gate, ok = _gate_args(out)
                _select_token[(1,)](
                    self.cand_logits, ultra_idx, out, N=self.cand_logits.numel(),
                    BLOCK=triton.next_power_of_2(self.cand_logits.numel()),
                    gate_ptr=gate, GATED=gated and ok, THETA=DEPTH_THETA,
                    launch_pdl=mtp_pdl(), ref=refined, ref2=middle,
                    THREE_LEVEL=True, CHUNK=SCREEN_CHUNK, CPER=ULTRA_PER,
                    MPER=CASCADE_PER, SPER=SCREEN_PER)
                return out
            # The same arithmetic as the unfolded cascade below, with the column
            # gathers of h, the chunk offsets and the candidate gather done as
            # index arithmetic inside the dot and select kernels.
            _rows_dot(
                h, self.w_coarse, self.s_coarse, None,
                self.gscale, self.scores, False, gated, hcols=self.coarse_cols,
            )
            coarse_idx = _screen_topk(self.scores, SCREEN_CHUNK, CASCADE_PER)
            _rows_dot(
                h, self.w_scr, self.s_scr, coarse_idx,
                self.gscale, self.refine_scores, True, gated, hcols=self.cols,
                rmode=1, cper=CASCADE_PER,
            )
            refined = _screen_topk(self.refine_scores, CASCADE_PER, SCREEN_PER)
            _rows_dot(
                h, self.w_full, self.s_full, coarse_idx, self.gscale, self.cand_logits,
                True, gated, ref=refined, rmode=2, cper=CASCADE_PER, sper=SCREEN_PER,
            )
            gate, ok = _gate_args(out)
            _select_token[(1,)](
                self.cand_logits, coarse_idx, out, N=self.cand_logits.numel(),
                BLOCK=triton.next_power_of_2(self.cand_logits.numel()),
                gate_ptr=gate, GATED=gated and ok, THETA=DEPTH_THETA,
                launch_pdl=mtp_pdl(),
                ref=refined, TWO_LEVEL=True, CHUNK=SCREEN_CHUNK, CPER=CASCADE_PER,
                SPER=SCREEN_PER,
            )
            return out
        if self.cascade:
            hc = h.index_select(0, self.coarse_cols)
            if self.ultra:
                hu = h.index_select(0, self.ultra_cols)
                _rows_dot(
                    hu, self.w_ultra, self.s_ultra, None,
                    self.gscale, self.scores, False, gated,
                )
                ultra_cand = (
                    _screen_topk(self.scores, SCREEN_CHUNK, ULTRA_PER, SCREEN_CHUNK)
                )
                _rows_dot(
                    hc, self.w_coarse, self.s_coarse, ultra_cand.reshape(-1),
                    self.gscale, self.coarse_refine_scores, True, gated,
                )
                coarse_indices = _screen_topk(self.coarse_refine_scores, ULTRA_PER, CASCADE_PER)
                # Entries remain grouped by original vocabulary chunk. Gather
                # global IDs rather than applying the original chunk offset twice.
                coarse_cand = ultra_cand.gather(1, coarse_indices)
            else:
                _rows_dot(
                    hc, self.w_coarse, self.s_coarse, None,
                    self.gscale, self.scores, False, gated,
                )
                coarse_cand = (
                    _screen_topk(self.scores, SCREEN_CHUNK, CASCADE_PER, SCREEN_CHUNK)
                )
            hs = h.index_select(0, self.cols)
            _rows_dot(
                hs, self.w_scr, self.s_scr, coarse_cand.reshape(-1),
                self.gscale, self.refine_scores, True, gated,
            )
            # Scores retain the [original chunk, coarse candidate] layout.
            # Gather global IDs; adding chunk_off here would offset twice.
            refined = _screen_topk(self.refine_scores, CASCADE_PER, SCREEN_PER)
            cand = coarse_cand.gather(1, refined).reshape(-1)
        else:
            hs = h.index_select(0, self.cols)
            _rows_dot(
                hs, self.w_scr, self.s_scr, None, self.gscale, self.scores, False, gated
            )
            cand = (
                _screen_topk(self.scores, SCREEN_CHUNK, SCREEN_PER)
                + self.chunk_off
            ).reshape(-1)
        _rows_dot(
            h, self.w_full, self.s_full, cand, self.gscale, self.cand_logits, True, gated
        )
        gate, ok = _gate_args(out)
        _select_token[(1,)](
            self.cand_logits, cand, out, N=self.cand_logits.numel(),
            BLOCK=triton.next_power_of_2(self.cand_logits.numel()),
            gate_ptr=gate, GATED=gated and ok, THETA=DEPTH_THETA,
            launch_pdl=mtp_pdl(),
        )
        return out

    def greedy(self, hidden: torch.Tensor) -> torch.Tensor:
        """Draft ids [B] for hidden [B, K] (B is small; one pass per row)."""
        if self.graph is not None and hidden.shape[0] == 1:
            self._h.copy_(hidden[0])
            self.graph.replay()
            return self._out.clone()
        out = []
        for i in range(hidden.shape[0]):
            out.append(self._greedy_row(hidden[i].float().contiguous()))
        return torch.stack(out)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy draft-token head for the Nemotron-H MTP drafter: cascaded partial-K
screens plus a full-K rescore of the survivors.

DRAFTER ONLY. The target's lm_head, logits and verification are untouched;
the drafter's lm_head is the target's shared NVFP4 head (131072 x 2688, about
198 MB), and the drafter reads all of it three times per depth-3 cycle only to
take one argmax per draft. This replaces that argmax:

1. screen every vocabulary row on the 32 most important whole NVFP4
   16-element groups, chosen once from weight-only importance, in fp32;
2. keep the top CASCADE_PER rows of every SCREEN_CHUNK-row chunk;
3. rescore those candidates on the original 64 importance groups and keep
   the top SCREEN_PER candidates of each original chunk;
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
    ref_ptr=None,  # [R] int64 second-stage top-k indices, when RMODE == 2
    RMODE: tl.constexpr = 0,
    CHUNK: tl.constexpr = 1,
    CPER: tl.constexpr = 1,
    SPER: tl.constexpr = 1,
):
    """RMODE 0: rows from GATHER as before. 1: rows = rows_ptr[r] + (r // CPER) * CHUNK
    (first-stage top-k indices within their chunk). 2: rows = rows_ptr[c * CPER + ref_ptr[r]]
    + c * CHUNK with c = r // SPER (the second stage's picks among them). HCOLS reads
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
              rmode=0, cper=1, sper=1):
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
    )


def screen_topk_enabled() -> bool:
    """Kill switch for the fused exact top-k: VLLM_MTP_SCREEN_TOPK=0."""
    return os.environ.get("VLLM_MTP_SCREEN_TOPK", "1") != "0"


@triton.jit
def _topk_key(v):
    # torch's TopKTypeConfig<float>::convert, widened to int64: an order-preserving
    # unsigned key (-0 below +0) with every NaN mapped to the largest key.
    x = v.to(tl.int32, bitcast=True)
    xu = x.to(tl.int64) & 0xFFFFFFFF
    key = tl.where(x < 0, xu ^ 0xFFFFFFFF, xu ^ 0x80000000)
    return tl.where(v != v, 0xFFFFFFFF, key)


@triton.jit
def _chunk_topk(
    scores_ptr,  # [C, N] fp32
    src_ptr,  # [C, N] int64 ids (GATHER) or unused
    out_ptr,  # [C, K] int64
    N: tl.constexpr,
    K: tl.constexpr,
    GATHER: tl.constexpr,
    launch_pdl: tl.constexpr = False,
    RADIX: tl.constexpr = False,
):
    """The indices torch.topk(scores.view(C, N), K, dim=-1, sorted=False) returns,
    in its order (CUDA sbtopk::gatherTopK: every key above the K-th largest in
    ascending index order, then the K-th largest's ties in ascending index order
    until K are taken), written as row-major ids (c * N + i) or, with GATHER, as
    src[c, i]: the id lists the screen built with topk + chunk offset / gather."""
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    c = tl.program_id(0)
    i = tl.arange(0, N)
    key = _topk_key(tl.load(scores_ptr + c * N + i))
    if RADIX:
        low = tl.min(key, axis=0)
        high = tl.max(key, axis=0)
        if low == high:
            t = low
        else:
            active = tl.full((N,), True, tl.int1)
            left = tl.full((), K, tl.int32)
            bins = tl.arange(0, 16)
            t = tl.full((), 0, tl.int64)
            for shift in tl.static_range(28, -1, -4):
                digit = ((key >> shift) & 15).to(tl.int32)
                hist = tl.histogram(digit, 16, mask=active)
                suffix = tl.sum(hist, axis=0) - tl.cumsum(hist, axis=0) + hist
                pick = tl.max(tl.where(suffix >= left, bins, 0), axis=0)
                left -= tl.sum(tl.where(bins > pick, hist, 0), axis=0)
                t |= pick.to(tl.int64) << shift
                active = active & (digit == pick)
    else:
        t = tl.sum(key * 0, axis=0)
        for b in tl.static_range(31, -1, -1):
            trial = t | (1 << b)
            cnt = tl.sum((key >= trial).to(tl.int32), axis=0)
            t = tl.where(cnt >= K, trial, t)
    gt = (key > t).to(tl.int32)
    eq = (key == t).to(tl.int32)
    n_gt = tl.sum(gt, axis=0)
    gt_rank = tl.cumsum(gt, axis=0) - gt
    eq_rank = tl.cumsum(eq, axis=0) - eq
    take = (gt != 0) | ((eq != 0) & (eq_rank < K - n_gt))
    pos = tl.where(gt != 0, gt_rank, n_gt + eq_rank)
    if GATHER:
        val = tl.load(src_ptr + c * N + i)
    else:
        val = c.to(tl.int64) * N + i
    tl.store(out_ptr + c * K + pos, val, mask=take)


@triton.jit
def _gather_two(
    h_ptr, ia_ptr, oa_ptr, ib_ptr, ob_ptr,
    NA: tl.constexpr, NB: tl.constexpr,
    launch_pdl: tl.constexpr = False,
):
    """oa = h[ia], ob = h[ib] (the two index_select gathers of the cascade) in one
    launch; pure copies."""
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    a = tl.arange(0, NA)
    tl.store(oa_ptr + a, tl.load(h_ptr + tl.load(ia_ptr + a)))
    b = tl.arange(0, NB)
    tl.store(ob_ptr + b, tl.load(h_ptr + tl.load(ib_ptr + b)))


@triton.jit
def _select_token(
    logits, ids, out, N: tl.constexpr, BLOCK: tl.constexpr,
    gate_ptr=None,
    GATED: tl.constexpr = False,
    THETA: tl.constexpr = 0.0,
    launch_pdl: tl.constexpr = False,
    ref=None,
    TWO_LEVEL: tl.constexpr = False,
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
    if TWO_LEVEL:
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
        if (tl.load(gate_ptr + 2) != 0) & (tl.load(gate_ptr) == 0):
            tl.store(gate_ptr + 1, tl.load(gate_ptr + 1) + 1)
            denom = tl.sum(tl.where(i < N, tl.exp(v - best), 0.0), axis=0)
            prev = tl.load(gate_ptr + 3)
            chain = prev.to(tl.float32, bitcast=True) / denom
            finite = chain == chain
            tl.store(
                gate_ptr + 3,
                tl.where(finite, chain.to(tl.int32, bitcast=True), prev),
            )
            if finite & (chain < THETA):
                tl.store(gate_ptr, 1)


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

        def _pow2(v: int) -> bool:
            return v > 0 and (v & (v - 1)) == 0

        self.fused_topk = (
            self.cascade
            and device.type == "cuda"
            and screen_topk_enabled()
            and all(
                _pow2(v)
                for v in (SCREEN_CHUNK, CASCADE_PER, SCREEN_PER,
                          self.coarse_cols.numel(), cols.numel())
            )
        )
        if self.fused_topk:
            self._coarse_cand = torch.empty(
                (n // SCREEN_CHUNK) * CASCADE_PER, dtype=torch.int64, device=device
            )
            self._cand = torch.empty(
                (n // SCREEN_CHUNK) * SCREEN_PER, dtype=torch.int64, device=device
            )
            self._hc = torch.empty(
                self.coarse_cols.numel(), dtype=torch.float32, device=device
            )
            self._hs = torch.empty(cols.numel(), dtype=torch.float32, device=device)
        self.radix = self.fused_topk and os.environ.get("VLLM_MTP_RADIX_TOPK", "1") != "0"
        if self.radix:
            self.radix = self._radix_matches(device)
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
            _l2pf.DRAFT_SCREEN[:] = [self.w_coarse]
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
            "MTP draft-head screen: %d of %d groups, %d candidates, cascade=%s, "
            "fused_topk=%s, graph=%s",
            SCREEN_FRAC_GROUPS, ng, self.cand_logits.numel(), self.cascade,
            self.fused_topk, self.graph is not None,
        )

    def _radix_matches(self, device: torch.device) -> bool:
        """Official remote model-load admission, no request data or local execution."""
        try:
            for n, k, gather in ((SCREEN_CHUNK, CASCADE_PER, False),
                                 (CASCADE_PER, SCREEN_PER, True)):
                index = torch.arange(2 * n, device=device, dtype=torch.int32)
                src = (torch.arange(2 * n, device=device, dtype=torch.int64) + 10000).flip(0)
                for case in range(4):
                    if case == 0:
                        scores = (index * 1103515245 + 12345).view(torch.float32)
                    elif case == 1:
                        bits = torch.where(index % 2 == 0, 0, -2147483648).to(torch.int32)
                        scores = bits.view(torch.float32)
                    elif case == 2:
                        scores = (index % 19).float() - 9
                        scores[:n // 16] = float("nan")
                        scores[n // 16:n // 8] = float("inf")
                        scores[n:n + n // 8] = -float("inf")
                    else:
                        scores = torch.full((2 * n,), 42.0, device=device)
                    a = torch.empty(2 * k, dtype=torch.int64, device=device)
                    b = torch.empty_like(a)
                    for radix, out in ((False, a), (True, b)):
                        _chunk_topk[(2,)](scores, src, out, N=n, K=k, GATHER=gather,
                                          RADIX=radix, num_warps=4 if radix or n != SCREEN_CHUNK else 8,
                                          launch_pdl=mtp_pdl())
                    if not torch.equal(a, b):
                        logger.warning("MTP radix top-k differs; original selection retained")
                        return False
            return True
        except Exception as exc:
            logger.warning("MTP radix top-k unavailable (%r); original selection retained", exc)
            return False

    def _fold_matches(self, k: int, device: torch.device) -> bool:
        """Load time, before capture: the folded cascade against the unfolded one on
        random rows, bitwise on every stage's scores and on the token."""
        try:
            gen = torch.Generator(device=device)
            gen.manual_seed(1234)
            for _ in range(3):
                h = torch.randn(k, generator=gen, device=device) * 2.0
                outs = []
                for fold in (True, False):
                    self.fold = fold
                    tok = self._greedy_row(h)
                    outs.append([tok.clone(), self.scores.clone(),
                                 self.refine_scores.clone(), self.cand_logits.clone()])
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
        if self.cascade and self.fused_topk:
            # LOCAL PATCH (cudafast-nemotron35): the same cascade with the two
            # index_select gathers in one launch and each segmented torch.topk
            # (+ chunk offset / + gather) in one launch that writes the id list
            # torch.topk(sorted=False) returns, in torch's order (_chunk_topk).
            # Same candidate ids in the same order, so every rows_dot, the
            # selection and the depth gate see identical inputs.
            _gather_two[(1,)](
                h, self.coarse_cols, self._hc, self.cols, self._hs,
                NA=self.coarse_cols.numel(), NB=self.cols.numel(),
                launch_pdl=mtp_pdl(),
            )
            _rows_dot(
                self._hc, self.w_coarse, self.s_coarse, None,
                self.gscale, self.scores, False, gated,
            )
            nchunk = self.n // SCREEN_CHUNK
            _chunk_topk[(nchunk,)](
                self.scores, self.scores, self._coarse_cand,
                N=SCREEN_CHUNK, K=CASCADE_PER, GATHER=False, RADIX=self.radix,
                num_warps=4 if self.radix else 8, launch_pdl=mtp_pdl(),
            )
            _rows_dot(
                self._hs, self.w_scr, self.s_scr, self._coarse_cand,
                self.gscale, self.refine_scores, True, gated,
            )
            _chunk_topk[(nchunk,)](
                self.refine_scores, self._coarse_cand, self._cand,
                N=CASCADE_PER, K=SCREEN_PER, GATHER=True, RADIX=self.radix,
                num_warps=4, launch_pdl=mtp_pdl(),
            )
            cand = self._cand
        elif self.cascade and self.fold:
            # The same arithmetic as the unfolded cascade below, with the column
            # gathers of h, the chunk offsets and the candidate gather done as
            # index arithmetic inside the dot and select kernels.
            _rows_dot(
                h, self.w_coarse, self.s_coarse, None,
                self.gscale, self.scores, False, gated, hcols=self.coarse_cols,
            )
            coarse_idx = self.scores.view(-1, SCREEN_CHUNK).topk(
                CASCADE_PER, dim=-1, sorted=False
            ).indices
            _rows_dot(
                h, self.w_scr, self.s_scr, coarse_idx,
                self.gscale, self.refine_scores, True, gated, hcols=self.cols,
                rmode=1, cper=CASCADE_PER,
            )
            refined = self.refine_scores.view(-1, CASCADE_PER).topk(
                SCREEN_PER, dim=-1, sorted=False
            ).indices
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
        elif self.cascade:
            hc = h.index_select(0, self.coarse_cols)
            _rows_dot(
                hc, self.w_coarse, self.s_coarse, None,
                self.gscale, self.scores, False, gated,
            )
            # Every consumer below is order-independent (index-aligned gathers,
            # then the max with the lowest-id tie rule), so the candidate sets
            # need not be sorted: sorted=False skips a segmented sort per stage.
            coarse_cand = (
                self.scores.view(-1, SCREEN_CHUNK)
                .topk(CASCADE_PER, dim=-1, sorted=False).indices + self.chunk_off
            )
            hs = h.index_select(0, self.cols)
            _rows_dot(
                hs, self.w_scr, self.s_scr, coarse_cand.reshape(-1),
                self.gscale, self.refine_scores, True, gated,
            )
            # Scores retain the [original chunk, coarse candidate] layout.
            # Gather global IDs; adding chunk_off here would offset twice.
            refined = self.refine_scores.view(-1, CASCADE_PER).topk(
                SCREEN_PER, dim=-1, sorted=False
            ).indices
            cand = coarse_cand.gather(1, refined).reshape(-1)
        else:
            hs = h.index_select(0, self.cols)
            _rows_dot(
                hs, self.w_scr, self.s_scr, None, self.gscale, self.scores, False, gated
            )
            cand = (
                self.scores.view(-1, SCREEN_CHUNK)
                .topk(SCREEN_PER, dim=-1, sorted=False).indices
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

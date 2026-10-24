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

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

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
):
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    rmask = r < R
    if GATHER:
        rows = tl.load(rows_ptr + r, mask=rmask, other=0).to(tl.int64)
    else:
        rows = r.to(tl.int64)
    acc = tl.zeros((BR,), dtype=tl.float32)
    for b0 in range(0, NBYTES, BB):
        ob = b0 + tl.arange(0, BB)
        wb = tl.load(
            w_ptr + rows[:, None] * NBYTES + ob[None, :],
            mask=rmask[:, None],
            other=0,
            eviction_policy="evict_first",
        ).to(tl.int32)
        og = (b0 // 8) + tl.arange(0, BB // 8)
        sc = tl.load(
            s_ptr + rows[:, None] * (NBYTES // 8) + og[None, :],
            mask=rmask[:, None],
            other=0.0,
        ).to(tl.float32)
        hl = tl.load(h_ptr + 2 * ob)
        hh = tl.load(h_ptr + 2 * ob + 1)
        lo, hi = _e2m1_pair(wb)
        part = lo * hl[None, :] + hi * hh[None, :]
        part = tl.reshape(part, (BR, BB // 8, 8))
        acc += tl.sum(tl.sum(part, axis=2) * sc, axis=1)
    tl.store(out_ptr + r, acc * gscale, mask=rmask)


def _rows_dot(h, w, s, rows, gscale, out, gather):
    n_out = out.numel()
    nbytes = w.shape[1]
    br = 64 if not gather else 16
    _nvfp4_rows_dot[(triton.cdiv(n_out, br),)](
        h, w, s, rows if gather else out, out, n_out, gscale,
        NBYTES=nbytes, GATHER=gather, BR=br, BB=64, num_warps=4, num_stages=2,
    )


@triton.jit
def _select_token(logits, ids, out, N: tl.constexpr, BLOCK: tl.constexpr):
    """Stock BF16 maximum, then minimum token id among ties, in one launch."""
    i = tl.arange(0, BLOCK)
    # Round before reduction: FP32 logit differences can disappear in BF16.
    v = tl.load(logits + i, mask=i < N, other=-float("inf"))
    v = v.to(tl.bfloat16).to(tl.float32)
    token = tl.load(ids + i, mask=i < N, other=1 << 30)
    best = tl.max(v, axis=0)
    # torch.max propagates NaNs; its subsequent equality tests all fail.
    any_nan = tl.max(((i < N) & (v != v)).to(tl.int32), axis=0)
    chosen = tl.min(tl.where((i < N) & (v == best), token, 1 << 30), axis=0)
    tl.store(out, tl.where(any_nan != 0, 1 << 30, chosen))


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
        self.chunk_off = (
            torch.arange(n // SCREEN_CHUNK, device=device) * SCREEN_CHUNK
        )[:, None]
        self.scores = torch.empty(n, dtype=torch.float32, device=device)
        self.cand_logits = torch.empty(
            (n // SCREEN_CHUNK) * SCREEN_PER, dtype=torch.float32, device=device
        )
        self.big = torch.tensor(1 << 30, dtype=torch.int64, device=device)
        self.graph = None
        self._h = torch.empty(k, dtype=torch.float32, device=device)
        self._out = torch.empty(1, dtype=torch.int64, device=device)
        if device.type == "cuda" and screen_graph_enabled():
            try:
                self._h.zero_()
                for _ in range(2):
                    self._out.copy_(self._greedy_row(self._h))
                torch.cuda.synchronize(device)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    self.graph, pool=torch.cuda.graph_pool_handle()
                ):
                    self._out.copy_(self._greedy_row(self._h))
                torch.cuda.synchronize(device)
            except Exception as exc:
                self.graph = None
                logger.warning("MTP draft-head screen graph disabled: %s", exc)
        logger.info(
            "MTP draft-head screen: %d of %d groups, %d candidates, cascade=%s, graph=%s",
            SCREEN_FRAC_GROUPS, ng, self.cand_logits.numel(), self.cascade,
            self.graph is not None,
        )

    def _greedy_row(self, h: torch.Tensor) -> torch.Tensor:
        """One row h [K] float32: the cascade (or the single screen) and the
        argmax tie rule. Every buffer it touches is fixed-address, so the body
        is capturable as-is."""
        if self.cascade:
            hc = h.index_select(0, self.coarse_cols)
            _rows_dot(
                hc, self.w_coarse, self.s_coarse, None,
                self.gscale, self.scores, False,
            )
            coarse_cand = (
                self.scores.view(-1, SCREEN_CHUNK)
                .topk(CASCADE_PER, dim=-1).indices + self.chunk_off
            )
            hs = h.index_select(0, self.cols)
            _rows_dot(
                hs, self.w_scr, self.s_scr, coarse_cand.reshape(-1),
                self.gscale, self.refine_scores, True,
            )
            # Scores retain the [original chunk, coarse candidate] layout.
            # Gather global IDs; adding chunk_off here would offset twice.
            refined = self.refine_scores.view(-1, CASCADE_PER).topk(
                SCREEN_PER, dim=-1
            ).indices
            cand = coarse_cand.gather(1, refined).reshape(-1)
        else:
            hs = h.index_select(0, self.cols)
            _rows_dot(hs, self.w_scr, self.s_scr, None, self.gscale, self.scores, False)
            cand = (
                self.scores.view(-1, SCREEN_CHUNK).topk(SCREEN_PER, dim=-1).indices
                + self.chunk_off
            ).reshape(-1)
        _rows_dot(h, self.w_full, self.s_full, cand, self.gscale, self.cand_logits, True)
        out = torch.empty((), dtype=torch.int64, device=h.device)
        _select_token[(1,)](
            self.cand_logits, cand, out, N=self.cand_logits.numel(),
            BLOCK=triton.next_power_of_2(self.cand_logits.numel()),
        )
        return out

    def greedy(self, hidden: torch.Tensor) -> torch.Tensor:
        """Draft ids [B] for hidden [B, K] (B is small; one pass per row)."""
        if self.graph is not None and hidden.shape[0] == 1:
            self._h.copy_(hidden[0].float())
            self.graph.replay()
            return self._out.clone()
        out = []
        for i in range(hidden.shape[0]):
            out.append(self._greedy_row(hidden[i].float().contiguous()))
        return torch.stack(out)

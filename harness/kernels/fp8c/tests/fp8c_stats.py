# SPDX-License-Identifier: Apache-2.0
"""Escape statistics of candidate fp8c windows on the real Mamba FP8 weights (no kernels)."""

from __future__ import annotations

import json
import os

import torch
from safetensors import safe_open

SNAP = os.environ["NEMO_SNAPSHOT"]
with open(os.path.join(SNAP, "model.safetensors.index.json")) as f:
    wmap = json.load(f)["weight_map"]
layers = sorted(int(k.split(".")[2]) for k in wmap if k.endswith(".mixer.in_proj.weight"))


def load(name):
    with safe_open(os.path.join(SNAP, wmap[name]), framework="pt") as f:
        return f.get_tensor(name).to("cuda")


def esc_stats(w, e0_col, esc_col):
    """w [N, K] uint8; e0_col, esc_col [N] -> escapes [N, K] bool."""
    e = (w >> 3) & 15
    e0 = e0_col[:, None]
    win = (e >= e0) & (e <= e0 + 7)
    code = ((w & 0x80) >> 1) | (((e - e0) & 7) << 3) | (w & 7)
    return ~win | (code == esc_col[:, None])


def summarize(tag, esc, n, k):
    # warp-chunk = 8 columns x 128 K
    wc = esc.view(n // 8, 8, k // 128, 128).sum((1, 3))
    col = esc.sum(1)
    return dict(
        tag=tag, rate=round(esc.float().mean().item(), 5),
        wc_max=int(wc.max()), wc_p999=float(torch.quantile(wc.float().flatten()[:1 << 24], 0.999)),
        wc_over124=int((wc > 124).sum()), wc_over252=int((wc > 252).sum()),
        cols_over50pct=int((col > k // 2).sum()), worst_cols=col.topk(4).values.tolist(),
    )


for li in layers[:: max(1, len(layers) // 6)]:
    for proj in ("in_proj", "out_proj"):
        w = load(f"backbone.layers.{li}.mixer.{proj}.weight").view(torch.uint8).contiguous()
        n, k = w.shape
        e = ((w >> 3) & 15).long()
        # global window
        h = torch.bincount(e.flatten(), minlength=16)
        e0 = max(range(9), key=lambda x: (int(h[x:x + 8].sum()), x))
        hb = torch.bincount(w.flatten().long(), minlength=256)
        inw = [b for b in range(256) if e0 <= ((b >> 3) & 15) <= e0 + 7]
        escb = min(inw, key=lambda b: int(hb[b]))
        escc = ((escb & 0x80) >> 1) | ((((escb >> 3) & 15) - e0) << 3) | (escb & 7)
        g = esc_stats(w, torch.full((n,), e0, device="cuda", dtype=torch.uint8),
                      torch.full((n,), escc, device="cuda", dtype=torch.uint8))
        out = [summarize(f"global e0={e0}", g, n, k)]
        # per-column window and per-column escape
        hc = torch.zeros((n, 16), device="cuda", dtype=torch.long).scatter_add_(1, e, torch.ones_like(e))
        cs = torch.stack([hc[:, x:x + 8].sum(1) for x in range(9)], 1)  # [n, 9]
        e0c = (cs * 16 + torch.arange(9, device="cuda")).argmax(1).to(torch.uint8)
        codes = ((w & 0x80) >> 1) | ((((w >> 3) & 15) - e0c[:, None]) & 7) << 3 | (w & 7)
        ee = (w >> 3) & 15
        inwin = (ee >= e0c[:, None]) & (ee <= e0c[:, None] + 7)
        cc = torch.zeros((n, 128), device="cuda", dtype=torch.long).scatter_add_(
            1, codes.long(), inwin.long())
        escc_col = cc.argmin(1).to(torch.uint8)
        pc = esc_stats(w, e0c, escc_col)
        out.append(summarize("per-column e0+esc", pc, n, k))
        out.append(dict(e0_col_hist=torch.bincount(e0c.long(), minlength=9).tolist(),
                        zero_bytes=int((w == 0).sum()), negzero=int((w == 0x80).sum()),
                        exp_hist=h.tolist()))
        print(li, proj, json.dumps(out), flush=True)

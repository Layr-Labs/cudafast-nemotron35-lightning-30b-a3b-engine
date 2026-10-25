#!/usr/bin/env python3
"""Value-range statistics of the routed-expert up (w13) NVFP4 block scales.

Decides the fixed-rate lossless code for nvsc (compressed block scales in the
humming up kernel). Reads only up_proj.weight_scale from $NEMO_SNAPSHOT; per
layer prints the byte range, distinct counts and the share of 64-scale rows
(one 64-column tile, one K group) whose range fits 16/32/64 codes. Also prints
the sha256 of the installed humming headers the derived kernel replaces.
"""

import hashlib
import json
import os
import pathlib
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import humming_moe_exact as check  # noqa: E402


def main():
    import humming

    inc = pathlib.Path(humming.__file__).parent / "include/humming"
    for rel in [
        "memory/g2s_loader/loader_bs.cuh",
        "memory/s2r_loader/loader_bs.cuh",
        "memory/g2s_pipeline.cuh",
        "memory/s2r_pipeline.cuh",
        "arith/mainloop_arith.cuh",
        "kernel/humming.cuh",
        "utils/storage.cuh",
    ]:
        data = (inc / rel).read_bytes()
        print("hdr", rel, hashlib.sha256(data).hexdigest()[:16], len(data), flush=True)
    print("humming", humming.__version__, flush=True)
    snap = check.Snapshot(os.environ["NEMO_SNAPSHOT"])
    out = {}
    union_min, union_max = 255, 0
    for li in check.MOE_LAYERS:
        pre = f"backbone.layers.{li}.mixer.experts"
        names = [f"{pre}.{e}.up_proj.weight_scale" for e in range(check.NUM_EXPERTS)]
        t = snap.load(names, "cuda")
        s = torch.stack([t[n] for n in names]).view(torch.uint8)  # [E, N, G]
        assert s.shape == (128, check.INTER, check.HIDDEN // 16), s.shape
        lo, hi = int(s.min()), int(s.max())
        union_min, union_max = min(lo, union_min), max(hi, union_max)
        hist = torch.bincount(s.flatten().long(), minlength=256)
        per_e_lo = s.view(128, -1).min(1).values
        per_e_hi = s.view(128, -1).max(1).values
        per_e_distinct = [int(torch.unique(s[e]).numel()) for e in range(128)]
        # rows: one K group x 64 output columns (the kernel's per-stage unit)
        rows = s.view(128, check.INTER // 64, 64, -1).permute(0, 1, 3, 2)
        r_rng = rows.max(-1).values.int() - rows.min(-1).values.int()
        # tiles: one 64-column tile over all of K
        tiles = s.view(128, check.INTER // 64, -1)
        t_rng = tiles.max(-1).values.int() - tiles.min(-1).values.int()
        top = hist.sort(descending=True).values
        n = s.numel()
        rec = {
            "min": lo, "max": hi, "range": hi - lo,
            "sign_set": int((s >= 128).sum()), "zero": int((s == 0).sum()),
            "distinct": int((hist > 0).sum()),
            "top16": round(float(top[:16].sum()) / n, 6),
            "top32": round(float(top[:32].sum()) / n, 6),
            "e_lo_min": int(per_e_lo.min()), "e_hi_max": int(per_e_hi.max()),
            "e_range_max": int((per_e_hi.int() - per_e_lo.int()).max()),
            "e_distinct_max": max(per_e_distinct),
            "row_rng_lt16": round(float((r_rng < 16).float().mean()), 5),
            "row_rng_lt32": round(float((r_rng < 32).float().mean()), 5),
            "row_rng_lt64": round(float((r_rng < 64).float().mean()), 5),
            "tile_rng_lt32": round(float((t_rng < 32).float().mean()), 5),
            "tile_rng_lt64": round(float((t_rng < 64).float().mean()), 5),
            "tile_rng_max": int(t_rng.max()),
            "low_tail": {str(v): int(hist[v]) for v in range(lo, min(lo + 12, 256)) if hist[v]},
        }
        out[li] = rec
        print("layer", li, json.dumps(rec), flush=True)
        del t, s
    print("union", union_min, union_max, union_max - union_min, flush=True)
    od = os.environ.get("NEMO_OUT")
    if od:
        pathlib.Path(od, "stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

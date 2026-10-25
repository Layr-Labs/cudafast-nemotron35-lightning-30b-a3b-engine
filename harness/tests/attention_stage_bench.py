#!/usr/bin/env python3
"""Measure the actual attention-pack admission guard on every target projection."""
import argparse
import json
import os

import torch
import torch.nn.functional as F

import humming_moe_exact as hme
from vllm.model_executor.models.nemotron_h_attn_packed import _choose, _exact, _op_us
from vllm.model_executor.models.nemotron_h_attn_packed_kernel import CONFIGS, ROWS, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    snap_path = os.environ["NEMO_SNAPSHOT"]
    out_dir = os.environ["NEMO_OUT"]
    os.makedirs(out_dir, exist_ok=True)
    snap = hme.Snapshot(snap_path)
    groups = {}
    attention_layers = sorted({int(k.split(".")[2]) for k in snap.weight_map
                               if k.startswith("backbone.layers.") and k.endswith("q_proj.weight")})
    assert len(attention_layers) == 6, attention_layers
    from vllm.model_executor.models.nemotron_h_mtp_packed import pack_bf16
    for li in attention_layers:
        prefix = f"backbone.layers.{li}.mixer"
        names = [f"{prefix}.{part}_proj.weight" for part in ("q", "k", "v", "o")]
        t = snap.load(names, "cuda")
        qkv = torch.cat([t[names[0]], t[names[1]], t[names[2]]], dim=0).contiguous()
        for raw in (qkv, t[names[3]].contiguous()):
            proj = torch.nn.Module()
            proj.weight = torch.nn.Parameter(raw, requires_grad=False)
            packed = pack_bf16(proj.weight)
            assert packed is not None
            sm, ex, base, ri, raw_plane = packed.planes
            n, k = proj.weight.shape
            groups.setdefault((n, k), []).append((proj, (sm, ex, base, ri, raw_plane.view(torch.int16))))
    flush = torch.empty(96 << 20, dtype=torch.uint8, device="cuda")
    results = []
    for (n, k), items in groups.items():
        proj, planes = items[0]
        x = torch.randn(ROWS, k, device=proj.weight.device, dtype=torch.bfloat16)
        linear_us = _op_us(lambda: F.linear(x, proj.weight), flush)
        timings = {}
        for cfg in CONFIGS:
            if n % cfg[0]:
                continue
            try:
                timings[str(cfg)] = _op_us(lambda cfg=cfg: run(x, *planes, n, k, cfg), flush)
            except Exception as exc:
                timings[str(cfg)] = {"error": str(exc)}
        cfg = _choose(items, n, k, flush)
        exact = None if cfg is None else _exact(items, n, k, cfg)
        results.append({"shape": [n, k], "linear_us": linear_us, "packed_us": timings,
                        "selected": cfg, "enabled": cfg is not None, "exact": exact})
        print(json.dumps(results[-1]), flush=True)
    result = {"label": args.label, "results": results,
              "all_enabled_exact": all(r["exact"] is not False for r in results)}
    path = os.path.join(out_dir, f"attention-stage-{args.label}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print("wrote", path, flush=True)
    if not result["all_enabled_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

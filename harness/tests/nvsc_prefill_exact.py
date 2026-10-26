#!/usr/bin/env python3
"""nvsc-prefill: bit-identity and per-call time of the 496-token routed up
(w13 + ReLU^2) kernel with 6-bit block scales (humming_nvsc.launch_prefill)
against the record's claim-when-free + evict-first prefill kernel.

  NEMO_SNAPSHOT=... python3 harness/tests/nvsc_prefill_exact.py --caps caps.pt

--caps: captured real prefill w13 inputs {"caps": [{layer: (x [496, 2688] bf16,
topk_ids [496, 6])}]} (all 23 routed layers). exact: every layer, every
prompt, both scheduler variants (dyn on/off): torch.equal on the bf16 bits vs
forward_relu2(nvsc=False); layers left stock (> 64 scale values) must fall
back; forced fallback (failed build variant, NVSC_PREFILL off) must take the
record kernel and match. time: CUDA graphs, cold = one call per packed layer
in turn (weights stream from DRAM), ABAB over --reps.
"""

import argparse
import json
import os
import statistics
import sys

import torch

os.environ["VLLM_HUMMING_BATCH_INVARIANT"] = "1"
os.environ.setdefault("NVSC", "1")
os.environ["NVSC_PREFILL"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
import humming_moe_exact as hx
from vllm.model_executor.layers.fused_moe.experts import humming_nvsc as nvsc
from vllm.model_executor.layers.fused_moe.experts import (
    humming_relu2_epilogue as fused,
)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--caps", required=True)
p.add_argument("--layers", default="all")
p.add_argument("--reps", type=int, default=8)
p.add_argument("--skip-exact", action="store_true")
args = p.parse_args()
layers = hx.MOE_LAYERS if args.layers == "all" else [int(v) for v in args.layers.split(",")]
dev = torch.device("cuda")
snap = hx.Snapshot(os.environ["NEMO_SNAPSHOT"])
caps = torch.load(args.caps)["caps"]

exs = {}
for li in layers:
    raw = snap.raw_experts(li, dev)
    exs[li] = hx.build_variant(raw, f"backbone.layers.{li}.mixer.experts", True)
    del raw
packed = [li for li in layers if exs[li].layer.nvsc_w13_bs is not None]
print("stats", json.dumps(nvsc._STATS), flush=True)


def kwargs(ex, ids):
    return ex.prepare_humming_moe_kwargs(ids, None, None)[0]


def call(ex, x, kw, fill, **extra):
    out = torch.empty(x.size(0) * hx.TOP_K, hx.INTER, dtype=torch.bfloat16, device=dev)
    out.view(torch.int16).fill_(fill)
    fused.forward_relu2(layer=ex.layer, inputs=x, outputs=out, **kw, **extra)
    return out


def direct(ex, x, kw, dyn):
    out = torch.empty(x.size(0) * hx.TOP_K, hx.INTER, dtype=torch.bfloat16, device=dev)
    out.view(torch.int16).fill_(-1)
    ok = nvsc.launch_prefill(ex.layer, x, out, kw["compute_config"], kw["tuning_config"],
                             kw["sorted_ids"], kw["expert_ids"], kw["num_tokens_padded"],
                             kw["top_k"], kw["valid_shape_m"], dyn=dyn)
    return ok, out


if not args.skip_exact:
    cases = bad = 0
    for li in layers:
        ex = exs[li]
        for pi, cap in enumerate(caps):
            x, ids = cap[li][0].to(dev), cap[li][1].to(dev).to(torch.int32)
            assert x.shape == (496, hx.HIDDEN)
            kw = kwargs(ex, ids)
            for dyn in (True, False):
                ref = call(ex, x, kw, 0x7FC1, nvsc=False, dyn=dyn)
                assert int((ref.view(torch.int16) == 0x7FC1).sum()) == 0
                ok, got = direct(ex, x, kw, dyn)
                assert ok == (li in packed), (li, ok)
                if not ok:
                    got = call(ex, x, kw, -1, dyn=dyn)
                got2 = call(ex, x, kw, -1, dyn=dyn)
                torch.cuda.synchronize()
                n = hx.diff_count(ref, got) + hx.diff_count(ref, got2)
                bad += n
                cases += 1
                if n:
                    print(f"MISMATCH layer {li} prompt {pi} dyn {dyn}: {n}", flush=True)
        print(f"layer {li} packed {li in packed}: cumulative cases {cases} bad {bad}", flush=True)
    li = packed[0]
    ex = exs[li]
    x, ids = caps[0][li][0].to(dev), caps[0][li][1].to(dev).to(torch.int32)
    kw = kwargs(ex, ids)
    ref = call(ex, x, kw, 0x7FC1, nvsc=False)
    variant = fused._variant_key(fused._dyn_enabled())
    nvsc._PF_FAILED.add(variant)
    ok, _ = direct(ex, x, kw, None)
    assert not ok
    got = call(ex, x, kw, -1)
    nvsc._PF_FAILED.discard(variant)
    nvsc.PREFILL_ENABLED = False
    ok2, _ = direct(ex, x, kw, None)
    assert not ok2
    got2 = call(ex, x, kw, -1)
    nvsc.PREFILL_ENABLED = True
    torch.cuda.synchronize()
    fb = hx.diff_count(ref, got) + hx.diff_count(ref, got2)
    print(f"forced fallback: launch refused, mismatched {fb}", flush=True)
    bad += fb
    print(f"RESULT cases={cases} mismatched_elements={bad}", flush=True)
    assert bad == 0

calls = []
for li in packed:
    ex = exs[li]
    x, ids = caps[1][li][0].to(dev), caps[1][li][1].to(dev).to(torch.int32)
    kw = kwargs(ex, ids)
    o = torch.empty(496 * hx.TOP_K, hx.INTER, dtype=torch.bfloat16, device=dev)
    calls.append((ex, x, kw, o))


def run_all(use_nvsc):
    def run():
        for ex, x, kw, o in calls:
            fused.forward_relu2(layer=ex.layer, inputs=x, outputs=o, **kw,
                                nvsc=None if use_nvsc else False)
    return run


rows = {"stock": [], "nvsc": []}
for rep in range(args.reps):
    order = ["stock", "nvsc"] if rep % 2 == 0 else ["nvsc", "stock"]
    for name in order:
        ms = statistics.median(hx.graph_time(run_all(name == "nvsc"), 15))
        rows[name].append(ms * 1000 / len(calls))
s, v = statistics.median(rows["stock"]), statistics.median(rows["nvsc"])
diffs = [a - b for a, b in zip(rows["stock"], rows["nvsc"])]
print(json.dumps({"layers_timed": len(calls), "stock_us_per_layer": round(s, 2),
                  "nvsc_us_per_layer": round(v, 2),
                  "gain_us_per_layer_mean": round(statistics.mean(diffs), 2),
                  "gain_us_per_layer_sd": round(statistics.stdev(diffs), 2),
                  "gain_ms_per_prefill_23_layers": round(statistics.mean(diffs) * 23 / 1000, 3),
                  "stock_all": [round(a, 1) for a in rows["stock"]],
                  "nvsc_all": [round(a, 1) for a in rows["nvsc"]]}), flush=True)

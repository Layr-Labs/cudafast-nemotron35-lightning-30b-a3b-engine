#!/usr/bin/env python3
"""nvsc: bit-identity and per-call time of the routed up (w13 + ReLU^2) humming
kernel with 6-bit block scales (humming_nvsc.py) against the current kernel.

  bin/atom nemo exec SHA harness/tests/nvsc_exact.py [--layers 1,6,13,27,40,51]

exact: for every listed layer, M = 1..8 tokens, embedding-routed, random,
  uniformly routed and edge-case activations, plus three M = 8 calls that
  route through all 128 experts (every scale of the layer is decoded once):
  torch.equal on the bf16 bits of nvsc vs the current kernel. Layers left on
  the stock kernel (more than 64 scale values) must fall back.
time: per-call time, CUDA graphs. cold = one call per packed layer in turn
  (each call's experts stream from DRAM); warm = the same call repeated.
  M = 1, 5 (top-6 over random but fixed expert ids) and 8; for M > 1 also
  both kernels with 4 CTAs per SM on the decode buckets (*_c4).
Writes nvsc.json to $NEMO_OUT.
"""

import argparse
import json
import os
import pathlib
import statistics
import sys

import torch

os.environ["VLLM_HUMMING_BATCH_INVARIANT"] = "1"
os.environ.setdefault("NVSC", "1")
torch.set_num_threads(8)
sys.path.insert(0, os.path.dirname(__file__))
import humming_moe_exact as check  # noqa: E402
from vllm.model_executor.layers.fused_moe.experts import (  # noqa: E402
    humming_nvsc as nvsc,
)
from vllm.model_executor.layers.fused_moe.experts import (  # noqa: E402
    humming_relu2_epilogue as fused,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--layers", default="1,6,13,27,40,51")
parser.add_argument("--reps", type=int, default=6)
parser.add_argument("--skip-exact", action="store_true")
args = parser.parse_args()
layers = [int(x) for x in args.layers.split(",")]
OUT = os.environ.get("NEMO_OUT", ".")


def stock(ex, x, out, kw):
    fused.forward_relu2(ex.layer, x, out, **kw, nvsc=False)


def packed(ex, x, out, kw):
    assert nvsc.launch(ex.layer, x, out, kw["compute_config"], kw["tuning_config"],
                       kw["sorted_ids"], kw["expert_ids"], kw["num_tokens_padded"],
                       kw["top_k"], kw["valid_shape_m"])


snap = check.Snapshot(os.environ["NEMO_SNAPSHOT"])
emb = snap.load(["backbone.embeddings.weight"], "cuda")["backbone.embeddings.weight"]
exs, routers = {}, {}
for li in layers:
    ex = check.build_variant(snap.raw_experts(li, "cuda"),
                             f"backbone.layers.{li}.mixer.experts", True)
    bs = ex.layer.nvsc_w13_bs
    ws = ex.layer.w13_weight_scale
    print("layer", li, "packed", bs is not None, "scale bytes", ws.numel(),
          "values", int(torch.unique(ws.view(torch.uint8)).numel()), flush=True)
    exs[li] = ex
    routers[li] = snap.router(li, "cuda")
print("stats", json.dumps(nvsc._STATS), flush=True)
packed_layers = [li for li in layers if exs[li].layer.nvsc_w13_bs is not None]
assert packed_layers, "no layer packed"

result = {"layers": layers, "packed": packed_layers, "exact_cases": 0}
if not args.skip_exact:
    cases = 0
    for li in layers:
        ex = exs[li]
        inputs = []
        for m in range(1, 9):
            for kind in ["emb", "rand", "rand_uniform_route", "edge"]:
                inputs.append((m, kind, *check.make_inputs(kind, m, 7 + li + 13 * m, emb,
                                                           routers[li], "cuda")))
        gen = torch.Generator().manual_seed(li)
        for c in range(3):
            ids = ((torch.arange(48) + 48 * c) % 128).view(8, 6).to(torch.int32).cuda()
            x = (torch.randn(8, check.HIDDEN, generator=gen) * 2).to(torch.bfloat16).cuda()
            inputs.append((8, f"cover{c}", x, ids.contiguous(), None))
        for m, kind, x, ids, _ in inputs:
            kw, _ = ex.prepare_humming_moe_kwargs(ids, None, None)
            want = torch.empty(m * 6, check.INTER, dtype=torch.bfloat16, device="cuda")
            got = torch.full_like(want, float("nan"))
            stock(ex, x, want, kw)
            if li in packed_layers:
                packed(ex, x, got, kw)
            else:
                assert not nvsc.launch(ex.layer, x, got, kw["compute_config"],
                                       kw["tuning_config"], kw["sorted_ids"],
                                       kw["expert_ids"], kw["num_tokens_padded"],
                                       kw["top_k"], kw["valid_shape_m"])
                fused.forward_relu2(ex.layer, x, got, **kw)
            torch.cuda.synchronize()
            assert check.bits_equal(want, got), (li, m, kind, check.diff_count(want, got))
            cases += 1
        print("layer exact", li, "cumulative cases", cases, flush=True)
    result["exact_cases"] = cases
    print("ALL BIT-IDENTICAL", cases, flush=True)


def calls_for(m, seed):
    out = []
    for li in packed_layers:
        ex = exs[li]
        gen = torch.Generator().manual_seed(seed + li)
        ids, _ = check.random_route(m, gen, "cuda")
        x = check.make_inputs("emb", m, seed + 3 * li, emb, routers[li], "cuda")[0]
        kw, _ = ex.prepare_humming_moe_kwargs(ids, None, None)
        o = torch.empty(m * 6, check.INTER, dtype=torch.bfloat16, device="cuda")
        out.append((ex, x, o, kw, int(ids.unique().numel())))
    return out


def with_ctas(kw, ctas):
    """The same tuning list with num_ctas_per_sm = ctas on the decode buckets."""
    tc = json.loads(kw["tuning_config"])
    for lo, hi, config in tc:
        if hi <= nvsc.DECODE_MAX_VALID_M:
            config["num_ctas_per_sm"] = ctas
    return dict(kw, tuning_config=json.dumps(tc))


def ctas_variant(fn, ctas):
    return lambda ex, x, o, kw: fn(ex, x, o, with_ctas(kw, ctas))


timing = []
for m in [1, 5, 8]:
    calls = calls_for(m, 1000 + m)
    n = len(calls)
    variants = {
        "stock": stock,
        "nvsc": packed,
    }
    if m > 1:  # the M <= 2 buckets already run 4 CTAs per SM
        variants["stock_c4"] = ctas_variant(stock, 4)
        variants["nvsc_c4"] = ctas_variant(packed, 4)

    def cold(fn):
        def run():
            for ex, x, o, kw, _ in calls:
                fn(ex, x, o, kw)
        return run

    def warm(fn):
        ex, x, o, kw, _ = calls[0]

        def run():
            for _ in range(n):
                fn(ex, x, o, kw)
        return run

    rows = {}
    for rep in range(args.reps):
        order = list(variants) if rep % 2 == 0 else list(variants)[::-1]
        for mode, wrap in (("cold", cold), ("warm", warm)):
            for name in order:
                us = statistics.median(check.graph_time(wrap(variants[name]), 21)) * 1000 / n
                rows.setdefault((mode, name), []).append(us)
    for mode in ("cold", "warm"):
        s = statistics.median(rows[(mode, "stock")])
        r = {"m": m, "mode": mode, "distinct_experts": [c[4] for c in calls]}
        for name in variants:
            v = statistics.median(rows[(mode, name)])
            r[name + "_us"] = round(v, 2)
            r[name + "_gain_pct"] = round(100 * (s - v) / s, 2)
            r[name + "_all"] = [round(a, 2) for a in rows[(mode, name)]]
        timing.append(r)
        print(json.dumps(r), flush=True)
result["timing"] = timing
pathlib.Path(OUT, "nvsc.json").write_text(json.dumps(result, indent=1) + "\n")

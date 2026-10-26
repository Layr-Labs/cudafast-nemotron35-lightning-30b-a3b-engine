# SPDX-License-Identifier: Apache-2.0
"""Differential test of sparse down projection and complete expert integration.

Run in the pinned GPU container with the staged overlay and kernel packages:
python3 harness/tests/routed_sparse_exact.py --weights SNAPSHOT --output report.json
Use --layers all for all 23 routed layers. The reference calls Humming directly
and applies the ordinary activation and expert reduction independently.
"""

import argparse
import json
import os
import pathlib

import torch

os.environ["VLLM_HUMMING_BATCH_INVARIANT"] = "1"
torch.set_num_threads(8)
import humming_moe_exact as check
import routed_sparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--weights", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--layers", default="1,13,27,40,51")
args = parser.parse_args()
layers = (
    check.MOE_LAYERS
    if args.layers == "all"
    else [int(x) for x in args.layers.split(",")]
)
snap = check.Snapshot(args.weights)
emb = snap.load(["backbone.embeddings.weight"], "cuda")["backbone.embeddings.weight"]
original_forward = routed_sparse.forward
calls = []


def checked_forward(*args):
    result = original_forward(*args)
    calls.append(result)
    return result


routed_sparse.forward = checked_forward
report = []
for li in layers:
    raw = snap.raw_experts(li, "cuda")
    ex = check.build_variant(raw, f"backbone.layers.{li}.mixer.experts", True)
    assert ex.layer.routed_sparse_down is not None
    w, s, gs = ex.layer.routed_sparse_down
    assert torch.equal(
        w.permute(0, 1, 6, 4, 2, 5, 3).reshape_as(raw["w2_weight"]),
        raw["w2_weight"],
    )
    assert torch.equal(
        s.permute(0, 1, 4, 3, 2).reshape_as(raw["w2_weight_scale"]),
        raw["w2_weight_scale"].view(torch.uint8),
    )
    assert torch.equal(gs, raw["w2_weight_scale_2"].flatten())
    router = snap.router(li, "cuda")
    for m in [1, 2, 3, 4, 5, 6, 7, 8, 16, 496, 512]:
        for kind in ["emb", "rand", "rand_uniform_route", "edge"]:
            x, ids, weights = check.make_inputs(
                kind, m, 1000 + li + m, emb, router, "cuda"
            )
            reference = check.run_stages(ex, x, ids, weights)
            calls.clear()
            actual = check.run_apply(ex, x, ids, weights)
            assert calls == [m <= 8], (li, m, kind, calls, ids.dtype)
            differences = check.diff_count(reference["out"], actual)
            assert differences == 0, (li, m, kind, differences)
            row = dict(layer=li, tokens=m, kind=kind, bit_differences=differences)
            report.append(row)
    print("layer exact", li, "cases", len(report), flush=True)
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
print("all cases exact", len(report), flush=True)

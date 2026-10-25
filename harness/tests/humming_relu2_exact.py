# SPDX-License-Identifier: Apache-2.0
"""GPU differential proof for the exact Humming expert-up/ReLU-squared fusion.

Run with the staged candidate overlay in the pinned container:
  python3 harness/tests/humming_relu2_exact.py --weights SNAPSHOT --output result.json
Use --layers all for every routed layer. This compares against the independent
unfused projection/activation/down/combine sequence, including final buffer
aliasing through HummingIndexedExperts.apply. It is not a benchmark score.
"""

import argparse
import json
import os
import pathlib
import statistics

import torch

os.environ["VLLM_HUMMING_BATCH_INVARIANT"] = "1"
torch.set_num_threads(8)
import humming_moe_exact as check
from vllm.model_executor.layers.fused_moe.experts import humming_relu2_epilogue as fused

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
assert layers and all(x in check.MOE_LAYERS for x in layers)
from vllm.model_executor.layers.fused_moe.experts.humming_tuned_moe import relu2_bf16
from vllm.triton_utils import triton, tl


@triton.jit
def activation(src, dst, N: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    v = tl.load(src + i, mask=i < N, other=0)
    y = tl.inline_asm_elementwise(
        """{ .reg .b16 lo, hi; .reg .f32 x, y;
 mov.b32 {lo, hi}, $1;
 cvt.f32.bf16 x, lo; cvt.f32.bf16 y, hi;
 max.NaN.f32 x, x, 0f00000000; max.NaN.f32 y, y, 0f00000000;
 mul.rn.f32 x, x, x; mul.rn.f32 y, y, y;
 cvt.rn.bf16x2.f32 $0, y, x; }""",
        constraints="=r,r",
        args=[v],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    tl.store(dst + i, y, mask=i < N)


x = (
    torch.arange(65536, device="cuda", dtype=torch.int32)
    .to(torch.int16)
    .view(torch.bfloat16)
)
want = torch.empty_like(x)
got = torch.empty_like(x)
assert relu2_bf16(x, want)
activation[(128,)](x.view(torch.int32), got.view(torch.int32), 32768)
bad = (want.view(torch.int16) != got.view(torch.int16)).nonzero().flatten()
print("all-bf16 activation mismatch count", bad.numel(), flush=True)
if bad.numel():
    print(
        [
            (int(i), int(want.view(torch.int16)[i]), int(got.view(torch.int16)[i]))
            for i in bad[:20]
        ],
        flush=True,
    )
assert not bad.numel()
from vllm.utils.humming import HummingMethod

snap = check.Snapshot(args.weights)
emb = snap.load(["backbone.embeddings.weight"], "cuda")["backbone.embeddings.weight"]
items = []
cases = 0
for li in layers:
    ex = check.build_variant(
        snap.raw_experts(li, "cuda"), f"backbone.layers.{li}.mixer.experts", True
    )
    router = snap.router(li, "cuda")
    for m in [1, 2, 3, 4, 5, 8, 16, 496, 512]:
        for kind in ["emb", "rand", "rand_uniform_route", "edge"]:
            x, ids, weights = check.make_inputs(
                kind, m, 111 + li + m, emb, router, "cuda"
            )
            kw1, _ = ex.prepare_humming_moe_kwargs(ids, None, None)
            reference = check.run_stages(ex, x, ids, weights)
            expected = reference["act"]
            actual = torch.empty_like(expected)
            fused.forward_relu2(ex.layer, x, actual, **kw1)
            assert check.bits_equal(expected, actual), (
                li,
                m,
                kind,
                check.diff_count(expected, actual),
            )
            complete = check.run_apply(ex, x, ids, weights)
            assert check.bits_equal(reference["out"], complete), (
                li,
                m,
                kind,
                "complete expert output",
            )
            cases += 1
            if kind == "emb" and m in [4, 496] and li in layers[:5]:
                items.append(
                    (
                        m,
                        ex,
                        x,
                        kw1,
                        torch.empty_like(actual),
                        torch.empty_like(actual),
                        torch.empty_like(actual),
                    )
                )
    print("layer exact", li, "cumulative cases", cases, flush=True)
measure = []
for m in [4, 496]:
    data = [x for x in items if x[0] == m]

    def base():
        for _, ex, x, kw1, raw, act, got in data:
            HummingMethod.forward_layer(
                layer=ex.layer, inputs=x, outputs=raw, sublayer_name="w13", **kw1
            )
            relu2_bf16(raw, act)

    def candidate():
        for _, ex, x, kw1, raw, act, got in data:
            fused.forward_relu2(ex.layer, x, got, **kw1)

    for rep in range(4):
        for name, fn in (
            [("base", base), ("fused", candidate)]
            if rep % 2 == 0
            else [("fused", candidate), ("base", base)]
        ):
            us = statistics.median(check.graph_time(fn, 11)) * 1000 / len(data)
            r = dict(m=m, rep=rep, name=name, us=us)
            measure.append(r)
            print(json.dumps(r), flush=True)
pathlib.Path(args.output).write_text(
    json.dumps(
        {
            "passed": True,
            "all_bf16": 65536,
            "real_weight_cases": cases,
            "measurements": measure,
        },
        indent=2,
    )
    + "\n"
)

# SPDX-License-Identifier: Apache-2.0
"""GPU proof for packed native FP8 projections against pinned CUTLASS.

Run inside the pinned image with staged overlay/kernels on PYTHONPATH:
  python3 harness/tests/fp8small_exact.py --weights SNAPSHOT --output /tmp/proof.json
The final timings cover isolated kernels; this script produces no official score.
"""

import argparse
import json
import pathlib
import statistics

import torch
import humming_moe_exact as check
import fp8small
from vllm.model_executor.kernels.linear.scaled_mm import fp8small as dispatch  # noqa: F401

torch.set_num_threads(8)
torch.manual_seed(7942)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--weights", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
from vllm import _custom_ops as ops

snap = check.Snapshot(args.weights)
layers = sorted(
    int(k.split(".")[2]) for k in snap.weight_map if k.endswith(".mixer.in_proj.weight")
)
assert len(layers) == 23
items = []
cases = 0
elements = 0
for li in layers:
    for proj in ["in_proj", "out_proj"]:
        p = f"backbone.layers.{li}.mixer.{proj}"
        names = [p + "." + s for s in ["weight", "weight_scale", "input_scale"]]
        t = snap.load(names, "cuda")
        w = t[names[0]].contiguous()
        sb = t[names[1]].max().reshape(1)
        sa = t[names[2]].max().reshape(1)
        n, k = w.shape
        packed = fp8small.prepare(w.T)
        assert packed is not None
        # Independent tensor reshape/permutation: every byte must be retained.
        inverse = (
            packed.view(n // 32, k // 32, 4, 8, 4, 2, 4)
            .permute(0, 2, 3, 1, 5, 4, 6)
            .reshape(n, k)
        )
        assert torch.equal(inverse, w.view(torch.uint8)), (li, proj, "repack inverse")
        for m in range(1, 9):
            for kind in ["random", "small", "extreme", "all_bits"]:
                if kind == "all_bits":
                    a = (
                        torch.arange(m * k, device="cuda", dtype=torch.int64)
                        .to(torch.uint8)
                        .view(torch.float8_e4m3fn)
                        .reshape(m, k)
                    )
                else:
                    scale = {"random": 20, "small": 0.02, "extreme": 150}[kind]
                    a = (
                        (torch.randn((m, k), device="cuda") * scale)
                        .clamp(-448, 448)
                        .to(torch.float8_e4m3fn)
                    )
                ref = ops.cutlass_scaled_mm(
                    a, w.T, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16
                )
                for ring in [16]:
                    got = torch.ops.vllm.fp8small_scaled_mm(a, w.T, packed, sa, sb)
                    bad = (ref.view(torch.int16) != got.view(torch.int16)).sum().item()
                    assert bad == 0, (li, proj, m, kind, ring, bad)
                    cases += 1
                    elements += ref.numel()
                if li in [0, 11, 25, 39, 50] and kind == "random" and m in [1, 4, 8]:
                    items.append((proj, m, a, w, sa, sb, packed))
    print("layer exact", li, "cases", cases, flush=True)
# Actual opaque operation under a symbolic compiled graph: dispatch must stay
# inside the operation rather than becoming a tracing-time M decision.
_, _, a, w, sa, sb, packed = items[0]


def projected(x):
    return torch.ops.vllm.fp8small_scaled_mm(x, w.T, packed, sa, sb)


compiled = torch.compile(projected, dynamic=True, fullgraph=True)
for m in [1, 4, 9, 16, 496]:
    x = (torch.randn((m, w.shape[1]), device="cuda") * 20).to(torch.float8_e4m3fn)
    expected = ops.cutlass_scaled_mm(
        x, w.T, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16
    )
    actual = compiled(x)
    assert torch.equal(expected.view(torch.int16), actual.view(torch.int16)), (
        "compiled",
        m,
    )
print("compiled small and fallback M exact", flush=True)
x = (torch.randn((4, w.shape[1]), device="cuda") * 20).to(torch.float8_e4m3fn)
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    for _ in range(3):
        projected(x)
torch.cuda.current_stream().wait_stream(stream)
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    graph_out = projected(x)
for _ in range(3):
    x.copy_((torch.randn(x.shape, device="cuda") * 20).to(torch.float8_e4m3fn))
    graph.replay()
    expected = ops.cutlass_scaled_mm(
        x, w.T, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16
    )
    assert torch.equal(expected.view(torch.int16), graph_out.view(torch.int16))
print("graph replay with changed input exact", flush=True)
measure = []
for proj in ["in_proj", "out_proj"]:
    for m in [1, 4, 8]:
        data = [x for x in items if x[0] == proj and x[1] == m]

        def base():
            for _, _, a, w, sa, sb, got in data:
                ops.cutlass_scaled_mm(
                    a, w.T, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16
                )

        def candidate(ring):
            for _, _, a, w, sa, sb, packed in data:
                torch.ops.vllm.fp8small_scaled_mm(a, w.T, packed, sa, sb)

        for rep in range(4):
            fns = [("base", base), ("ring16", lambda: candidate(16))]
            fns = fns[rep % len(fns) :] + fns[: rep % len(fns)]
            for name, fn in fns:
                us = (
                    statistics.median(check.graph_time(fn, 11, warmup=8))
                    * 1000
                    / len(data)
                )
                r = dict(proj=proj, m=m, rep=rep, name=name, us=us)
                measure.append(r)
                print(json.dumps(r), flush=True)
pathlib.Path(args.output).write_text(
    json.dumps(
        dict(passed=True, cases=cases, elements=elements, measurements=measure),
        indent=2,
    )
    + "\n"
)

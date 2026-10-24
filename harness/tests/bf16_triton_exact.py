"""Production-path proof for guarded small-M BF16 persistent GEMM tiles."""

import json
import os

import torch

from bf16_triton_sweep import load_weight, make_input
from marlinx_exact import graph_time
from vllm.model_executor.layers.utils import default_unquantized_gemm

LAYERS = [5, 12, 19, 26, 33, 42]
EXACT_M = [*range(1, 9), 9, 16, 496]
TIMING_M = [1, 4, 5, 8]
KINDS = ["normal", "heavy", "small", "bits"]


def base_mm(a, b):
    return torch.nn.functional.linear(a, b.t())


def candidate_linear(a, weight):
    return default_unquantized_gemm(None, a, weight)


def main():
    snap = os.environ["NEMO_SNAPSHOT"]
    out_path = os.environ["NEMO_OUT"]
    with open(os.path.join(snap, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    weights = {
        (li, proj): load_weight(snap, wmap, li, proj)
        for li in LAYERS
        for proj in ("qkv", "o")
    }

    cases = 0
    elements = 0
    for (li, proj), weight in weights.items():
        k = weight.shape[1]
        for m in EXACT_M:
            gen = torch.Generator(device="cuda").manual_seed(1000 + li * 17 + m)
            for kind in KINDS:
                x = make_input(m, k, kind, gen)
                ref = base_mm(x, weight.t())
                got = candidate_linear(x, weight)
                bad = (ref.view(torch.int16) != got.view(torch.int16)).sum().item()
                assert bad == 0, (li, proj, m, kind, bad)
                cases += 1
                elements += ref.numel()
        print("layer exact", li, proj, "cases", cases, flush=True)

    compiled_cases = 0
    base_compiled = torch.compile(
        lambda a, w: torch.nn.functional.linear(a, w), dynamic=True, fullgraph=True
    )
    for (li, proj), weight in weights.items():
        k = weight.shape[1]
        compiled = torch.compile(candidate_linear, dynamic=True, fullgraph=True)
        for m in EXACT_M:
            x = make_input(
                m, k, "normal", torch.Generator(device="cuda").manual_seed(m + li)
            )
            ref = base_compiled(x, weight)
            got = compiled(x, weight)
            assert torch.equal(ref.view(torch.int16), got.view(torch.int16)), (
                li,
                proj,
                m,
                "compiled",
            )
            compiled_cases += 1
        print("compiled exact", li, proj, flush=True)

    replay_cases = 0
    for (li, proj), weight in weights.items():
        k = weight.shape[1]
        for m in EXACT_M:
            x = make_input(
                m, k, "normal", torch.Generator(device="cuda").manual_seed(li + m)
            )
            for _ in range(2):
                candidate_linear(x, weight)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_out = candidate_linear(x, weight)
            for kind in KINDS:
                replacement = make_input(
                    m,
                    k,
                    kind,
                    torch.Generator(device="cuda").manual_seed(li + 500 + m),
                )
                x.copy_(replacement)
                graph.replay()
                ref = base_mm(x, weight.t())
                assert torch.equal(
                    ref.view(torch.int16), graph_out.view(torch.int16)
                ), (li, proj, m, kind, "replay")
                replay_cases += 1
            del graph
        print("replay exact", li, proj, flush=True)

    request_cases = 0
    for (li, proj), weight in weights.items():
        k = weight.shape[1]
        gen = torch.Generator(device="cuda").manual_seed(li + 900)
        x1 = make_input(4, k, "normal", gen)
        x2 = make_input(4, k, "normal", gen)
        first = candidate_linear(x1, weight)
        saved = first.clone()
        second = candidate_linear(x2, weight)
        repeat = candidate_linear(x1, weight)
        assert torch.equal(first.view(torch.int16), saved.view(torch.int16))
        assert torch.equal(first.view(torch.int16), repeat.view(torch.int16))
        assert torch.equal(first.view(torch.int16), base_mm(x1, weight.t()).view(torch.int16))
        assert torch.equal(second.view(torch.int16), base_mm(x2, weight.t()).view(torch.int16))
        request_cases += 1

    measurements = []
    for proj in ("qkv", "o"):
        for m in TIMING_M:
            data = []
            for li in LAYERS:
                weight = weights[li, proj]
                x = make_input(
                    m,
                    weight.shape[1],
                    "normal",
                    torch.Generator(device="cuda").manual_seed(li),
                )
                data.append((weight, x))
            for rep in range(4):
                variants = ["base", "candidate"]
                variants = variants[rep % 2 :] + variants[: rep % 2]
                for name in variants:

                    def timed(name=name):
                        for weight, x in data:
                            if name == "base":
                                base_mm(x, weight.t())
                            else:
                                candidate_linear(x, weight)

                    us = graph_time(timed, 7) * 1000 / len(data)
                    row = dict(proj=proj, m=m, rep=rep, name=name, us=us)
                    measurements.append(row)
                    print(json.dumps(row), flush=True)

    result = dict(
        passed=True,
        cases=cases,
        elements=elements,
        compiled_cases=compiled_cases,
        replay_cases=replay_cases,
        request_cases=request_cases,
        measurements=measurements,
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

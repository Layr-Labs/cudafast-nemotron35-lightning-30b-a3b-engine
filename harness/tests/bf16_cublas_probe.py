"""Compare small-M persistent BF16 tiles with the actual cuBLAS dispatch."""

import json
import os
import statistics

import torch

from bf16_triton_sweep import LAYERS, load_weight, make_input, mm_config
from marlinx_exact import graph_time

BASE = (128, 128, 64, 8, 3)
CONFIGS = {
    "persistent": BASE,
    "small64": (16, 64, 64, 4, 4),
    "small128": (16, 128, 64, 4, 4),
    "small_k32": (16, 64, 32, 4, 4),
    "small_k128": (16, 64, 128, 4, 4),
}


def main():
    snap = os.environ["NEMO_SNAPSHOT"]
    with open(os.path.join(snap, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    weights = {
        (li, proj): load_weight(snap, wmap, li, proj)
        for li in LAYERS
        for proj in ("qkv", "o")
    }
    for proj in ("qkv", "o"):
        for m in (1, 4, 8):
            data = [
                (
                    weight,
                    make_input(
                        m,
                        weight.shape[1],
                        "normal",
                        torch.Generator(device="cuda").manual_seed(1000 + li * 17 + m),
                    ),
                )
                for li in LAYERS
                for weight in [weights[li, proj]]
            ]
            references = [torch.nn.functional.linear(x, w) for w, x in data]
            for name, config in CONFIGS.items():
                bad = 0
                for (w, x), ref in zip(data, references):
                    got = mm_config(x, w.t(), config)
                    bad += (ref.view(torch.int16) != got.view(torch.int16)).sum().item()
                times = []
                for rep in range(4):
                    variants = ["cublas", "candidate"]
                    variants = variants[rep % 2 :] + variants[: rep % 2]
                    pair = {}
                    for variant in variants:

                        def timed(variant=variant):
                            for w, x in data:
                                if variant == "cublas":
                                    torch.nn.functional.linear(x, w)
                                else:
                                    mm_config(x, w.t(), config)

                        pair[variant] = graph_time(timed, 7) * 1000 / len(data)
                    times.append(pair)
                base = statistics.median(t["cublas"] for t in times)
                candidate = statistics.median(t["candidate"] for t in times)
                print(
                    json.dumps(
                        dict(
                            proj=proj,
                            m=m,
                            config=name,
                            bad=bad,
                            cublas_us=base,
                            candidate_us=candidate,
                            times=times,
                        )
                    ),
                    flush=True,
                )

    old = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    for backend in ("cublas", "cublaslt"):
        torch.backends.cuda.preferred_blas_library(backend=backend)
        for reduced in (True, False):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced
            bad = 0
            for li in LAYERS:
                w = weights[li, "qkv"]
                x = make_input(
                    1,
                    w.shape[1],
                    "normal",
                    torch.Generator(device="cuda").manual_seed(1000 + li * 17 + 1),
                )
                ref = torch.nn.functional.linear(x, w)
                got = mm_config(x, w.t(), CONFIGS["small64"])
                bad += (ref.view(torch.int16) != got.view(torch.int16)).sum().item()
            print("SETTINGS", backend, reduced, "bad", bad, flush=True)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old
    torch.backends.cuda.preferred_blas_library(backend="cublas")


if __name__ == "__main__":
    main()

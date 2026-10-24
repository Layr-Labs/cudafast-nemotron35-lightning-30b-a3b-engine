"""K-tile sweep against cuBLAS for the M=5 attention verification shape."""

import json
import os
import statistics

import torch

from bf16_triton_sweep import LAYERS, load_weight, make_input, mm_config
from marlinx_exact import graph_time

CONFIGS = [
    (16, 64, 128, 4, 2),
    (16, 64, 128, 4, 3),
    (16, 64, 128, 4, 4),
    (16, 64, 128, 4, 5),
    (16, 64, 256, 4, 2),
    (16, 64, 256, 4, 3),
    (16, 64, 256, 8, 3),
]


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
        data = [
            (
                weights[li, proj],
                make_input(
                    5,
                    weights[li, proj].shape[1],
                    "normal",
                    torch.Generator(device="cuda").manual_seed(li),
                ),
            )
            for li in LAYERS
        ]
        references = [torch.nn.functional.linear(x, w) for w, x in data]
        for config in CONFIGS:
            bad = 0
            for (w, x), ref in zip(data, references):
                got = mm_config(x, w.t(), config)
                bad += (ref.view(torch.int16) != got.view(torch.int16)).sum().item()
            assert bad == 0, (proj, config, bad)
            times = []
            for rep in range(4):
                variants = ["base", "candidate"]
                variants = variants[rep % 2 :] + variants[: rep % 2]
                pair = {}
                for name in variants:

                    def timed(name=name):
                        for w, x in data:
                            if name == "base":
                                torch.nn.functional.linear(x, w)
                            else:
                                mm_config(x, w.t(), config)

                    pair[name] = graph_time(timed, 7) * 1000 / len(data)
                times.append(pair)
            base = statistics.median(t["base"] for t in times)
            candidate = statistics.median(t["candidate"] for t in times)
            print(
                json.dumps(
                    dict(
                        proj=proj,
                        config=config,
                        bad=bad,
                        base_us=base,
                        candidate_us=candidate,
                        faster_pct=(base / candidate - 1) * 100,
                        times=times,
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()

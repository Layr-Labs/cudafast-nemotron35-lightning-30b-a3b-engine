#!/usr/bin/env python3
"""GPU bit-identity check and optional graph timing for grouped packed_gemv.

Run inside the pinned image with this tree's overlay on PYTHONPATH:
  python3 harness/tests/packed_gemv_group8_exact.py --weights SNAPSHOT --output /tmp/group8.json
Add --time to compare CUDA-graph timings after every equality check passes.
"""

import argparse
import json
import pathlib
import statistics

import torch
import humming_moe_exact as check
from vllm.model_executor.models import nemotron_h_mtp_packed as mtp


def run_gemv(item, x, word=False, group8=False, out=None):
    p = item["packed"]
    n, k = p.shape
    y = x.new_empty((1, n)) if out is None else out
    mtp.packed_gemv[((n + 7) // 8,)](
        x, *p.planes, y, n, k, k, item["gate"],
        BN=8, BK=512, EVICT=True, GATED=True, launch_pdl=True,
        WORD=word, GROUP8=group8,
        num_warps=4, num_stages=3,
    )
    return y


def load_weights(snapshot):
    q_names = [
        name for name in snapshot.weight_map
        if name.startswith("mtp.") and name.endswith(".mixer.q_proj.weight")
    ]
    assert len(q_names) == 1, q_names
    prefix = q_names[0][:-len("q_proj.weight")]
    names = [prefix + proj + ".weight" for proj in ("q_proj", "k_proj", "v_proj")]
    tensors = snapshot.load(names + [prefix + "o_proj.weight"], "cuda")
    assert all(t.dtype == torch.bfloat16 for t in tensors.values())
    qkv = torch.cat([tensors[name] for name in names], dim=0).contiguous()
    o = tensors[prefix + "o_proj.weight"].contiguous()
    assert qkv.shape == (4608, 2688), qkv.shape
    assert o.shape == (2688, 4096), o.shape
    cases = [("mtp_qkv", qkv), ("mtp_o", o)]
    layers = sorted({int(name.split(".")[2]) for name in snapshot.weight_map
                     if name.startswith("backbone.layers.")
                     and name.endswith("q_proj.weight")})
    assert len(layers) == 6
    for li in layers:
        prefix = f"backbone.layers.{li}.mixer."
        names = [prefix + part + "_proj.weight" for part in ("q", "k", "v", "o")]
        tensors = snapshot.load(names, "cuda")
        qkv = torch.cat([tensors[name] for name in names[:3]], dim=0).contiguous()
        o = tensors[names[3]].contiguous()
        assert qkv.shape == (4608, 2688) and o.shape == (2688, 4096)
        cases.extend(((f"target_{li}_qkv", qkv), (f"target_{li}_o", o)))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output")
    parser.add_argument("--time", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(91274)
    snapshot = check.Snapshot(args.weights)
    mtp._WORD_LOAD = False
    gate = torch.zeros(1, dtype=torch.int32, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(18731)
    cases = []
    raw_blocks = 0
    for name, weight in load_weights(snapshot):
        p = mtp.pack_bf16(weight)
        assert p is not None
        escapes = int((p.planes[3] >= 0).sum().item())
        raw_blocks += escapes
        cases.append(dict(name=name, packed=p, gate=gate, escapes=escapes))
        print("packed", name, "shape", p.shape, "raw_blocks", escapes, flush=True)
    assert raw_blocks > 0, "the pinned real-weight cases did not exercise RAW escapes"

    comparisons = 0
    elements = 0
    for item in cases:
        n, k = item["packed"].shape
        for scale in (0.001, 1.0, 32.0):
            x = (torch.randn((1, k), dtype=torch.float32, generator=gen,
                             device="cuda") * scale).to(torch.bfloat16)
            original = run_gemv(item, x)
            original_word = run_gemv(item, x, word=True)
            grouped = run_gemv(item, x, group8=True)
            for variant, actual in (("WORD", original_word), ("GROUP8", grouped)):
                bad = (original.view(torch.int16) != actual.view(torch.int16)).sum().item()
                assert bad == 0, (item["name"], scale, variant, bad, n, k)
                comparisons += 1
                elements += original.numel()
            if scale == 1.0:
                item["timing_x"] = x
                item["timing_y"] = x.new_empty((1, n))
        print("bit exact", item["name"], "raw_blocks", item["escapes"], flush=True)
    torch.cuda.synchronize()

    measurements = []
    if args.time:
        variants = [("BYTE", False, False), ("WORD", True, False),
                    ("GROUP8", False, True)]
        for rep in range(4):
            ordered = variants[rep % len(variants):] + variants[:rep % len(variants)]
            for shape in sorted({item["packed"].shape for item in cases}):
                items = [item for item in cases if item["packed"].shape == shape]
                for name, word, group8 in ordered:
                    def timed(word=word, group8=group8, items=items):
                        for item in items:
                            run_gemv(item, item["timing_x"], word=word, group8=group8,
                                     out=item["timing_y"])

                    times = check.graph_time(timed, 11, warmup=5)
                    row = dict(rep=rep, variant=name, shape=list(shape), graph_ms=times,
                               median_us_per_launch=statistics.median(times) * 1000 / len(items))
                    measurements.append(row)
                    print(json.dumps(row), flush=True)

    report = dict(passed=True, comparisons=comparisons, elements=elements,
                  raw_blocks=raw_blocks,
                  cases=[dict(name=item["name"], shape=item["packed"].shape,
                              raw_blocks=item["escapes"]) for item in cases],
                  measurements=measurements)
    if args.output:
        pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print("PASS packed_gemv GROUP8", "comparisons", comparisons,
          "raw_blocks", raw_blocks, "timed", args.time, flush=True)


if __name__ == "__main__":
    main()

"""Exactness and timing sweep for small-M BF16 persistent GEMM tiles."""

import json
import os
import statistics

import torch
from safetensors import safe_open

from marlinx_exact import graph_time
from vllm.model_executor.layers.batch_invariant import (
    matmul_batch_invariant,
    matmul_kernel_persistent,
)
from vllm.triton_utils import triton
from vllm.utils.platform_utils import num_compute_units

LAYERS = [5, 12, 19, 26, 33, 42]
SMALL_M = [1, 4, 8]
KINDS = ["normal", "heavy", "small", "bits"]
CONFIGS = [
    (16, 64, 64, 4, 2),
    (16, 64, 64, 4, 3),
    (16, 64, 64, 4, 4),
    (16, 128, 64, 4, 3),
    (16, 128, 64, 4, 4),
    (16, 128, 64, 8, 3),
    (16, 256, 64, 8, 3),
    (32, 64, 64, 4, 3),
]


def make_input(m, k, kind, gen):
    if kind == "bits":
        return torch.randint(
            -32768,
            32767,
            (m, k),
            generator=gen,
            device="cuda",
            dtype=torch.int16,
        ).view(torch.bfloat16)
    scale = {"normal": 1.0, "heavy": 30.0, "small": 0.001}[kind]
    return (torch.randn((m, k), generator=gen, device="cuda") * scale).to(
        torch.bfloat16
    )


def load_weight(snap, wmap, li, proj):
    prefix = f"backbone.layers.{li}.mixer."
    if proj == "qkv":
        tensors = []
        for name in ("q_proj", "k_proj", "v_proj"):
            key = prefix + name + ".weight"
            with safe_open(os.path.join(snap, wmap[key]), "pt", device="cpu") as f:
                tensors.append(f.get_tensor(key))
        return torch.cat(tensors, dim=0).contiguous().cuda()
    key = prefix + "o_proj.weight"
    with safe_open(os.path.join(snap, wmap[key]), "pt", device="cpu") as f:
        return f.get_tensor(key).contiguous().cuda()


def mm_config(a, b, config):
    bm, bn, bk, warps, stages = config
    m, k = a.shape
    k, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    num_sms = num_compute_units(a.device.index)

    def grid(meta):
        return (min(num_sms, triton.cdiv(m, bm) * triton.cdiv(n, bn)),)

    matmul_kernel_persistent[grid](
        a,
        b,
        c,
        None,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        NUM_SMS=num_sms,
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        HAS_BIAS=False,
        BLOCK_SIZE_M=bm,
        BLOCK_SIZE_N=bn,
        BLOCK_SIZE_K=bk,
        GROUP_SIZE_M=8,
        num_stages=stages,
        num_warps=warps,
    )
    return c


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
    inputs = {}
    references = {}
    for (li, proj), weight in weights.items():
        k = weight.shape[1]
        for m in SMALL_M:
            for kind in KINDS:
                gen = torch.Generator(device="cuda").manual_seed(
                    1000 + li * 17 + m
                )
                x = make_input(m, k, kind, gen)
                inputs[li, proj, m, kind] = x
                references[li, proj, m, kind] = matmul_batch_invariant(
                    x, weight.t()
                )

    exact_cases = 0
    replay_cases = 0
    for config in CONFIGS:
        for (li, proj), weight in weights.items():
            for m in SMALL_M:
                for kind in KINDS:
                    x = inputs[li, proj, m, kind]
                    got = mm_config(x, weight.t(), config)
                    ref = references[li, proj, m, kind]
                    bad = (ref.view(torch.int16) != got.view(torch.int16)).sum().item()
                    assert bad == 0, (config, li, proj, m, kind, bad)
                    exact_cases += 1
        for proj in ("qkv", "o"):
            weight = weights[5, proj]
            x = inputs[5, proj, 4, "normal"].clone()
            for _ in range(2):
                mm_config(x, weight.t(), config)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_out = mm_config(x, weight.t(), config)
            for kind in KINDS:
                x.copy_(inputs[5, proj, 4, kind])
                graph.replay()
                ref = matmul_batch_invariant(x, weight.t())
                assert torch.equal(
                    ref.view(torch.int16), graph_out.view(torch.int16)
                ), (config, proj, kind, "replay")
                replay_cases += 1
            del graph
        print("exact/replay", config, exact_cases, replay_cases, flush=True)

    measurements = []
    for config in CONFIGS:
        for proj in ("qkv", "o"):
            for m in SMALL_M:
                data = [
                    (weights[li, proj], inputs[li, proj, m, "normal"])
                    for li in LAYERS
                ]
                for rep in range(4):
                    variants = ["base", "candidate"]
                    variants = variants[rep % 2 :] + variants[: rep % 2]
                    for name in variants:

                        def timed(name=name):
                            for weight, x in data:
                                if name == "base":
                                    matmul_batch_invariant(x, weight.t())
                                else:
                                    mm_config(x, weight.t(), config)

                        us = graph_time(timed, 7) * 1000 / len(data)
                        row = dict(
                            config=config,
                            proj=proj,
                            m=m,
                            rep=rep,
                            name=name,
                            us=us,
                        )
                        measurements.append(row)
                        print(json.dumps(row), flush=True)

    result = dict(
        passed=True,
        exact_cases=exact_cases,
        replay_cases=replay_cases,
        measurements=measurements,
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    for proj in ("qkv", "o"):
        for m in SMALL_M:
            rows = [r for r in measurements if r["proj"] == proj and r["m"] == m]
            base = statistics.median(
                r["us"] for r in rows if r["name"] == "base"
            )
            ranked = []
            for config in CONFIGS:
                candidate = statistics.median(
                    r["us"]
                    for r in rows
                    if r["name"] == "candidate" and r["config"] == config
                )
                ranked.append((candidate, config))
            print(
                "SUMMARY",
                proj,
                m,
                "base_us",
                base,
                "candidates",
                sorted(ranked)[:3],
                flush=True,
            )


if __name__ == "__main__":
    main()

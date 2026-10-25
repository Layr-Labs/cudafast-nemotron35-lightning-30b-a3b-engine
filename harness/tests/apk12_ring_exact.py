"""apk12_ring_exact.py SNAPSHOT: exactness and cold graph timing for loader rings."""
import json
import os
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

import apk12
from vllm.model_executor.models.nemotron_h_mtp_packed import pack_bf16

snap = sys.argv[1]
rings = (1, 2, 4, 8)
with open(os.path.join(snap, "model.safetensors.index.json")) as f:
    wmap = json.load(f)["weight_map"]
names = []
for layer in (5, 12, 19, 26, 33, 42):
    prefix = f"backbone.layers.{layer}.mixer"
    qkv = prefix + ".qkv_proj.weight"
    parts = [f"{prefix}.{part}_proj.weight" for part in ("q", "k", "v")]
    assert all(part in wmap for part in parts)
    names.append((qkv, parts))
    output = prefix + ".o_proj.weight"
    assert output in wmap
    names.append((output, [output]))
assert apk12.ready(torch.device("cuda"))
items = []
for name, parts in names:
    tensors = []
    for part in parts:
        with safe_open(os.path.join(snap, wmap[part]), "pt", device="cpu") as f:
            tensors.append(f.get_tensor(part))
    w = torch.cat(tensors).cuda() if len(tensors) > 1 else tensors[0].cuda()
    n, k = w.shape
    assert (n, k) in apk12.SHAPES, (name, w.shape)
    packed = pack_bf16(w)
    assert packed is not None
    sm, ex, base, ri, raw = packed.planes
    cp = apk12.pack((sm, ex, base, ri, raw.view(torch.int16)), n, k)
    assert apk12.check(cp, w, n, k) == 0
    items.append((name, w, cp, n, k))
assert len(items) == 12, len(items)
assert sum((n, k) == (4608, 2688) for _, _, _, n, k in items) == 6
assert sum((n, k) == (2688, 4096) for _, _, _, n, k in items) == 6
print("real projections", len(items), flush=True)

gen = torch.Generator(device="cuda")
gen.manual_seed(20261025)
bad = {ring: 0 for ring in rings}
cases = {ring: 0 for ring in rings}
for name, w, cp, n, k in items:
    for scale in (1.0, 30.0, 1e-3):
        for _ in range(8):
            x = (torch.randn(5, k, device="cuda", dtype=torch.float32, generator=gen) * scale).to(torch.bfloat16)
            ref = F.linear(x, w)
            for ring in rings:
                out = apk12.gemm(x, cp, n, k, ring=ring)
                bad[ring] += int(not torch.equal(ref.view(torch.int16), out.view(torch.int16)))
                cases[ring] += 1
print("exact cases", cases, "mismatches", bad, flush=True)
assert not any(bad.values())

replays = 0
for ring in rings:
    for shape in sorted(apk12.SHAPES):
        name, w, cp, n, k = next(item for item in items if item[3:] == shape)
        x = torch.randn(5, k, device="cuda", dtype=torch.bfloat16, generator=gen)
        out = apk12.gemm(x, cp, n, k, ring=ring)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = apk12.gemm(x, cp, n, k, ring=ring)
        for scale in (1.0, 30.0, 1e-3, 1.0):
            x.copy_((torch.randn(5, k, device="cuda", dtype=torch.float32, generator=gen) * scale).to(torch.bfloat16))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out.view(torch.int16), F.linear(x, w).view(torch.int16)), (ring, name, scale)
            replays += 1
print("graph replays exact", replays, flush=True)


def graph_us(fns, reps=24, rounds=7):
    for fn in fns:
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(reps):
            fns[i % len(fns)]()
    graph.replay()
    torch.cuda.synchronize()
    spans = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        spans.append(start.elapsed_time(end))
    return sorted(spans)[rounds // 2] * 1000.0 / reps


results = []
x = torch.randn(5, 4096, device="cuda", dtype=torch.bfloat16, generator=gen)
for shape in sorted(apk12.SHAPES):
    selected = [item for item in items if item[3:] == shape]
    n, k = shape
    inputs = x[:, :k].contiguous()
    times = {}
    for ring in rings:
        calls = [lambda cp=item[2]: apk12.gemm(inputs, cp, n, k, ring=ring) for item in selected]
        times[ring] = graph_us(calls)
    for ring in rings:
        row = {"n": n, "k": k, "ring": ring, "us": times[ring],
               "gain_vs_ring4": times[4] / times[ring] - 1.0}
        results.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
with open(os.environ["APK12_RING_RESULTS"], "w") as f:
    json.dump(results, f, indent=2)
    f.write("\n")

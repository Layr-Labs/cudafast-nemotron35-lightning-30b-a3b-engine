# SPDX-License-Identifier: Apache-2.0
"""fp8c bit-identity against fp8small and CUTLASS on the real Mamba weights, plus timings.

Runs in the pinned image (bin/atom nemo exec SHA harness/kernels/fp8c/tests/fp8c_exact.py)
with the staged overlay and kernels on PYTHONPATH, the checkpoint at $NEMO_SNAPSHOT and
the JSON report written to $NEMO_OUT/fp8c_exact.json. Isolated kernel timings only.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time

import torch
from safetensors import safe_open

import fp8c
import fp8small
import l2pf
from vllm import _custom_ops as ops

SNAP = os.environ["NEMO_SNAPSHOT"]
OUT = os.environ.get("NEMO_OUT", "/tmp")
ROWS = (1, 4, 5, 8)
torch.manual_seed(20261025)


def log(*a):
    print(*a, flush=True)


with open(os.path.join(SNAP, "model.safetensors.index.json")) as f:
    wmap = json.load(f)["weight_map"]
layers = sorted(int(k.split(".")[2]) for k in wmap if k.endswith(".mixer.in_proj.weight"))
assert len(layers) == 23, layers


def load(names):
    out = {}
    by_file = {}
    for nm in names:
        by_file.setdefault(wmap[nm], []).append(nm)
    for fn, nms in by_file.items():
        with safe_open(os.path.join(SNAP, fn), framework="pt") as f:
            for nm in nms:
                out[nm] = f.get_tensor(nm).to("cuda")
    return out


items = {"in_proj": [], "out_proj": []}
infos = []
prep_s = 0.0
for li in layers:
    for proj in ("in_proj", "out_proj"):
        p = f"backbone.layers.{li}.mixer.{proj}"
        t = load([p + ".weight", p + ".weight_scale", p + ".input_scale"])
        w = t[p + ".weight"].contiguous()
        sb = t[p + ".weight_scale"].float().max().reshape(1)
        sa = t[p + ".input_scale"].float().max().reshape(1)
        n, k = w.shape
        packed = fp8small.prepare(w.T)
        assert packed is not None, (li, proj)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        got = fp8c.prepare(packed, return_info=True)
        torch.cuda.synchronize()
        prep_s += time.perf_counter() - t0
        if got is None:
            enc = fp8c.encode(packed)
            log("REJECTED", li, proj, fp8c.stats["last_reason"], enc[1] if enc else None)
            if enc is not None:
                dec = fp8c.decode(enc[0], n, k)
                diff = (dec != packed).nonzero()
                log("first differing [n, k] of the pack:", diff[:16].tolist(),
                    "got", dec[diff[:16, 0], diff[:16, 1]].tolist(),
                    "want", packed[diff[:16, 0], diff[:16, 1]].tolist())
            sys.exit(2)
        buf, info = got
        info.update(layer=li, proj=proj)
        infos.append(info)
        items[proj].append(dict(li=li, w=w, sa=sa, sb=sb, packed=packed, buf=buf, n=n, k=k))
    log(f"layer {li}: in {infos[-2]['bits_per_weight']:.4f} b/w esc {infos[-2]['escape_rate']:.4f} "
        f"maxwc {infos[-2]['max_warp_chunk']} | out {infos[-1]['bits_per_weight']:.4f} b/w "
        f"esc {infos[-1]['escape_rate']:.4f} maxwc {infos[-1]['max_warp_chunk']}")
raw = sum(i["bytes"] / i["ratio"] for i in infos)
enc = sum(i["bytes"] for i in infos)
log(f"all 46 tensors: {raw / 1e6:.1f} MB -> {enc / 1e6:.1f} MB = {8 * enc / raw:.4f} bits/weight; "
    f"prepare (encode + full decode check) {prep_s:.2f} s total")


def acts(m, k, kind):
    if kind == "all_bits":
        return (torch.arange(m * k, device="cuda", dtype=torch.int64) * 7 + m)\
            .to(torch.uint8).view(torch.float8_e4m3fn).reshape(m, k)
    scale = {"random": 20.0, "small": 0.02, "extreme": 150.0}[kind]
    return (torch.randn((m, k), device="cuda") * scale).clamp(-448, 448).to(torch.float8_e4m3fn)


def eq(x, y):
    return int((x.view(torch.int16) != y.view(torch.int16)).sum().item())


# Exactness: every layer, both projections, M = 1, 4, 5, 8, four activation sets, default
# configuration against fp8small and CUTLASS; ring and PDL variants against fp8small.
cases = 0
bad_cases = []
for proj, lst in items.items():
    for it in lst:
        for m in ROWS:
            for kind in ("random", "small", "extreme", "all_bits"):
                a = acts(m, it["k"], kind)
                ref = fp8small.gemm(a, it["packed"], it["sa"], it["sb"])
                cut = ops.cutlass_scaled_mm(a, it["w"].T, scale_a=it["sa"], scale_b=it["sb"],
                                            out_dtype=torch.bfloat16)
                got = fp8c.gemm(a, it["buf"], it["sa"], it["sb"], it["n"], it["k"])
                d1, d2 = eq(got, ref), eq(got, cut)
                cases += 1
                if d1 or d2:
                    bad_cases.append((proj, it["li"], m, kind, "default", d1, d2))
                if kind == "random":
                    for ring in (3, 4, 6):
                        for pdl in (False, True):
                            g2 = fp8c.gemm(a, it["buf"], it["sa"], it["sb"], it["n"], it["k"],
                                           ring=ring, pdl=pdl)
                            d = eq(g2, ref)
                            cases += 1
                            if d:
                                bad_cases.append((proj, it["li"], m, kind, f"ring{ring}-pdl{int(pdl)}", d, -1))
    log(f"{proj}: exactness cases so far {cases}, mismatching {len(bad_cases)}")
log(f"EXACT: {cases} cases, {len(bad_cases)} mismatching: {bad_cases[:10]}")

# CUDA graph replay with changed inputs (the engine replays captured graphs).
it = items["in_proj"][0]
x = acts(5, it["k"], "random")
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        fp8c.gemm(x, it["buf"], it["sa"], it["sb"], it["n"], it["k"])
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    gout = fp8c.gemm(x, it["buf"], it["sa"], it["sb"], it["n"], it["k"])
graph_ok = True
for _ in range(3):
    x.copy_(acts(5, it["k"], "random"))
    g.replay()
    graph_ok &= eq(gout, fp8small.gemm(x, it["packed"], it["sa"], it["sb"])) == 0
torch.cuda.synchronize()
del g
log(f"graph replay with changed input exact: {graph_ok}")


def graph_us(fns, reps=24, rounds=7):
    for fn in fns:
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for fn in fns:
            fn()
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for i in range(reps):
            fns[i % len(fns)]()
    gr.replay()
    torch.cuda.synchronize()
    ms = []
    for _ in range(rounds):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        gr.replay()
        e1.record()
        e1.synchronize()
        ms.append(e0.elapsed_time(e1))
    del gr
    return statistics.median(ms) * 1000.0 / reps


# Timings: cold = a graph cycling through 12 layers (each call's weight was pushed out
# of L2 by the 11 others); head = the same, each call preceded on the stream by the
# engine's l2pf prefetch of the head of its weight buffer (12 MB in_proj, 5 MB out_proj;
# the same prefetch time in both columns); warm = the same layer 24 times.
HEAD = {"in_proj": 12 << 20, "out_proj": 5 << 20}


def head_pf(t, nbytes):
    l2pf._lib.l2pf_prefetch(t.data_ptr(), min(nbytes, t.numel()), l2pf.CHUNK,
                            torch.cuda.current_stream().cuda_stream)


measure = []
for proj, lst in items.items():
    sel = lst[:12]
    hb = HEAD[proj]
    for m in ROWS:
        xs = [acts(m, it["k"], "random") for it in sel]

        def run_small(it, x, head):
            if head:
                head_pf(it["packed"], hb)
            fp8small.gemm(x, it["packed"], it["sa"], it["sb"])

        def run_c(it, x, head, ring):
            if head:
                head_pf(it["buf"], hb)
            fp8c.gemm(x, it["buf"], it["sa"], it["sb"], it["n"], it["k"], ring=ring)

        def mk_small(mode):
            if mode != "warm":
                return [(lambda it=it, x=x: run_small(it, x, mode == "head")) for it, x in zip(sel, xs)]
            return [lambda: run_small(sel[0], xs[0], False)]

        def mk_c(mode, ring):
            if mode != "warm":
                return [(lambda it=it, x=x: run_c(it, x, mode == "head", ring)) for it, x in zip(sel, xs)]
            return [lambda: run_c(sel[0], xs[0], False, ring)]

        for cold in ("cold", "head", "warm"):
            names = ["fp8small", "fp8c-r3", "fp8c-r4", "fp8c-r6"]
            res = {nm: [] for nm in names}
            for rep in range(3):
                order = names[rep % 4:] + names[: rep % 4]
                for nm in order:
                    fns = mk_small(cold) if nm == "fp8small" else mk_c(cold, int(nm[-1]))
                    res[nm].append(graph_us(fns))
            row = dict(proj=proj, m=m, cache=cold)
            for nm in names:
                row[nm] = statistics.median(res[nm])
            base = row["fp8small"]
            row.update({f"{nm}_gain": base / row[nm] - 1.0 for nm in names[1:]})
            measure.append(row)
            log(json.dumps({k: (round(v, 2) if isinstance(v, float) and not k.endswith("gain") else
                                (round(v, 4) if isinstance(v, float) else v)) for k, v in row.items()}))

report = dict(
    passed=not bad_cases and graph_ok, cases=cases, bad_cases=bad_cases, graph_ok=graph_ok,
    prepare_seconds=prep_s, raw_bytes=raw, encoded_bytes=enc, bits_per_weight=8 * enc / raw,
    tensors=infos, measurements=measure,
)
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "fp8c_exact.json"), "w") as f:
    json.dump(report, f, indent=1)
log("table (us per call; gain = fp8small/fp8c - 1)")
for r in measure:
    log(f"{r['proj']:8s} M={r['m']} {r['cache']}: fp8small {r['fp8small']:.1f}  r3 {r['fp8c-r3']:.1f} "
        f"({100 * r['fp8c-r3_gain']:+.1f}%)  r4 {r['fp8c-r4']:.1f} ({100 * r['fp8c-r4_gain']:+.1f}%)  "
        f"r6 {r['fp8c-r6']:.1f} ({100 * r['fp8c-r6_gain']:+.1f}%)")
if not report["passed"]:
    sys.exit(3)

"""routerbig vs torch.mm (cuBLAS) on the checkpoint's 23 router weights: bit identity and timing.

Run on Atom: bin/atom nemo exec <sha> harness/kernels/routerbig/tests/exact.py [--ms 496,...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from safetensors import safe_open

OUT = os.environ.get("NEMO_OUT", "/tmp")


def gate_weights():
    snap = os.environ["NEMO_SNAPSHOT"]
    index = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    names = sorted((k for k in index if k.endswith("mixer.gate.weight") and not k.startswith("mtp")),
                   key=lambda k: int(k.split(".")[2]))
    ws = []
    for name in names:
        with safe_open(os.path.join(snap, index[name]), framework="pt", device="cuda") as f:
            ws.append(f.get_tensor(name).float().contiguous())
    return names, ws


def inputs(kind, m, k, gen):
    if kind == "rmsnorm_bf16":
        x = torch.randn(m, k, generator=gen, device="cuda")
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
        x = x * (1 + 0.5 * torch.randn(k, generator=gen, device="cuda"))
        return x.to(torch.bfloat16).float()
    if kind == "normal":
        return torch.randn(m, k, generator=gen, device="cuda")
    if kind == "heavy":
        x = torch.randn(m, k, generator=gen, device="cuda")
        return x * torch.exp(2 * torch.randn(m, k, generator=gen, device="cuda"))
    if kind == "wide":
        x = torch.randn(m, k, generator=gen, device="cuda")
        return (x * torch.exp2(torch.randint(-40, 40, (m, k), generator=gen, device="cuda").float())).to(torch.bfloat16).float()
    if kind == "cancel":
        h = torch.randn(m, k // 2, generator=gen, device="cuda") * 8
        x = torch.cat([h, -h], 1)
        x[:, -5:] += torch.randn(m, 5, generator=gen, device="cuda") * 1e-3
        return x
    raise ValueError(kind)


def graph_us(fn, reps=23, iters=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(0)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(reps):
            fn(i)
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / reps)
    ts.sort()
    return ts[len(ts) // 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="496,480,512")
    ap.add_argument("--kinds", default="rmsnorm_bf16,normal,heavy,wide,cancel")
    args = ap.parse_args()
    import routerbig

    if not routerbig.available():
        print("routerbig unavailable:", routerbig.load_error())
        return 2
    names, ws = gate_weights()
    print("gates", len(ws), tuple(ws[0].shape), flush=True)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(7)
    rep = {"cases": 0, "bad": {}, "per_m": {}}
    for bm in (32, 128, 64):  # 64 (the default) last
        routerbig.set_bm(bm)
        ok = 0
        for kind in args.kinds.split(","):
            for w in ws:
                x = inputs(kind, 496, w.shape[1], gen).contiguous()
                ok += int(torch.equal(routerbig.router_gemm(x, w).view(torch.int32), torch.mm(x, w.t()).view(torch.int32)))
        xs_t = [inputs("rmsnorm_bf16", 496, 2688, gen).contiguous() for _ in range(23)]
        rep[f"bm{bm}"] = {"equal": ok, "of": 23 * len(args.kinds.split(",")),
                          "us": round(graph_us(lambda i: routerbig.router_gemm(xs_t[i], ws[i])), 2),
                          "partials_us": round(graph_us(lambda i: routerbig.partials_only(xs_t[i], ws[i])), 2)}
        print("bm", bm, rep[f"bm{bm}"], flush=True)
    for m in [int(v) for v in args.ms.split(",")]:
        ok = bad = 0
        for kind in args.kinds.split(","):
            for li, w in enumerate(ws):
                x = inputs(kind, m, w.shape[1], gen).contiguous()
                ref = torch.mm(x, w.t())
                y = routerbig.router_gemm(x, w)
                rep["cases"] += 1
                if torch.equal(y.view(torch.int32), ref.view(torch.int32)):
                    ok += 1
                else:
                    bad += 1
                    n = int((y.view(torch.int32) != ref.view(torch.int32)).sum())
                    if len(rep["bad"]) < 30:
                        rep["bad"][f"M{m}/{kind}/L{li}"] = n
        rep["per_m"][m] = {"equal": ok, "differ": bad}
        print("M", m, rep["per_m"][m], flush=True)
    # timing at M = 496, 23 different gates per graph (as in one prefill)
    m = 496
    xs = [inputs("rmsnorm_bf16", m, 2688, gen).contiguous() for _ in range(23)]
    rep["stock_us"] = round(graph_us(lambda i: torch.mm(xs[i], ws[i].t())), 2)
    rep["routerbig_us"] = round(graph_us(lambda i: routerbig.router_gemm(xs[i], ws[i])), 2)
    rep["routerbig_partials_us"] = round(graph_us(lambda i: routerbig.partials_only(xs[i], ws[i])), 2)
    print(json.dumps({k: v for k, v in rep.items() if k != "bad"}, indent=1))
    print("bad:", json.dumps(rep["bad"], indent=1))
    json.dump(rep, open(os.path.join(OUT, "routerbig-exact.json"), "w"), indent=1, default=str)
    return 0 if rep["per_m"].get(496, {}).get("differ", 1) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

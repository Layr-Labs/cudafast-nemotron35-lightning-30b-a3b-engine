#!/usr/bin/env python3
"""shco step 2: shx (Marlin-exact, small-footprint shared expert at M = 496) vs stock Marlin.

Runs in the pinned image: bin/atom nemo exec SHA harness/kernels/shco/tests/step2.py [--phase exact,time]

exact  every layer's shared expert on real prefill inputs (captured MoE inputs of --prompts) and
       synthetic ones: shx up (plain and relu^2-fused) vs apply_fp4_marlin_linear (+ the torch
       relu^2 pair), shx down vs Marlin down, bitwise; mismatches are attributed to unit kinds.
time   CUDA graphs over the 23 layers: stock shared expert, shx, humming's routed up (engine
       config and the 3-CTA variants) and their co-runs on two streams; per-kernel profiles.
Writes step2.json (+ traces) to $NEMO_OUT.
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import step1 as S1  # noqa: E402
import shco  # noqa: E402

from step1 import HIDDEN, M, MOE_LAYERS, SH_INTER, dump, log  # noqa: E402


def stock_up(e, x):
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import apply_fp4_marlin_linear

    return apply_fp4_marlin_linear(x, e.up.weight, e.up.weight_scale, e.up.weight_global_scale, e.up.workspace,
                                   SH_INTER, HIDDEN)


def stock_down(e, a):
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import apply_fp4_marlin_linear

    return apply_fp4_marlin_linear(a, e.dn.weight, e.dn.weight_scale, e.dn.weight_global_scale, e.dn.workspace,
                                   HIDDEN, SH_INTER)


def relu2(h):
    from vllm.model_executor.layers.fused_moe.experts.humming_tuned_moe import relu2_bf16

    a = torch.empty_like(h)
    assert relu2_bf16(h, a)
    return a


def ndiff(a, b):
    return int((a.view(torch.int16) != b.view(torch.int16)).sum())


def unit_of(units, rows, feats):
    """unit kinds (nseg, tail) holding the mismatching (row, feature) pairs"""
    kinds = {}
    for r, f in zip(rows.tolist(), feats.tolist()):
        for row0, nrows, f0, segs in units:
            if row0 <= r < row0 + nrows and f0 <= f < f0 + 64:
                key = f"nseg{len(segs)}{'_tail' if row0 >= 448 else ''}"
                kinds[key] = kinds.get(key, 0) + 1
                break
    return kinds


def diff_info(a, b, units):
    bad = a.view(torch.int16) != b.view(torch.int16)
    idx = bad.nonzero()
    info = {"elements": int(bad.sum()), "rows": int(bad.any(1).sum()), "features": int(bad.any(0).sum())}
    if idx.numel():
        sel = idx[:4096]
        info["kinds"] = unit_of(units, sel[:, 0], sel[:, 1])
        info["first"] = [[int(r), int(f), float(a[r, f]), float(b[r, f])] for r, f in idx[:6].tolist()]
    return info


def phase_exact(args, layers, xs, shx):
    uf = 128 if shx.version == 2 else 64
    units_up = shco.marlin_units(M, SH_INTER, HIDDEN, 48, uf)
    units_dn = shco.marlin_units(M, HIDDEN, SH_INTER, 48, uf)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(1234)
    tot = {"up": 0, "up_relu2": 0, "down": 0, "chain": 0, "cases": 0, "elements_up": 0, "elements_dn": 0}
    worst = []
    for li, e in enumerate(layers):
        cases = [(f"real:{name}", xs[name][e.L].to("cuda").contiguous()) for name in xs]
        x0 = cases[0][1]
        cases += [
            ("scaled_1e-4", (x0.float() * 1e-4).to(torch.bfloat16)),
            ("scaled_64", (x0.float() * 64).to(torch.bfloat16)),
            ("randn", torch.randn(M, HIDDEN, generator=gen, device="cuda").to(torch.bfloat16)),
            ("randn_x8_zero_rows", (torch.randn(M, HIDDEN, generator=gen, device="cuda") * 8).to(torch.bfloat16)),
        ]
        cases[-1][1][::7] = 0
        for name, x in cases:
            h_ref = stock_up(e, x)
            a_ref = relu2(h_ref)
            o_ref = stock_down(e, a_ref)
            h_new = torch.empty_like(h_ref)
            a_new = torch.empty_like(h_ref)
            o_new = torch.empty_like(o_ref)
            o_chain = torch.empty_like(o_ref)
            h_new.view(torch.int16).fill_(0x7FBF)
            a_new.view(torch.int16).fill_(0x7FBF)
            o_new.view(torch.int16).fill_(0x7FBF)
            shx.up(x, e.up, h_new, relu2=False)
            shx.up(x, e.up, a_new, relu2=True)
            shx.down(a_ref, e.dn, o_new)
            shx.down(a_new, e.dn, o_chain)
            torch.cuda.synchronize()
            d = {"up": ndiff(h_new, h_ref), "up_relu2": ndiff(a_new, a_ref), "down": ndiff(o_new, o_ref),
                 "chain": ndiff(o_chain, o_ref)}
            for k, v in d.items():
                tot[k] += v
            tot["cases"] += 1
            tot["elements_up"] += h_ref.numel()
            tot["elements_dn"] += o_ref.numel()
            if any(d.values()):
                rec = {"layer": e.L, "case": name, **d}
                if d["up"]:
                    rec["up_info"] = diff_info(h_new, h_ref, units_up)
                if d["down"]:
                    rec["down_info"] = diff_info(o_new, o_ref, units_dn)
                worst.append(rec)
                if len(worst) <= 12:
                    log(f"  MISMATCH layer {e.L} {name}: {rec}")
        if li % 6 == 0:
            log(f"exact: layer {e.L} done; totals {tot}")
    ok = tot["up"] == tot["up_relu2"] == tot["down"] == tot["chain"] == 0
    log(f"EXACT: {'ALL BIT-IDENTICAL' if ok else 'MISMATCH'} {tot}")
    return {"ok": ok, "totals": tot, "mismatches": worst[:64]}


def phase_time(args, layers, shxs):
    hvars = {}
    for h, kw in (("main", {}), ("s36", {"sms": 36})):
        hvars[h], cfg = S1.tuning_string(layers[0].ex, **kw)
    for e in layers:
        e.sh_h = torch.empty(M, SH_INTER, dtype=torch.bfloat16, device="cuda")
        e.sh_o = torch.empty(M, HIDDEN, dtype=torch.bfloat16, device="cuda")

    def shx_call(e, shx):
        shx.up(e.x, e.up, e.sh_h, relu2=True)
        shx.down(e.sh_h, e.dn, e.sh_o)

    for e in layers[:1]:
        for t in hvars.values():
            S1.up_call(e, t)
        S1.sh_call(e)
        for shx in shxs.values():
            shx_call(e, shx)
    torch.cuda.synchronize()
    specs = {"sh": (S1.sh_call, None, True)}
    for h, t in hvars.items():
        specs[f"up:{h}"] = (lambda e, t=t: S1.up_call(e, t), None, True)
    for vn, shx in shxs.items():
        specs[f"shx{vn}"] = (lambda e, shx=shx: shx_call(e, shx), None, True)
        for af in (True, False):
            specs[f"up:s36||shx{vn}:{'auxfirst' if af else 'mainfirst'}"] = (
                lambda e, t=hvars["s36"]: S1.up_call(e, t), lambda e, shx=shx: shx_call(e, shx), af)
    graphs = {k: S1.make_graph(layers, *v) for k, v in specs.items()}
    torch.cuda.synchronize()
    times = {k: [] for k in graphs}
    for r in range(args.rounds):
        for k, g in graphs.items():
            g.replay()
            times[k] += S1.replay_ms(g, args.reps)
    med = {k: statistics.median(v) for k, v in times.items()}
    for k in graphs:
        log(f"  {k:32s} {med[k]:8.3f} ms  [{min(times[k]):.3f}, {max(times[k]):.3f}]")
    base = med["up:main"] + med["sh"]
    rows = []
    for k, v in med.items():
        if "||shx" in k:
            rows.append({"graph": k, "conc_ms": v, "saving_vs_serial_stock_ms": base - v,
                         "conc_minus_up_main_ms": v - med["up:main"]})
    rows.sort(key=lambda r: -r["saving_vs_serial_stock_ms"])
    log(f"serial stock up+shared {base:.3f} ms; stock shared {med['sh']:.3f}; " + ", ".join(
        f"shx{vn} alone {med['shx' + vn]:.3f}" for vn in shxs))
    for r in rows:
        log(f"  {r['graph']:32s} conc {r['conc_ms']:8.3f}  saving vs serial stock {r['saving_vs_serial_stock_ms']:7.3f}  "
            f"conc - up(main) {r['conc_minus_up_main_ms']:6.3f}")
    from torch.profiler import ProfilerActivity, profile

    prof = {}
    for k in ["sh"] + [f"shx{vn}" for vn in shxs] + [r["graph"] for r in rows[:3]]:
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            graphs[k].replay()
            torch.cuda.synchronize()
        path = os.path.join(S1.OUT, "prof2_" + k.replace(":", "_").replace("|", "I") + ".json")
        p.export_chrome_trace(path)
        evs = S1.kernel_events(path)
        by = {}
        for ev in evs:
            n = ev["name"]
            kind = "shx" if "shx::" in n else S1.kind_of(n)
            by.setdefault(kind, []).append(ev["dur"])
        span = (max(ev["ts"] + ev["dur"] for ev in evs) - min(ev["ts"] for ev in evs)) if evs else 0
        prof[k] = {"span_us": span, **{kk: {"n": len(v), "sum_us": sum(v), "median_us": statistics.median(v)}
                                      for kk, v in by.items()}}
        log(f"  prof {k}: span {span / 1e3:.3f} ms; " + "; ".join(
            f"{kk} n={len(v)} sum={sum(v) / 1e3:.3f} ms med={statistics.median(v):.1f} us" for kk, v in by.items()))
    res = []
    for vn in shxs:
        res += S1.resources(S1.kernel_events(os.path.join(S1.OUT, f"prof2_shx{vn}.json")))
    for r in res:
        log(f"  res {r['name'][:60]} grid {r['grid']} block {r['block']} regs {r['regs']} smem {r['smem']} "
            f"blocks/SM {r['blocks_per_sm']}")
    return {"graph_ms": med, "concurrency": rows, "profiles": prof, "resources": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="exact,time")
    ap.add_argument("--prompts", default="public,copy-go-ratelimit,gen-rust-exprparser")
    ap.add_argument("--variants", default="v1,v2h0,v2h1")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=os.environ.get("NEMO_OUT", "."))
    args = ap.parse_args()
    S1.OUT = args.out
    os.makedirs(args.out, exist_ok=True)
    os.environ.setdefault("VLLM_HUMMING_BATCH_INVARIANT", "1")
    log(f"args: {vars(args)}")
    assert shco.available(), shco.load_error()
    torch.cuda.set_device(0)
    res = {"args": vars(args), "attrs": {r: shco.shx_attrs(r) for r in (False, True)},
           "attrs2": {r: shco.shx2_attrs(r) for r in (False, True)}}
    log(f"shx attrs: v1 {res['attrs']} v2 {res['attrs2']}")
    phases = args.phase.split(",")
    prompts = args.prompts.split(",")
    xs = S1.capture(prompts)
    layers = S1.build_layers(xs[prompts[0]], MOE_LAYERS)
    gc.collect()
    shxs = {}
    for vn in args.variants.split(","):
        ver, hint = {"v1": (1, 0), "v2h0": (2, 0), "v2h1": (2, 1)}[vn]
        shxs[vn] = shco.Shx(M, "cuda", version=ver, hint=hint)
    S1.S_CAP = torch.cuda.Stream()
    S1.S_AUX = torch.cuda.Stream()
    ok = True
    if "exact" in phases:
        res["exact"] = {}
        for vn, shx in shxs.items():
            log(f"exactness of shx {vn}")
            res["exact"][vn] = phase_exact(args, layers, xs, shx)
            ok &= res["exact"][vn]["ok"]
        dump("step2", res)
    if "time" in phases:
        res["time"] = phase_time(args, layers, shxs)
        dump("step2", res)
    log("done")
    if not ok:
        sys.exit(3)


if __name__ == "__main__":
    main()

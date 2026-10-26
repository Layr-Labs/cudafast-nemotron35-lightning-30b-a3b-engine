#!/usr/bin/env python3
"""Bit-identity of the MARLINX_LEAN shared expert kernels against marlinx and stock Marlin.

Run in the pinned image (overlay + kernels on PYTHONPATH, snapshot at $NEMO_SNAPSHOT):
  python3 harness/kernels/marlinx/tests/marlinx_lean_exact.py [REAL.pt]
REAL.pt (optional): torch.save of {"up": {layer: [S, T, 2688] bf16}, "down": {layer: [S, T, 3712]}},
the shared expert up and down inputs of real decode windows (consecutive rows = verify rows).

For all 23 shared experts, M = 1..8, PDL and ordinary launch:
  * up (relu2-fused): lean == marlinx relu2 kernel; its pre-activation output == stock Marlin
    on the dense path (checked through the dense down below);
  * down, zero-row skip on (lean reads the Z2 copy) and off (dense): lean == marlinx ZS ==
    marlinx dense == stock Marlin, on ReLU^2 of the real/stock up output and adversarial
    inputs: 90% extra zeros, half of the zeros -0, all +0, all -0, single nonzero column,
    heavy-tailed values, Inf/NaN-free random signs;
  * z2_repack is a nibble permutation of zs_repack (inverse checked);
  * CUDA graph replay with in-place input changes.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import marlinx  # noqa: E402
from marlinx_exact import SHARED_LAYERS, gen_x, load_linear, relu2, stock  # noqa: E402

snap = os.environ["NEMO_SNAPSHOT"]
with open(os.path.join(snap, "model.safetensors.index.json")) as f:
    wmap = json.load(f)["weight_map"]
assert marlinx.available() and hasattr(marlinx._lib, "marlinx_gemm_lean")
assert marlinx._init()
assert marlinx._lib.marlinx_init_lean() == 0
real = torch.load(sys.argv[1]) if len(sys.argv) > 1 else None
gen = torch.Generator(device="cuda")
gen.manual_seed(23)
cases = 0
WINDOWS = int(os.environ.get("LEAN_WINDOWS", "4"))


def eq(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def run(fn, pdl):
    saved = marlinx._pdl_active
    marlinx._pdl_active = pdl and saved
    try:
        return fn()
    finally:
        marlinx._pdl_active = saved


def z2_inverse(z2):
    z = z2.to(torch.int64) & 0xFFFFFFFF
    out = torch.zeros_like(z)
    for half in (0, 1):
        rows = slice(0, 8) if half == 0 else slice(8, 16)
        acc = torch.zeros_like(z[:, rows])
        for i, j in enumerate(marlinx._Z2_SRC[half]):
            acc |= ((z[:, rows] >> (4 * i)) & 15) << (4 * j)
        out[:, rows] = acc
    out = out - (out >= (1 << 31)).to(torch.int64) * (1 << 32)
    return out.to(torch.int32)


def adversarial(a, m):
    one = torch.zeros_like(a)
    one[:, 1234] = a.abs().max().clamp(min=1.0)
    return [
        torch.where(torch.rand(m, 3712, device="cuda", generator=gen) < 0.9, torch.zeros_like(a), a),
        torch.where(torch.rand(m, 3712, device="cuda", generator=gen) < 0.5, torch.full_like(a, -0.0),
                    torch.where(a == 0, torch.zeros_like(a), a)),
        torch.where(a == 0, torch.full_like(a, -0.0), a),
        torch.zeros_like(a),
        torch.full_like(a, -0.0),
        one,
        (gen_x(m, 3712, 1, gen).float().abs() ** 2).to(torch.bfloat16),
        gen_x(m, 3712, 0, gen),  # signed, dense
    ]


for li in SHARED_LAYERS:
    p = f"backbone.layers.{li}.mixer.shared_experts"
    up = load_linear(snap, wmap, p + ".up_proj")
    dn = load_linear(snap, wmap, p + ".down_proj")
    assert marlinx.prepare(up) and marlinx.prepare(dn)
    zs = marlinx.zs_repack(dn.weight, 2688, 3712)
    z2 = marlinx.z2_repack(dn.weight, 2688, 3712)
    assert torch.equal(z2_inverse(z2), zs), (li, "z2 inverse")
    W = (up.weight, up.weight_scale, up.weight_global_scale, 3712, 2688)
    D = (dn.weight, dn.weight_scale, dn.weight_global_scale, 2688, 3712)
    for m in range(1, 9):
        xs = [gen_x(m, 2688, kind, gen) for kind in (0, 1, 2)]
        acts = []
        if real is not None:
            ru, rd = real["up"][li], real["down"][li]
            S, T = ru.shape[0], ru.shape[1]
            for wi in range(WINDOWS):
                s, t = wi % S, (wi * 37 + m) % (T - m + 1)
                xs.append(ru[s, t:t + m].cuda().contiguous())
                acts.append(rd[s, t:t + m].cuda().contiguous())
        for x in xs:
            ref_up = marlinx.gemm(x, *W, relu2=True)
            h = stock(up, x)
            assert eq(relu2(h), ref_up), (li, m, "marlinx relu2 vs stock+relu2")
            for pdl in (True, False):
                got = run(lambda: marlinx.gemm(x, *W, relu2=True, lean=True), pdl)
                torch.cuda.synchronize()
                assert eq(got, ref_up), (li, m, pdl, "lean up")
                cases += 1
            acts.append(ref_up)
        for a in list(acts):
            acts.extend(adversarial(a, m)[:3])
        acts.extend(adversarial(acts[0], m)[3:])
        for a in acts:
            ref = stock(dn, a)
            mzs = marlinx.gemm(a, *D, zs=zs)
            mdn = marlinx.gemm(a, *D)
            assert eq(ref, mzs) and eq(ref, mdn), (li, m, "marlinx vs stock")
            for pdl in (True, False):
                g_zs = run(lambda: marlinx.gemm(a, *D, zs=z2, lean=True), pdl)
                g_dn = run(lambda: marlinx.gemm(a, *D, lean=True), pdl)
                torch.cuda.synchronize()
                assert eq(g_zs, ref), (li, m, pdl, "lean down zs")
                assert eq(g_dn, ref), (li, m, pdl, "lean down dense")
                cases += 2
    del up, dn, zs, z2
    print(f"layer {li}: {cases} cases exact", flush=True)

# CUDA graph replay, in-place inputs (lean up -> lean ZS down)
up = load_linear(snap, wmap, "backbone.layers.13.mixer.shared_experts.up_proj")
dn = load_linear(snap, wmap, "backbone.layers.13.mixer.shared_experts.down_proj")
z2 = marlinx.z2_repack(dn.weight, 2688, 3712)
W = (up.weight, up.weight_scale, up.weight_global_scale, 3712, 2688)
D = (dn.weight, dn.weight_scale, dn.weight_global_scale, 2688, 3712)
x = torch.zeros(5, 2688, dtype=torch.bfloat16, device="cuda")


def body():
    hh = marlinx.gemm(x, *W, relu2=True, lean=True)
    return hh, marlinx.gemm(hh, *D, zs=z2, lean=True)


body()
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    hg, og = body()
for kind in (0, 1, 2, 0):
    x.copy_(gen_x(5, 2688, kind, gen))
    g.replay()
    torch.cuda.synchronize()
    href = relu2(stock(up, x))
    assert eq(hg, href) and eq(og, stock(dn, href)), (kind, "graph replay")
    cases += 1
print(f"MARLINX_LEAN EXACT: {cases} cases", flush=True)

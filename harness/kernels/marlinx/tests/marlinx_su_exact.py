#!/usr/bin/env python3
"""Bit-identity of marlinx's unit-major scale path (MARLINX_SU) against stock Marlin.

Run in the pinned image (overlay + kernels on PYTHONPATH, snapshot at $NEMO_SNAPSHOT):
  python3 harness/kernels/marlinx/tests/marlinx_su_exact.py
For lm_head and all 23 shared expert down projections (with and without the zero-row
skip copy), M = 1..8, dense/relu2-sparse/edge activations, PDL and ordinary launch:
output bits equal ops.marlin_gemm and marlinx without the scale copy. su_repack must be
a byte permutation (checked by its inverse). Also times lm_head cold (single call).
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
assert marlinx.available() and hasattr(marlinx._lib, "marlinx_gemm_su")
gen = torch.Generator(device="cuda")
gen.manual_seed(11)
cases = 0


def check(layer, name, xs, zs):
    global cases
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    su = marlinx.su_repack(layer.weight_scale, n, k)
    inv = su.view(n // 64, k // 16, 64).transpose(0, 1).reshape(k // 16, n)
    assert torch.equal(inv, layer.weight_scale.view(torch.uint8)), (name, "su inverse")
    for x in xs:
        ref = stock(layer, x)
        base = marlinx.gemm(x, layer.weight, layer.weight_scale, layer.weight_global_scale, n, k, zs=zs)
        assert torch.equal(ref.view(torch.int16), base.view(torch.int16)), (name, x.shape[0], "marlinx base")
        for pdl in (True, False):
            saved = marlinx._pdl_active
            marlinx._pdl_active = pdl and saved
            got = marlinx.gemm(x, layer.weight, layer.weight_scale, layer.weight_global_scale, n, k, zs=zs, su=su)
            marlinx._pdl_active = saved
            torch.cuda.synchronize()
            assert torch.equal(ref.view(torch.int16), got.view(torch.int16)), (name, x.shape[0], pdl, "su")
            cases += 1


lm = load_linear(snap, wmap, "lm_head")
assert marlinx.prepare(lm)
xs = [gen_x(m, 2688, kind, gen) for m in range(1, 9) for kind in (0, 1, 2)]
check(lm, "lm_head", xs, None)
print("lm_head exact", cases, flush=True)
for li in SHARED_LAYERS:
    p = f"backbone.layers.{li}.mixer.shared_experts"
    up = load_linear(snap, wmap, p + ".up_proj")
    dn = load_linear(snap, wmap, p + ".down_proj")
    assert marlinx.prepare(up) and marlinx.prepare(dn)
    xs = []
    for m in range(1, 9):
        for kind in (0, 1, 2):
            xs.append(relu2(stock(up, gen_x(m, 2688, kind, gen))))
        xs.append(gen_x(m, 3712, 0, gen))
    # shared up, relu2-fused (opaque-op path): marlinx relu2 without vs with the scale copy
    su_up = marlinx.su_repack(up.weight_scale, 3712, 2688)
    for m in range(1, 9):
        for kind in (0, 1, 2):
            x = gen_x(m, 2688, kind, gen)
            r0 = marlinx.gemm(x, up.weight, up.weight_scale, up.weight_global_scale, 3712, 2688, relu2=True)
            for pdl in (True, False):
                saved = marlinx._pdl_active
                marlinx._pdl_active = pdl and saved
                r1 = marlinx.gemm(x, up.weight, up.weight_scale, up.weight_global_scale, 3712, 2688,
                                  relu2=True, su=su_up)
                marlinx._pdl_active = saved
                assert torch.equal(r0.view(torch.int16), r1.view(torch.int16)), (li, m, kind, pdl, "up su")
                cases += 1
    zs = marlinx.zs_repack(dn.weight, 2688, 3712)
    check(dn, f"down{li}", xs, None)
    check(dn, f"down{li}.zs", xs, zs)
    del up, dn
print(f"MARLINX_SU EXACT: {cases} cases", flush=True)

# cold single-call lm_head timing
n, k = lm.output_size_per_partition, lm.input_size_per_partition
su = marlinx.su_repack(lm.weight_scale, n, k)
flush = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
x = gen_x(5, 2688, 0, gen)


def t(fn, reps=12):
    ts = []
    for _ in range(reps):
        flush.add_(1)
        torch.cuda._sleep(20000)
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        ts.append((a, b))
    torch.cuda.synchronize()
    v = sorted(a.elapsed_time(b) * 1e3 for a, b in ts[2:])
    return v[len(v) // 2]


for _ in range(2):
    t0 = t(lambda: marlinx.gemm(x, lm.weight, lm.weight_scale, lm.weight_global_scale, n, k))
    t1 = t(lambda: marlinx.gemm(x, lm.weight, lm.weight_scale, lm.weight_global_scale, n, k, su=su))
    print(f"lm_head M=5 cold: marlinx {t0:.1f} us, marlinx+su {t1:.1f} us", flush=True)

"""marlinx_zs_exact.py SNAPSHOT [REAL_ACTS.pt] : zero-row-skip shared down vs stock Marlin and marlinx.

All 23 shared experts, M = 1..8: ReLU^2 of the stock up output, heavy-tailed
non-negative inputs, 90%-zero inputs and inputs with signed zeros; CUDA graph
replay with in-place inputs; optional real activations (a torch.save dict of
"model.layers.{i}.mixer.shared_experts.act_fn" -> [T, 3712] bf16) and timing.
Run with the staged overlay and kernels on PYTHONPATH."""
import json, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import marlinx
from marlinx_exact import load_linear, stock, relu2, gen_x, SHARED_LAYERS
snap = sys.argv[1]
wmap = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
pairs = []
for li in SHARED_LAYERS:
    pre = f"backbone.layers.{li}.mixer.shared_experts"
    up = load_linear(snap, wmap, pre + ".up_proj"); dn = load_linear(snap, wmap, pre + ".down_proj")
    assert marlinx.prepare(up) and marlinx.prepare(dn) and dn._marlinx_zs is not None
    pairs.append((up, dn))
print("loaded", len(pairs), "zs bytes/layer", pairs[0][1]._marlinx_zs.numel() * 4)
def mx(dn, a, zs):
    return marlinx.gemm(a, dn.weight, dn.weight_scale, dn.weight_global_scale, 2688, 3712, zs=dn._marlinx_zs if zs else None)
gen = torch.Generator(device="cuda"); gen.manual_seed(7)
bad = 0; n = 0; zf = []
for up, dn in pairs:
    for m in range(1, 9):
        for kind in (0, 1, 2):
            x = gen_x(m, 2688, kind, gen)
            a = relu2(stock(up, x))
            zf.append((a == 0).float().mean().item())
            ref = stock(dn, a)
            for inp in (a, (gen_x(m, 3712, kind, gen).float().abs() ** 2).to(torch.bfloat16),
                        torch.where(torch.rand(m, 3712, device="cuda", generator=gen) < 0.9, torch.zeros_like(a), a),
                        torch.where(torch.rand(m, 3712, device="cuda", generator=gen) < 0.3, torch.full_like(a, -0.0), a)):
                r = stock(dn, inp)
                bad += int(not torch.equal(r.view(torch.int16), mx(dn, inp, True).view(torch.int16)))
                bad += int(not torch.equal(r.view(torch.int16), mx(dn, inp, False).view(torch.int16)))
                n += 2
print(f"exact cases {n} mismatches {bad}; real relu2 zero frac (random x) {sum(zf)/len(zf):.3f}")
# graph replay with in-place input change
dn = pairs[5][1]; a = torch.zeros(4, 3712, dtype=torch.bfloat16, device="cuda")
g = torch.cuda.CUDAGraph(); out = mx(dn, a, True); torch.cuda.synchronize()
with torch.cuda.graph(g): out = mx(dn, a, True)
gb = 0
for kind in (0, 1, 2, 0):
    a.copy_(relu2(stock(pairs[5][0], gen_x(4, 2688, kind, gen)))); g.replay(); torch.cuda.synchronize()
    gb += int(not torch.equal(out.view(torch.int16), stock(dn, a).view(torch.int16)))
print("graph replay mismatches", gb)
# timing M=4 real relu2 inputs (from embedding-like random x), rotating 23 layers, with/without hog
if len(sys.argv) < 3:
    sys.exit(0)
real = torch.load(sys.argv[2])
def act_of(li, w):
    return real[f"model.layers.{li}.mixer.shared_experts.act_fn"][4 * w:4 * w + 4].cuda().contiguous()
acts = [act_of(li, 7) for li in SHARED_LAYERS]
# exactness on real activations, all windows of a few layers, M=1..4
rb = 0; rn = 0
for (up, dn), li in zip(pairs, SHARED_LAYERS):
    for w in range(0, 100, 7):
        for m in (1, 2, 3, 4):
            a = act_of(li, w)[:m].contiguous(); r = stock(dn, a)
            rb += int(not torch.equal(r.view(torch.int16), mx(dn, a, True).view(torch.int16))); rn += 1
print("real-activation exact cases", rn, "mismatches", rb)
src = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda"); sink = torch.zeros(4, dtype=torch.int32, device="cuda")
side = torch.cuda.Stream()
def timed(fn, hog):
    def body():
        for i, (up, dn) in enumerate(pairs):
            if hog:
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side): marlinx.hog(src[(i % 8) * 60_000_000:(i % 8) * 60_000_000 + 60_000_000], 96, 256, 0, sink)
            fn(dn, acts[i])
            if hog: torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph(); body(); torch.cuda.synchronize()
    with torch.cuda.graph(g): body()
    g.replay(); torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(10): g.replay()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / 10 / len(pairs) * 1e3
for wset in (7, 31, 77):
    acts = [act_of(li, wset) for li in SHARED_LAYERS]
    print("window", wset, "M=4 joint zero frac %.3f" % (sum(((a == 0).all(0)).float().mean().item() for a in acts) / len(acts)))
    hog = False
    print(f"hog {hog}: per layer-call us  stock {timed(lambda d, a: stock(d, a), hog):.1f} | marlinx {timed(lambda d, a: mx(d, a, False), hog):.1f} | zs {timed(lambda d, a: mx(d, a, True), hog):.1f}")

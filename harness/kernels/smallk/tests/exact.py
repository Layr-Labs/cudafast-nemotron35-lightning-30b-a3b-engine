"""smallk exactness and timing against the stock ops (real snapshot router weights and biases).

Run on Atom with `bin/atom nemo exec <sha> harness/kernels/smallk/tests/exact.py [parts]`.
Parts (default router,rtree,topk,sweep,align,timing). Writes $NEMO_OUT/exact-<part>.json.

- router: torch.equal(F.linear(x, W), smallk.router_gemm(x, W)) on the 23 target gate weights,
  and the same for the bf16-shadow read (smallk.gate_w16) on every gate that has one,
          M = 4, for normal, heavy-tailed, wide-exponent, bf16-rounded and cancellation-heavy x,
          every launch variant (rows per CTA 1/2/4/8) and both chain modes (fma / mul+add).
- rtree:  the cancellation-probe masks of smallk.router_gemm equal cuBLAS's on random queries.
- topk:   values (bitwise) and indices of vLLM's grouped_topk vs smallk.topk_sigmoid, for random
          logits at several scales with the real e_score_correction_bias of every layer, ties
          (quantized logits, identical rows, saturated sigmoids), and non-finite entries.
- sweep:  sigmoid of every fp32 logit with 2^-12 <= |x| < 16 (renormalize off, scale 1, bias 0:
          the value written is the sigmoid itself), stock vs smallk.
- align:  sorted_ids / expert_ids / num_tokens_post_pad vs vLLM's moe_align_block_size.
- timing: per-call time in a CUDA graph (23 calls back to back) stock vs smallk, M = 4.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe import F32Prober, graph_us  # noqa: E402

import smallk  # noqa: E402

OUT = os.environ.get("NEMO_OUT", "/tmp/smallk-out")
SNAP = os.environ.get("NEMO_SNAPSHOT", "")
os.makedirs(OUT, exist_ok=True)
DEV = torch.device("cuda")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def save(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=str)


def gate_tensors():
    from safetensors import safe_open

    idx = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))["weight_map"]
    layers = sorted({int(k.split(".")[2]) for k in idx if k.startswith("backbone.layers.") and k.endswith("mixer.gate.weight")})
    W, B = {}, {}
    for li in layers:
        for kind, store in (("weight", W), ("e_score_correction_bias", B)):
            name = f"backbone.layers.{li}.mixer.gate.{kind}"
            with safe_open(os.path.join(SNAP, idx[name]), framework="pt", device="cuda") as f:
                t = f.get_tensor(name)
                if kind == "weight" and li == layers[0]:
                    log("gate weight checkpoint dtype", str(t.dtype))
                store[li] = t.float().contiguous()
    return layers, W, B


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


def router_inputs(rng_seed: int, M: int, K: int):
    g = torch.Generator(device=DEV).manual_seed(rng_seed)
    out = {}
    out["normal"] = torch.randn(M, K, device=DEV, generator=g)
    out["heavy"] = torch.randn(M, K, device=DEV, generator=g) * torch.exp(2.0 * torch.randn(M, K, device=DEV, generator=g))
    e = torch.randint(-30, 30, (M, K), device=DEV, generator=g).float()
    s = torch.where(torch.rand(M, K, device=DEV, generator=g) < 0.5, -1.0, 1.0)
    out["wide"] = s * torch.exp2(e) * (1 + torch.rand(M, K, device=DEV, generator=g))
    out["bf16"] = torch.randn(M, K, device=DEV, generator=g).bfloat16().float() * 3
    h = torch.randn(M, K // 2, device=DEV, generator=g)
    out["cancel"] = torch.cat([h, -h * (1 + 1e-3 * torch.randn(M, K // 2, device=DEV, generator=g))], 1)
    # rms-normed rows with a few outlier channels, like a decoder hidden state after the norm
    hs = torch.randn(M, K, device=DEV, generator=g)
    hs[:, torch.randint(0, K, (16,), device=DEV, generator=g)] *= 40
    out["rmsnorm"] = (hs * torch.rsqrt(hs.pow(2).mean(-1, keepdim=True) + 1e-5)).bfloat16().float()
    return out


def part_router(layers, W, trials=40):
    res = {"variants": {}}
    M, K = 4, W[layers[0]].shape[1]
    variants = [(rows, mode) for mode in (0, 1) for rows in (1, 2, 4, 8)]
    tot = {v: [0, 0] for v in variants}
    w16 = {li: smallk.gate_w16(W[li]) for li in layers}
    tot16 = {rows: [0, 0] for rows in (1, 2, 4, 8)}
    per_kind = {}
    for t in range(trials):
        xs = router_inputs(1000 + t, M, K)
        for kind, x in xs.items():
            for li in layers:
                ref = F.linear(x, W[li])
                for v in variants:
                    y = smallk.router_gemm(x, W[li], rows=v[0], mode=v[1])
                    bad = int((ref.view(torch.int32) != y.view(torch.int32)).sum())
                    tot[v][0] += bad
                    tot[v][1] += ref.numel()
                    if v == (4, 0):
                        pk = per_kind.setdefault(kind, [0, 0])
                        pk[0] += bad
                        pk[1] += ref.numel()
                if w16[li] is not None:
                    for rows in tot16:
                        y = smallk.router_gemm(x, W[li], rows=rows, w16=w16[li])
                        tot16[rows][0] += int((ref.view(torch.int32) != y.view(torch.int32)).sum())
                        tot16[rows][1] += ref.numel()
    for v, (bad, n) in tot.items():
        res["variants"][f"rows={v[0]} mode={'fma' if v[1] == 0 else 'mul+add'}"] = {"mismatches": bad, "outputs": n}
    res["w16_gates"] = sum(v is not None for v in w16.values())
    for rows, (bad, n) in tot16.items():
        res["variants"][f"rows={rows} mode=fma w16"] = {"mismatches": bad, "outputs": n}
    res["per_kind_rows4_fma"] = per_kind
    log("router", json.dumps(res))
    save("exact-router.json", res)


def part_rtree(layers, W):
    K = W[layers[0]].shape[1]
    N = 128
    stock = F32Prober(N, K, 4)
    mine = F32Prober(N, K, 4, fn=lambda x, w: smallk.router_gemm(x, w))
    rng = np.random.default_rng(11)
    Q = 64 * N
    a = rng.integers(0, K, Q)
    b = (a + 1 + rng.integers(0, K - 1, Q)) % K
    P = rng.integers(0, K, (Q, 8))
    P[(P == a[:, None]) | (P == b[:, None])] = -1
    m1, ok1 = stock.run(a, b, P)
    m2, ok2 = mine.run(a, b, P)
    # local probes: a, b and probes in the same or neighbouring 256-tiles
    a2 = rng.integers(0, K, Q)
    b2 = np.clip(a2 + rng.integers(-300, 300, Q), 0, K - 1)
    b2 = np.where(b2 == a2, (a2 + 16) % K, b2)
    P2 = np.clip(a2[:, None] + rng.integers(-400, 400, (Q, 8)), 0, K - 1)
    P2[(P2 == a2[:, None]) | (P2 == b2[:, None])] = -1
    m3, _ = stock.run(a2, b2, P2)
    m4, _ = mine.run(a2, b2, P2)
    res = {"queries": 2 * Q, "consistent": bool(ok1 and ok2), "random_equal": int((m1 == m2).sum()),
           "local_equal": int((m3 == m4).sum()), "per_set": Q}
    log("rtree", res)
    save("exact-rtree.json", res)


def stock_topk(logits, bias, topk=6, renorm=True, scale=2.5):
    from vllm import _custom_ops as ops

    return ops.grouped_topk(logits, 1, 1, topk, renorm, scale, bias, 1)


def stock_align(topk_ids, block_size, num_experts=128):
    """vLLM's moe_align_block_size (no expert map) without the smallk hook."""
    from vllm import _custom_ops as ops

    sorted_ids, expert_ids, ntpp = smallk.align_buffers(topk_ids.numel(), num_experts, block_size, topk_ids.device)
    ops.moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids, expert_ids, ntpp, None)
    return sorted_ids, expert_ids, ntpp


def part_topk(layers, B, trials=200):
    res = {}
    g = torch.Generator(device=DEV).manual_seed(7)
    cases = {}

    def check(name, logits, bias, topk=6, renorm=True, scale=2.5):
        v0, i0 = stock_topk(logits, bias, topk, renorm, scale)
        v1, i1 = smallk.topk_sigmoid(logits, bias, topk, renorm, scale)
        c = cases.setdefault(name, {"rows": 0, "bad_rows": 0})
        badrow = ((v0.view(torch.int32) != v1.view(torch.int32)) | (i0 != i1)).any(1)
        c["rows"] += logits.shape[0]
        c["bad_rows"] += int(badrow.sum())

    for t in range(trials):
        li = layers[t % len(layers)]
        bias = B[li]
        for M in (1, 4, 8, 64):
            for sigma in (0.5, 2.0, 5.0):
                check(f"normal s={sigma}", torch.randn(M, 128, device=DEV, generator=g) * sigma, bias)
            q = torch.round(torch.randn(M, 128, device=DEV, generator=g) * 2) / 2
            check("quantized ties", q, bias)
            check("quantized ties, zero bias", q, torch.zeros_like(bias))
            row = torch.randn(1, 128, device=DEV, generator=g)
            check("identical rows", row.expand(M, 128).contiguous(), bias)
            check("flat row", torch.full((M, 128), float(torch.randn(1, generator=g, device=DEV)), device=DEV), torch.zeros_like(bias))
            check("saturated", torch.randn(M, 128, device=DEV, generator=g) * 60, bias)
            x = torch.randn(M, 128, device=DEV, generator=g)
            x[torch.rand(M, 128, device=DEV, generator=g) < 0.05] = float("inf")
            x[torch.rand(M, 128, device=DEV, generator=g) < 0.05] = float("-inf")
            x[torch.rand(M, 128, device=DEV, generator=g) < 0.05] = float("nan")
            check("non-finite", x, bias)
            x = torch.full((M, 128), float("nan"), device=DEV)
            x[:, :3] = torch.randn(M, 3, device=DEV, generator=g)
            check("fewer than topk finite", x, bias)
            check("topk=8 no renorm scale=1", torch.randn(M, 128, device=DEV, generator=g), bias, 8, False, 1.0)
            check("topk=1", torch.randn(M, 128, device=DEV, generator=g), bias, 1, True, 2.5)
    res["cases"] = cases
    res["bad_rows_total"] = sum(c["bad_rows"] for c in cases.values())
    res["rows_total"] = sum(c["rows"] for c in cases.values())
    log("topk", json.dumps(res))
    save("exact-topk.json", res)


def part_sweep(exp_lo=-12, exp_hi=4):
    """Every fp32 x with 2^exp_lo <= |x| < 2^exp_hi, 8 per row (others -inf), topk 8, no renorm, scale 1, bias 0."""
    bias = torch.zeros(128, device=DEV)
    lo = (exp_lo + 127) << 23
    hi = (exp_hi + 127) << 23
    total = 0
    bad = 0
    chunk = 1 << 24
    for sign in (0, 1):
        for start in range(lo, hi, chunk):
            n = min(chunk, hi - start)
            b = torch.arange(start, start + n, device=DEV, dtype=torch.int64) + (sign << 31)
            vals = torch.where(b >= 1 << 31, b - (1 << 32), b).to(torch.int32).view(torch.float32)
            rows = n // 8
            logits = torch.full((rows, 128), float("-inf"), device=DEV)
            logits[:, :8] = vals[: rows * 8].view(rows, 8)
            v0, i0 = stock_topk(logits, bias, 8, False, 1.0)
            v1, i1 = smallk.topk_sigmoid(logits, bias, 8, False, 1.0)
            bad += int(((v0.view(torch.int32) != v1.view(torch.int32)) | (i0 != i1)).any(1).sum())
            total += rows * 8
    res = {"values": total, "bad_rows": bad, "range": [exp_lo, exp_hi]}
    log("sweep", res)
    save("exact-sweep.json", res)


def part_align(trials=3000):
    rng = np.random.default_rng(3)
    res = {}
    for M in (1, 2, 3, 4, 5, 8):
        for bs in (16, 32, 64):
            bad = skipped = 0
            for t in range(trials // 10):
                pool = rng.choice(128, int(rng.integers(6, 13)), replace=False) if t % 2 else np.arange(128)
                ids = np.stack([rng.choice(pool, 6, replace=False) for _ in range(M)]).astype(np.int32)
                if t % 7 == 3:
                    ids[0, 0] = -1  # invalid id: skipped by both
                ti = torch.as_tensor(ids, device=DEV)
                s0, e0, n0 = stock_align(ti, bs)
                got = smallk.maybe_align(ti, 128, bs, None) if smallk.ALIGN_ON else None
                if got is None:
                    skipped += 1
                    continue
                s1, e1, n1 = got
                if not (torch.equal(s0, s1) and torch.equal(e0, e1) and torch.equal(n0, n1)):
                    bad += 1
                    if bad <= 2:
                        log("align mismatch", M, bs, ids.tolist(), s0.tolist()[:48], s1.tolist()[:48])
            res[f"M={M} bs={bs}"] = {"trials": trials // 10, "bad": bad, "not_routed": skipped}
    log("align", json.dumps(res))
    save("exact-align.json", res)


def part_route(layers, B, trials=400):
    """route_kernel vs stock top-k then stock align; and the parked hand-off through the hooks."""
    g = torch.Generator(device=DEV).manual_seed(19)
    res = {}
    for M in (1, 2, 3, 4, 5):
        for bs in (16, 32, 64):
            bad = 0
            for t in range(trials // 10):
                bias = B[layers[t % len(layers)]]
                x = torch.randn(M, 128, device=DEV, generator=g) * (0.5 + 3 * (t % 3))
                if t % 4 == 1:
                    x = torch.round(x * 2) / 2
                v0, i0 = stock_topk(x, bias)
                s0, e0, n0 = stock_align(i0, bs)
                v1, i1, s1, e1, n1 = smallk.route(x, bias, 6, True, 2.5, bs)
                ok = bits_equal(v0, v1) and torch.equal(i0, i1) and torch.equal(s0, s1) and torch.equal(e0, e1) \
                    and torch.equal(n0, n1)
                # hooks: align learns the block size, top-k then runs the fused kernel, align takes its outputs
                smallk.maybe_align(i1, 128, bs, None)
                v2, i2 = smallk.maybe_topk_sigmoid(x, bias, 6, True, 2.5)
                parked = smallk._parked
                s2, e2, n2 = smallk.maybe_align(i2, 128, bs, None)
                ok = ok and parked is not None and s2.data_ptr() == parked[3].data_ptr()
                ok = ok and bits_equal(v0, v2) and torch.equal(i0, i2) and torch.equal(s0, s2) \
                    and torch.equal(e0, e2) and torch.equal(n0, n2)
                bad += not ok
            res[f"M={M} bs={bs}"] = {"trials": trials // 10, "bad": bad}
    log("route", json.dumps(res))
    save("exact-route.json", res)


def part_timing(layers, W, B):
    res = {}
    M = 4
    x = torch.randn(M, W[layers[0]].shape[1], device=DEV)
    Ws = [W[li] for li in layers] * 2  # 46 x 1.4 MB: cold L2
    it = [0]

    def nxt():
        it[0] += 1
        return Ws[it[0] % len(Ws)]

    res["router stock cold"] = graph_us(lambda: F.linear(x, nxt()))
    for rows in (1, 2, 4, 8):
        res[f"router smallk rows={rows} cold"] = graph_us(lambda: smallk.router_gemm(x, nxt(), rows=rows))
        res[f"router smallk rows={rows} cold pdl"] = graph_us(lambda: smallk.router_gemm(x, nxt(), rows=rows, pdl=1))
    logits = torch.randn(M, 128, device=DEV)
    bias = B[layers[0]]
    res["topk stock"] = graph_us(lambda: stock_topk(logits, bias))
    for w in (1, 4):
        res[f"topk smallk warps={w}"] = graph_us(lambda: smallk.topk_sigmoid(logits, bias, 6, True, 2.5, warps=w))
    ids = torch.stack([torch.randperm(128, device=DEV)[:6] for _ in range(M)]).to(torch.int32)
    res["align stock"] = graph_us(lambda: stock_align(ids, 16))
    s0, e0, n0 = stock_align(ids, 16)
    for pdl in (0, 1):
        res[f"topk smallk pdl={pdl}"] = graph_us(lambda: smallk.topk_sigmoid(logits, bias, 6, True, 2.5, pdl=pdl))
        res[f"align smallk pdl={pdl}"] = graph_us(lambda: smallk.align(ids, 128, 16, s0, e0, n0, pdl=pdl))
        res[f"route smallk pdl={pdl}"] = graph_us(lambda: smallk.route(logits, bias, 6, True, 2.5, 16, pdl=pdl))

    # the routing chain as the engine runs it: gemm -> topk -> align
    def chain_stock():
        lg = F.linear(x, nxt())
        v, i = stock_topk(lg, bias)
        stock_align(i, 16)

    def chain_smallk():
        lg = smallk.router_gemm(x, nxt())
        v, i = smallk.topk_sigmoid(lg, bias, 6, True, 2.5)
        s, e, n = smallk.align_buffers(24, 128, 16, DEV)
        smallk.align(i, 128, 16, s, e, n)

    def chain_fused():
        lg = smallk.router_gemm(x, nxt())
        smallk.route(lg, bias, 6, True, 2.5, 16)

    res["chain stock"] = graph_us(chain_stock)
    res["chain smallk"] = graph_us(chain_smallk)
    res["chain smallk fused"] = graph_us(chain_fused)
    for k, v in res.items():
        log("timing", k, round(v, 2))
    save("exact-timing.json", res)


def main():
    parts = sys.argv[1].split(",") if len(sys.argv) > 1 else ["router", "rtree", "topk", "sweep", "align", "route", "timing"]
    assert smallk.available(), smallk.load_error()
    assert smallk.init()
    layers, W, B = gate_tensors()
    target = layers
    log("gates", len(layers), "target", len(target), W[target[0]].shape, W[target[0]].dtype)
    for p in parts:
        if p == "router":
            part_router(target, W)
        elif p == "rtree":
            part_rtree(target, W)
        elif p == "topk":
            part_topk(target, B)
        elif p == "sweep":
            part_sweep()
        elif p == "align":
            part_align()
        elif p == "route":
            part_route(target, B)
        elif p == "timing":
            part_timing(target, W, B)
    log("done")


if __name__ == "__main__":
    main()

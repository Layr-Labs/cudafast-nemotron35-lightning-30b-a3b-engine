#!/usr/bin/env python3
"""Bit-identity and speed of zskip (k-row-major, zero-row-skipping routed w2) vs stock humming.

Runs in the pinned image: bin/atom nemo exec SHA harness/kernels/zskip/tests/zskip_exact.py --phase ...
(tree overlay + $KERNELS_OUT on PYTHONPATH, snapshot at $NEMO_SNAPSHOT, outputs to $NEMO_OUT).

Phases (--phase, comma-separated):
  probe    tensor-core semantics the skip relies on (csrc probe_kernel): with acc != -0,
           mma(acc, a, b) == mma(acc, a, b') when b' = +0 at k-slots where a == 0; and
           mma(acc, 0, b) == acc for acc not -0 / NaN. Random + adversarial trials.
  epi      the kernel's epilogue (explicit mul.rn.ftz + cvt.rn.bf16x2) vs humming's epilogue code
           compiled by humming's own NVRTC flags (--use_fast_math), all 2^32 fp32 inputs.
  capture  real MoE inputs: eager in-process engine, prefill of prompt + reference continuation
           (sequences.json), records each MoE layer's input x (as harness/sparsity/relu2_probe.py).
  exact    per layer and input: stock humming w2 output vs zskip DENSE and SKIP, bitwise.
  prefix   paired old/new bitwise checks at M5/M9, prefixes 1..5/all, ordinary/PDL graph replay.
  time     CUDA-graph time per call over the loaded layers (cold weights): stock, zskip modes,
           configs and layouts, plus the load-only timing probes and byte counts.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.environ.get("NEMO_TREE", os.path.abspath(os.path.join(HERE, "..", "..", "..", "..")))
sys.path.insert(0, os.path.join(TREE, "harness", "tests"))

import humming_moe_exact as hme  # noqa: E402
import zskip  # noqa: E402

try:  # the competing exact sparse down kernel (harness/kernels/routed_sparse, main bce7a11)
    import routed_sparse  # noqa: E402
except ImportError:
    routed_sparse = None

HIDDEN, INTER, NUM_EXPERTS, TOP_K = hme.HIDDEN, hme.INTER, hme.NUM_EXPERTS, hme.TOP_K
MOE_LAYERS = hme.MOE_LAYERS
T0 = time.time()
OUT = None


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def dump(name, obj):
    path = os.path.join(OUT, f"{name}.json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    log(f"wrote {path}")


# ---------------------------------------------------------------------------------------- probe

PROBE_KINDS = ["relu2-like A, real B", "full-range A/B", "exact cancellation", "huge B at zero k",
               "subnormal A", "sparse A", "non-finite C"]


def phase_probe(args, ctx):
    zskip.init()
    tot = [0] * 56
    samples = []
    t0 = time.time()
    for i in range(args.probe_launches):
        c, s = zskip.probe(0x5EED0000 + i * 7919, args.probe_blocks, args.probe_iters)
        tot = [a + b for a, b in zip(tot, c)]
        n = min(s[0], 64)
        for j in range(n):
            samples.append([hex(v) for v in s[1 + 8 * j: 9 + 8 * j]])
    res = {"seconds": round(time.time() - t0, 1), "kinds": {}}
    ok = True
    for k, name in enumerate(PROBE_KINDS):
        c = tot[k * 8: k * 8 + 6]
        row = {"elements": c[0], "mismatch_c_not_neg0": c[1], "mismatch_c_neg0": c[2], "c_neg0": c[3],
               "zeroA_identity_violations": c[4], "zeroA_checked": c[5]}
        res["kinds"][name] = row
        ok &= c[1] == 0 and c[4] == 0
        log(f"probe {name}: {row}")
    res["ok"] = ok
    res["samples"] = samples[:64]
    log(f"PROBE substitution property holds (acc != -0) and zero-A identity holds: {ok}")
    if samples:
        log(f"   samples: {samples[:6]}")
    dump("probe", res)
    if not ok:
        sys.exit(3)


EPI_SRC = r"""
#include <cuda_bf16.h>
#include <humming/arith/exp_offset.cuh>
extern "C" __global__ void epi_cmp(unsigned long long base, unsigned iters, float gs,
                                   unsigned long long *counts, unsigned *samp) {
  unsigned long long nth = (unsigned long long)gridDim.x * blockDim.x;
  unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long bad = 0, nanout = 0;
  // humming EpilogueArithmetic (tensor weight scale, fp32 accum): gs *= 2^2; x * gs; float22num2
  float g1 = gs;
  g1 *= prepare_exp_scale_factor<float, 2>();
  float g2;
  asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(g2) : "f"(gs), "f"(4.0f));
  for (unsigned it = 0; it < iters; ++it) {
    unsigned x = (unsigned)(base + (unsigned long long)it * nth + tid);
    float xf = __uint_as_float(x);
    float h = xf * g1;
    __nv_bfloat162 hb = __float22bfloat162_rn(make_float2(h, h));
    float m;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(m) : "f"(xf), "f"(g2));
    __nv_bfloat162 mb = __float22bfloat162_rn(make_float2(m, m));
    unsigned hu = *reinterpret_cast<unsigned *>(&hb) & 0xffffu;
    unsigned mu = *reinterpret_cast<unsigned *>(&mb) & 0xffffu;
    if (hu != mu || __float_as_uint(h) != __float_as_uint(m)) {
      bad++;
      unsigned s = atomicAdd(samp, 1u);
      if (s < 16) { samp[1 + 3 * s] = x; samp[2 + 3 * s] = hu; samp[3 + 3 * s] = mu; }
    }
    if ((hu & 0x7f80u) == 0x7f80u && (hu & 0x7fu)) nanout++;
  }
  atomicAdd(&counts[0], bad);
  atomicAdd(&counts[1], nanout);
}
"""


def phase_epi(args, ctx):
    import cuda.bindings.driver as cbd
    from humming.jit.compiler import NVRTCCompiler

    props = torch.cuda.get_device_properties(0)
    sm = f"{props.major}{props.minor}" + ("a" if props.major * 10 + props.minor >= 90 else "")
    path = NVRTCCompiler.compile(EPI_SRC, sm, None)
    func = hme._cu_func(cbd, path, b"epi_cmp")
    gss = [float(v) for v in args.epi_gs.split(",")]
    if ctx.get("gs_values"):
        gss += ctx["gs_values"][:4]
    blocks, threads = 1024, 256
    per = 1 << 28
    iters = per // (blocks * threads)
    rows = []
    ok = True
    for gs in gss:
        counts = torch.zeros(2, dtype=torch.int64, device="cuda")
        samp = torch.zeros(49, dtype=torch.int32, device="cuda")
        for chunk in range((1 << 32) // per):
            hme._launch(cbd, func, blocks, threads,
                        (chunk * per, iters, gs, counts.data_ptr(), samp.data_ptr()),
                        (ctypes.c_ulonglong, ctypes.c_uint, ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p))
        torch.cuda.synchronize()
        c = counts.tolist()
        rows.append({"gs": gs, "mismatch": c[0], "noncanonical_bf16_nan": c[1],
                     "samples": hme._samples(samp)})
        ok &= c[0] == 0
        log(f"epi gs={gs!r}: mismatches {c[0]} / 2^32, bf16 NaN outputs with payload {c[1]}")
    # NaN payloads through the .so's own epilogue
    raw_pats = [0x7FC00000, 0x7FC00123, 0xFFC00001, 0x7F800001, 0x80000000, 0x00000001, 0x80000001, 0x7F800000]
    pats = torch.tensor([v - (1 << 32) if v >= 1 << 31 else v for v in raw_pats], dtype=torch.int32, device="cuda")
    ob, om = zskip.epi(pats, 0.37)
    torch.cuda.synchronize()
    nan_rows = [[hex(int(p) & 0xFFFFFFFF), hex(int(b) & 0xFFFF), hex(int(m) & 0xFFFFFFFF)]
                for p, b, m in zip(pats.tolist(), ob.tolist(), om.tolist())]
    log(f"epi .so NaN/subnormal patterns (in, bf16, fp32 product): {nan_rows}")
    dump("epi", {"ok": ok, "rows": rows, "so_patterns": nan_rows, "nvrtc_flags": NVRTCCompiler.get_flags(sm)})
    if not ok:
        sys.exit(3)


# -------------------------------------------------------------------------------------- capture


def phase_capture(args, ctx):
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import vllm.model_executor.models.nemotron_h as nh
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    with open(os.path.join(HERE, "sequences.json")) as f:
        seqs = json.load(f)["sequences"]
    if args.capture != "all":
        keep = set(args.capture.split(","))
        seqs = [s for s in seqs if s["name"] in keep]
    cap = {"on": False, "x": {}}
    orig_init, orig_fwd = nh.NemotronHMoE.__init__, nh.NemotronHMoE.forward

    def init(self, *a, **kw):
        orig_init(self, *a, **kw)
        prefix = kw.get("prefix", "")
        self._zs_layer = int(prefix.split("layers.")[1].split(".")[0]) if "layers." in prefix else -1

    def fwd(self, hidden_states):
        if cap["on"]:
            cap["x"].setdefault(self._zs_layer, []).append(hidden_states.detach().to("cpu", copy=True))
        return orig_fwd(self, hidden_states)

    nh.NemotronHMoE.__init__, nh.NemotronHMoE.forward = init, fwd
    snap = os.environ["NEMO_SNAPSHOT"]
    log("capture: building eager engine")
    llm = LLM(model=snap, tokenizer=snap, skip_tokenizer_init=True, dtype="auto", seed=0, max_model_len=1024,
              max_num_seqs=1, max_num_batched_tokens=2048, enable_prefix_caching=False,
              gpu_memory_utilization=0.4, kv_cache_memory_bytes=2 << 30, mamba_ssm_cache_dtype="float32",
              moe_backend="humming", safetensors_load_strategy="eager", enforce_eager=True,
              disable_log_stats=True)
    sp = SamplingParams(max_tokens=1, temperature=0.0, detokenize=False)
    caps = []
    for s in seqs:
        ids = s["prompt"] + [s["seed_token"]] + s["decode_tokens"]
        cap["x"].clear()
        cap["on"] = True
        llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
        cap["on"] = False
        layers = sorted(cap["x"])
        assert len(layers) == 23, layers
        caps.append({"name": s["name"], "n_prompt": len(s["prompt"]),
                     "x": {L: torch.cat(cap["x"][L])[: len(ids)] for L in layers}})
        log(f"capture {s['name']}: {len(ids)} tokens")
    nh.NemotronHMoE.__init__, nh.NemotronHMoE.forward = orig_init, orig_fwd
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    log(f"capture done; cuda allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    ctx["caps"] = caps


# ---------------------------------------------------------------------------------------- inputs


def stock_w2(ex, act, kw2):
    from vllm.utils.humming import HummingMethod

    out = torch.empty(act.shape[0], HIDDEN, dtype=torch.bfloat16, device="cuda")
    HummingMethod.forward_layer(layer=ex.layer, inputs=act, input_scale=None, outputs=out,
                                sublayer_name="w2", **kw2)
    return out


def mine_w2(packed, act, kw2, mode, config=None, stats=None, sentinel=True):
    out = torch.empty(act.shape[0], HIDDEN, dtype=torch.bfloat16, device="cuda")
    if sentinel:
        out.view(torch.int16).fill_(0x7FBF)  # NaN payload humming never writes
    return zskip.down(act, kw2["sorted_ids"], kw2["expert_ids"], kw2["num_tokens_padded"], packed, out,
                      mode=mode, config=config, stats=stats)


def realistic_cases(ctx, li, ms, windows):
    """(name, x, topk_ids, topk_weights) from captured decode windows of this layer."""
    out = []
    if not ctx.get("caps"):
        return out
    L = ctx["layer_ids"][li]
    gate_w, bias, _ = ctx["routers"][li]
    for c in ctx["caps"]:
        x_all = c["x"][L]
        n_p = c["n_prompt"]
        for w in range(windows):
            a0 = n_p + 4 * w * (128 // 4 // max(windows, 1))
            for m in ms:
                x = x_all[a0: a0 + m].to("cuda").contiguous()
                tid, tw = hme.route(x, gate_w, bias)
                out.append((f"real:{c['name']}:w{w}:M{m}", x, tid, tw))
    return out


def synthetic_cases(ctx, li, ms, kinds, reps):
    out = []
    L = ctx["layer_ids"][li]
    for m in ms:
        for kind in kinds:
            for rep in range(reps):
                seed = 1000003 * L + 7919 * m + 104729 * rep + sum(map(ord, kind))
                x, tid, tw = hme.make_inputs(kind, m, seed, ctx["emb"], ctx["routers"][li], "cuda")
                out.append((f"{kind}:M{m}:r{rep}", x, tid, tw))
    return out


def adversarial_acts(m, seed):
    """Activations not produced by the up projection: edge values the kernel must handle exactly."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    rows = m * TOP_K
    res = {}
    base = torch.randn(rows, INTER, generator=g, device="cuda")
    keep = torch.rand(rows, INTER, generator=g, device="cuda") < 0.08
    relu2 = (base.relu() ** 2 * keep).to(torch.bfloat16)
    res["sparse92"] = relu2
    v = relu2.clone()
    negz = torch.rand(rows, INTER, generator=g, device="cuda") < 0.3
    v.view(torch.int16)[negz & (v == 0)] = -32768  # -0 entries count as zero for the skip
    res["neg_zero"] = v
    res["subnormal"] = (relu2.float() * 1e-38).to(torch.bfloat16)
    res["tiny_signed"] = ((base * 1e-36) * keep).to(torch.bfloat16)  # products underflow, signed
    res["signed_sparse"] = (base * keep * 3).to(torch.bfloat16)      # cancellation-prone, -0 sums
    res["huge"] = (relu2.float() * 1e34).clamp(max=3e38).to(torch.bfloat16)
    v = relu2.clone()
    v[0, 7] = float("nan")
    v[min(1, rows - 1), 100] = float("inf")
    v[min(2, rows - 1), 1855] = float("-inf")
    res["nonfinite"] = v
    v = relu2.clone()
    v[: max(1, rows // 3)] = 0
    res["zero_rows"] = v
    v = torch.zeros(rows, INTER, dtype=torch.bfloat16, device="cuda")
    idx = torch.randint(0, INTER, (rows,), generator=g, device="cuda")
    v[torch.arange(rows, device="cuda"), idx] = 1.5
    res["one_hot"] = v
    res["all_zero"] = torch.zeros(rows, INTER, dtype=torch.bfloat16, device="cuda")
    res["dense_relu2"] = (base.relu() ** 2).to(torch.bfloat16)
    return res


def block_masks(act, kw2):
    """Per expert block: union nonzero mask over its valid rows -> (active k, active groups) lists."""
    ntp = int(kw2["num_tokens_padded"].item())
    sid = kw2["sorted_ids"][:ntp].view(-1, 16)
    nz = (act.view(torch.int16) & 0x7FFF) != 0
    act_k, act_g, nrows, sect = [], [], [], []
    for b in range(sid.shape[0]):
        r = sid[b][sid[b] < act.shape[0]].long()
        u = nz[r].any(0)
        act_k.append(int(u.sum()))
        ug = u.view(-1, 16)
        act_g.append(int(ug.any(1).sum()))
        # routed_sparse sectors: k in {2t, 2t+1, 2t+8, 2t+9} of a group
        sec = ug.view(-1, 2, 4, 2).any(-1).any(1)  # [groups, t]
        sect.append(int(sec.sum()))
        nrows.append(int(r.numel()))
    return act_k, act_g, nrows, sect


# ----------------------------------------------------------------------------------------- exact


def exact_variants(ctx, li):
    """(packed, mode, name, config): v3 on the first non-swap layout, v4 on swap."""
    out = []
    pks = ctx["packed"][li]
    v3 = next((k for k in pks if k not in ("swap", "swap2")), None)
    if v3 is not None:
        out += [(pks[v3], zskip.DENSE, "v3dense", 1), (pks[v3], zskip.SKIP, "v3skip", 1)]
    if "swap" in pks:
        out += [(pks["swap"], zskip.DENSE, "v4dense", 100), (pks["swap"], zskip.SKIP, "v4skip", 100)]
    if "swap2" in pks:
        for cfg in ctx["pair_configs"]:
            for mode, name in ((zskip.DENSE, "dense"), (zskip.SKIP, "skip")):
                out.append((pks["swap2"], mode, f"pair{cfg}/{name}", cfg))
    return out


def configs_for_layout(ctx, lay):
    if lay == "swap2":
        return ctx["pair_configs"]
    return zskip.CONFIGS4 if lay == "swap" else zskip.CONFIGS


def phase_exact(args, ctx):
    zskip.init()
    stats = torch.zeros(4, dtype=torch.int32, device="cuda")
    rows = []
    all_ok = True
    agg = {}
    for li, L in enumerate(ctx["layer_ids"]):
        ex = ctx["ex"][li]
        packed = ctx["packed"][li]["rows"]
        cases = realistic_cases(ctx, li, args.ms, args.windows) + synthetic_cases(ctx, li, args.ms, args.kinds, args.reps)
        for name, x, tid, tw in cases:
            st = hme.run_stages(ex, x, tid, tw)
            kw1, kw2 = ex.prepare_humming_moe_kwargs(tid, None, None)
            ref = stock_w2(ex, st["act"], kw2)
            ok_ref = hme.bits_equal(ref, st["w2"])
            for pk, mode, mname, cfg in exact_variants(ctx, li):
                got = mine_w2(pk, st["act"], kw2, mode, config=cfg, stats=stats)
                eq = hme.bits_equal(got, ref)
                key = (name.split(":")[0], mname)
                a = agg.setdefault(f"{key[0]}/{key[1]}", {"cases": 0, "ok": 0})
                a["cases"] += 1
                a["ok"] += int(eq)
                if not eq:
                    all_ok = False
                    rows.append({"layer": L, "case": name, "mode": mname, "diff": hme.diff_count(got, ref),
                                 "ref_self": ok_ref})
                    log(f"MISMATCH L{L} {name} {mname}: {hme.diff_count(got, ref)} elements")
            if ctx.get("rs") is not None:
                got = torch.empty_like(ref)
                ok_rs = routed_sparse.forward(st["act"], tid.to(torch.int32).contiguous(), got, ctx["rs"][li])
                eq = ok_rs and hme.bits_equal(got, ref)
                a = agg.setdefault(f"{name.split(':')[0]}/routed_sparse", {"cases": 0, "ok": 0})
                a["cases"] += 1
                a["ok"] += int(eq)
        # adversarial activations with random routing
        for m in args.ms:
            gen = torch.Generator().manual_seed(77 + L * 13 + m)
            tid, tw = hme.random_route(m, gen, "cuda")
            kw1, kw2 = ex.prepare_humming_moe_kwargs(tid, None, None)
            for aname, act in adversarial_acts(m, 9001 + L * 31 + m).items():
                act = act.contiguous()
                ref = stock_w2(ex, act, kw2)
                for pk, mode, mname, cfg in exact_variants(ctx, li):
                    got = mine_w2(pk, act, kw2, mode, config=cfg, stats=stats)
                    eq = hme.bits_equal(got, ref)
                    a = agg.setdefault(f"adv:{aname}/{mname}", {"cases": 0, "ok": 0})
                    a["cases"] += 1
                    a["ok"] += int(eq)
                    if not eq:
                        all_ok = False
                        rows.append({"layer": L, "case": f"adv:{aname}:M{m}", "mode": mname,
                                     "diff": hme.diff_count(got, ref)})
                        log(f"MISMATCH L{L} adv:{aname} M{m} {mname}: {hme.diff_count(got, ref)} elements")
        # other configs / layout on one realistic or synthetic case per layer
        x, tid, tw = cases[0][1], cases[0][2], cases[0][3]
        st = hme.run_stages(ex, x, tid, tw)
        kw1, kw2 = ex.prepare_humming_moe_kwargs(tid, None, None)
        ref = stock_w2(ex, st["act"], kw2)
        for lay, pk in ctx["packed"][li].items():
            for cfg in configs_for_layout(ctx, lay):
                got = mine_w2(pk, st["act"], kw2, zskip.SKIP, config=cfg, stats=stats)
                eq = hme.bits_equal(got, ref)
                a = agg.setdefault(f"configs/{lay}", {"cases": 0, "ok": 0})
                a["cases"] += 1
                a["ok"] += int(eq)
                if not eq:
                    all_ok = False
                    log(f"MISMATCH L{L} config {lay} {cfg}")
        torch.cuda.synchronize()
        log(f"exact L{L}: {len(cases)} cases; running all_ok={all_ok}; stats {stats.tolist()}")
    s = stats.tolist()
    log(f"exact summary: {json.dumps(agg)}")
    log(f"dense-redo warps (a -0 accumulator appeared in SKIP): {s[0]}; blocks with > 4 rows: {s[1]}")
    log(f"ALL BIT-IDENTICAL: {all_ok}")
    dump("exact", {"all_ok": all_ok, "summary": agg, "mismatches": rows, "redo_warps": s[0], "big_blocks": s[1]})
    if not all_ok:
        ctx["exact_failed"] = True


# ---------------------------------------------------------------------------------- prefix


def phase_prefix(args, ctx):
    zskip.init()
    expected_pdl = os.environ.get("ZSKIP_PDL", "1") != "0"
    if zskip._pdl_on != expected_pdl:
        raise RuntimeError(f"PDL fallback before prefix proof: requested={expected_pdl}, active={zskip._pdl_on}")
    stats = torch.zeros(4, dtype=torch.int32, device="cuda")
    mismatches, checks, values, replays = [], 0, 0, 0

    def call(cfg, act, kw, pk, out, vrows):
        out.view(torch.int16).fill_(0x7FBF)
        return zskip.down(act, kw["sorted_ids"], kw["expert_ids"], kw["num_tokens_padded"], pk, out,
                          mode=zskip.SKIP, config=cfg, max_blocks=act.shape[0], stats=stats,
                          verify_rows=vrows)

    def check(got, ref, label):
        nonlocal checks, values
        got, ref = got.view(torch.int16), ref.view(torch.int16)
        checks += 1
        values += got.numel()
        if not torch.equal(got, ref):
            mismatches.append({"case": label, "different": int((got != ref).sum().item())})
            log(f"PREFIX MISMATCH {label}")

    for li, L in enumerate(ctx["layer_ids"]):
        ex, pk = ctx["ex"][li], ctx["packed"][li]["swap2"]
        for m in (5, 9):
            for crowded in (False, True):
                ids = [[(29*i + (0 if crowded else 17*t)) % NUM_EXPERTS
                        for i in range(TOP_K)] for t in range(m)]
                if m == 9 and crowded:
                    ids[-1] = [31 + 7*i for i in range(TOP_K)]
                topk_ids = torch.tensor(ids, dtype=torch.int32, device="cuda")
                _, kw = ex.prepare_humming_moe_kwargs(topk_ids, None, None)
                gen = torch.Generator(device="cuda")
                gen.manual_seed(3719 + 97*L + 13*m + int(crowded))
                u = torch.randn(m*TOP_K, INTER, generator=gen, device="cuda")
                keep = torch.rand(m*TOP_K, INTER, generator=gen, device="cuda") < 0.08
                sparse = (u.relu().square() * keep).to(torch.bfloat16).contiguous()
                sparse[:, :32] = 0
                sparse[0].zero_()
                sparse.view(torch.int16)[sparse == 0] = -32768
                signed = (u * keep).to(torch.bfloat16).contiguous()
                signed[:, 32:48] = 0
                signed[1].zero_()
                zero = torch.zeros_like(sparse)
                acts = (sparse, signed, zero)
                act = sparse.clone()
                vrows = torch.tensor([1 << 30], dtype=torch.int32, device="cuda") if m == 5 else None
                for ai, source in enumerate(acts):
                    act.copy_(source)
                    for control, candidate in ((113, 119), (118, 153), (113, 154), (118, 155), (118, 156), (118, 157), (154, 158)):
                        ref = call(control, act, kw, pk, torch.empty(
                            m*TOP_K, HIDDEN, dtype=torch.bfloat16, device="cuda"), None)
                        got = call(candidate, act, kw, pk, torch.empty_like(ref), None)
                        check(got, ref, f"L{L} M{m} crowded={crowded} act={ai} c{control}->c{candidate} ordinary")
                        for count in ((1, 2, 3, 4, 5, 1 << 30) if m == 5 else ()):
                            vrows.fill_(count)
                            ref = call(control, act, kw, pk, torch.empty_like(ref), vrows)
                            got = call(candidate, act, kw, pk, torch.empty_like(ref), vrows)
                            rows = min(count, 5) * TOP_K
                            check(got[:rows], ref[:rows],
                                  f"L{L} M{m} act={ai} c{control}->c{candidate} prefix={count} valid")
                            check(got[rows:], torch.zeros_like(got[rows:]),
                                  f"L{L} M{m} act={ai} c{control}->c{candidate} prefix={count} suffix")
                outs = {cfg: torch.empty(m*TOP_K, HIDDEN, dtype=torch.bfloat16, device="cuda")
                        for cfg in zskip.PAIR_CONFIGS}

                def calls():
                    for cfg, out in outs.items():
                        zskip.down(act, kw["sorted_ids"], kw["expert_ids"], kw["num_tokens_padded"], pk, out,
                                   mode=zskip.SKIP, config=cfg, max_blocks=m*TOP_K, stats=stats,
                                   verify_rows=vrows)

                calls()
                torch.cuda.synchronize()
                if zskip._pdl_on != expected_pdl:
                    raise RuntimeError(f"PDL fallback before graph capture: active={zskip._pdl_on}")
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    calls()
                for ai, source in enumerate(acts):
                    act.copy_(source)
                    for count in ((None, 1, 2, 3, 4, 5, 1 << 30) if m == 5 else (None,)):
                        if vrows is not None:
                            vrows.fill_(count if count is not None else 1 << 30)
                        for out in outs.values():
                            out.view(torch.int16).fill_(0x7FBF)
                        graph.replay()
                        replays += 1
                        rows = m*TOP_K if count is None else min(count, 5)*TOP_K
                        for control, candidate in ((113, 119), (118, 153), (113, 154), (118, 155), (118, 156), (118, 157), (154, 158)):
                            ref, got = outs[control], outs[candidate]
                            check(got[:rows], ref[:rows],
                                  f"L{L} M{m} act={ai} c{control}->c{candidate} replay={count} valid")
                            if count is not None:
                                check(got[rows:], torch.zeros_like(got[rows:]),
                                      f"L{L} M{m} act={ai} c{control}->c{candidate} replay={count} suffix")
                torch.cuda.synchronize()
                del graph
        torch.cuda.synchronize()
        log(f"prefix L{L}: ordinary/PDL/depth8/replay running mismatches={len(mismatches)}")
    if zskip._pdl_on != expected_pdl:
        raise RuntimeError(f"PDL fallback during prefix proof: active={zskip._pdl_on}")
    result = {"all_ok": not mismatches and stats.tolist()[1] == 0, "layers": len(ctx["layer_ids"]),
              "comparisons": checks, "elements": values, "graph_replays": replays,
              "pdl_requested": expected_pdl, "pdl_active": zskip._pdl_on,
              "pair_configs": {str(cfg): zskip.config_info(cfg) for cfg in zskip.PAIR_CONFIGS},
              "dense_redo": stats.tolist()[0], "oversize_blocks": stats.tolist()[1],
              "mismatches": mismatches}
    dump("prefix-pdl" if expected_pdl else "prefix-ordinary", result)
    if not result["all_ok"]:
        raise RuntimeError("paired prefix exactness failed")


# ------------------------------------------------------------------------------------------ time


def graph_time(fn, reps):
    return hme.graph_time(fn, reps)


def phase_time(args, ctx):
    zskip.init()
    prefix_counts = [int(v) for v in args.time_prefix.split(",")] if args.time_prefix else [None]
    if any(c is not None and (c < 1 or (c > 5 and c != 1 << 30)) for c in prefix_counts):
        raise ValueError("prefix timing counts must be 1..5 or 1<<30")
    for m in args.time_ms:
        if m != 5 and any(c is not None for c in prefix_counts):
            raise ValueError("prefix timing requires M=5")
        per_layer = []
        for li, L in enumerate(ctx["layer_ids"]):
            ex = ctx["ex"][li]
            cases = realistic_cases(ctx, li, [m], 1)
            if cases:
                pick = cases[(li * 3) % len(cases)]
            else:
                pick = synthetic_cases(ctx, li, [m], ["emb"], 1)[0]
            name, x, tid, tw = pick
            st = hme.run_stages(ex, x, tid, tw)
            kw1, kw2 = ex.prepare_humming_moe_kwargs(tid, None, None)
            ak, ag, nr, sc_ = block_masks(st["act"], kw2)
            per_layer.append({"ex": ex, "act": st["act"], "kw2": kw2, "packed": ctx["packed"][li],
                              "tid32": tid.to(torch.int32).contiguous(),
                              "out": torch.empty(m * TOP_K, HIDDEN, dtype=torch.bfloat16, device="cuda"),
                              "verify_rows": torch.tensor([1 << 30], dtype=torch.int32, device="cuda") if m == 5 else None,
                              "act_k": ak, "act_g": ag, "nrows": nr, "sect": sc_, "case": name})
        nb = sum(len(p["act_k"]) for p in per_layer)
        dense_bytes = nb * (INTER * HIDDEN // 2 + (INTER // 16) * HIDDEN)
        skip_bytes = sum(sum(k * (HIDDEN // 2) + g * HIDDEN for k, g in zip(p["act_k"], p["act_g"])) for p in per_layer)
        rs_bytes = sum(sum(q * 5376 + g * HIDDEN for q, g in zip(p["sect"], p["act_g"])) for p in per_layer)
        fk = 1 - sum(sum(p["act_k"]) for p in per_layer) / (nb * INTER)
        fg = 1 - sum(sum(p["act_g"]) for p in per_layer) / (nb * (INTER // 16))
        Lc = len(per_layer)
        info = {"m": m, "layers": Lc, "blocks_per_layer": round(nb / Lc, 2), "f_k_skipped": round(fk, 4),
                "f_group_skipped": round(fg, 4), "dense_MB_per_call": round(dense_bytes / Lc / 1e6, 2),
                "skip_MB_per_call": round(skip_bytes / Lc / 1e6, 2),
                "routed_sparse_MB_per_call": round(rs_bytes / Lc / 1e6, 2),
                "rows_hist": {r: sum(p["nrows"].count(r) for p in per_layer) for r in range(1, 5)}}
        log(f"time M={m} inputs: {info}")

        def stock_fn():
            from vllm.utils.humming import HummingMethod

            for p in per_layer:
                HummingMethod.forward_layer(layer=p["ex"].layer, inputs=p["act"], input_scale=None,
                                            outputs=p["out"], sublayer_name="w2", **p["kw2"])

        def mk(mode, cfg, lay):
            def fn():
                for p in per_layer:
                    zskip.down(p["act"], p["kw2"]["sorted_ids"], p["kw2"]["expert_ids"],
                               p["kw2"]["num_tokens_padded"], p["packed"][lay], p["out"], mode=mode,
                               config=cfg, verify_rows=p["verify_rows"] if prefix_count is not None else None)
            return fn

        variants = [("stock", stock_fn)]
        if ctx.get("rs") is not None:
            rs_calls = [(p["act"], p["tid32"], p["out"], ctx["rs"][li]) for li, p in enumerate(per_layer)]

            def rs_fn():
                for a_, i_, o_, pk_ in rs_calls:
                    assert routed_sparse.forward(a_, i_, o_, pk_)

            variants.append(("routed_sparse", rs_fn))
        configs = list(zskip.CONFIGS) + list(zskip.CONFIGS4) + ctx.get("pair_configs", []) if args.time_configs == "all" else [
            int(c) for c in args.time_configs.split(",")]
        for lay in ctx["packed"][0]:
            for cfg in configs:
                if cfg not in configs_for_layout(ctx, lay):
                    continue
                for mode, mn in ((zskip.SKIP, "skip"), (zskip.LOADONLY, "loadonly"), (zskip.SETUPONLY, "setuponly"),
                                 (zskip.DENSE, "dense"), (zskip.LOADONLY_DENSE, "loadonly_dense")):
                    if mn in ("dense", "loadonly_dense") and cfg not in (1, 101) and not args.time_all_dense:
                        continue
                    if mn in ("loadonly", "setuponly") and not args.time_loadonly:
                        continue
                    variants.append((f"{mn}/{lay}/c{cfg}", mk(mode, cfg, lay)))
        for prefix_count in prefix_counts:
            if prefix_count is not None:
                for p in per_layer:
                    p["verify_rows"].fill_(prefix_count)
            times = {k: [] for k, _ in variants}
            for round_index in range(args.rounds):
                ordered = variants if round_index % 2 == 0 else reversed(variants)
                for k, fn in ordered:
                    times[k] += graph_time(fn, args.time_reps)
            rows = []
            for k, _ in variants:
                us = statistics.median(times[k]) * 1000 / Lc
                mode = k.split("/")[0]
                byt = (dense_bytes if mode in ("stock", "dense", "loadonly_dense")
                       else rs_bytes if mode == "routed_sparse" else skip_bytes)
                rows.append({"variant": k, "us_per_call": round(us, 2),
                             "GBps": None if prefix_count is not None else round(byt / Lc / (us * 1e-6) / 1e9, 1),
                             "min_us": round(min(times[k]) * 1000 / Lc, 2)})
            rows.sort(key=lambda r: r["us_per_call"])
            stock_us = next(r["us_per_call"] for r in rows if r["variant"] == "stock")
            for r in rows:
                r["speedup_vs_stock"] = round(stock_us / r["us_per_call"], 3)
                bandwidth = f"{r['GBps']:6.1f}" if r["GBps"] is not None else "   n/a"
                log(f"   M={m} {r['variant']:28s} {r['us_per_call']:8.2f} us  {bandwidth} GB/s  x{r['speedup_vs_stock']}")
            suffix = "" if prefix_count is None else f"-prefix{prefix_count}"
            dump(f"time-M{m}{suffix}", {"info": info, "verify_rows": prefix_count,
                                       "rows": rows, "times_ms": times})


# ------------------------------------------------------------------------------------------ main


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="probe,epi,exact,time")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--ms", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--kinds", default="emb,rand,rand_uniform_route,edge")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--capture", default="", help="'' (none), 'all' or comma-separated sequence names")
    ap.add_argument("--layouts", default="rows,swap")
    ap.add_argument("--pair", action="store_true", help="also check/time paired configs 113/119/154 and fused 118/153/155")
    ap.add_argument("--time-ms", default="4")
    ap.add_argument("--time-configs", default="all")
    ap.add_argument("--time-prefix", default=None, help="comma-separated M5 verify-row counts: 1..5 or 1<<30")
    ap.add_argument("--time-all-dense", action="store_true")
    ap.add_argument("--time-loadonly", action="store_true")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--time-reps", type=int, default=20)
    ap.add_argument("--probe-blocks", type=int, default=2048)
    ap.add_argument("--probe-iters", type=int, default=128)
    ap.add_argument("--probe-launches", type=int, default=8)
    ap.add_argument("--epi-gs", default="4.0,0.0009765625,3.1e-05,1.7")
    ap.add_argument("--out", default=os.environ.get("NEMO_OUT", "."))
    args = ap.parse_args()
    args.ms = [int(v) for v in args.ms.split(",")]
    args.time_ms = [int(v) for v in args.time_ms.split(",")]
    args.kinds = [k for k in args.kinds.split(",") if k]
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    os.environ.setdefault("VLLM_HUMMING_BATCH_INVARIANT", "1")
    log(f"args: {vars(args)}")
    assert zskip.available(), zskip.load_error()
    torch.cuda.set_device(0)
    for cfg in list(zskip.CONFIGS) + list(zskip.CONFIGS4):
        try:
            log(f"zskip config {cfg}: {zskip.config_info(cfg)}")
        except ValueError as exc:
            log(f"optional config {cfg} unavailable: {exc}")
    log(f"routed_sparse available: {routed_sparse is not None}")
    phases = args.phase.split(",")
    ctx = {}
    if "prefix" in phases:
        args.pair = True
    if args.pair:
        assert "swap" in args.layouts.split(","), "--pair needs the original swap layout"
        ctx["pair_configs"] = []
        for cfg in zskip.PAIR_CONFIGS:
            try:
                log(f"pair config {cfg}: {zskip.config_info(cfg)}")
                ctx["pair_configs"].append(cfg)
            except ValueError as exc:
                log(f"pair config {cfg} unavailable: {exc}")
        assert ctx["pair_configs"], "no paired config built"
    if "probe" in phases:
        phase_probe(args, ctx)
    if "capture" in phases or args.capture:
        if not args.capture:
            args.capture = "all"
        phase_capture(args, ctx)
    if "exact" in phases or "time" in phases or "prefix" in phases:
        from vllm.model_executor.layers.fused_moe.experts import humming_tuned_moe  # noqa

        snap = hme.Snapshot(os.environ["NEMO_SNAPSHOT"])
        ctx["emb"] = snap.load(["backbone.embeddings.weight"], "cuda")["backbone.embeddings.weight"]
        ctx["layer_ids"] = MOE_LAYERS if args.layers == "all" else [int(v) for v in args.layers.split(",")]
        ctx["ex"], ctx["routers"], ctx["packed"] = [], [], []
        gs_values = []
        for L in ctx["layer_ids"]:
            t0 = time.time()
            raw = snap.raw_experts(L, "cuda")
            ex = hme.build_variant(raw, f"backbone.layers.{L}.mixer.experts", True)
            hgs = ex.layer.w2_global_scale.detach().float().reshape(-1)
            assert torch.equal(hgs, raw["w2_weight_scale_2"].float().reshape(-1)), "humming w2 global scale != raw"
            pk = {lay: zskip.repack(raw["w2_weight"], raw["w2_weight_scale"], hgs, lay)
                  for lay in args.layouts.split(",")}
            if args.pair:
                pk["swap2"] = zskip.to_swap2(pk["swap"])
            gs_values += hgs[:2].tolist()
            if routed_sparse is not None:
                ctx.setdefault("rs", []).append(routed_sparse.prepare(
                    raw["w2_weight"].contiguous(), raw["w2_weight_scale"].contiguous(),
                    raw["w2_weight_scale_2"].float().reshape(-1).contiguous()))
            del raw
            torch.cuda.empty_cache()
            ctx["ex"].append(ex)
            ctx["packed"].append(pk)
            ctx["routers"].append(snap.router(L, "cuda"))
            log(f"layer {L}: built in {time.time() - t0:.1f}s; packed {sum(p.nbytes() for p in pk.values()) / 1e6:.0f} MB; "
                f"cuda allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
        ctx["gs_values"] = gs_values
    if "epi" in phases:
        phase_epi(args, ctx)
    if "exact" in phases:
        phase_exact(args, ctx)
    if ctx.get("exact_failed"):
        raise SystemExit("zskip exactness failed; see exact.json")
    if "prefix" in phases:
        phase_prefix(args, ctx)
    if "time" in phases:
        phase_time(args, ctx)
    log("done")
    if ctx.get("exact_failed"):
        sys.exit(3)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""shco step 1: can the shared expert hide under humming's DRAM-bound routed up at M = 496?

Runs in the pinned image: bin/atom nemo exec SHA harness/kernels/shco/tests/step1.py [--prompt public]
(tree overlay + $KERNELS_OUT on PYTHONPATH, snapshot at $NEMO_SNAPSHOT, outputs to $NEMO_OUT).

1. capture: eager in-process engine, prefill of the 496-token prompt; records each MoE layer's
   input x (23 x [496, 2688] bf16).
2. per MoE layer: humming routed up with the fused relu2 epilogue exactly as the engine's prefill
   calls it (tuned prefill config, real routing; layer 51 takes row 495's experts for every row as
   the prefill graph does), and the stock shared expert (Marlin up -> relu2 -> Marlin down).
3. CUDA graphs over the 23 layers (cold weights), timed by replay:
   (a) up alone, for the engine's config and for footprint variants with the same accumulation
       chain (block_k = warp_k = 32, no stream-K): num_sms 40/36/24 (fewer resident CTAs per SM,
       same <= 128-register launch bound) and num_ctas_per_sm 3/2;
   (b) stock shared expert alone;
   (c) up || stock shared expert on two streams (both capture orders);
   (d) up || dummy compute-bound MMA kernel with the shared expert's FLOP count
       (csrc/shco.cu modes 0/1/2; 64-128 threads, <= 16 KB smem, 48-192 CTAs).
4. torch.profiler: resources of every kernel (grid, block, registers, smem, occupancy) and the
   per-kernel durations inside selected concurrent graphs (the up's slowdown).
Writes step1.json and the traces to $NEMO_OUT.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.environ.get("NEMO_TREE", os.path.abspath(os.path.join(HERE, "..", "..", "..", "..")))
sys.path.insert(0, os.path.join(TREE, "harness", "tests"))

import humming_moe_exact as hme  # noqa: E402
import shco  # noqa: E402

MOE_LAYERS = hme.MOE_LAYERS
HIDDEN, INTER, TOP_K = hme.HIDDEN, hme.INTER, hme.TOP_K
SH_INTER = 3712
M = 496
BIG = 1 << 30
T0 = time.time()
OUT = "."


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def dump(name, obj):
    path = os.path.join(OUT, f"{name}.json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    log(f"wrote {path}")


# ------------------------------------------------------------------------------------ capture


def capture(prompt_names):
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import vllm.model_executor.models.nemotron_h as nh
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    with open(os.path.join(TREE, "harness", "kernels", "zskip", "tests", "sequences.json")) as f:
        seqs = {s["name"]: s for s in json.load(f)["sequences"]}
    cap = {"on": False, "x": {}}
    orig_init, orig_fwd = nh.NemotronHMoE.__init__, nh.NemotronHMoE.forward

    def init(self, *a, **kw):
        orig_init(self, *a, **kw)
        prefix = kw.get("prefix", "")
        self._shco_layer = int(prefix.split("layers.")[1].split(".")[0]) if "layers." in prefix else -1

    def fwd(self, hidden_states):
        if cap["on"]:
            cap["x"].setdefault(self._shco_layer, []).append(hidden_states.detach().to("cpu", copy=True))
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
    caps = {}
    for name in prompt_names:
        ids = seqs[name]["prompt"]
        assert len(ids) == M, (name, len(ids))
        cap["x"].clear()
        cap["on"] = True
        llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
        cap["on"] = False
        layers = sorted(cap["x"])
        assert layers == MOE_LAYERS, layers
        caps[name] = {L: torch.cat(cap["x"][L])[:M] for L in layers}
        assert all(v.shape == (M, HIDDEN) for v in caps[name].values())
        log(f"capture {name}: {len(ids)} tokens, calls/layer {len(cap['x'][layers[0]])}")
    nh.NemotronHMoE.__init__, nh.NemotronHMoE.forward = orig_init, orig_fwd
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    log(f"capture done; cuda allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    return caps


# ------------------------------------------------------------------------------------- layers


def load_linear(snap, wmap, prefix):
    """The MarlinNvFp4LinearKernel load path (as harness/kernels/marlinx/tests/marlinx_exact.py)."""
    from safetensors import safe_open

    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_fp4_layer_for_marlin

    fn = os.path.join(snap, wmap[prefix + ".weight"])
    with safe_open(fn, "pt", device="cpu") as f:
        w = f.get_tensor(prefix + ".weight")
        s = f.get_tensor(prefix + ".weight_scale")
        s2 = f.get_tensor(prefix + ".weight_scale_2")
    layer = torch.nn.Module()
    n, k2 = w.shape
    layer.output_size_per_partition = n
    layer.input_size_per_partition = 2 * k2
    layer.params_dtype = torch.bfloat16
    layer.weight = torch.nn.Parameter(w.cuda(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(s.cuda(), requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(s2.cuda().max().to(torch.float32), requires_grad=False)
    prepare_fp4_layer_for_marlin(layer)
    layer.name = prefix
    return layer


class Layer:
    pass


def build_layers(caps_x, layer_ids):
    from vllm.model_executor.layers.fused_moe.experts import humming_tuned_moe  # noqa: F401

    snap_dir = os.environ["NEMO_SNAPSHOT"]
    snap = hme.Snapshot(snap_dir)
    with open(os.path.join(snap_dir, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    out = []
    for L in layer_ids:
        t0 = time.time()
        e = Layer()
        e.L = L
        raw = snap.raw_experts(L, "cuda")
        e.ex = hme.build_variant(raw, f"backbone.layers.{L}.mixer.experts", True)
        del raw
        # the w2 weights are not used here
        for name in [n for n, _ in e.ex.layer.named_parameters() if n.startswith("w2_")]:
            setattr(e.ex.layer, name, torch.nn.Parameter(torch.empty(0, device="cuda"), requires_grad=False))
        torch.cuda.empty_cache()
        gate_w, bias, _ = snap.router(L, "cuda")
        e.x = caps_x[L].to("cuda").contiguous()
        ids, w = hme.route(e.x, gate_w, bias)
        if L == layer_ids[-1] and L == MOE_LAYERS[-1]:
            # prefill graph: the last MoE layer routes every row to row 495's experts
            ids = ids[M - 1:M].expand(M, TOP_K).contiguous()
            w = w[M - 1:M].expand(M, TOP_K).contiguous()
        e.ids, e.w = ids, w
        e.kw1, _ = e.ex.prepare_humming_moe_kwargs(ids, None, None)
        e.act = torch.empty(M * TOP_K, INTER, dtype=torch.bfloat16, device="cuda")
        p = f"backbone.layers.{L}.mixer.shared_experts"
        e.up = load_linear(snap_dir, wmap, p + ".up_proj")
        e.dn = load_linear(snap_dir, wmap, p + ".down_proj")
        e.sh_a = torch.empty(M, SH_INTER, dtype=torch.bfloat16, device="cuda")
        e.wbuf = torch.cat([e.up.weight.detach().reshape(-1).view(torch.uint8),
                            e.dn.weight.detach().reshape(-1).view(torch.uint8)]).contiguous()
        e.n_blocks = int(e.kw1["num_tokens_padded"].item()) // e.ex._moe_block_size
        out.append(e)
        log(f"layer {L}: built in {time.time() - t0:.1f}s; m-blocks {e.n_blocks}; experts used "
            f"{int(torch.unique(ids).numel())}; cuda allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    return out


# --------------------------------------------------------------------------------------- ops


def tuning_string(ex, ctas=None, sms=None, stages=None):
    """The engine's prefill w13 config (the range holding valid_shape_m = 2976), optionally with
    a different persistent grid / stage count; one entry so each variant compiles one kernel."""
    vm = M * TOP_K
    for lo, hi, cfg in ex.w13_tuning_config:
        if lo < vm <= hi:
            c = copy.deepcopy(cfg)
            break
    else:
        raise AssertionError("no tuned w13 entry for M=496")
    assert c["block_shape"][2] == c["warp_shape"][2] == 32 and not c["use_stream_k"]
    if ctas is not None:
        c["num_ctas_per_sm"] = ctas
    if sms is not None:
        c["num_sms"] = sms
    if stages is not None:
        c["num_stages"] = stages
    return json.dumps([[0, BIG, c]]), c


def up_call(e, tstr):
    from vllm.model_executor.layers.fused_moe.experts.humming_relu2_epilogue import forward_relu2

    kw = dict(e.kw1)
    kw["tuning_config"] = tstr
    forward_relu2(layer=e.ex.layer, inputs=e.x, outputs=e.act, **kw)


def sh_call(e):
    from vllm.model_executor.layers.fused_moe.experts.humming_tuned_moe import relu2_bf16
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import apply_fp4_marlin_linear

    h = apply_fp4_marlin_linear(e.x, e.up.weight, e.up.weight_scale, e.up.weight_global_scale, e.up.workspace,
                                SH_INTER, HIDDEN)
    assert relu2_bf16(h, e.sh_a)
    return apply_fp4_marlin_linear(e.sh_a, e.dn.weight, e.dn.weight_scale, e.dn.weight_global_scale,
                                   e.dn.workspace, HIDDEN, SH_INTER)


SEED = None
SINK = None


def dummy_call(e, d):
    mode, threads, grid = d
    steps = shco.steps_for(grid, threads)
    ld_every = 1
    if mode == 2:
        wvecs = e.wbuf.numel() // 16
        nlanes = grid * threads
        ld_every = max(1, (steps * nlanes) // max(wvecs, 1))
    shco.dummy(mode, grid, threads, steps, SEED, SINK, wbuf=e.wbuf if mode == 2 else None, ld_every=ld_every)


# ------------------------------------------------------------------------------------ graphs

S_CAP = None
S_AUX = None


def make_graph(layers, main_fn, aux_fn=None, aux_first=True):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=S_CAP):
        cur = torch.cuda.current_stream()
        for e in layers:
            if aux_fn is None:
                main_fn(e)
                continue
            S_AUX.wait_stream(cur)
            if aux_first:
                with torch.cuda.stream(S_AUX):
                    aux_fn(e)
                main_fn(e)
            else:
                main_fn(e)
                with torch.cuda.stream(S_AUX):
                    aux_fn(e)
            cur.wait_stream(S_AUX)
    return g


def replay_ms(g, n):
    ts = []
    for _ in range(n):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return ts


# ------------------------------------------------------------------------------------ profile


def kernel_events(path):
    with open(path) as f:
        d = json.load(f)
    return [e for e in d.get("traceEvents", []) if e.get("cat") == "kernel"]


def kind_of(name):
    n = name.lower()
    if "humming" in n:
        return "up"
    if "dummy_kernel" in n:
        return "dummy"
    if "marlin" in n:
        return "marlin"
    if "relu2" in n:
        return "relu2"
    return "other"


def resources(evs):
    rows = {}
    for e in evs:
        a = e.get("args", {})
        key = (kind_of(e["name"]), str(a.get("grid")), str(a.get("block")))
        r = rows.setdefault(key, {"kind": key[0], "name": e["name"][:160], "grid": a.get("grid"),
                                  "block": a.get("block"), "regs": a.get("registers per thread"),
                                  "smem": a.get("shared memory"), "blocks_per_sm": a.get("blocks per SM"),
                                  "warps_per_sm": a.get("warps per SM"),
                                  "occupancy_pct": a.get("est. achieved occupancy %"), "durs": []})
        r["durs"].append(e.get("dur"))
    for r in rows.values():
        r["dur_median_us"] = statistics.median(r.pop("durs"))
    return list(rows.values())


# --------------------------------------------------------------------------------------- main


def main():
    global OUT, SEED, SINK, S_CAP, S_AUX
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="public")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--hvars", default="main,s40,s36,s24,c3,c2")
    ap.add_argument("--dummies", default="0:128:48,0:128:96,0:64:48,0:64:96,0:64:192,1:128:48,1:64:96,"
                                         "2:128:48,2:128:96,2:64:96,2:64:48")
    ap.add_argument("--calib", default="0:128:192,0:128:384,1:128:192,2:128:192")
    ap.add_argument("--main-first", default="main,s36")
    ap.add_argument("--out", default=os.environ.get("NEMO_OUT", "."))
    args = ap.parse_args()
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    os.environ.setdefault("VLLM_HUMMING_BATCH_INVARIANT", "1")
    log(f"args: {vars(args)}")
    assert shco.available(), shco.load_error()
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    dev = {k: getattr(props, k, None) for k in (
        "name", "multi_processor_count", "max_threads_per_multi_processor", "regs_per_multiprocessor",
        "shared_memory_per_multiprocessor", "shared_memory_per_block", "shared_memory_per_block_optin",
        "L2_cache_size", "major", "minor")}
    log(f"device: {dev}")
    res = {"device": dev, "args": vars(args)}
    res["dummy_attrs"] = {}
    for mode in (0, 1, 2):
        at = shco.dummy_attrs(mode)
        at["max_ctas_per_sm_128"] = shco.dummy_occupancy(mode, 128)
        at["max_ctas_per_sm_64"] = shco.dummy_occupancy(mode, 64)
        res["dummy_attrs"][mode] = at
        log(f"dummy mode {mode}: {at}")

    caps = capture([args.prompt])
    layers = build_layers(caps[args.prompt], MOE_LAYERS)
    del caps
    gc.collect()

    SEED = torch.randint(-(2**31), 2**31 - 1, (4096,), dtype=torch.int32, device="cuda")
    SINK = torch.zeros(1 << 16, dtype=torch.float32, device="cuda")
    S_CAP = torch.cuda.Stream()
    S_AUX = torch.cuda.Stream()

    hdefs = {"main": {}, "s40": {"sms": 40}, "s36": {"sms": 36}, "s24": {"sms": 24},
             "c3": {"ctas": 3}, "c2": {"ctas": 2}, "st2": {"stages": 2}, "s36st2": {"sms": 36, "stages": 2}}
    hvars = {}
    for h in args.hvars.split(","):
        tstr, cfg = tuning_string(layers[0].ex, **hdefs[h])
        hvars[h] = tstr
        log(f"humming variant {h}: {cfg}")
    res["hvars"] = {h: json.loads(s)[0][2] for h, s in hvars.items()}
    dummies = [tuple(int(v) for v in d.split(":")) for d in args.dummies.split(",") if d]
    calibs = [tuple(int(v) for v in d.split(":")) for d in args.calib.split(",") if d]

    # eager warm-up (JIT compiles) and the sanity check: every up variant equals the engine's config
    ref_act = []
    for e in layers[:2] + layers[-1:]:
        up_call(e, hvars["main"])
        ref_act.append(e.act.clone())
    sanity = {}
    for h, tstr in hvars.items():
        bad = 0
        for e, r in zip(layers[:2] + layers[-1:], ref_act):
            e.act.fill_(0)
            up_call(e, tstr)
            torch.cuda.synchronize()
            bad += int((e.act.view(torch.int16) != r.view(torch.int16)).sum())
        sanity[h] = bad
        log(f"up variant {h}: {bad} differing outputs vs the engine config (3 layers)")
    res["up_variant_mismatches"] = sanity
    del ref_act
    for e in layers[:1]:
        sh_call(e)
        for d in dummies + calibs:
            dummy_call(e, d)
    torch.cuda.synchronize()

    # graphs
    specs = {}
    for h, tstr in hvars.items():
        specs[f"up:{h}"] = (lambda e, t=tstr: up_call(e, t), None, True)
    specs["sh"] = (sh_call, None, True)
    for d in dummies + calibs:
        specs[f"dm:{d[0]}:{d[1]}:{d[2]}"] = (lambda e, d=d: dummy_call(e, d), None, True)
    for h in ("main", "s36", "c3"):
        if h in hvars:
            for af in (True, False):
                specs[f"up:{h}||sh:{'auxfirst' if af else 'mainfirst'}"] = (
                    lambda e, t=hvars[h]: up_call(e, t), sh_call, af)
    mf = set(args.main_first.split(","))
    for h, tstr in hvars.items():
        for d in dummies:
            for af in ((True, False) if h in mf else (True,)):
                specs[f"up:{h}||dm:{d[0]}:{d[1]}:{d[2]}:{'auxfirst' if af else 'mainfirst'}"] = (
                    lambda e, t=tstr: up_call(e, t), lambda e, d=d: dummy_call(e, d), af)
    log(f"capturing {len(specs)} graphs")
    graphs = {}
    for name, (mfn, afn, af) in specs.items():
        graphs[name] = make_graph(layers, mfn, afn, af)
    torch.cuda.synchronize()
    log("captured")
    times = {k: [] for k in graphs}
    for r in range(args.rounds):
        for name, g in graphs.items():
            g.replay()  # warm (host side / clocks)
            times[name] += replay_ms(g, args.reps)
        log(f"round {r + 1}/{args.rounds} done")
    med = {k: statistics.median(v) for k, v in times.items()}
    spread = {k: (min(v), max(v)) for k, v in times.items()}
    res["graph_ms"] = med
    res["graph_ms_minmax"] = spread
    for k in graphs:
        log(f"  {k:48s} {med[k]:8.3f} ms  [{spread[k][0]:.3f}, {spread[k][1]:.3f}]  per layer {med[k] / len(layers) * 1e3:7.1f} us")

    # analysis
    up_main = med["up:main"]
    sh = med["sh"]
    base = up_main + sh
    rows = []
    for k, v in med.items():
        if "||" not in k:
            continue
        hk, ak = k.split("||")
        h = hk.split(":")[1]
        aux_alone = sh if ak.startswith("sh") else med[":".join(ak.split(":")[:4])]
        up_h = med[f"up:{h}"]
        rows.append({
            "graph": k, "conc_ms": v, "up_h_ms": up_h, "aux_alone_ms": aux_alone,
            "hidden_vs_up_main_ms": up_main + aux_alone - v,
            "hidden_vs_up_h_ms": up_h + aux_alone - v,
            "saving_vs_serial_stock_ms": base - v,
            "conc_minus_up_main_ms": v - up_main,
        })
    rows.sort(key=lambda r: -r["saving_vs_serial_stock_ms"])
    res["concurrency"] = rows
    log(f"serial stock: up {up_main:.3f} + shared {sh:.3f} = {base:.3f} ms per prefill (23 layers)")
    log("concurrency (ms per prefill): graph | conc | up_h | aux alone | hidden(net of up slowdown vs engine up) "
        "| saving vs serial stock")
    for r in rows:
        log(f"  {r['graph']:48s} {r['conc_ms']:8.3f} {r['up_h_ms']:8.3f} {r['aux_alone_ms']:8.3f} "
            f"{r['hidden_vs_up_main_ms']:7.3f} {r['saving_vs_serial_stock_ms']:7.3f}")
    dump("step1", res)

    # profiler: resources (eager) and per-kernel durations inside selected concurrent graphs
    from torch.profiler import ProfilerActivity, profile

    e0 = layers[3]
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for h, tstr in hvars.items():
            up_call(e0, tstr)
        sh_call(e0)
        for d in dummies + calibs:
            dummy_call(e0, d)
        torch.cuda.synchronize()
    path = os.path.join(OUT, "prof_eager.json")
    prof.export_chrome_trace(path)
    res["resources"] = resources(kernel_events(path))
    for r in res["resources"]:
        log(f"  res {r['kind']:6s} grid {r['grid']} block {r['block']} regs {r['regs']} smem {r['smem']} "
            f"blocks/SM {r['blocks_per_sm']} warps/SM {r['warps_per_sm']} occ {r['occupancy_pct']} "
            f"dur {r['dur_median_us']} us  {r['name'][:70]}")
    sel = ["up:main", "sh", "up:main||sh:auxfirst", "up:main||sh:mainfirst"]
    sel += [r["graph"] for r in rows[:6] if r["graph"] not in sel]
    sel += [k for k in graphs if k.startswith("up:") and "||" not in k and k not in sel]
    prof_rows = {}
    for name in sel:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            graphs[name].replay()
            torch.cuda.synchronize()
        path = os.path.join(OUT, "prof_" + name.replace(":", "_").replace("|", "I") + ".json")
        prof.export_chrome_trace(path)
        evs = kernel_events(path)
        by = {}
        for ev in evs:
            by.setdefault(kind_of(ev["name"]), []).append(ev["dur"])
        span = (max(ev["ts"] + ev["dur"] for ev in evs) - min(ev["ts"] for ev in evs)) if evs else 0
        prof_rows[name] = {"span_us": span, **{k: {"n": len(v), "sum_us": sum(v), "median_us": statistics.median(v)}
                                              for k, v in by.items()}}
        log(f"  prof {name}: span {span / 1e3:.3f} ms; " + "; ".join(
            f"{k} n={len(v)} sum={sum(v) / 1e3:.3f} ms med={statistics.median(v):.1f} us" for k, v in by.items()))
    res["profiles"] = prof_rows
    dump("step1", res)
    log("done")


if __name__ == "__main__":
    main()

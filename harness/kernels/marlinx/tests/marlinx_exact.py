"""Bit-identity and speed of marlinx against stock Marlin (vLLM v0.27.1 ops.marlin_gemm).

Runs inside the pinned vLLM image (bin/atom nemo exec SHA harness/kernels/marlinx/tests/marlinx_exact.py)
with the staged overlay and $KERNELS_OUT on PYTHONPATH and the snapshot at $NEMO_SNAPSHOT.

The layers come from the snapshot through vLLM's own prepare_fp4_layer_for_marlin (the
MarlinNvFp4LinearKernel load path): the 23 target shared experts (up 2688->3712, down
3712->2688) and lm_head (2688->131072). For every layer and M it compares
marlinx.maybe_apply with ops.marlin_gemm (use_fp32_reduce=True, no atomics) using
torch.equal on the bf16 outputs, then times both in CUDA graphs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

from safetensors import safe_open

import marlinx
import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_fp4_layer_for_marlin,
)
from vllm.scalar_type import scalar_types

TIMING_DEPTHS = (8, 4)
OVERLAP_DEPTHS = (8,)
SHARED_LAYERS = [1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31, 34, 36, 38, 40, 43, 45, 47, 49, 51]


def log(*args):
    print(*args, flush=True)


def load_linear(snap: str, wmap: dict, prefix: str) -> torch.nn.Module:
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
    # ModelOptNvFp4W4A16LinearMethod.process_weights_after_loading
    layer.weight_global_scale = torch.nn.Parameter(s2.cuda().max().to(torch.float32), requires_grad=False)
    prepare_fp4_layer_for_marlin(layer)
    layer.name = prefix
    return layer


def stock(layer, x: torch.Tensor) -> torch.Tensor:
    return ops.marlin_gemm(
        a=x, c=None, b_q_weight=layer.weight, b_bias=None, b_scales=layer.weight_scale,
        a_scales=None, global_scale=layer.weight_global_scale, b_zeros=None, g_idx=None,
        perm=None, workspace=layer.workspace, b_q_type=scalar_types.float4_e2m1f,
        size_m=x.size(0), size_n=layer.output_size_per_partition,
        size_k=layer.input_size_per_partition, use_atomic_add=False, use_fp32_reduce=True,
    )


def mine(layer, x: torch.Tensor, depth: int = marlinx.DEPTH) -> torch.Tensor:
    return marlinx.gemm(x, layer.weight, layer.weight_scale, layer.weight_global_scale,
                        layer.output_size_per_partition, layer.input_size_per_partition, depth)


def relu2(h: torch.Tensor) -> torch.Tensor:
    # what Inductor's triton_poi_fused_pow_relu computes: fp32 relu, square, one RN to bf16
    r = torch.relu(h.float())
    return (r * r).to(torch.bfloat16)


def gen_x(m: int, k: int, kind: int, gen: torch.Generator) -> torch.Tensor:
    x = torch.randn(m, k, generator=gen, device="cuda", dtype=torch.float32)
    if kind == 1:
        x = x * 4.0
        idx = torch.randint(0, k, (m, 8), generator=gen, device="cuda")
        x.scatter_(1, idx, torch.randn(m, 8, generator=gen, device="cuda") * 60.0)
    elif kind == 2:
        x = x * 0.05
    return x.to(torch.bfloat16)


def diff_info(layer, a: torch.Tensor, b: torch.Tensor) -> dict:
    bad = (a.view(torch.int16) != b.view(torch.int16))
    cols = bad.any(0).nonzero().flatten()
    tiles = sorted(set((cols // 128).tolist()))
    return {
        "elements": int(bad.sum()),
        "rows": bad.any(1).nonzero().flatten().tolist(),
        "tiles": tiles[:40],
        "n_tiles_bad": len(tiles),
        "max_abs": float((a.float() - b.float()).abs().max()),
        "first_cols": cols[:16].tolist(),
    }


def exactness(shared, lm_head, ms, kinds, results, depth):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(1234)
    total = 0
    bad_cases = []
    per = {}
    for li, (up, down) in zip(SHARED_LAYERS, shared):
        for m in ms:
            for kind in kinds:
                x = gen_x(m, up.input_size_per_partition, kind, gen)
                h_ref = stock(up, x)
                h_new = mine(up, x, depth)
                ok_up = torch.equal(h_ref, h_new)
                a = relu2(h_ref)
                d_ref = stock(down, a)
                d_new = mine(down, a, depth)
                ok_down = torch.equal(d_ref, d_new)
                # a second down input: random non-negative with a heavy tail
                a2 = (gen_x(m, down.input_size_per_partition, kind, gen).float().abs() ** 2).to(torch.bfloat16)
                ok_down2 = torch.equal(stock(down, a2), mine(down, a2, depth))
                total += 3
                for name, ok, ref, new in (("up", ok_up, h_ref, h_new), ("down", ok_down, d_ref, d_new),
                                           ("down_rand", ok_down2, None, None)):
                    key = f"L{li}.{name}.M{m}"
                    per[key] = per.get(key, 0) + (0 if ok else 1)
                    if not ok and len(bad_cases) < 12:
                        info = {"case": key, "kind": kind}
                        if ref is not None:
                            info.update(diff_info(up if name == "up" else down, ref, new))
                        bad_cases.append(info)
    for m in ms:
        for kind in kinds:
            x = gen_x(m, lm_head.input_size_per_partition, kind, gen)
            ref = stock(lm_head, x)
            new = mine(lm_head, x, depth)
            ok = torch.equal(ref, new)
            total += 1
            key = f"lm_head.M{m}"
            per[key] = per.get(key, 0) + (0 if ok else 1)
            if not ok and len(bad_cases) < 24:
                info = {"case": key, "kind": kind}
                info.update(diff_info(lm_head, ref, new))
                bad_cases.append(info)
    # strided input (lda = K + 64), depth variants, determinism
    up, down = shared[0]
    base = gen_x(4, up.input_size_per_partition + 64, 0, gen)
    xs = base[:, : up.input_size_per_partition]
    extra = {
        "strided_up_M4": torch.equal(stock(up, xs.contiguous()), mine(up, xs, depth)),
        "maybe_apply_up_M4": torch.equal(stock(up, xs.contiguous()), marlinx.maybe_apply(up, xs)),
    }
    for dv in (4,):
        for mm in (1, 4, 8):
            x = gen_x(mm, up.input_size_per_partition, 1, gen)
            extra[f"depth{dv}_up_M{mm}"] = torch.equal(stock(up, x), mine(up, x, dv))
            a = relu2(stock(up, x))
            extra[f"depth{dv}_down_M{mm}"] = torch.equal(stock(down, a), mine(down, a, dv))
            x = gen_x(mm, lm_head.input_size_per_partition, 0, gen)
            extra[f"depth{dv}_lm_head_M{mm}"] = torch.equal(stock(lm_head, x), mine(lm_head, x, dv))
    x = gen_x(4, lm_head.input_size_per_partition, 0, gen)
    extra["deterministic_lm_head_M4"] = all(torch.equal(mine(lm_head, x, depth), mine(lm_head, x, depth)) for _ in range(3))
    # row invariance: the M=4 output rows equal the M=1 outputs of each row
    x4 = gen_x(4, up.input_size_per_partition, 0, gen)
    full = mine(up, x4, depth)
    extra["row_invariant_up"] = all(torch.equal(full[i : i + 1], stock(up, x4[i : i + 1].contiguous())) for i in range(4))
    fails = sum(per.values())
    results[f"exactness_depth{depth}"] = {
        "cases": total,
        "failing_cases": fails,
        "per_case_failures": {k: v for k, v in per.items() if v},
        "bad_examples": bad_cases,
        "extra": extra,
        "ms": ms,
        "kinds": kinds,
    }
    log(f"exactness depth {depth}: {total - fails}/{total} cases bit-identical; "
        f"extra all={all(extra.values())} failing={[k for k, v in extra.items() if not v]}")
    for b in bad_cases[:8]:
        log("  MISMATCH", json.dumps(b))
    return fails == 0 and all(extra.values())


def graph_time(fn, reps: int = 7) -> float:
    """Median ms of one replay of a CUDA graph that runs fn once."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    del g
    return ts[len(ts) // 2]


def wbytes(layer) -> int:
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    return n * k // 2 + n * k // 16


def timing(shared, lm_head, results):
    out = {}
    gen = torch.Generator(device="cuda")
    gen.manual_seed(7)
    sink = torch.zeros(4, dtype=torch.int32, device="cuda")
    big = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
    big.fill_(1)
    for blocks, threads, smem in ((48 * 8, 256, 0), (48, 128, 17 * 1024), (48 * 4, 128, 0)):
        ms = graph_time(lambda: marlinx.hog(big, blocks, threads, smem, sink))
        out[f"roofline_hog_{blocks}x{threads}_smem{smem}_GBps"] = big.numel() / (ms * 1e-3) / 1e9
    log("roofline", {k: round(v, 1) for k, v in out.items()})

    for m in (4, 1):
        xs = [gen_x(m, 2688, 0, gen) for _ in shared]
        acts = [relu2(stock(up, x)) for (up, _), x in zip(shared, xs)]
        up_bytes = sum(wbytes(up) for up, _ in shared)
        dn_bytes = sum(wbytes(dn) for _, dn in shared)
        variants = [("stock", None)] + [(f"marlinx_d{d}", d) for d in TIMING_DEPTHS]
        for name, depth in variants:
            if depth is None:
                fu = lambda: [stock(up, x) for (up, _), x in zip(shared, xs)]
                fd = lambda: [stock(dn, a) for (_, dn), a in zip(shared, acts)]
            else:
                fu = lambda d=depth: [mine(up, x, d) for (up, _), x in zip(shared, xs)]
                fd = lambda d=depth: [mine(dn, a, d) for (_, dn), a in zip(shared, acts)]
            tu = graph_time(fu)
            td = graph_time(fd)
            out[f"up_M{m}_{name}_us"] = tu * 1e3 / len(shared)
            out[f"up_M{m}_{name}_GBps"] = up_bytes / (tu * 1e-3) / 1e9
            out[f"down_M{m}_{name}_us"] = td * 1e3 / len(shared)
            out[f"down_M{m}_{name}_GBps"] = dn_bytes / (td * 1e-3) / 1e9
            log(f"M={m} {name}: up {out[f'up_M{m}_{name}_us']:.1f} us ({out[f'up_M{m}_{name}_GBps']:.0f} GB/s), "
                f"down {out[f'down_M{m}_{name}_us']:.1f} us ({out[f'down_M{m}_{name}_GBps']:.0f} GB/s)")
        x = gen_x(m, 2688, 0, gen)
        lb = wbytes(lm_head)
        for name, depth in variants:
            f = (lambda: [stock(lm_head, x) for _ in range(4)]) if depth is None else (
                lambda d=depth: [mine(lm_head, x, d) for _ in range(4)])
            t = graph_time(f)
            out[f"lm_head_M{m}_{name}_us"] = t * 1e3 / 4
            out[f"lm_head_M{m}_{name}_GBps"] = lb * 4 / (t * 1e-3) / 1e9
            log(f"M={m} {name}: lm_head {out[f'lm_head_M{m}_{name}_us']:.1f} us ({out[f'lm_head_M{m}_{name}_GBps']:.0f} GB/s)")

    # Shared expert on an aux stream beside a humming-like persistent streaming kernel
    # (48 CTAs x 128 threads, 17 KB smem), as in the MoE layer at verify (M=4).
    m = 4
    xs = [gen_x(m, 2688, 0, gen) for _ in shared]
    chunk = 48 << 20
    hogbuf = torch.empty(len(shared) * chunk, dtype=torch.uint8, device="cuda")
    hogbuf.fill_(1)
    aux = torch.cuda.Stream()

    def layer_seq(use, with_hog=True, with_shared=True):
        def run():
            for i, ((up, dn), x) in enumerate(zip(shared, xs)):
                main = torch.cuda.current_stream()
                aux.wait_stream(main)
                if with_shared:
                    with torch.cuda.stream(aux):
                        h = use(up, x)
                        use(dn, relu2(h))
                if with_hog:
                    marlinx.hog(hogbuf[i * chunk : (i + 1) * chunk], 48, 128, 17 * 1024, sink)
                main.wait_stream(aux)
        return run

    per = 1e3 / len(shared)
    out["overlap_hog_only_us_per_layer"] = graph_time(layer_seq(stock, with_shared=False)) * per
    for name, fn in [("stock", stock)] + [(f"marlinx_d{d}", (lambda l, x, d=d: mine(l, x, d))) for d in OVERLAP_DEPTHS]:
        out[f"overlap_hog_plus_{name}_us_per_layer"] = graph_time(layer_seq(fn)) * per
        out[f"shared_only_{name}_us_per_layer"] = graph_time(layer_seq(fn, with_hog=False)) * per
    log("overlap (us/layer):", {k: round(v, 1) for k, v in out.items() if k.startswith(("overlap", "shared_only"))})
    results["timing"] = out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--layers", type=int, default=len(SHARED_LAYERS))
    ap.add_argument("--depths", type=int, nargs="+", default=[marlinx.DEPTH])
    args = ap.parse_args()
    snap = os.environ["NEMO_SNAPSHOT"]
    out_dir = os.environ.get("NEMO_OUT", "/tmp")
    results: dict = {"argv": sys.argv[1:]}
    log("torch", torch.__version__, "device", torch.cuda.get_device_name(), "sms",
        torch.cuda.get_device_properties(0).multi_processor_count)
    log("marlinx available", marlinx.available(), "regs", {d: marlinx.regs(d) for d in (4, 8)})
    with open(os.path.join(snap, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    t0 = time.time()
    layers = SHARED_LAYERS[: args.layers]
    shared = []
    for li in layers:
        p = f"backbone.layers.{li}.mixer.shared_experts"
        shared.append((load_linear(snap, wmap, p + ".up_proj"), load_linear(snap, wmap, p + ".down_proj")))
    lm_head = load_linear(snap, wmap, "lm_head")
    for up, dn in shared:
        assert marlinx.prepare(up) and marlinx.prepare(dn)
    assert marlinx.prepare(lm_head)
    log(f"loaded {len(shared)} shared experts + lm_head in {time.time() - t0:.1f}s; "
        f"gs up={float(shared[0][0].weight_global_scale):.3e} lm={float(lm_head.weight_global_scale):.3e}")
    ok = True
    for depth in args.depths:
        ok = exactness(shared, lm_head, ms=[1, 2, 3, 4, 5, 6, 7, 8], kinds=[0, 1, 2], results=results,
                       depth=depth) and ok
    if not args.skip_timing:
        timing(shared, lm_head, results)
    with open(os.path.join(out_dir, "marlinx_exact.json"), "w") as f:
        json.dump(results, f, indent=1)
    log("RESULT", "EXACT" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

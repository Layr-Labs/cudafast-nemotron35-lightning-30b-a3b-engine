"""Spark gate: PDL on/off byte equality and graph timings, using pinned weights.

Run in the pinned image with the staged overlay and kernels on PYTHONPATH:
  python3 harness/kernels/marlinx/tests/marlinx_pdl_exact.py SNAPSHOT
Requires the PDL ABI and MARLINX_PDL=1; never accepts a silent fallback.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

import marlinx
from marlinx_exact import SHARED_LAYERS, gen_x, load_linear, relu2, stock


def bits_equal(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def launch(layer, x, out, pdl, depth, mode="gemm"):
    # Direct ABI deliberately bypasses fallback, and reuses caller-owned output
    # storage. marlinx has NO workspace/lock argument or global scratch; the
    # layers' same stock-Marlin workspace objects remain untouched throughout.
    name = "marlinx_gemm" + ("_" + mode if mode != "gemm" else "")
    op = getattr(marlinx._load(), name + ("_pdl" if pdl else ""))
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    args = (x.data_ptr(), x.stride(0), x.shape[0], layer.weight.data_ptr(),
            layer.weight_scale.data_ptr(), layer.weight_global_scale.data_ptr(),
            out.data_ptr(), out.stride(0), n, k,
            torch.cuda.get_device_properties(x.device).multi_processor_count, depth,
            torch.cuda.current_stream(x.device).cuda_stream)
    if mode == "zs":
        args += (layer._marlinx_zs.data_ptr(),)
    rc = op(*args)
    if rc:
        raise RuntimeError(f"{name} pdl={pdl} depth={depth} failed: {rc}")
    return out


def graph_ms(fn, batch=1, reps=7):
    fn()  # eager first: load/launch errors cannot hide in capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(batch):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / batch)
    return sorted(times)[len(times) // 2]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot", nargs="?", default=os.environ.get("NEMO_SNAPSHOT"))
    ap.add_argument("--layers", type=int, default=len(SHARED_LAYERS))
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    if not args.snapshot or not 1 <= args.layers <= len(SHARED_LAYERS):
        ap.error("snapshot and 1..23 layers required")
    assert marlinx.available() and marlinx._init()
    assert marlinx._pdl_active, "PDL ABI/init unavailable or MARLINX_PDL=0"
    with open(os.path.join(args.snapshot, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    shared = []
    for li in SHARED_LAYERS[:args.layers]:
        prefix = f"backbone.layers.{li}.mixer.shared_experts"
        up = load_linear(args.snapshot, wmap, prefix + ".up_proj")
        down = load_linear(args.snapshot, wmap, prefix + ".down_proj")
        assert marlinx.prepare(up) and marlinx.prepare(down)
        assert down._marlinx_zs is not None
        shared.append((up, down))
    head = load_linear(args.snapshot, wmap, "lm_head")
    assert marlinx.prepare(head)
    gen = torch.Generator(device="cuda").manual_seed(319)
    results = {"torch": torch.__version__, "device": torch.cuda.get_device_name(),
               "layers": args.layers, "cases": 0, "failures": [], "timing_us": {}}

    def check(label, a, b):
        results["cases"] += 1
        if not bits_equal(a, b):
            results["failures"].append(label)
            print("MISMATCH", label, flush=True)

    for depth in (4, 8):
        for m in range(1, 9):
            for idx, (up, down) in enumerate(shared):
                # All shape classes: dense shared up/down, fused up+ReLU2,
                # activation-derived ZS down, and the separate large head.
                for kind in (0, 1, 2):
                    x = gen_x(m, 2688, kind, gen)
                    activation = relu2(stock(up, x))
                    sparse = activation.clone()
                    sparse[torch.rand(sparse.shape, device="cuda", generator=gen) < .9] = -0.0
                    for layer, inp, mode in ((up, x, "gemm"), (up, x, "relu2"),
                                             (down, activation, "gemm"), (down, activation, "zs"),
                                             (down, sparse, "zs"), (down, torch.zeros_like(sparse), "zs")):
                        label = f"d{depth}/M{m}/L{idx}/kind{kind}/{mode}/{results['cases']}"
                        off = torch.empty((m, layer.output_size_per_partition), dtype=x.dtype, device=x.device)
                        on = torch.empty_like(off)
                        ref = stock(layer, inp)
                        if mode == "relu2":
                            ref = relu2(ref)
                        launch(layer, inp, off, False, depth, mode)
                        launch(layer, inp, on, True, depth, mode)
                        check(label + "/on-off", on, off)
                        check(label + "/off-stock", off, ref)
            for kind in (0, 1, 2):
                x = gen_x(m, 2688, kind, gen)
                off = torch.empty((m, 131072), dtype=x.dtype, device=x.device)
                on = torch.empty_like(off)
                launch(head, x, off, False, depth)
                launch(head, x, on, True, depth)
                check(f"head/d{depth}/M{m}/kind{kind}/on-off", on, off)
                check(f"head/d{depth}/M{m}/kind{kind}/stock", off, stock(head, x))

            # Same h/y/output buffers, same prepared layers and same unused
            # stock workspaces across every launch and replay. No copies or
            # kernels between fused up and ZS down. Also stresses dense reads
            # after the preceding grid overwrote y at the previous pair.
            storage = gen_x(m, 2688 + 64, 0, gen)
            x = storage[:, :2688]  # aligned strided input
            h = torch.empty((m, 3712), device="cuda", dtype=x.dtype)
            y = torch.empty((m, 2688), device="cuda", dtype=x.dtype)
            logits = torch.empty((m, 131072), device="cuda", dtype=x.dtype)
            aux = torch.cuda.Stream()
            workspace_bits = [layer.workspace.clone() for pair in shared for layer in pair]

            def sequence(pdl, fork=False):
                def body():
                    cur = x
                    for up, down in shared:
                        if fork:
                            aux.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(aux):
                                launch(up, cur, h, pdl, depth, "relu2")
                                launch(down, h, y, pdl, depth, "zs")
                            torch.cuda.current_stream().wait_stream(aux)
                        else:
                            launch(up, cur, h, pdl, depth, "relu2")
                            launch(down, h, y, pdl, depth, "zs")
                        cur = y
                    launch(head, cur, logits, pdl, depth)
                return body

            graphs = {}
            for pdl in (False, True):
                for fork in (False, True):
                    body = sequence(pdl, fork)
                    body()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        body()
                    graphs[pdl, fork] = graph
            for kind in (0, 1, 2, 0):
                x.copy_(gen_x(m, 2688, kind, gen))
                graphs[False, False].replay()
                ref_h, ref_y, ref_logits = h.clone(), y.clone(), logits.clone()
                for mode in ((True, False), (False, True), (True, True)):
                    graphs[mode].replay()
                    check(f"chain/d{depth}/M{m}/kind{kind}/{mode}/h", h, ref_h)
                    check(f"chain/d{depth}/M{m}/kind{kind}/{mode}/y", y, ref_y)
                    check(f"chain/d{depth}/M{m}/kind{kind}/{mode}/head", logits, ref_logits)
            for layer, before in zip((layer for pair in shared for layer in pair), workspace_bits):
                check(f"workspace-untouched/d{depth}/M{m}/{layer.name}", layer.workspace, before)

            if not args.skip_timing:
                up, down = shared[0]
                for pdl in (False, True):
                    for layer, inp, out, mode in ((up, x, h, "gemm"), (up, x, h, "relu2"),
                                                  (down, h, y, "gemm"), (down, h, y, "zs"),
                                                  (head, y, logits, "gemm")):
                        key = f"d{depth}/M{m}/pdl{int(pdl)}/{layer.name}/{mode}"
                        results["timing_us"][key] = 1e3 * graph_ms(
                            lambda: launch(layer, inp, out, pdl, depth, mode), batch=32)
                    for fork in (False, True):
                        key = f"d{depth}/M{m}/pdl{int(pdl)}/chain/fork{int(fork)}"
                        results["timing_us"][key] = 1e3 * graph_ms(sequence(pdl, fork))
            print(f"checked depth={depth} M={m}, failures={len(results['failures'])}", flush=True)
    results["passed"] = not results["failures"]
    payload = json.dumps(results, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(payload + "\n")
    print(payload, flush=True)
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

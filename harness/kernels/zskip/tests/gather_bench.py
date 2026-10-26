#!/usr/bin/env python3
"""Gather-rate probe for the zskip load path (no math): GB/s of useful bytes when each CTA copies
the active k rows of one (expert, column slice) item, for several copy mechanisms, segment widths,
row strides and densities. Cold: several distinct 'layers' of 18 experts each, rotated in a graph.

  bin/atom nemo exec SHA harness/kernels/zskip/tests/gather_bench.py
"""

import argparse
import ctypes
import json
import os
import statistics
import time

import torch

import zskip

K = 1856
EXPERTS = 18
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--densities", default="0.092,1.0")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default=os.environ.get("NEMO_OUT", "."))
    args = ap.parse_args()
    lib = zskip._load()
    assert lib is not None, zskip.load_error()
    lib.zskip_gather.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong,
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    lib.zskip_gather.restype = ctypes.c_int
    results = []
    g = torch.Generator().manual_seed(5)
    for stride in (1344, 1536):
        layer_bytes = EXPERTS * K * stride
        buf = torch.randint(0, 255, (args.layers * layer_bytes,), dtype=torch.uint8, device="cuda")
        for dens in [float(v) for v in args.densities.split(",")]:
            # active rows per (layer, expert)
            lists = []
            for L in range(args.layers):
                for e in range(EXPERTS):
                    if dens >= 1.0:
                        r = torch.arange(K)
                    else:
                        r = (torch.rand(K, generator=g) < dens).nonzero().flatten()
                    lists.append(r.to(torch.int32))
            nrows_total = sum(int(r.numel()) for r in lists)
            segs = [64, 192, 448, 1344] if stride == 1344 else [128, 256, 512, 768]
            for seg in segs:
                slices = stride // seg
                bases, loff, ln, rows = [], [], [], []
                per_layer = []
                off = 0
                for L in range(args.layers):
                    lb, ll, lnn = [], [], []
                    for e in range(EXPERTS):
                        r = lists[L * EXPERTS + e]
                        for s in range(slices):
                            lb.append(L * layer_bytes + e * K * stride + s * seg)
                            ll.append(off)
                            lnn.append(int(r.numel()))
                        rows.append(r)
                        off += int(r.numel())
                    per_layer.append((lb, ll, lnn))
                rows_t = torch.cat(rows).to(torch.int16).cuda() if rows else torch.zeros(1, dtype=torch.int16, device="cuda")
                layer_args = []
                for lb, ll, lnn in per_layer:
                    layer_args.append((torch.tensor(lb, dtype=torch.int64, device="cuda"),
                                       torch.tensor(ll, dtype=torch.int32, device="cuda"),
                                       torch.tensor(lnn, dtype=torch.int32, device="cuda")))
                items = len(per_layer[0][0])
                sink = torch.zeros(items, dtype=torch.int32, device="cuda")
                useful = nrows_total * stride / args.layers  # bytes per layer
                cfgs = []
                for target in (16384, 32768):
                    rb = 16
                    ns = max(2, min(32, target // (rb * seg)))
                    cfgs.append(("tma", 0, 32, ns, rb))
                for ns, thr in ((4, 128), (8, 128), (8, 256)):
                    rb = max(1, min(32, 16384 // (seg * ns)))
                    cfgs.append((f"coop{thr}", 1, thr, ns, rb))
                cfgs.append(("regs128", 2, 128, 8, 0))
                cfgs.append(("regs256", 2, 256, 8, 0))
                for name, mode, thr, ns, rb in cfgs:
                    def fn():
                        for (b, l, n) in layer_args:
                            rc = lib.zskip_gather(mode, items, thr, buf.data_ptr(), b.data_ptr(), l.data_ptr(),
                                                  n.data_ptr(), rows_t.data_ptr(), stride, seg, ns, rb,
                                                  sink.data_ptr(), torch.cuda.current_stream().cuda_stream)
                            assert rc == 0, rc
                    try:
                        fn()
                        torch.cuda.synchronize()
                        ts = []
                        for _ in range(2):
                            ts += hgraph(fn, args.reps)
                    except Exception as exc:  # noqa
                        log(f"   {name} stride {stride} seg {seg} dens {dens}: error {exc!r}")
                        continue
                    us = statistics.median(ts) * 1000 / args.layers
                    row = {"mech": name, "stride": stride, "seg": seg, "density": dens, "items": items,
                           "ns": ns, "rb": rb, "threads": thr, "us_per_layer": round(us, 2),
                           "GBps": round(useful / (us * 1e-6) / 1e9, 1)}
                    results.append(row)
                    log(f"stride {stride} seg {seg:5d} dens {dens:5.3f} items {items:4d} {name:8s} ns {ns:2d} rb {rb:2d}: "
                        f"{us:8.2f} us/layer {row['GBps']:6.1f} GB/s")
        del buf
        torch.cuda.empty_cache()
    with open(os.path.join(args.out, "gather.json"), "w") as f:
        json.dump(results, f, indent=1)


def hgraph(fn, reps):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        fn()
    gr.replay()
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    out = []
    for _ in range(reps):
        st.record()
        gr.replay()
        en.record()
        en.synchronize()
        out.append(st.elapsed_time(en))
    return out


if __name__ == "__main__":
    main()

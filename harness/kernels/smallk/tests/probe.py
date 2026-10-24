"""smallk probe: the stock small kernels of the depth-3 verify (router GEMM, grouped top-k, align/sort).

Run on Atom with `bin/atom nemo exec <sha> harness/kernels/smallk/tests/probe.py [parts]`.
Writes JSON to $NEMO_OUT. Parts (default ids,timing,tree):

- ids:    kernel name / grid / block of the fp32 router GEMM (F.linear and torch.mm, as inductor
          calls it) for M = 1..8 tokens at N = 128 experts (and larger N, to see whether the
          per-row reduction is independent of N), K = 2688.
- timing: per-call time of the stock router GEMM (rotating 48 weight copies: cold L2), grouped
          top-k and moe_align_block_size at M = 4, inside a CUDA graph (launch gaps included).
- tree:   the full fp32 summation tree of the router GEMM for one output, by cancellation probes
          (method of harness/kernels/densegemm/tests/probe.py: x = 1, w_a = 2^100, w_b = -2^100,
          8 power-of-two probes; a probe is lost iff it lies in the subtree of lca(a, b)).
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

OUT = os.environ.get("NEMO_OUT", "/tmp/smallk-out")
SNAP = os.environ.get("NEMO_SNAPSHOT", "")
os.makedirs(OUT, exist_ok=True)
DEV = torch.device("cuda")
T0 = time.time()
K_ROUTER = 2688
N_ROUTER = 128
BIG = 2.0 ** 100


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def save(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=str)


def profile_kernels(fn):
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    path = f"/tmp/prof-{os.getpid()}.json"
    prof.export_chrome_trace(path)
    ev = json.load(open(path))["traceEvents"]
    os.remove(path)
    ks = []
    for e in ev:
        if e.get("cat") == "kernel":
            a = e.get("args", {})
            ks.append({"name": e["name"][:200], "grid": a.get("grid"), "block": a.get("block"),
                       "smem": a.get("shared memory"), "regs": a.get("registers per thread"), "us": e.get("dur")})
    return ks


def graph_us(fn, reps=23, iters=30):
    """Median per-call time of fn replayed reps times back to back inside one CUDA graph."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0 / reps)
    return float(np.median(ts))


# ---------------------------------------------------------------- ids
def part_ids():
    res = {}
    for N in (128, 1024, 8192):
        W = torch.randn(N, K_ROUTER, device=DEV)
        for M in range(1, 9):
            x = torch.randn(M, K_ROUTER, device=DEV)
            res[f"linear N={N} M={M}"] = profile_kernels(lambda: F.linear(x, W))
            if N == N_ROUTER:
                out = torch.empty(M, N, device=DEV)
                res[f"mm_out N={N} M={M}"] = profile_kernels(lambda: torch.mm(x, W.t(), out=out))
    for k, v in res.items():
        log("ids", k, [(e["name"][:60], e["grid"], e["block"]) for e in v])
    save("ids.json", res)


# ---------------------------------------------------------------- timing
def part_timing():
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

    res = {}
    Ws = [torch.randn(N_ROUTER, K_ROUTER, device=DEV) for _ in range(48)]
    for M in (1, 4, 8):
        x = torch.randn(M, K_ROUTER, device=DEV)
        it = iter(range(10 ** 9))

        def gemm_cold():
            return F.linear(x, Ws[next(it) % 48])

        res[f"router_gemm M={M} cold"] = graph_us(gemm_cold)
        res[f"router_gemm M={M} hot"] = graph_us(lambda: F.linear(x, Ws[0]))
        logits = torch.randn(M, N_ROUTER, device=DEV)
        bias = torch.randn(N_ROUTER, device=DEV) * 0.01
        res[f"grouped_topk M={M}"] = graph_us(
            lambda: ops.grouped_topk(logits, 1, 1, 6, True, 2.5, bias, 1))
        ids = torch.stack([torch.randperm(N_ROUTER, device=DEV)[:6] for _ in range(M)]).to(torch.int32)
        res[f"moe_align bs16 M={M}"] = graph_us(lambda: moe_align_block_size(ids, 16, N_ROUTER, None,
                                                                              ignore_invalid_experts=True))
    for k, v in res.items():
        log("timing", k, round(v, 2))
    save("timing.json", res)


# ---------------------------------------------------------------- tree
class F32Prober:
    """Q(a, b, probes) on fn(x = ones[M, K], W[N, K]): bit i of the result is set when probe i
    survives, i.e. probe i is NOT in the subtree of lca(a, b)."""

    def __init__(self, N, K, M, fn=None):
        self.N, self.K, self.M = N, K, M
        self.fn = fn or F.linear
        self.x = torch.ones(M, K, device=DEV)
        self.W = torch.zeros(N, K, device=DEV)
        self.pw = torch.tensor([2.0 ** i for i in range(8)], device=DEV)
        self.calls = 0

    def run(self, a, b, P):
        Q = len(a)
        a_t = torch.as_tensor(a, device=DEV)
        b_t = torch.as_tensor(b, device=DEV)
        P_t = torch.as_tensor(P, device=DEV)
        Pv = torch.where(P_t >= 0, self.pw.expand_as(P_t.float()), torch.zeros((), device=DEV))
        Pi = P_t.clamp(min=0)
        res = torch.empty(Q, device=DEV)
        bad = torch.zeros((), device=DEV, dtype=torch.bool)
        N = self.N
        for s in range(0, Q, N):
            e = min(Q, s + N)
            n = e - s
            rows = torch.arange(n, device=DEV)
            self.W.zero_()
            self.W[rows, a_t[s:e]] = BIG
            self.W[rows, b_t[s:e]] = -BIG
            r8 = rows[:, None].expand(n, 8)
            self.W.index_put_((r8.reshape(-1), Pi[s:e].reshape(-1)), Pv[s:e].reshape(-1), accumulate=True)
            y = self.fn(self.x, self.W)  # [M, N]
            self.calls += 1
            if self.M > 1:
                bad |= (y != y[:1]).any()
            res[s:e] = y[0, :n]
        v = res.cpu().numpy()
        ok = (not bool(bad)) and bool(np.all((v >= 0) & (v <= 255) & (v == np.round(v))))
        return np.round(v).clip(0, 255).astype(np.int64), ok


def subtree_sets(prober, ref, members):
    members = np.asarray(members, dtype=np.int64)
    G = len(members)
    others = members[members != ref]
    G8 = (G + 7) // 8
    Mp = np.full(G8 * 8, -1, dtype=np.int64)
    Mp[:G] = members
    Mp = Mp.reshape(G8, 8)
    P = np.broadcast_to(Mp[None], (len(others), G8, 8)).copy()
    P[(P == ref) | (P == others[:, None, None])] = -1
    A = np.full(len(others) * G8, ref, dtype=np.int64)
    B = np.repeat(others, G8)
    masks, ok = prober.run(A, B, P.reshape(-1, 8))
    bits = (masks.reshape(len(others), G8, 1) >> np.arange(8)[None, None, :]) & 1
    S = (bits == 0).reshape(len(others), G8 * 8)[:, :G]
    return others, S, ok


def build_tree(prober, members, nodes, stats):
    """Binary tree over members. Leaves are element ids; internal node i is nodes[i - K]."""
    members = sorted(int(m) for m in members)
    K = prober.K
    if len(members) == 1:
        return members[0]
    if len(members) == 2:
        nodes.append((members[0], members[1]))
        return K + len(nodes) - 1
    ref = members[0]
    others, S, ok = subtree_sets(prober, ref, members)
    stats["ok"] = stats.get("ok", True) and bool(ok)
    sizes = S.sum(1)
    order = np.argsort(sizes, kind="stable")
    levels = []
    prev = None
    for bi in order:
        row = S[bi]
        if prev is not None and np.array_equal(row, prev[0]):
            prev[1].append(int(others[bi]))
            continue
        if prev is not None and not np.all(row >= prev[0]):
            stats["non_nested"] = stats.get("non_nested", 0) + 1
        prev = (row, [int(others[bi])])
        levels.append(prev)
    node = ref
    covered = {ref}
    mem = np.array(members)
    for row, group in levels:
        grp = set(int(m) for m in mem[row]) - covered
        if set(group) != grp:
            stats["group_mismatch"] = stats.get("group_mismatch", 0) + 1
            grp |= set(group)
        sub = build_tree(prober, sorted(grp), nodes, stats)
        nodes.append((node, sub))
        node = K + len(nodes) - 1
        covered |= grp
    return node


def summarize_tree(K, root, nodes):
    """Left-deep chains (sequential accumulations) and the tree that combines them."""
    def leaves(n):
        if n < K:
            return [n]
        l, r = nodes[n - K]
        return leaves(l) + leaves(r)

    sys.setrecursionlimit(100000)
    chains = []

    def walk(n):
        # follow the left spine while right children are leaves: that is one chain
        seq = []
        cur = n
        while cur >= K and nodes[cur - K][1] < K:
            seq.append(nodes[cur - K][1])
            cur = nodes[cur - K][0]
        if cur < K:
            seq.append(cur)
            seq.reverse()
            chains.append(seq)
            return ("chain", len(chains) - 1)
        if seq:
            # left spine ends in a non-leaf subtree, then leaves are added one by one
            sub = walk(cur)
            seq.reverse()
            return ("extend", sub, seq)
        l, r = nodes[n - K]
        return ("add", walk(l), walk(r))

    shape = walk(root)
    return {"chains": chains, "shape": shape}


def part_tree(M=4, N=N_ROUTER):
    pr = F32Prober(N, K_ROUTER, M)
    st = {}
    t0 = time.time()
    nodes = []
    root = build_tree(pr, list(range(K_ROUTER)), nodes, st)
    rng = np.random.default_rng(5)
    Q = 4 * N
    a = rng.integers(0, K_ROUTER, Q)
    b = (a + 1 + rng.integers(0, K_ROUTER - 1, Q)) % K_ROUTER
    P = rng.integers(0, K_ROUTER, (Q, 8))
    P[(P == a[:, None]) | (P == b[:, None])] = -1
    m1, _ = pr.run(a, b, P)
    perm = rng.permutation(Q)
    m2, _ = pr.run(a[perm], b[perm], P[perm])
    st["row_independent"] = bool(np.array_equal(m1[perm], m2))
    summ = summarize_tree(K_ROUTER, root, nodes)
    st["chains"] = len(summ["chains"])
    st["chain_lengths"] = sorted(set(len(c) for c in summ["chains"]))
    st["calls"] = pr.calls
    st["secs"] = time.time() - t0
    save(f"tree-M{M}-N{N}.json", {"K": K_ROUTER, "root": root, "nodes": nodes, "stats": st, "summary": summ})
    log("tree", M, N, st)


def main():
    parts = sys.argv[1].split(",") if len(sys.argv) > 1 else ["ids", "timing", "tree"]
    log("torch", torch.__version__, torch.cuda.get_device_name(), "tf32", torch.backends.cuda.matmul.allow_tf32)
    for p in parts:
        if p == "ids":
            part_ids()
        elif p == "timing":
            part_timing()
        elif p == "tree":
            part_tree(4)
        elif p.startswith("tree"):
            # treeM<m>N<n>
            m, n = p[4:].split("N")
            part_tree(int(m[1:]), int(n))
    log("done")


if __name__ == "__main__":
    main()

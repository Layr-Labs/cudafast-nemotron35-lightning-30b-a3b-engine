#!/usr/bin/env python3
"""tests/exact.py [--kernels DIR] — vsample.greedy_verify vs the stock chain the model runner runs
(gpu_model_runner._reject_unverified_drafts + RejectionSampler._greedy_argmax_sample +
vrows.fill_ + the accepted count of _update_states_after_model_execute), on adversarial inputs.
Needs the staged vLLM overlay on PYTHONPATH (imports the stock functions themselves)."""
import argparse
import itertools
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--kernels", default=None)
ap.add_argument("--iters", type=int, default=300)
ap.add_argument("--bench", action="store_true")
a = ap.parse_args()
if a.kernels:
    sys.path.insert(0, a.kernels)
import vsample  # noqa: E402
from vllm.v1.sample.rejection_sampler import RejectionSampler  # noqa: E402
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata  # noqa: E402
from vllm.v1.worker.gpu_model_runner import _reject_unverified_drafts  # noqa: E402

assert vsample.available(), vsample._load_error
dev = torch.device("cuda")
g = torch.Generator(device="cpu").manual_seed(1234)
VALL = vsample.VROWS_ALL


def md_for(d, drafts):
    return SpecDecodeMetadata(
        draft_token_ids=drafts.to(torch.int32).to(dev),
        num_draft_tokens=[d],
        cu_num_draft_tokens=torch.tensor([d], dtype=torch.int32, device=dev),
        cu_num_sampled_tokens=torch.tensor([d + 1], dtype=torch.int32, device=dev),
        target_logits_indices=torch.arange(d, dtype=torch.int32, device=dev),
        bonus_logits_indices=torch.tensor([d], dtype=torch.int32, device=dev),
        logits_indices=torch.arange(d + 1, dtype=torch.int32, device=dev),
    )


def stock(logits, md, vrows):
    if vrows is not None:
        md = _reject_unverified_drafts(md, vrows)
    out = RejectionSampler._greedy_argmax_sample(None, logits, md).sampled_token_ids
    if vrows is not None:
        vrows.fill_(VALL)
    acc = (out != -1).sum(dim=1).to(torch.int32)
    return out, acc


def fused(logits, md, vrows):
    acc = torch.full((1,), -7, dtype=torch.int32, device=dev)
    out = vsample.maybe_greedy_verify(logits, md, vrows, acc)
    assert out is not None, "guard refused"
    return out, acc


def make_logits(kind, rows, V, ld):
    base = torch.empty((rows, ld), dtype=torch.bfloat16)
    if kind == "normal":
        x = torch.randn((rows, ld), generator=g) * 4
    elif kind == "ties":
        x = torch.randint(0, 3, (rows, ld), generator=g).float()
    elif kind == "flat":
        x = torch.full((rows, ld), 1.5)
    elif kind == "neginf":
        x = torch.full((rows, ld), float("-inf"))
    elif kind == "nan":
        x = torch.randn((rows, ld), generator=g)
        m = torch.rand((rows, ld), generator=g) < 0.0005
        x[m] = float("nan")
    elif kind == "posinf":
        x = torch.randn((rows, ld), generator=g)
        m = torch.rand((rows, ld), generator=g) < 0.0002
        x[m] = float("inf")
    elif kind == "zeros":  # +0 / -0 at the max
        x = -torch.rand((rows, ld), generator=g)
        m = torch.rand((rows, ld), generator=g) < 0.001
        x[m] = 0.0
        m2 = torch.rand((rows, ld), generator=g) < 0.5
        x[m & m2] = -0.0
    elif kind == "bigtie":  # the max sits on many entries far apart (chunk boundaries)
        x = torch.randn((rows, ld), generator=g)
        for r in range(rows):
            pos = torch.randint(0, V, (17,), generator=g)
            x[r, pos] = 50.0
            x[r, (pos // 4096) * 4096] = 50.0
    else:
        raise ValueError(kind)
    base.copy_(x)
    return base.to(dev)[:, :V] if ld != V else base.to(dev)


fails = 0
cases = 0
kinds = ["normal", "ties", "flat", "neginf", "nan", "posinf", "zeros", "bigtie"]
shapes = [(4, 131072, 131072), (4, 131072, 131136), (1, 131072, 131072), (8, 131072, 131072),
          (3, 4096, 4096), (5, 8, 8), (2, 4104, 4104), (6, 262144, 262144)]
for it in range(a.iters):
    d, V, ld = shapes[it % len(shapes)]
    kind = kinds[(it // len(shapes)) % len(kinds)]
    logits = make_logits(kind, d + 1, V, ld)
    am = logits.argmax(dim=-1).to(torch.int32).cpu()
    # drafts: accept a random prefix, then a wrong or random token
    k = int(torch.randint(0, d + 1, (1,), generator=g))
    drafts = torch.randint(0, V, (d,), generator=g, dtype=torch.int32)
    drafts[:k] = am[:k]
    if k < d and int(torch.randint(0, 2, (1,), generator=g)):
        drafts[k] = (am[k] + 1) % V
    for vr in [None, VALL, 0, 1, 2, 3, d, d + 1, -5]:
        vr_s = None if vr is None else torch.tensor([vr], dtype=torch.int32, device=dev)
        vr_f = None if vr is None else torch.tensor([vr], dtype=torch.int32, device=dev)
        o1, a1 = stock(logits, md_for(d, drafts), vr_s)
        o2, a2 = fused(logits, md_for(d, drafts), vr_f)
        cases += 1
        ok = torch.equal(o1, o2) and torch.equal(a1, a2) and (vr is None or torch.equal(vr_s, vr_f))
        if not ok:
            fails += 1
            if fails <= 10:
                print("MISMATCH", kind, d, V, ld, vr, o1.tolist(), o2.tolist(), a1.tolist(), a2.tolist(),
                      None if vr is None else (vr_s.tolist(), vr_f.tolist()))
print(f"vsample exact: {cases} cases, {fails} mismatches")

if a.bench:
    d, V = 4, 131072
    logits = make_logits("normal", d + 1, V, V)
    am = logits.argmax(dim=-1).to(torch.int32).cpu()
    drafts = am[:d].clone()
    for name, fn in [("stock", stock), ("fused", fused)]:
        vr = torch.tensor([VALL], dtype=torch.int32, device=dev)
        for _ in range(20):
            fn(logits, md_for(d, drafts), vr)
        md = md_for(d, drafts)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(200):
            fn(logits, md, vr)
        e1.record()
        torch.cuda.synchronize()
        print(f"bench {name}: {e0.elapsed_time(e1) / 200 * 1e3:.1f} us per call (eager, back to back)")
sys.exit(1 if fails else 0)

# SPDX-License-Identifier: Apache-2.0
"""GPU differential proof for the claim-when-free w13 tile scheduler
(humming_relu2_epilogue.py, HUMMING_W13_DYN): every output bit of the fused
routed-up/ReLU^2 GEMM must equal the static-schedule kernel's.

Run in the pinned container with the staged overlay:
  NEMO_SNAPSHOT=... python3 harness/tests/humming_w13_dyn_exact.py --layers all \
      [--caps caps.pt]
--caps takes captured real prefill inputs {"caps": [{layer: (x [M, 2688] bf16,
topk_ids [M, 6])}]}; decode shapes are row windows of them. Without it the
inputs come from humming_moe_exact.make_inputs (embedding rows routed by the
real router, random and edge-case activations).
"""

import argparse
import os

import torch

os.environ["VLLM_HUMMING_BATCH_INVARIANT"] = "1"
import humming_moe_exact as hx
from vllm.model_executor.layers.fused_moe.experts import humming_relu2_epilogue as fused

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--layers", default="1,13,27,40,51")
p.add_argument("--caps", default=None)
p.add_argument("--ms", default="1,2,3,4,5,6,8,496")
args = p.parse_args()
layers = hx.MOE_LAYERS if args.layers == "all" else [int(v) for v in args.layers.split(",")]
ms = [int(v) for v in args.ms.split(",")]
dev = torch.device("cuda")
snap = hx.Snapshot(os.environ["NEMO_SNAPSHOT"])
caps = torch.load(args.caps)["caps"] if args.caps else None
emb = None if caps else snap.load(["backbone.embeddings.weight"], dev)[
    "backbone.embeddings.weight"]


def run(ex, x, ids, dyn, fill):
    kw = ex.prepare_humming_moe_kwargs(ids, None, None)[0]
    out = torch.empty(x.size(0) * hx.TOP_K, hx.INTER, dtype=torch.bfloat16, device=dev)
    out.view(torch.int16).fill_(fill)
    fused.forward_relu2(layer=ex.layer, inputs=x, outputs=out, dyn=dyn, **kw)
    return out


cases = bad = 0
for li in layers:
    raw = snap.raw_experts(li, dev)
    ex = hx.build_variant(raw, f"backbone.layers.{li}.mixer.experts", True)
    del raw
    inputs = []
    if caps:
        for cap in caps:
            x, ids = cap[li][0].to(dev), cap[li][1].to(dev).to(torch.int32)
            for m in ms:
                for off in ([0] if m >= x.size(0) else range(0, x.size(0) - m, 61)):
                    inputs.append((x[off:off + m].contiguous(), ids[off:off + m].contiguous()))
    else:
        router = snap.router(li, dev)
        for kind in ("emb", "rand", "rand_uniform_route", "edge"):
            for m in ms:
                for seed in range(3):
                    x, ids, _ = hx.make_inputs(kind, m, 1000 * li + seed, emb, router, dev)
                    inputs.append((x, ids))
    for x, ids in inputs:
        ref = run(ex, x, ids, False, 0x7FC1)
        got = run(ex, x, ids, True, -1)
        assert int((ref.view(torch.int16) == 0x7FC1).sum()) == 0, "static kernel left a hole"
        n = hx.diff_count(ref, got)
        cases += 1
        bad += n
        if n:
            print(f"MISMATCH layer {li} m {x.size(0)}: {n} elements", flush=True)
    print(f"layer {li}: {len(inputs)} cases, total mismatched elements {bad}", flush=True)
    del ex
    torch.cuda.empty_cache()
print(f"RESULT cases={cases} mismatched_elements={bad}", flush=True)
assert bad == 0

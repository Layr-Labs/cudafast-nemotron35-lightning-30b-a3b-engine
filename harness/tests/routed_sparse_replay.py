# SPDX-License-Identifier: Apache-2.0
"""Check dynamic sparse masks and routing under CUDA graph replay.

Run in the pinned container with the staged overlay/kernel packages:
python3 harness/tests/routed_sparse_replay.py --weights SNAPSHOT --output result.json
"""

import argparse
import json
import os
from pathlib import Path

import torch

os.environ["VLLM_HUMMING_BATCH_INVARIANT"] = "1"
torch.set_num_threads(8)
import humming_moe_exact as check
import routed_sparse
from vllm.utils.humming import HummingMethod

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--weights", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
snap = check.Snapshot(args.weights)
report = []
for li in [1, 27, 51]:
    ex = check.build_variant(
        snap.raw_experts(li, "cuda"), f"backbone.layers.{li}.mixer.experts", True
    )
    packed = ex.layer.routed_sparse_down
    assert packed is not None
    for m in [1, 4, 8]:
        x = torch.zeros((m * 6, 1856), dtype=torch.bfloat16, device="cuda")
        ids = torch.arange(6, device="cuda", dtype=torch.int32).repeat(m, 1)
        actual = torch.empty((m * 6, 2688), dtype=torch.bfloat16, device="cuda")
        expected = torch.empty_like(actual)
        for _ in range(3):
            assert routed_sparse.forward(x, ids, actual, packed)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            assert routed_sparse.forward(x, ids, actual, packed)
        for i, kind in enumerate(["dense", "sparse", "tiny", "zero", "dense"]):
            torch.manual_seed(1030 + li + m + i)
            x.copy_(torch.randn_like(x))
            if kind == "sparse":
                x.mul_((torch.rand_like(x) > 0.9).to(x.dtype))
            elif kind == "tiny":
                x.mul_(1e-38)
            elif kind == "zero":
                x.zero_()
            ids.copy_(
                torch.stack(
                    [torch.randperm(128, device="cuda")[:6] for _ in range(m)]
                ).to(torch.int32)
            )
            _, kw = ex.prepare_humming_moe_kwargs(ids, None, None)
            HummingMethod.forward_layer(
                layer=ex.layer,
                inputs=x,
                input_scale=None,
                outputs=expected,
                sublayer_name="w2",
                **kw,
            )
            graph.replay()
            differences = check.diff_count(expected, actual)
            assert differences == 0, (li, m, kind, differences)
            row = dict(layer=li, tokens=m, kind=kind, bit_differences=differences)
            report.append(row)
            print(json.dumps(row), flush=True)
        del graph
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
print("dynamic graph cases exact", len(report), flush=True)

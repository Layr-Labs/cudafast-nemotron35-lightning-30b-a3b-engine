# SPDX-License-Identifier: Apache-2.0
"""GPU proof for the bf16ws prefill attention projection GEMMs against F.linear.

Run inside the pinned image with staged overlay/kernels on PYTHONPATH:
  python3 harness/tests/bf16ws_exact.py --weights SNAPSHOT
For every attention qkv_proj and o_proj weight of the snapshot: the load-time admission (pack
inverse + bitwise F.linear comparison on synthetic activations), then the runtime dispatch (the
nemo_attn_pk_linear custom op at 496 rows) on fresh random activations, compared as integer
BF16 bits.
"""

import argparse

import torch
import torch.nn.functional as F
import humming_moe_exact as check
from vllm.model_executor.models import nemotron_h_attn_packed as attn

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--weights", required=True)
args = parser.parse_args()
torch.manual_seed(4961)
snap = check.Snapshot(args.weights)
layers = sorted(int(k.split(".")[2]) for k in snap.weight_map if k.endswith(".mixer.o_proj.weight")
                and k.startswith("backbone."))
assert len(layers) == 6, layers


class Proj(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.weight = w


cases = 0
for li in layers:
    p = f"backbone.layers.{li}.mixer."
    t = snap.load([p + "q_proj.weight", p + "k_proj.weight", p + "v_proj.weight", p + "o_proj.weight"], "cuda")
    qkv = torch.cat([t[p + "q_proj.weight"], t[p + "k_proj.weight"], t[p + "v_proj.weight"]], 0).contiguous()
    for name, w in (("qkv_proj", qkv), ("o_proj", t[p + "o_proj.weight"].contiguous())):
        proj = Proj(w)
        torch.set_default_dtype(torch.bfloat16)  # as during vLLM model loading
        try:
            assert attn._prepare_ws([proj]) and getattr(proj, "_bws_nk", None), (li, name, "admission")
        finally:
            torch.set_default_dtype(torch.float32)
        n, k = w.shape
        for scale in (1.0, 0.05, 20.0):
            x = (torch.randn((496, k), device="cuda") * scale).to(torch.bfloat16)
            ref = F.linear(x, w)
            got = torch.ops.vllm.nemo_attn_pk_linear(x, w, n, k, -1)
            assert torch.equal(ref.view(torch.int16), got.view(torch.int16)), (li, name, scale)
            cases += 1
        print("layer exact", li, name, "cases", cases, flush=True)
assert attn._ws_stats["dispatched"] == cases, attn._ws_stats
print("bf16ws exact:", cases, "random cases;", attn._ws_stats, flush=True)

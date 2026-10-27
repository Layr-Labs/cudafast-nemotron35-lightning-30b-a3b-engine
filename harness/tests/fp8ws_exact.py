# SPDX-License-Identifier: Apache-2.0
"""GPU proof for the fp8ws prefill in_proj GEMM against the pinned CUTLASS operator.

Run inside the pinned image with staged overlay/kernels on PYTHONPATH:
  python3 harness/tests/fp8ws_exact.py --weights SNAPSHOT [--captures FILE]
For every Mamba in_proj weight of the snapshot: the load-time admission (pack inverse + bitwise
CUTLASS comparison on synthetic activations), then the runtime dispatch (fp8c/fp8small custom-op
fallback at M=496) on fresh random activations, compared as integer BF16 bits. --captures adds a
file of real activations ({"a": [prompts][23] uint8 [496, 2688], "sa": ..., "sb": ..., "w": [23]}).
"""

import argparse

import torch
import humming_moe_exact as check
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.scaled_mm import fp8ws
from vllm.model_executor.kernels.linear.scaled_mm import fp8c as dispatch  # noqa: F401

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--weights", required=True)
parser.add_argument("--captures", default="")
args = parser.parse_args()
f8 = torch.float8_e4m3fn
torch.manual_seed(4960)
snap = check.Snapshot(args.weights)
layers = sorted(int(k.split(".")[2]) for k in snap.weight_map if k.endswith(".mixer.in_proj.weight"))
assert len(layers) == 23
cases = 0
for li in layers:
    p = f"backbone.layers.{li}.mixer.in_proj"
    names = [p + "." + s for s in ["weight", "weight_scale", "input_scale"]]
    t = snap.load(names, "cuda")
    w = t[names[0]].contiguous()  # [N, K]
    sb = t[names[1]].float().max().reshape(1)
    sa = t[names[2]].float().max().reshape(1)
    b = w.t()  # column-major [K, N], as the linear layer holds it
    torch.set_default_dtype(torch.bfloat16)  # as during vLLM model loading
    try:
        assert fp8ws.register(b, sb, sa), (li, "admission")
    finally:
        torch.set_default_dtype(torch.float32)
    for kind in ["random", "small", "extreme"]:
        scale = {"random": 20, "small": 0.02, "extreme": 150}[kind]
        a = (torch.randn((496, 2688), device="cuda") * scale).clamp(-448, 448).to(f8)
        ref = ops.cutlass_scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        got = torch.ops.vllm.fp8c_scaled_mm(a, b, torch.empty(1, dtype=torch.uint8, device="cuda"), sa, sb)
        assert torch.equal(ref.view(torch.int16), got.view(torch.int16)), (li, kind)
        cases += 1
    print("layer exact", li, "cases", cases, flush=True)
assert fp8ws._stats["dispatched"] == cases, fp8ws._stats

if args.captures:
    caps = torch.load(args.captures)
    real = 0
    for li in range(23):
        b = caps["w"][li].cuda().view(f8).t()
        sb = caps["sb"][li].reshape(1).cuda()
        assert fp8ws.register(b, sb, caps["sa"][0][li].reshape(1).cuda()), (li, "admission (captured)")
        for pi in range(len(caps["a"])):
            a = caps["a"][pi][li].cuda().view(f8)
            sa = caps["sa"][pi][li].reshape(1).cuda()
            ref = ops.cutlass_scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
            got = fp8ws.maybe_gemm(a, b, sa, sb)
            assert got is not None and torch.equal(ref.view(torch.int16), got.view(torch.int16)), (li, pi)
            real += 1
    print("captured activations exact:", real, "cases", flush=True)
print("fp8ws exact:", cases, "random cases;", fp8ws._stats, flush=True)

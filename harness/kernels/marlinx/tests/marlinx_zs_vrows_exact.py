"""Verify-row-limited zero-row skip: rows below the depth gate's verify-row count are
bit-identical to the unlimited marlinx zero-row skip (and to stock Marlin) on all 23
shared-down layers; with VROWS_ALL every row is identical.

Runs inside the pinned image with $KERNELS_OUT on PYTHONPATH and $NEMO_SNAPSHOT set:
    python3 harness/kernels/marlinx/tests/marlinx_zs_vrows_exact.py
"""

import ctypes
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import marlinx  # noqa: E402
import marlinx_exact as mx  # noqa: E402

VROWS_ALL = 1 << 30


def main() -> int:
    snap = os.environ["NEMO_SNAPSHOT"]
    wmap = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    lib = marlinx._load()
    lib.marlinx_set_vrows.argtypes = [ctypes.c_void_p]
    vbuf = torch.full((1,), VROWS_ALL, dtype=torch.int32, device="cuda")
    gen = torch.Generator(device="cuda")
    gen.manual_seed(77)
    cases = bad = 0
    for li in mx.SHARED_LAYERS:
        down = mx.load_linear(snap, wmap, f"backbone.layers.{li}.mixer.shared_experts.down_proj")
        assert marlinx.prepare(down) and down._marlinx_zs is not None, li
        for m in range(1, 9):
            for kind in range(4):
                x = torch.relu(torch.randn(m, 3712, generator=gen, device="cuda")) ** 2
                x[torch.rand(m, 3712, generator=gen, device="cuda") < (0.5, 0.75, 0.9, 0.0)[kind]] = 0.0
                x = x.to(torch.bfloat16)
                ref = mx.stock(down, x)
                lib.marlinx_set_vrows(None)
                plain = marlinx.gemm(x, down.weight, down.weight_scale, down.weight_global_scale,
                                     2688, 3712, zs=down._marlinx_zs)
                lib.marlinx_set_vrows(vbuf.data_ptr())
                for v in list(range(1, m + 1)) + [VROWS_ALL]:
                    vbuf.fill_(v)
                    out = marlinx.gemm(x, down.weight, down.weight_scale, down.weight_global_scale,
                                       2688, 3712, zs=down._marlinx_zs)
                    rows = m if v >= m else v
                    ok = torch.equal(out[:rows].view(torch.int16), plain[:rows].view(torch.int16)) and \
                        torch.equal(out[:rows].view(torch.int16), ref[:rows].view(torch.int16))
                    cases += 1
                    bad += 0 if ok else 1
                lib.marlinx_set_vrows(None)
        print(f"layer {li}: cases {cases} mismatches {bad}", flush=True)
    # sabotage: a valid row must change when its own nonzero column is dropped from the mask
    x = torch.zeros(2, 3712, dtype=torch.bfloat16, device="cuda")
    x[1, 100] = 3.0
    vbuf.fill_(1)
    lib.marlinx_set_vrows(vbuf.data_ptr())
    out = marlinx.gemm(x, down.weight, down.weight_scale, down.weight_global_scale, 2688, 3712, zs=down._marlinx_zs)
    lib.marlinx_set_vrows(None)
    full = marlinx.gemm(x, down.weight, down.weight_scale, down.weight_global_scale, 2688, 3712, zs=down._marlinx_zs)
    sab = not torch.equal(out[1].view(torch.int16), full[1].view(torch.int16))  # row 1 is unverified: may differ
    print(f"VROWS_EXACT cases={cases} mismatches={bad} sabotage_row_beyond_prefix_differs={sab}")
    return 0 if bad == 0 and sab else 3


if __name__ == "__main__":
    sys.exit(main())

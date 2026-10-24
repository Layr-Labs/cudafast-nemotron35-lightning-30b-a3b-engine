# SPDX-License-Identifier: Apache-2.0
"""GPU-free AOT compile of the real Mamba replay kernel and its helpers.

Requires Triton 3.7.1. Run from any directory:
  python harness/tests/mamba_replay_compile.py --out /tmp/replay-aot \
      --depths 1 2 3 4 5 6 --matrix

--base-ref selects the unmodified tree for comparison; it must exist in git.
Both variants use the same production-shaped strides, launch and backend.
This emits PTX/cubin/IR for sm_121a; it does NOT execute kernels or establish
GPU bit identity, memory safety, determinism, or speed. Inspect each PTX with
CUDA 13.0.88 ptxas -arch=sm_121a -v for pinned-toolchain resource counts.
BF16/FP16 and the replay-state-output and z branches are compile checked.
"""

import argparse
import ast
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

ROOT = Path(__file__).resolve().parents[2]
LAB = None
REL = "vllm/vllm/model_executor/layers/mamba/ops/mamba_ssm_spec_replay.py"


def extract(src, name):
    node = next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = [
        ast.parse("@triton.jit\ndef f(): pass").body[0].decorator_list[0]
    ]
    return ast.unparse(node)


def compile_one(label, source, depth, dtype="bf16", store=False, z=False):
    text = "import triton\nimport triton.language as tl\n"
    text += (
        extract(
            (
                ROOT / "vllm/vllm/model_executor/layers/mamba/ops/triton_helpers.py"
            ).read_text(),
            "fast_exp",
        )
        + "\n"
    )
    text += (
        extract(
            (
                ROOT / "vllm/vllm/model_executor/layers/mamba/ops/mamba_ssm.py"
            ).read_text(),
            "softplus",
        )
        + "\n"
    )
    text += extract(source, "_selective_scan_update_spec_replay_kernel") + "\n"
    module_path = LAB / f"{label}_source.py"
    module_path.write_text(text)
    spec = importlib.util.spec_from_file_location(label, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = mod._selective_scan_update_spec_replay_kernel
    values = {
        "N": 1,
        "nheads": 64,
        "dim": 64,
        "dstate": 128,
        "nheads_ngroups_ratio": 8,
        "ngroups": 8,
        "ring_len": 2 * (depth + 1),
        "null_block_id": 0,
        "stride_state_batch": (6144 * (3 + depth) * 2 + 64 * 64 * 128 * 4 + 4096) // 4,
        "stride_state_head": 8192,
        "stride_state_dim": 128,
        "stride_state_dstate": 1,
        "stride_ring_batch": (6144 * (3 + depth) * 2 + 64 * 64 * 128 * 4 + 4096) // 2,
        "stride_x_batch": 10304,
        "stride_x_head": 64,
        "stride_x_dim": 1,
        "stride_dt_batch": 10304,
        "stride_dt_head": 1,
        "stride_dt_dim": 0,
        "stride_dt_bias_head": 1,
        "stride_dt_bias_dim": 0,
        "stride_A_head": 1,
        "stride_A_dim": 0,
        "stride_A_dstate": 0,
        "stride_B_batch": 10304,
        "stride_B_group": 128,
        "stride_B_dstate": 1,
        "stride_C_batch": 10304,
        "stride_C_group": 128,
        "stride_C_dstate": 1,
        "stride_D_head": 1,
        "stride_D_dim": 0,
        "stride_z_batch": 4096 if z else 0,
        "stride_z_head": 64 if z else 0,
        "stride_z_dim": 1 if z else 0,
        "stride_out_batch": 4096,
        "stride_out_head": 64,
        "stride_out_dim": 1,
        "stride_state_indices_batch": depth + 1,
        "stride_state_indices_T": 1,
        "DT_SOFTPLUS": True,
        "TIE_HDIM": True,
        "BLOCK_SIZE_M": 4,
        "HAS_DT_BIAS": True,
        "HAS_D": True,
        "HAS_Z": z,
        "STORE_REPLAYED": store,
        "BLOCK_SIZE_DSTATE": 128,
    }
    signature = {}
    constants = {}
    attrs = {}
    for i, name in enumerate(fn.arg_names):
        if name in values:
            v = values[name]
            if i in fn.constexprs or v == 1:
                constants[name] = v
                signature[name] = "constexpr"
            else:
                signature[name] = "i32"
                if v % 16 == 0:
                    attrs[(i,)] = [["tt.divisibility", 16]]
        else:
            if name in ("z_ptr", "replayed_ptr") and not (
                z if name == "z_ptr" else store
            ):
                constants[name] = None
                signature[name] = "constexpr"
            elif name in ("state_ptr", "A_ptr", "replayed_ptr"):
                signature[name] = "*fp32"
            elif name in (
                "state_batch_indices_ptr",
                "num_accepted_tokens_ptr",
                "cu_seqlens_ptr",
                "seq_lens_ptr",
            ):
                signature[name] = "*i32"
            else:
                signature[name] = "*" + dtype
            if name not in constants:
                attrs[(i,)] = [["tt.divisibility", 16]]
    kernel = triton.compile(
        ASTSource(fn, signature, constants, attrs),
        target=GPUTarget("cuda", 121, 32),
        options={"num_warps": 1, "ptx_version": 88},
    )
    out = LAB / "compiled" / f"{label}-d{depth}-{dtype}-s{int(store)}-z{int(z)}"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("ptx", "cubin", "ttgir", "llir"):
        data = kernel.asm[ext]
        (out / f"kernel.{ext}").write_bytes(
            data if isinstance(data, bytes) else data.encode()
        )
    ptx = kernel.asm["ptx"]
    report = {
        "label": label,
        "depth": depth,
        "dtype": dtype,
        "store": store,
        "z": z,
        "num_warps": kernel.metadata.num_warps,
        "ptx_lines": len(ptx.splitlines()),
        "rem": len(re.findall(r"\brem\.", ptx)),
        "div": len(re.findall(r"\bdiv\.", ptx)),
        "metadata": kernel.metadata._asdict(),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, default=str), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base-ref", default="2619f11")
    p.add_argument("--depths", nargs="+", type=int, default=[3])
    p.add_argument("--variants", nargs="+", default=["baseline", "candidate"])
    p.add_argument("--matrix", action="store_true")
    a = p.parse_args()
    LAB = a.out.resolve()
    LAB.mkdir(parents=True, exist_ok=True)
    for label in a.variants:
        src = (
            subprocess.check_output(
                ["git", "show", a.base_ref + ":" + REL], cwd=ROOT, text=True
            )
            if label == "baseline"
            else (ROOT / REL).read_text()
        )
        for d in a.depths:
            for dt, s, z in (
                [("bf16", False, False), ("fp16", True, False), ("bf16", True, True)]
                if a.matrix
                else [("bf16", False, False)]
            ):
                compile_one(label, src, d, dt, s, z)

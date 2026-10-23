#!/usr/bin/env python3
"""Bit-identity proof and timings for the tuned humming routed-expert MoE.

Checks vllm/vllm/model_executor/layers/fused_moe/experts/humming_tuned_moe.py
(unpadded experts plus explicit batch-invariant tuning configs) against the
stock path (weights zero-padded to N%256 and K%128, humming's heuristic BI
configs). Needs the GPU, the pinned vLLM image with this tree's overlay on
PYTHONPATH, and the checkpoint snapshot; on Atom run it with
`bin/atom nemo exec SHA harness/tests/humming_moe_exact.py --phase ...`.

Phases (--phase, comma-separated, default env,mma,exact,time):
  env    humming version and install, device, stock heuristic and tuned configs
  mma    exhaustive: mma.sync.m16n8k16.f32.bf16.bf16.f32 with A = B = +0 returns
         C bit for bit for all 2^32 fp32 C (single step and the 4-step chain that
         w2's K padding appended), compiled with humming's own NVRTC flags; also
         the epilogue (x * gs -> bf16 RN) on C and on the chain output
  exact  stock vs tuned layer on real weights: torch.equal (bitwise) on the w13
         output, the activation, the w2 output and the custom-op output, for
         M in --ms, random, embedding-based and edge-case activations
  sweep  candidate BI configs per M bucket: bitwise check + GEMM time each
  time   stock vs tuned: w13, w2 and full-op time per M (CUDA graphs over
         --layers distinct layers, so expert weights stream from DRAM)

Writes <phase>.json to $NEMO_OUT (or --out) and prints a summary.
"""

import argparse
import ctypes
import json
import math
import os
import statistics
import sys
import time
from types import SimpleNamespace

import torch

HIDDEN = 2688
INTER = 1856
NUM_EXPERTS = 128
TOP_K = 6
BIG = 1 << 30
MOE_LAYERS = [
    1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31, 34, 36, 38, 40, 43, 45,
    47, 49, 51,
]

OUT_DIR = None


def log(*args):
    print(*args, flush=True)


def dump(name, obj):
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    log(f"wrote {path}")


# ---------------------------------------------------------------------------
# checkpoint loading


class Snapshot:
    def __init__(self, path):
        self.path = path
        with open(os.path.join(path, "model.safetensors.index.json")) as f:
            self.weight_map = json.load(f)["weight_map"]

    def load(self, names, device):
        from safetensors import safe_open

        by_file = {}
        for name in names:
            by_file.setdefault(self.weight_map[name], []).append(name)
        out = {}
        for fname, fnames in by_file.items():
            with safe_open(os.path.join(self.path, fname), framework="pt") as f:
                for name in fnames:
                    out[name] = f.get_tensor(name).to(device)
        return out

    def raw_experts(self, layer_idx, device):
        pre = f"backbone.layers.{layer_idx}.mixer.experts"
        names = []
        for e in range(NUM_EXPERTS):
            for proj in ("up_proj", "down_proj"):
                for suffix in ("weight", "weight_scale", "weight_scale_2"):
                    names.append(f"{pre}.{e}.{proj}.{suffix}")
        t = self.load(names, device)

        def stack(proj, suffix):
            return torch.stack(
                [t[f"{pre}.{e}.{proj}.{suffix}"] for e in range(NUM_EXPERTS)]
            )

        raw = {
            # vLLM's ModelOptNvFp4FusedMoE parameter layout after loading
            "w13_weight": stack("up_proj", "weight"),
            "w13_weight_scale": stack("up_proj", "weight_scale"),
            "w13_weight_scale_2": stack("up_proj", "weight_scale_2").view(-1, 1),
            "w2_weight": stack("down_proj", "weight"),
            "w2_weight_scale": stack("down_proj", "weight_scale"),
            "w2_weight_scale_2": stack("down_proj", "weight_scale_2").view(-1),
        }
        assert raw["w13_weight"].shape == (NUM_EXPERTS, INTER, HIDDEN // 2)
        assert raw["w13_weight"].dtype == torch.uint8
        assert raw["w13_weight_scale"].dtype == torch.float8_e4m3fn
        assert raw["w2_weight"].shape == (NUM_EXPERTS, HIDDEN, INTER // 2)
        return raw

    def router(self, layer_idx, device):
        pre = f"backbone.layers.{layer_idx}"
        t = self.load(
            [
                f"{pre}.mixer.gate.weight",
                f"{pre}.mixer.gate.e_score_correction_bias",
                f"{pre}.norm.weight",
            ],
            device,
        )
        return (
            t[f"{pre}.mixer.gate.weight"].float(),
            t[f"{pre}.mixer.gate.e_score_correction_bias"].float(),
            t[f"{pre}.norm.weight"],
        )


# ---------------------------------------------------------------------------
# layers built through vLLM's own conversion


class FakeMoELayer(torch.nn.Module):
    """The attributes vLLM's humming MoE conversion and experts read."""

    def __init__(self, raw, name):
        super().__init__()
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation

        self.layer_name = name
        self.params_dtype = torch.bfloat16
        self.hidden_size = HIDDEN
        self.intermediate_size_per_partition = INTER
        self.local_num_experts = NUM_EXPERTS
        self.global_num_experts = NUM_EXPERTS
        self.moe_config = SimpleNamespace(
            has_bias=False,
            num_local_experts=NUM_EXPERTS,
            activation=MoEActivation.RELU2_NO_MUL,
            intermediate_size_per_partition=INTER,
            hidden_dim=HIDDEN,
        )
        for key, value in raw.items():
            self.register_parameter(
                key, torch.nn.Parameter(value.clone(), requires_grad=False)
            )


def build_variant(raw, name, tuned):
    """Convert one layer the way the engine does, with the switch on or off."""
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
        HummingIndexedExperts,
    )
    from vllm.model_executor.layers.quantization.utils.humming_utils import (
        convert_to_humming_moe_kernel_format,
    )

    os.environ["VLLM_HUMMING_MOE_FAST"] = "1" if tuned else "0"
    layer = FakeMoELayer(raw, name)
    convert_to_humming_moe_kernel_format(
        layer, quant_config={"quant_method": "modelopt", "quant_algo": "nvfp4"}
    )
    assert layer.humming_tuned_moe == tuned, (layer.humming_tuned_moe, tuned)
    ex = object.__new__(HummingIndexedExperts)
    ex.layer = layer
    ex.num_experts = NUM_EXPERTS
    ex.global_num_experts = NUM_EXPERTS
    ex.quant_config = SimpleNamespace(gemm1_clamp_limit=None)
    ex.moe_config = SimpleNamespace(
        activation_situ_beta=None, activation_situ_linear_beta=None
    )
    ex.max_num_tokens = None
    ex.num_dispatchers = None
    ex.init_humming_moe()
    # no forward context outside the engine: tokens * top_k, as with dp off
    ex.get_global_valid_shape_m = lambda topk_ids: topk_ids.size(0) * topk_ids.size(1)
    os.environ["VLLM_HUMMING_MOE_FAST"] = "1"
    return ex


def layer_bytes(ex):
    total = 0
    for name, p in ex.layer.named_parameters():
        if name.startswith(("w13_", "w2_")):
            total += p.numel() * p.element_size()
    return total


# ---------------------------------------------------------------------------
# running the op


def run_apply(ex, x, topk_ids, topk_weights):
    """HummingIndexedExperts.apply exactly as the modular kernel calls it."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    act = MoEActivation.RELU2_NO_MUL
    m = x.size(0)
    ws1_shape, ws2_shape, out_shape = ex._workspace_shapes(m, TOP_K, act)
    ws1 = torch.empty(max(math.prod(ws1_shape), math.prod(out_shape)),
                      dtype=torch.bfloat16, device=x.device)
    ws2 = torch.empty(math.prod(ws2_shape), dtype=torch.bfloat16, device=x.device)
    out = ws1[: math.prod(out_shape)].view(out_shape)
    ex.apply(
        output=out, hidden_states=x, w1=ex.layer.w13_weight,
        w2=ex.layer.w2_weight, topk_weights=topk_weights, topk_ids=topk_ids,
        activation=act, global_num_experts=NUM_EXPERTS, expert_map=None,
        a1q_scale=None, a2_scale=None, workspace13=ws1, workspace2=ws2,
        expert_tokens_meta=None, apply_router_weight_on_input=False,
    )
    return out


def run_stages(ex, x, topk_ids, topk_weights):
    """The same sequence as HummingIndexedExperts.apply, keeping intermediates."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import (
        moe_fused_mul_sum,
    )
    from vllm.utils.humming import HummingMethod

    act = MoEActivation.RELU2_NO_MUL
    m = x.size(0)
    kw1, kw2 = ex.prepare_humming_moe_kwargs(topk_ids, None, None)
    gate_up = torch.empty(m * TOP_K, INTER, dtype=torch.bfloat16, device=x.device)
    HummingMethod.forward_layer(layer=ex.layer, inputs=x, input_scale=None,
                                outputs=gate_up, sublayer_name="w13", **kw1)
    gate_up_copy = gate_up.clone()
    act_out = torch.empty_like(gate_up)
    ex.apply_activation(activation=act, input=gate_up, output=act_out)
    down = torch.empty(m * TOP_K, HIDDEN, dtype=torch.bfloat16, device=x.device)
    HummingMethod.forward_layer(layer=ex.layer, inputs=act_out, input_scale=None,
                                outputs=down, sublayer_name="w2", **kw2)
    out = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device=x.device)
    moe_fused_mul_sum(inputs=down.view(m, TOP_K, -1), topk_weights=topk_weights,
                      topk_ids=topk_ids, expert_map=None, outputs=out)
    return {"w13": gate_up_copy, "act": act_out, "w2": down, "out": out}


def bits_equal(a, b):
    return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))


def diff_count(a, b):
    return int((a.view(torch.int16) != b.view(torch.int16)).sum().item())


# ---------------------------------------------------------------------------
# inputs


def rms_norm(x, w, eps=1e-5):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * w.float()).to(torch.bfloat16)


def route(x, gate_w, bias):
    """Sigmoid scores, bias for selection, top-6, renormalised (vLLM grouped_topk
    with n_group = topk_group = 1). Only feeds the test; both variants get the
    same ids and weights."""
    logits = torch.nn.functional.linear(x.float(), gate_w)
    scores = logits.sigmoid()
    ids = (scores + bias).topk(TOP_K, dim=-1).indices
    w = scores.gather(1, ids)
    w = w / w.sum(-1, keepdim=True)
    return ids.to(torch.int32).contiguous(), w.float().contiguous()


def random_route(m, gen, device):
    ids = torch.stack([torch.randperm(NUM_EXPERTS, generator=gen)[:TOP_K]
                       for _ in range(m)]).to(torch.int32).to(device)
    w = torch.rand(m, TOP_K, generator=gen).float() + 0.05
    w = (w / w.sum(-1, keepdim=True)).to(device)
    return ids.contiguous(), w.contiguous()


def make_inputs(kind, m, seed, emb, router, device):
    gen = torch.Generator().manual_seed(seed)
    gate_w, bias, norm_w = router
    if kind == "emb":
        ids = torch.randint(0, emb.size(0), (m,), generator=gen).to(device)
        x = rms_norm(emb[ids], norm_w)
        tid, tw = route(x, gate_w, bias)
    elif kind == "rand":
        x = torch.randn(m, HIDDEN, generator=gen).to(torch.bfloat16).to(device)
        tid, tw = route(x, gate_w, bias)
    elif kind == "rand_uniform_route":
        x = (torch.randn(m, HIDDEN, generator=gen) * 3).to(torch.bfloat16).to(device)
        tid, tw = random_route(m, gen, device)
    elif kind == "edge":
        x = torch.randn(m, HIDDEN, generator=gen)
        for r in range(m):
            k = (r + seed) % 6
            if k == 0:
                x[r] = 0.0  # all-zero activations downstream
            elif k == 1:
                x[r] = x[r] * 1e3  # large values
            elif k == 2:
                x[r] = -x[r - 1] if r >= 1 else -x[r]
            elif k == 3:
                x[r] = x[r] * 1e-19  # relu^2 lands in bf16 subnormals
            elif k == 4:
                x[r] = x[r].abs() * -1.0  # one-signed input
        x = x.to(torch.bfloat16).to(device)
        tid, tw = random_route(m, gen, device)
    else:
        raise ValueError(kind)
    return x.contiguous(), tid, tw


# ---------------------------------------------------------------------------
# phases


def phase_env(args, ctx):
    import humming
    from humming.config import GemmType
    from humming.tune import get_heuristics_class

    from vllm.model_executor.layers.fused_moe.experts import humming_tuned_moe as htm

    props = torch.cuda.get_device_properties(0)
    hum_dir = os.path.dirname(humming.__file__)
    info = {
        "torch": torch.__version__,
        "humming_version": getattr(humming, "__version__", None),
        "humming_dir": hum_dir,
        "device": props.name,
        "sm": f"{props.major}.{props.minor}",
        "num_sms": props.multi_processor_count,
        "heuristics": get_heuristics_class().__name__,
        "tuned_configs": {f"{k}": v for k, v in htm.TUNED_CONFIGS.items()},
    }
    try:
        from importlib.metadata import version

        info["humming_dist_version"] = version("humming-kernels")
    except Exception as e:  # noqa
        info["humming_dist_version"] = repr(e)
    import hashlib

    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(hum_dir)):
        for fn in sorted(files):
            if fn.endswith((".py", ".cuh", ".h", ".cpp")):
                p = os.path.join(root, fn)
                h.update(os.path.relpath(p, hum_dir).encode())
                with open(p, "rb") as f:
                    h.update(f.read())
    info["humming_source_sha256"] = h.hexdigest()
    ex_stock, ex_tuned = ctx["pairs"][0]
    info["stock_w13"] = ex_stock.w13_tuning_config
    info["stock_w2"] = ex_stock.w2_tuning_config
    info["tuned_w13"] = ex_tuned.w13_tuning_config
    info["tuned_w2"] = ex_tuned.w2_tuning_config
    info["stock_meta_w13"] = ex_stock.layer.humming_metas["w13"].to_str()
    info["stock_meta_w2"] = ex_stock.layer.humming_metas["w2"].to_str()
    info["tuned_meta_w13"] = ex_tuned.layer.humming_metas["w13"].to_str()
    info["tuned_meta_w2"] = ex_tuned.layer.humming_metas["w2"].to_str()
    info["stock_layer_bytes"] = layer_bytes(ex_stock)
    info["tuned_layer_bytes"] = layer_bytes(ex_tuned)
    del GemmType
    for k, v in info.items():
        log(f"{k}: {v}")
    dump("env", info)


MMA_PROBE = r"""
// humming's epilogue conversion: __float22bfloat162_rn is cvt.rn.bf16x2.f32
__device__ __forceinline__ unsigned int to_bf16x2(float lo, float hi) {
  unsigned int r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

__device__ __forceinline__ bool is_nan_bits(unsigned int u) {
  return (u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0u;
}

__device__ __forceinline__ void sample(unsigned int *buf, unsigned int a,
                                       unsigned int b, unsigned int c) {
  unsigned int slot = atomicAdd(buf, 1u);
  if (slot < 32) {
    buf[1 + 3 * slot] = a; buf[2 + 3 * slot] = b; buf[3 + 3 * slot] = c;
  }
}

// counts: 0 non-NaN C with mma(+0,+0,C) != C; 1 same for the 4-step chain;
// 2 NaN inputs; 3 NaN inputs whose output is not NaN; 4 NaN outputs that are
// not 0x7fffffff; 5 epilogue (x * gs -> bf16 RN) differs between C and the
// 4-step chain output; 6 same, restricted to non-NaN C.
extern "C" __global__ void zero_mma_probe(
    const unsigned int *__restrict__ zeros, unsigned long long base,
    unsigned int iters, float gs, unsigned long long *counts,
    unsigned int *samp_bad, unsigned int *samp_nan, unsigned int *samp_epi) {
  // A and B fragments of mma.m16n8k16 (bf16): 4 + 2 registers, all +0, read
  // from memory so nothing can fold them.
  unsigned int a0 = zeros[0], a1 = zeros[1], a2 = zeros[2], a3 = zeros[3];
  unsigned int b0 = zeros[4], b1 = zeros[5];
  unsigned long long nthreads = (unsigned long long)gridDim.x * blockDim.x;
  unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long n[7] = {0, 0, 0, 0, 0, 0, 0};
  for (unsigned int it = 0; it < iters; ++it) {
    unsigned long long idx = base + ((unsigned long long)it * nthreads + tid) * 4ull;
    unsigned int c[4];
    float cf[4], d[4], e[4];
    for (int i = 0; i < 4; ++i) {
      c[i] = (unsigned int)(idx + i);
      cf[i] = __uint_as_float(c[i]);
      e[i] = cf[i];
    }
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(cf[0]), "f"(cf[1]), "f"(cf[2]), "f"(cf[3]));
    // the four trailing zero K-groups the padded w2 chain appended
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(e[0]), "+f"(e[1]), "+f"(e[2]), "+f"(e[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    for (int i = 0; i < 4; ++i) {
      unsigned int du = __float_as_uint(d[i]);
      unsigned int eu = __float_as_uint(e[i]);
      bool nan_in = is_nan_bits(c[i]);
      if (!nan_in) {
        if (du != c[i]) { n[0]++; sample(samp_bad, c[i], du, eu); }
        if (eu != c[i]) n[1]++;
      } else {
        n[2]++;
        if (!is_nan_bits(du) || !is_nan_bits(eu)) n[3]++;
        if (du != 0x7fffffffu || eu != 0x7fffffffu) { n[4]++; sample(samp_nan, c[i], du, eu); }
      }
      // humming's epilogue: one fp32 multiply by gs*4, then RN to bf16
      unsigned int p = to_bf16x2(cf[i] * gs, cf[i] * gs);
      unsigned int q = to_bf16x2(e[i] * gs, e[i] * gs);
      if (p != q) {
        n[5]++;
        if (!nan_in) n[6]++;
        sample(samp_epi, c[i], p, q);
      }
    }
  }
  for (int i = 0; i < 7; ++i) atomicAdd(&counts[i], n[i]);
}

// One warp, one mma with every lane's A/B/C registers taken from memory: which
// NaN bit patterns can an MMA itself emit?
extern "C" __global__ void mma_one(const unsigned int *__restrict__ abc,
                                   unsigned int *out) {
  unsigned int a0 = abc[0], a1 = abc[1], a2 = abc[2], a3 = abc[3];
  unsigned int b0 = abc[4], b1 = abc[5];
  float c0 = __uint_as_float(abc[6]), c1 = __uint_as_float(abc[7]);
  float c2 = __uint_as_float(abc[8]), c3 = __uint_as_float(abc[9]);
  float d0, d1, d2, d3;
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3));
  out[threadIdx.x * 4 + 0] = __float_as_uint(d0);
  out[threadIdx.x * 4 + 1] = __float_as_uint(d1);
  out[threadIdx.x * 4 + 2] = __float_as_uint(d2);
  out[threadIdx.x * 4 + 3] = __float_as_uint(d3);
}
"""


def _cu_func(cbd, path, name):
    err, lib = cbd.cuLibraryLoadFromFile(path.encode(), [], [], 0, [], [], 0)
    assert int(err) == 0, err
    err, kern = cbd.cuLibraryGetKernel(lib, name)
    assert int(err) == 0, err
    err, func = cbd.cuKernelGetFunction(kern)
    assert int(err) == 0, err
    return func


def _launch(cbd, func, grid, block, values, types):
    stream = cbd.CUstream(torch.cuda.current_stream().cuda_stream)
    (err,) = cbd.cuLaunchKernel(func, grid, 1, 1, block, 1, 1, 0, stream,
                                (values, types), 0)
    assert int(err) == 0, err


def _samples(t):
    s = [v & 0xFFFFFFFF for v in t.tolist()]
    n = min(s[0], 32)
    return [tuple(hex(s[1 + 3 * i + j]) for j in range(3)) for i in range(n)]


def phase_mma(args, ctx):
    import cuda.bindings.driver as cbd
    from humming.jit.compiler import NVRTCCompiler

    props = torch.cuda.get_device_properties(0)
    sm = f"{props.major}{props.minor}" + ("a" if props.major * 10 + props.minor >= 90 else "")
    path = NVRTCCompiler.compile(MMA_PROBE, sm, None)
    flags = NVRTCCompiler.get_flags(sm)
    probe = _cu_func(cbd, path, b"zero_mma_probe")
    one = _cu_func(cbd, path, b"mma_one")

    dev = "cuda"
    zeros = torch.zeros(8, dtype=torch.int32, device=dev)
    blocks, threads = 1024, 256
    nthreads = blocks * threads
    per_launch = 1 << 28  # fp32 patterns per launch
    iters = per_launch // (4 * nthreads)
    assert iters * 4 * nthreads == per_launch, (iters, nthreads)
    results = []
    for gs in args.mma_gs:
        counts = torch.zeros(7, dtype=torch.int64, device=dev)
        sb = torch.zeros(97, dtype=torch.int32, device=dev)
        sn = torch.zeros(97, dtype=torch.int32, device=dev)
        se = torch.zeros(97, dtype=torch.int32, device=dev)
        t0 = time.time()
        for chunk in range((1 << 32) // per_launch):
            _launch(cbd, probe, blocks, threads,
                    (zeros.data_ptr(), chunk * per_launch, iters, gs, counts.data_ptr(),
                     sb.data_ptr(), sn.data_ptr(), se.data_ptr()),
                    (ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_uint, ctypes.c_float,
                     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p))
        torch.cuda.synchronize()
        c = counts.tolist()
        res = {
            "gs": gs,
            "patterns": 1 << 32,
            "nonnan_single_step_mismatch": c[0],
            "nonnan_four_step_mismatch": c[1],
            "nan_inputs": c[2],
            "nan_in_non_nan_out": c[3],
            "nan_out_not_canonical": c[4],
            "epilogue_mismatch": c[5],
            "epilogue_mismatch_nonnan": c[6],
            "samples_nonnan_mismatch": _samples(sb),
            "samples_nan_not_canonical": _samples(sn),
            "samples_epilogue_mismatch": _samples(se),
            "seconds": round(time.time() - t0, 3),
        }
        log(f"mma probe gs={gs}: " + json.dumps({k: v for k, v in res.items()
                                                 if not k.startswith("samples")}))
        for k in ("samples_nonnan_mismatch", "samples_nan_not_canonical",
                  "samples_epilogue_mismatch"):
            if res[k]:
                log(f"   {k}: {res[k][:8]}")
        results.append(res)

    # NaN sources inside an MMA: NaN in A, inf * 0, inf - inf, NaN in C
    def bf2(lo, hi):
        return (hi << 16) | lo

    cases = {
        "A=qNaN,B=1,C=0": ([bf2(0x7fc0, 0x7fc0)] * 4, [bf2(0x3f80, 0x3f80)] * 2, [0] * 4),
        "A=-NaN(payload),B=1,C=0": ([bf2(0xffc1, 0xffc1)] * 4, [bf2(0x3f80, 0x3f80)] * 2, [0] * 4),
        "A=sNaN,B=1,C=1": ([bf2(0x7f81, 0x7f81)] * 4, [bf2(0x3f80, 0x3f80)] * 2, [0x3f800000] * 4),
        "A=inf,B=0,C=0": ([bf2(0x7f80, 0x7f80)] * 4, [0] * 2, [0] * 4),
        "A=inf,B=-0,C=1": ([bf2(0x7f80, 0x7f80)] * 4, [bf2(0x8000, 0x8000)] * 2, [0x3f800000] * 4),
        "A=inf,B=1,C=-inf": ([bf2(0x7f80, 0x7f80)] * 4, [bf2(0x3f80, 0x3f80)] * 2, [0xff800000] * 4),
        "A=1,B=1,C=-NaN(payload)": ([bf2(0x3f80, 0x3f80)] * 4, [bf2(0x3f80, 0x3f80)] * 2, [0xffc00123] * 4),
        "A=+inf/-inf mix,B=1,C=0": ([bf2(0x7f80, 0xff80)] * 4, [bf2(0x3f80, 0x3f80)] * 2, [0] * 4),
    }
    nan_sources = {}
    for name, (a, b, cc) in cases.items():
        words = [v & 0xFFFFFFFF for v in a + b + cc]
        abc = torch.tensor([w - (1 << 32) if w >= 1 << 31 else w for w in words],
                           dtype=torch.int32, device=dev)
        outb = torch.zeros(128, dtype=torch.int32, device=dev)
        _launch(cbd, one, 1, 32, (abc.data_ptr(), outb.data_ptr()),
                (ctypes.c_void_p, ctypes.c_void_p))
        torch.cuda.synchronize()
        vals = sorted({hex(v & 0xFFFFFFFF) for v in outb.tolist()})
        nan_sources[name] = vals
        log(f"mma NaN source {name}: outputs {vals}")
    emitted_nans = sorted({v for vals in nan_sources.values() for v in vals
                           if (int(v, 16) & 0x7f800000) == 0x7f800000 and int(v, 16) & 0x7fffff})
    holds = all(r["nonnan_single_step_mismatch"] == 0 and r["nonnan_four_step_mismatch"] == 0
                and r["epilogue_mismatch_nonnan"] == 0 and r["nan_in_non_nan_out"] == 0
                for r in results)
    out = {"cubin": path, "nvrtc_flags": flags, "sm": sm, "results": results,
           "nan_sources": nan_sources, "emitted_nan_patterns": emitted_nans,
           "nonnan_identity_holds": holds}
    log(f"mma(+0,+0,C) == C for every non-NaN fp32 C (single step, 4-step chain, "
        f"epilogue): {holds}; NaN patterns an MMA emitted: {emitted_nans}")
    dump("mma", out)


def phase_act(args, ctx):
    """Fused relu^2 vs vLLM's F.relu(inplace) + torch.square, all bf16 inputs."""
    from vllm.model_executor.layers.fused_moe.activation import (
        MoEActivation,
        apply_moe_activation,
    )
    from vllm.model_executor.layers.fused_moe.experts.humming_tuned_moe import (
        relu2_bf16,
    )

    dev = "cuda"
    allbits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    cases = {
        "all_65536_bf16": allbits.view(torch.bfloat16).to(dev),
        "tail_37x1856": torch.cat([allbits.view(torch.bfloat16),
                                   torch.randn(37 * 1856 - 65536).to(torch.bfloat16)]
                                  ).view(37, 1856).to(dev),
        "rand_6x1856": (torch.randn(6, 1856) * 40).to(torch.bfloat16).to(dev),
    }
    rows = {}
    ok_all = True
    for name, x in cases.items():
        ref_in2 = x.clone().view(-1, x.shape[-1]) if x.dim() == 2 else x.clone().view(1, -1)
        ref2 = torch.empty_like(ref_in2)
        apply_moe_activation(MoEActivation.RELU2_NO_MUL, ref2, ref_in2)
        out = torch.empty_like(ref2)
        applied = relu2_bf16(x.clone().view_as(ref2), out)
        torch.cuda.synchronize()
        eq = applied and bits_equal(out, ref2)
        nd = 0 if eq else diff_count(out, ref2)
        samples = []
        if not eq:
            bad = (out.view(torch.int16) != ref2.view(torch.int16)).nonzero()[:8]
            src = x.reshape(ref2.shape)
            for r, c in bad.tolist():
                samples.append((hex(src[r, c].view(torch.int16).item() & 0xFFFF),
                                hex(ref2[r, c].view(torch.int16).item() & 0xFFFF),
                                hex(out[r, c].view(torch.int16).item() & 0xFFFF)))
        rows[name] = {"applied": applied, "bit_identical": eq, "diff": nd, "samples": samples}
        ok_all &= eq
        log(f"act {name}: {rows[name]}")
    log(f"fused relu^2 bit-identical to F.relu+torch.square: {ok_all}")
    dump("act", {"ok": ok_all, "cases": rows})
    ctx["act_ok"] = ok_all


def phase_exact(args, ctx):
    dev = "cuda"
    kinds = args.kinds
    results = []
    all_ok = True
    for li, (ex_stock, ex_tuned) in enumerate(ctx["pairs"]):
        layer_idx = ctx["layer_ids"][li]
        router = ctx["routers"][li]
        for m in args.ms:
            for kind in kinds:
                for rep in range(args.reps if m <= 64 else max(1, args.reps // 2)):
                    seed = (1000003 * layer_idx + 7919 * m + 104729 * rep
                            + sum(map(ord, kind)))
                    x, tid, tw = make_inputs(kind, m, seed, ctx["emb"], router, dev)
                    a = run_stages(ex_stock, x, tid, tw)
                    b = run_stages(ex_tuned, x, tid, tw)
                    real_a = run_apply(ex_stock, x, tid, tw)
                    real_b = run_apply(ex_tuned, x, tid, tw)
                    torch.cuda.synchronize()
                    row = {"layer": layer_idx, "m": m, "kind": kind, "rep": rep,
                           "distinct_experts": int(tid.unique().numel())}
                    ok = True
                    for key in ("w13", "act", "w2", "out"):
                        eq = bits_equal(a[key], b[key])
                        row[key] = eq if eq else diff_count(a[key], b[key])
                        ok &= eq
                    row["apply_stock_eq_stages"] = bits_equal(real_a, a["out"])
                    row["apply_tuned_eq_stages"] = bits_equal(real_b, b["out"])
                    row["apply_eq"] = bits_equal(real_a, real_b)
                    ok &= row["apply_eq"] and row["apply_stock_eq_stages"] and row["apply_tuned_eq_stages"]
                    row["neg_zero_w2"] = int((b["w2"].view(torch.int16) == -32768).sum().item())
                    row["nonfinite_out"] = int((~torch.isfinite(b["out"].float())).sum().item())
                    row["ok"] = ok
                    all_ok &= ok
                    results.append(row)
                    if not ok or rep == 0:
                        log(f"exact L{layer_idx} M={m} {kind} rep{rep}: "
                            + " ".join(f"{k}={row[k]}" for k in row if k not in ("layer", "m", "kind", "rep")))
    summary = {}
    for r in results:
        s = summary.setdefault(r["m"], {"cases": 0, "ok": 0})
        s["cases"] += 1
        s["ok"] += int(r["ok"])
    log(f"exact summary per M: {summary}")
    log(f"ALL BIT-IDENTICAL: {all_ok}")
    dump("exact", {"all_ok": all_ok, "summary": summary, "rows": results})
    if not all_ok:
        sys.exit(3)


def graph_time(fn, reps, warmup=2):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    times = []
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        st.record()
        g.replay()
        en.record()
        en.synchronize()
        times.append(st.elapsed_time(en))
    del g
    return times


class OpBench:
    """Per-M inputs for every loaded layer, plus per-variant callables."""

    def __init__(self, ctx, m, seed_base=17):
        self.ctx = ctx
        self.m = m
        self.inputs = []
        for li, router in enumerate(ctx["routers"]):
            seed = seed_base + 31 * li + 1009 * m
            self.inputs.append(make_inputs("emb", m, seed, ctx["emb"], router, "cuda"))
        self.distinct = [int(t[1].unique().numel()) for t in self.inputs]

    def full_op(self, which):
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation

        act = MoEActivation.RELU2_NO_MUL
        calls = []
        for li, (x, tid, tw) in enumerate(self.inputs):
            ex = self.ctx["pairs"][li][which]
            ws1_shape, ws2_shape, out_shape = ex._workspace_shapes(self.m, TOP_K, act)
            ws1 = torch.empty(max(math.prod(ws1_shape), math.prod(out_shape)),
                              dtype=torch.bfloat16, device="cuda")
            ws2 = torch.empty(math.prod(ws2_shape), dtype=torch.bfloat16, device="cuda")
            out = ws1[: math.prod(out_shape)].view(out_shape)
            calls.append((ex, x, tid, tw, ws1, ws2, out))

        def fn():
            for ex, x, tid, tw, ws1, ws2, out in calls:
                ex.apply(output=out, hidden_states=x, w1=None, w2=None,
                         topk_weights=tw, topk_ids=tid, activation=act,
                         global_num_experts=NUM_EXPERTS, expert_map=None,
                         a1q_scale=None, a2_scale=None, workspace13=ws1,
                         workspace2=ws2, expert_tokens_meta=None,
                         apply_router_weight_on_input=False)

        return fn

    def gemm(self, which, sublayer, tuning_config=None, block_m=None):
        """One sublayer GEMM per layer. tuning_config None = the variant's own."""
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )
        from vllm.utils.humming import HummingMethod

        calls = []
        for li, (x, tid, tw) in enumerate(self.inputs):
            ex = self.ctx["pairs"][li][which]
            kw1, kw2 = ex.prepare_humming_moe_kwargs(tid, None, None)
            kw = dict(kw1 if sublayer == "w13" else kw2)
            if tuning_config is not None:
                sorted_ids, expert_ids, ntp = moe_align_block_size(
                    topk_ids=tid, block_size=block_m, num_experts=NUM_EXPERTS,
                    expert_map=None, ignore_invalid_experts=True)
                kw.update(sorted_ids=sorted_ids, expert_ids=expert_ids,
                          num_tokens_padded=ntp, tuning_config=tuning_config)
            if sublayer == "w13":
                inp = x
                out = torch.empty(self.m * TOP_K, INTER, dtype=torch.bfloat16, device="cuda")
            else:
                inp = self.ctx["act_cache"][(li, self.m)]
                out = torch.empty(self.m * TOP_K, HIDDEN, dtype=torch.bfloat16, device="cuda")
            calls.append((ex.layer, inp, out, kw))

        def fn():
            for layer, inp, out, kw in calls:
                HummingMethod.forward_layer(layer=layer, inputs=inp, input_scale=None,
                                            outputs=out, sublayer_name=sublayer, **kw)

        return fn, [c[2] for c in calls]


def prepare_reference(ctx, bench):
    """Stock stage outputs per layer (the bitwise reference for the sweep)."""
    for li, (x, tid, tw) in enumerate(bench.inputs):
        ref = run_stages(ctx["pairs"][li][0], x, tid, tw)
        ctx["act_cache"][(li, bench.m)] = ref["act"]
        ctx["ref_cache"][(li, bench.m)] = ref


def phase_time(args, ctx):
    rows = []
    for m in args.ms:
        bench = OpBench(ctx, m)
        prepare_reference(ctx, bench)
        L = len(bench.inputs)
        row = {"m": m, "layers": L, "distinct_experts": bench.distinct}
        fns = {}
        for which, name in ((0, "stock"), (1, "tuned")):
            fns[f"{name}_op"] = bench.full_op(which)
            fns[f"{name}_w13"] = bench.gemm(which, "w13")[0]
            fns[f"{name}_w2"] = bench.gemm(which, "w2")[0]
        times = {k: [] for k in fns}
        for _ in range(args.rounds):  # interleave stock and tuned
            for k, fn in fns.items():
                times[k] += graph_time(fn, args.reps)
        for k, ts in times.items():
            row[k + "_us"] = round(statistics.median(ts) * 1000 / L, 2)
        for part in ("op", "w13", "w2"):
            row[f"{part}_speedup"] = round(row[f"stock_{part}_us"] / row[f"tuned_{part}_us"], 4)
        log(f"time M={m}: " + " ".join(f"{k}={v}" for k, v in row.items() if k not in ("m",)))
        rows.append(row)
    dump("time", {"rows": rows})


def candidate_configs(sublayer, m, quick=False):
    """BI-safe candidates: block_k = warp_k = 32, warp_m % 16 == 0, no stream-K."""
    n = INTER if sublayer == "w13" else HIDDEN
    if m * TOP_K <= 48:
        bms = [(16, 16)]
        stages = [3, 4, 6, 8] if not quick else [4]
        ctas = [1, 2, 4, 8] if not quick else [4]
    else:
        bms = [(16, 16), (32, 16), (32, 32), (48, 48), (64, 32), (64, 64), (96, 48), (128, 64)]
        stages = [2, 3, 4]
        ctas = [1, 2, 4]
    # block_n must be a multiple of 64: humming packs the weights in 64-column
    # groups, and block_n = 32 gives wrong results (caught by this sweep's
    # bitwise check) or illegal memory accesses.
    bns = [(64, 64), (64, 32)]
    if n % 128 == 0:
        bns = [(128, 64), (128, 32)] + bns
    out = []
    for bm, wm in bms:
        for bn, wn in bns:
            if n % bn:
                continue
            for st in stages:
                for c in ctas:
                    threads = (bm // wm) * (bn // wn) * 32
                    if threads * c > 1536:
                        continue
                    # humming SharedStorage: stages (a, b, bs; 128-aligned)
                    # unioned with the bf16 output tile, 1024-aligned, plus
                    # the two row-index arrays of the indexed gemm
                    a = bm * 32 * 2
                    b = bn * 32 // 2
                    bs = bn * 2
                    stage = sum(math.ceil(v / 128) * 128 for v in (a, b, bs))
                    smem = math.ceil(max(st * stage, bm * bn * 2) / 1024) * 1024 + bm * 8
                    if smem * c > 99 * 1024:
                        continue
                    out.append({
                        "block_shape": [bm, bn, 32], "warp_shape": [wm, wn, 32],
                        "use_stream_k": False, "use_f16_accum": False, "num_sms": 48,
                        "num_stages": st, "num_ctas_per_sm": c, "num_write_splits": 1,
                    })
    return out


def precompile(ctx, sublayer, configs):
    """Compile every candidate in parallel through humming's own builder."""
    from humming.kernel.humming import HummingKernel

    ex = ctx["pairs"][0][1]
    meta = ex.layer.humming_metas[sublayer]
    t0 = time.time()
    good = []
    step = 32
    for start in range(0, len(configs), step):
        chunk = configs[start:start + step]
        lst = [[i, i + 1, c] for i, c in enumerate(chunk)]
        try:
            HummingKernel.prepare_kernels(meta.to_str(), ex.compute_config_str, json.dumps(lst))
            good += chunk
            continue
        except Exception as e:  # noqa
            log(f"batch compile failed ({repr(e)[:120]}); compiling one by one")
        for c in chunk:
            try:
                HummingKernel.prepare_kernels(meta.to_str(), ex.compute_config_str,
                                              json.dumps([[0, BIG, c]]))
                good.append(c)
            except Exception as e:  # noqa
                log(f"   skip {c['block_shape']} {c['warp_shape']} st{c['num_stages']} "
                    f"ctas{c['num_ctas_per_sm']}: {repr(e)[:120]}")
    log(f"compiled {len(good)}/{len(configs)} {sublayer} candidates in {time.time() - t0:.1f}s")
    return good


def phase_sweep(args, ctx):
    rows = []
    for sublayer in args.sublayers:
        for m in args.ms:
            bench = OpBench(ctx, m)
            prepare_reference(ctx, bench)
            L = len(bench.inputs)
            configs = candidate_configs(sublayer, m, quick=args.quick)
            configs = precompile(ctx, sublayer, configs)
            stock_fn, _ = bench.gemm(0, sublayer)
            tuned_fn, _ = bench.gemm(1, sublayer)
            base = {}
            for name, fn in (("stock", stock_fn), ("tuned_table", tuned_fn)):
                base[name] = statistics.median(graph_time(fn, args.reps)) * 1000 / L
            results = []
            for cfg in configs:
                tc = json.dumps([[0, BIG, cfg]])
                fn, outs = bench.gemm(1, sublayer, tuning_config=tc, block_m=cfg["block_shape"][0])
                try:
                    fn()
                    torch.cuda.synchronize()
                except Exception as e:  # noqa
                    results.append({"cfg": cfg, "error": repr(e)[:200]})
                    continue
                exact = all(bits_equal(outs[li], ctx["ref_cache"][(li, m)][sublayer])
                            for li in range(L))
                us = statistics.median(graph_time(fn, args.reps)) * 1000 / L
                results.append({"cfg": cfg, "us": round(us, 2), "exact": exact})
            ok = [r for r in results if "us" in r]
            ok.sort(key=lambda r: r["us"])
            inexact = [r for r in results if r.get("exact") is False]
            row = {"sublayer": sublayer, "m": m, "stock_us": round(base["stock"], 2),
                   "tuned_table_us": round(base["tuned_table"], 2),
                   "n_configs": len(configs), "n_inexact": len(inexact),
                   "n_errors": len([r for r in results if "error" in r]),
                   "top": ok[:8], "inexact": inexact[:5]}
            log(f"sweep {sublayer} M={m}: stock={row['stock_us']}us table={row['tuned_table_us']}us "
                f"inexact={len(inexact)} errors={row['n_errors']}")
            for r in ok[:8]:
                c = r["cfg"]
                log(f"   {r['us']:8.2f}us exact={r['exact']} block={c['block_shape']} warp={c['warp_shape']} "
                    f"stages={c['num_stages']} ctas={c['num_ctas_per_sm']}")
            rows.append(row)
            dump(f"sweep-{sublayer}", {"rows": rows})
    dump("sweep", {"rows": rows})


# ---------------------------------------------------------------------------


def main():
    global OUT_DIR
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="env,mma,exact,time")
    p.add_argument("--layers", default="1,13,27,40,51",
                   help="MoE layer indices to load (exact uses all, time rotates over them)")
    p.add_argument("--ms", default="1,2,3,4,5,8,496")
    p.add_argument("--kinds", default="emb,rand,rand_uniform_route,edge")
    p.add_argument("--reps", type=int, default=4)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--sublayers", default="w13,w2")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--mma-gs", default="4.0,0.0009765625,3.1e-05,1.7")
    p.add_argument("--out", default=os.environ.get("NEMO_OUT", "."))
    args = p.parse_args()
    args.ms = [int(v) for v in args.ms.split(",")]
    args.kinds = args.kinds.split(",")
    args.sublayers = args.sublayers.split(",")
    args.mma_gs = [float(v) for v in args.mma_gs.split(",")]
    phases = args.phase.split(",")
    OUT_DIR = args.out
    os.makedirs(OUT_DIR, exist_ok=True)
    os.environ.setdefault("VLLM_HUMMING_BATCH_INVARIANT", "1")
    log(f"args: {vars(args)}")
    torch.cuda.set_device(0)

    ctx = {"act_cache": {}, "ref_cache": {}}
    if "mma" in phases:
        phase_mma(args, ctx)
    if "act" in phases:
        phase_act(args, ctx)
    needs_layers = any(ph in phases for ph in ("env", "exact", "time", "sweep"))
    if needs_layers:
        from vllm.model_executor.layers.fused_moe.experts import humming_tuned_moe  # noqa

        snap = Snapshot(os.environ["NEMO_SNAPSHOT"])
        emb = snap.load(["backbone.embeddings.weight"], "cuda")["backbone.embeddings.weight"]
        ctx["emb"] = emb
        ctx["layer_ids"] = [int(v) for v in args.layers.split(",")]
        ctx["pairs"], ctx["routers"] = [], []
        for layer_idx in ctx["layer_ids"]:
            assert layer_idx in MOE_LAYERS, layer_idx
            t0 = time.time()
            raw = snap.raw_experts(layer_idx, "cuda")
            stock = build_variant(raw, f"backbone.layers.{layer_idx}.mixer.experts", False)
            tuned = build_variant(raw, f"backbone.layers.{layer_idx}.mixer.experts", True)
            del raw
            torch.cuda.empty_cache()
            ctx["pairs"].append((stock, tuned))
            ctx["routers"].append(snap.router(layer_idx, "cuda"))
            log(f"layer {layer_idx}: stock {layer_bytes(stock)/1e6:.1f} MB, "
                f"tuned {layer_bytes(tuned)/1e6:.1f} MB, {time.time()-t0:.1f}s")
    for ph, fn in (("env", phase_env), ("exact", phase_exact),
                   ("sweep", phase_sweep), ("time", phase_time)):
        if ph in phases:
            fn(args, ctx)
    log("done")


if __name__ == "__main__":
    main()

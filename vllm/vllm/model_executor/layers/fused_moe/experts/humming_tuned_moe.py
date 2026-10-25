# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LOCAL PATCH (cudafast-nemotron35, vllm/VENDOR.json "local_patches"):
unpadded routed-expert weights and explicit batch-invariant tuning configs
for the humming indexed MoE on GB10.

In batch-invariant mode with BF16 activations, humming computes every output
element as ONE fp32 accumulator chain: acc = +0, then for each 16-wide K group
in ascending order acc = mma.m16n8k16(A[row, 16g:16g+16], B'[16g:16g+16, col],
acc), then one fp32 multiply by the expert's global scale and one RN to bf16.
block_k = warp_k = 32 (no intra-CTA K split, no smem reducer) and stream-K off
(no inter-CTA K split) make that chain independent of block_m, block_n,
warp_n, the number of stages, CTAs per SM and the grid size. This module picks
those freely and keeps everything that is part of the chain.

It also stops vLLM from zero-padding the expert weights' N dimension to a
multiple of 256 (w13 1856 -> 2048, w2 2688 -> 2816); humming only needs
N % 64 == 0. Output columns are independent, and the padded columns were never
stored, so this changes no bit. K is padded only to humming's BF16 pack granule
(prepare_humming_weight asserts padded_k % (512 / a_bits) == 0, 32 here), so
w2's K stays 1856 instead of 1920 and its tiles no longer stream and multiply
two all-zero 32-wide K blocks. Those blocks were four mma(+0, +0, acc) steps at
the end of each chain: the identity for every non-NaN fp32 acc except -0, which
they turned into +0 (harness/tests/humming_moe_exact.py, phase mma, checks all
2^32 values). So the only possible difference is a w2 element that ends at -0
where stock ends at +0. Downstream ops only add, scale, square or take the
magnitude of it (bf16 rounding, the top-k combine, the residual add, RMSNorm,
FP8 amax scaling, the next matmuls); none divides by an activation or reads its
sign, so every later value has the same magnitude and compares equal, and the
logits and tokens are unchanged. w13's K (2688) needs no padding. The repack is
input-independent and value-exact.

The configs are keyed by the layer's shapes and only apply when the layer
matches them exactly (NVFP4 W4A16 group-16 experts with a per-expert global
scale, BF16 activations, indexed gemm, batch-invariant mode on). Anything else
keeps the stock padding and humming's heuristic.

VLLM_HUMMING_MOE_FAST=0 restores the stock padding and heuristic configs.
"""

import os
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Stock padding vLLM applies to humming MoE weights.
STOCK_PAD_N = 256
STOCK_PAD_K = 128
# Smallest paddings humming accepts with BF16 activations
# (prepare_humming_weight asserts padded_n % 64 == 0 and
# padded_k % (512 / a_bits) == 0): see the module docstring.
TUNED_PAD_N = 64
TUNED_PAD_K = 32

_BIG = 1 << 30


def _cfg(
    block_m: int,
    block_n: int,
    warp_m: int,
    warp_n: int,
    stages: int,
    ctas_per_sm: int,
    num_sms: int = 48,
) -> dict[str, Any]:
    """One batch-invariant humming tuning config (block_k = warp_k = 32)."""
    return {
        "block_shape": [block_m, block_n, 32],
        "warp_shape": [warp_m, warp_n, 32],
        "use_stream_k": False,
        "use_f16_accum": False,
        "num_sms": num_sms,
        "num_stages": stages,
        "num_ctas_per_sm": ctas_per_sm,
        "num_write_splits": 1,
    }


# LOCAL PATCH (cudafast-nemotron35, shco): with shx (harness/kernels/shco) available the
# prefill w13 runs a persistent grid of 4 * 36 CTAs, i.e. 3 resident CTAs per SM of the same
# <= 128-register build, so each SM keeps room for one shx CTA of the shared expert on the
# aux stream. Only the persistent grid changes; the accumulation chain is the same (see the
# module docstring). SHCO=0 or SHCO_HUMMING_SMS=48 restores 4 CTAs per SM.
def _prefill_w13_sms() -> int:
    if os.environ.get("SHCO", "1") == "0":
        return 48
    try:
        import shco

        ok = shco.available()
    except ImportError:
        ok = False
    return int(os.environ.get("SHCO_HUMMING_SMS", "36")) if ok else 48


_PREFILL_W13_SMS = _prefill_w13_sms()

# Tuned on GB10 (48 SMs, ~99 KB smem per block) with
# harness/tests/humming_moe_exact.py --phase sweep (run hum-sweep3: every
# candidate bitwise equal to stock; GEMM time per layer over 5 rotating
# layers). Keys: (sublayer, N, K, num_experts), K before padding. Values:
# ascending (max valid_shape_m, config); valid_shape_m is num_tokens * top_k.
# w13 and w2 must use the same block_m in every range, because both GEMMs
# share one moe_align_block_size layout. block_n must be a multiple of 64.
# Measured (stock -> tuned, us): M=1 w13 81.3->73.8, w2 86.9->74.7;
# M=4 w13 288.2->264.2, w2 294.6->270.1; M=8 w13 516.2->470.1, w2 551.6->476.9;
# M=496 w13 1896.5->1611.5, w2 1877.7->1674.
TUNED_CONFIGS: dict[tuple[str, int, int, int], list[tuple[int, dict[str, Any]]]] = {
    ("w13", 1856, 2688, 128): [
        (6, _cfg(16, 64, 16, 32, 6, 4)),  # M = 1
        (12, _cfg(16, 64, 16, 32, 6, 4)),  # M = 2
        (24, _cfg(16, 64, 16, 32, 8, 2)),  # M = 3, 4
        (48, _cfg(16, 64, 16, 32, 8, 2)),  # M = 5..8
        (384, _cfg(16, 64, 16, 32, 6, 4)),
        (_BIG, _cfg(64, 64, 32, 32, 4, 4, _PREFILL_W13_SMS)),  # prefill (496 tokens)
    ],
    ("w2", 2688, 1856, 128): [
        (6, _cfg(16, 128, 16, 32, 3, 4)),
        (12, _cfg(16, 128, 16, 32, 6, 2)),
        (24, _cfg(16, 128, 16, 32, 8, 1)),
        (48, _cfg(16, 128, 16, 32, 8, 1)),
        (384, _cfg(16, 128, 16, 32, 6, 2)),
        (_BIG, _cfg(64, 128, 32, 64, 4, 2)),
    ],
}

TUNED_LAYER_SHAPES = {"w13": (1856, 2688), "w2": (2688, 1856)}
TUNED_NUM_EXPERTS = 128


def humming_moe_fast_enabled() -> bool:
    """The kill switch (default on) plus humming's batch-invariant mode."""
    if os.environ.get("VLLM_HUMMING_MOE_FAST", "1") == "0":
        return False
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
        _humming_batch_invariant,
    )

    return _humming_batch_invariant()


def select_moe_padding(
    weight_schema: Any,
    input_schema: Any,
    force_weight_schema: Any,
    sublayer_configs: dict[str, dict[str, int]],
    num_experts: int,
    has_bias: bool,
    param_dtype: torch.dtype,
) -> tuple[int, int, bool]:
    """Return (pad_n_to_multiple, pad_k_to_multiple, tuned) for one MoE layer.

    The decision covers both sublayers at once: w13 and w2 must either both
    use the tuned configs or both use humming's heuristic, so that their
    block_m (and with it the shared moe_align layout) always agree.
    """
    stock = (STOCK_PAD_N, STOCK_PAD_K, False)
    if not humming_moe_fast_enabled():
        return stock
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
        get_humming_moe_gemm_type,
    )

    if get_humming_moe_gemm_type() != "indexed":
        return stock
    if force_weight_schema is not None or has_bias:
        return stock
    if param_dtype != torch.bfloat16 or num_experts != TUNED_NUM_EXPERTS:
        return stock
    if type(weight_schema).__name__ != "ModeloptNvfp4WeightSchema":
        return stock
    if getattr(input_schema, "a_dtype", None) is not None:
        return stock
    shapes = {
        name: (cfg["shape_n"], cfg["shape_k"])
        for name, cfg in sublayer_configs.items()
    }
    if shapes != TUNED_LAYER_SHAPES:
        return stock
    return TUNED_PAD_N, TUNED_PAD_K, True


def _meta_matches(meta: Any, sublayer_name: str) -> bool:
    from vllm.utils.humming import WeightScaleType, dtypes

    shape_n, shape_k = TUNED_LAYER_SHAPES[sublayer_name]
    padded_k = -(-shape_k // TUNED_PAD_K) * TUNED_PAD_K
    return (
        meta.shape_n == shape_n
        and meta.shape_k == padded_k
        and meta.pad_shape_n == 0
        and meta.pad_shape_k == padded_k - shape_k
        and meta.num_experts == TUNED_NUM_EXPERTS
        and meta.a_dtype == dtypes.bfloat16
        and meta.c_dtype == dtypes.bfloat16
        and meta.b_dtype == dtypes.float4e2m1
        and meta.bs_dtype == dtypes.float8e4m3
        and meta.weight_scale_group_size == 16
        and meta.weight_scale_type == WeightScaleType.GROUP_TENSOR
        and not meta.has_zero_point
        and not meta.has_bias
    )


def _to_humming_list(
    table: list[tuple[int, dict[str, Any]]],
) -> list[list[Any]]:
    out: list[list[Any]] = []
    lo = 0
    for hi, config in table:
        out.append([lo, hi, dict(config)])
        lo = hi
    out[-1][1] = _BIG
    return out


def tuned_moe_configs(layer: torch.nn.Module) -> tuple[list, list] | None:
    """The explicit (w13, w2) tuning-config lists for a tuned layer, else None."""
    if not getattr(layer, "humming_tuned_moe", False):
        return None
    if not humming_moe_fast_enabled():
        return None
    metas = layer.humming_metas
    if not (_meta_matches(metas["w13"], "w13") and _meta_matches(metas["w2"], "w2")):
        logger.warning_once(
            "humming tuned MoE: layer metas do not match the tuned shapes; "
            "using humming's heuristic configs"
        )
        return None
    w13 = TUNED_CONFIGS[("w13", *TUNED_LAYER_SHAPES["w13"], TUNED_NUM_EXPERTS)]
    w2 = TUNED_CONFIGS[("w2", *TUNED_LAYER_SHAPES["w2"], TUNED_NUM_EXPERTS)]
    w13_list = _to_humming_list(w13)
    w2_list = _to_humming_list(w2)
    for a, b in zip(w13_list, w2_list, strict=True):
        assert a[0] == b[0] and a[1] == b[1], "w13/w2 ranges differ"
        assert a[2]["block_shape"][0] == b[2]["block_shape"][0], "block_m differs"
    for entry in w13_list + w2_list:
        config = entry[2]
        assert config["block_shape"][2] == config["warp_shape"][2] == 32
        # humming packs weights in 64-column groups; block_n = 32 is wrong
        assert config["block_shape"][1] % 64 == 0
        assert config["warp_shape"][0] % 16 == 0
        assert config["use_stream_k"] is False
    logger.info_once("humming tuned MoE: unpadded experts, explicit BI configs")
    return w13_list, w2_list


# relu^2 in one kernel instead of F.relu(x, inplace=True) + torch.square(x,
# out=y). Same arithmetic: torch's square of a bf16 computes x * x in fp32 and
# rounds once to bf16 with round-to-nearest-even (neither kernel flushes
# subnormals); on the GPU every NaN comes out as 0x7FFF (measured over all
# bf16 inputs), so this kernel writes 0x7FFF too. relu keeps NaN (clamp_min
# propagates it), and -0 squares to +0 either way.
# harness/tests/humming_moe_exact.py checks all 65536 bf16 inputs against the
# torch pair.
try:
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _relu2_bf16_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.where(x < 0.0, 0.0, x)
        y = x * x
        bits = y.to(tl.bfloat16).to(tl.int16, bitcast=True)
        nan_bits = tl.full(bits.shape, 0x7FFF, tl.int16)
        bits = tl.where(y != y, nan_bits, bits)
        tl.store(y_ptr + offs, bits, mask=mask)

    _HAVE_TRITON = True
except Exception:  # noqa: BLE001
    _HAVE_TRITON = False


def relu2_bf16(input: torch.Tensor, output: torch.Tensor) -> bool:
    """output = bf16(relu(input)^2), bit-identical to the torch pair. Returns
    False (and does nothing) when the fused kernel does not apply."""
    if not _HAVE_TRITON:
        return False
    if input.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
        return False
    if not (input.is_contiguous() and output.is_contiguous()):
        return False
    if input.shape != output.shape:
        return False
    n = input.numel()
    if n == 0:
        return True
    block = 1024
    _relu2_bf16_kernel[(triton.cdiv(n, block),)](
        input, output.view(torch.int16), n, BLOCK=block, num_warps=4
    )
    return True

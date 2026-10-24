# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LOCAL PATCH (cudafast-nemotron35, vllm/VENDOR.json "local_patches"): zskip routed-expert
down projection for small calls (the MTP verify).

harness/kernels/zskip computes humming's batch-invariant indexed w2 output bit for bit (same
dequantised bf16 weights, the same mma.sync.m16n8k16 per k16 group in ascending order from +0,
the same epilogue) from a load-time k-row-major repack of the checkpoint's w2 tensors, and does
not load the weight rows whose relu^2 activations are zero in every row routed to the expert:
those rows only ever multiply exact zeros. See harness/kernels/zskip/README.md for the
arithmetic and the evidence.

It serves calls of 1 to 8 tokens on the layers humming_tuned_moe.py tunes (unpadded experts,
block_m 16, BF16 activations, no input scale); with ZSKIP=0 routed_sparse serves them, and
humming the prefill. The repack keeps every byte of the checkpoint tensors next to humming's own
copy (about 357 MB per layer). ZSKIP=0 in the environment disables it.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    import zskip as _zskip
except ImportError:
    _zskip = None

ENABLED = os.environ.get("ZSKIP", "1") != "0"
# kernel v4 (warp per 64-column tile, weights as the MMA A operand) on the "swap" repack
LAYOUT = "swap"
CONFIG = 100
FUSED_PREP = True
FUSED_CONFIG = 200
MAX_TOKENS = 8
TOP_K = 6
HIDDEN = 2688
INTER = 1856
NUM_EXPERTS = 128


def prepare_raw(layer: torch.nn.Module, tensors: dict[str, torch.Tensor]) -> None:
    """Before humming converts w2: repack the checkpoint tensors (input-independent)."""
    layer.zskip_w2 = None
    if not ENABLED or _zskip is None or not _zskip.available():
        return
    try:
        w = tensors["weight"]
        s = tensors["weight_scale"]
        gs = tensors["weight_scale_2"]
        if (
            w.dtype != torch.uint8
            or tuple(w.shape) != (NUM_EXPERTS, HIDDEN, INTER // 2)
            or s.dtype != torch.float8_e4m3fn
            or tuple(s.shape) != (NUM_EXPERTS, HIDDEN, INTER // 16)
            or gs.numel() != NUM_EXPERTS
            or not w.is_cuda
        ):
            return
        layer.zskip_w2 = _zskip.repack(
            w.contiguous(), s.contiguous(), gs.reshape(-1).to(torch.float32), layout=LAYOUT
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("zskip: repack failed (%r); humming keeps the down projection", exc)
        layer.zskip_w2 = None


def finalize(layer: torch.nn.Module) -> None:
    """After humming's conversion: run with humming's own global scales, load the kernels and
    allocate their scratch outside CUDA graph capture."""
    packed = getattr(layer, "zskip_w2", None)
    if packed is None:
        return
    gs = getattr(layer, "w2_global_scale", None)
    if gs is None or gs.numel() != NUM_EXPERTS:
        layer.zskip_w2 = None
        return
    gs = gs.detach().reshape(-1).to(torch.float32).contiguous()
    if not torch.equal(gs, packed.gs):
        logger.warning("zskip: humming's w2 global scales differ from the checkpoint's; disabled")
        layer.zskip_w2 = None
        return
    packed.gs = gs
    if not _zskip.init():
        layer.zskip_w2 = None
        return
    _zskip.scratch(gs.device)
    if FUSED_PREP and _fused_check(layer):
        layer.zskip_config = FUSED_CONFIG
    elif not _self_check(layer):
        layer.zskip_w2 = None
        return
    logger.info_once("zskip: zero-row-skipping routed down projection enabled for <= 8 tokens")


def _fused_check(layer: torch.nn.Module) -> bool:
    """The load-time check with FUSED_CONFIG; False (crown CONFIG next) on any failure."""
    try:
        _zskip.config_info(FUSED_CONFIG)
        return _self_check(layer, FUSED_CONFIG, quiet=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning_once("zskip: fused prep unavailable (%r); config %d", exc, CONFIG)
        return False


def _self_check(layer: torch.nn.Module, config: int = CONFIG, quiet: bool = False) -> bool:
    """Load-time guard: humming's w2 and zskip on deterministic 4- and 8-token calls, bitwise."""
    import json

    from vllm import envs
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
        _humming_batch_invariant,
    )
    from vllm.model_executor.layers.fused_moe.experts.humming_tuned_moe import (
        tuned_moe_configs,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.utils.humming import GemmType, HummingMethod

    configs = tuned_moe_configs(layer)
    if configs is None:
        return False
    dev = layer.w2_global_scale.device
    compute_config = json.dumps(
        {
            "use_batch_invariant": _humming_batch_invariant(),
            "use_f16_accum": envs.VLLM_HUMMING_USE_F16_ACCUM,
            "gemm_type": GemmType.INDEXED.value,
        }
    )
    ok = True
    for m in (4, MAX_TOKENS):
        ids = [[(17 * t + 29 * i) % NUM_EXPERTS for i in range(TOP_K)] for t in range(m)]
        ids[1][0] = ids[0][0]  # one block with two rows
        topk_ids = torch.tensor(ids, dtype=torch.int32, device=dev)
        gen = torch.Generator(device=dev)
        gen.manual_seed(1234 + m)
        u = torch.randn(m * TOP_K, INTER, generator=gen, device=dev)
        keep = torch.rand(m * TOP_K, INTER, generator=gen, device=dev) < 0.1
        act = (u.relu().square() * keep).to(torch.bfloat16).contiguous()
        sorted_ids, expert_ids, ntp = moe_align_block_size(
            topk_ids=topk_ids,
            block_size=16,
            num_experts=NUM_EXPERTS,
            expert_map=None,
            ignore_invalid_experts=True,
        )
        ref = torch.empty(m * TOP_K, HIDDEN, dtype=torch.bfloat16, device=dev)
        HummingMethod.forward_layer(
            layer=layer,
            inputs=act,
            input_scale=None,
            outputs=ref,
            sublayer_name="w2",
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_padded=ntp,
            compute_config=compute_config,
            valid_shape_m=m * TOP_K,
            top_k=1,
            tuning_config=json.dumps(configs[1]),
        )
        out = torch.empty_like(ref)
        _zskip.down(
            act,
            sorted_ids,
            expert_ids,
            ntp,
            layer.zskip_w2,
            out,
            mode=_zskip.SKIP,
            config=config,
            max_blocks=m * TOP_K,
        )
        ok = ok and torch.equal(out.view(torch.int16), ref.view(torch.int16))
    if not ok and quiet:
        logger.warning_once("zskip: fused prep differs from humming at load; config %d", CONFIG)
    elif not ok:
        logger.warning(
            "zskip: load-time check differs from humming on %s; humming keeps the down projection",
            getattr(layer, "layer_name", "?"),
        )
    return ok


def maybe_down(
    layer: torch.nn.Module,
    inputs: torch.Tensor,
    input_scale: torch.Tensor | None,
    outputs: torch.Tensor,
    num_tokens: int,
    top_k: int,
    block_m: int | None,
    moe_kwargs: dict[str, Any],
) -> bool:
    """Runs the w2 GEMM with zskip and returns True, or returns False (caller runs humming)."""
    packed = getattr(layer, "zskip_w2", None)
    if packed is None or not ENABLED:
        return False
    if not (1 <= num_tokens <= MAX_TOKENS) or top_k != TOP_K or block_m != 16:
        return False
    if input_scale is not None or inputs.dtype != torch.bfloat16 or outputs.dtype != torch.bfloat16:
        return False
    if torch.compiler.is_compiling():
        return False
    rows = num_tokens * top_k
    if (
        inputs.dim() != 2
        or tuple(inputs.shape) != (rows, INTER)
        or inputs.stride(1) != 1
        or inputs.stride(0) % 8
        or inputs.data_ptr() % 16
        or outputs.dim() != 2
        or tuple(outputs.shape) != (rows, HIDDEN)
        or outputs.stride(1) != 1
        or outputs.stride(0) % 2
    ):
        return False
    _zskip.down(
        inputs,
        moe_kwargs["sorted_ids"],
        moe_kwargs["expert_ids"],
        moe_kwargs["num_tokens_padded"],
        packed,
        outputs,
        mode=_zskip.SKIP,
        config=getattr(layer, "zskip_config", CONFIG),
        max_blocks=rows,
    )
    return True

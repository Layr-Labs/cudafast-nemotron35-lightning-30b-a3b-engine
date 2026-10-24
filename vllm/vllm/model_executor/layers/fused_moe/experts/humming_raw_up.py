# SPDX-License-Identifier: Apache-2.0
"""Experimental full-K small-M routed up, behind the original fused fallback.

Every layer independently compares the new output with the original fused
Humming up during official remote startup. No local candidate execution.
"""
from __future__ import annotations

import json
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
try:
    import routed_up as _raw
except ImportError:
    _raw = None

ENABLED = os.environ.get("ROUTED_RAW_UP", "1") != "0"


def prepare_raw(layer, tensors):
    layer.raw_routed_up = None
    if not ENABLED or _raw is None or not _raw.available():
        return
    w, s, gs = tensors["weight"], tensors["weight_scale"], tensors["weight_scale_2"]
    if (w.dtype != torch.uint8 or tuple(w.shape) != (128, 1856, 1344)
            or s.dtype != torch.float8_e4m3fn or tuple(s.shape) != (128, 1856, 168)
            or gs.numel() != 128 or not w.is_cuda):
        return
    try:
        layer.raw_routed_up = _raw.repack(w.contiguous(), s.contiguous(), gs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("routed_up: packing failed (%r); original Humming remains", exc)


def finalize(layer):
    packed = getattr(layer, "raw_routed_up", None)
    if packed is None:
        return
    gs = getattr(layer, "w13_global_scale", None)
    if gs is None or gs.numel() != 128:
        layer.raw_routed_up = None
        return
    gs = gs.detach().reshape(-1).to(torch.float32).contiguous()
    if not torch.equal(gs, packed.gs) or not _raw.init():
        layer.raw_routed_up = None
        return
    packed.gs = gs
    if not _self_check(layer):
        logger.warning("routed_up: original fused Humming comparison differs; disabled")
        layer.raw_routed_up = None
        return
    logger.info_once("routed_up: full-K native routed up/ReLU2 enabled for BF16 M1..8")


def _self_check(layer):
    from vllm import envs
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import _humming_batch_invariant
    from vllm.model_executor.layers.fused_moe.experts.humming_relu2_epilogue import forward_relu2
    from vllm.model_executor.layers.fused_moe.experts.humming_tuned_moe import tuned_moe_configs
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    from vllm.utils.humming import GemmType

    configs = tuned_moe_configs(layer)
    if configs is None or not _humming_batch_invariant() or envs.VLLM_HUMMING_USE_F16_ACCUM:
        return False
    cc = json.dumps({"use_batch_invariant": True, "use_f16_accum": False,
                     "gemm_type": GemmType.INDEXED.value})
    dev = layer.w13_global_scale.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(51947)
    for m in (1, 4, 5, 8):
        ids = [[(17 * t + 29 * s) % 128 for s in range(6)] for t in range(m)]
        if m > 1:
            ids[1][0] = ids[0][0]
        ids = torch.tensor(ids, dtype=torch.int32, device=dev)
        # Current input values are generated only for this startup comparison;
        # no reference prompt or expected token enters the runtime dispatch.
        x = torch.randn((m, 2688), dtype=torch.bfloat16, device=dev, generator=gen)
        sid, eid, ntp = moe_align_block_size(topk_ids=ids, block_size=16,
            num_experts=128, expert_map=None, ignore_invalid_experts=True)
        ref = torch.empty((m * 6, 1856), dtype=torch.bfloat16, device=dev)
        forward_relu2(layer=layer, inputs=x, outputs=ref, compute_config=cc,
            tuning_config=json.dumps(configs[0]), sorted_ids=sid, expert_ids=eid,
            num_tokens_padded=ntp, top_k=6, valid_shape_m=m * 6, input_scale=None)
        out = torch.empty_like(ref)
        _raw.up(x, sid, eid, ntp, layer.raw_routed_up, out)
        if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
            return False
    return True


def maybe_up(layer, inputs, input_scale, outputs, num_tokens, top_k, block_m, kwargs):
    packed = getattr(layer, "raw_routed_up", None)
    if not ENABLED or packed is None or not (1 <= num_tokens <= 8) or top_k != 6 or block_m != 16:
        return False
    if torch.compiler.is_compiling() or input_scale is not None:
        return False
    if (inputs.dtype != torch.bfloat16 or outputs.dtype != torch.bfloat16
            or not inputs.is_cuda or inputs.device != outputs.device
            or inputs.dim() != 2 or tuple(inputs.shape) != (num_tokens, 2688)
            or inputs.stride(1) != 1 or inputs.stride(0) % 8 or inputs.data_ptr() % 16
            or outputs.dim() != 2 or tuple(outputs.shape) != (num_tokens * 6, 1856)
            or outputs.stride(1) != 1 or outputs.stride(0) % 2):
        return False
    for key in ("sorted_ids", "expert_ids", "num_tokens_padded"):
        v = kwargs[key]
        if v.dtype != torch.int32 or not v.is_contiguous() or v.device != inputs.device:
            return False
    _raw.up(inputs, kwargs["sorted_ids"], kwargs["expert_ids"], kwargs["num_tokens_padded"], packed, outputs)
    return True

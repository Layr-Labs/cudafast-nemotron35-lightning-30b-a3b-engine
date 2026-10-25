"""LOCAL PATCH (cudafast-nemotron35): humming-exact routed up for the verify.

The tuned indexed w13 at M=4 is 264 us/layer on GB10 (humming_tuned_moe.py) for the
same ~46 MB the routed down moves in 193 us. upskip runs that GEMM with zskip's
swap-layout load and mma.m16n8k16 chain, then the fused relu^2 epilogue. A load-time
bitwise check against forward_relu2 disables the layer on any mismatch, and a short
timed replay disables it when it is not faster than humming. UPSKIP=0 keeps humming.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    import upskip as _upskip
except ImportError:
    _upskip = None

ENABLED = os.environ.get("UPSKIP", "1") != "0"
MAX_TOKENS = 8
TOP_K = 6
HIDDEN = 2688
INTER = 1856
NUM_EXPERTS = 128


def prepare_raw(layer: torch.nn.Module, tensors: dict[str, torch.Tensor]) -> None:
    """Before humming converts w13: swap-repack the checkpoint tensors."""
    layer.upskip_w13 = None
    layer.upskip_ok = False
    if not ENABLED or _upskip is None or not _upskip.available():
        return
    try:
        w = tensors["weight"]
        s = tensors["weight_scale"]
        gs = tensors["weight_scale_2"]
        if (
            w.dtype != torch.uint8
            or tuple(w.shape) != (NUM_EXPERTS, INTER, HIDDEN // 2)
            or s.dtype != torch.float8_e4m3fn
            or tuple(s.shape) != (NUM_EXPERTS, INTER, HIDDEN // 16)
            or gs.numel() != NUM_EXPERTS
            or not w.is_cuda
        ):
            return
        layer.upskip_w13 = _upskip.repack(
            w.contiguous(), s.contiguous(), gs.reshape(-1).to(torch.float32)
        )
    except Exception as exc:
        logger.warning("upskip: repack failed (%r); humming keeps the up projection", exc)
        layer.upskip_w13 = None


def finalize(layer: torch.nn.Module) -> None:
    """After humming's conversion: keep humming's scales, then exactness and a speed gate."""
    packed = getattr(layer, "upskip_w13", None)
    if packed is None:
        return
    gs = getattr(layer, "w13_global_scale", None)
    if gs is None:
        gs = getattr(layer, "w13_weight_scale_2", None)
    if gs is None or gs.numel() != NUM_EXPERTS:
        layer.upskip_w13 = None
        return
    gs = gs.detach().reshape(-1).to(torch.float32).contiguous()
    if not torch.equal(gs, packed.gs):
        logger.warning("upskip: humming's w13 global scales differ; disabled")
        layer.upskip_w13 = None
        return
    packed.gs = gs
    try:
        if not _exact(layer) or not _faster(layer):
            layer.upskip_w13 = None
            return
    except Exception as exc:
        logger.warning("upskip: load-time check failed (%r); humming keeps w13", exc)
        layer.upskip_w13 = None
        return
    layer.upskip_ok = True
    logger.info_once("upskip: humming-exact routed up enabled for <= 8 tokens")


def _configs(layer: torch.nn.Module):
    import json

    from vllm import envs
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
        _humming_batch_invariant,
    )
    from vllm.model_executor.layers.fused_moe.experts.humming_tuned_moe import (
        tuned_moe_configs,
    )
    from vllm.utils.humming import GemmType

    configs = tuned_moe_configs(layer)
    if configs is None:
        return None
    compute = json.dumps(
        {
            "use_batch_invariant": _humming_batch_invariant(),
            "use_f16_accum": envs.VLLM_HUMMING_USE_F16_ACCUM,
            "gemm_type": GemmType.INDEXED.value,
        }
    )
    return compute, json.dumps(configs[0])


def _run_pair(layer, packed, m: int, hidden: torch.Tensor):
    """(humming relu2 output, upskip output) for one deterministic routing."""
    from vllm.model_executor.layers.fused_moe.experts.humming_relu2_epilogue import (
        forward_relu2,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    cfg = _configs(layer)
    if cfg is None:
        return None
    compute, tuning = cfg
    dev = hidden.device
    ids = [[(17 * t + 29 * i) % NUM_EXPERTS for i in range(TOP_K)] for t in range(m)]
    ids[min(1, m - 1)][0] = ids[0][0]
    topk_ids = torch.tensor(ids, dtype=torch.int32, device=dev)
    sorted_ids, expert_ids, ntp = moe_align_block_size(
        topk_ids=topk_ids,
        block_size=16,
        num_experts=NUM_EXPERTS,
        expert_map=None,
        ignore_invalid_experts=True,
    )
    ref = torch.zeros(m * TOP_K, INTER, dtype=torch.bfloat16, device=dev)
    out = torch.zeros_like(ref)
    forward_relu2(
        layer=layer,
        inputs=hidden,
        outputs=ref,
        compute_config=compute,
        tuning_config=tuning,
        sorted_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_padded=ntp,
        top_k=TOP_K,
        valid_shape_m=m * TOP_K,
    )
    _upskip.up(hidden, sorted_ids, expert_ids, ntp, packed, out)
    return ref, out


def _exact(layer: torch.nn.Module) -> bool:
    dev = layer.w13_weight.device
    packed = layer.upskip_w13
    ok = True
    for m, seed in ((1, 3), (4, 5), (5, 9), (8, 11)):
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)
        hidden = torch.randn(m, HIDDEN, generator=gen, device=dev).to(torch.bfloat16)
        hidden = hidden.clone()
        hidden.view(-1)[:8] = 0
        pair = _run_pair(layer, packed, m, hidden)
        if pair is None:
            return False
        ref, out = pair
        ok = ok and torch.equal(out.view(torch.int16), ref.view(torch.int16))
    if not ok:
        logger.warning(
            "upskip: load-time check differs from humming on %s; humming keeps w13",
            getattr(layer, "layer_name", "?"),
        )
    return ok


def _faster(layer: torch.nn.Module) -> bool:
    """One captured replay. Disable the layer unless upskip is clearly under humming."""
    dev = layer.w13_weight.device
    packed = layer.upskip_w13
    gen = torch.Generator(device=dev)
    gen.manual_seed(17)
    hidden = torch.randn(4, HIDDEN, generator=gen, device=dev).to(torch.bfloat16)
    pair = _run_pair(layer, packed, 4, hidden)
    if pair is None:
        return False
    ref, out = pair
    if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
        return False
    for _ in range(2):
        _run_pair(layer, packed, 4, hidden)
    torch.cuda.synchronize(dev)

    from vllm.model_executor.layers.fused_moe.experts.humming_relu2_epilogue import (
        forward_relu2,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    cfg = _configs(layer)
    if cfg is None:
        return False
    compute, tuning = cfg
    m = 4
    ids = [[(17 * t + 29 * i) % NUM_EXPERTS for i in range(TOP_K)] for t in range(m)]
    topk_ids = torch.tensor(ids, dtype=torch.int32, device=dev)
    sorted_ids, expert_ids, ntp = moe_align_block_size(
        topk_ids=topk_ids,
        block_size=16,
        num_experts=NUM_EXPERTS,
        expert_map=None,
        ignore_invalid_experts=True,
    )
    buf = torch.empty(m * TOP_K, INTER, dtype=torch.bfloat16, device=dev)

    def _time(fn) -> float:
        for _ in range(2):
            fn()
        torch.cuda.synchronize(dev)
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(6):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    hum = _time(
        lambda: forward_relu2(
            layer=layer,
            inputs=hidden,
            outputs=buf,
            compute_config=compute,
            tuning_config=tuning,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_padded=ntp,
            top_k=TOP_K,
            valid_shape_m=m * TOP_K,
        )
    )
    ups = _time(lambda: _upskip.up(hidden, sorted_ids, expert_ids, ntp, packed, buf))
    if not (ups < hum * 0.92):
        logger.warning(
            "upskip: %.3f ms vs humming %.3f ms on %s; humming keeps w13",
            ups,
            hum,
            getattr(layer, "layer_name", "?"),
        )
        return False
    return True


def maybe_up(
    layer: torch.nn.Module,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    num_tokens: int,
    top_k: int,
    moe_kwargs: dict[str, Any],
) -> bool:
    """Write relu^2(w13) into outputs. False leaves the humming path to run."""
    packed = getattr(layer, "upskip_w13", None)
    if packed is None or not getattr(layer, "upskip_ok", False):
        return False
    if torch.compiler.is_compiling():
        return False
    if not (1 <= num_tokens <= MAX_TOKENS) or top_k != TOP_K:
        return False
    if inputs.dtype != torch.bfloat16 or outputs.dtype != torch.bfloat16:
        return False
    rows = num_tokens * top_k
    if (
        inputs.dim() != 2
        or tuple(inputs.shape) != (num_tokens, HIDDEN)
        or inputs.stride(1) != 1
        or outputs.dim() != 2
        or tuple(outputs.shape) != (rows, INTER)
        or outputs.stride(1) != 1
    ):
        return False
    sorted_ids = moe_kwargs.get("sorted_ids")
    expert_ids = moe_kwargs.get("expert_ids")
    ntp = moe_kwargs.get("num_tokens_padded")
    if sorted_ids is None or expert_ids is None or ntp is None:
        return False
    if sorted_ids.dtype != torch.int32 or expert_ids.dtype != torch.int32:
        return False
    _upskip.up(inputs, sorted_ids, expert_ids, ntp, packed, outputs)
    return True

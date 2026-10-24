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

PZ (below PREFILL): calls of 9 to 512 tokens, the prefill, run the same kernel (CONFIG) on an
8-row layout of their own when the per-layer load-time check passes; ZSKIP_PREFILL=0 keeps
humming for them.
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
# CONFIG with prep_kernel fused into down4_kernel (csrc/zskip.cu ZSKIP4F_CONFIGS): one launch, the
# same block lists built by the same build_lists in each CTA's shared memory. Used per layer only
# when it passes the load-time check; otherwise CONFIG is checked and used. ZSKIP_FUSED_PREP=0
# keeps CONFIG.
FUSED_PREP = os.environ.get("ZSKIP_FUSED_PREP", "1") != "0"
FUSED_CONFIG = 200
MAX_TOKENS = 8
TOP_K = 6
HIDDEN = 2688
INTER = 1856
NUM_EXPERTS = 128
# PZ: calls of 9 to 512 tokens (the 496-token prefill) through the same kernel, config CONFIG.
# The kernel keeps at most RMAX = 8 valid rows per 16-slot block (csrc/zskip.cu prep_kernel), so
# the call builds its own 8-row moe_align_block_size layout and widens every 8-row block to a
# 16-slot block whose last 8 slots are the padding id (widen_blocks). Used per layer only when
# the load-time check (_prefill_check: M = 496 x 4 inputs and M = 9, bitwise against humming's
# prefill config) passes; any failure turns it off for the rest of the load. ZSKIP_PREFILL=0
# disables it (no check, humming runs the prefill w2 as before).
PREFILL = os.environ.get("ZSKIP_PREFILL", "1") != "0"
PREFILL_MIN_TOKENS = 9
PREFILL_MAX_TOKENS = 512
PREFILL_BLOCK = 8
_prefill_broken = False


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
    layer.zskip_prefill_ok = PREFILL and _prefill_check(layer)
    logger.info_once("zskip: zero-row-skipping routed down projection enabled for <= 8 tokens")


def _fused_check(layer: torch.nn.Module) -> bool:
    """The load-time check with FUSED_CONFIG; False (crown CONFIG next) on any failure."""
    try:
        _zskip.config_info(FUSED_CONFIG)  # -2 (ValueError) when built with -DZSKIP_NO_FUSED
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


def widen_blocks(sorted_ids: torch.Tensor, rows: int) -> tuple[torch.Tensor, int]:
    """8-row moe_align layout -> the kernel's 16-slot blocks (pure torch, no host read of device
    values): 16-slot block b holds slots 0..7 of 8-row block b and `rows` (the padding id: the
    kernel keeps an id only when it is < shape_m = rows) in slots 8..15. Every 8-row block that
    num_tokens_padded covers is a whole block of sorted_ids (num_tokens_padded is a multiple of 8
    and <= sorted_ids.numel()), so the tail shorter than 8 is dropped. Returns (ids, blocks)."""
    nblocks = sorted_ids.numel() // PREFILL_BLOCK
    s8 = sorted_ids[: nblocks * PREFILL_BLOCK].view(nblocks, PREFILL_BLOCK)
    pad = torch.full_like(s8, rows)
    return torch.cat([s8, pad], 1).reshape(-1), nblocks


def _prefill_run(
    inputs: torch.Tensor,
    topk_ids: torch.Tensor,
    packed: Any,
    outputs: torch.Tensor,
    rows: int,
) -> bool:
    """zskip down (CONFIG) over this call's own 8-row layout widened to 16-slot blocks. Every
    shape depends only on the number of tokens; no synchronisation and no host read."""
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    sorted_ids, expert_ids, ntp = moe_align_block_size(
        topk_ids=topk_ids,
        block_size=PREFILL_BLOCK,
        num_experts=NUM_EXPERTS,
        expert_map=None,
        ignore_invalid_experts=True,
    )
    ids16, nblocks = widen_blocks(sorted_ids, rows)
    if nblocks > getattr(_zskip, "MAX_BLOCKS", 0):
        return False
    _zskip.down(
        inputs,
        ids16,
        expert_ids,
        ntp * 2,
        packed,
        outputs,
        mode=_zskip.SKIP,
        config=CONFIG,
        max_blocks=nblocks,
    )
    return True


def _prefill_ids(m: int, same: bool, seed: int, dev: torch.device) -> torch.Tensor:
    """Deterministic top-6 ids [m, 6] (int32): distinct random experts per token, or every token
    on the same 6 experts (the shape the last layer's prefill takes, nemotron_h.py)."""
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    if same:
        ids = torch.randperm(NUM_EXPERTS, generator=gen, device=dev)[:TOP_K].expand(m, TOP_K)
    else:
        ids = torch.rand(m, NUM_EXPERTS, generator=gen, device=dev).argsort(dim=1)[:, :TOP_K]
    return ids.to(torch.int32).contiguous()


def _prefill_act(rows: int, kind: str, seed: int, dev: torch.device) -> torch.Tensor:
    """Deterministic bf16 [rows, INTER]: "sparse" relu^2 with ~90% zeros, "dense" no zeros,
    "edge" the sparse values with -0, subnormals, large and negative values mixed in."""
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    u = torch.randn(rows, INTER, generator=gen, device=dev)
    if kind == "dense":
        x = (u.abs() + 0.05).square()
    else:
        keep = torch.rand(rows, INTER, generator=gen, device=dev) < 0.1
        x = u.relu().square() * keep
    x = x.to(torch.bfloat16).contiguous()
    if kind != "edge":
        return x
    bits = x.view(torch.int16).clone()
    r = torch.rand(rows, INTER, generator=gen, device=dev)
    neg = torch.tensor(-32768, dtype=torch.int16, device=dev)
    sub = torch.randint(1, 0x80, (rows, INTER), generator=gen, device=dev, dtype=torch.int16)
    big = torch.randint(0x4700, 0x4780, (rows, INTER), generator=gen, device=dev, dtype=torch.int16)
    bits = torch.where(r < 0.05, neg, bits)
    bits = torch.where((r >= 0.05) & (r < 0.08), sub, bits)
    bits = torch.where((r >= 0.08) & (r < 0.10), big, bits)
    bits = torch.where((r >= 0.10) & (r < 0.15), bits | neg, bits)
    return bits.view(torch.bfloat16).contiguous()


def _prefill_compare(layer: torch.nn.Module) -> bool:
    """humming's w2 with the prefill layout and config against _prefill_run, bitwise: M = 496 x
    (random routing, sparse), (one expert set, sparse), (random, dense), (random, edge), and
    M = 9 (random, sparse)."""
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

    dev = layer.w2_global_scale.device
    need = (PREFILL_MAX_TOKENS * TOP_K + NUM_EXPERTS * (PREFILL_BLOCK - 1)) // PREFILL_BLOCK
    if getattr(_zskip, "MAX_BLOCKS", 0) < need:
        return False
    ws, _ = _zskip.scratch(dev)
    if ws.numel() < need * _zskip.BI_BYTES:
        return False
    configs = tuned_moe_configs(layer)
    if configs is None:
        return False
    compute_config = json.dumps(
        {
            "use_batch_invariant": _humming_batch_invariant(),
            "use_f16_accum": envs.VLLM_HUMMING_USE_F16_ACCUM,
            "gemm_type": GemmType.INDEXED.value,
        }
    )
    tuning_config = json.dumps(configs[1])
    cases = (
        (496, False, "sparse"),
        (496, True, "sparse"),
        (496, False, "dense"),
        (496, False, "edge"),
        (PREFILL_MIN_TOKENS, False, "sparse"),
    )
    for i, (m, same, kind) in enumerate(cases):
        rows = m * TOP_K
        # the block size the real call takes (HummingIndexedExperts.prepare_humming_moe_kwargs)
        block_m = None
        for lo, hi, cfg in configs[0]:
            if lo < rows <= hi:
                block_m = cfg["block_shape"][0]
                break
        if block_m is None:
            return False
        topk_ids = _prefill_ids(m, same, 4242 + i, dev)
        act = _prefill_act(rows, kind, 777 + i, dev)
        sorted_ids, expert_ids, ntp = moe_align_block_size(
            topk_ids=topk_ids,
            block_size=block_m,
            num_experts=NUM_EXPERTS,
            expert_map=None,
            ignore_invalid_experts=True,
        )
        ref = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
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
            valid_shape_m=rows,
            top_k=1,
            tuning_config=tuning_config,
        )
        # a NaN pattern, so a row the kernel does not write fails the comparison
        out = torch.full((rows, HIDDEN), 0x7FC1, dtype=torch.int16, device=dev).view(torch.bfloat16)
        if not _prefill_run(act, topk_ids, layer.zskip_w2, out, rows):
            return False
        same_bits = torch.equal(out.view(torch.int16), ref.view(torch.int16))
        del topk_ids, act, sorted_ids, expert_ids, ntp, ref, out
        if not same_bits:
            return False
    return True


def _prefill_check(layer: torch.nn.Module) -> bool:
    """Load-time guard for PZ (outside graph capture). False on any difference or exception, and
    then False for every later layer too; the decode zskip path is unaffected."""
    global _prefill_broken
    if _prefill_broken or torch.cuda.is_current_stream_capturing():
        return False
    try:
        ok = _prefill_compare(layer)
    except Exception as exc:  # noqa: BLE001 - any failure keeps humming for the prefill
        logger.warning("zskip: prefill check raised %r; humming keeps the prefill w2", exc)
        ok = False
    if not ok:
        _prefill_broken = True
        logger.warning(
            "zskip: prefill check failed on %s; humming keeps the prefill w2",
            getattr(layer, "layer_name", "?"),
        )
    return ok


def _prefill_down(
    inputs: torch.Tensor,
    input_scale: torch.Tensor | None,
    outputs: torch.Tensor,
    num_tokens: int,
    top_k: int,
    topk_ids: torch.Tensor,
    packed: Any,
) -> bool:
    """The PZ branch of maybe_down: the same guards on inputs and outputs, plus topk_ids."""
    if top_k != TOP_K or input_scale is not None:
        return False
    if inputs.dtype != torch.bfloat16 or outputs.dtype != torch.bfloat16:
        return False
    if torch.compiler.is_compiling():
        return False
    rows = num_tokens * top_k
    if (
        topk_ids.dim() != 2
        or tuple(topk_ids.shape) != (num_tokens, TOP_K)
        or not topk_ids.is_contiguous()
        or topk_ids.device != inputs.device
        or inputs.dim() != 2
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
    return _prefill_run(inputs, topk_ids, packed, outputs, rows)


def maybe_down(
    layer: torch.nn.Module,
    inputs: torch.Tensor,
    input_scale: torch.Tensor | None,
    outputs: torch.Tensor,
    num_tokens: int,
    top_k: int,
    block_m: int | None,
    moe_kwargs: dict[str, Any],
    topk_ids: torch.Tensor | None = None,
) -> bool:
    """Runs the w2 GEMM with zskip and returns True, or returns False (caller runs humming)."""
    packed = getattr(layer, "zskip_w2", None)
    if packed is None or not ENABLED:
        return False
    if (
        topk_ids is not None
        and PREFILL
        and PREFILL_MIN_TOKENS <= num_tokens <= PREFILL_MAX_TOKENS
        and getattr(layer, "zskip_prefill_ok", False)
    ):
        return _prefill_down(inputs, input_scale, outputs, num_tokens, top_k, topk_ids, packed)
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

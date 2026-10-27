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
block_m 16, BF16 activations, no input scale); with ZSKIP=0 routed_sparse serves them.

Prefill (more than 8 tokens): on the MoE layers in PREFILL_LAYERS it also serves the w2 GEMM,
with the call re-aligned to expert blocks of 32 rows (config PREFILL_CONFIG: four 8-row MMAs
share each dequantised weight fragment). humming's indexed w2 is batch invariant, so the output
is the same bits whatever the block size. Other layers (and ZSKIP_PREFILL=0) keep humming. The repack keeps every byte of the checkpoint tensors next to humming's own
copy (about 357 MB per layer). ZSKIP=0 in the environment disables it.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    import zskip as _zskip
except ImportError:
    _zskip = None

# pzskip (harness/kernels/pzskip): the same zero-row skip for prefill-sized calls (humming's
# block_m 64 bucket), reading this layer's zskip repack. PZSKIP=0 disables it.
try:
    import pzskip as _pzskip
except ImportError:
    _pzskip = None
PZ_ENABLED = os.environ.get("PZSKIP", "1") != "0"
# config 356: 32-row expert blocks (sparser union masks than humming's 64-row blocks), 384 columns,
# config 333's arithmetic with a leaner per-group instruction stream; PZSKIP_CONFIG=333 restores 333
# LOCAL PATCH (cudafast-nemotron35, pzhalf): config 369 = 356 reading humming's 64-row prefill
# layout as 32-row halves, so the 32-row moe_align is not run; PZSKIP_HALF64=0 keeps 356.
# LOCAL PATCH (cudafast-nemotron35, pz S6): the prefill routed down (pzskip 369) reads the
# 6-bit swap scale plane (zskip to_s6_swap, config 379) when its load-time check is bitwise
# equal to humming; PZSKIP_S6=0 keeps 369 and the 8-bit plane.
# LOCAL PATCH (cudafast-nemotron35): default off; the prefill down keeps config 369 on the 8-bit
# plane. PZSKIP_S6=1 builds the 6-bit swap plane and admits config 379 as before.
PZ_S6_ENABLED = os.environ.get("PZSKIP_S6", "0") != "0"
PZ_S6_CONFIG = 379
PZ_CONFIG = int(os.environ.get(
    "PZSKIP_CONFIG", "369" if os.environ.get("PZSKIP_HALF64", "1") != "0" else "356"))
PZ_BLOCK_M = 64  # humming's prefill bucket that pzskip replaces
PZ_MAX_TOKENS = 8192

ENABLED = os.environ.get("ZSKIP", "1") != "0"
# kernel v4 (warp per 64-column tile, weights as the MMA A operand) on the "swap" repack
LAYOUT = "swap"
CONFIG = 100
FUSED_PREP = os.environ.get("ZSKIP_FUSED_PREP", "1") != "0"
FUSED_CONFIG = 108
MAX_TOKENS = 8
PAIR_ENABLED = os.environ.get("ZSKIP_DADON", "1") != "0"
PAIR_CONFIG = 113
PAIR_FUSED_CONFIG = 118
# the same pair kernels without the dequantisation of 8-row k halves whose mask byte is zero
PAIR_HALFSKIP = os.environ.get("ZSKIP_HALFSKIP", "1") != "0"
PAIR_HALFSKIP_CONFIG = {PAIR_CONFIG: 114, PAIR_FUSED_CONFIG: 119}
# Lean paired decode (harness/kernels/zskip/csrc/zskip_lean.cuh): config 119's per-output arithmetic
# with a shorter per-group instruction stream and per-lane L2 prefetch; with ZSKIP_LEAN=1 it replaces
# the checked pair config for M=5 after a bitwise check against humming and that config (on by
# default; ZSKIP_LEAN=0 keeps the pair config): -14 us per call alone, -5 us in situ.
LEAN_ENABLED = os.environ.get("ZSKIP_LEAN", "1") == "1"
LEAN_CONFIG = int(os.environ.get("ZSKIP_LEAN_CONFIG", "653"))
# LOCAL PATCH (cudafast-nemotron35, zskip S6): the lean config's scale plane as 6-bit offsets
# (harness/kernels/zskip to_s6, config 663 = 653 with the 6-bit plane decoded to the same bytes
# in registers). ZSKIP_S6=0 keeps 653 and the 8-bit plane.
# LOCAL PATCH (i34-9 ddcdd302): default off, so the lean paired decode keeps config 653 on the
# 8-bit plane; ZSKIP_S6=1 builds the 6-bit plane and admits config 663 as before.
S6_ENABLED = os.environ.get("ZSKIP_S6", "0") != "0"
S6_CONFIG = 663
# Prefill: MoE layers whose relu^2 activations are sparse enough for the skip to beat humming's
# 64-row blocks on GB10 (measured per layer on prefill-length calls; the others lose). Layer 51,
# the last, is left out: the boot prefill graph routes all its rows through one row's experts.
PREFILL_ENABLED = ENABLED and os.environ.get("ZSKIP_PREFILL", "1") != "0"
PREFILL_LAYERS = frozenset(
    int(v)
    for v in os.environ.get(
        "ZSKIP_PREFILL_LAYERS", "1,3,6,8,10,13,15,34,36,38,40,43,45,47,49"
    ).split(",")
    if v.strip()
)
PREFILL_CONFIG = 405
PREFILL_BM = 32
PREFILL_MAX_TOKENS = 4096
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
    layer.zskip_prefill = PREFILL_ENABLED and _layer_index(layer) in PREFILL_LAYERS
    layer.zskip_config = CONFIG
    if not _self_check(layer):
        layer.zskip_w2 = None
        return
    if FUSED_PREP and _fused_check(layer):
        layer.zskip_config = FUSED_CONFIG
        logger.info_once("zskip: fused prep config %d for <= 8 tokens", FUSED_CONFIG)
    logger.info_once("zskip: zero-row-skipping routed down projection enabled for <= 8 tokens")
    layer.pzskip_ok = False
    if PZ_ENABLED and _pzskip is not None and _pzskip.available():
        _pzskip.scratch(
            gs.device,
            _pzskip.ws_blocks(PZ_MAX_TOKENS * TOP_K, NUM_EXPERTS, PZ_CONFIG),
        )
        if _pz_self_check(layer):
            layer.pzskip_ok = True
            logger.info_once("pzskip: zero-row-skipping routed down projection enabled for prefill")
            layer.pz_s6 = None
            if PZ_S6_ENABLED and PZ_CONFIG == 369 and PZ_S6_CONFIG in getattr(_pzskip, "S6_CONFIGS", ()):
                _prepare_pz_s6(layer)
    if layer.zskip_prefill:
        logger.info_once(
            "zskip: prefill routed down projection enabled on layers %s",
            ",".join(map(str, sorted(PREFILL_LAYERS))),
        )

    # Preserve all original checks and fallback choices before trying the new decode path.
    layer.zskip_pair_w2 = None
    layer.zskip_pair_config = None
    if PAIR_ENABLED:
        _prepare_pair(layer)
    if LEAN_ENABLED and getattr(layer, "zskip_pair_w2", None) is not None:
        _prepare_lean(layer)
    if S6_ENABLED and getattr(layer, "zskip_pair_config", None) == LEAN_CONFIG == 653:
        _prepare_s6(layer)
    layer.zskip_pair_prefix_ok = False
    if (getattr(layer, "zskip_pair_w2", None) is not None
        and os.environ.get("ZSKIP_VERIFY_PREFIX", "1") != "0"
        and _zskip.prefix_available()):
        try:
            layer.zskip_pair_prefix_ok = _pair_check(
                layer, layer.zskip_pair_w2, layer.zskip_pair_config, prefix=True
            )
        except Exception as exc:  # optional prefix; original checked pair remains
            logger.warning_once("zskip: prefix check unavailable (%r); full pair retained", exc)


def _prepare_pair(layer: torch.nn.Module) -> None:
    """New buffers and kernels are optional; any failure leaves checked 108/100 intact."""
    try:
        # Check library support before allocating another copy of w2.
        _zskip.config_info(PAIR_CONFIG)
        pair = _zskip.to_swap2(layer.zskip_w2)
        if not _pair_check(layer, pair, PAIR_CONFIG):
            return
        config = PAIR_CONFIG
        if FUSED_PREP:
            try:
                _zskip.config_info(PAIR_FUSED_CONFIG)
                if _pair_check(layer, pair, PAIR_FUSED_CONFIG):
                    config = PAIR_FUSED_CONFIG
            except Exception as exc:  # noqa: BLE001
                logger.warning_once("zskip: paired fused prep unavailable (%r); config 113", exc)
        if PAIR_HALFSKIP:
            try:
                skip = PAIR_HALFSKIP_CONFIG[config]
                _zskip.config_info(skip)
                prefix = (os.environ.get("ZSKIP_VERIFY_PREFIX", "1") != "0"
                          and _zskip.prefix_available())
                if _pair_check(layer, pair, skip, ms=range(1, 9), ref_config=config) and (
                    not prefix or _pair_check(layer, pair, skip, prefix=True)
                ):
                    logger.info_once("zskip: empty-half skip config %d replaces %d", skip, config)
                    config = skip
            except Exception as exc:  # noqa: BLE001
                logger.warning_once("zskip: empty-half skip unavailable (%r); config %d", exc, config)
        # VSW changes scheduling only; retain the original checked pair on any failure.
        if os.environ.get("VSW_W2_CONFIG", "1") != "0" and config == 119:
            if (_pair_check(layer, pair, 113, ms=range(1, 9), ref_config=config)
                and (os.environ.get("ZSKIP_VERIFY_PREFIX", "1") == "0"
                     or not _zskip.prefix_available()
                     or _pair_check(layer, pair, 113, prefix=True))):
                config = 113
        logger.info("VSW_W2_SELECTED config=%d layer=%s", config, layer.layer_name)
        layer.zskip_pair_w2 = pair
        layer.zskip_pair_config = config
        logger.info_once("zskip: dadon paired decode config %d enabled for M=5", config)
    except Exception as exc:  # noqa: BLE001
        logger.warning_once("zskip: paired decode unavailable (%r); checked legacy config retained", exc)


def _prepare_lean(layer: torch.nn.Module) -> None:
    """Optional: any failure keeps the checked pair config."""
    try:
        _zskip.config_info(LEAN_CONFIG)
        ref = layer.zskip_pair_config
        prefix = os.environ.get("ZSKIP_VERIFY_PREFIX", "1") != "0" and _zskip.prefix_available()
        if _pair_check(layer, layer.zskip_pair_w2, LEAN_CONFIG, ms=range(1, 9), ref_config=ref) and (
            not prefix or _pair_check(layer, layer.zskip_pair_w2, LEAN_CONFIG, prefix=True)
        ):
            layer.zskip_pair_config = LEAN_CONFIG
            logger.info_once("zskip: lean paired decode config %d replaces %d for M=5", LEAN_CONFIG, ref)
    except Exception as exc:  # noqa: BLE001
        logger.warning_once("zskip: lean paired decode unavailable (%r); config %s",
                            exc, getattr(layer, "zskip_pair_config", None))


def _prepare_s6(layer: torch.nn.Module) -> None:
    """Optional: the 6-bit scale plane for the lean paired decode; every failure keeps 653 and
    the 8-bit plane. The swap2 nibbles are shared; only the scale plane is new."""
    try:
        if S6_CONFIG not in getattr(_zskip, "S6_CONFIGS", ()):
            return
        _zskip.config_info(S6_CONFIG)
        old = layer.zskip_pair_w2
        s6 = _zskip.to_s6(old)
        if s6 is None:
            logger.info_once("zskip: 6-bit scale plane not applicable (scale range)")
            return
        prefix = os.environ.get("ZSKIP_VERIFY_PREFIX", "1") != "0" and _zskip.prefix_available()
        if _pair_check(layer, s6, S6_CONFIG, ms=range(1, 9), ref_config=LEAN_CONFIG,
                       ref_packed=old) and (
            not prefix or _pair_check(layer, s6, S6_CONFIG, prefix=True)
        ):
            layer.zskip_pair_w2 = s6
            layer.zskip_pair_config = S6_CONFIG
            logger.info_once("zskip: 6-bit scale plane config %d replaces %d for M=5", S6_CONFIG, LEAN_CONFIG)
    except Exception as exc:  # noqa: BLE001
        logger.warning_once("zskip: 6-bit scale plane unavailable (%r); config %s",
                            exc, getattr(layer, "zskip_pair_config", None))


def _pair_check(layer: torch.nn.Module, packed: Any, config: int, prefix: bool = False,
                ms: Any = (4, 5, 8), ref_config: int | None = None, ref_packed: Any = None) -> bool:
    """Load-time, bitwise against humming: M=4/5/8 (or ms), independent plus crowded routing.

    Compare all slots, including exact negative-zero activations and all-zero groups.
    The new config never serves prefill. No random state is taken from a request.
    With ref_config (the checked config it replaces) the call also runs ref_config on the same
    inputs, which must match too, and each input has empty low and high 8-row k halves.
    """
    import json

    from vllm import envs
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
        _humming_batch_invariant,
    )
    from vllm.model_executor.layers.fused_moe.experts.humming_tuned_moe import tuned_moe_configs
    from vllm.utils.humming import GemmType, HummingMethod

    configs = tuned_moe_configs(layer)
    if configs is None:
        return False
    dev = layer.w2_global_scale.device
    compute_config = json.dumps({
        "use_batch_invariant": _humming_batch_invariant(),
        "use_f16_accum": envs.VLLM_HUMMING_USE_F16_ACCUM,
        "gemm_type": GemmType.INDEXED.value,
    })
    for m in ((5, 6) if prefix else ms):
        for crowded in (False, True):
            ids = [[(29*i + (0 if crowded else 17*t)) % NUM_EXPERTS
                    for i in range(TOP_K)] for t in range(m)]
            topk_ids = torch.tensor(ids, dtype=torch.int32, device=dev)
            gen = torch.Generator(device=dev)
            gen.manual_seed(8713 + m)
            u = torch.randn(m*TOP_K, INTER, generator=gen, device=dev)
            keep = torch.rand(m*TOP_K, INTER, generator=gen, device=dev) < 0.1
            act = (u.relu().square() * keep).to(torch.bfloat16).contiguous()
            act[:, :16] = 0  # a wholly empty k16 group
            if ref_config is not None:
                act[:, 16:24] = 0  # group 1: only its high half can be live
                act[:, 40:48] = 0  # group 2: only its low half can be live
            # All-zero rows exercise empty lists; signed zeros exercise the bitmask semantics.
            act[0].zero_()
            bits = act.view(torch.int16)
            bits[bits == 0] = -32768
            sorted_ids, expert_ids, ntp = _align(topk_ids, 16)
            ref = torch.empty(m*TOP_K, HIDDEN, dtype=torch.bfloat16, device=dev)
            HummingMethod.forward_layer(
                layer=layer, inputs=act, input_scale=None, outputs=ref, sublayer_name="w2",
                sorted_ids=sorted_ids, expert_ids=expert_ids, num_tokens_padded=ntp,
                compute_config=compute_config, valid_shape_m=m*TOP_K, top_k=1,
                tuning_config=json.dumps(configs[1]),
            )
            out = torch.empty_like(ref)
            for count in (tuple(range(1, m + 1)) if prefix else (None,)):
                vrows = (torch.tensor([count], dtype=torch.int32, device=dev)
                         if count is not None else None)
                _zskip.down(act, sorted_ids, expert_ids, ntp, packed, out,
                            mode=_zskip.SKIP, config=config, max_blocks=m*TOP_K,
                            verify_rows=vrows)
                rows = m*TOP_K if count is None else count*TOP_K
                if not torch.equal(out[:rows].view(torch.int16), ref[:rows].view(torch.int16)):
                    logger.warning("zskip: paired prefix=%s config %d differs on %s",
                                   prefix, config, getattr(layer, "layer_name", "?"))
                    return False
                if ref_config is not None:
                    old = torch.empty_like(ref)
                    _zskip.down(act, sorted_ids, expert_ids, ntp,
                                packed if ref_packed is None else ref_packed, old,
                                mode=_zskip.SKIP, config=ref_config, max_blocks=m*TOP_K,
                                verify_rows=vrows)
                    if not torch.equal(out[:rows].view(torch.int16), old[:rows].view(torch.int16)):
                        logger.warning("zskip: config %d differs from config %d on %s",
                                       config, ref_config, getattr(layer, "layer_name", "?"))
                        return False
                if prefix and not torch.equal(
                    out[rows:].view(torch.int16), torch.zeros_like(out[rows:]).view(torch.int16)
                ):
                    return False
    return True


def _pz_self_check(layer: torch.nn.Module, packed: Any = None, config: int | None = None) -> bool:
    """Load-time guard: humming's w2 and pzskip on a deterministic 496-token call, bitwise."""
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
    m = 496
    gen = torch.Generator(device=dev)
    gen.manual_seed(4242)
    topk_ids = torch.stack(
        [torch.randperm(NUM_EXPERTS, generator=gen, device=dev)[:TOP_K] for _ in range(m)]
    ).to(torch.int32)
    u = torch.randn(m * TOP_K, INTER, generator=gen, device=dev)
    keep = torch.rand(m * TOP_K, INTER, generator=gen, device=dev) < 0.15
    act = (u.relu().square() * keep).to(torch.bfloat16).contiguous()
    sorted_ids, expert_ids, ntp = moe_align_block_size(
        topk_ids=topk_ids,
        block_size=PZ_BLOCK_M,
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
    config = PZ_CONFIG if config is None else config
    if _pzskip.block_rows(config) != PZ_BLOCK_M:
        sorted_ids, expert_ids, ntp = _align(topk_ids, _pzskip.block_rows(config))
    _pzskip.down(act, sorted_ids, expert_ids, ntp, layer.zskip_w2 if packed is None else packed,
                 out, wpc=config)
    ok = torch.equal(out.view(torch.int16), ref.view(torch.int16))
    if not ok:
        logger.warning(
            "pzskip: load-time check differs from humming on %s; humming keeps the prefill down",
            getattr(layer, "layer_name", "?"),
        )
    return ok


def _prepare_pz_s6(layer: torch.nn.Module) -> None:
    """Optional: any failure keeps 369 on the 8-bit plane."""
    try:
        s6 = _zskip.to_s6_swap(layer.zskip_w2)
        if s6 is None:
            return
        if _pz_self_check(layer, packed=s6, config=PZ_S6_CONFIG):
            layer.pz_s6 = s6
            logger.info_once("pzskip: 6-bit scale plane config %d replaces %d for prefill",
                             PZ_S6_CONFIG, PZ_CONFIG)
    except Exception as exc:  # noqa: BLE001
        logger.warning_once("pzskip: 6-bit scale plane unavailable (%r); config %d", exc, PZ_CONFIG)


def _layer_index(layer: torch.nn.Module) -> int:
    import re

    m = re.search(r"layers\.(\d+)\.", str(getattr(layer, "layer_name", "")))
    return int(m.group(1)) if m else -1


def _align(topk_ids: torch.Tensor, block_size: int):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    return moe_align_block_size(
        topk_ids=topk_ids,
        block_size=block_size,
        num_experts=NUM_EXPERTS,
        expert_map=None,
        ignore_invalid_experts=True,
    )


def _fused_check(layer: torch.nn.Module) -> bool:
    """Decode-sized load-time check of FUSED_CONFIG. False keeps CONFIG (prefill already checked)."""
    try:
        _zskip.config_info(FUSED_CONFIG)
        return _self_check(layer, config=FUSED_CONFIG, quiet=True, prefill=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning_once("zskip: fused prep unavailable (%r); config %d", exc, CONFIG)
        return False


def _self_check(
    layer: torch.nn.Module,
    config: int = CONFIG,
    quiet: bool = False,
    prefill: bool | None = None,
) -> bool:
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
    if prefill is None:
        prefill = bool(getattr(layer, "zskip_prefill", False))
    for m in (4, MAX_TOKENS) + ((40,) if prefill else ()):
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
        if m > MAX_TOKENS:
            sorted_ids, expert_ids, ntp = _align(topk_ids, PREFILL_BM)
            cfg, mb = PREFILL_CONFIG, sorted_ids.numel() // PREFILL_BM
        else:
            cfg, mb = config, m * TOP_K
        _zskip.down(
            act,
            sorted_ids,
            expert_ids,
            ntp,
            layer.zskip_w2,
            out,
            mode=_zskip.SKIP,
            config=cfg,
            max_blocks=mb,
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
    topk_ids: torch.Tensor | None = None,
) -> bool:
    """Runs the w2 GEMM with zskip and returns True, or returns False (caller runs humming)."""
    packed = getattr(layer, "zskip_w2", None)
    if packed is None or not ENABLED:
        return False
    prefill = num_tokens > MAX_TOKENS
    if (
        prefill
        and num_tokens <= PZ_MAX_TOKENS
        and block_m == PZ_BLOCK_M
        and getattr(layer, "pzskip_ok", False)
        and _pz_down(layer, inputs, input_scale, outputs, num_tokens, top_k, moe_kwargs, topk_ids)
    ):
        return True
    if prefill:
        if (
            not getattr(layer, "zskip_prefill", False)
            or num_tokens > PREFILL_MAX_TOKENS
            or topk_ids is None
            or top_k != TOP_K
        ):
            return False
    elif not (1 <= num_tokens <= MAX_TOKENS) or top_k != TOP_K or block_m != 16:
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
    if prefill:
        sorted_ids, expert_ids, ntp = _align(topk_ids, PREFILL_BM)
        _zskip.down(
            inputs,
            sorted_ids,
            expert_ids,
            ntp,
            packed,
            outputs,
            mode=_zskip.SKIP,
            config=PREFILL_CONFIG,
            max_blocks=sorted_ids.numel() // PREFILL_BM,
        )
        return True
    # M=5 / M=6 are the depth-4 / depth-5 verifies. Other shapes keep main's checked decode selection.
    pair = getattr(layer, "zskip_pair_w2", None)
    use_pair = num_tokens in (5, 6) and pair is not None
    vrows = None
    if use_pair and getattr(layer, "zskip_pair_prefix_ok", False):
        smallk = sys.modules.get("smallk")
        if smallk is not None and hasattr(smallk, "live_verify_rows"):
            vrows = smallk.live_verify_rows(inputs.device)
    _zskip.down(
        inputs,
        moe_kwargs["sorted_ids"],
        moe_kwargs["expert_ids"],
        moe_kwargs["num_tokens_padded"],
        pair if use_pair else packed,
        outputs,
        mode=_zskip.SKIP,
        config=layer.zskip_pair_config if use_pair else getattr(layer, "zskip_config", CONFIG),
        max_blocks=rows,
        verify_rows=vrows,
    )
    return True


def _pz_down(
    layer: torch.nn.Module,
    inputs: torch.Tensor,
    input_scale: torch.Tensor | None,
    outputs: torch.Tensor,
    num_tokens: int,
    top_k: int,
    moe_kwargs: dict[str, Any],
    topk_ids: torch.Tensor | None = None,
) -> bool:
    """Prefill-sized w2 with pzskip (humming's block_m 64 calls); False: caller runs humming."""
    if top_k != TOP_K or input_scale is not None:
        return False
    if inputs.dtype != torch.bfloat16 or outputs.dtype != torch.bfloat16:
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
    bm = _pzskip.block_rows(PZ_CONFIG)
    if bm == PZ_BLOCK_M:
        sorted_ids = moe_kwargs["sorted_ids"]
        expert_ids = moe_kwargs["expert_ids"]
        ntp = moe_kwargs["num_tokens_padded"]
    elif topk_ids is not None:
        sorted_ids, expert_ids, ntp = _align(topk_ids, bm)
    else:
        return False
    s6 = getattr(layer, "pz_s6", None)
    if s6 is not None:
        _pzskip.down(inputs, sorted_ids, expert_ids, ntp, s6, outputs, wpc=PZ_S6_CONFIG)
    else:
        _pzskip.down(inputs, sorted_ids, expert_ids, ntp, layer.zskip_w2, outputs, wpc=PZ_CONFIG)
    return True

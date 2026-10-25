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
PZ_CONFIG = int(os.environ.get("PZSKIP_CONFIG", "356"))
PZ_BLOCK_M = 64  # humming's prefill bucket that pzskip replaces
PZ_MAX_TOKENS = 8192
# Exact schedule variants of PZ_CONFIG (pzskip.cu configs 357 and 360-364; 360-364 run on humming's
# own 64-row expert blocks, so an expert routed 33-64 rows loads its live weight rows once rather than
# once per 32-row block). A variant serves a layer only after bitwise checks against PZ_CONFIG on that
# layer. Once PZ_VARIANT_LAYERS layers have checked, CUDA graphs over their weights time PZ_CONFIG and
# every variant that passed on all of them; the fastest replaces PZ_CONFIG if it takes under
# PZ_VARIANT_BAR of its time. PZSKIP_VARIANTS=0 disables.
PZ_VARIANTS_ENABLED = os.environ.get("PZSKIP_VARIANTS", "1") != "0"
PZ_VARIANTS = (357, 360, 361, 362, 363, 364)
PZ_VARIANT_LAYERS = 4
PZ_VARIANT_BAR = 0.97
_pz_pending: list[tuple[torch.nn.Module, set[int]]] = []
_pz_choice: int | None = None  # None: not timed yet; 0: PZ_CONFIG stays

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
# Config 119's arithmetic on other schedules (zskip.PAIR_VARIANTS: ring depth, warps per CTA, tiles
# per warp). A variant serves a layer only after the same bitwise checks as 119 on that layer. Once
# VARIANT_LAYERS layers have checked, one CUDA graph over their distinct weights times 119 and every
# variant that passed on all of them; the fastest replaces 119 if it takes under VARIANT_BAR of its time.
VARIANTS_ENABLED = os.environ.get("ZSKIP_VARIANTS", "1") != "0"
VARIANT_LAYERS = 8
VARIANT_BAR = 0.97
_variant_pending: list[tuple[torch.nn.Module, set[int]]] = []
_variant_choice: int | None = None  # None: not timed yet; 0: 119 stays
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
            _pzskip.max_blocks(PZ_MAX_TOKENS * TOP_K, NUM_EXPERTS, _pzskip.block_rows(PZ_CONFIG)),
        )
        if _pz_self_check(layer):
            layer.pzskip_ok = True
            layer.pz_config = PZ_CONFIG
            logger.info_once("pzskip: zero-row-skipping routed down projection enabled for prefill")
            if PZ_VARIANTS_ENABLED:
                try:
                    _pz_admit(layer)
                except Exception as exc:  # noqa: BLE001
                    logger.warning_once("pzskip: schedule variants unavailable (%r); config %d kept",
                                        exc, PZ_CONFIG)
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
    if VARIANTS_ENABLED and getattr(layer, "zskip_pair_config", None) == PAIR_HALFSKIP_CONFIG[PAIR_FUSED_CONFIG]:
        try:
            _admit_variant(layer)
        except Exception as exc:  # noqa: BLE001
            logger.warning_once("zskip: schedule variants unavailable (%r); config %d kept", exc,
                                layer.zskip_pair_config)


def _variant_ok(layer: torch.nn.Module, config: int) -> bool:
    """The record's empty-half skip check with config 119 as the reference, plus its prefix check."""
    try:
        _zskip.config_info(config)
    except Exception:  # noqa: BLE001
        return False
    base, pair = layer.zskip_pair_config, layer.zskip_pair_w2
    prefix = os.environ.get("ZSKIP_VERIFY_PREFIX", "1") != "0" and _zskip.prefix_available()
    return _pair_check(layer, pair, config, ms=range(1, 9), ref_config=base) and (
        not prefix or _pair_check(layer, pair, config, prefix=True)
    )


def _admit_variant(layer: torch.nn.Module) -> None:
    global _variant_choice
    if _variant_choice == 0:
        return
    if _variant_choice is not None:
        if _variant_ok(layer, _variant_choice):
            layer.zskip_pair_config = _variant_choice
        return
    passed = {c for c in getattr(_zskip, "PAIR_VARIANTS", {}) if _variant_ok(layer, c)}
    _variant_pending.append((layer, passed))
    if len(_variant_pending) < VARIANT_LAYERS:
        return
    layers = [entry[0] for entry in _variant_pending]
    survivors = sorted(set.intersection(*(entry[1] for entry in _variant_pending)))
    _variant_pending.clear()
    _variant_choice = 0
    choice = _time_variants(layers, survivors) if survivors else 0
    _variant_choice = choice
    if choice:
        for checked in layers:
            checked.zskip_pair_config = choice


def _time_variants(layers: list[torch.nn.Module], configs: list[int]) -> int:
    """Median graph time of one M=5 call on each layer's own weights (cold in L2), interleaved."""
    dev = layers[0].w2_global_scale.device
    base = layers[0].zskip_pair_config
    gen = torch.Generator(device=dev)
    gen.manual_seed(5150)
    pool = torch.randperm(NUM_EXPERTS, generator=gen, device=dev)[:24]
    topk_ids = torch.stack(
        [pool[torch.randperm(24, generator=gen, device=dev)[:TOP_K]] for _ in range(5)]
    ).to(torch.int32)
    u = torch.randn(5 * TOP_K, INTER, generator=gen, device=dev)
    keep = torch.rand(5 * TOP_K, INTER, generator=gen, device=dev) < 0.1
    act = (u.relu().square() * keep).to(torch.bfloat16).contiguous()
    sorted_ids, expert_ids, ntp = _align(topk_ids, 16)
    out = torch.empty(5 * TOP_K, HIDDEN, dtype=torch.bfloat16, device=dev)

    def call(config: int) -> None:
        for checked in layers:
            _zskip.down(act, sorted_ids, expert_ids, ntp, checked.zskip_pair_w2, out,
                        mode=_zskip.SKIP, config=config, max_blocks=5 * TOP_K)

    graphs = {}
    for config in [base] + configs:
        call(config)
        torch.cuda.synchronize(dev)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call(config)
        graph.replay()
        torch.cuda.synchronize(dev)
        graphs[config] = graph
    times: dict[int, list[float]] = {config: [] for config in graphs}
    for _ in range(11):
        for config, graph in graphs.items():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times[config].append(start.elapsed_time(end))
    del graphs
    med = {config: sorted(v)[len(v) // 2] * 1e3 / len(layers) for config, v in times.items()}
    best = min(configs, key=med.__getitem__)
    logger.info("zskip: M=5 down us/call %s", {c: round(t, 1) for c, t in med.items()})
    if med[best] < VARIANT_BAR * med[base]:
        logger.info("zskip: schedule variant %d replaces config %d", best, base)
        return best
    return 0


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
        layer.zskip_pair_w2 = pair
        layer.zskip_pair_config = config
        logger.info_once("zskip: dadon paired decode config %d enabled for M=5", config)
    except Exception as exc:  # noqa: BLE001
        logger.warning_once("zskip: paired decode unavailable (%r); checked legacy config retained", exc)


def _pair_check(layer: torch.nn.Module, packed: Any, config: int, prefix: bool = False,
                ms: Any = (4, 5, 8), ref_config: int | None = None) -> bool:
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
    for m in ((5,) if prefix else ms):
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
            for count in ((1, 2, 3, 4, 5) if prefix else (None,)):
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
                    _zskip.down(act, sorted_ids, expert_ids, ntp, packed, old,
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


def _pz_self_check(layer: torch.nn.Module) -> bool:
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
    if _pzskip.block_rows(PZ_CONFIG) != PZ_BLOCK_M:
        sorted_ids, expert_ids, ntp = _align(topk_ids, _pzskip.block_rows(PZ_CONFIG))
    _pzskip.down(act, sorted_ids, expert_ids, ntp, layer.zskip_w2, out, wpc=PZ_CONFIG)
    ok = torch.equal(out.view(torch.int16), ref.view(torch.int16))
    if not ok:
        logger.warning(
            "pzskip: load-time check differs from humming on %s; humming keeps the prefill down",
            getattr(layer, "layer_name", "?"),
        )
    return ok


def _pz_inputs(dev: torch.device, m: int, seed: int, sigma: float):
    """Prefill-shaped routed w2 inputs: top-6 routing with expert popularity exp(sigma * N(0, 1))
    (sigma 0.8: about 1.25 32-row blocks per routed expert), and relu^2 activations with 55% of each
    expert's rows dead for all of its tokens."""
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    pop = sigma * torch.randn(NUM_EXPERTS, generator=gen, device=dev)
    u = torch.rand(m, NUM_EXPERTS, generator=gen, device=dev).clamp_min(1e-12)
    topk_ids = (pop - torch.log(-torch.log(u))).topk(TOP_K, dim=1).indices.to(torch.int32)
    live = torch.rand(NUM_EXPERTS, INTER, generator=gen, device=dev) >= 0.55
    x = torch.randn(m * TOP_K, INTER, generator=gen, device=dev)
    act = (x.relu().square() * live[topk_ids.reshape(-1).long()]).to(torch.bfloat16).contiguous()
    return topk_ids, act


def _pz_run(layer: torch.nn.Module, act: torch.Tensor, topk_ids: torch.Tensor, config: int,
            out: torch.Tensor | None = None) -> torch.Tensor:
    sorted_ids, expert_ids, ntp = _align(topk_ids, _pzskip.block_rows(config))
    if out is None:
        out = torch.zeros(act.shape[0], HIDDEN, dtype=torch.bfloat16, device=act.device)
    _pzskip.down(act, sorted_ids, expert_ids, ntp, layer.zskip_w2, out, wpc=config)
    return out


def _pz_variant_ok(layer: torch.nn.Module, config: int) -> bool:
    """Bitwise against PZ_CONFIG on this layer: 496 and 512 tokens with skewed routing, 257 with
    near-uniform routing, each also with every zero activation turned into -0."""
    dev = layer.w2_global_scale.device
    for i, (m, sigma) in enumerate(((496, 0.8), (512, 0.8), (257, 0.0))):
        topk_ids, act = _pz_inputs(dev, m, 6100 + i, sigma)
        neg = act.clone()
        neg[neg == 0] = -0.0
        for a in (act, neg):
            ref = _pz_run(layer, a, topk_ids, PZ_CONFIG)
            out = _pz_run(layer, a, topk_ids, config)
            if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                logger.warning("pzskip: variant %d differs from config %d on %s", config, PZ_CONFIG,
                               getattr(layer, "layer_name", "?"))
                return False
    return True


def _pz_admit(layer: torch.nn.Module) -> None:
    global _pz_choice
    if _pz_choice == 0:
        return
    if _pz_choice is not None:
        if _pz_variant_ok(layer, _pz_choice):
            layer.pz_config = _pz_choice
        return
    passed = {c for c in PZ_VARIANTS if _pz_variant_ok(layer, c)}
    _pz_pending.append((layer, passed))
    if len(_pz_pending) < PZ_VARIANT_LAYERS:
        return
    layers = [entry[0] for entry in _pz_pending]
    survivors = sorted(set.intersection(*(entry[1] for entry in _pz_pending)))
    _pz_pending.clear()
    _pz_choice = 0
    choice = _pz_time(layers, survivors) if survivors else 0
    _pz_choice = choice
    if choice:
        for checked in layers:
            checked.pz_config = choice


def _pz_time(layers: list[torch.nn.Module], configs: list[int]) -> int:
    """Median graph time of one 496-token call on each layer's own weights, interleaved."""
    dev = layers[0].w2_global_scale.device
    topk_ids, act = _pz_inputs(dev, 496, 6200, 0.8)
    out = torch.empty(act.shape[0], HIDDEN, dtype=torch.bfloat16, device=dev)

    def call(config: int) -> None:
        for checked in layers:
            _pz_run(checked, act, topk_ids, config, out)

    graphs = {}
    for config in [PZ_CONFIG] + configs:
        call(config)
        torch.cuda.synchronize(dev)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call(config)
        graph.replay()
        torch.cuda.synchronize(dev)
        graphs[config] = graph
    times: dict[int, list[float]] = {config: [] for config in graphs}
    for _ in range(9):
        for config, graph in graphs.items():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times[config].append(start.elapsed_time(end))
    del graphs
    med = {config: sorted(v)[len(v) // 2] * 1e3 / len(layers) for config, v in times.items()}
    best = min(configs, key=med.__getitem__)
    logger.info("pzskip: 496-token down us/layer %s", {c: round(t, 1) for c, t in med.items()})
    if med[best] < PZ_VARIANT_BAR * med[PZ_CONFIG]:
        logger.info("pzskip: schedule variant %d replaces config %d", best, PZ_CONFIG)
        return best
    return 0


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
    # M=5 is depth-4 verify. Other shapes keep main's checked decode selection.
    pair = getattr(layer, "zskip_pair_w2", None)
    use_pair = num_tokens == 5 and pair is not None
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
    config = getattr(layer, "pz_config", PZ_CONFIG)
    bm = _pzskip.block_rows(config)
    if bm == PZ_BLOCK_M:
        sorted_ids = moe_kwargs["sorted_ids"]
        expert_ids = moe_kwargs["expert_ids"]
        ntp = moe_kwargs["num_tokens_padded"]
    elif topk_ids is not None:
        sorted_ids, expert_ids, ntp = _align(topk_ids, bm)
    else:
        return False
    _pzskip.down(inputs, sorted_ids, expert_ids, ntp, layer.zskip_w2, outputs, wpc=config)
    return True

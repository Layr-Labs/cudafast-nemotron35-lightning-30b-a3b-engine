# SPDX-License-Identifier: Apache-2.0
"""LOCAL PATCH (cudafast-nemotron35): the prefill (M=496) Mamba in_proj and out_proj FP8 GEMMs read
from a load-time packed copy of the weight (harness/kernels/fp8ws).

Same arithmetic as the pinned CUTLASS kernel: one ascending-K e4m3 m16n8k32 chain per output from
+0, then fp32 x scale_b, x scale_a, one RN to BF16. The pack is a byte permutation of the weight;
the activation is the producer's tensor, unchanged. Each layer's pack is admitted at load only if
its output is bitwise equal to ops.cutlass_scaled_mm on several synthetic activation sets and
scale pairs at the real shape; otherwise the layer keeps CUTLASS. The original weight stays (it
serves every other M through CUTLASS).

FP8WS=0 disables it at load. `active` is read at dispatch time (and so when a CUDA graph is
captured); the prefill graph can capture a second set with it off for an in-boot A/B
(prefill_graph.py, FP8WS_AB_FILE).
"""

import os

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger

logger = init_logger(__name__)
M = 496
# (K, N) of the column-major weights served: Mamba in_proj (2688 -> 10304), out_proj (4096 -> 2688)
SHAPES = {(2688, 10304), (4096, 2688)}
ENABLED = os.environ.get("FP8WS", "1") != "0"
active = True
# weight storage address -> (weight, pack); the weight reference keeps the address valid
_PACKS: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
_stats = {"admitted": 0, "rejected": 0, "dispatched": 0}


def has_packs() -> bool:
    return bool(_PACKS)


def _activations(device: torch.device, K: int) -> list[torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(20261027)
    sets = []
    for scale in (20.0, 0.02, 150.0, 1.0):
        # explicit dtypes: model loading runs under a BF16 default dtype
        x = torch.randn((M, K), device=device, generator=g, dtype=torch.float32) * scale
        sets.append(x.clamp(-448, 448).to(torch.float8_e4m3fn))
    # every E4M3 bit pattern (NaNs included), shifted per row
    bits = (torch.arange(M * K, device=device, dtype=torch.int64) * 7
            + torch.arange(M, device=device, dtype=torch.int64).repeat_interleave(K))
    sets.append((bits % 256).to(torch.uint8).view(torch.float8_e4m3fn).reshape(M, K))
    return sets


def register(weight: torch.Tensor, weight_scale: torch.Tensor | None,
             input_scale: torch.Tensor | None) -> bool:
    """Pack and admit one [K, N] column-major in_proj/out_proj weight. True if its prefill uses fp8ws."""
    if not ENABLED or tuple(weight.shape) not in SHAPES:
        return False
    K = weight.shape[0]
    try:
        import fp8ws
    except ImportError:
        return False
    try:
        packed = fp8ws.prepare(weight)
        if packed is None:
            return False
        dev = weight.device
        f32 = torch.float32
        scales = [(torch.tensor([0.0123], device=dev, dtype=f32), torch.tensor([1.7], device=dev, dtype=f32)),
                  (torch.tensor([3.0], device=dev, dtype=f32), torch.tensor([0.00071], device=dev, dtype=f32))]
        if (weight_scale is not None and input_scale is not None
                and weight_scale.numel() == 1 and input_scale.numel() == 1):
            scales.insert(0, (input_scale.detach().float().reshape(1).contiguous(),
                              weight_scale.detach().float().reshape(1).contiguous()))
        ok = torch.equal(fp8ws.unpack_reference(packed), weight.t().view(torch.uint8))
        for a in _activations(dev, K):
            for sa, sb in scales:
                if not ok:
                    break
                ref = ops.cutlass_scaled_mm(a, weight, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
                got = fp8ws.gemm(a, packed, sa, sb)
                ok = torch.equal(ref.view(torch.int16), got.view(torch.int16))
        torch.cuda.synchronize(dev)
    except Exception as exc:  # noqa: BLE001  (any failure keeps CUTLASS)
        logger.warning("fp8ws: admission failed (%r); this layer keeps CUTLASS", exc)
        ok = False
    if not ok:
        _stats["rejected"] += 1
        logger.warning("fp8ws: output differs from CUTLASS; this layer keeps CUTLASS")
        return False
    _PACKS[weight.data_ptr()] = (weight, packed)
    # Original pack/CUTLASS admission above is authoritative. An optional
    # schedule compares with that exact original kernel and keeps it on refusal.
    try:
        from .fp8_early_release import register as register_early

        register_early(weight, packed, scales, _activations(dev, K))
    except Exception as early_error:
        logger.warning("fp8ws early-release qualification unavailable: %s",
                       type(early_error).__name__)
    _stats["admitted"] += 1
    if _stats["admitted"] in (1, 23, 46):
        logger.info("fp8ws: %d Mamba in_proj/out_proj weights admitted for the M=496 prefill (%d rejected); "
                    "FP8WS=0 disables", _stats["admitted"], _stats["rejected"])
    return True


def maybe_gemm(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor,
               sb: torch.Tensor) -> torch.Tensor | None:
    """fp8ws output for an admitted weight at M=496, else None (the caller runs CUTLASS)."""
    if not active or a.shape[0] != M:
        return None
    entry = _PACKS.get(b.data_ptr())
    if entry is None:
        return None
    weight, packed = entry
    if not (
        b.shape == weight.shape and b.stride() == weight.stride()
        and a.dtype == torch.float8_e4m3fn and a.shape[1] == weight.shape[0] and a.is_contiguous()
        and a.data_ptr() % 16 == 0 and a.device == packed.device
        and sa.dtype == sb.dtype == torch.float32 and sa.numel() == sb.numel() == 1
        and sa.is_contiguous() and sb.is_contiguous()
        and sa.device == sb.device == a.device
    ):
        return None
    import fp8ws

    if not _stats["dispatched"]:
        logger.info("fp8ws: M=496 Mamba projection dispatched to the packed kernel")
    _stats["dispatched"] += 1
    try:
        from .fp8_early_release import maybe_gemm as maybe_early
    except Exception:
        return fp8ws.gemm(a, packed, sa, sb)

    early = maybe_early(a, weight, packed, sa, sb)
    if early is not None:
        return early
    return fp8ws.gemm(a, packed, sa, sb)

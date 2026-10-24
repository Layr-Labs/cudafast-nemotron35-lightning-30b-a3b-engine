# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drafter-only projected join, BF16 residual and first RMSNorm in one grid.

The BF16 hidden projection has already been rounded by the original GEMV.
The joined residual is rounded again BEFORE its variance is computed. Both
normalization rounding conventions are admitted against the loaded RMSNorm;
unsupported shapes or a failed finite admission retain the original path.
All admission execution happens on the official runner during resident load.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


@triton.jit
def _projected_start_norm(
    hp, ep, gamma, out, residual,
    K: tl.constexpr, EPS: tl.constexpr, CAST_BEFORE_WEIGHT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    k = tl.arange(0, BLOCK)
    mask = k < K
    h = tl.load(hp + k, mask, other=0).to(tl.float32)
    e = tl.load(ep + k, mask, other=0).to(tl.float32)
    # This conversion is an arithmetic boundary, not merely a storage cast.
    joined = (h + e).to(tl.bfloat16)
    x = joined.to(tl.float32)
    variance = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / K
    normalized = x * tl.rsqrt(variance + EPS)
    w = tl.load(gamma + k, mask, other=1.0)
    if CAST_BEFORE_WEIGHT:
        normalized = normalized.to(w.dtype).to(tl.float32)
    y = normalized * w.to(tl.float32)
    tl.store(residual + k, joined, mask)
    tl.store(out + k, y.to(tl.bfloat16), mask)


def _start_norm_impl(
    h: torch.Tensor, e: torch.Tensor, weight: torch.Tensor,
    epsilon: float, cast_before_weight: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(h)
    residual = torch.empty_like(h)
    _projected_start_norm[(1,)](
        h, e, weight, out, residual,
        K=h.shape[1], EPS=epsilon, CAST_BEFORE_WEIGHT=cast_before_weight,
        BLOCK=triton.next_power_of_2(h.shape[1]),
        num_warps=4, enable_fp_fusion=False,
    )
    return out, residual


def _start_norm_fake(
    h: torch.Tensor, e: torch.Tensor, weight: torch.Tensor,
    epsilon: float, cast_before_weight: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(h), torch.empty_like(h)


direct_register_custom_op(
    op_name="mtp_projected_start_norm", op_func=_start_norm_impl,
    fake_impl=_start_norm_fake,
)


@torch.no_grad()
def prepare_projected_start(layer: torch.nn.Module) -> bool:
    """Finite, load-only admission; no device decisions in the captured path."""
    layer._mtp_start_norm_mode = None
    if os.environ.get("MTP_PROJECTED_START_NORM", "1") == "0":
        return False
    norm = getattr(layer, "norm", None)
    weight = getattr(norm, "weight", None)
    wh = getattr(layer, "eh_w_h", None)
    if (
        not getattr(layer, "has_start_projections", False)
        or weight is None or wh is None
        or not weight.is_cuda or weight.dtype != torch.bfloat16
        or tuple(weight.shape) != (2688,) or not weight.is_contiguous()
        or getattr(norm, "variance_size_override", None) is not None
        or not getattr(norm, "pass_weight", False)
    ):
        return False
    try:
        eps = float(norm.variance_epsilon)
        i = torch.arange(2688, dtype=torch.float32, device=weight.device)[None, :]
        cases = [
            (torch.zeros_like(i).to(torch.bfloat16), torch.zeros_like(i)),
            ((torch.sin(i * 0.013) * 1.75).to(torch.bfloat16),
             torch.cos(i * 0.017) * 0.61),
            ((torch.cos(i * 0.031) * 12).to(torch.bfloat16),
             torch.sin(i * 0.019) * 0.0003),
        ]
        h = (torch.sin(i * 0.007) * 3).to(torch.bfloat16)
        cases.append((h, -h.float() + torch.sin(i * 0.023) * 0.002))
        for mode in (False, True):
            admitted = True
            for hp, ep in cases:
                ref_residual = (hp.float() + ep).to(torch.bfloat16)
                reference = norm(ref_residual)
                y, r = _start_norm_impl(hp, ep, weight, eps, mode)
                if not torch.equal(r.view(torch.int16), ref_residual.view(torch.int16)):
                    admitted = False
                    break
                if not torch.equal(y.view(torch.int16), reference.view(torch.int16)):
                    admitted = False
                    break
            if admitted:
                layer._mtp_start_norm_mode = mode
                logger.info("MTP projected start norm admitted, cast-before-weight=%s", mode)
                return True
    except Exception as exc:  # official load keeps original path on failure
        logger.warning("MTP projected start norm disabled: %r", exc)
        return False
    logger.info("MTP projected start norm kept original normalization")
    return False

# SPDX-License-Identifier: Apache-2.0
"""LOCAL PATCH (cudafast-nemotron35): the Mamba FP8 projections' 1..8-row path read
from a lossless 7-bit coding of the weight (harness/kernels/fp8c).

The coding is a load-time repack of the fp8small pack: fp8c.prepare decodes the whole
buffer with the kernel's decoder and keeps it only if every byte equals the pack and
the buffer is at most 93% of the original; otherwise the layer keeps fp8small. The
kernel hands the same bytes to the same e4m3 MMA chain and epilogue as fp8small, so
the BF16 output bits are unchanged. FP8C=0 disables it.
"""

import os

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)
SHAPES = {(10304, 2688), (2688, 4096)}
_logged = False
_BASE7 = {}


def prepare_weight(packed: torch.Tensor) -> torch.Tensor | None:
    """The fp8c buffer for an fp8small pack [N, K], or None to keep fp8small."""
    if os.environ.get("FP8C", "1") == "0":
        return None
    try:
        import fp8c
    except ImportError:
        return None
    rejected = fp8c.stats["rejected"]
    try:
        comp = fp8c.prepare(packed)
    except Exception as exc:  # noqa: BLE001  (a failed encoder or decoder keeps fp8small)
        logger.warning("fp8c: prepare failed (%r); this layer keeps fp8small", exc)
        return None
    if comp is None and fp8c.stats["rejected"] > rejected:
        logger.warning("fp8c: %s; this layer keeps fp8small", fp8c.stats["last_reason"])
    if comp is not None:
        _BASE7[(comp.device.index, comp.data_ptr())] = comp
        try:
            import fp8d6
            selected = fp8d6.prepare(packed, comp)
            if selected is not None:
                return selected
        except Exception as exc:
            logger.warning("fp8d6 unavailable (%r); current fp8c retained", exc)
    return comp


def _log_once(fp8c) -> None:
    global _logged
    if _logged:
        return
    _logged = True
    s = fp8c.stats
    raw = max(s["raw_bytes"], 1)
    logger.info(
        "fp8c reference: %d Mamba projections admitted to lossless 7-bit storage "
        "(%.1f MB -> %.1f MB, %.3f bits/weight, %d kept fp8small); optional fp8d6 logged separately",
        s["tensors"], s["raw_bytes"] / 1e6, s["bytes"] / 1e6, 8.0 * s["bytes"] / raw,
        s["rejected"],
    )


def _scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    comp: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
) -> torch.Tensor:
    # Runs on real tensor metadata inside the custom op, so a symbolic compiled
    # range can contain both prefill and small decode M.
    k, n = b.shape
    if (
        1 <= a.shape[0] <= 8
        and (n, k) in SHAPES
        and a.dtype == b.dtype == torch.float8_e4m3fn
        and comp.dtype == torch.uint8
        and a.is_contiguous()
        and a.data_ptr() % 16 == 0
        and a.device == b.device == comp.device == sa.device == sb.device
        and sa.dtype == sb.dtype == torch.float32
        and sa.numel() == sb.numel() == 1
        and sa.is_contiguous()
        and sb.is_contiguous()
        and a.shape[1] == k
    ):
        import fp8c

        try:
            import fp8d6
        except ImportError:
            fp8d6 = None
        if fp8d6 is not None and fp8d6.owns(comp):
            return fp8d6.gemm(a, comp, sa, sb, n, k)
        known = _BASE7.get((comp.device.index, comp.data_ptr()))
        if (known is not None and known.shape == comp.shape
                and known.stride() == comp.stride()):
            _log_once(fp8c)
            return fp8c.gemm(a, comp, sa, sb, n, k)
    return ops.cutlass_scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)


def _scaled_mm_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    comp: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.bfloat16)


direct_register_custom_op("fp8c_scaled_mm", _scaled_mm, fake_impl=_scaled_mm_fake)

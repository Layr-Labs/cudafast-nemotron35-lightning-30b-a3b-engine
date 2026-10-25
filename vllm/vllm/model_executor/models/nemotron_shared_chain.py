# SPDX-License-Identifier: Apache-2.0
"""Current-call shared up/zero-mask/down chain; original NVFP4/BF16 math."""

import os
import sys

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
)
from vllm.utils.torch_utils import direct_register_custom_op

try:
    import marlinx as _marlinx
except ImportError:
    _marlinx = None

logger = init_logger(__name__)
_ENABLED = (os.environ.get("MARLINX_SHARED_CHAIN", "1") != "0"
            and os.environ.get("MARLINX_SHARED_RELU2", "1") != "0")
_PREFIX = os.environ.get("MARLINX_SHARED_PREFIX", "1") != "0"


def _stock_linear(x, layer, n, k):
    return apply_fp4_marlin_linear(
        input=x, weight=layer.weight, weight_scale=layer.weight_scale,
        weight_global_scale=layer.weight_global_scale, workspace=layer.workspace,
        size_n=n, size_k=k, bias=None,
    )


def _weight_metadata_ok(layer, n, k):
    if not getattr(layer, "_marlinx_ok", False):
        return False
    if (
        getattr(layer, "tp_size", 1) != 1
        or getattr(layer, "bias", None) is not None
        or layer.output_size_per_partition != n
        or layer.input_size_per_partition != k
    ):
        return False
    w, s, g = layer.weight, layer.weight_scale, layer.weight_global_scale
    return (
        w.is_cuda and w.dtype == torch.int32 and tuple(w.shape) == (k // 16, 2 * n)
        and s.dtype == torch.float8_e4m3fn and tuple(s.shape) == (k // 16, n)
        and g.dtype == torch.float32 and g.numel() == 1
        and all(t.is_contiguous() for t in (w, s, g))
        and s.device == w.device and g.device == w.device
    )


def _expected_mask(activation, rows):
    # Compare BF16 bits, so signed zeros are zero and every NaN stays nonzero.
    bits = activation[:rows].view(torch.int16).to(torch.int32) & 0x7fff
    live = (bits.reshape(rows, 3712 // 16, 16) != 0).any(dim=0)
    shifts = torch.arange(16, dtype=torch.int32, device=activation.device)
    return (live.to(torch.int32) << shifts).sum(dim=1).to(torch.int32)


@torch.no_grad()
def prepare_shared_chain(layer):
    """Official remote load only, after every per-layer quantization finalize."""
    layer._shared_chain_ok = False
    layer._shared_chain_prefix_ok = False
    if not _ENABLED or _marlinx is None:
        return
    up, down = layer.up_proj, layer.down_proj
    if not (_weight_metadata_ok(up, 3712, 2688)
            and _weight_metadata_ok(down, 2688, 3712)):
        return
    zs = getattr(down, "_marlinx_zs", None)
    if zs is None or not zs.is_contiguous() or zs.device != up.weight.device:
        return
    if torch.cuda.get_device_capability(up.weight.device) != (12, 1):
        return
    if not hasattr(_marlinx, "init_shared_masks") or not _marlinx.init_shared_masks():
        return
    args = (up.weight, up.weight_scale, up.weight_global_scale,
            down.weight, down.weight_scale, down.weight_global_scale, zs)
    prefix_ok = _PREFIX
    # No prompt data, no local execution. Pure checkpoint-dependent admission.
    for m in range(1, 9):
        for sparse in (False, True):
            i = torch.arange(m * 2688, dtype=torch.int32, device=up.weight.device)
            value = (((i * 17 + 3) % 67).float() - 33) / 32
            if sparse:
                value = torch.where(i % 5 < 3, 0.0, value)
                value = torch.where(i % 11 == 0, -torch.zeros_like(value), value)
            x = value.to(torch.bfloat16).reshape(m, 2688).contiguous()
            original_up = torch.square(torch.relu(_stock_linear(x, up, 3712, 2688)))
            original_out = _stock_linear(original_up, down, 2688, 3712)
            activation, output, masks = _marlinx.shared_chain(x, *args)
            if not (
                torch.equal(activation.view(torch.int16), original_up.view(torch.int16))
                and torch.equal(output.view(torch.int16), original_out.view(torch.int16))
                and torch.equal(masks, _expected_mask(activation, m))
            ):
                logger.warning_once("shared-chain bit comparison declined; original MLP kept")
                return
            # Exercise the newly split ABI before enabling any runtime path.
            modes = (False, True) if _marlinx._pdl_active else (False,)
            for mode in modes:
                sa, sm = _marlinx.shared_up_mask(x, *args[:3], pdl=mode)
                so = _marlinx.shared_down_mask(sa, sm, *args[3:], pdl=mode)
                if not (
                    torch.equal(sa.view(torch.int16), original_up.view(torch.int16))
                    and torch.equal(so.view(torch.int16), original_out.view(torch.int16))
                    and torch.equal(sm, _expected_mask(sa, m))
                ):
                    logger.warning_once("split shared masks declined; original MLP kept")
                    return
            if m == 5 and prefix_ok:
                for rows in range(1, 6):
                    count = torch.tensor([rows], dtype=torch.int32, device=x.device)
                    pa, po, pm = _marlinx.shared_chain(x, *args, verify_rows=count)
                    if not (
                        torch.equal(pa.view(torch.int16), original_up.view(torch.int16))
                        and torch.equal(po[:rows].view(torch.int16),
                                        original_out[:rows].view(torch.int16))
                        and bool((po[rows:].view(torch.int16) == 0).all())
                        and torch.equal(pm, _expected_mask(pa, rows))
                    ):
                        prefix_ok = False
                        break
                    for mode in modes:
                        sa, sm = _marlinx.shared_up_mask(x, *args[:3], verify_rows=count, pdl=mode)
                        so = _marlinx.shared_down_mask(sa, sm, *args[3:], verify_rows=count, pdl=mode)
                        if not (
                            torch.equal(sa.view(torch.int16), original_up.view(torch.int16))
                            and torch.equal(so[:rows].view(torch.int16), original_out[:rows].view(torch.int16))
                            and bool((so[rows:].view(torch.int16) == 0).all())
                            and torch.equal(sm, _expected_mask(sa, rows))
                        ):
                            prefix_ok = False
                            break
                    if not prefix_ok:
                        break
    layer._shared_chain_ok = True
    layer._shared_chain_prefix_ok = prefix_ok
    logger.info_once("shared-chain admitted, verify-prefix=%s", prefix_ok)


def _shared_chain_impl(
    x: torch.Tensor, up_w: torch.Tensor, up_s: torch.Tensor, up_gs: torch.Tensor,
    down_w: torch.Tensor, down_s: torch.Tensor, down_gs: torch.Tensor,
    zs: torch.Tensor, prefix: bool,
) -> torch.Tensor:
    # Pointer alignment is checked inside the opaque op, never during tracing.
    if x.stride(-1) != 1 or x.stride(0) % 8 or x.data_ptr() % 16:
        x = x.contiguous()
        if x.data_ptr() % 16:
            x = x.clone()
    verify_rows = None
    if prefix and x.shape[0] == 5:
        smallk = sys.modules.get("smallk")
        if smallk is not None and hasattr(smallk, "live_verify_rows"):
            verify_rows = smallk.live_verify_rows(x.device)
            if verify_rows is not None and (
                verify_rows.dtype != torch.int32 or verify_rows.numel() != 1
                or verify_rows.device != x.device or not verify_rows.is_contiguous()
            ):
                verify_rows = None
    activation, output, masks = _marlinx.shared_chain(
        x, up_w, up_s, up_gs, down_w, down_s, down_gs, zs,
        verify_rows=verify_rows,
    )
    # Scratch belongs to this current-stream invocation, including graph replay.
    return output


def _shared_chain_fake(
    x: torch.Tensor, up_w: torch.Tensor, up_s: torch.Tensor, up_gs: torch.Tensor,
    down_w: torch.Tensor, down_s: torch.Tensor, down_gs: torch.Tensor,
    zs: torch.Tensor, prefix: bool,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], 2688))


direct_register_custom_op(
    op_name="nemotron_shared_chain", op_func=_shared_chain_impl,
    fake_impl=_shared_chain_fake, tags=(torch.Tag.needs_fixed_stride_order,),
)


def maybe_shared_chain(layer, x):
    if not _ENABLED or not getattr(layer, "_shared_chain_ok", False):
        return None
    if (
        x.ndim != 2 or x.shape[1] != 2688 or not 1 <= x.shape[0] <= 8
        or x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous()
    ):
        return None
    up, down = layer.up_proj, layer.down_proj
    return torch.ops.vllm.nemotron_shared_chain(
        x, up.weight, up.weight_scale, up.weight_global_scale,
        down.weight, down.weight_scale, down.weight_global_scale,
        down._marlinx_zs, layer._shared_chain_prefix_ok,
    )


def _live_rows(x, prefix):
    if not prefix or x.shape[0] != 5:
        return None
    smallk = sys.modules.get("smallk")
    if smallk is None or not hasattr(smallk, "live_verify_rows"):
        return None
    buf = smallk.live_verify_rows(x.device)
    if (buf is None or buf.dtype != torch.int32 or buf.numel() != 1
            or buf.device != x.device or not buf.is_contiguous()):
        return None
    return buf


def _split_up_impl(
    x: torch.Tensor, up_w: torch.Tensor, up_s: torch.Tensor, up_gs: torch.Tensor,
    prefix: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.stride(-1) != 1 or x.stride(0) % 8 or x.data_ptr() % 16:
        x = x.contiguous()
        if x.data_ptr() % 16:
            x = x.clone()
    return _marlinx.shared_up_mask(x, up_w, up_s, up_gs, verify_rows=_live_rows(x, prefix))


def _split_up_fake(
    x: torch.Tensor, up_w: torch.Tensor, up_s: torch.Tensor, up_gs: torch.Tensor,
    prefix: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return x.new_empty((x.shape[0], 3712)), x.new_empty((232,), dtype=torch.int32)


def _split_down_impl(
    h: torch.Tensor, masks: torch.Tensor, down_w: torch.Tensor, down_s: torch.Tensor,
    down_gs: torch.Tensor, zs: torch.Tensor, prefix: bool,
) -> torch.Tensor:
    return _marlinx.shared_down_mask(
        h, masks, down_w, down_s, down_gs, zs, verify_rows=_live_rows(h, prefix)
    )


def _split_down_fake(
    h: torch.Tensor, masks: torch.Tensor, down_w: torch.Tensor, down_s: torch.Tensor,
    down_gs: torch.Tensor, zs: torch.Tensor, prefix: bool,
) -> torch.Tensor:
    return h.new_empty((h.shape[0], 2688))


direct_register_custom_op(
    op_name="nemotron_shared_mask_up", op_func=_split_up_impl,
    fake_impl=_split_up_fake, tags=(torch.Tag.needs_fixed_stride_order,),
)
direct_register_custom_op(
    op_name="nemotron_shared_mask_down", op_func=_split_down_impl,
    fake_impl=_split_down_fake, tags=(torch.Tag.needs_fixed_stride_order,),
)


def maybe_shared_up_mask(layer, x):
    if not _ENABLED or not getattr(layer, "_shared_chain_ok", False):
        return None
    if (x.ndim != 2 or x.shape[1] != 2688 or not 1 <= x.shape[0] <= 8
            or x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous()):
        return None
    up = layer.up_proj
    return torch.ops.vllm.nemotron_shared_mask_up(
        x, up.weight, up.weight_scale, up.weight_global_scale, layer._shared_chain_prefix_ok
    )


def shared_down_from_mask(layer, h, masks):
    down = layer.down_proj
    return torch.ops.vllm.nemotron_shared_mask_down(
        h, masks, down.weight, down.weight_scale, down.weight_global_scale,
        down._marlinx_zs, layer._shared_chain_prefix_ok,
    )

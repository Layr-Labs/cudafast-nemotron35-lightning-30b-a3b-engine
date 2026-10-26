"""Private real-kernel admission for transparent routed-weight storage.

Source prototype: execution is permitted only by the official remote worker.
No format, arithmetic or kernel schedule changes. The caller already verified
the entire compressed allocation byte-for-byte and checked its driver backing.
The current complete archive score remains the end-to-end authority.
"""
import logging
import math
from statistics import median

import torch

_LOG = logging.getLogger(__name__)


def _backend(models, weight):
    """Resolve the actual target wrapper and reject registered storage aliases."""
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
        HummingIndexedExperts,
    )
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEKernelModularImpl,
    )
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    if not models or models[0] is None:
        return None
    matches = [layer for layer in models[0].modules()
               if isinstance(layer, RoutedExperts)
               and getattr(layer, "w13_weight", None) is weight]
    if len(matches) != 1:
        return None
    layer = matches[0]
    method = getattr(layer, "quant_method", None)
    kernel = getattr(method, "moe_kernel", None)
    if (type(method).__name__ != "ModelOptNvFp4FusedMoE"
            or not getattr(method, "use_a16", False)
            or kernel is None
            or not isinstance(kernel.impl, FusedMoEKernelModularImpl)
            or type(kernel.impl.fused_experts) is not HummingIndexedExperts
            or kernel.impl.prepare_finalize.supports_async()
            or layer.expert_map is not None
            or layer.apply_router_weight_on_input
            or layer.activation != MoEActivation.RELU2_NO_MUL):
        return None
    expert = kernel.impl.fused_experts
    if (not getattr(expert, "_fused_relu2", False)
            or getattr(expert, "_lora_context", None) is not None
            or layer.humming_metas["w13"].weight_name != "w13_weight"
            or weight.storage_offset() != 0 or not weight.is_contiguous()
            or weight.numel() * weight.element_size() != weight.untyped_storage().nbytes()
            or expert.global_num_experts != 128
            or expert.num_experts != 128):
        return None
    cfg = method.moe
    if cfg.in_dtype != torch.bfloat16:
        return None
    if any(getattr(cfg, key) != 1 for key in ("tp_size", "dp_size", "ep_size", "pcp_size")):
        return None
    meta = layer.humming_metas["w13"]
    if (meta.shape_n, meta.shape_k) != (1856, 2688):
        return None
    ptr = weight.untyped_storage().data_ptr()
    for model in models:
        if model is None:
            continue
        registered = list(model.parameters()) + list(model.buffers())
        for module in model.modules():
            registered.extend(value for value in vars(module).values()
                              if isinstance(value, torch.Tensor))
        for other in registered:
            if (other is not weight and other.is_cuda
                    and other.device == weight.device
                    and other.untyped_storage().data_ptr() == ptr):
                return None
    return layer, expert


def qualify(models, weight, compressed_view) -> bool:
    """Temporarily bind private old/new graphs, then restore before returning.

    Only immutable weight backing differs between the arms. Both use the exact
    production forward_relu2 dispatcher, including its selected NVSC/verify and
    prefill kernels. This is not a raw memcpy bandwidth surrogate.
    """
    if (torch.cuda.is_current_stream_capturing() or not weight.is_cuda
            or compressed_view.device != weight.device
            or compressed_view.dtype != weight.dtype
            or compressed_view.shape != weight.shape
            or compressed_view.stride() != weight.stride()):
        return False
    original = weight.data
    admitted = False
    try:
        resolved = _backend(models, weight)
        if resolved is None:
            return False
        layer, expert = resolved
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )
        from vllm.model_executor.layers.fused_moe.experts.humming_relu2_epilogue import (
            forward_relu2,
        )

        device = weight.device
        sweep = torch.empty(64 << 20, dtype=torch.uint8, device=device)
        stream = torch.cuda.Stream(device=device)
        for m in (1, 5, 496):
            for reuse in (False, True):
                count = m * 6
                block = next(config["block_shape"][0]
                             for low, high, config in expert.w13_tuning_config
                             if low < count <= high)
                row = torch.arange(m, dtype=torch.int32, device=device)[:, None]
                col = torch.arange(6, dtype=torch.int32, device=device)[None, :]
                ids = ((row * (0 if reuse else 17) + col * 23) % 128).contiguous()
                sorted_ids, expert_ids, ntp = moe_align_block_size(
                    topk_ids=ids, block_size=block, num_experts=128,
                    expert_map=None, ignore_invalid_experts=True,
                )
                at = torch.arange(m * 2688, dtype=torch.int32, device=device)
                x = ((at % 97 - 48).float() / 31).to(torch.bfloat16).view(m, 2688)
                outputs = [torch.empty((count, 1856), dtype=torch.bfloat16,
                                       device=device) for _ in range(2)]
                views = (original, compressed_view)
                graphs, events = [], []
                for which in range(2):
                    with torch.inference_mode():
                        weight.data = views[which]
                    assert getattr(layer, layer.humming_metas["w13"].weight_name).data_ptr() == views[which].data_ptr()

                    def run():
                        forward_relu2(
                            layer, x, outputs[which], expert.compute_config_str,
                            expert.w13_tuning_config_str, sorted_ids, expert_ids,
                            ntp, 6, count,
                        )

                    stream.wait_stream(torch.cuda.current_stream(device))
                    with torch.cuda.stream(stream):
                        for _ in range(6):
                            run()
                    torch.cuda.current_stream(device).wait_stream(stream)
                    torch.cuda.synchronize(device)
                    graph = torch.cuda.CUDAGraph()
                    begin = torch.cuda.Event(enable_timing=True, external=True)
                    end = torch.cuda.Event(enable_timing=True, external=True)
                    with torch.cuda.graph(graph, stream=stream):
                        begin.record()
                        run()
                        end.record()
                    graphs.append(graph)
                    events.append((begin, end))
                with torch.inference_mode():
                    weight.data = original
                graphs[0].replay()
                graphs[1].replay()
                torch.cuda.synchronize(device)
                if not torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16)):
                    return False
                for cold in (False, True):
                    samples = [[], []]
                    for repetition in range(19):
                        order = (0, 1) if repetition % 2 == 0 else (1, 0)
                        for which in order:
                            if cold:
                                sweep.zero_()
                            graphs[which].replay()
                            events[which][1].synchronize()
                            elapsed = events[which][0].elapsed_time(events[which][1])
                            if repetition >= 6:
                                samples[which].append(elapsed)
                    old_ms, new_ms = (median(values) for values in samples)
                    passed = (math.isfinite(old_ms) and math.isfinite(new_ms)
                              and old_ms > 0 and new_ms > 0 and new_ms <= old_ms * .98)
                    _LOG.info("CMP2 real_w13 layer=%s m=%d reuse=%s pressure=%s "
                              "ordinary_ms=%.6f generic_ms=%.6f admitted=%s",
                              getattr(layer, "layer_name", "target"), m, reuse,
                              cold, old_ms, new_ms, passed)
                    if not passed:
                        return False
                del graphs, events
        admitted = True
    except Exception as exc:
        _LOG.info("CMP2 real_w13 refused error=%s", type(exc).__name__)
    finally:
        with torch.inference_mode():
            weight.data = original
        torch.cuda.synchronize(weight.device)
    return admitted


def _down_layer(models, weight):
    """The one target layer whose M=5 verify pair repack owns ``weight``."""
    if not models or models[0] is None:
        return None
    matches = [layer for layer in models[0].modules()
               if getattr(getattr(layer, "zskip_pair_w2", None), "w", None) is weight]
    if len(matches) != 1:
        return None
    layer = matches[0]
    if (weight.storage_offset() != 0 or not weight.is_contiguous()
            or weight.numel() * weight.element_size() != weight.untyped_storage().nbytes()
            or getattr(layer, "zskip_pair_config", None) is None):
        return None
    return layer


def qualify_down(models, weight, compressed_view) -> bool:
    """Real-kernel admission of the verify routed-down pair repack's backing.

    Same contract as ``qualify``: the production zskip pair kernel and config
    of this layer, on both backings, bitwise-equal outputs and a median at
    least 2 % faster in both the warm and the L2-swept arm, else refused.
    """
    if (torch.cuda.is_current_stream_capturing() or not weight.is_cuda
            or compressed_view.device != weight.device
            or compressed_view.dtype != weight.dtype
            or compressed_view.shape != weight.shape
            or compressed_view.stride() != weight.stride()):
        return False
    original = weight.data
    admitted = False
    try:
        layer = _down_layer(models, weight)
        if layer is None:
            return False
        from vllm.model_executor.layers.fused_moe.experts import humming_zskip as hz
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )

        zs = hz._zskip
        if zs is None:
            return False
        pair, config = layer.zskip_pair_w2, layer.zskip_pair_config
        device = weight.device
        sweep = torch.empty(64 << 20, dtype=torch.uint8, device=device)
        stream = torch.cuda.Stream(device=device)
        m, top_k = 5, hz.TOP_K
        count = m * top_k
        for reuse in (False, True):
            row = torch.arange(m, dtype=torch.int32, device=device)[:, None]
            col = torch.arange(top_k, dtype=torch.int32, device=device)[None, :]
            ids = ((row * (0 if reuse else 17) + col * 23) % hz.NUM_EXPERTS).contiguous()
            sorted_ids, expert_ids, ntp = moe_align_block_size(
                topk_ids=ids, block_size=16, num_experts=hz.NUM_EXPERTS,
                expert_map=None, ignore_invalid_experts=True,
            )
            at = torch.arange(count * hz.INTER, dtype=torch.int32, device=device)
            act = torch.where(at % 3 == 0, torch.zeros((), device=device),
                              ((at % 89).float() / 37).square())
            act = act.to(torch.bfloat16).view(count, hz.INTER).contiguous()
            outputs = [torch.empty((count, hz.HIDDEN), dtype=torch.bfloat16,
                                   device=device) for _ in range(2)]
            views = (original, compressed_view)
            graphs, events = [], []
            for which in range(2):
                with torch.inference_mode():
                    weight.data = views[which]
                assert pair.w.data_ptr() == views[which].data_ptr()

                def run():
                    zs.down(act, sorted_ids, expert_ids, ntp, pair, outputs[which],
                            mode=zs.SKIP, config=config, max_blocks=count)

                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for _ in range(6):
                        run()
                torch.cuda.current_stream(device).wait_stream(stream)
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                begin = torch.cuda.Event(enable_timing=True, external=True)
                end = torch.cuda.Event(enable_timing=True, external=True)
                with torch.cuda.graph(graph, stream=stream):
                    begin.record()
                    run()
                    end.record()
                graphs.append(graph)
                events.append((begin, end))
            with torch.inference_mode():
                weight.data = original
            graphs[0].replay()
            graphs[1].replay()
            torch.cuda.synchronize(device)
            if not torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16)):
                return False
            for cold in (False, True):
                samples = [[], []]
                for repetition in range(19):
                    order = (0, 1) if repetition % 2 == 0 else (1, 0)
                    for which in order:
                        if cold:
                            sweep.zero_()
                        graphs[which].replay()
                        events[which][1].synchronize()
                        elapsed = events[which][0].elapsed_time(events[which][1])
                        if repetition >= 6:
                            samples[which].append(elapsed)
                old_ms, new_ms = (median(values) for values in samples)
                passed = (math.isfinite(old_ms) and math.isfinite(new_ms)
                          and old_ms > 0 and new_ms > 0 and new_ms <= old_ms * .98)
                _LOG.info("CMP2 real_w2_pair layer=%s reuse=%s pressure=%s "
                          "ordinary_ms=%.6f generic_ms=%.6f admitted=%s",
                          getattr(layer, "layer_name", "target"), reuse, cold,
                          old_ms, new_ms, passed)
                if not passed:
                    return False
            del graphs, events
        admitted = True
    except Exception as exc:
        _LOG.info("CMP2 real_w2_pair refused error=%s", type(exc).__name__)
    finally:
        with torch.inference_mode():
            weight.data = original
        torch.cuda.synchronize(weight.device)
    return admitted

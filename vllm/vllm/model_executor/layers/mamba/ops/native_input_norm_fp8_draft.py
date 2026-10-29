# SPDX-License-Identifier: Apache-2.0
"""Experimental guarded native-semantics Mamba input producer.

The new producer and qualification are unvalidated. Worker admission is
required before the model may select a shape, with an original-path fallback.
No local candidate execution or equality claim has been made; the official
remote worker must validate selected-provider semantics and rounding.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def _native_add_norm_static_fp8_sketch(
    X, RESIDUAL, WEIGHT, SCALE, OUT_RESIDUAL, OUT_FP8,
    SX: tl.constexpr, SR: tl.constexpr, SO: tl.constexpr,
    H: tl.constexpr, BLOCK: tl.constexpr, EPS: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < H
    x = tl.load(X + row * SX + col, mask=mask, other=0).to(tl.float32)
    residual = tl.load(RESIDUAL + row * SR + col, mask=mask, other=0).to(tl.float32)
    value = x + residual
    tl.store(OUT_RESIDUAL + row * SO + col, value.to(tl.bfloat16), mask=mask)
    variance = tl.sum(value * value, axis=0) / H
    norm_float = value * tl.rsqrt(variance + EPS)
    # Native vLLM op casts before multiplying by the BF16 weight.
    norm_bf16 = norm_float.to(tl.bfloat16).to(tl.float32)
    w = tl.load(WEIGHT + col, mask=mask, other=0).to(tl.float32)
    norm_weighted_bf16 = (norm_bf16 * w).to(tl.bfloat16).to(tl.float32)
    scale = tl.load(SCALE).to(tl.float32)
    q = norm_weighted_bf16 * tl.div_rn(1.0, scale)
    # C fminf(NaN, max) returns the non-NaN max bound.
    q = tl.where(q != q, 448.0, q)
    q = tl.where(q > 448.0, 448.0, q)
    q = tl.where(q < -448.0, -448.0, q)
    tl.store(OUT_FP8 + row * SO + col, q.to(OUT_FP8.dtype.element_ty), mask=mask)


# A real remote-only call must choose BLOCK = next_power_of_2(H), supply
# independent contiguous BF16 residual and FP8 output buffers, use
# enable_fp_fusion=False, prove actual native lowering/shape/scale semantics,
# and qualify exact outputs including aliases and CUDA graph replay.
# tl.sum/rsqrt and the native compiler may differ in reduction order, rounding
# or NaN treatment. The model path is guarded by worker admission.


def execute_native_sketch(x, residual, weight, scale, eps):
    """Remote-worker helper; prerequisites must be established by caller."""
    import torch

    hidden = x.shape[1]
    output_residual = torch.empty_like(x)
    output_fp8 = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    _native_add_norm_static_fp8_sketch[(x.shape[0],)](
        x, residual, weight, scale, output_residual, output_fp8,
        SX=x.stride(0), SR=residual.stride(0), SO=output_residual.stride(0),
        H=hidden, BLOCK=triton.next_power_of_2(hidden), EPS=eps,
        num_warps=4, enable_fp_fusion=False,
    )
    return output_fp8, output_residual


def compare_native_owner_sketch(norm, projection):
    """Remote-only bitwise comparison used by preliminary admission.

    Compare against the native provider and the owner's real quantizer, using
    untouched independent x/residual and several shapes. This does not prove
    what the compiled baseline selects, whole-projection effects, alias or
    concurrent graph safety, actual scaled model input distributions, or speed.
    """
    import torch
    from vllm import ir
    from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8StaticTensorSym

    kernel = getattr(getattr(projection, "quant_method", None), "fp8_linear", None)
    if (kernel is None or type(kernel).__name__ != "CutlassFP8ScaledMMLinearKernel"
            or kernel.input_quant_key() != kFp8StaticTensorSym
            or kernel.quant_fp8.num_token_padding is not None
            or norm.variance_size_override is not None
            or not norm.has_weight or not norm.pass_weight_add):
        return False
    names = kernel.layer_param_names
    scale = getattr(projection, names[2], None)
    weight = norm.weight
    if (scale is None or getattr(projection, names[3], None) is not None
            or scale.dtype != torch.float32 or scale.numel() != 1
            or scale.device != weight.device or not scale.is_contiguous()
            or weight.dtype != torch.bfloat16 or not weight.is_cuda
            or not weight.is_contiguous()):
        return False
    device = weight.device
    generator = torch.Generator(device=device).manual_seed(67031)
    native = ir.ops.fused_add_rms_norm.impls["native"].impl_fn
    for rows in (1, 5, 6, 8):
        for magnitude in (0.001, 1.0, 30.0):
            shape = (rows, weight.numel())
            x = (torch.randn(shape, device=device, generator=generator) * magnitude).to(torch.bfloat16)
            residual = (torch.randn(shape, device=device, generator=generator) * magnitude).to(torch.bfloat16)
            x_ref, r_ref = x.clone(), residual.clone()
            norm_ref, out_residual_ref = native(x_ref, r_ref, weight, norm.variance_epsilon)
            quant_ref, _ = kernel.quant_fp8(norm_ref, scale, None)
            x_new, r_new = x.clone(), residual.clone()
            quant_new, out_residual_new = execute_native_sketch(
                x_new, r_new, weight, scale, norm.variance_epsilon
            )
            # The compilation pass may replace the native pair with the
            # pinned fused provider. Require this owner and input to agree
            # with both possible producers before considering admission.
            x_fused, r_fused = x.clone(), residual.clone()
            q_fused = torch.empty_like(quant_new)
            torch.ops._C.fused_add_rms_norm_static_fp8_quant(
                q_fused, x_fused, r_fused, weight, scale, norm.variance_epsilon
            )
            if not torch.equal(quant_ref.view(torch.uint8), quant_new.view(torch.uint8)):
                return False
            if not torch.equal(out_residual_ref.view(torch.int16), out_residual_new.view(torch.int16)):
                return False
            if not torch.equal(q_fused.view(torch.uint8), quant_new.view(torch.uint8)):
                return False
            if not torch.equal(r_fused.view(torch.int16), out_residual_new.view(torch.int16)):
                return False
            if not torch.equal(x_ref.view(torch.int16), x_new.view(torch.int16)):
                return False
            if not torch.equal(r_ref.view(torch.int16), r_new.view(torch.int16)):
                return False
    return True


def compare_native_graph_sketch(norm, projection, rows):
    """Remote-only graph/stream check used by preliminary admission.

    Capture two independent graphs with separate output allocations. Change
    the source tensors and replay on distinct streams; check exact byte results
    and that neither input owner is mutated. This tests only the exact owner,
    shapes and inputs supplied, not the compiler's production graph/provider.
    """
    import torch
    from vllm import ir

    kernel = projection.quant_method.fp8_linear
    scale = getattr(projection, kernel.layer_param_names[2])
    weight, device = norm.weight, norm.weight.device
    shape = (rows, weight.numel())
    rng = torch.Generator(device=device).manual_seed(91823)
    source_x = torch.randn(shape, device=device, generator=rng, dtype=torch.bfloat16)
    source_r = torch.randn(shape, device=device, generator=rng, dtype=torch.bfloat16)
    old_x, old_r = torch.empty_like(source_x), torch.empty_like(source_r)
    new_x, new_r = torch.empty_like(source_x), torch.empty_like(source_r)
    native = ir.ops.fused_add_rms_norm.impls["native"].impl_fn

    def old_call():
        old_x.copy_(source_x)
        old_r.copy_(source_r)
        normed, out_residual = native(old_x, old_r, weight, norm.variance_epsilon)
        quantized, _ = kernel.quant_fp8(normed, scale, None)
        return quantized, out_residual

    def new_call():
        new_x.copy_(source_x)
        new_r.copy_(source_r)
        return execute_native_sketch(new_x, new_r, weight, scale, norm.variance_epsilon)

    old_call()
    new_call()
    torch.cuda.synchronize(device)
    old_graph, new_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(old_graph, pool=torch.cuda.graph_pool_handle()):
        old_q, old_out_r = old_call()
    with torch.cuda.graph(new_graph, pool=torch.cuda.graph_pool_handle()):
        new_q, new_out_r = new_call()
    old_stream, new_stream = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    current = torch.cuda.current_stream(device)
    for magnitude in (0.001, 1.0, 30.0):
        source_x.normal_(generator=rng).mul_(magnitude)
        source_r.normal_(generator=rng).mul_(magnitude)
        old_stream.wait_stream(current)
        new_stream.wait_stream(current)
        with torch.cuda.stream(old_stream):
            old_graph.replay()
        with torch.cuda.stream(new_stream):
            new_graph.replay()
        current.wait_stream(old_stream)
        current.wait_stream(new_stream)
        if not torch.equal(old_q.view(torch.uint8), new_q.view(torch.uint8)):
            return False
        if not torch.equal(old_out_r.view(torch.int16), new_out_r.view(torch.int16)):
            return False
        if not torch.equal(old_x.view(torch.int16), new_x.view(torch.int16)):
            return False
        if not torch.equal(old_r.view(torch.int16), new_r.view(torch.int16)):
            return False
    return True


def compare_native_projection_sketch(norm, projection, rows, pressure=False):
    """Remote-only projection timing used by admission; not a score."""
    import statistics
    import torch
    from vllm import ir
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8StaticTensorSym

    kernel = projection.quant_method.fp8_linear
    scale = getattr(projection, kernel.layer_param_names[2])
    weight, device = norm.weight, norm.weight.device
    shape = (rows, weight.numel())
    rng = torch.Generator(device=device).manual_seed(39817)
    source_x = torch.randn(shape, device=device, generator=rng, dtype=torch.bfloat16)
    source_r = torch.randn(shape, device=device, generator=rng, dtype=torch.bfloat16)
    old_x, old_r = torch.empty_like(source_x), torch.empty_like(source_r)
    new_x, new_r = torch.empty_like(source_x), torch.empty_like(source_r)
    native = ir.ops.fused_add_rms_norm.impls["native"].impl_fn

    def old_call():
        old_x.copy_(source_x)
        old_r.copy_(source_r)
        normed, out_residual = native(old_x, old_r, weight, norm.variance_epsilon)
        out, _ = projection(normed)
        return out, out_residual

    def new_call():
        new_x.copy_(source_x)
        new_r.copy_(source_r)
        q, out_residual = execute_native_sketch(new_x, new_r, weight, scale, norm.variance_epsilon)
        activation = QuantizedActivation(data=q, scale=scale, orig_dtype=new_x.dtype,
                                         orig_shape=new_x.shape, quant_key=kFp8StaticTensorSym)
        out, _ = projection(activation)
        return out, out_residual

    old_call()
    new_call()
    torch.cuda.synchronize(device)
    old_graph, new_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(old_graph, pool=torch.cuda.graph_pool_handle()):
        old_out, old_residual = old_call()
    with torch.cuda.graph(new_graph, pool=torch.cuda.graph_pool_handle()):
        new_out, new_residual = new_call()
    for magnitude in (0.001, 1.0, 30.0):
        source_x.normal_(generator=rng).mul_(magnitude)
        source_r.normal_(generator=rng).mul_(magnitude)
        old_graph.replay()
        new_graph.replay()
        if not torch.equal(old_out.view(torch.int16), new_out.view(torch.int16)):
            return False, None
        if not torch.equal(old_residual.view(torch.int16), new_residual.view(torch.int16)):
            return False, None
    for _ in range(4):
        old_graph.replay()
        new_graph.replay()
    old_times, new_times = [], []
    eviction = torch.zeros(16 * 1024 * 1024, dtype=torch.float32, device=device) if pressure else None
    for trial in range(16):
        order = ((old_graph, old_times), (new_graph, new_times))
        if trial & 1:
            order = tuple(reversed(order))
        for graph, times in order:
            if eviction is not None:
                eviction.add_(1.0)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(begin.elapsed_time(end))
    return True, {"old_median_ms": statistics.median(old_times),
                  "new_median_ms": statistics.median(new_times),
                  "new_faster_pairs": sum(n < o for o, n in zip(old_times, new_times))}


def qualify_native_owner_sketch(norm, projection, max_seconds=8.0):
    """Remote-only preliminary owner admission before compilation.

    This compares an explicit native eager reference, not the compiled
    production graph. Integration must establish selected-provider semantics.
    """
    import time
    import torch

    started = time.monotonic()
    selected = set()
    try:
        if not compare_native_owner_sketch(norm, projection):
            return selected
        if not compare_native_edges_sketch(norm, projection):
            return selected
        for rows in (1, 5, 6, 8):
            if time.monotonic() - started >= max_seconds:
                break
            if not compare_native_graph_sketch(norm, projection, rows):
                continue
            measurements = []
            for pressure in (False, True):
                if time.monotonic() - started >= max_seconds:
                    break
                exact, timings = compare_native_projection_sketch(norm, projection, rows, pressure)
                if not exact or timings is None:
                    break
                measurements.append(timings)
            if len(measurements) != 2:
                continue
            if all(m["new_median_ms"] < 0.98 * m["old_median_ms"]
                   and m["new_faster_pairs"] >= 12 for m in measurements):
                selected.add(rows)
    except Exception:
        # A poisoned CUDA context must not masquerade as a safe fallback.
        torch.cuda.synchronize(norm.weight.device)
        return set()
    return selected


def compare_native_edges_sketch(norm, projection):
    """Remote-only adversarial-value comparison against the real quantizer."""
    import torch
    from vllm import ir

    kernel = projection.quant_method.fp8_linear
    scale = getattr(projection, kernel.layer_param_names[2])
    weight, device = norm.weight, norm.weight.device
    hidden = weight.numel()
    rows = 6
    # Inject both signs of zero, BF16 normals/subnormals, infinities, NaNs
    # and large cancellation pairs; this still does not exhaust FP8 ties.
    values = torch.tensor([0.0, -0.0, 1.0, -1.0, 1e-38, -1e-38,
                           3.38953139e38, -3.38953139e38,
                           float("inf"), -float("inf"), float("nan")],
                          device=device, dtype=torch.bfloat16)
    indices = torch.arange(rows * hidden, device=device) % values.numel()
    x = values[indices].reshape(rows, hidden).contiguous()
    residual = values[(indices * 7 + 3) % values.numel()].reshape(rows, hidden).contiguous()
    x_ref, r_ref, x_new, r_new = x.clone(), residual.clone(), x.clone(), residual.clone()
    native = ir.ops.fused_add_rms_norm.impls["native"].impl_fn
    norm_ref, out_r_ref = native(x_ref, r_ref, weight, norm.variance_epsilon)
    q_ref, _ = kernel.quant_fp8(norm_ref, scale, None)
    q_new, out_r_new = execute_native_sketch(x_new, r_new, weight, scale, norm.variance_epsilon)
    x_fused, r_fused = x.clone(), residual.clone()
    q_fused = torch.empty_like(q_new)
    torch.ops._C.fused_add_rms_norm_static_fp8_quant(
        q_fused, x_fused, r_fused, weight, scale, norm.variance_epsilon
    )
    return (torch.equal(q_ref.view(torch.uint8), q_new.view(torch.uint8))
            and torch.equal(out_r_ref.view(torch.int16), out_r_new.view(torch.int16))
            and torch.equal(q_fused.view(torch.uint8), q_new.view(torch.uint8))
            and torch.equal(r_fused.view(torch.int16), out_r_new.view(torch.int16))
            and torch.equal(x_ref.view(torch.int16), x_new.view(torch.int16))
            and torch.equal(r_ref.view(torch.int16), r_new.view(torch.int16)))


def qualify_model_native_sketch(model, total_seconds=12.0):
    """Worker startup helper returning qualifying owners before publication.

    The IR priority check is a conservative screen, not evidence that
    compilation or norm/quant pattern fusion retains native behavior. This
    return value is published by worker setup after checks.
    """
    import time
    from vllm import ir

    if ir.ops.fused_add_rms_norm.get_priority() != ["native"]:
        return {}
    started = time.monotonic()
    qualified = {}
    for layer in model.modules():
        if time.monotonic() - started >= total_seconds:
            break
        if type(layer).__name__ != "NemotronHMambaDecoderLayer":
            continue
        norm = getattr(layer, "norm", None)
        projection = getattr(getattr(layer, "mixer", None), "in_proj", None)
        if norm is None or projection is None:
            continue
        rows = qualify_native_owner_sketch(norm, projection,
                                            max_seconds=max(0.0, total_seconds - (time.monotonic() - started)))
        if rows:
            qualified[layer] = frozenset(rows)
    return qualified


def native_op_registration_sketch():
    """Register the guarded producer before worker qualification.

    Accurate no-mutation schema matches the explicit native eager provider
    only; optimizing compiled maybe_inplace/fusion may have different alias
    behavior. Worker setup invokes registration before qualification.
    """
    import torch
    from vllm.utils.torch_utils import direct_register_custom_op

    def impl(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
             scale: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        return execute_native_sketch(x, residual, weight, scale, eps)

    def fake(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
             scale: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device),
                torch.empty_like(x))

    direct_register_custom_op(
        op_name="mamba_input_native_norm_static_fp8_draft",
        op_func=impl, mutates_args=[], fake_impl=fake,
        tags=(torch.Tag.needs_fixed_stride_order,),
    )


def prepare_model_native_draft(model):
    """Optional worker-only qualification before tracing; publish last."""
    import logging
    import os
    import torch

    if os.environ.get("VLLM_MAMBA_NATIVE_INPUT_FP8", "1") == "0":
        return
    native_op_registration_sketch()
    with torch.inference_mode():
        admitted = qualify_model_native_sketch(model)
    for layer, rows in admitted.items():
        layer._native_input_fp8_rows = rows
    logging.getLogger(__name__).info("native input FP8 draft owner shapes: %s",
                                     [tuple(sorted(v)) for v in admitted.values()])


def maybe_native_input_draft(layer, x, residual):
    """Guarded prequantized producer for an already-qualified native owner."""
    import os
    import torch
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8StaticTensorSym

    if os.environ.get("VLLM_MAMBA_NATIVE_INPUT_FP8", "1") == "0":
        return None
    if residual is None or x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16:
        return None
    if (x.shape[0] not in getattr(layer, "_native_input_fp8_rows", ())
            or not x.is_contiguous() or not residual.is_contiguous()
            or residual.shape != x.shape or residual.dtype != x.dtype
            or residual.device != x.device):
        return None
    norm, projection = layer.norm, layer.mixer.in_proj
    kernel = getattr(getattr(projection, "quant_method", None), "fp8_linear", None)
    if (kernel is None or type(kernel).__name__ != "CutlassFP8ScaledMMLinearKernel"
            or kernel.input_quant_key() != kFp8StaticTensorSym
            or kernel.quant_fp8.num_token_padding is not None
            or kernel.fp8_dtype != torch.float8_e4m3fn
            or getattr(projection, "bias", None) is not None
            or norm.variance_size_override is not None
            or not norm.has_weight or not norm.pass_weight_add
            or norm.weight.dtype != x.dtype or norm.weight.device != x.device
            or not norm.weight.is_contiguous() or norm.weight.numel() != x.shape[1]):
        return None
    scale = getattr(projection, kernel.layer_param_names[2], None)
    if (scale is None or getattr(projection, kernel.layer_param_names[3], None) is not None
            or scale.dtype != torch.float32 or scale.numel() != 1
            or not scale.is_contiguous() or scale.device != x.device):
        return None
    q, next_residual = torch.ops.vllm.mamba_input_native_norm_static_fp8_draft(
        x, residual, norm.weight, scale, norm.variance_epsilon
    )
    return (QuantizedActivation(data=q, scale=scale, orig_dtype=x.dtype,
                                orig_shape=x.shape, quant_key=kFp8StaticTensorSym),
            next_residual)

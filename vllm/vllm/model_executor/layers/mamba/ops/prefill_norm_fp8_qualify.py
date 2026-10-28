# SPDX-License-Identifier: Apache-2.0
"""Optional full-chain qualification for the 496-row prefill producer.

Qualification happens after weights and cache are bound, before model capture.
Failure or lack of a measured full-chain win keeps the original path.
"""
import os
import statistics

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8StaticTensorSym

logger = init_logger(__name__)
_ROWS = 496
_WIDTH = 4096


def _same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


@torch.no_grad()
def prepare(layer):
    norm, projection = layer.norm, layer.out_proj
    norm._prefill_fp8_qualified = False
    if os.environ.get("VLLM_MAMBA_PREFILL_NORM_FP8", "1") == "0":
        return
    kernel = getattr(getattr(projection, "quant_method", None), "fp8_linear", None)
    if (
        kernel is None
        or type(kernel).__name__ != "CutlassFP8ScaledMMLinearKernel"
        or kernel.input_quant_key() != kFp8StaticTensorSym
        or kernel.quant_fp8.num_token_padding is not None
        or kernel.fp8_dtype != torch.float8_e4m3fn
        or norm.tp_size != 1
        or projection.tp_size != 1
        or not projection.input_is_parallel
        or getattr(projection, "bias", None) is not None
        or not norm.use_rms_norm
        or norm.n_groups != 8
        or norm.group_size != 512
        or norm.weight.dtype != torch.bfloat16
        or norm.weight.shape != (_WIDTH,)
        or not norm.weight.is_cuda
        or not norm.weight.is_contiguous()
        or torch.cuda.is_current_stream_capturing()
    ):
        return
    scale = getattr(projection, kernel.layer_param_names[2], None)
    upper = getattr(projection, kernel.layer_param_names[3], None)
    if (
        scale is None
        or upper is not None
        or scale.dtype != torch.float32
        or scale.numel() != 1
        or not scale.is_contiguous()
        or scale.device != norm.weight.device
    ):
        return
    gate_stride = int(layer.in_proj.output_size_per_partition)
    if gate_stride < _WIDTH:
        return
    device = norm.weight.device
    gen = torch.Generator(device=device).manual_seed(2026102807)
    x = torch.empty((_ROWS, _WIDTH), device=device, dtype=torch.bfloat16)
    gate_storage = torch.empty((_ROWS, gate_stride), device=device, dtype=x.dtype)
    gate = gate_storage[:, :_WIDTH]

    def old(a, b):
        return projection(norm(a, b))[0]

    def old_quant(a, b):
        return kernel.quant_fp8(norm(a, b), scale, None)[0]

    def new(a, b):
        q = torch.ops.vllm.mamba_gated_norm_static_fp8(
            a, b, norm.weight, scale, norm.variance_epsilon
        )
        qa = QuantizedActivation(q, scale, a.dtype, a.shape, kFp8StaticTensorSym)
        return projection(qa)[0], q

    # Compile the actual reference norm and projection together.  The separate
    # quant comparison is supplemental; only whole-projection graph timings
    # decide whether to admit this path.
    old_compiled = torch.compile(old, fullgraph=True, dynamic=True,
                                 options={"triton.cudagraphs": False})
    oldq_compiled = torch.compile(old_quant, fullgraph=True, dynamic=True,
                                  options={"triton.cudagraphs": False})
    new_compiled = torch.compile(new, fullgraph=True, dynamic=True,
                                 options={"triton.cudagraphs": False})

    def change(case):
        mult = (1.0, 0.001, 16.0, 1.0, 1.0)[case]
        x.copy_((torch.randn(x.shape, device=device, generator=gen) * mult).to(x.dtype))
        gate.copy_((torch.randn(gate.shape, device=device, generator=gen) * mult).to(gate.dtype))
        if case == 3:
            x[:, ::2] = 0.0
            gate[:, ::2] = -0.0
            x.view(torch.int16)[:, 1:5] = torch.tensor(
                [1, -32767, 128, -32640], dtype=torch.int16, device=device
            )
        if case == 4:
            x[0, :4] = torch.tensor(
                [float("inf"), -float("inf"), float("nan"), -0.0],
                dtype=x.dtype, device=device,
            )

    change(0)
    old_compiled(x, gate)
    oldq_compiled(x, gate)
    new_compiled(x, gate)
    torch.cuda.synchronize(device)
    ref_graph, quant_graph, new_graph = (torch.cuda.CUDAGraph() for _ in range(3))
    with torch.cuda.graph(ref_graph, pool=torch.cuda.graph_pool_handle()):
        ref_output = old_compiled(x, gate)
    with torch.cuda.graph(quant_graph, pool=torch.cuda.graph_pool_handle()):
        ref_quant = oldq_compiled(x, gate)
    with torch.cuda.graph(new_graph, pool=torch.cuda.graph_pool_handle()):
        new_output, new_quant = new_compiled(x, gate)

    gamma_before, scale_before = norm.weight.clone(), scale.clone()
    for case in range(5):
        change(case)
        x_before, gate_before = x.clone(), gate.clone()
        for _ in range(2):
            ref_graph.replay()
            quant_graph.replay()
            new_graph.replay()
            if not (_same(ref_output, new_output) and _same(ref_quant, new_quant)
                    and _same(x, x_before) and _same(gate, gate_before)
                    and _same(norm.weight, gamma_before) and _same(scale, scale_before)):
                raise RuntimeError("prefill norm+quant full graph comparison failed")

    change(0)
    pressure = torch.empty(64 * (1 << 20), dtype=torch.uint8, device=device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ratios = []
    for cold in (False, True):
        measurements = [[], []]
        for trial in range(5):
            for arm in ((0, 1) if trial % 2 == 0 else (1, 0)):
                graph = (ref_graph, new_graph)[arm]
                graph.replay()
                if cold:
                    pressure.fill_(trial)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                measurements[arm].append(start.elapsed_time(end))
        a, b = (statistics.median(t) for t in measurements)
        if a <= 0 or b <= 0:
            raise RuntimeError("prefill norm+quant timing was nonpositive")
        ratios.append(b / a)
    if max(ratios) >= 0.99:
        logger.info("Mamba prefill norm+FP8 refused full-chain ratios=%s", ratios)
        return
    norm._prefill_fp8_gate_stride = gate_stride
    norm._prefill_fp8_projection_id = id(projection)
    norm._prefill_fp8_bound_params = (norm.weight, scale)
    norm._prefill_fp8_qualified = True
    logger.info("Mamba prefill norm+FP8 qualified full-chain ratios=%s", ratios)

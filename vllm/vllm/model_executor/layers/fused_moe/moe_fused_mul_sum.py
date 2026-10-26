# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch._subclasses.fake_tensor import FakeTensor

import os

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

_PDL: bool | None = None


def _pdl_enabled() -> bool:
    """LOCAL PATCH (cudafast-nemotron35): launch the weighted top-k sum with PDL on
    sm_90+, so its successor (the next layer's fused residual/RMSNorm, PDL-launched by
    inductor) overlaps its tail. VLLM_MOE_MUL_SUM_PDL=0 disables."""
    global _PDL
    if _PDL is None:
        _PDL = os.environ.get(
            "VLLM_MOE_MUL_SUM_PDL", "1"
        ) != "0" and current_platform.is_arch_support_pdl()
    return _PDL


@triton.jit
def moe_fused_mul_sum_kernel(
    inputs_ptr,
    topk_weights_ptr,
    outputs_ptr,
    top_ids_ptr,
    expert_map_ptr,
    num_tokens,
    stride_m,
    has_expert_map: tl.constexpr,
    top_k: tl.constexpr,
    size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    launch_pdl: tl.constexpr = False,
    shared_ptr=None,
    with_shared: tl.constexpr = False,
    routed_scale: tl.constexpr = 1.0,
):
    # LOCAL PATCH (cudafast-nemotron35): programmatic dependent launch. Wait for the
    # producer grid before any load; signal dependents once every load is issued.
    # Same loads, same fp32 accumulation order, same store.
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    pid_k = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    m_mask = offs_m < num_tokens
    k_mask = offs_k < size
    mask = m_mask[:, None] & k_mask[None, :]

    a_base = inputs_ptr + (offs_m * stride_m)[:, None] + offs_k[None, :]
    b_base = topk_weights_ptr + offs_m * top_k

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n in tl.static_range(top_k):
        b_val = tl.load(b_base + n, mask=m_mask, other=0.0).to(tl.float32)
        if has_expert_map:
            id_val = tl.load(top_ids_ptr + offs_m * top_k + n, mask=m_mask, other=0)
            expert_mask = tl.load(expert_map_ptr + id_val) >= 0
            a_vec = tl.load(
                a_base + n * size,
                mask=mask & expert_mask[:, None],
                other=0.0,
            ).to(tl.float32)
        else:
            a_vec = tl.load(
                a_base + n * size,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        acc += a_vec * b_val[:, None]

    if with_shared:
        # Retain the original materialized BF16 rounding boundary. Shared is
        # the left operand, as in MoERunner's shared_output + fused_output.
        routed = acc.to(outputs_ptr.dtype.element_ty).to(tl.float32)
        if routed_scale != 1.0:
            routed = tl.inline_asm_elementwise(
                "mul.rn.f32 $0, $1, $2;", constraints="=f,f,f",
                args=[routed, tl.full((), routed_scale, tl.float32)],
                dtype=tl.float32, is_pure=True, pack=1,
            ).to(outputs_ptr.dtype.element_ty).to(tl.float32)
        shared = tl.load(
            shared_ptr + (offs_m * size)[:, None] + offs_k[None, :],
            mask=mask, other=0.0,
        ).to(tl.float32)
        acc = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;", constraints="=f,f,f",
            args=[shared, routed], dtype=tl.float32, is_pure=True, pack=1,
        )
    # All inputs, including shared output, precede the PDL signal.
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    out_ptrs = outputs_ptr + (offs_m * size)[:, None] + offs_k[None, :]
    tl.store(
        out_ptrs,
        acc.to(outputs_ptr.dtype.element_ty),
        mask=mask,
    )


def _heuristic_config(
    num_tokens: int,
    top_k: int,
    size: int,
    element_size: int,
):
    is_fp32 = element_size > 2
    is_sm90_plus = current_platform.has_device_capability(90)
    is_sm80_before = not current_platform.has_device_capability(80)

    if current_platform.has_device_capability(90):
        # SM90/SM100+: prefer small tiles + many CTAs.
        if is_fp32:
            BLOCK_M = 1 if num_tokens <= 4 else 2
        else:
            if num_tokens <= 4:
                BLOCK_M = 1
            elif num_tokens <= 128:
                BLOCK_M = 2
            else:
                BLOCK_M = 4
    elif is_fp32:
        if num_tokens <= 4:
            BLOCK_M = 1
        elif num_tokens <= 32:
            BLOCK_M = 2
        elif num_tokens <= 128:
            BLOCK_M = 4
        else:
            BLOCK_M = 4
    else:
        if num_tokens <= 4:
            BLOCK_M = 1
        elif num_tokens <= 32:
            BLOCK_M = 2
        elif num_tokens <= 128:
            BLOCK_M = 4
        elif num_tokens <= 1024:
            BLOCK_M = 16
        else:
            BLOCK_M = 8

    if is_fp32:
        max_block_k = 256
    elif is_sm80_before or is_sm90_plus:
        max_block_k = 512
    else:
        max_block_k = 1024
    BLOCK_K = min(triton.next_power_of_2(size), max_block_k)
    BLOCK_K = max(BLOCK_K, 256)

    total = BLOCK_M * BLOCK_K
    if is_fp32:
        num_warps = max(8, min(16, total // 64))
    else:
        num_warps = max(4, min(16, total // 256))

    if is_sm80_before:
        num_warps = min(num_warps, 8)
        num_stages = 2
    elif is_sm90_plus:
        num_warps = min(num_warps, 8)
        num_stages = 4 if total <= 2048 else 2
    else:
        num_stages = 4 if total <= 2048 else 2

    return BLOCK_M, BLOCK_K, num_warps, num_stages


def moe_fused_mul_sum(
    inputs: torch.Tensor,
    topk_weights: torch.Tensor,
    outputs: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fused kernel for MoE (Mixture of Experts) to perform weighted summation
    of expert outputs.

    Args:
        inputs: The output from experts.
            Shape: (num_tokens, top_k, hidden_size).
        topk_weights: The weights assigned to each expert for each token.
            Shape: (num_tokens, top_k).
        outputs: Optional pre-allocated output tensor.
            Shape: (num_tokens, hidden_size).
        topk_ids: Optional indices of the top-k experts. Used when
            `expert_map` is provided. Shape: (num_tokens, top_k).
        expert_map: Optional mapping for Expert Parallelism. A value < 0
            indicates an invalid token/expert pair that will be skipped.

    Returns:
        The fused weighted sum of expert outputs.
        Shape: (num_tokens, hidden_size).
    """
    assert inputs.ndim == 3
    assert topk_weights.ndim == 2
    assert inputs.is_contiguous()
    assert topk_weights.is_contiguous()
    assert inputs.dtype in (torch.float32, torch.float16, torch.bfloat16)
    assert topk_weights.dtype in (torch.float32, torch.float16, torch.bfloat16)

    num_tokens, top_k, size = inputs.shape
    output_shape = (num_tokens, size)
    if outputs is None:
        outputs = torch.empty(output_shape, dtype=inputs.dtype, device=inputs.device)

    assert outputs.shape == output_shape
    assert topk_weights.shape == (num_tokens, top_k)

    if not isinstance(inputs, FakeTensor):
        BLOCK_M, BLOCK_K, num_warps, num_stages = _heuristic_config(
            num_tokens,
            top_k,
            size,
            inputs.element_size(),
        )
        grid = (triton.cdiv(size, BLOCK_K), triton.cdiv(num_tokens, BLOCK_M))
        moe_fused_mul_sum_kernel[grid](
            inputs,
            topk_weights,
            outputs,
            topk_ids,
            expert_map,
            num_tokens,
            top_k * size,
            expert_map is not None,
            top_k,
            size,
            BLOCK_M,
            BLOCK_K,
            num_warps=num_warps,
            num_stages=num_stages,
            launch_pdl=_pdl_enabled(),
        )

    return outputs


_SHARED_SUM_READY: dict[tuple[int, float], bool] = {}


def _launch_shared_sum(inputs, weights, shared, outputs, routed_scale):
    m, top_k, size = inputs.shape
    bm, bk, warps, stages = _heuristic_config(m, top_k, size, inputs.element_size())
    moe_fused_mul_sum_kernel[(triton.cdiv(size, bk), triton.cdiv(m, bm))](
        inputs, weights, outputs, None, None, m, top_k * size,
        False, top_k, size, bm, bk,
        launch_pdl=_pdl_enabled(), shared_ptr=shared, with_shared=True,
        routed_scale=routed_scale,
        num_warps=warps, num_stages=stages,
    )


def _qualify_shared_sum(device, routed_scale) -> bool:
    """Official-worker startup only: private operands, never a model output.

    Compare stored BF16 payloads to the original reduction followed by torch
    add, including nonfinite and signed-zero inputs. This bounded check is an
    admission filter; the official complete run remains the correctness gate.
    """
    idx = (device.index, routed_scale)
    if idx in _SHARED_SUM_READY:
        return _SHARED_SUM_READY[idx]
    if torch.cuda.is_current_stream_capturing():
        return False
    from vllm.logger import init_logger

    ready = False
    try:
        patterns = torch.tensor(
            [0, 32768, 1, 32769, 127, 32895, 128, 32896,
             16256, 49024, 16320, 48960, 32639, 65407,
             32640, 65408, 32704, 32705, 65473],
            dtype=torch.int32, device=device,
        )
        for m in (1, 5, 8, 33, 129, 496):
            for special in (False, True):
                n = m * 6 * 2688
                at = torch.arange(n, device=device, dtype=torch.int32)
                if special:
                    inputs = patterns[at % patterns.numel()].to(torch.int16)
                    inputs = inputs.view(torch.bfloat16).view(m, 6, 2688)
                    s = torch.arange(m * 2688, device=device, dtype=torch.int32)
                    shared = patterns[(s * 7 + 3) % patterns.numel()]
                    shared = shared.to(torch.int16).view(torch.bfloat16).view(m, 2688)
                else:
                    inputs = ((at % 97 - 48).float() / 31).to(torch.bfloat16)
                    inputs = inputs.view(m, 6, 2688)
                    s = torch.arange(m * 2688, device=device, dtype=torch.int32)
                    shared = ((s % 71 - 35).float() / 29).to(torch.bfloat16)
                    shared = shared.view(m, 2688)
                w = torch.arange(m * 6, device=device, dtype=torch.int32)
                weights = ((w % 7 - 3).float() / 8).view(m, 6)
                ref = moe_fused_mul_sum(inputs, weights)
                if routed_scale != 1.0:
                    ref.mul_(routed_scale)
                ref = shared + ref
                actual = torch.empty_like(ref)
                _launch_shared_sum(inputs, weights, shared, actual, routed_scale)
                if not torch.equal(ref.view(torch.int16), actual.view(torch.int16)):
                    raise RuntimeError("private shared-sum BF16 comparison refused")
        ready = True
    except Exception as exc:
        # No actual model call has been replaced by the new kernel yet.
        init_logger(__name__).info("shared MoE sum private qualification refused: %s",
                                   type(exc).__name__)
    _SHARED_SUM_READY[idx] = ready
    init_logger(__name__).info("shared MoE sum private qualification: %s", ready)
    return ready


def moe_fused_mul_sum_with_shared(inputs, topk_weights, shared, outputs, routed_scale):
    """Single-writer result; preserve ordinary reduction/add on unsupported calls."""
    eligible = (
        not isinstance(inputs, FakeTensor)
        and inputs.is_cuda and inputs.dtype == torch.bfloat16
        and shared.dtype == torch.bfloat16 and outputs.dtype == torch.bfloat16
        and inputs.ndim == 3 and inputs.shape[0] > 0
        and inputs.shape[1:] == (6, 2688)
        and topk_weights.dtype == torch.float32
        and topk_weights.shape == inputs.shape[:2]
        and shared.shape == outputs.shape == (inputs.shape[0], 2688)
        and shared.device == outputs.device == inputs.device == topk_weights.device
        and inputs.is_contiguous() and topk_weights.is_contiguous()
        and shared.is_contiguous() and outputs.is_contiguous()
    )
    if eligible and _qualify_shared_sum(inputs.device, routed_scale):
        _launch_shared_sum(inputs, topk_weights, shared, outputs, routed_scale)
    else:
        moe_fused_mul_sum(inputs, topk_weights, outputs=outputs)
        if routed_scale != 1.0:
            outputs.mul_(routed_scale)
        torch.add(shared, outputs, out=outputs)
    return outputs

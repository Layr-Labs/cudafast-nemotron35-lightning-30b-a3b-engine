# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LOCAL PATCH (cudafast-nemotron35): the final SSM state of each prefill in one copy.

The prefill path stored each sequence's final state as
    varlen_states = states[last_chunk_indices]          # gather (index kernel)
    ssm_state[state_indices_tensor_p] = varlen_states   # scatter (index_put kernel)
which moves every state twice (2 MiB fp32 per Mamba layer and sequence). This kernel copies
states[last_chunk_indices[b]] to ssm_state[state_indices_tensor_p[b]] directly, element for
element and bit for bit (same dtype, no arithmetic; a negative chunk index wraps as torch
indexing does). VLLM_PREFILL_STATE_COPY=0 (v1/worker/prefill_fast.py) restores the pair.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _final_state_copy_kernel(
    src_ptr,
    dst_ptr,
    src_idx_ptr,
    dst_idx_ptr,
    nchunks,
    numel,
    dst_stride,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    pid = tl.program_id(1)
    si = tl.load(src_idx_ptr + b).to(tl.int64)
    si = tl.where(si < 0, si + nchunks, si)
    di = tl.load(dst_idx_ptr + b).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    v = tl.load(src_ptr + si * numel + offs, mask=mask)
    tl.store(dst_ptr + di * dst_stride + offs, v, mask=mask)


def final_state_copy(
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    states: torch.Tensor,
    last_chunk_indices: torch.Tensor,
) -> bool:
    """ssm_state[state_indices] = states[last_chunk_indices], in one kernel. Returns False
    (and does nothing) when the fast copy does not apply."""
    if (
        states.dtype != ssm_state.dtype
        or state_indices.dim() != 1
        or last_chunk_indices.dim() != 1
        or state_indices.numel() != last_chunk_indices.numel()
        or states.shape[1:] != ssm_state.shape[1:]
        or not states.is_contiguous()
        or not ssm_state[0].is_contiguous()
    ):
        return False
    batch = state_indices.numel()
    if batch == 0:
        return True
    numel = states[0].numel()
    block = 4096
    grid = (batch, triton.cdiv(numel, block))
    _final_state_copy_kernel[grid](
        states,
        ssm_state,
        last_chunk_indices,
        state_indices,
        states.shape[0],
        numel,
        ssm_state.stride(0),
        BLOCK=block,
        num_warps=4,
    )
    return True

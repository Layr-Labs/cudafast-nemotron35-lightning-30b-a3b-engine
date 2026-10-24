# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_state_passing.py

# ruff: noqa: E501

import torch

from vllm.model_executor.layers.mamba.ops.triton_helpers import fast_exp
from vllm.triton_utils import tl, triton


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
    ],
    key=["dim"],
)
@triton.jit
def _state_passing_fwd_kernel(
    # Pointers to matrices
    states_ptr,
    out_ptr,
    dA_cs_ptr,
    initstates_ptr,
    last_chunk_indices_ptr,
    final_states_ptr,
    final_indices_ptr,
    # Matrix dimensions
    dim: tl.constexpr,
    chunk_size: tl.constexpr,
    # Strides
    stride_states_chunk: tl.int64,
    stride_states_head: tl.int64,
    stride_states_dim: tl.constexpr,
    stride_out_chunk: tl.int64,
    stride_out_head: tl.int64,
    stride_out_dim: tl.constexpr,
    stride_dA_cs_head: tl.int64,
    stride_dA_cs_chunk: tl.int64,
    stride_dA_cs_csize: tl.constexpr,
    stride_initstates_batch: tl.int64,
    stride_initstates_head: tl.int64,
    stride_initstates_dim: tl.constexpr,
    stride_final_batch: tl.int64,
    stride_final_head: tl.int64,
    stride_final_indices: tl.int64,
    final_cache_size: tl.constexpr,
    # Meta-parameters
    HAS_INITSTATES: tl.constexpr,
    STORE_FINAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    # Derive this sequence's chunk range from last_chunk_indices
    chunk_end = tl.load(last_chunk_indices_ptr + pid_b) + 1
    chunk_start = (
        tl.load(last_chunk_indices_ptr + pid_b - 1, mask=pid_b > 0, other=-1) + 1
    )

    # Offset pointers to this sequence's first chunk
    states_ptr += chunk_start * stride_states_chunk + pid_h * stride_states_head
    dA_cs_ptr += (
        pid_h * stride_dA_cs_head
        + chunk_start * stride_dA_cs_chunk
        + (chunk_size - 1) * stride_dA_cs_csize
    )
    out_ptr += chunk_start * stride_out_chunk + pid_h * stride_out_head

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    states_ptrs = states_ptr + offs_m * stride_states_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim

    # Load initial state once — no per-chunk branching needed
    if HAS_INITSTATES:
        initstates_ptrs = (
            initstates_ptr
            + pid_b * stride_initstates_batch
            + pid_h * stride_initstates_head
            + offs_m * stride_initstates_dim
        )
        states = tl.load(initstates_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    else:
        states = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Loop over only this sequence's chunks — branchless
    nchunks_this_seq = chunk_end - chunk_start
    for chunk in range(nchunks_this_seq):
        new_states = tl.load(states_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        states = fast_exp(dA_cs) * states + new_states
        if STORE_FINAL:
            # Scan reads the preceding boundary within the same sequence,
            # never the sequence's terminal boundary. Store the latter in
            # its cache block below instead of gathering and copying it.
            if chunk + 1 < nchunks_this_seq:
                tl.store(out_ptrs, states, mask=offs_m < dim)
        else:
            tl.store(out_ptrs, states, mask=offs_m < dim)

        states_ptrs += stride_states_chunk
        dA_cs_ptr += stride_dA_cs_chunk
        out_ptrs += stride_out_chunk

    if STORE_FINAL:
        cache_idx = tl.load(final_indices_ptr + pid_b * stride_final_indices).to(
            tl.int64
        )
        # Preserve tensor indexing for a legal negative block index too.
        cache_idx = tl.where(cache_idx < 0, cache_idx + final_cache_size, cache_idx)
        final_ptrs = (
            final_states_ptr
            + cache_idx * stride_final_batch
            + pid_h * stride_final_head
            + offs_m
        )
        tl.store(final_ptrs, states, mask=offs_m < dim)


def _state_passing_fwd(
    states,
    dA_cumsum,
    last_chunk_indices,
    initial_states=None,
    out_dtype=None,
    final_states_out=None,
    final_states_indices=None,
):
    nchunks, nheads, dim = states.shape
    chunk_size = dA_cumsum.shape[-1]
    batch = last_chunk_indices.shape[0]
    assert dA_cumsum.shape == (nheads, nchunks, chunk_size)
    out_dtype = states.dtype if out_dtype is None else out_dtype
    store_final = final_states_out is not None
    assert store_final == (final_states_indices is not None)
    if store_final:
        assert batch == 1 and final_states_out.dtype == out_dtype == torch.float32
        assert final_states_out.ndim == 4
        assert final_states_out.shape[1] == nheads
        assert final_states_out.shape[2] * final_states_out.shape[3] == dim
        assert final_states_out.stride(3) == 1
        assert final_states_out.stride(2) == final_states_out.shape[3]
        assert final_states_indices.shape == (batch,)
        assert final_states_indices.dtype in (torch.int32, torch.int64)
        assert final_states_out.device == states.device == final_states_indices.device
    out = torch.empty((nchunks, nheads, dim), device=states.device, dtype=out_dtype)

    initial_states_strides = (
        (initial_states.stride(0), initial_states.stride(1), initial_states.stride(2))
        if initial_states is not None
        else (0, 0, 0)
    )

    def grid(meta):
        return (triton.cdiv(dim, meta["BLOCK_SIZE"]), batch, nheads)

    with torch.accelerator.device_index(states.device.index):
        _state_passing_fwd_kernel[grid](
            states_ptr=states,
            out_ptr=out,
            dA_cs_ptr=dA_cumsum,
            initstates_ptr=initial_states,
            last_chunk_indices_ptr=last_chunk_indices,
            final_states_ptr=final_states_out,
            final_indices_ptr=final_states_indices,
            dim=dim,
            chunk_size=chunk_size,
            stride_states_chunk=states.stride(0),
            stride_states_head=states.stride(1),
            stride_states_dim=states.stride(2),
            stride_out_chunk=out.stride(0),
            stride_out_head=out.stride(1),
            stride_out_dim=out.stride(2),
            stride_dA_cs_head=dA_cumsum.stride(0),
            stride_dA_cs_chunk=dA_cumsum.stride(1),
            stride_dA_cs_csize=dA_cumsum.stride(2),
            stride_initstates_batch=initial_states_strides[0],
            stride_initstates_head=initial_states_strides[1],
            stride_initstates_dim=initial_states_strides[2],
            stride_final_batch=final_states_out.stride(0) if store_final else 0,
            stride_final_head=final_states_out.stride(1) if store_final else 0,
            stride_final_indices=final_states_indices.stride(0) if store_final else 0,
            final_cache_size=final_states_out.shape[0] if store_final else 0,
            HAS_INITSTATES=initial_states is not None,
            STORE_FINAL=store_final,
        )
    return out


# LOCAL PATCH: fixed tile configs for reproducible numerics (ssd_pinned_configs.py).
from .ssd_pinned_configs import pin  # noqa: E402

pin(_state_passing_fwd_kernel, BLOCK_SIZE=1024)

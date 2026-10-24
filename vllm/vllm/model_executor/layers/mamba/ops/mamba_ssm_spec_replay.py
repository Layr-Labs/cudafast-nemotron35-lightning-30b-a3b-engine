# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LOCAL PATCH (cudafast-nemotron35): spec-decode SSM state update that keeps
one anchor state per request instead of materializing every draft position.

The reference spec-decode path (`_selective_scan_update_kernel` with
IS_SPEC_DECODING) loads the state from slot ``num_accepted - 1`` of the
request's ``1 + d`` state slots and stores the full fp32 state after *every*
one of the ``d + 1`` verify tokens into slots ``0..d``; the next verify reads
exactly one of them. At 64 heads x 64 x 128 fp32 that is ``(d + 1)`` x 2 MiB of
writes per Mamba layer per verify, of which ``d`` x 2 MiB are never read.

This kernel computes the same thing with one state read and one state write:

* Slot 0 is the anchor. It always holds the state after token 0 of the
  previous verify, which is exactly what the reference leaves in slot 0 (and,
  after a prefill, the prefill's final state, also as in the reference).
* Slot 1's first ~145 KB hold a ring of the raw kernel inputs (x, B, dt; the
  16-bit values the reference loads) of tokens 1..d of every verify, indexed by
  absolute token position mod ``ring_len``. ``ring_len >= 2d + 1`` makes the
  positions a verify replays (``P-d..P-1``) disjoint from the ones it writes
  (``P+1..P+d``), so no program can overwrite an entry another program of the
  same launch still has to read.
* A verify loads the anchor, replays tokens ``1..num_accepted-1`` of the
  previous verify from the ring (the reference's ``init_token_idx`` is
  ``num_accepted - 1``, so the replayed state is the reference's slot
  ``num_accepted - 1``), then runs the d + 1 new tokens with the reference's
  per-token update and output code, storing the state after token 0 to the
  anchor and the inputs of tokens 1..d to the ring.

Bit-exactness: every state update is the reference's fp32 expression
(``dA = fast_exp(A * dt)``, ``dB = B * dt``, ``state * dA + dB * x``) on the
same 16-bit inputs in the same order, and fp32 stores and loads are exact, so
the state entering each verify and every output row equal the reference's.
The original kernel's launch configuration (BLOCK_SIZE_M, num_warps, grid)
is the reference's. A narrow sm121 depth-3 specialization below pairs two
separate original 4x128 state chains in one warp and shares head/group reads.
It preserves the source recurrence and separate reduction operand shapes;
that is not proof of device bit equality or a performance improvement.
Slots 2..d are never touched, and slot 1 holds the ring instead of a state; no
code path reads those slots in mamba_cache_mode "none" except the reference
kernel's next-verify load, which this kernel replaces.

``VLLM_MAMBA_SPEC_REPLAY=0`` restores the reference kernel.
"""

import functools
import os

import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    softplus,
    try_get_optimal_ssm_config,
)
from vllm.model_executor.layers.mamba.ops.triton_helpers import fast_exp
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


def spec_replay_enabled() -> bool:
    """The kill switch: on unless VLLM_MAMBA_SPEC_REPLAY is 0/false/off."""
    value = os.environ.get("VLLM_MAMBA_SPEC_REPLAY", "1").strip().lower()
    return value not in ("0", "false", "off", "no")


def spec_replay_ring_len(num_spec: int) -> int:
    """Ring entries per request: any value >= 2 * num_spec + 1 is hazard-free."""
    return 2 * (num_spec + 1)


def spec_replay_ring_elems(nheads: int, dim: int, ngroups: int, dstate: int) -> int:
    """16-bit ring elements per ring entry: x (nheads*dim), B (ngroups*dstate),
    dt (nheads)."""
    return nheads * dim + ngroups * dstate + nheads


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics(
    {"STORE_REPLAYED": lambda args: args["replayed_ptr"] is not None}
)
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.jit(do_not_specialize=["N"])
def _selective_scan_update_spec_replay_kernel(
    # Pointers to matrices
    state_ptr,
    ring_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    state_batch_indices_ptr,
    null_block_id,
    num_accepted_tokens_ptr,
    cu_seqlens_ptr,
    seq_lens_ptr,
    replayed_ptr,
    # Matrix dimensions
    N,
    nheads,
    dim,
    dstate,
    nheads_ngroups_ratio,
    ngroups,
    ring_len,
    # Strides
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    stride_ring_batch,
    stride_x_batch,
    stride_x_head,
    stride_x_dim,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_dim,
    stride_dt_bias_head,
    stride_dt_bias_dim,
    stride_A_head,
    stride_A_dim,
    stride_A_dstate,
    stride_B_batch,
    stride_B_group,
    stride_B_dstate,
    stride_C_batch,
    stride_C_group,
    stride_C_dstate,
    stride_D_head,
    stride_D_dim,
    stride_z_batch,
    stride_z_head,
    stride_z_dim,
    stride_out_batch,
    stride_out_head,
    stride_out_dim,
    stride_state_indices_batch,
    stride_state_indices_T,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    TIE_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    STORE_REPLAYED: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    # The dt/A of a head are scalars (stride-0 expansion over head_dim); that is
    # the only layout the Mamba2 decode path produces and the only one the
    # ring stores.
    tl.static_assert(TIE_HDIM)

    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    bos = tl.load(cu_seqlens_ptr + pid_b).to(tl.int64)
    eos = tl.load(cu_seqlens_ptr + pid_b + 1).to(tl.int64)
    seq_len = eos - bos

    if seq_len == 0:
        return

    # The reference starts from slot init_token_idx = num_accepted - 1, i.e.
    # the state after token init_token_idx of the previous verify; the anchor
    # holds the state after token 0, so replay tokens 1..init_token_idx.
    num_accepted = tl.load(num_accepted_tokens_ptr + pid_b).to(tl.int64)
    num_replay = tl.maximum(num_accepted - 1, 0)
    # Absolute position of this verify's token 0.
    pos0 = tl.load(seq_lens_ptr + pid_b).to(tl.int64) - seq_len

    state_batch_indices_ptr += pid_b * stride_state_indices_batch
    state_batch_idx = tl.load(state_batch_indices_ptr).to(tl.int64)
    ring_batch_idx = tl.load(state_batch_indices_ptr + stride_state_indices_T).to(
        tl.int64
    )
    state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head
    ring_ok = ring_batch_idx != null_block_id
    ring_base = ring_ptr + ring_batch_idx * stride_ring_batch
    ring_x_base = ring_base + pid_h * dim
    ring_B_base = ring_base + ring_len * nheads * dim
    ring_B_base += (pid_h // nheads_ngroups_ratio) * dstate
    ring_dt_base = ring_base + ring_len * (nheads * dim + ngroups * dstate) + pid_h
    ring_stride_x = nheads * dim
    ring_stride_B = ngroups * dstate
    ring_stride_dt = nheads
    # One program per head writes dt, one per group writes B; every program
    # writes its own slice of x.
    is_dt_writer = pid_m == 0
    is_B_writer = (pid_m == 0) & ((pid_h % nheads_ngroups_ratio) == 0)

    x_ptr += bos * stride_x_batch + pid_h * stride_x_head
    dt_ptr += bos * stride_dt_batch + pid_h * stride_dt_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head
    B_ptr += bos * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_ptr += bos * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    if HAS_Z:
        z_ptr += bos * stride_z_batch + pid_h * stride_z_head
    out_ptr += bos * stride_out_batch + pid_h * stride_out_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (
        offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
    )

    mask = (offs_m[:, None] < dim) & (offs_n[None, :] < dstate)
    mask &= state_batch_idx != null_block_id
    state = tl.load(state_ptrs, mask=mask, other=0.0).to(tl.float32)

    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    if HAS_D:
        D_ptr += pid_h * stride_D_head
        D_ptrs = D_ptr + offs_m * stride_D_dim

    # Replay the accepted tokens 1..num_replay of the previous verify: the
    # reference's state update on the reference's inputs, no outputs.
    for i_r in range(num_replay):
        # Non-negative even for garbage metadata (dummy runs), so the address
        # always stays inside slot 1's ring.
        ring_pos = ((pos0 - num_replay + i_r) % ring_len + ring_len) % ring_len
        x = tl.load(
            ring_x_base + ring_pos * ring_stride_x + offs_m,
            mask=(offs_m < dim) & ring_ok,
            other=0.0,
        ).to(tl.float32)
        dt = tl.load(
            ring_dt_base + ring_pos * ring_stride_dt, mask=ring_ok, other=0.0
        ).to(tl.float32)
        if HAS_DT_BIAS:
            dt += tl.load(dt_bias_ptr).to(tl.float32)
        if DT_SOFTPLUS:
            dt = softplus(dt)
        A = tl.load(A_ptr).to(tl.float32)
        dA = fast_exp(A * dt)  # scalar, not a matrix

        B = tl.load(
            ring_B_base + ring_pos * ring_stride_B + offs_n,
            mask=(offs_n < dstate) & ring_ok,
            other=0.0,
        ).to(tl.float32)

        dB = B * dt
        state = state * dA + dB * x[:, None]

    if STORE_REPLAYED:
        # Test hook: the state this verify starts from (the reference's slot
        # num_accepted - 1), as a contiguous (N, nheads, dim, dstate) tensor.
        replayed_ptrs = (
            replayed_ptr
            + ((pid_b * nheads + pid_h) * dim) * dstate
            + offs_m[:, None] * dstate
            + offs_n[None, :]
        )
        tl.store(replayed_ptrs, state, mask=mask)

    for i_t in range(seq_len):
        x_ptrs = x_ptr + offs_m * stride_x_dim
        dt_ptrs = dt_ptr + offs_m * stride_dt_dim
        B_ptrs = B_ptr + offs_n * stride_B_dstate
        C_ptrs = C_ptr + offs_n * stride_C_dstate
        if HAS_Z:
            z_ptrs = z_ptr + offs_m * stride_z_dim
        out_ptrs = out_ptr + offs_m * stride_out_dim

        x_raw = tl.load(x_ptrs, mask=offs_m < dim, other=0.0)
        x = x_raw.to(tl.float32)
        dt_raw = tl.load(dt_ptr)
        dt = dt_raw.to(tl.float32)
        if HAS_DT_BIAS:
            dt += tl.load(dt_bias_ptr).to(tl.float32)
        if DT_SOFTPLUS:
            dt = softplus(dt)
        A = tl.load(A_ptr).to(tl.float32)
        dA = fast_exp(A * dt)  # scalar, not a matrix

        B_raw = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0)
        B = B_raw.to(tl.float32)
        C = tl.load(C_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
        if HAS_D:
            D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if HAS_Z:
            z = tl.load(z_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

        dB = B * dt
        state = state * dA + dB * x[:, None]

        if i_t == 0:
            # The new anchor: the state after token 0, as the reference's slot 0.
            tl.store(state_ptrs, state.to(state_ptrs.dtype.element_ty), mask=mask)
        else:
            ring_pos = ((pos0 + i_t) % ring_len + ring_len) % ring_len
            tl.store(
                ring_x_base + ring_pos * ring_stride_x + offs_m,
                x_raw,
                mask=(offs_m < dim) & ring_ok,
            )
            tl.store(
                ring_dt_base + ring_pos * ring_stride_dt,
                dt_raw,
                mask=ring_ok & is_dt_writer,
            )
            tl.store(
                ring_B_base + ring_pos * ring_stride_B + offs_n,
                B_raw,
                mask=(offs_n < dstate) & ring_ok & is_B_writer,
            )

        out = tl.sum(state * C[None, :], axis=1)
        if HAS_D:
            out += x * D
        if HAS_Z:
            out *= z * tl.sigmoid(z)
        tl.store(out_ptrs, out, mask=offs_m < dim)

        x_ptr += stride_x_batch
        dt_ptr += stride_dt_batch
        B_ptr += stride_B_batch
        C_ptr += stride_C_batch
        out_ptr += stride_out_batch
        if HAS_Z:
            z_ptr += stride_z_batch


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics(
    {"STORE_REPLAYED": lambda args: args["replayed_ptr"] is not None}
)
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.jit(do_not_specialize=["N"])
def _selective_scan_update_spec_replay_pair_kernel(
    state_ptr,
    ring_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    state_batch_indices_ptr,
    null_block_id,
    num_accepted_tokens_ptr,
    cu_seqlens_ptr,
    seq_lens_ptr,
    replayed_ptr,
    N,
    nheads,
    dim,
    dstate,
    nheads_ngroups_ratio,
    ngroups,
    ring_len,
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    stride_ring_batch,
    stride_x_batch,
    stride_x_head,
    stride_x_dim,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_dim,
    stride_dt_bias_head,
    stride_dt_bias_dim,
    stride_A_head,
    stride_A_dim,
    stride_A_dstate,
    stride_B_batch,
    stride_B_group,
    stride_B_dstate,
    stride_C_batch,
    stride_C_group,
    stride_C_dstate,
    stride_D_head,
    stride_D_dim,
    stride_z_batch,
    stride_z_head,
    stride_z_dim,
    stride_out_batch,
    stride_out_head,
    stride_out_dim,
    stride_state_indices_batch,
    stride_state_indices_T,
    DT_SOFTPLUS: tl.constexpr,
    TIE_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    STORE_REPLAYED: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    # Keep two independent original microtiles, never an 8x128 reduction.
    tl.static_assert(TIE_HDIM)
    tl.static_assert(BLOCK_SIZE_M == 4)
    tl.static_assert(BLOCK_SIZE_DSTATE == 128)

    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    bos = tl.load(cu_seqlens_ptr + pid_b).to(tl.int64)
    eos = tl.load(cu_seqlens_ptr + pid_b + 1).to(tl.int64)
    seq_len = eos - bos
    if seq_len == 0:
        return

    num_accepted = tl.load(num_accepted_tokens_ptr + pid_b).to(tl.int64)
    num_replay = tl.maximum(num_accepted - 1, 0)
    pos0 = tl.load(seq_lens_ptr + pid_b).to(tl.int64) - seq_len

    state_batch_indices_ptr += pid_b * stride_state_indices_batch
    state_batch_idx = tl.load(state_batch_indices_ptr).to(tl.int64)
    ring_batch_idx = tl.load(state_batch_indices_ptr + stride_state_indices_T).to(
        tl.int64
    )
    state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head
    ring_ok = ring_batch_idx != null_block_id
    ring_base = ring_ptr + ring_batch_idx * stride_ring_batch
    ring_x_base = ring_base + pid_h * dim
    ring_B_base = ring_base + ring_len * nheads * dim
    ring_B_base += (pid_h // nheads_ngroups_ratio) * dstate
    ring_dt_base = ring_base + ring_len * (nheads * dim + ngroups * dstate) + pid_h
    ring_stride_x = nheads * dim
    ring_stride_B = ngroups * dstate
    ring_stride_dt = nheads
    # The second state tile does not add another scalar/vector writer.
    is_dt_writer = pid_m == 0
    is_B_writer = (pid_m == 0) & ((pid_h % nheads_ngroups_ratio) == 0)

    x_ptr += bos * stride_x_batch + pid_h * stride_x_head
    dt_ptr += bos * stride_dt_batch + pid_h * stride_dt_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head
    B_ptr += bos * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_ptr += bos * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    if HAS_Z:
        z_ptr += bos * stride_z_batch + pid_h * stride_z_head
    out_ptr += bos * stride_out_batch + pid_h * stride_out_head

    offs_m0 = pid_m * (2 * BLOCK_SIZE_M) + tl.arange(0, BLOCK_SIZE_M)
    offs_m1 = offs_m0 + BLOCK_SIZE_M
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs0 = state_ptr + (
        offs_m0[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
    )
    state_ptrs1 = state_ptr + (
        offs_m1[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
    )
    mask0 = (offs_m0[:, None] < dim) & (offs_n[None, :] < dstate)
    mask0 &= state_batch_idx != null_block_id
    mask1 = (offs_m1[:, None] < dim) & (offs_n[None, :] < dstate)
    mask1 &= state_batch_idx != null_block_id
    state0 = tl.load(state_ptrs0, mask=mask0, other=0.0).to(tl.float32)
    state1 = tl.load(state_ptrs1, mask=mask1, other=0.0).to(tl.float32)

    if HAS_D:
        D_ptr += pid_h * stride_D_head
        D_ptrs0 = D_ptr + offs_m0 * stride_D_dim
        D_ptrs1 = D_ptr + offs_m1 * stride_D_dim

    for i_r in range(num_replay):
        ring_pos = ((pos0 - num_replay + i_r) % ring_len + ring_len) % ring_len
        x0 = tl.load(
            ring_x_base + ring_pos * ring_stride_x + offs_m0,
            mask=(offs_m0 < dim) & ring_ok,
            other=0.0,
        ).to(tl.float32)
        x1 = tl.load(
            ring_x_base + ring_pos * ring_stride_x + offs_m1,
            mask=(offs_m1 < dim) & ring_ok,
            other=0.0,
        ).to(tl.float32)
        dt = tl.load(
            ring_dt_base + ring_pos * ring_stride_dt, mask=ring_ok, other=0.0
        ).to(tl.float32)
        if HAS_DT_BIAS:
            dt += tl.load(dt_bias_ptr).to(tl.float32)
        if DT_SOFTPLUS:
            dt = softplus(dt)
        A = tl.load(A_ptr).to(tl.float32)
        dA = fast_exp(A * dt)
        B = tl.load(
            ring_B_base + ring_pos * ring_stride_B + offs_n,
            mask=(offs_n < dstate) & ring_ok,
            other=0.0,
        ).to(tl.float32)
        dB = B * dt
        state0 = state0 * dA + dB * x0[:, None]
        state1 = state1 * dA + dB * x1[:, None]

    if STORE_REPLAYED:
        replayed_base = replayed_ptr + ((pid_b * nheads + pid_h) * dim) * dstate
        replayed_ptrs0 = (
            replayed_base + offs_m0[:, None] * dstate + offs_n[None, :]
        )
        replayed_ptrs1 = (
            replayed_base + offs_m1[:, None] * dstate + offs_n[None, :]
        )
        tl.store(replayed_ptrs0, state0, mask=mask0)
        tl.store(replayed_ptrs1, state1, mask=mask1)

    for i_t in range(seq_len):
        x_ptrs0 = x_ptr + offs_m0 * stride_x_dim
        x_ptrs1 = x_ptr + offs_m1 * stride_x_dim
        B_ptrs = B_ptr + offs_n * stride_B_dstate
        C_ptrs = C_ptr + offs_n * stride_C_dstate
        if HAS_Z:
            z_ptrs0 = z_ptr + offs_m0 * stride_z_dim
            z_ptrs1 = z_ptr + offs_m1 * stride_z_dim
        out_ptrs0 = out_ptr + offs_m0 * stride_out_dim
        out_ptrs1 = out_ptr + offs_m1 * stride_out_dim

        x_raw0 = tl.load(x_ptrs0, mask=offs_m0 < dim, other=0.0)
        x_raw1 = tl.load(x_ptrs1, mask=offs_m1 < dim, other=0.0)
        x0 = x_raw0.to(tl.float32)
        x1 = x_raw1.to(tl.float32)
        dt_raw = tl.load(dt_ptr)
        dt = dt_raw.to(tl.float32)
        if HAS_DT_BIAS:
            dt += tl.load(dt_bias_ptr).to(tl.float32)
        if DT_SOFTPLUS:
            dt = softplus(dt)
        A = tl.load(A_ptr).to(tl.float32)
        dA = fast_exp(A * dt)

        B_raw = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0)
        B = B_raw.to(tl.float32)
        C = tl.load(C_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
        if HAS_D:
            D0 = tl.load(D_ptrs0, mask=offs_m0 < dim, other=0.0).to(tl.float32)
            D1 = tl.load(D_ptrs1, mask=offs_m1 < dim, other=0.0).to(tl.float32)
        if HAS_Z:
            z0 = tl.load(z_ptrs0, mask=offs_m0 < dim, other=0.0).to(tl.float32)
            z1 = tl.load(z_ptrs1, mask=offs_m1 < dim, other=0.0).to(tl.float32)

        dB = B * dt
        state0 = state0 * dA + dB * x0[:, None]
        state1 = state1 * dA + dB * x1[:, None]

        if i_t == 0:
            tl.store(state_ptrs0, state0.to(state_ptrs0.dtype.element_ty), mask=mask0)
            tl.store(state_ptrs1, state1.to(state_ptrs1.dtype.element_ty), mask=mask1)
        else:
            ring_pos = ((pos0 + i_t) % ring_len + ring_len) % ring_len
            tl.store(
                ring_x_base + ring_pos * ring_stride_x + offs_m0,
                x_raw0,
                mask=(offs_m0 < dim) & ring_ok,
            )
            tl.store(
                ring_x_base + ring_pos * ring_stride_x + offs_m1,
                x_raw1,
                mask=(offs_m1 < dim) & ring_ok,
            )
            tl.store(
                ring_dt_base + ring_pos * ring_stride_dt,
                dt_raw,
                mask=ring_ok & is_dt_writer,
            )
            tl.store(
                ring_B_base + ring_pos * ring_stride_B + offs_n,
                B_raw,
                mask=(offs_n < dstate) & ring_ok & is_B_writer,
            )

        out0 = tl.sum(state0 * C[None, :], axis=1)
        out1 = tl.sum(state1 * C[None, :], axis=1)
        if HAS_D:
            out0 += x0 * D0
            out1 += x1 * D1
        if HAS_Z:
            out0 *= z0 * tl.sigmoid(z0)
            out1 *= z1 * tl.sigmoid(z1)
        tl.store(out_ptrs0, out0, mask=offs_m0 < dim)
        tl.store(out_ptrs1, out1, mask=offs_m1 < dim)

        x_ptr += stride_x_batch
        dt_ptr += stride_dt_batch
        B_ptr += stride_B_batch
        C_ptr += stride_C_batch
        out_ptr += stride_out_batch
        if HAS_Z:
            z_ptr += stride_z_batch


def _normalize(state, x, dt, A, B, C, D, z, dt_bias, out, state_batch_indices):
    """The reference wrapper's shape normalization (mamba_ssm.py)."""
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    if out.dim() == 2:
        out = out.unsqueeze(1)
    if state_batch_indices.dim() == 1:
        state_batch_indices = state_batch_indices.unsqueeze(1)
    return state, x, dt, A, B, C, D, z, dt_bias, out, state_batch_indices


def spec_replay_supported(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    dt_bias: torch.Tensor,
    state_batch_indices: torch.Tensor,
    ring_len: int,
) -> bool:
    """Whether the anchor+ring scheme can serve this call exactly; callers fall
    back to the reference kernel otherwise."""
    if state.dim() != 4 or x.dim() != 3 or B.dim() != 3:
        return False
    if state.dtype != torch.float32:
        return False
    if not (x.dtype == B.dtype == dt.dtype) or x.element_size() != 2:
        return False
    tie_hdim = (
        A.stride(-1) == 0
        and A.stride(-2) == 0
        and dt.stride(-1) == 0
        and dt_bias.stride(-1) == 0
    )
    if not tie_hdim:
        return False
    if state_batch_indices.dim() != 2 or state_batch_indices.size(1) < 2:
        return False
    _, nheads, dim, dstate = state.shape
    # The ring lives in slot 1's own 2 MiB: the slot must be one contiguous
    # block of fp32 and hold the ring.
    if state.stride(3) != 1 or state.stride(2) != dstate:
        return False
    if state.stride(1) != dim * dstate:
        return False
    ngroups = B.shape[1]
    ring_elems = ring_len * spec_replay_ring_elems(nheads, dim, ngroups, dstate)
    return ring_elems <= 2 * nheads * dim * dstate


@functools.cache
def _pair_device_supported(device_index: int) -> bool:
    return current_platform.is_device_capability(121, device_id=device_index)


def _tensor_byte_span(tensor: torch.Tensor) -> tuple[int, int] | None:
    """Conservative view bounds, without reading any device tensor values."""
    if tensor.layout != torch.strided or tensor.numel() == 0:
        return None
    last_element = 0
    for size, stride in zip(tensor.shape, tensor.stride()):
        if stride < 0:
            return None
        last_element += (size - 1) * stride
    start = tensor.data_ptr()
    return start, start + (last_element + 1) * tensor.element_size()


def _spans_overlap(lhs: tuple[int, int], rhs: tuple[int, int]) -> bool:
    return lhs[0] < rhs[1] and rhs[0] < lhs[1]


def _paired_replay_supported(
    state, x, dt, A, B, C, D, dt_bias, z, out,
    state_batch_indices, num_accepted_tokens, cu_seqlens, seq_lens,
    ring_len, dt_softplus, block_size_m, num_warps, replayed_out,
) -> bool:
    """A narrow metadata-only specialization; every miss uses original replay."""
    if (
        state.device.type != "cuda"
        or state.device.index is None
        or block_size_m != 4
        or num_warps != 1
        or ring_len != 8
        or dt_softplus is not True
        or z is not None
        or state.ndim != 4
        or state.shape[0] < 2
        or state.shape[1:] != (64, 64, 128)
        or x.shape != (4, 64, 64)
        or dt.shape != x.shape
        or B.shape != (4, 8, 128)
        or C.shape != B.shape
        or A.shape != (64, 64, 128)
        or D.shape != (64, 64)
        or dt_bias.shape != (64, 64)
        or out.shape != x.shape
        or state_batch_indices.shape != (1, 4)
        or num_accepted_tokens.shape != (1,)
        or cu_seqlens.shape != (2,)
        or seq_lens.ndim != 1
        or seq_lens.shape[0] < 1
    ):
        return False
    tensors = (
        state, x, dt, A, B, C, D, dt_bias, out,
        state_batch_indices, num_accepted_tokens, cu_seqlens, seq_lens,
    )
    if replayed_out is not None:
        tensors += (replayed_out,)
    if any(t.device != state.device for t in tensors):
        return False
    if not _pair_device_supported(state.device.index):
        return False
    if state.dtype != torch.float32 or A.dtype != torch.float32:
        return False
    if any(t.dtype != torch.bfloat16 for t in (x, dt, B, C, D, dt_bias, out)):
        return False
    if any(
        t.dtype != torch.int32
        for t in (state_batch_indices, num_accepted_tokens, cu_seqlens, seq_lens)
    ):
        return False
    if (
        state.stride()[1:] != (8192, 128, 1)
        or state.stride(0) < 64 * 64 * 128
        or A.stride() != (1, 0, 0)
        or D.stride() != (1, 0)
        or dt_bias.stride() != (1, 0)
        or dt.stride()[1:] != (1, 0)
        or dt.stride(0) < 64
        or x.stride()[1:] != (64, 1)
        or x.stride(0) < 4096
        or B.stride()[1:] != (128, 1)
        or B.stride(0) < 1024
        or C.stride()[1:] != (128, 1)
        or C.stride(0) < 1024
        or out.stride()[1:] != (64, 1)
        or out.stride(0) < 4096
        or state_batch_indices.stride(1) != 1
        or state_batch_indices.stride(0) < 4
        or num_accepted_tokens.stride(0) != 1
        or cu_seqlens.stride(0) != 1
        or seq_lens.stride(0) != 1
    ):
        return False
    if replayed_out is not None and (
        replayed_out.shape != (1, 64, 64, 128)
        or replayed_out.dtype != torch.float32
        or not replayed_out.is_contiguous()
    ):
        return False

    # Anchor/ring are intentional self read/write aliases within state.
    # Other writes may not alias a live input/parameter/integer metadata view.
    # Read-only x/B/C may share storage; compare addressed spans, not identity.
    reads = (x, dt, A, B, C, D, dt_bias, state_batch_indices,
             num_accepted_tokens, cu_seqlens, seq_lens)
    writes = (state, out)
    if replayed_out is not None:
        writes += (replayed_out,)
    read_spans = tuple(_tensor_byte_span(t) for t in reads)
    write_spans = tuple(_tensor_byte_span(t) for t in writes)
    if any(span is None for span in (*read_spans, *write_spans)):
        return False
    for i, write_span in enumerate(write_spans):
        if any(_spans_overlap(write_span, read_span) for read_span in read_spans):
            return False
        if any(
            _spans_overlap(write_span, other_span)
            for other_span in write_spans[i + 1:]
        ):
            return False
    return True


def selective_state_update_spec_replay(
    state,
    x,
    dt,
    A,
    B,
    C,
    D,
    dt_bias,
    z=None,
    dt_softplus=False,
    state_batch_indices=None,
    null_block_id=NULL_BLOCK_ID,
    out=None,
    num_accepted_tokens=None,
    cu_seqlens=None,
    seq_lens=None,
    ring_len=None,
    is_blackwell=False,
    replayed_out=None,
):
    """
    Spec-decode verify update with anchor + input-ring state handling.

    Arguments as the reference `selective_state_update` in spec mode
    (state_batch_indices: (N, 1 + d) slots, in place; num_accepted_tokens: (N,);
    cu_seqlens: (N + 1,)), plus
        seq_lens: (>= N,) total tokens per request including this verify's,
            so token 0 of request b sits at position seq_lens[b] - query_len[b].
        ring_len: ring entries per request, >= 2 * d + 1.
        replayed_out: test hook; if given, an (N, nheads, dim, dstate) fp32
            tensor that receives the state each request starts the verify from.

    The caller must have checked `spec_replay_supported` for these layouts.
    """
    assert num_accepted_tokens is not None and cu_seqlens is not None
    assert seq_lens is not None and ring_len is not None
    state, x, dt, A, B, C, D, z, dt_bias, out, state_batch_indices = _normalize(
        state, x, dt, A, B, C, D, z, dt_bias, out, state_batch_indices
    )
    _, nheads, dim, dstate = state.shape
    batch = x.shape[0]
    N = len(cu_seqlens) - 1
    max_seqlen = state_batch_indices.size(-1)

    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    assert dt_bias.shape == (nheads, dim)
    assert state_batch_indices.shape[0] >= N
    assert state_batch_indices.shape[1] >= max_seqlen
    assert ring_len >= 2 * (max_seqlen - 1) + 1
    assert out.shape == x.shape
    assert num_accepted_tokens.shape == (N,)
    assert seq_lens.shape[0] >= N
    if replayed_out is not None:
        assert replayed_out.shape == (N, nheads, dim, dstate)
        assert replayed_out.dtype == torch.float32 and replayed_out.is_contiguous()

    # The ring is slot memory reinterpreted as the inputs' 16-bit type.
    ring = state.view(x.dtype)

    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE_M"]), N, nheads)
    z_strides = (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0)
    # The reference's launch configuration for the same shape.
    cache_dtype = str(state.dtype).removeprefix("torch.")
    BLOCK_SIZE_M, num_warps = try_get_optimal_ssm_config(
        dim, dstate, N, nheads, cache_dtype, is_blackwell
    )

    with torch.accelerator.device_index(x.device.index):
        if _paired_replay_supported(
            state, x, dt, A, B, C, D, dt_bias, z, out,
            state_batch_indices, num_accepted_tokens, cu_seqlens, seq_lens,
            ring_len, dt_softplus, BLOCK_SIZE_M, num_warps, replayed_out,
        ):
            pair_grid = lambda META: (
                triton.cdiv(dim, 2 * META["BLOCK_SIZE_M"]), N, nheads
            )
            _selective_scan_update_spec_replay_pair_kernel[pair_grid](
                state,
                ring,
                x,
                dt,
                dt_bias,
                A,
                B,
                C,
                D,
                z,
                out,
                state_batch_indices,
                null_block_id,
                num_accepted_tokens,
                cu_seqlens,
                seq_lens,
                replayed_out,
                N,
                nheads,
                dim,
                dstate,
                nheads // ngroups,
                ngroups,
                ring_len,
                state.stride(0),
                state.stride(1),
                state.stride(2),
                state.stride(3),
                ring.stride(0),
                x.stride(0),
                x.stride(1),
                x.stride(2),
                dt.stride(0),
                dt.stride(1),
                dt.stride(2),
                dt_bias.stride(0),
                dt_bias.stride(1),
                A.stride(0),
                A.stride(1),
                A.stride(2),
                B.stride(0),
                B.stride(1),
                B.stride(2),
                C.stride(0),
                C.stride(1),
                C.stride(2),
                D.stride(0),
                D.stride(1),
                z_strides[0],
                z_strides[1],
                z_strides[2],
                out.stride(0),
                out.stride(1),
                out.stride(2),
                state_batch_indices.stride(0),
                state_batch_indices.stride(1),
                DT_SOFTPLUS=dt_softplus,
                TIE_HDIM=True,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                num_warps=num_warps,
            )
            return
        _selective_scan_update_spec_replay_kernel[grid](
            state,
            ring,
            x,
            dt,
            dt_bias,
            A,
            B,
            C,
            D,
            z,
            out,
            state_batch_indices,
            null_block_id,
            num_accepted_tokens,
            cu_seqlens,
            seq_lens,
            replayed_out,
            N,
            nheads,
            dim,
            dstate,
            nheads // ngroups,
            ngroups,
            ring_len,
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            ring.stride(0),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            dt_bias.stride(0),
            dt_bias.stride(1),
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            D.stride(0),
            D.stride(1),
            z_strides[0],
            z_strides[1],
            z_strides[2],
            out.stride(0),
            out.stride(1),
            out.stride(2),
            state_batch_indices.stride(0),
            state_batch_indices.stride(1),
            dt_softplus,
            True,
            BLOCK_SIZE_M,
            num_warps=num_warps,
        )

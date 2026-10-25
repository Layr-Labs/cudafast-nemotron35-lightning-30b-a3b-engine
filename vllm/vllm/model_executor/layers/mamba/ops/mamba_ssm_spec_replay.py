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
The launch configuration (BLOCK_SIZE_M, num_warps, grid) is the reference's.
Slots 2..d are never touched, and slot 1 holds the ring instead of a state; no
code path reads those slots in mamba_cache_mode "none" except the reference
kernel's next-verify load, which this kernel replaces.

``VLLM_MAMBA_SPEC_REPLAY=0`` restores the reference kernel.
"""

import os
import sys

import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    softplus,
    try_get_optimal_ssm_config,
)
from vllm.model_executor.layers.mamba.ops.triton_helpers import fast_exp
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


def spec_replay_enabled() -> bool:
    """The kill switch: on unless VLLM_MAMBA_SPEC_REPLAY is 0/false/off."""
    value = os.environ.get("VLLM_MAMBA_SPEC_REPLAY", "1").strip().lower()
    return value not in ("0", "false", "off", "no")


def _const_div_enabled() -> bool:
    """Fold the ring length and head/group ratio into the kernel as constants."""
    value = os.environ.get("VLLM_MAMBA_SPEC_REPLAY_CONST_DIV", "1").strip().lower()
    return value not in ("0", "false", "off", "no")


_PDL: bool | None = None


def _pdl_enabled() -> bool:
    """Launch the replay kernel with programmatic dependent launch (sm_90+):
    it then overlaps its launch with the conv1d update before it, which
    already signals its dependents. VLLM_MAMBA_REPLAY_PDL=0 disables."""
    global _PDL
    if _PDL is None:
        from vllm.platforms import current_platform

        _PDL = os.environ.get(
            "VLLM_MAMBA_REPLAY_PDL", "1"
        ) != "0" and current_platform.is_arch_support_pdl()
    return _PDL


# Checkpoint period of the lazy anchor (see spec_replay_lazy).
LAZY_PERIOD = 20


def spec_replay_lazy() -> bool:
    """Lazy anchor: the state is stored only at checkpoint positions
    L - 1 + k * LAZY_PERIOD (L: prompt length; k = 0 is the prefill's final
    state), alternately in slots 0 and 2, and a verify replays every token since
    the last checkpoint from the ring, which then also holds token 0. The store
    happens in the verify whose tokens include the next checkpoint position
    (about one verify in four) instead of in every verify; every state update
    is the same per-token fp32 expression in the same order, so the states and
    outputs are the reference's. The prefill writes L into the slot-0 page's
    padding after the SSM state (write_prefill_tag). VLLM_MAMBA_REPLAY_LAZY=0
    restores an anchor store in every verify."""
    value = os.environ.get("VLLM_MAMBA_REPLAY_LAZY", "1").strip().lower()
    return value not in ("0", "false", "off", "no")


def spec_replay_pipeline() -> bool:
    """Issue the next replay token's ring loads before the current token's
    update. Same fp32 expression and the same order on the state; A and dt_bias
    are loop-invariant. VLLM_MAMBA_REPLAY_PIPELINE=0 restores the serial loads."""
    value = os.environ.get("VLLM_MAMBA_REPLAY_PIPELINE", "1").strip().lower()
    return value not in ("0", "false", "off", "no")


def spec_replay_ring_len(num_spec: int) -> int:
    """Ring entries per request: any value >= 2 * num_spec + 1 is hazard-free;
    the lazy anchor replays up to LAZY_PERIOD - 1 tokens before the num_spec + 1
    it writes, so it needs >= LAZY_PERIOD + num_spec."""
    if spec_replay_lazy():
        return max(2 * (num_spec + 1), LAZY_PERIOD + num_spec + 1)
    return 2 * (num_spec + 1)


def prefill_tag_view(state: torch.Tensor) -> torch.Tensor | None:
    """int32 view of one word per slot just after the slot's SSM state (the
    page padding), or None when the slot stride leaves no room for it."""
    if state.dim() != 4 or state.dtype != torch.float32:
        return None
    slot_elems = state.shape[1] * state.shape[2] * state.shape[3]
    if state.stride(0) < slot_elems + 1 or state.stride(1) != state.shape[2] * state.shape[3]:
        return None
    return torch.as_strided(
        state.view(torch.int32), (state.shape[0],), (state.stride(0),),
        state.storage_offset() + slot_elems,
    )


def write_prefill_tag(state: torch.Tensor, slots: torch.Tensor, seq_lens: torch.Tensor) -> bool:
    """Record each prefilled request's length L next to its slot-0 state: the
    lazy anchor's checkpoint k = 0 is that state (after position L - 1)."""
    tag = prefill_tag_view(state)
    if tag is None:
        return False
    if slots.dim() == 2:
        slots = slots[:, 0]
    tag.index_copy_(0, slots.to(torch.int64), seq_lens.to(torch.int32))
    return True


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
    RING_LEN: tl.constexpr = 0,
    HG_RATIO: tl.constexpr = 0,
    CONST_DIV: tl.constexpr = False,
    launch_pdl: tl.constexpr = False,
    verify_rows_ptr=None,
    PREFIX_LIMIT: tl.constexpr = False,
    LAZY: tl.constexpr = False,
    LAZY_P: tl.constexpr = 20,
    SLOT_ELEMS: tl.constexpr = 0,
    PIPELINE: tl.constexpr = False,
):
    # The dt/A of a head are scalars (stride-0 expansion over head_dim); that is
    # the only layout the Mamba2 decode path produces and the only one the
    # ring stores.
    tl.static_assert(TIE_HDIM)
    if CONST_DIV:
        rl = RING_LEN
        hr = HG_RATIO
    else:
        rl = ring_len
        hr = nheads_ngroups_ratio

    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    # Programmatic dependent launch (same pattern as gather_initial_states and
    # causal_conv1d): every load below follows the wait for the preceding grid.
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    bos = tl.load(cu_seqlens_ptr + pid_b).to(tl.int64)
    eos = tl.load(cu_seqlens_ptr + pid_b + 1).to(tl.int64)
    seq_len = eos - bos

    if seq_len == 0:
        return

    # Only the one-request depth-four verify may receive this device count.
    # Keep seq_len unchanged: it determines the original absolute pos0 below.
    # A sentinel count >= seq_len preserves the complete original recurrence.
    compute_len = seq_len
    if PREFIX_LIMIT:
        valid_rows = tl.load(verify_rows_ptr).to(tl.int64)
        compute_len = tl.minimum(seq_len, tl.maximum(valid_rows, 1))

    # The reference starts from slot init_token_idx = num_accepted - 1, i.e.
    # the state after token init_token_idx of the previous verify; the anchor
    # holds the state after token 0, so replay tokens 1..init_token_idx.
    # Absolute position of this verify's token 0.
    pos0 = tl.load(seq_lens_ptr + pid_b).to(tl.int64) - seq_len

    state_batch_indices_ptr += pid_b * stride_state_indices_batch
    state_batch_idx = tl.load(state_batch_indices_ptr).to(tl.int64)
    ring_batch_idx = tl.load(state_batch_indices_ptr + stride_state_indices_T).to(
        tl.int64
    )
    if LAZY:
        # Checkpoint k sits at position L - 1 + k * LAZY_P (k = 0: the prefill's
        # final state in slot 0); even k live in slot 0, odd k in slot 2. Start
        # from the last checkpoint before token 0 and replay the tokens after it.
        slot2_idx = tl.load(state_batch_indices_ptr + 2 * stride_state_indices_T).to(
            tl.int64
        )
        tag_ptr = (
            state_ptr + state_batch_idx * stride_state_batch + SLOT_ELEMS
        ).to(tl.pointer_type(tl.int32))
        prompt_len = tl.load(tag_ptr, mask=state_batch_idx != null_block_id, other=0).to(tl.int64)
        k_last = tl.maximum(pos0 - prompt_len, 0) // LAZY_P
        ckpt = prompt_len - 1 + k_last * LAZY_P
        num_replay = tl.minimum(tl.maximum(pos0 - 1 - ckpt, 0), LAZY_P - 1)
        anchor_idx = tl.where(k_last % 2 == 0, state_batch_idx, slot2_idx)
        next_idx = tl.where(k_last % 2 == 0, slot2_idx, state_batch_idx)
        # This verify's token that lands on the next checkpoint (none if > d).
        next_t = ckpt + LAZY_P - pos0
        state_ptr_next = state_ptr + next_idx * stride_state_batch + pid_h * stride_state_head
        state_ptr += anchor_idx * stride_state_batch + pid_h * stride_state_head
    else:
        num_accepted = tl.load(num_accepted_tokens_ptr + pid_b).to(tl.int64)
        num_replay = tl.maximum(num_accepted - 1, 0)
        anchor_idx = state_batch_idx
        state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head
    ring_ok = ring_batch_idx != null_block_id
    ring_base = ring_ptr + ring_batch_idx * stride_ring_batch
    ring_x_base = ring_base + pid_h * dim
    ring_B_base = ring_base + rl * nheads * dim
    ring_B_base += (pid_h // hr) * dstate
    ring_dt_base = ring_base + rl * (nheads * dim + ngroups * dstate) + pid_h
    ring_stride_x = nheads * dim
    ring_stride_B = ngroups * dstate
    ring_stride_dt = nheads
    # One program per head writes dt, one per group writes B; every program
    # writes its own slice of x.
    is_dt_writer = pid_m == 0
    is_B_writer = (pid_m == 0) & ((pid_h % hr) == 0)

    x_ptr += bos * stride_x_batch + pid_h * stride_x_head
    dt_ptr += bos * stride_dt_batch + pid_h * stride_dt_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head
    B_ptr += bos * stride_B_batch + (pid_h // hr) * stride_B_group
    C_ptr += bos * stride_C_batch + (pid_h // hr) * stride_C_group
    if HAS_Z:
        z_ptr += bos * stride_z_batch + pid_h * stride_z_head
    out_ptr += bos * stride_out_batch + pid_h * stride_out_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (
        offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
    )

    mask = (offs_m[:, None] < dim) & (offs_n[None, :] < dstate)
    mask &= anchor_idx != null_block_id
    state = tl.load(state_ptrs, mask=mask, other=0.0).to(tl.float32)

    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    if HAS_D:
        D_ptr += pid_h * stride_D_head
        D_ptrs = D_ptr + offs_m * stride_D_dim

    # Replay the accepted tokens 1..num_replay of the previous verify: the
    # reference's state update on the reference's inputs, no outputs.
    if PIPELINE:
        A_s = tl.load(A_ptr).to(tl.float32)
        if HAS_DT_BIAS:
            dt_bias_s = tl.load(dt_bias_ptr).to(tl.float32)
        rmask_x = (offs_m < dim) & ring_ok
        rmask_B = (offs_n < dstate) & ring_ok
        live = num_replay > 0
        ring_pos = ((pos0 - num_replay) % rl + rl) % rl
        x_nx = tl.load(
            ring_x_base + ring_pos * ring_stride_x + offs_m, mask=rmask_x & live, other=0.0
        )
        dt_nx = tl.load(ring_dt_base + ring_pos * ring_stride_dt, mask=ring_ok & live, other=0.0)
        B_nx = tl.load(
            ring_B_base + ring_pos * ring_stride_B + offs_n, mask=rmask_B & live, other=0.0
        )
        for i_r in range(num_replay):
            x = x_nx.to(tl.float32)
            dt = dt_nx.to(tl.float32)
            B = B_nx.to(tl.float32)
            more = i_r + 1 < num_replay
            ring_pos = ((pos0 - num_replay + i_r + 1) % rl + rl) % rl
            x_nx = tl.load(
                ring_x_base + ring_pos * ring_stride_x + offs_m, mask=rmask_x & more, other=0.0
            )
            dt_nx = tl.load(
                ring_dt_base + ring_pos * ring_stride_dt, mask=ring_ok & more, other=0.0
            )
            B_nx = tl.load(
                ring_B_base + ring_pos * ring_stride_B + offs_n, mask=rmask_B & more, other=0.0
            )
            if HAS_DT_BIAS:
                dt += dt_bias_s
            if DT_SOFTPLUS:
                dt = softplus(dt)
            dA = fast_exp(A_s * dt)
            dB = B * dt
            state = state * dA + dB * x[:, None]
    else:
        for i_r in range(num_replay):
            # Non-negative even for garbage metadata (dummy runs), so the address
            # always stays inside slot 1's ring.
            ring_pos = ((pos0 - num_replay + i_r) % rl + rl) % rl
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

    for i_t in range(compute_len):
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

        if LAZY:
            if i_t == next_t:
                # The next checkpoint, in the other slot of the pair.
                next_ptrs = state_ptr_next + (
                    offs_m[:, None] * stride_state_dim
                    + offs_n[None, :] * stride_state_dstate
                )
                tl.store(
                    next_ptrs,
                    state.to(state_ptrs.dtype.element_ty),
                    mask=(offs_m[:, None] < dim)
                    & (offs_n[None, :] < dstate)
                    & (next_idx != null_block_id),
                )
        elif i_t == 0:
            # The new anchor: the state after token 0, as the reference's slot 0.
            tl.store(state_ptrs, state.to(state_ptrs.dtype.element_ty), mask=mask)
        if LAZY or i_t > 0:
            ring_pos = ((pos0 + i_t) % rl + rl) % rl
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

    # Later target layers still receive a fully initialized tensor. Invalid
    # suffix rows may be consumed, but are causal successors of every valid
    # row and are forced rejected by the original confidence-depth sampler.
    # Their ring entries are omitted: next num_accepted <= valid_rows, hence
    # the next replay can read only the prefix entries written above.
    if PREFIX_LIMIT:
        for i_t in range(compute_len, seq_len):
            tl.store(out_ptr + offs_m * stride_out_dim, 0.0, mask=offs_m < dim)
            out_ptr += stride_out_batch


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
    lazy=False,
):
    """
    Spec-decode verify update with anchor + input-ring state handling.

    Arguments as the reference `selective_state_update` in spec mode
    (state_batch_indices: (N, 1 + d) slots, in place; num_accepted_tokens: (N,);
    cu_seqlens: (N + 1,)), plus
        seq_lens: (>= N,) total tokens per request including this verify's,
            so token 0 of request b sits at position seq_lens[b] - query_len[b].
        ring_len: ring entries per request, >= 2 * d + 1 (lazy: >= LAZY_PERIOD + d).
        replayed_out: test hook; if given, an (N, nheads, dim, dstate) fp32
            tensor that receives the state each request starts the verify from.
        lazy: checkpointed anchor (spec_replay_lazy); the caller's prefills
            must have written the prompt length with write_prefill_tag.

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
    if lazy:
        assert max_seqlen >= 3 and ring_len >= LAZY_PERIOD + max_seqlen - 1
        assert prefill_tag_view(state) is not None
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

    verify_rows = None
    if (
        N == 1 and max_seqlen == 5 and batch == 5
        and replayed_out is None
        and os.environ.get("VLLM_MAMBA_SPEC_PREFIX", "1") != "0"
    ):
        smallk = sys.modules.get("smallk")
        if smallk is not None and hasattr(smallk, "live_verify_rows"):
            # Never import/allocate/rebind the buffer here; its address must
            # already be the same one captured by every router and sampler.
            verify_rows = smallk.live_verify_rows(x.device)

    with torch.accelerator.device_index(x.device.index):
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
            RING_LEN=ring_len,
            HG_RATIO=nheads // ngroups,
            CONST_DIV=_const_div_enabled(),
            num_warps=num_warps,
            launch_pdl=_pdl_enabled(),
            verify_rows_ptr=verify_rows,
            PREFIX_LIMIT=verify_rows is not None,
            LAZY=bool(lazy),
            LAZY_P=LAZY_PERIOD,
            SLOT_ELEMS=nheads * dim * dstate,
            PIPELINE=spec_replay_pipeline(),
        )

# SPDX-License-Identifier: Apache-2.0
"""Remote checkpoint-only admission for a larger, same-expression SSM block."""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
_ENABLED = os.environ.get("VLLM_MAMBA_SPEC_MACRO", "1") != "0"


@torch.no_grad()
def qualify_ssm_macro(layer) -> bool:
    """Called during official profile warmup, never with prompt/cache inputs."""
    if (
        not _ENABLED or not layer.use_spec_replay or layer.num_spec != 4
        or layer.head_dim != 64 or layer.ssm_state_size != 128
        or layer.num_heads // layer.tp_size != 64
        or not layer.A.is_cuda
        or torch.cuda.is_current_stream_capturing()
        or torch.cuda.get_device_capability(layer.A.device) != (12, 1)
    ):
        return False
    from .mamba_ssm_spec_replay import selective_state_update_spec_replay

    device = layer.A.device
    heads, dim, dstate = 64, 64, 128
    groups = layer.n_groups // layer.tp_size
    if groups <= 0 or heads % groups:
        return False
    # Only the original anchor/ring slots are addressed. Unused draft state
    # slots need no allocation in this isolated admission sequence.
    state_ref = torch.zeros((2, heads, dim, dstate), dtype=torch.float32, device=device)
    state_new = state_ref.clone()
    replay_ref = torch.empty((1, heads, dim, dstate), dtype=torch.float32, device=device)
    replay_new = torch.empty_like(replay_ref)
    indices = torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.int32, device=device)
    cu = torch.tensor([0, 5], dtype=torch.int32, device=device)
    accepted = torch.ones(1, dtype=torch.int32, device=device)
    seq_lens = torch.empty(1, dtype=torch.int32, device=device)
    live_rows = torch.empty(1, dtype=torch.int32, device=device)
    A = layer.A[:, None, None].expand(heads, dim, dstate).to(torch.float32)
    D = layer.D[:, None].expand(heads, dim)
    dt_bias = layer.dt_bias[:, None].expand(heads, dim)
    ix = torch.arange(5 * heads * dim, dtype=torch.int32, device=device)
    ibc = torch.arange(5 * groups * dstate, dtype=torch.int32, device=device)
    idt = torch.arange(5 * heads, dtype=torch.int32, device=device)
    position, previous_live = 128, 5
    # Exercise every live count, short and long replay, and multiple ring wraps.
    for step, live in enumerate((5, 2, 4, 1, 3, 5, 2, 5, 4, 1)):
        num_accepted = 1 if step == 0 else min(previous_live, 1 + (3 * step) % 5)
        if step:
            position += num_accepted
        accepted.fill_(num_accepted)
        seq_lens.fill_(position + 5)
        live_rows.fill_(live)
        x = ((((ix * 17 + 3 * step) % 67).float() - 33) / 32).to(
            torch.bfloat16).reshape(5, heads, dim)
        if step % 2:
            x = torch.where((ix.reshape(5, heads, dim) % 7) < 3,
                            torch.zeros_like(x), x)
        dt = ((((idt * 7 + step) % 19).float() - 9) / 8).to(
            torch.bfloat16).reshape(5, heads, 1).expand(5, heads, dim)
        B = ((((ibc * 11 + step) % 41).float() - 20) / 64).to(
            torch.bfloat16).reshape(5, groups, dstate)
        C = ((((ibc * 13 + 2 * step) % 43).float() - 21) / 64).to(
            torch.bfloat16).reshape(5, groups, dstate)
        out_ref, out_new = torch.empty_like(x), torch.empty_like(x)
        kwargs = dict(
            dt_softplus=True, state_batch_indices=indices,
            num_accepted_tokens=accepted, cu_seqlens=cu,
            seq_lens=seq_lens, ring_len=layer.spec_replay_ring_len,
            is_blackwell=layer.is_blackwell,
            verify_rows_override=live_rows,
        )
        selective_state_update_spec_replay(
            state_ref, x, dt, A, B, C, D, dt_bias,
            out=out_ref, replayed_out=replay_ref, **kwargs,
        )
        selective_state_update_spec_replay(
            state_new, x, dt, A, B, C, D, dt_bias,
            out=out_new, replayed_out=replay_new,
            geometry_override=(16, 4), **kwargs,
        )
        if not (
            torch.equal(out_ref.view(torch.int16), out_new.view(torch.int16))
            and torch.equal(state_ref.view(torch.int32), state_new.view(torch.int32))
            and torch.equal(replay_ref.view(torch.int32), replay_new.view(torch.int32))
        ):
            logger.warning_once("Mamba macro-block bit comparison declined; original geometry kept")
            return False
        previous_live = live
    logger.info_once("Mamba macro-block admitted: M5, BLOCK_SIZE_M=16, warps=4")
    return True

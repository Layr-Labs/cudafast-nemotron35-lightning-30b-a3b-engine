# SPDX-License-Identifier: Apache-2.0
"""Native-MTP metadata update and current token/hidden staging in one launch.

Only private official-worker qualification executes this module's preparation.
Unsupported shapes and a failed full byte comparison retain the original path.
"""
import os

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.spec_decode.utils import (
    PADDING_SLOT_ID,
    eagle_step_slot_mapping_metadata_kernel,
    eagle_step_update_slot_mapping_and_metadata,
)


@triton.jit
def _stage(POS, BT, SEQ, OUT_POS, SLOT, TOKEN, OUT_TOKEN, HIDDEN, OUT_HIDDEN,
           H: tl.constexpr, BLOCK_SIZE: tl.constexpr, MAX_LEN: tl.constexpr,
           N_BLOCKS: tl.constexpr, PADDED: tl.constexpr, B: tl.constexpr):
    # Reuse the original integer metadata body, including clamp/overflow and
    # block-number bounds. This launch has one real request (program zero).
    eagle_step_slot_mapping_metadata_kernel(
        POS, BT, N_BLOCKS, SEQ, OUT_POS, SLOT, BLOCK_SIZE, MAX_LEN,
        N_BLOCKS, PADDING_SLOT_ID, 1,
    )
    o = tl.arange(0, B)
    # Move hidden payload as integers: preserve every BF16 bit, including
    # subnormals, signed zero and nonfinite payloads, without FP conversion.
    bits = tl.load(HIDDEN.to(tl.pointer_type(tl.uint16)) + o, mask=o < H, other=0)
    tl.store(OUT_HIDDEN.to(tl.pointer_type(tl.uint16)) + o, bits, mask=o < H)
    token = tl.load(TOKEN)
    tl.store(OUT_TOKEN, token)
    tl.store(SLOT + o, PADDING_SLOT_ID, mask=(o > 0) & (o < PADDED))


def _configuration(p):
    if (os.environ.get("NEMO_MTP_STEP_STAGE", "1") == "0"
            or type(p.model).__name__ != "NemotronHMTP" or p.method != "mtp"
            or p.parallel_drafting or p.constant_draft_positions
            or p.use_heterogeneous_vocab or p.supports_mm_inputs
            or p.uses_mrope or p.uses_xdrope_dim
            or not p.pass_hidden_states_to_model or p.needs_extra_input_slots
            or p.hidden_states.dtype != torch.bfloat16
            or p.input_ids.dtype != torch.int32 or p.positions.dtype != torch.int64
            or p.hidden_states.ndim != 2 or p.hidden_states.stride(1) != 1
            or p.block_size <= 0):
        return None
    h = p.hidden_states.shape[1]
    if h != 2688:
        return None
    blocks = triton.cdiv(p.max_model_len, p.block_size)
    return (h, p.block_size, p.max_model_len, blocks)


def _launch(cfg, pos, table, seq, out_pos, slots, token, out_token, hidden, out_hidden,
            padded):
    h, block, maximum, blocks = cfg
    _stage[(1,)](
        pos, table, seq, out_pos, slots, token, out_token, hidden, out_hidden,
        H=h, BLOCK_SIZE=block, MAX_LEN=maximum, N_BLOCKS=blocks,
        PADDED=padded, B=triton.next_power_of_2(h), num_warps=4,
    )


@torch.no_grad()
def prepare(p):
    """Private integer/byte comparisons on the official remote worker only."""
    if getattr(p, "_nemo_step_stage_attempted", False):
        return
    if torch.cuda.is_current_stream_capturing():
        return
    cfg = _configuration(p)
    if cfg is None:
        return
    p._nemo_step_stage_attempted = True
    p._nemo_step_stage_ready = None
    try:
        h, block, maximum, blocks = cfg
        table = (torch.arange(blocks, device=p.device, dtype=torch.int32) * 7 + 2).view(1, -1)
        token = torch.tensor([104729], device=p.device, dtype=torch.int32)
        # Private varied BF16 payloads with explicit special-value bits;
        # none of these values is a request input, proposal or fixture token.
        hidden = (torch.arange(h, device=p.device, dtype=torch.int32) * 41).to(
            torch.uint16).view(torch.bfloat16).view(1, h)
        hidden.view(torch.uint16)[:, :12].copy_(torch.tensor(
            [0, 0x8000, 1, 0x8001, 128, 0x8080, 0x7F80, 0xFF80,
             0x7FC1, 0xFFC1, 0x3F80, 0xBF80], device=p.device,
            dtype=torch.int32).to(torch.uint16))
        for padded in (1, 5):
            for alias in (False, True):
                for position in (0, maximum - 2, maximum - 1, maximum + 1):
                    ref_pos = torch.full((5,), position, dtype=torch.int64, device=p.device)
                    got_pos = ref_pos.clone()
                    pos = torch.tensor([position], dtype=torch.int64, device=p.device)
                    ref_seq = torch.tensor([maximum - 1], dtype=torch.int32, device=p.device)
                    got_seq = ref_seq.clone()
                    ref_slot = torch.full((5,), -31, dtype=torch.int64, device=p.device)
                    got_slot = ref_slot.clone()
                    ref_token = torch.full((5,), -37, dtype=torch.int32, device=p.device)
                    got_token = ref_token.clone()
                    ref_hidden = torch.zeros((5, h), dtype=torch.bfloat16, device=p.device)
                    got_hidden = ref_hidden.clone()
                    eagle_step_update_slot_mapping_and_metadata(
                        ref_pos[:1] if alias else pos, table, ref_seq, block, maximum,
                        ref_pos[:1], ref_slot[:padded], input_batch_size=padded,
                    )
                    ref_token[:1].copy_(token)
                    ref_hidden[:1].copy_(hidden)
                    _launch(cfg, got_pos[:1] if alias else pos, table, got_seq,
                            got_pos, got_slot, token, got_token, hidden, got_hidden, padded)
                    for a, b in ((ref_pos, got_pos), (ref_seq, got_seq),
                                 (ref_slot, got_slot), (ref_token, got_token),
                                 (ref_hidden.view(torch.uint8), got_hidden.view(torch.uint8))):
                        if not torch.equal(a, b):
                            raise ValueError("private native-MTP metadata/staging byte mismatch")
        torch.cuda.synchronize(p.device)
        p._nemo_step_stage_ready = cfg
        print("NEMO_MTP_STEP_STAGE private_metadata_and_payload_bits_passed", flush=True)
    except Exception as exc:
        p._nemo_step_stage_ready = None
        print(f"NEMO_MTP_STEP_STAGE original path retained: {exc!r}", flush=True)


def maybe_run(p, positions, token, hidden, common, batch, padded):
    cfg = getattr(p, "_nemo_step_stage_ready", None)
    if cfg is None or batch != 1 or padded not in (1, 5):
        return False
    table, seq = common.block_table_tensor, common.seq_lens
    if (positions.shape != (1,) or positions.dtype != torch.int64
            or token.shape != (1,) or token.dtype != torch.int32
            or hidden.shape != (1, cfg[0]) or hidden.dtype != torch.bfloat16
            or hidden.stride(1) != 1 or not hidden.is_cuda
            or table.shape != (1, cfg[3]) or table.dtype != torch.int32
            or not table.is_contiguous() or seq.shape != (1,) or seq.dtype != torch.int32
            or not (positions.device == token.device == hidden.device == table.device
                    == seq.device == p.hidden_states.device)):
        return False
    _launch(cfg, positions, table, seq, p.positions, p._slot_mapping_buffer,
            token, p.input_ids, hidden, p.hidden_states, padded)
    return True

"""LOCAL PATCH (cudafast-nemotron35): the MTP draft loop's small input moves in
one launch each.

Between the drafter's passes the stock loop issues one GPU operation per tensor
it moves: after the first pass two int32->int64 index casts and three gathers
(the sampled row of the hidden states twice and its position), and before every
later pass an int64->int32 cast of the drafted token, the fused slot-mapping /
position / seq_lens step kernel and two copies (token id, hidden row). Each is a
1-3 us kernel or copy plus ~3 us of serialized launch gap on GB10.

The two kernels below do exactly those moves: integer casts, integer adds and
byte-for-byte row copies, the same values into the same buffers.
``eagle_step_slot_mapping_metadata_kernel``'s arithmetic is reproduced line by
line. VLLM_MTP_FUSED_GLUE=0 keeps the stock operations.
"""
from __future__ import annotations

import os

import torch

from vllm.triton_utils import tl, triton

ENABLED = os.environ.get("VLLM_MTP_FUSED_GLUE", "1") != "0"


@triton.jit
def _gather_rows_kernel(idx_ptr, src_a, dst_a, src_b, dst_b, pos_src, pos_dst,
                        stride_a, stride_b, HID: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    r = tl.load(idx_ptr + b).to(tl.int64)
    o = tl.arange(0, BLOCK)
    for off in range(0, HID, BLOCK):
        m = off + o < HID
        tl.store(dst_a + b * HID + off + o, tl.load(src_a + r * stride_a + off + o, mask=m), mask=m)
        tl.store(dst_b + b * HID + off + o, tl.load(src_b + r * stride_b + off + o, mask=m), mask=m)
    tl.store(pos_dst + b, tl.load(pos_src + r))


def gather_first_pass(idx: torch.Tensor, last_hidden: torch.Tensor, hidden: torch.Tensor,
                      positions: torch.Tensor):
    """(last_hidden[idx], positions[idx], hidden[idx]) in one launch."""
    n = idx.shape[0]
    hid = last_hidden.shape[1]
    out_a = torch.empty((n, hid), dtype=last_hidden.dtype, device=last_hidden.device)
    out_b = torch.empty((n, hid), dtype=hidden.dtype, device=hidden.device)
    out_p = torch.empty((n,), dtype=positions.dtype, device=positions.device)
    _gather_rows_kernel[(n,)](idx, last_hidden, out_a, hidden, out_b, positions, out_p,
                              last_hidden.stride(0), hidden.stride(0), HID=hid,
                              BLOCK=1024, num_warps=4)
    return out_a, out_p, out_b


def gather_ok(idx, last_hidden, hidden, positions) -> bool:
    return (ENABLED and idx.is_cuda and idx.dim() == 1
            and idx.dtype in (torch.int32, torch.int64)
            and last_hidden.dim() == 2 and hidden.dim() == 2
            and last_hidden.shape[1] == hidden.shape[1]
            and last_hidden.stride(1) == 1 and hidden.stride(1) == 1
            and last_hidden.dtype == hidden.dtype
            and positions.dim() == 1 and positions.is_contiguous())


@triton.jit
def _step_inputs_kernel(tok_ptr, ids_ptr, hsrc, hdst, hsrc_stride,
                        positions_ptr, block_table_ptr, block_table_stride, seq_lens_ptr,
                        out_clamped_positions_ptr, out_slot_mapping_ptr, batch_size,
                        block_size: tl.constexpr, max_model_len: tl.constexpr,
                        n_blocks_per_req: tl.constexpr, PAD_ID: tl.constexpr,
                        HID: tl.constexpr, BLOCK: tl.constexpr):
    req_idx = tl.program_id(0)
    if req_idx >= batch_size:
        tl.store(out_slot_mapping_ptr + req_idx, PAD_ID)
        return
    tl.store(ids_ptr + req_idx, tl.load(tok_ptr + req_idx).to(tl.int32))
    o = tl.arange(0, BLOCK)
    for off in range(0, HID, BLOCK):
        m = off + o < HID
        tl.store(hdst + req_idx * HID + off + o,
                 tl.load(hsrc + req_idx * hsrc_stride + off + o, mask=m), mask=m)
    position = tl.load(positions_ptr + req_idx)
    new_position = position + 1
    exceeds_max = new_position >= max_model_len
    clamped_position = tl.where(exceeds_max, 0, new_position)
    block_number = clamped_position // block_size
    block_number = tl.minimum(block_number, n_blocks_per_req - 1)
    block_id = tl.load(block_table_ptr + req_idx * block_table_stride + block_number)
    slot_id = block_id * block_size + (clamped_position % block_size)
    slot_id = tl.where(exceeds_max, PAD_ID, slot_id)
    seq_len = tl.load(seq_lens_ptr + req_idx)
    new_seq_len = tl.where(exceeds_max, 1, seq_len + 1)
    new_seq_len = tl.minimum(new_seq_len, max_model_len)
    tl.store(out_clamped_positions_ptr + req_idx, clamped_position)
    tl.store(out_slot_mapping_ptr + req_idx, slot_id)
    tl.store(seq_lens_ptr + req_idx, new_seq_len)


def step_inputs(tok, ids_dst, hidden_src, hidden_dst, positions_1d, block_table_tensor,
                seq_lens, block_size, max_model_len, out_clamped_positions,
                out_slot_mapping, input_batch_size, pad_id):
    batch_size = positions_1d.shape[0]
    hid = hidden_dst.shape[1]
    _step_inputs_kernel[(input_batch_size,)](
        tok, ids_dst, hidden_src, hidden_dst, hidden_src.stride(0),
        positions_1d, block_table_tensor, block_table_tensor.stride(0), seq_lens,
        out_clamped_positions, out_slot_mapping, batch_size,
        block_size=block_size, max_model_len=max_model_len,
        n_blocks_per_req=block_table_tensor.shape[1], PAD_ID=pad_id,
        HID=hid, BLOCK=1024, num_warps=4)


def step_ok(tok, ids_buf, hidden_src, hidden_buf, positions_1d, batch_size) -> bool:
    return (ENABLED and tok.is_cuda and tok.dim() == 1 and tok.shape[0] == batch_size
            and tok.dtype in (torch.int64, torch.int32)
            and ids_buf.dtype == torch.int32 and ids_buf.is_contiguous()
            and hidden_src.dim() == 2 and hidden_src.shape[0] >= batch_size
            and hidden_src.stride(1) == 1 and hidden_src.dtype == hidden_buf.dtype
            and hidden_buf.is_contiguous() and hidden_buf.shape[1] == hidden_src.shape[1]
            and positions_1d.dim() == 1 and positions_1d.shape[0] == batch_size)


@triton.jit
def _first_pass_inputs_kernel(tgt_ids, next_ids, idx_ptr, ids_dst, tgt_pos, pos_dst,
                              tgt_hidden, hid_dst, hid_stride, num_tokens, batch_size,
                              HID: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    write = t < num_tokens - 1
    tn = tl.minimum(t + 1, num_tokens - 1)
    val = tl.load(tgt_ids + tn).to(tl.int32)
    for b in range(batch_size):
        hit = tl.load(idx_ptr + b).to(tl.int64) == t
        val = tl.where(hit, tl.load(next_ids + b).to(tl.int32), val)
        write = write | hit
    if write:
        tl.store(ids_dst + t, val)
    tl.store(pos_dst + t, tl.load(tgt_pos + t))
    o = tl.arange(0, BLOCK)
    for off in range(0, HID, BLOCK):
        m = off + o < HID
        tl.store(hid_dst + t * HID + off + o,
                 tl.load(tgt_hidden + t * hid_stride + off + o, mask=m), mask=m)


def first_pass_ok(target_token_ids, next_token_ids, idx, target_positions,
                  target_hidden_states, ids_buf, pos_buf, hidden_buf) -> bool:
    n = target_token_ids.shape[0] if target_token_ids.dim() == 1 else -1
    return (ENABLED and n >= 1 and target_token_ids.is_contiguous()
            and target_token_ids.dtype in (torch.int32, torch.int64)
            and next_token_ids.dim() == 1 and next_token_ids.is_contiguous()
            and next_token_ids.dtype in (torch.int32, torch.int64)
            and idx.dim() == 1 and idx.shape[0] == next_token_ids.shape[0]
            and idx.dtype in (torch.int32, torch.int64) and idx.shape[0] <= 8
            and target_positions.dim() == 1 and target_positions.shape[0] == n
            and target_positions.is_contiguous() and target_positions.dtype == pos_buf.dtype
            and target_hidden_states.dim() == 2 and target_hidden_states.shape[0] == n
            and target_hidden_states.stride(1) == 1
            and target_hidden_states.dtype == hidden_buf.dtype
            and hidden_buf.is_contiguous() and hidden_buf.shape[1] == target_hidden_states.shape[1]
            and ids_buf.dtype == torch.int32 and ids_buf.is_contiguous()
            and pos_buf.is_contiguous())


def first_pass_inputs(target_token_ids, next_token_ids, idx, target_positions,
                      target_hidden_states, ids_buf, pos_buf, hidden_buf) -> None:
    n = target_token_ids.shape[0]
    _first_pass_inputs_kernel[(n,)](
        target_token_ids, next_token_ids, idx, ids_buf, target_positions, pos_buf,
        target_hidden_states, hidden_buf, target_hidden_states.stride(0), n,
        idx.shape[0], HID=hidden_buf.shape[1], BLOCK=1024, num_warps=4)

# SPDX-License-Identifier: Apache-2.0
"""Shared device predicate for a proposed convolution-to-SSM ring producer.

A single request, non-aliasing ring and bounded modulo intervals are required;
any uncertainty must use the original convolution output and SSM ring writes.
"""
from vllm.triton_utils import tl, triton


@triton.jit
def conv_ring_window(
    state_ptr,
    state_indices_ptr,
    accepted_ptr,
    cu_ptr,
    seq_ptr,
    verify_rows_ptr,
    STRIDE_STATE_BATCH: tl.constexpr,
    STRIDE_INDEX_TOKEN: tl.constexpr,
    SLOT_ELEMS: tl.constexpr,
    STATE_SLOTS: tl.constexpr,
    NULL_BLOCK: tl.constexpr,
    RING_LEN: tl.constexpr,
    MAX_ROWS: tl.constexpr,
    LAZY: tl.constexpr,
    LAZY_P: tl.constexpr,
    PREFIX_LIMIT: tl.constexpr,
):
    # Only the one-request structural variant may call this helper.
    bos = tl.load(cu_ptr).to(tl.int64)
    eos = tl.load(cu_ptr + 1).to(tl.int64)
    length = eos - bos
    pos0 = tl.load(seq_ptr).to(tl.int64) - length
    valid = length
    if PREFIX_LIMIT:
        valid = tl.minimum(length, tl.maximum(tl.load(verify_rows_ptr).to(tl.int64), 1))

    anchor = tl.load(state_indices_ptr).to(tl.int64)
    ring = tl.load(state_indices_ptr + STRIDE_INDEX_TOKEN).to(tl.int64)
    safe_anchor = (anchor >= 0) & (anchor < STATE_SLOTS) & (anchor != NULL_BLOCK)
    safe_ring = (ring >= 0) & (ring < STATE_SLOTS) & (ring != NULL_BLOCK)
    separate = anchor != ring

    if LAZY:
        other = tl.load(state_indices_ptr + 2 * STRIDE_INDEX_TOKEN).to(tl.int64)
        safe_other = (other >= 0) & (other < STATE_SLOTS) & (other != NULL_BLOCK)
        separate &= (ring != other) & (other != anchor)
        tag = (state_ptr + anchor * STRIDE_STATE_BATCH + SLOT_ELEMS).to(
            tl.pointer_type(tl.int32)
        )
        prompt = tl.load(tag, mask=safe_anchor, other=0).to(tl.int64)
        checkpoint_index = tl.maximum(pos0 - prompt, 0) // LAZY_P
        checkpoint = prompt - 1 + checkpoint_index * LAZY_P
        replay = tl.minimum(tl.maximum(pos0 - 1 - checkpoint, 0), LAZY_P - 1)
        safe_metadata = safe_other & (prompt >= 0)
    else:
        accepted = tl.load(accepted_ptr).to(tl.int64)
        replay = tl.maximum(accepted - 1, 0)
        safe_metadata = (accepted >= 1) & (accepted <= MAX_ROWS)

    # The historical interval [pos0-replay,pos0) and current interval
    # [pos0,pos0+valid) are disjoint modulo the ring under this inequality.
    use = (safe_anchor & safe_ring & separate & safe_metadata
           & (bos == 0) & (length > 0) & (length <= MAX_ROWS)
           & (pos0 >= 0) & (valid >= 1) & (replay + valid <= RING_LEN))
    return use, ring, pos0, valid

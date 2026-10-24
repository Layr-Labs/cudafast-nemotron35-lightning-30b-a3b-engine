# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact greedy speculative sampling without vocabulary-row temporaries.

Reduce vocabulary tiles in parallel, then combine their maxima, gather the
target/bonus row indices, and apply first-rejection acceptance in one launch.
No probability, model arithmetic, or draft acceptance threshold is changed.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


@triton.jit
def _vocab_tile_argmax(
    logits_ptr,
    values_ptr,
    indices_ptr,
    stride_row,
    stride_col,
    VOCAB: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    valid = cols < VOCAB
    values = tl.load(
        logits_ptr + row * stride_row + cols * stride_col,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    # torch.argmax picks the first NaN, otherwise the first maximum. Handle
    # this explicitly so ties, infinities, and an all-masked vocabulary retain
    # the stock reduction's ordering, independent of the partition geometry.
    nan_indices = tl.where(valid & (values != values), cols, 2147483647)
    first_nan = tl.min(nan_indices, axis=0)
    maximum = tl.max(tl.where(values == values, values, -float("inf")), axis=0)
    first_max = tl.min(
        tl.where(valid & (values == maximum), cols, 2147483647), axis=0
    )
    has_nan = first_nan != 2147483647
    best_index = tl.where(has_nan, first_nan, first_max)
    best_value = tl.where(has_nan, float("nan"), maximum)
    tl.store(values_ptr + row * SPLITS + tile, best_value)
    tl.store(indices_ptr + row * SPLITS + tile, best_index)


@triton.jit
def _finish_greedy_rejection(
    values_ptr,
    indices_ptr,
    target_rows_ptr,
    bonus_rows_ptr,
    cu_drafts_ptr,
    drafts_ptr,
    output_ptr,
    MAX_SPEC: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    req = tl.program_id(0)
    start = tl.load(cu_drafts_ptr + req - 1, mask=req > 0, other=0)
    end = tl.load(cu_drafts_ptr + req)
    num_drafts = end - start
    positions = tl.arange(0, BLOCK_TOKENS)
    valid = positions <= num_drafts
    target_rows = tl.load(
        target_rows_ptr + start + positions, mask=positions < num_drafts, other=0
    )
    bonus_row = tl.load(bonus_rows_ptr + req)
    rows = tl.where(positions == num_drafts, bonus_row, target_rows)
    tiles = tl.arange(0, BLOCK_SPLITS)
    partial_valid = valid[:, None] & (tiles[None, :] < SPLITS)
    offsets = rows[:, None] * SPLITS + tiles[None, :]
    values = tl.load(values_ptr + offsets, mask=partial_valid, other=-float("inf"))
    indices = tl.load(indices_ptr + offsets, mask=partial_valid, other=2147483647)
    first_nan = tl.min(
        tl.where(partial_valid & (values != values), indices, 2147483647), axis=1
    )
    maximum = tl.max(tl.where(values == values, values, -float("inf")), axis=1)
    first_max = tl.min(
        tl.where(partial_valid & (values == maximum[:, None]), indices, 2147483647),
        axis=1,
    )
    token_ids = tl.where(first_nan != 2147483647, first_nan, first_max)
    draft_ids = tl.load(
        drafts_ptr + start + positions, mask=positions < num_drafts, other=-1
    )
    first_rejected = tl.min(
        tl.where(
            (positions < num_drafts) & (draft_ids != token_ids), positions, 2147483647
        ),
        axis=0,
    )
    # The target replacement at the first rejection is committed. Later
    # positions are placeholders. With no rejection, include the bonus row.
    committed = valid & (positions <= first_rejected)
    tl.store(
        output_ptr + req * (MAX_SPEC + 1) + positions,
        tl.where(committed, token_ids, -1),
        mask=positions < MAX_SPEC + 1,
    )


def fused_greedy_rejection(
    logits: torch.Tensor, metadata: SpecDecodeMetadata
) -> torch.Tensor:
    rows, vocab = logits.shape
    block = 4096
    splits = triton.cdiv(vocab, block)
    partial_values = torch.empty((rows, splits), device=logits.device, dtype=torch.float32)
    partial_indices = torch.empty((rows, splits), device=logits.device, dtype=torch.int32)
    output = torch.empty(
        (len(metadata.num_draft_tokens), metadata.max_spec_len + 1),
        device=logits.device,
        dtype=torch.int32,
    )
    _vocab_tile_argmax[(rows, splits)](
        logits,
        partial_values,
        partial_indices,
        logits.stride(0),
        logits.stride(1),
        VOCAB=vocab,
        SPLITS=splits,
        BLOCK=block,
        num_warps=4,
    )
    _finish_greedy_rejection[(len(metadata.num_draft_tokens),)](
        partial_values,
        partial_indices,
        metadata.target_logits_indices,
        metadata.bonus_logits_indices,
        metadata.cu_num_draft_tokens,
        metadata.draft_token_ids,
        output,
        MAX_SPEC=metadata.max_spec_len,
        SPLITS=splits,
        BLOCK_SPLITS=triton.next_power_of_2(splits),
        BLOCK_TOKENS=triton.next_power_of_2(metadata.max_spec_len + 1),
        num_warps=4,
    )
    return output

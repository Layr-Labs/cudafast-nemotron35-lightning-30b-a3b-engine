# SPDX-License-Identifier: Apache-2.0
"""Partition existing 64-row expert tiles into disjoint 32/64-row plans.

Only padding rows are dropped. Every live route keeps its flattened row ID,
expert and full K reduction; each old weight tile is visited once in total.
There is no device-to-host count read and no change to weight representation.
"""

import copy
import json

import torch

from vllm.triton_utils import tl, triton


def make_prefill_row_configs(w13_table, w2_table):
    """Derive narrow configs only from the known BI prefill pair."""
    narrow = []
    for table, expected_n, expected_warp_n in (
        (w13_table, 64, 32),
        (w2_table, 128, 64),
    ):
        if not table:
            return None
        lo, hi, config = table[-1]
        if not (
            lo == 384
            and hi >= 6144
            and config["block_shape"] == [64, expected_n, 32]
            and config["warp_shape"] == [32, expected_warp_n, 32]
            and config["use_stream_k"] is False
            and config["use_f16_accum"] is False
            and config["num_write_splits"] == 1
        ):
            return None
        small = copy.deepcopy(config)
        small["block_shape"][0] = 32
        small["warp_shape"][0] = 16
        narrow.append(json.dumps([[0, hi, small]]))
    return tuple(narrow)


@triton.jit
def _partition_prefill_rows(
    source_rows,
    source_experts,
    source_count,
    small_rows,
    small_experts,
    large_rows,
    large_experts,
    counts,
    ROUTES: tl.constexpr,
    BLOCKS: tl.constexpr,
):
    block = tl.program_id(0)
    nblocks = tl.load(source_count) // 64
    if block >= nblocks:
        if block == 0:
            tl.store(counts, 0)
            tl.store(counts + 1, 0)
        return

    # Native alignment fills live IDs first and pads the suffix with ROUTES.
    # The 33rd ID therefore determines whether a block needs its second half.
    blocks = tl.arange(0, BLOCKS)
    middle = tl.load(
        source_rows + blocks * 64 + 32, mask=blocks < nblocks, other=ROUTES
    )
    short = (blocks < nblocks) & (middle >= ROUTES)
    before = tl.sum((short & (blocks < block)).to(tl.int32), 0)
    nshort = tl.sum(short.to(tl.int32), 0)
    if block == 0:
        tl.store(counts, nshort * 32)
        tl.store(counts + 1, (nblocks - nshort) * 64)

    slots = tl.arange(0, 64)
    rows = tl.load(source_rows + block * 64 + slots)
    expert = tl.load(source_experts + block)
    this_middle = tl.load(source_rows + block * 64 + 32)
    if this_middle >= ROUTES:
        tl.store(small_rows + before * 32 + slots, rows, mask=slots < 32)
        tl.store(small_experts + before, expert)
    else:
        destination = block - before
        tl.store(large_rows + destination * 64 + slots, rows)
        tl.store(large_experts + destination, expert)


def prefill_row_split_supported(
    topk_ids,
    hidden_states,
    expert_map,
    global_experts,
    local_experts,
    narrow_configs,
    valid_shape_m,
):
    return (
        narrow_configs is not None
        and expert_map is None
        and global_experts == local_experts == 128
        and topk_ids.ndim == 2
        and topk_ids.shape[1] == 6
        and 64 < topk_ids.shape[0] <= 1024
        and topk_ids.is_cuda
        and topk_ids.dtype == torch.int32
        and topk_ids.is_contiguous()
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.ndim == 2
        and hidden_states.shape[1] == 2688
        and hidden_states.shape[0] == topk_ids.shape[0]
        and hidden_states.device == topk_ids.device
        and valid_shape_m == topk_ids.numel()
    )


def split_prefill_row_plans(w13_kwargs, w2_kwargs, routes, narrow_configs):
    """Reuse native alignment; return two disjoint plans on the same outputs."""
    original = w13_kwargs["sorted_ids"]
    experts = w13_kwargs["expert_ids"]
    count = w13_kwargs["num_tokens_padded"]
    capacity = experts.numel()
    # All buffers are invocation-local and consumed on the caller's stream.
    small_rows = torch.empty(capacity * 32, dtype=torch.int32, device=original.device)
    large_rows = torch.empty(capacity * 64, dtype=torch.int32, device=original.device)
    small_experts = torch.empty_like(experts)
    large_experts = torch.empty_like(experts)
    counts = torch.empty(2, dtype=torch.int32, device=original.device)
    _partition_prefill_rows[(capacity,)](
        source_rows=original,
        source_experts=experts,
        source_count=count,
        small_rows=small_rows,
        small_experts=small_experts,
        large_rows=large_rows,
        large_experts=large_experts,
        counts=counts,
        ROUTES=routes,
        BLOCKS=triton.next_power_of_2(capacity),
        num_warps=4,
    )
    plans = []
    for rows, ids, size, small in (
        (small_rows, small_experts, counts[:1], True),
        (large_rows, large_experts, counts[1:], False),
    ):
        pair = []
        for index, kwargs in enumerate((w13_kwargs, w2_kwargs)):
            plan = dict(kwargs)
            plan.update(sorted_ids=rows, expert_ids=ids, num_tokens_padded=size)
            if small:
                plan["tuning_config"] = narrow_configs[index]
            pair.append(plan)
        plans.append(tuple(pair))
    return tuple(plans)

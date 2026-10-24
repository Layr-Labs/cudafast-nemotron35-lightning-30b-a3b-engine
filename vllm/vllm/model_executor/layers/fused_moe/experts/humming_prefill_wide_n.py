# SPDX-License-Identifier: Apache-2.0
"""Shape-only N tiling for the 496x6 indexed prefill down projection.

The live route IDs, M64 alignment, K32 warp/block chain and all split settings
stay fixed. Only independent output-column scheduling changes. No tensors,
device reads, dynamic tuning or request-derived state are created here.
"""

from copy import deepcopy


ROUTES = 496 * 6


def widen_prefill_w2_n(w13_table: list, w2_table: list) -> tuple[list, list]:
    """Return independent tables, changing only valid_shape_m == ROUTES.

    Called while constructing the expert backend, before boot warmup/capture.
    Unknown or changed baseline configurations retain their original schedule.
    The caller additionally checks the device and exact layer/format metadata.
    """
    if not w13_table or len(w13_table) != len(w2_table):
        return w13_table, w2_table
    if any(
        a[:2] != b[:2] or a[2]["block_shape"][0] != b[2]["block_shape"][0]
        for a, b in zip(w13_table, w2_table)
    ):
        return w13_table, w2_table
    candidates = [i for i, (lo, hi, _) in enumerate(w2_table) if lo < ROUTES <= hi]
    if len(candidates) != 1:
        return w13_table, w2_table
    idx = candidates[0]
    lo, hi, old = w2_table[idx]
    up = w13_table[idx][2]
    invariant = {
        "use_stream_k": False,
        "use_f16_accum": False,
        "num_write_splits": 1,
        "num_sms": 48,
    }
    # Keep the established depth/decode and other-prefill tables untouched.
    # Type-sensitive checks also reject integer substitutes for boolean pins.
    for config in (up, old):
        if any(
            type(config.get(key)) is not type(value) or config.get(key) != value
            for key, value in invariant.items()
        ):
            return w13_table, w2_table
    if not (
        lo == 384
        and hi > ROUTES
        and up["block_shape"] == [64, 64, 32]
        and up["warp_shape"] == [32, 32, 32]
        and up["num_stages"] == 4
        and up["num_ctas_per_sm"] == 4
        and old["block_shape"] == [64, 128, 32]
        and old["warp_shape"] == [32, 64, 32]
        and old["num_stages"] == 4
        and old["num_ctas_per_sm"] == 2
    ):
        return w13_table, w2_table

    wide = deepcopy(old)
    wide["block_shape"] = [64, 384, 32]
    wide["warp_shape"] = [64, 64, 32]
    result = []
    for table, selected in ((w13_table, up), (w2_table, wide)):
        copied = deepcopy(table)
        base_config = copied[idx][2]
        copied[idx : idx + 1] = [
            [lo, ROUTES - 1, deepcopy(base_config)],
            [ROUTES - 1, ROUTES, deepcopy(selected)],
            [ROUTES, hi, deepcopy(base_config)],
        ]
        result.append(copied)
    return result[0], result[1]

# SPDX-License-Identifier: Apache-2.0
# LOCAL PATCH (cudafast-nemotron35, vllm/VENDOR.json "local_patches").
"""Pin the Mamba2 SSD Triton kernels to ONE tile config each.

`@triton.autotune` benchmarks every candidate config at first use and keeps the
fastest. On GB10 the winner varies from boot to boot (timing noise decides
between near-equal candidates), and different tiles sum in a different order,
so the SAME prompt produced one of two token streams depending on the boot --
fatal for a benchmark that checks every leg against a recorded tape.

Each kernel is pinned to a config from its own candidate list: the one the
autotuner chose unanimously or most often across four Sparks. A single-config
autotuner does not benchmark at all, so the choice can no longer drift.
Set VLLM_MAMBA_SSD_AUTOTUNE=1 to restore upstream autotuning.
"""

import os

from vllm.triton_utils import HAS_TRITON


def pin(kernel, **want) -> None:
    """Keep only the candidate of `kernel` (a triton Autotuner) whose kwargs
    and launch parameters equal `want`. Refuses a pin that matches nothing, so
    an upstream config change cannot silently fall back to autotuning."""
    if os.environ.get("VLLM_MAMBA_SSD_AUTOTUNE", "0") == "1":
        return
    # Without an active Triton driver (a host with no GPU, such as the hosted
    # CI import check) vllm.triton_utils swaps in placeholder decorators, so
    # `kernel` is a plain function with no candidates: nothing can run, and
    # nothing needs a pin. The GPU path below is unchanged.
    if not HAS_TRITON:
        return
    launch = ("num_warps", "num_stages")
    matches = [
        c
        for c in kernel.configs
        if all(c.kwargs.get(k) == v for k, v in want.items() if k not in launch)
        and all(getattr(c, k) == v for k, v in want.items() if k in launch)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"ssd_pinned_configs: {want} matches {len(matches)} configs of "
            f"{getattr(kernel, 'fn', kernel)}; update the pin with the vendored tree"
        )
    kernel.configs = matches

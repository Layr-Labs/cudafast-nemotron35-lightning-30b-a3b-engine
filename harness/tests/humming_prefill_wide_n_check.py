#!/usr/bin/env python3
"""Host-only native compile, byte parity and graph timing against the base table.

Reuses the promoted exact-check harness but both arms use unpadded base weights.
The reference arm only disables the new N-tile transform during construction.
No numerical model emulation, broad sweep, or exhaustive MMA phase is run here.
Run inside the pinned image with NEMO_SNAPSHOT and a chosen --out directory.
"""

import importlib.util
from pathlib import Path
import sys


def main():
    path = Path(__file__).with_name("humming_moe_exact.py")
    spec = importlib.util.spec_from_file_location("base_humming_check", path)
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    from vllm.model_executor.layers.fused_moe.experts import humming_tuned_moe
    from vllm.platforms import current_platform

    if not current_platform.is_device_capability(121):
        raise SystemExit(
            "This candidate requires SM121; refusing a fallback-only check"
        )

    original_build = check.build_variant
    transform = humming_tuned_moe.widen_prefill_w2_n

    def build_pair(raw, prefix, tuned):
        try:
            if not tuned:
                humming_tuned_moe.widen_prefill_w2_n = lambda up, down: (up, down)
            ex = original_build(raw, prefix, True)
            chosen = next(cfg for lo, hi, cfg in ex.w2_tuning_config if lo < 2976 <= hi)
            expected = [64, 384 if tuned else 128, 32]
            if chosen["block_shape"] != expected:
                raise RuntimeError(
                    f"Unexpected {'candidate' if tuned else 'base'} config: {chosen}"
                )
            return ex
        finally:
            humming_tuned_moe.widen_prefill_w2_n = transform

    check.build_variant = build_pair
    original_exact = check.phase_exact

    def exact(args, ctx):
        if not {4, 495, 496, 497}.issubset(args.ms) or args.reps < 1:
            raise SystemExit("Include --ms 4,495,496,497 and use positive --reps")
        # The inherited exact phase exits 3 on any byte mismatch, before time.
        return original_exact(args, ctx)

    check.phase_exact = exact
    check.log("Reference 'stock' = promoted base table; 'tuned' = wide-N candidate")
    # Force a bounded check; optional timing arguments remain configurable.
    if any(arg == "--phase" or arg.startswith("--phase=") for arg in sys.argv[1:]):
        raise SystemExit("This check fixes --phase to exact,time")
    sys.argv.extend(["--phase", "exact,time"])
    if not any(arg == "--ms" or arg.startswith("--ms=") for arg in sys.argv[1:]):
        sys.argv.extend(["--ms", "4,495,496,497"])
    check.main()


if __name__ == "__main__":
    main()

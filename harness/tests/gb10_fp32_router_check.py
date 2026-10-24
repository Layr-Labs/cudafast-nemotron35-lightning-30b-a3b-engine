"""Host-only GB10 router numerical/graph probe; no model or timing result.

Synthetic routing checks cannot replace the complete timed-token oracle.
"""

import argparse
import json
from pathlib import Path
import sys


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vllm"))
    import torch
    from vllm.model_executor.layers.fused_moe.router.gb10_fp32_router import (
        small_fp32_router,
        small_fp32_router_supported,
    )

    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 1)
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    cases = []

    def compare(x, weight, label):
        reference = torch.nn.functional.linear(x.float(), weight)
        actual = small_fp32_router(x, weight)
        torch.cuda.synchronize()
        assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference, rtol=3e-4, atol=3e-5)
        # Check a generic sigmoid+bias top-6 routing proxy. The real model's
        # full routing/output/token checks remain a separate mandatory gate.
        bias = torch.linspace(-0.02, 0.02, 128, device="cuda", dtype=torch.float32)
        ids = [
            torch.topk(torch.sigmoid(t) + bias, 6, dim=-1).indices
            for t in (reference, actual)
        ]
        assert torch.equal(*ids), ("synthetic router IDs differ", label)
        assert torch.equal(
            actual.view(torch.int32), small_fp32_router(x, weight).view(torch.int32)
        )
        cases.append(
            dict(
                label=label,
                max_abs_error=float((actual - reference).abs().max()),
                changed_fp32_words=int(
                    torch.count_nonzero(
                        actual.view(torch.int32) != reference.view(torch.int32)
                    )
                ),
                synthetic_top6_identical=True,
            )
        )
        return actual

    try:
        for seed in (813, 491):
            torch.manual_seed(seed)
            for dtype in (torch.bfloat16, torch.float32):
                for padding in (0, 19):
                    weight = (torch.randn(128, 2688 + padding, device="cuda") * 0.02)[
                        :, :2688
                    ]
                    x_all = (
                        torch.randn(4, 2688 + padding, device="cuda", dtype=dtype) * 0.5
                    )[:, :2688]
                    for rows in (1, 2, 3, 4):
                        x = x_all[:rows]
                        assert small_fp32_router_supported(x, weight)
                        compare(
                            x,
                            weight,
                            dict(
                                seed=seed, dtype=str(dtype), rows=rows, padding=padding
                            ),
                        )
                    four = small_fp32_router(x_all, weight)
                    for row in range(4):
                        one = small_fp32_router(x_all[row : row + 1], weight)
                        assert torch.equal(
                            four[row : row + 1].view(torch.int32), one.view(torch.int32)
                        ), "batch invariance"

        x = torch.randn(4, 2688, device="cuda", dtype=torch.bfloat16) * 0.5
        weight = torch.randn(128, 2688, device="cuda", dtype=torch.float32) * 0.02
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            small_fp32_router(x, weight)
            torch.ops.vllm.gb10_small_fp32_router(x, weight)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = torch.ops.vllm.gb10_small_fp32_router(x, weight)
        for seed in (819, 820):
            torch.manual_seed(seed)
            x.copy_(torch.randn_like(x) * 0.5)
            expected = compare(x, weight, dict(kind="graph-replay", seed=seed))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(captured.view(torch.int32), expected.view(torch.int32))
        assert len(cases) == 34
        print(
            json.dumps(
                dict(
                    cases=cases,
                    passed=len(cases),
                    bit_identical_to_cublas_claimed=False,
                    performance_measured=False,
                    full_engine_timed_tokens_verified=False,
                )
            )
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_tf32


if __name__ == "__main__":
    main()

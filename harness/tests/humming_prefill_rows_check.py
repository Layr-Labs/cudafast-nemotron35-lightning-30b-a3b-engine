"""Host-only pinned GB10 equivalence probe. Run under the global resource guard.

Loads selected pinned expert weights. No timing, oracle, fetch or submission.
"""

import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--layers", default="1,13,27,40,51")
    args = parser.parse_args()
    # Match the fixture inside this standalone probe, never a ranked launcher.
    os.environ.setdefault("VLLM_HUMMING_BATCH_INVARIANT", "1")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vllm"))
    import torch
    from humming_moe_exact import Snapshot, build_variant, run_apply

    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 1)
    torch.manual_seed(912)
    snapshot = Snapshot(args.snapshot)
    cases = []
    for layer in map(int, args.layers.split(",")):
        raw = snapshot.raw_experts(layer, "cuda")
        expert = build_variant(raw, f"backbone.layers.{layer}.mixer.experts", True)
        del raw
        split_config = expert._prefill_row_configs
        assert split_config is not None, (
            "candidate did not select GB10 row split config"
        )
        for m in (64, 65, 128, 496, 512, 1024, 1025):
            x = torch.randn(m, 2688, device="cuda", dtype=torch.bfloat16) * 0.1
            weights = torch.full((m, 6), 1 / 6, device="cuda", dtype=torch.float32)
            for kind in ("balanced", "hot", "random"):
                if kind == "balanced":
                    ids = (
                        (torch.arange(m * 6, device="cuda") % 128)
                        .reshape(m, 6)
                        .to(torch.int32)
                    )
                elif kind == "hot":
                    ids = (
                        torch.arange(6, device="cuda", dtype=torch.int32)
                        .expand(m, 6)
                        .contiguous()
                    )
                else:
                    ids = (
                        torch.rand(m, 128, device="cuda")
                        .topk(6, dim=-1)
                        .indices.to(torch.int32)
                    )
                expert._prefill_row_configs = None
                reference = run_apply(expert, x, ids, weights).clone()
                expert._prefill_row_configs = split_config
                result = run_apply(expert, x, ids, weights)
                torch.cuda.synchronize()
                assert torch.isfinite(reference).all() and torch.isfinite(result).all()
                assert torch.equal(
                    reference.view(torch.int16), result.view(torch.int16)
                ), (layer, m, kind)
                cases.append(dict(layer=layer, rows=m, kind=kind))
        # Stable-address graph replay must observe changed IDs and activations.
        m = 496
        x = torch.randn(m, 2688, device="cuda", dtype=torch.bfloat16) * 0.1
        ids = (torch.arange(m * 6, device="cuda") % 128).reshape(m, 6).to(torch.int32)
        weights = torch.full((m, 6), 1 / 6, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run_apply(expert, x, ids, weights)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = run_apply(expert, x, ids, weights)
        for kind in ("hot", "balanced"):
            replacement = (
                torch.arange(6, device="cuda", dtype=torch.int32)
                .expand(m, 6)
                .contiguous()
                if kind == "hot"
                else (torch.arange(m * 6, device="cuda") % 128)
                .reshape(m, 6)
                .to(torch.int32)
            )
            ids.copy_(replacement)
            x.copy_(torch.randn_like(x) * 0.1)
            expert._prefill_row_configs = None
            reference = run_apply(expert, x, ids, weights).clone()
            expert._prefill_row_configs = split_config
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(
                reference.view(torch.int16), captured.view(torch.int16)
            ), (layer, "graph", kind)
            cases.append(dict(layer=layer, rows=m, kind="graph-" + kind))
        del captured, graph, expert, reference, result
        torch.cuda.empty_cache()
    print(json.dumps(dict(passed=len(cases), cases=cases, performance_measured=False)))


if __name__ == "__main__":
    main()

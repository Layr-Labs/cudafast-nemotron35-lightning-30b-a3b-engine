"""Unexecuted here: exact GPU comparison on guarded pinned GB10, no timing."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vllm"))
    import torch
    from vllm.model_executor.layers.mamba.ops import ssd_combined as combined

    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 1)
    torch.manual_seed(107)
    original_guard = combined.can_fuse_prefill
    selected = []

    def guard(*args):
        result = original_guard(*args)
        selected.append(result)
        assert result, "GPU case unexpectedly fell back"
        return result

    def inputs(lengths, chunk, initial, option=0):
        cu, last, seq = [0], [], []
        for batch, length in enumerate(lengths):
            for start in range(0, length, chunk):
                cu.append(cu[-1] + min(chunk, length - start))
                seq.append(batch)
            last.append(len(seq) - 1)
        length = sum(lengths)

        def feature(heads, width):
            # Noncontiguous outer stride; the inner feature stride remains one.
            return (
                torch.randn(
                    length, heads + 1, width, device="cuda", dtype=torch.bfloat16
                )[:, :heads, :]
                * 0.025
                if option % 2 == 0
                else (
                    torch.randn(
                        length, heads + 1, width, device="cuda", dtype=torch.bfloat16
                    )
                    * 0.025
                )[:, :heads, :]
            )

        return dict(
            x=feature(64, 64),
            B=feature(8, 128),
            C=feature(8, 128),
            dt=torch.randn(length, 64, device="cuda", dtype=torch.bfloat16) * 0.1,
            A=-torch.rand(64, device="cuda", dtype=torch.float32),
            chunk_size=chunk,
            cu_seqlens=torch.tensor(
                [0] + list(__import__("itertools").accumulate(lengths)),
                device="cuda",
                dtype=torch.int32,
            ),
            cu_chunk_seqlens=torch.tensor(cu, device="cuda", dtype=torch.int32),
            last_chunk_indices=torch.tensor(last, device="cuda", dtype=torch.int32),
            seq_idx=torch.tensor(seq, device="cuda", dtype=torch.int32),
            initial_states=torch.randn(
                len(lengths), 64, 64, 128, device="cuda", dtype=torch.float32
            )
            * 0.001
            if initial
            else None,
            D=None
            if option % 3 == 0
            else torch.randn(
                (64,) if option % 3 == 1 else (64, 64),
                device="cuda",
                dtype=torch.float32,
            ),
            z=feature(64, 64) if option % 3 == 2 else None,
            dt_bias=torch.full((64,), -3.0, device="cuda"),
            dt_softplus=True,
            state_dtype=torch.float32,
        )

    def run(args, enabled, intermediate=True):
        combined.can_fuse_prefill = guard if enabled else lambda *a: False
        out = torch.empty_like(args["x"])
        states = combined._mamba_chunk_scan_combined_fwd(
            **args,
            out=out,
            return_intermediate_states=intermediate,
        )
        return out, states

    def exact(reference, actual, label):
        torch.cuda.synchronize()
        for a, b in zip(reference, actual):
            assert torch.isfinite(a).all() and torch.isfinite(b).all(), label
            dtype = torch.int32 if a.dtype == torch.float32 else torch.int16
            assert torch.equal(a.view(dtype), b.view(dtype)), (label, str(a.dtype))

    passed = []
    try:
        for chunk in (128, 256):
            for i, lengths in enumerate(
                ((1,), (127,), (128,), (129,), (496,), (513,), (1537,), (3, 129, 17))
            ):
                for initial in (False, True):
                    args = inputs(lengths, chunk, initial, i)
                    saved = args["initial_states"].clone() if initial else None
                    label = dict(
                        chunk=chunk,
                        lengths=lengths,
                        initial=initial,
                        all_boundaries=True,
                    )
                    exact(run(args, False), run(args, True), label)
                    if saved is not None:
                        assert torch.equal(
                            saved.view(torch.int32),
                            args["initial_states"].view(torch.int32),
                        )
                    passed.append(label)
        args = inputs((496,), 256, True)
        exact(run(args, False, False), run(args, True, False), "terminal gather")
        passed.append(dict(kind="terminal-gather"))

        args = inputs((496,), 128, True, 2)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run(args, True)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = run(args, True)
        for bounds in ([0, 127, 255, 383, 496], [0, 128, 256, 384, 496]):
            args["cu_chunk_seqlens"].copy_(
                torch.tensor(bounds, device="cuda", dtype=torch.int32)
            )
            args["x"].copy_(torch.randn_like(args["x"]) * 0.025)
            expected = run(args, False)
            graph.replay()
            exact(expected, captured, ("graph-replay", bounds))
            passed.append(dict(kind="graph-replay", bounds=bounds))
        assert len(passed) == 35 and all(selected)
        print(
            json.dumps(
                dict(passed=len(passed), cases=passed, performance_measured=False)
            )
        )
    finally:
        combined.can_fuse_prefill = original_guard


if __name__ == "__main__":
    main()

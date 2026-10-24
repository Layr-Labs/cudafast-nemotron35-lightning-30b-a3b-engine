"""Host-only guarded CUDA check; no model weights, official oracle or timing."""

import json
from pathlib import Path
import sys


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vllm"))
    import torch
    from vllm.model_executor.layers.mamba.ops.ssd_combined import (
        mamba_chunk_scan_combined_varlen,
        prefill_state_sink_supported,
    )

    assert torch.cuda.is_available(), "CUDA required; a CPU skip is not a pass"
    torch.manual_seed(481)
    chunk_size, heads, hdim, groups, dstate = 128, 64, 64, 8, 128
    passed = []
    for length in (1, 127, 128, 129, 496, 513):
        for use_initial in (False, True):
            x = torch.randn(length, heads, hdim, device="cuda", dtype=torch.bfloat16)
            dt = torch.randn(length, heads, device="cuda", dtype=torch.bfloat16)
            B = torch.randn(length, groups, dstate, device="cuda", dtype=torch.bfloat16)
            C = torch.randn_like(B)
            A = -torch.rand(heads, device="cuda", dtype=torch.float32)
            D = torch.randn(heads, device="cuda", dtype=torch.float32)
            bias = torch.randn(heads, device="cuda", dtype=torch.float32)
            cuts = list(range(0, length, chunk_size)) + [length]
            nchunks = len(cuts) - 1
            initial = (
                torch.randn(1, heads, hdim, dstate, device="cuda", dtype=torch.float32)
                if use_initial
                else None
            )
            initial_copy = initial.clone() if initial is not None else None
            common = dict(
                x=x,
                dt=dt,
                A=A,
                B=B,
                C=C,
                D=D,
                dt_bias=bias,
                chunk_size=chunk_size,
                dt_softplus=True,
                cu_seqlens=torch.tensor([0, length], device="cuda", dtype=torch.int32),
                cu_chunk_seqlens=torch.tensor(cuts, device="cuda", dtype=torch.int32),
                last_chunk_indices=torch.tensor(
                    [nchunks - 1], device="cuda", dtype=torch.int32
                ),
                seq_idx=torch.zeros(nchunks, device="cuda", dtype=torch.int32),
                initial_states=initial,
                state_dtype=torch.float32,
            )
            reference_out = torch.empty_like(x)
            final = mamba_chunk_scan_combined_varlen(**common, out=reference_out)
            assert torch.isfinite(reference_out).all() and torch.isfinite(final).all()
            cache = torch.full((4, heads, hdim, dstate), -917.0, device="cuda")
            for index in (0, 2, -1):
                indices = torch.tensor(
                    [[index, 0, 0, 0]], device="cuda", dtype=torch.int32
                )[:, 0]
                assert prefill_state_sink_supported(
                    cache, indices, 1, (heads, hdim, dstate)
                )
                before, sink_out = cache.clone(), torch.empty_like(x)
                result = mamba_chunk_scan_combined_varlen(
                    **common,
                    out=sink_out,
                    final_states_out=cache,
                    final_states_indices=indices,
                )
                torch.cuda.synchronize()
                assert result is None
                assert torch.equal(
                    reference_out.view(torch.int16), sink_out.view(torch.int16)
                )
                assert torch.equal(
                    final[0].view(torch.int32), cache[index].view(torch.int32)
                )
                for block in range(4):
                    if block != index % 4:
                        assert torch.equal(
                            before[block].view(torch.int32),
                            cache[block].view(torch.int32),
                        )
                if initial is not None:
                    assert torch.equal(
                        initial.view(torch.int32), initial_copy.view(torch.int32)
                    )
                passed.append(
                    dict(length=length, initial=use_initial, cache_index=index)
                )
    print(json.dumps(dict(passed=len(passed), cases=passed, timing_measured=False)))


if __name__ == "__main__":
    main()

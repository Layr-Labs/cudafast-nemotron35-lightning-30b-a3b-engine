# smallk: bit-exact small kernels of the MoE routing in the MTP verify

Follow-on experiment: native MTP depth four verifies five rows. This source
also instantiates the same independent FP32 chains for M5 and admits only
the additional shape `(5, 128, 2688)`. The M4 evidence below does not prove
equality to cuBLAS's M5 choice. No local execution was performed for this
extension; the official remote token oracle and complete paired score decide
correctness and usefulness. `SMALLK_ROUTER_M5=0` restores the M5 torch fallback.
The matching configuration retains an exact five-row graph alongside the
one- and two-row drafter graphs. The older depth-only experiment's negative
whole result is not adopted; this evaluates a different router and sparse
expert cost regime on the newest full verified engine.

`smallk` replaces the latency-bound routing kernels of each of the 23 target MoE layers in the
depth-3 verify (4 rows, one FULL CUDA graph) with kernels that compute the same bits:

| Stock (per MoE layer, a0e1cae profile) | Launched by | smallk |
|---|---|---|
| cuBLAS `gemmSN_TN_kernel<float, 128, 16, 2, 4, 4, 4, true>`, 16 CTAs, 24.3 us | `GateLinear` -> `F.linear` (fp32 router) | `router_gemm_kernel<1, 4>`: 128 CTAs x 176 threads |
| `single_group_topk_warp_kernel<float, float, int, SIGMOID, 128, 8>`, 8.1 us | `GroupedTopKRouter` -> `ops.grouped_topk` | `route_kernel` (one CTA) |
| `moe_align_block_size_kernel` 11.8 us + `count_and_sort_expert_tokens_kernel` 3.8 us | humming `prepare_humming_moe_kwargs` -> `moe_align_block_size` | (same `route_kernel`) |

Hooks (vLLM tree): `fused_moe/router/gate_linear.py` (custom op `vllm::smallk_router_gemm`),
`fused_moe/router/grouped_topk_router.py`, `fused_moe/moe_align_block_size.py`, and the schedule
change in `models/nemotron_h.py` + `fused_moe/runner/moe_runner.py` (early router).

Switches (environment): `SMALLK=0` (all), `SMALLK_ROUTER=0`, `SMALLK_EARLY_ROUTER=0` (both on
by default); `SMALLK_TOPK=1 SMALLK_ALIGN=1` opt into the top-k/align kernels (off by default, see
below), `SMALLK_FUSE=0` then keeps them separate. Every call outside a kernel's verified guard runs
the stock op.

Verify-row count (dormant until the spec loop calls `smallk.enable_verify_rows(device)` before
graph capture; it returns an int32[1] buffer holding `VROWS_ALL`): the early-router path of
target-model gates (not the MTP layer's) then passes it to the router kernel, which gives every row
at or past its value the last valid row's logits bit for bit. Those rows route to experts the valid
row already loads, so a verify that masks its trailing rows reads no extra expert weights for them.
Valid rows are unchanged (each row's logits are its own chain). Only a spec loop whose sampler
stops acceptance at the same row may lower it; write it in place, since the verify graph captures
its address. Until it is enabled the kernel gets nullptr and runs the old code.

## Arithmetic

**Router GEMM** (`M = 4, N = 128, K = 2688`, fp32): the cuBLAS summation tree was recovered by
cancellation probes (`tests/probe.py tree`, the method of `harness/kernels/densegemm`): with
lane `r = k mod 16` and tile `j = k div 256`, `C(r, j)` is an fma chain over
`k = 256 j + r + 16 i`, `i` ascending, from 0; `L(r) = ((C(r,0) + C(r,1)) + ...)` over tiles;
`y = ((L(0) + L(1)) + ...) + L(15)`. The kernel gives each `(tile, lane)` chain its own thread
(176 threads per output row) and combines the partials in shared memory in that order.

**Top-k** (vLLM v0.27.1 `single_group_topk_warp_kernel`): `u = 0.5f * tanhf(0.5f * x) + 0.5f`,
`sel = u + bias` (`-0` -> `+0`, non-finite -> `-inf`), selection by the 64-bit key
`(twiddled sel bits, 65535 - expert)`, `unbiased = sel - bias[e]`, warp sum by the xor butterfly
of `cg::reduce`, `w = unbiased * (2.5f / (sum + 1e-20f))`.

**Align**: the same integers as `moe_align_block_size` + `count_and_sort_expert_tokens` (no expert
map). Within an expert the slots ascend by flat index, which is what the stock atomics produce
when all ids sit in one warp; the hook only takes `numel <= 32` (stock order across two warps is
a race, see `tests/exact.py align` at M = 8).

The fused path: once the experts have asked for an align block size for a token count, the top-k
hook launches `route_kernel` (top-k + align in one CTA) and parks the align outputs; the align
hook returns them for the same `topk_ids` storage and block size.

## Evidence

`tests/exact.py` (on Atom, real snapshot gate weights and biases of all 23 layers):

- router: 0 mismatching outputs of 2,826,240 per launch variant (rows per CTA 1/2/4/8) for
  normal, heavy-tailed, wide-exponent, bf16-rounded, cancellation-heavy and rms-normed inputs;
  the mul+add variant mismatches 46% (so the chains are fma); probe masks equal on 16,384 queries.
- top-k: 0 differing rows of 184,800 (random scales, quantized ties, identical and flat rows,
  saturated sigmoids, inf/nan, fewer than top-k finite, top-k 1 and 8, no renormalization).
- sigmoid sweep: all 268,435,456 fp32 logits with 2^-12 <= |x| < 16 equal.
- align: 0 differences for M = 1..5 (block sizes 16/32/64, invalid ids, colliding experts).
- route and the parked hand-off through the hooks: 0 differences, M = 1..5.

Standalone in a CUDA graph (M = 4, rotating cold weights): router 20.0 -> 6.7 us; gemm + top-k
+ align chain 27.4 -> 10.7 us.

## In the engine (depth 3, 7 prompts x 3 reps, exact vs the 0afa471 oracle on every run)

| run (same window) | decode ms/tok | prefill ms | composite |
|---|---|---|---|
| a0e1cae (nemo-d3-sk-base4) | 7.863 | 122.32 | 1.517 |
| router + early router (nemo-d3-sk-c33nr) | 7.706 | 122.68 | 1.539 |
| router + early router + route (nemo-d3-sk-c33) | 7.718 | 122.63 | 1.537 |

Profile (public prompt, per MoE layer): router GEMM 24.3 -> 9.8 us; top-k + align + sort
23.7 -> 6.9 us with the route kernel, but the routed-expert GEMM then grows by about as much:
the routing chain runs while the shared-expert Marlin GEMMs stream on the side stream, so a
shorter chain only moves that DRAM traffic into the humming window. The router GEMM ran alone
before the fork, so its saving is real; running it after the fork (early router) hides most of
what is left. Hence top-k/align are opt-in.

## bf16 gate read (SMALLK_ROUTER_W16, on by default)

The checkpoint stores each gate weight in bf16; vLLM upcasts it to fp32 on GB10. `gate_w16` keeps a
bf16 copy of a gate only if its fp32 values widen back bit for bit (int32 compare, once per gate,
outside capture), and the router kernel widens each element with `bits << 16` into the same fma
chains. Same operands, same logits; half the gate bytes (0.69 MB per MoE layer per verify). A gate
that is not bf16-exact keeps the fp32 read. `tests/exact.py` checks the W16 launches against
`F.linear` and logs the checkpoint dtype.

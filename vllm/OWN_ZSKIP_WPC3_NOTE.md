# Three-warp lean zskip preparation sharing

Model: Claude Opus 5.5
Harness: Claude Code

## Summary

This candidate starts from the current promoted record, submission `f5904a37-e3aa-4c29-9a8a-4cecee7b6184` (main `8695be87`, official 2.67234), and changes nothing else in it. It adds one independent target-model mechanism inside the routed-expert lean zskip CUDA kernel. The inherited default launches one warp per CTA for each of the 21 output tile pairs. Every one of those CTAs reconstructs the same routed rows, scans the same BF16 activation rows, builds the same 116 group masks, and compacts the same ordered list before its warp begins the output-specific weight work. Config 663 places three adjacent output warps in one CTA. Those three warps share the row discovery and mask scan already implemented by the kernel template, then retain separate ordered lists and perform the unchanged output math.

The candidate changes no graph-level Torch expression. It does not move an add, cast, normalization, quantization boundary, or residual operation across a custom-op boundary. It changes neither the number of custom-op calls nor the ordering of surrounding kernels. Its only intended saving is work that is visibly duplicated inside `down_lean_kernel` itself.

## Actual execution path

The ranked source uses the Inductor compilation backend and the CUDA platform's native IR priority. Target-model Torch operations around the MoE are therefore compiled as graph code where possible. The routed down projection remains an opaque zskip custom operator. For the five-row verify shape, six routed experts per row produce `shape_m = 30`; the checked lean configuration replaces the earlier paired kernel during model finalization. CUDA graph capture records that selected custom-op launch and replays it for the timed target steps.

The inherited default is lean config 653: one warp per CTA, ring depth three, no split-row shortcut, literal zero fragments for empty halves, and the per-warp L2 prefetch mode with distance two. `launch_lean` uses a grid of `(21 / WPC, max_blocks)`. At WPC one this is 21 CTAs for every routed expert block. In each CTA, warp zero recovers up to eight routed rows. The CTA then scans all 116 k16 activation groups, producing a 16-bit mask per group, and compacts the active groups in ascending index order. Only after that common preparation does the output-tile value `P` select different weight and scale addresses.

The `down_lean_kernel<WPC, R, F>` implementation was already written to support more than one warp per CTA. It distributes the 128 mask slots across `WPC * 32` threads, uses a block barrier when WPC is greater than one, gives every warp its own `s_list[warp]`, and derives `P` as `blockIdx.x * WPC + warp`. Config 663 instantiates that existing path at WPC three while preserving config 653's ring depth and flags. Seven CTAs still contain exactly 21 output warps. Warp P reads the same output weight tile, executes the same active groups in the same order, and writes the same disjoint output columns as warp P in config 653.

## Cost model

For the normal five-row target verify call, routing produces 30 activation rows across the expert blocks. One complete logical scan reads `30 * 1856 * 2 = 111,360` BF16 activation bytes. Config 653 repeats that scan for 21 output CTAs, or 2,338,560 logical bytes per MoE layer. Config 663 repeats it for seven CTAs, or 779,520 logical bytes. The direct reduction is 1,559,040 logical activation bytes per layer, two thirds of the mask-scan loads. Across 23 target MoE layers that is 35,857,920 fewer logical bytes per target forward.

Those logical bytes are not claimed as 35.9 MB of LPDDR5X traffic. The activation is small enough for substantial L2 reuse, and concurrently scheduled CTAs may hit data fetched by another CTA. The reduction still removes L2 requests, address generation, two-row scan loops, nonzero-mask operations, repeated row ballots, and fourteen CTA instances per expert block. It leaves the 21 weight-producing warps and their dominant weight traffic unchanged. The launch count is also unchanged: one lean zskip custom-op launch per layer before and after the patch.

The historical target decode scale recorded for this record is about 7.083 milliseconds per output token. A new decode-only mechanism needs approximately a 0.266 percent target-time reduction to contribute 0.2 percent to the composite score with the decode exponent of 0.75. That is about 18.9 microseconds per token, or about 0.82 microseconds per one of the 23 MoE layers. The removed 1.559 MB per layer reaches 0.82 microseconds at an effective service rate near 1.9 TB/s, before counting the removed instructions and CTA setup. This makes a contribution above 0.2 percent plausible on an integrated, bandwidth-sensitive device, but it is not a measurement. L2 hits, three-warp CTA scheduling, and the two block barriers can make the saving smaller or negative. The official improvement remains unclaimed until the host compares the exact archive on GB10.

The candidate deliberately does not add a load-time timing chooser. A synthetic isolated kernel decision would not reproduce the captured target graph's concurrent shared expert and prefetch traffic, and selecting from that microbenchmark could make the ranked path less stable. The existing load-time bitwise admission remains the fallback boundary: compilation failure, unsupported configuration, or any byte mismatch leaves the previously checked paired configuration active rather than selecting config 663.

## Bit-exactness argument

The numerical chain is local to each output warp. Config 663 does not change `lean_main`, `lean_group`, weight or scale loads, dequantization, ring depth, prefetch distance, MMA sequence, negative-zero checks, dense recomputation, global scaling, BF16 conversion, or output addressing. It only shares input-derived metadata that config 653 independently recomputes to the same values in each one-warp CTA.

Row discovery remains warp-zero code over the same 16 aligned routing slots. The same ballot defines valid rows, the same rank expression stores them, the same eight-row cap applies, and the same expert id is selected. Other warps wait at the block barrier before reading this shared state. No output-specific value participates in row discovery.

For mask construction, config 653 has 32 threads and four group positions per thread. Config 663 has 96 threads and two compile-time positions per thread, of which only group indices below 128 are active. Every group index from zero through 127 still has exactly one owning thread. For a given group, that thread visits routed rows in the same increasing pairs and ORs the same two `nz_mask8` results into a local integer in the same order. The mapping from a group to a CUDA lane changes, but there is no cross-thread floating-point reduction and no atomic ordering. Each final `uint16_t` mask is therefore the same integer.

After the block barrier, each of the three warps independently walks group ranges 0--31, 32--63, 64--95, and 96--127. The ballots and prefix popcounts produce an ascending compact list. Each warp writes only `s_list[warp]`, so there is no list race. The list content and `nag` count match the one-warp configuration. From that point onward every warp has the same P, row list, mask list, activation pointer, weight pointer, scale pointer, and flags it would have had in config 653. The per-output FP32 accumulator sees the same operations in the same order. Output tiles are disjoint, so grouping three warps cannot change a write conflict.

Dense recomputation after a negative-zero accumulator is also per warp and unchanged. A warp detecting negative zero reruns its own output tile with all rows loaded; it does not consult another warp's accumulator. Prefix verification uses the same `row_limit`, and mask construction excludes the same rejected suffix rows before list generation. Padded routing slots, empty expert blocks, signed zeros, empty k16 groups, the one-row specialization, and ring wrap do not introduce a new state path.

The Python finalizer asks the zskip library for config 663, then invokes the inherited `_pair_check`. That check covers M=1 through M=8 for both crowded and distributed routing, compares every served BF16 word against the Humming reference and the previously admitted paired configuration, injects an all-zero row and signed zeros, and exercises empty low and high eight-row halves. When prefix support is present, it separately checks every accepted prefix count for M=5 and verifies that the rejected output suffix remains zero. Config 663 becomes active only if all of those comparisons pass. Any exception or mismatch retains the prior checked configuration.

The patch does not modify model weights, their NVFP4 representation, FP8 projections, the FP8 KV cache, FP32 SSM state, the BF16 MTP head, routing ids, top-k weights, logits, acceptance logic, resident state, or request invalidation. It creates no cache across requests or phases. It moves no prefill work across `free_decode_begin`, and it exports no environment variable. The `ZSKIP_LEAN_CONFIG` diagnostic override remains available inside the source's existing test mechanism; the ranked fixture does not need or export it.

## Failure boundaries

The principal execution risk is CUDA compilation of a template shape that has not been built in this preparation environment. Ninety-six threads per block are within the kernel's declared `__launch_bounds__(WPC * 32)`, and 21 is exactly divisible by three, but register allocation and occupancy must be checked by the host build. `zskip_config` queries the compiled function attributes before selection; a query failure is caught by `_prepare_lean`, which leaves the prior configuration selected.

The principal performance risk is CTA granularity. Three warps now share a CTA lifetime. Adjacent output tiles have the same active-group count, but memory-system variation can delay one warp and retain the other two warp resources until the whole CTA completes. The multi-warp path also uses `__syncthreads()` instead of `__syncwarp()` at the two metadata boundaries. These costs can offset the reduced scanning, and static inspection cannot settle that balance. A regression is a reason to discard the candidate even if exactness passes.

The change is limited to the M=5 lean decode selection. Prefill continues to use pzskip and the inherited prefill configurations. If lean admission fails, the checked paired configuration remains. Other models and shapes do not select this target-specific configuration through the guarded finalizer.

## Host measurement plan

Build the exact archive in the pinned CUDA image and confirm that config 663 reports valid register, shared-memory, warp, and occupancy attributes. Run the zskip exact suite for M=1 through M=8 in dense, skip, and verify-prefix modes, including signed zero, empty halves, all-zero rows, crowded routing, distributed routing, and capture replay. Compare config 663 byte for byte with config 653, the inherited paired configuration, and the Humming reference.

Measure config 653 and 663 first as isolated captured calls using real activation windows from every target MoE layer. Record setup-only, load-only, and full-kernel time rather than inferring the saving from end-to-end score. Then compare complete target decode on the same host allocation, same prompt pool, same depth, and same inherited source. A GPU trace should confirm seven three-warp CTAs per expert block, one zskip launch per layer, no extra synchronization launch, and unchanged surrounding graph pieces. Report raw target decode time and the official composite separately. Reject the candidate for any token mismatch, compilation or capture failure, occupancy collapse, or repeatable latency regression.

## Host checks

to be filled by the host

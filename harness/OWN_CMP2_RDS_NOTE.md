Model: Claude Opus 5.5
Harness: Claude Code

# Transparent compressed backing for the routed-down pair that decode actually reads

## Scope

This candidate keeps the complete current record and changes only the storage backing of two immutable tensors per eligible target MoE layer. The selected tensors are the packed FP4 code plane and FP8 scale plane in `zskip_pair_w2`, with layout `swap2`. That pair is the prepared representation read by the target verify's paired routed-down M=5 kernel. The ordinary Humming representation, the legacy zskip representation used by other shapes, every activation, every cache, and every output tensor stay outside the new selection.

The mechanism already used by the record is CUDA virtual-memory generic compression. It is transparent to the CUDA kernels: the virtual address maps a generic-compression allocation, but loads return the same bytes that were copied into it. There is no model codec, no quantization conversion, no numerical kernel change, and no request-dependent choice. The extension applies after model loading and all routed weight repacks, before compilation and CUDA graph capture.

## Why this is a distinct extension

The record selects immutable FP8 projection packs, attention planes, dense shared-expert Marlin representations, and the language head. It deliberately does not select routed experts. The routed-down M=5 path, however, does not read the large ordinary routed representation. It reads the already prepared paired zskip representation. Selecting only that hot representation avoids adding compressed copies for routed-up weights, ordinary fallback weights, the legacy routed-down representation, KV storage, SSM state, workspaces, masks, locks, or request buffers.

Each paired representation contains exactly the original NVFP4 code bytes and block-scale bytes in a different fixed order. One layer's pair is approximately 357 MB. The target has 23 routed MoE layers, so the extension can expose roughly 8.2 GB of immutable backing to hardware compression. That is a capacity figure, not a claim that all of those bytes are fetched in one token. A verify routes six experts per row and zskip omits exactly zero activation sectors, so the useful traffic is the selected experts and live sectors. The opportunity exists because E2M1 code nibbles are drawn from only sixteen bit patterns and trained quantized weights are centered around the zero and low-magnitude codes. Repeated zero or low-magnitude nibble pairs can be compressible to the memory controller even though the logical FP4 layout remains unchanged. FP8 scales are expected to compress less well. No compression ratio can be established without the target machine's hardware counters, so the candidate treats the ratio and the speed gain as measurements, not correctness assumptions.

## Implementation

`cmp2._select` now returns two lists. The first list is the record selection byte for byte in meaning. The second list is the new routed-pair family. A tensor enters that family only when all of the following are true:

- it belongs to the target model;
- the layer has a `zskip_pair_w2` object;
- the object's layout is exactly `swap2`;
- the `w` or `s` member is a nonempty CUDA Tensor;
- its Tensor identity has not already entered either transaction.

No name substring is sufficient by itself. The prepared object's explicit layout field is the admission contract. The global scale is excluded because it is tiny and may alias another retained representation. The ordinary routed tensors and the first zskip pack are excluded deliberately.

The allocation and rebinding body is factored into `_compress`. It gathers complete underlying storages, not just visible Tensor spans. Aliases sharing device, data pointer and storage byte count map to one new storage. For every storage, the code makes a byte view that includes padding and copies it to the host. It requests `CU_MEM_ALLOCATION_COMP_GENERIC`, checks the allocation's reported compression type, maps it at the same two-megabyte alignment used by the record, copies the bytes, copies them back, and compares the complete byte sequence with `torch.equal`. Tensor shape, stride and storage offset are rebuilt and checked before any pointer is changed.

Rebinding is atomic within a family. All allocations and all full-storage comparisons for that family must succeed before the first `Tensor.data` assignment. If a rebind raises, every Tensor already touched in that family is restored. Original storage byte views and compressed DLPack tensors remain strongly referenced for the engine lifetime.

## Failure isolation

The record selection and the routed-pair selection are two transactions. The record transaction runs first. If it fails, behavior is the record's ordinary-tensor fallback and the extension is not attempted. If the record transaction succeeds but the routed extension is unsupported, cannot allocate, cannot reproduce every byte, or cannot preserve metadata, only the routed extension is abandoned. The already successful record transaction remains active. This separation matters because an optional 8 GB family must not turn an allocation miss into loss of the record's measured compression benefit.

The extension also performs a conservative memory-reserve check after the record transaction. It sums unique routed-pair storage byte counts and queries free CUDA memory. The extension proceeds only when the worst-case requested bytes plus 8 GiB are free. Generic compression may consume less physical backing, but the admission assumes no compression. The retained reserve is for graph capture, scratch, allocator fragmentation and later engine setup. If the condition is false, the log states requested, free and reserve bytes and the engine continues on the record path. Allocation can still fail after the check; that exception is contained by the extension transaction and also returns to the record path.

A device without virtual-memory generic-compression support returns from the record mechanism exactly as before. A build without the optional allocator package follows the existing import fallback in the model runner. The environment switch `CMP2=0` still disables both transactions.

## Numerical argument

The target's state and output token must be bit invariant. This change does not alter an arithmetic instruction or an arithmetic operand. Before rebinding, the complete backing storage of every candidate tensor is compared byte for byte after a round trip through the requested allocation. This includes FP4 code bytes, FP8 scale bytes, padding, NaN payloads and signed-zero encodings. The reconstructed Tensor has the same dtype, shape, stride and storage offset. The CUDA graph is captured only after the new pointer is installed, so there is no stale captured routed pointer.

The paired zskip kernel receives the same logical addresses for its `w` and `s` arguments, except for the virtual base. Every byte returned by a load is the byte that occupied the corresponding offset in the old storage. Its activation masks, expert ordering, ascending K groups, dequantization, MMA chain, global scale and BF16 epilogue are untouched. Target logits, KV cache, fp32 SSM state, sampler inputs and committed tokens therefore remain bit invariant if the CUDA VMM contract holds. Failure of that contract is caught at load by the full byte comparison before the pointer is visible to inference.

The drafter is not selected. There is no permitted drafter-only perturbation to assess in this candidate.

## Performance model

The current record reported roughly 0.9 to 1.1 percent lower decode time from generic-compression backing on other immutable decode weights. The routed-down pair is a major remaining immutable stream, but its code bytes are only four bits per weight and may have less hardware-compressible structure than the record's other planes. A conservative planning range is 0.7 to 1.3 percent lower decode time when the extension is admitted, corresponding to about 0.5 to 1.0 percent composite improvement with the 0.75 decode exponent. The lower end can miss promotion; the upper end clears it. This range is an estimate, not measured evidence.

The strongest counterargument is that the paired FP4 stream may be close to incompressible after nibble packing. In that case generic compression can provide little bandwidth reduction while still changing physical placement, and the official result will be the record within noise or slightly worse. A second risk is that the memory-reserve check disables the extension on the assigned machine. The log must therefore be collected with the score so a no-op admission is not mistaken for negative compression evidence.

A useful host measurement records: whether all 23 layers are selected, unique selected bytes, free and reserve bytes, allocation type, the extension success log, decode time for each paired leg, and any available memory-compression counters. The candidate should be rejected as an optimization if the extension is not admitted, if its central decode improvement is below the noise floor, or if any target byte/token check differs.

## Changed surface

- `harness/kernels/cmp2/py/__init__.py`: split the measured record family from the routed-pair family, add exact `swap2` selection, add family-atomic fallback, and require an 8 GiB worst-case reserve.
- `vllm/VENDOR.json`: document the local storage-only extension and fallback.
- This note and its dependency draft live under the editable harness surface and are inert at runtime.

## Host checks

to be filled by the host

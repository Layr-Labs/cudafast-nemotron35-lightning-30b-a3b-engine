# Small-M routed up / ReLU-squared experiment

Model: GPT 6.1 Sol. Harness: Codex. Effort: ultra.

This is a new full-K native target up projection for NVFP4 W4A16 experts of
Nemotron 3.5: E128, N1856, K2688, BF16 M1..8, top-six and block16 indexed
routing. It is now composed on the complete highest runtime source
83376611-5b73-4602-b51e-82b5943b7451 (official1.962255807169652), which
retains a1c083d7-215a-4b8e-abeb-ea12e7cad022 (1.924184208672474,
tree4b6706eb4bc5ac901b64ba6211a2d1f13ba3987c). All833 source changes
are exact; seven generated pyc artifacts from the upstream archive are
deliberately omitted per the repository guide. No source/kernel is omitted.

The existing Humming up already has a fused ReLU-squared epilogue. This change
replaces its small-M GEMM schedule and weight addressing, rather than treating
that old fusion as a new saving. It does not sparsify the target input, prune
an expert, split K, change the router or alter any numerical format.

## Data layout

The load-time permutation is derived from the `swap` packing of a1's zskip
down. Every checkpoint nibble and E4M3 scale byte remains. W becomes
`[128,2688,928]` raw bytes and S `[128,168,1856]` raw scale bytes. Column
`n=64T+16c+8h+g` maps to byte `32T+4g+c`, nibble h, and its scale to byte
`64T+8g+2c+h`. FP32 expert global scales remain separate and are compared
with Humming's converted scales before the path is enabled. There is about
357MB additional residency per layer, about8.2GB across23 layers. The original
Humming weights and full fallback remain, as do a1's down copies. Startup
memory/load cost is a risk, not a claimed benefit.

## Kernel

One warp owns64 output columns of an aligned expert block, and two warps form
a CTA. Twenty-nine column tiles cover1856; the last CTA retires its unused
whole warp. All collectives are warp-only. The warp gathers that block's at
most8 current row IDs and uses `encoded_id // 6` to read the original token
input. It stores to the original encoded row, without a new permutation
kernel or scratch. Top-six IDs are unique per token in the existing router,
so one expert can contain at most M rows. Padding IDs are excluded.

Every one of168 K16 groups executes. A four-group register ring loads all
weight rows, eight scale bytes per lane and the current token's BF16 inputs.
The BF16 bit placement, two BF16 scale multiplies and
`mma.sync.m16n8k16.f32.bf16.bf16.f32` body are derived from a1 zskip v4. The
weights are the16-row MMA operand and tokens the8-column operand. Each output
has one ascending K16 FP32 accumulator chain with no CTA or cross-CTA reducer.
This schedule/orientation has not been proved bit-identical to the actual
Humming up for all inputs. The official token oracle remains decisive.

The epilogue computes the original explicit FTZ global-scale multiply and RN
BF16 GEMM rounding, then the existing Humming fused epilogue's explicit
non-FTZ `max.NaN`, RN FP32 square and RN BF16 rounding. No cancellation, zero
omission or sign-of-zero identity shortcut is used in this new up kernel.

## Integration and evidence

Only tuned unpadded no-bias NVFP4/BF16 layers prepare a copy before Humming
converts the checkpoint weights. After conversion the new path checks the
global scales, initializes CUDA attributes outside graph capture and compares
the fused output bitwise against original Humming on deterministic generated
M1,4,5,8 inputs. That comparison runs during official remote startup. A
mismatch disables the new path for that layer. It is a finite comparison,
not a proof over all values and not a replacement for target validation.

The runtime bridge also requires the original tuned ReLU2 branch, no expert
map or input quantization, block16, BF16 tensors with supported shapes/strides,
and int32 contiguous current routing metadata on the same device. Other calls
continue through the existing fused Humming branch. `ROUTED_RAW_UP=0` disables
both preparation and native dispatch. Prefill remains Humming. No new state
or work is reused between requests or phases.

No candidate code, compiler, setup, syntax/AST check, test, benchmark, profiler
or model was run locally. Local work consisted of source editing, text/Git
inspection and JSON metadata handling. There is no measured speedup or
correctness result for this new path. It is intended to reduce the dense
routed-up cost after a1 reduced routed-down traffic. Dense raw weight streaming
could be slower than Humming's staged schedule, and a runtime comparison could
disable it entirely; only an official whole-bundle result will decide.

## Material sources

- a1c083d7-215a-4b8e-abeb-ea12e7cad022: complete highest source, raw byte swap,
  dequantization helpers, transposed MMA fragment mapping and register-ring
  design. The new up has different dimensions, indexed input gather and direct
  fused activation store, and no sparse preparation/list/skip machinery.
- 20912575-b224-41df-92c5-f9a4039f2e2f: inherited expert-up fused Humming
  epilogue and its precise BF16/non-FTZ ReLU2 boundary, copied into the new
  native epilogue while retaining the original fallback.
- c2a77c1c-7c50-4cd9-9521-ca5cdd5ec009: complete predecessor carried by a1.

All other inherited mechanisms remain those of full a1; VENDOR records the
new files and integration. Whole-bundle regression above0.1percent is not
adopted as the next baseline. No negative prior MTP/slot-graph bundle is used
as this branch's baseline.

The new composition also reuses nativeM5/depth4/exact5graph and last-MoE
prefill routing from our full-correct286548fd-f052-4777-828b-f5fa3019b614
(1.9561806575737297, +1.66percent versus a1 but below833; not baseline),
and the gatednorm/staticFP8 producer component of full-correct
7b5bcade-e1ec-495a-9290-c7bd4931733f (1.94081609151312, not baseline).
Their isolated gains are not established. Complete833 compact draft source
is retained, originally from73d42dd1-a764-48c2-b125-55a43407a3a7, and
this new combination requires official remote evaluation.

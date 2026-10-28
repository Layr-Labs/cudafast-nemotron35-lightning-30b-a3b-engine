# Direct hidden gather and screen cascade

This is an optional derivative of `scrtail`, whose original files are unchanged.
It uses the existing `STAGE1=true` coarse dot and all the original radix,
mid/rest/full dot arithmetic. It does not replace token selection.

Each 512-thread CTA owns one 2048-row chunk, loads the full hidden vector into
the original shared allocation, gathers pre/mid/rest vectors there, then runs
the existing first dot and cascade without a global coarse-score read. CTA0
publishes the original auxiliary vectors before the stopped early return,
because the original gather was ungated. Production still writes every coarse
score, candidate and logit; debug also scatters the original mid scores.

This removes the gather and coarse-dot kernel boundaries from the admitted
screen body. Matrix products, quantized weights/scales and final selection
remain unchanged. It adds indirect column-map reads and one live CTA barrier;
the 64-CTA first-dot plan can lose latency hiding compared with the original
2048-CTA Triton plan. There is no measured resource or speed claim.

The ctypes ABI is the original 21 pointers, global scale, four integer controls,
three int64 column-map pointers, optional raw-first debug pointer, then stream.
Only `STAGE1=1`, `WIN=0`, 64 chunks are accepted. Both original PDL forms build.
The optional debug pointer receives first scores before mid scatter; it is null
in production. `scores_out` always points to the real screen score owner.

`nemotron_h_mtp_gather_tail.py` authors qualification against the actual original
screen: first scores/aux, all full body state, candidates/logits/token/gate,
read-only inputs, guarded private owners followed by actual input/output
addresses, changed-input eight-output graphs, original continuation, and exact
restoration. Full production body timing compares original gather + first dot +
tail + selection with this kernel + unchanged selection, with independent gate
resets outside timing and alternating warm/64 MiB pressure trials. The outer
greedy copy/graph wrapper/token clone is unchanged and outside this body timing.
Admission requires median ratio <0.995 and worst ratio <1.01. No private input,
graph or output survives preparation. Any runtime error after dispatch
propagates; unsupported paths fall back before GPU work.

`MTP_SCREEN_GATHER_TAIL=0` disables this path. Only if it is not admitted does
the separately qualified gather-first fallback from submission
`eca713b1-34f1-4a68-b96a-daed18115493` prepare. That predecessor failed in the
official sandbox before build; it has no correctness or performance evidence.

This package has only been edited and inspected as source locally. No local
setup, import, syntax/AST execution, compilation, test, benchmark or profiler was
run. Execution and qualification are reserved for official remote validation.
Source and idea provenance includes original scrtail
`fd5696c6-0b61-477c-9e28-d4caca4659e3`, inherited through complete highest
`1faceed1-7860-4e8d-b734-96600d6629aa`.

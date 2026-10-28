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

The ctypes ABI is the original 21 pointers, global scale, five integer controls,
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


## Source-level weight lookahead schedule

The additional LOOKAHEAD form preserves the original 16-warp row ownership,
64 rows per iteration and four-iteration batch. Its two automatic weight/scale
banks preload batch zero, issue all independent loads of the next batch before
consuming the current bank, compute the current rows with the original
`pre_block_term` and exact FP32 chains, then rotate banks. The final batch
issues no next load. Each row still reads its original two 64-byte blocks and
16 scales once; no matrix product or shared allocation changes. No extra CTA
barrier is added by lookahead.

The intent is to expose memory operations early enough for warp scheduling to
cover first-dot decode/ALU work. These are automatic C++ arrays intended for
registers; scalarization, physical register assignment, spills and generated
instruction scheduling have not been observed. The second bank extends live
ranges and may hurt the full kernel. No measured overlap or utilization claim
is made.

The C ABI adds one integer `lookahead` after `stage1`; both zero and one are
accepted, and each has PDL/non-PDL launch specializations. The wrapper and
actual-owner spec bind that choice. The lookahead form must independently pass
all original first-score, full-state, continuation, changed-output graph and
complete-body cost gates. If refused and state restored, the original 954
full-fusion dataflow is independently qualified against the original screen;
if that too refuses, gather-first and then the original path remain.
`MTP_SCREEN_WEIGHT_LOOKAHEAD=0` omits only the lookahead form. This is two
substantively different load/consume schedules, not a sweep of tile sizes or
relaxed numerical/cost thresholds.

The predecessor `16427993-8afc-42d8-a93b-804122b16dc2` failed on the official
runner at Sandbox start and cleanup before setup/build/GPU execution. Its
fusion is unvalidated, and it is not a verified source baseline. The previous verified whole was `1faceed1-7860-4e8d-b734-96600d6629aa`;
the current full verified base is `eb48e2f0-50e6-4c39-a1f3-0ac305d17c5b`. This further source has
not been uploaded at preparation. Four same-worker infrastructure failures
caused an intake hold. Subsequent ordinary official work dispatched after the
fault to three healthy workers (runs40609136225 on1,40610026901 on4,40610168656
on3), each passed sandbox and setup and entered GPU measurement. That is
healthy ordinary-queue progress, not proof spark-2 was repaired or that account
routing is sticky. One new substantive intake on the complete new highest
source may use the ordinary queue; no identical failed package retry or worker
selection is intended. No trusted cleanup or local execution is used.


## Complete EB48 depth-five base and selector compatibility

The current base is the entire sealed-valid EB48 result, score
2.7314909238146505, official commit e0e0b4410aec0c51830787a5be7d8482bfc3ac15,
root babd54c1aa16348df15dc6434815043767009f9d. All22 official full delta blobs
were fetched and Git-hash checked; the reconstructed entire root is exact.
Its declared depth five, six-row fast paths, low-acceptance 1.8/2.0/min4 rule,
original conditional-step default/admission and out-projection implementation
remain. The original EB48 `_select_token` code is unchanged.

Both new full-body and gather-first qualification pass the original THETA5,
adapt pointer, ADAPT and TAU arguments. The owner binds the selector/getter,
thresholds and adaptive tensor metadata, excludes aliases, observes all four
adaptive words, uses guarded private adaptive storage then the actual adaptive
address and restores exact original bytes/pointers. Changed cases include kept
counts before/after the fourth draft, live/stopped/inactive gates, adaptive
flags around0.5 and cumulative products1/.5. Analytical confident-token probes
with one BF16-exact +1000 logit and the rest-1000 make denom exactly1, so
threshold/adaptive-gate expectations do not depend on random hidden confidence.
These probes, firstscore/aux, full-state/continuation/changed-output graph and
full-body cost checks are authored only for official remote execution. All
private random inputs explicitly use FP32 despite model default dtype.

No new component speed, branch activation or old policy defect is inferred
from the baseline's whole score. The original depth/adaptive implementation
is not replaced with a separately tuned policy.

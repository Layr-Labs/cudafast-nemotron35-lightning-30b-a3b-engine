# Draft: reuse the speculative convolution input packet

Effort: ultra. Implementation author: GPT 6.1 Sol using Codex. The author and
effort were read from the native context of the implementation turn, rather
than inferred from an earlier submission. This is an unvalidated source draft.
No candidate code, import, parser, AST, syntax check, compiler, setup, test,
benchmark, preflight or profiler was run locally. Only source editing, text
inspection and Git metadata operations were used. The source is frozen for
review, not submitted by its author.

## Complete baseline and scope

The parent is the entire officially validated c41fc629-8c06-4bf2-8cb8-cad0edea9875
source, with official score 2.77776232031352. Its published source commit is
cdea0e5bb85ae36999f23c1f9de15de5901b8e45 and its complete Git tree is
3f3a3fa2636eb7c71bf871462134b3b18fbc5259. The local baseline object
4726d88aef2bfa8e04c110015568a9c68a840b30 has that same complete tree.
The score belongs only to that baseline. This draft has no remote result,
activation evidence or measured speedup.

The only executable source change is in
`vllm/model_executor/layers/mamba/ops/causal_conv1d.py`. The VENDOR entry and
this note record the method and limits. The Mamba FP8 in/out projections,
SSM replay kernels, quantization, sampler, speculation depth, graph settings
and all other baseline files are retained. The prefill convolution body is
unchanged. The selected decode kernel retains the original grid, warp and
stage choices and the original sequence of arithmetic operations. This is a
data reuse experiment, not a launch-parameter sweep.

The larger alternative considered during research was to compute the
causal convolution in the FP8 in-projection epilogue. It was not implemented.
It would cross a CUDA/Triton numerical boundary and require one shared
producer/consumer branch across the current projection and Mamba opaque
operations. This smaller draft deliberately does not require that protocol.

## Source-level repeated read

The original speculative update kernel first loads the raw input matrix into
`loaded_x[NP2_STATELEN, BLOCK_N]` to update the convolution history. After that
state update, its serial token loop loads each current raw input from global
memory again for the convolution's final tap. For the qualified case, every
one of these BF16 values is already present in the first packet.

The new `REUSE_INPUT_PACKET` constexpr defaults to false. When its host and
device predicates are satisfied, the width-four final tap uses a two-dimensional
integer gather from `loaded_x`, followed by a shape-only reshape to the original
one-dimensional channel vector. No floating reduction, addition, multiplication
or conversion is used to select the value. In particular, the implementation
does not use `sum(where(...))` to emulate a gather.

The original `matrix_w`, `acc += matrix_x * matrix_w`, bias preload, halo
shifts, `acc / (1 + tl.exp(-acc))`, output store, dtype conversions and state
update are retained. The gather returns the source packet's BF16 element type.
It does not promote the operand to FP32 or add a new BF16 rounding point.
The original raw `tl.load` remains both when the constexpr is false and when
the per-call device query condition rejects the packet index.

## Index proof

Let Qmax=6 be the launcher's maximum query length and q be this request's
actual length from `query_start_loc`. The host eligibility requires width4
and a non-null speculative accepted-token tensor. The original launcher sets

```
state_len_initial = width - 1 + (Qmax - 1) = 8
NP2_STATELEN = 8
```

The original varlen code then computes

```
state_len = 8 - (6 - q) = q + 2
seqlen = q
VAL = state_len - seqlen = 2
```

The first packet's load address for row r and feature f is

```
x_base[f] + (r - VAL) * stride_x_token
```

Consequently packet row `VAL + i` is exactly the original current-token
load address `x_base[f] + i * stride_x_token`. The token loop has 0<=i<q;
the device predicate additionally requires 1<=q<=6 and VAL==2. Its largest
index is 2+5=7, below the eight-row packet extent. The original input mask is
true for that row and every qualified feature because 6144 is divisible by
the retained BLOCK_N=128. A zero-length request returns at the original early
exit before the packet is used. NULL slots also retain their original early
exit. A rejected device predicate uses the original raw load.

The original query offsets are unchanged; this proof does not replace their
GPU values with a captured Python scalar. The accepted-token value and
`accepted - 1` history offset are also loaded in the original kernel at each
invocation. This draft does not derive or cache a state address on the host.

## Host eligibility and aliasing

The host helper inspects tensor metadata only. It does not read GPU tensor
values, launch work, allocate a GPU buffer or cache an admission decision.
The optional path requires all of the following:

- CUDA sm121; BF16 x with 1..6 rows, 6144 channels and the exact original
  projection-view stride `(10304, 1)`; output must be that same Python tensor
  object after the existing dtype handling.
- Maximum query length6; one-row-by-six-column int32 state indices, one int32
  accepted-token entry and two int32 query offsets, all contiguous.
- BF16 contiguous convolution weights of shape `(6144, 4)` and, if present,
  a BF16 contiguous bias of shape `(6144,)`.
- No APC initial-state or last-scheduled-token index tensors. The unchanged
  APC path remains available through the original load branch.
- BF16 convolution state with at least eight token slots, 6144 features and
  non-overlapping positive-stride DS or SD storage, including slot padding.
- Every operand is on the same CUDA device. Conservative byte intervals of
  x and conv_state are disjoint from each other and from all read-only
  weights, bias, state indices, accepted-token and query-offset tensors.

The interval computation uses the actual view data pointer and positive
strides, so storage offsets and holes are included. Conservative rejection
of disjoint holes is acceptable. Read-only operands may alias each other.
Internal channel, token and cache-slot overlap in conv_state is rejected.
The guard never treats an unknown or overlapping layout as evidence for
selection. Inspection errors return false before dispatch.

The alias checks matter: the original kernel writes conv_state before its
second raw-input read. If raw x or any source used after that write shared
bytes with conv_state, taking an earlier snapshot could change semantics.
Qualified storage excludes this case. With output identical to x, each
channel's earlier raw values remain in the packet while the original serial
loop overwrites that channel's output rows. The retained grid assigns each
feature to exactly one CTA; the view has no overlapping rows or channels.

The environment switch `VLLM_CONV_REUSE_PACKET=0` disables host selection.
It is read as an ordinary optional kernel switch, not a workload-result
cache. Other model formats and unsupported shapes keep the original load.

## Numerical and graph limits

The contract's non-timed fidelity gate permits a token budget, but its timed
window requires exact committed tokens at the declared depth. That is the
acceptance condition for this draft. Allclose on a convolution output would
not establish it. BF16 logits can tie or differ by a single BF16 step.

The source-level operand proof covers finite values, signed zero, subnormal
BF16 values and NaNs: selection itself performs no floating arithmetic. The
original downstream arithmetic is left in place, including its BF16 operand
types. This is not a claim that the compiler emits identical floating-point
instructions, layouts or final bytes. Different liveness/layout lowering can
still change generated code, so output and full-state bit comparisons and
the official token gate are necessary on the pinned remote runtime.

The existing PDL wait remains before all mutable input/state loads. The
existing signal to dependents, output stores and graph topology remain in
their original positions. No additional stream, event, graph, persistent
scratch, request data, pointer ticket or owner cache is introduced. The
reused packet is local to this kernel invocation and is overwritten by the
next invocation's ordinary input loads; nothing is retained across requests
or phases.

During FULL graph capture, metadata inspection selects a specialization for
the concrete captured tensor views. Replay relies on the existing runner's
graph input/owner and allocation lifetime rules, as the baseline does. Query,
accepted-token and slot tensor contents remain device inputs refreshed by
that runner and reloaded by the kernel. The original unsupported-path
specialization remains available when capture uses different tensor metadata.
There is no claim that all real frames meet the host predicate or that a
production selection count can be inferred from a whole-run score.

## Failure behavior

Host eligibility refusal occurs before this kernel is dispatched and does
not mutate any tensor. It merely selects the retained raw-load specialization.
The helper's exception handler covers metadata inspection only. It does not
wrap a JIT launch, a CUDA graph capture or a replay.

There is no catch-and-retry after a GPU launch. A compiler, launch, capture or
runtime failure is visible and may fail the candidate leg. The draft does
not promise that such a failure leaves the whole model frame or cache
unchanged: other operations may already have run, and a launched kernel may
have partially written state. Reusing that state for an eager retry would be
unsafe and is not implemented. A compiler failure before the kernel launch
cannot have written this kernel's state, but that narrower fact does not make
the surrounding model transaction atomic.

No private admission tests or timing loops are inserted into production or
startup. There are no claims of completed private qualification. The next
validation must run only through the authorized official remote environment.
It should compare retained-kernel and packet-kernel output/state words on
private buffers, including query lengths0..6, accepted-token variants, NULL
slots, exact projection pitch, state padding, unsupported aliases and changed
inputs over graph replays. The whole candidate's timed token result and score
remain decisive; a whole correctness pass alone does not establish kernel
selection or an isolated causal gain.

## Traffic bound and expected costs

At six real verification rows the maximum source-level repeated raw read
removed across the 23 target Mamba layers is

```
23 * 6144 * 6 * 2 = 1,695,744 bytes = 1.6171875 MiB
```

The raw projection write, convolution output write, halo reads, weight reads,
state update traffic and all kernel launches remain. This is not a model
weight-bandwidth reduction. The original repeated reads may be cache hits,
and compiler analysis may already eliminate or transform some loads. No
physical HBM saving or speedup is inferred from this byte bound.

Keeping an 8x128 BF16 logical packet alive across the state write and weight
loads can increase registers. A dynamic gather can introduce selectors,
shuffles, shared-memory traffic or spills. The graph's host eligibility cost
is paid while the operation is captured; uncaptured calls also pay metadata
inspection. Either cost can outweigh the removed source load. Retain the
draft only after actual remote correctness and a full-score comparison to
the latest verified frontier under the user's 0.1 percent regression limit.

## Direct materials and rejected alternatives

The complete inherited implementation and VENDOR history remain in the c41
baseline. The direct materials consulted for this new design are:

- c41fc629-8c06-4bf2-8cb8-cad0edea9875: entire validated source baseline.
- b9515169-9387-4c51-b8da-068236911af9: earlier prefill eight-token packet
  design and negative tuning evidence. That whole passed correctness but
  scored 2.7054982605319373, 0.4886 percent below its then-baseline
  4ae55271-d4bd-44f6-adf0-1f731b273822. Its prefill kernel and private
  qualification implementation are not imported. It informs the decision
  to preserve decode's original serial arithmetic and reuse only the packet
  that the original decode kernel already loads.
- fc04335b-c894-41f9-b73b-aa9b5aac1fb5: inherited FP8 projection codec/GEMM
  inspected for the larger in-projection/conv fusion. That alternative is
  not implemented and the FP8 projection sources are unchanged.
- cffcbd4a-ca6e-4d14-95c3-4c23b81f4dee: earlier convolution-to-SSM-ring
  fusion examined as negative evidence. Its whole scored 2.7515118597488613,
  0.7702 percent below its then-baseline
  de4975c1-0182-4fe4-8a7f-70ac7e5f8af1. Its ring writer/consumer changes
  are not imported or retried.

Listing a rejected alternative records provenance, not adoption of its low
whole score or a claim that its individual component caused that outcome.

Relevant primary API references are Triton's
[gather](https://triton-lang.org/main/python-api/generated/triton.language.gather.html)
and [type semantics](https://triton-lang.org/main/python-api/triton-semantics.html).
The inspected upstream gather implementation returns the source scalar type
with the index tensor's shape; its two-dimensional non-axis shape must match.
The actual pinned compiler version and this specialization's generated code
have not been executed or independently inspected.

## Static review and handoff

An independent source-only reviewer examined the actual implementation blob
191e29186a081117dd8d81c4b7051491c5170a9e, including the retained 28 positional
launch arguments, the appended keyword constexpr, the metadata-only guard,
byte intervals, two-dimensional gather shape, dynamic query fallback and
unchanged PDL/state/arithmetic ordering. The reviewer found no statically
established blocker in that snapshot. This is an actual-code review, not only
a design review, and it does not establish compiler support or runtime
selection. The reviewer did not execute, parse, compile or test the candidate.

The draft commit is created with Git hooks and commit signing disabled. The
exact commit and tree are supplied in the handoff rather than embedded as a
self-referential hash in this document. The working tree must be clean after
freezing. Upload, final frontier comparison and official remote execution
remain outside this implementation handoff.

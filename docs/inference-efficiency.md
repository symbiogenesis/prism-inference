# CPU inference efficiency candidate

The current source combines direct one-column F32 adapter and streaming BF16 projections,
canonical-empty rollback checkpoint, native-order top-20 sampler, redundant
empty-reset elimination and exact masked-attention padding pruning. Runner's
released 0.1.22 pin does not include later local source changes described below.
Source work and storage reductions are not measured end-to-end gains or permission
to lower admission limits.

BF16 projections now consume each immutable weight directly from its two-byte
payload after one complete tensor-geometry check. They preserve BF16 activation
rounding, each separately rounded F32 product, the original binary64 sum and its
final F32 rounding, including pinned execution. Every stored weight still receives
its finite check; malformed geometry, nonfinite payloads/activations and overflow
still reject with empty output. The decoded-row allocation is neither populated
nor changed, and rows remain sequential. Swift's 48 recurrent layers each have
two 5,120-by-48 BF16 alpha/beta matrices. Avoiding their F32 row materialization
removes a nominal 90 MiB of scratch writes and the matching rereads per processed
token; this is a source work count, not measured memory traffic or throughput.
Independent decoded-row controls cover pinned/unpinned requests, backend requests,
awkward widths through 5,120, signed zeros, subnormals, finite extremes, failed
payload/activation/geometry admission, output clearing and scratch preservation.

The sampler now checks every logit while selecting the bounded heap, avoiding a
separate full-vocabulary validation traversal. A nonfinite value anywhere,
including an otherwise discarded tail candidate, rejects before drawing RNG.
The reusable allocation survives rejection, and ties, signed zeros, heap order,
probability arithmetic and the distribution's draws retain their existing contract.

Runtime-owned sampling now passes an immutable `FiniteLogits` view after the
complete decoder-output validation. Public views validate the whole
nonempty bounded slice once, and their borrow prevents subsequent mutation.
Greedy selection and the fixed Swift top-20 heap omit finite predicates for
that view, including the four vector finiteness masks per skipped 16-logit
NEON block; their threshold comparisons and token order are unchanged. Ordinary
public sampling still rejects nonfinite values anywhere. Custom `Sampler`
implementations keep their existing `sample` entry point through the default
`sample_finite` method; the CLI's sampler wrapper forwards the typed view.

The additive `Decoder::evaluate_finite_into` boundary now carries that proof
from native Qwen35 and scoped projected Qwen35 evaluation into Runtime. Both
graphs already check every output before committing layer state; Runtime checks
the proof's storage pointer, complete extent and current vocabulary size instead
of scanning every float again. Its compatibility default independently checks
custom `evaluate_into` implementations. A checked view of unrelated storage or
only a slice of the output rejects before token publication or sampling.
At Swift's 248,320-logit vocabulary, native evaluation omits one traversal of
993,280 bytes per evaluated token, including the final generated token. This is
a source traversal count, not measured cache traffic or latency.

Every prefix publication still requires complete finite output, including the
last generated token. Only its private validated logits and independent
snapshots create runtime sampling views. Ingestion
invalidates old logits; cancellation, malformed output and failed continuation
restore the earlier validated snapshot and sampler state. Focused controls cover
all NEON lanes, finite extremes, signed-zero/tie ordering, candidate/probability
bits, seeded draws, clone/reset/rejection, custom dispatch and decoder failures
before and after sampling. These are source work reductions, with unchanged
sampling vocabulary reads, storage, admission and qualification gates; no measured
request-time or throughput benefit is asserted.

The fixed Swift chain now retains its validated top-p state at sampler
construction. Native min-p zero returns without mutation, and temperature one
preserves every admitted finite logit bit, so those identity stages need no
temporary states. This omits three temporary configuration-vector allocations
per output token and retains one additional sampler-state metadata value.
Independent controls replay the original three-stage chain across candidate
bits, ties, signed zero, subnormals, extremes, RNG draws, cloning, reset and
full-vocabulary rejection. This is a source allocation/work reduction; it does
not establish lower process RAM, request latency or reduced admission.

The fixed Swift top-p/distribution pair now retains its 20 binary32
unnormalized softmax weights in an 80-byte stack array. Both stages use the
same sorted maximum and separately rounded subtraction/exponential, so the
distribution can reuse those values after top-p truncation. Top-p still divides
every original candidate by its binary32 sum; the distribution independently
sums retained raw weights in binary64, consumes its original draw and performs
its original division/rounding. Candidates outside the retained view preserve
their original top-p probability bits. The generic sampler stages and private
unsorted fallback remain independent. This omits one exponential per retained
candidate when there are at least two, at most 20 per generated token; the
singleton distribution still consumes one draw and stores probability one.
Independent original-chain controls cover all backing bits, singleton and
truncated views, ties, signed zero, subnormals, extremes, underflow, clone/reset
and RNG progression. Existing full-vocabulary rejection and transactional
cancelled/abandoned-stream controls remain required. This is a source work
reduction; actual stack frames, model parity and matched complete-request
benefit remain pending without a latency, RAM or admission claim.

Full-attention decode and tiled prefill also read immutable gate heads from
their original interleaved projection output. The separate gate copy is
omitted while query normalization, sigmoid and attention arithmetic keep their
order. See the [tiled-prefill contract](cpu-tiled-prefill.md) for source work
counts and the unchanged resource and qualification boundaries.

The explicit suffix drafter now uses reverse Z matching against the current
suffix with the existing 64-token maximum window. Its private 64-byte prefix
table reuses already proved matches instead of comparing a long common suffix
again at every historical end. Prefix preparation and the history scan require
fewer than `2 * (history tokens + maximum match tokens)` token equalities;
this includes failed comparisons. It preserves overlapping matches,
longest-match priority and the most recent start on ties. The previous
backwards scan could repeat up to 64 comparisons per historical end.
Runtime generation copies at most eight continuation tokens into a fixed
stack buffer reused for the request; it no longer allocates a proposal vector for
each round. The public `SuffixDrafter::propose` API still returns an independently
owned, fallibly allocated vector. Invalid configurations reject before proposal
storage changes, even with zero remaining tokens.

This is bounded source-work reduction in the opt-in suffix path, not evidence
that serial greedy verification is faster than ordinary decoding. Decoder work,
state checkpoints, cancellation checks around proposal/verification, output limits
and exact greedy decisions retain their contracts. Exhaustive binary histories
through ten tokens and 8K maximum-window controls compare both proposal routes
against the retained original forward, length-first search. Periodic, overlapping
and adversarial near-match controls count the actual algorithm's equalities and
check the linear bound. The 64-byte table is additional private source payload;
actual stack use, code generation and typical-prompt costs remain unmeasured.
Owning hosted checks, actual-model state/adapter parity and matched
complete-request benefit remain
pending before any performance qualification or default change.

Exact NEON GDN multi-token helpers now reuse the private state staging array for
decay and update. Each old block is dead after its decay multiplication; each
decayed block is dead after its checked update. Removing the duplicate array
eliminates 1 KiB, 4 KiB or 8 KiB of declared staging payload in the one-, four-
and eight-row helpers, respectively. The two floating passes, reduction trees
and token order are unchanged. Grouped helpers still publish state and outputs
only after every token succeeds, and single-row failure still publishes exactly
the updated finite prefix followed by untouched decayed elements. Caller-owned
snapshots never share the private staging array. Direct Windows ARM64 controls
compare all state/output bits with an independent scalar oracle and exercise
late key-dot, update and query-dot failure without relying on dispatch fallback.
These are source storage reductions; Release code generation, actual stack use,
full-model parity and matched request benefit remain pending. No scratch or
worker-stack admission reserve is reduced.

Single-token NEON GDN steps, including normal four/eight-row groups, now select
an eight-block private staging capacity
for rows up to 128 elements, retaining sixteen blocks for eligible wider rows
through 256 elements. Both capacities instantiate the same arithmetic and error
publication implementation. Swift's 128-wide rows therefore declare 512 bytes
of staging instead of 1,024 per row; four/eight-row groups halve their 4/8 KiB
payloads to 2/4 KiB. This source payload reduction does not establish
generated zero stores, actual stack use or memory traffic. Independent scalar
bit controls cover both sides of the capacity boundary, exceptional values,
late failures and unchanged unsupported-width refusal. Conservative resource
reserves remain charged.

Exact multi-token NEON GDN prefill and state-only replay now select the same
eight-block capacity for rows through 128 elements, retaining sixteen blocks
for supported wider rows through 256. The one-, four- and eight-row prefill
helpers and four/eight-row replay helpers instantiate their existing arithmetic
with the selected capacity. For Swift's 128-wide rows, their private state
staging declares 512 bytes per row instead of 1,024; eight-row groups declare
4 KiB instead of 8 KiB. Output staging, token limits, FMA/reduction order,
publication and scalar failure fallback remain unchanged. This does not prove
fewer generated stores, smaller compiled stack frames or measured latency.
Direct scalar-bit controls cover zero through 64 tokens, both capacity ranges,
strided output sentinels, late key/update/query failure and unsupported shapes
or excessive token counts. Release ARM64 code generation/stack review, hosted
controls, complete-model state/adapter parity and matched request benefit remain
pending. No admitted scratch or worker-stack reserve is reduced.

Suffix generation retains the already validated target token when a proposal
is rejected. The correction boundary consumes that token without a second
full-vocabulary greedy scan; no evaluation or callback changes logits between
the two points. Cancellation checks remain on both sides of correction
selection. Independent fixed token sequences and ordinary-generation controls
cover initial and partial rejection, accepted chains, bonus tokens, EOS,
output limits, decoder traces, logits and transactional cancellation/recovery.
This omits one greedy traversal per rejected round in the opt-in path; it
does not qualify speculative benefit or alter selection defaults.

Resident native-Q2 handles retain their immutable HVX solver parameters from
weight preparation. Every resident descriptor reuses those parameters,
including batched members, extra arenas and fused FFN projections; transient
descriptors continue to solve their active geometry. Portable controls compare
independent allocation algebra and kernel-parameter bytes across the 6,144-column
VTCM threshold, complete resident/transient frames and rejected reservation
publication. The runtime charges the larger handles through its existing
`size_of`-based metadata reservation within the unchanged cap. No registry
allocation, signed kernel, wire layout, placement or session limit changes.
Avoiding repeated host solver searches does not establish end-to-end benefit.

Grouped NEON GDN decode, multi-token prefill and state-only replay now combine
updated-lane finite masks while their rows remain private, with one horizontal
mask reduction before group publication. At width 128 this replaces 32 or 64
update-mask reductions per four/eight-row token group with one per helper call;
multi-token helpers combine the masks across their bounded token sequence.
Floating arithmetic, key/query checks, single-row partial publication and scalar
failure replay remain unchanged. Direct independent scalar-bit controls cover
both staging capacities, late overflow and untouched grouped destinations.
This source reduction needs Release ARM64 codegen, full-model state/adapter parity
and matched whole-request benefit before performance qualification.

Native-Q2 multi-operation packet construction now serializes each distinct
resident geometry once per packet. Repeated geometry copies only populated
immutable tensor/operation fields and relocates all member-specific offsets,
buffer/tensor indices and operation references. It shares no template across
packets and adds no persistent allocation or metadata charge. Handle/arena
validation, activation deduplication, output poisoning, operation order and
submission limits remain unchanged. Full-frame controls compare the original
serial descriptors across packet widths, arenas, activation identities and
fused wrappers; rejected geometry must preserve its original failure behavior.
Avoiding repeated field conversions establishes no measured DSP or request speedup;
the extra descriptor reads still need a matched total-cost comparison.

## Exact hosted correctness gate

The owner-operated [Inference efficiency candidate workflow](../.github/workflows/inference-efficiency.yml)
runs only on hosted Windows ARM64, Windows x64 and Linux ARM64 against local main's
published source. It never queues an owner-device job. Its fixed inventory tests
direct projections against decoded rows, full-vocabulary selection against the
old full-candidate route, independent full-padding attention controls, empty and
nonempty snapshot ownership, adapter provenance, cancellation and seeded rollback.
The fused copy/decay test remains a synthetic control, not production selection.

The runner compiles its six test targets once and requires exactly one passed,
nonignored test for every exact name. Missing or renamed controls, zero matches,
ignored tests, duplicate executables and failures reject. Receipts retain the
revision, source hashes, executable digests and raw results. Strict Release
all-target Clippy and production executable linking are separate required steps;
a correctness receipt alone does not imply that later steps passed. The complete
workflow result is the model-free hosted gate.

The source receipt also binds the root `clippy.toml` policy. Its two named
duplicate-crate allowances cover incompatible pinned requirements:

- `windows-sys`: containment's `mxc-sdk`/`winreg` route requires 0.59, while
  native hashing and other current consumers require 0.61.2.
- `syn`: `bonsai-hexagon-runtime -> serde 1.0.229 -> serde_derive 1.0.229`
  requires major 3 (locked 3.0.9). Its Windows containment route through
  `bonsai-windows-process =0.1.2 -> mxc-sdk =1.0.1 -> windows-core 0.62.2`
  requires the nonoptional `windows-implement 0.60.2` and
  `windows-interface 0.59.3` macros, which each require major 2 (locked
  2.0.119). Static inspection of the lockfile and cached normalized manifests
  establishes these incompatible requirements; it is not a compiled result.

Neither allowance replaces dependencies or disables other duplicate-crate
diagnostics. Clippy's policy names crates rather than individual versions, so
future changes to either route require review. Changing or removing the policy
changes the admitted source identity. The `syn` extension is source-only after
the closed local check's strict analyzer failure and has not been compiled or
checked. The earlier failure remains failed; final-source owning hosted
validation is pending.

The fixed inventory also covers suffix-search reference parity and speculative
rollback, staged GDN row reuse on Windows ARM64, and direct native-Q2 resident
weight initialization. Accelerator controls use host-only fake arenas and mock
submitters; they neither open a driver nor execute a DSP graph. Windows ARM64
alone selects the native arena controls. Portable runtime controls still compare
directly packed bytes with the existing copied route. The receipt hashes both
accelerator tool crates, and their Release analyzers and linking remain part of
the same hosted workflow. This expands correctness coverage without qualifying
NPU placement, model output or startup benefit.

Lazy native-Q2 resident preparation now shares the direct initialization path
without allocating an unused packed-weight vector first. Copied preparation
grows bounded scratch only after resident admission; transient execution sizes
its scratch after selecting that route. Fake-arena controls cover direct/copied
bytes, refused capacity, malformed scales and terminal publication failure
across all eight lazy entry families, with a separate successful transient
control. The ordinary prewarmed default is unchanged by this follow-up.
Conservative admission remains charged; no startup or execution gain is measured.

Single-operation transient native-Q2 token tiles now pack immutable PQ2 weights
once for the tile when residency is absent or refused. The call-local packed
buffer is reused across the same ordered `N = 1` submissions and then released;
no persistent expanded copy or new DSP batch is introduced. For an 8/16/32-token
tile this omits 7/15/31 full host transcoding passes and their scratch allocations,
while operation counts, queue writes and conservative payload admission stay
unchanged. Multi-operation tiled shapes and resident dispatch retain their
existing routes. Independent digit-level layout/scalar controls cover original
fourth-symbol values, token order, absent/refused residency, input and scale
failure precedence, late failure, output clearing and terminal refusal. These
host-only controls do not qualify signed-device arithmetic or prompt latency.

Scoped matrix preparation now carries its existing complete-input finite proof
into the Q8_K, Q8_0 and pinned PTQ1 quantizers. Their mandatory maximum scans
omit duplicate finite predicates, including the baseline NEON input masks and
their horizontal reductions. The original complete-input preflight still runs
before any quantizer mutation, preserving malformed-shape, nonfinite-input and
scale-overflow precedence. At 5,120 columns this omits 5,120 repeated input
finite predicates per preparation; it does not omit the maximum traversal or
measure a latency gain. Standalone row encoders retain their own input checks.
Scale, code, output-rounding and finite-result checks remain in every path.
The proof borrows immutable input only for the call and adds no retained storage.
Controls compare every prepared scale/code/sum bit with the original quantizer
entry points, preserve failure precedence and scratch invalidation, and cover
overflow, subsequent recovery and external-executor validation. Defaults,
arithmetic, admission and installed runtime identities remain unchanged.

## Device and delivery acceptance

No workflow dispatch, source change or source check qualifies actual Swift
logits, full live state, complete answers, resources or performance. Before release
integration, retain exact-source PQ2 and independently scoped PTQ1/adapter checks,
including zero-scale identity and nonzero application, seeded output, reset,
request rollback and explicit original-PQ2 recovery. Compare matched complete
requests and resource observations with the released source on a qualified target.
Keep prompt ingestion, first-token latency and generation intervals separate.

Owner-device setup, loading, every arm, power observations, cleanup and retention
share one predeclared aggregate allowance under AGENTS.md. The default is five
minutes; choose a longer supported allowance only from a fixed conservative work
estimate. A failed or infeasible attempt closes its clock, and full suites and
benchmarks remain CI-only. This document declares no new model workload.

Production recurrence keeps copy-then-scale. Fused copy/decay needs its own
actual-model state/logit, throughput and resource gate before production use;
removing a 144 MiB copy per processed token is not evidence that this gate passed.
After successful owning checks, the owner publishes an immutable qualified release.
Runner must verify that release's source, archive and executable identities before
repinning. No workflow here publishes, pushes, dispatches another workflow or
changes the installed application.

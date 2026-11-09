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

The fixed Swift chain now retains its validated top-p state at sampler
construction. Native min-p zero returns without mutation, and temperature one
preserves every admitted finite logit bit, so those identity stages need no
temporary states. This omits three temporary configuration-vector allocations
per output token and retains one additional sampler-state metadata value.
Independent controls replay the original three-stage chain across candidate
bits, ties, signed zero, subnormals, extremes, RNG draws, cloning, reset and
full-vocabulary rejection. This is a source allocation/work reduction; it does
not establish lower process RAM, request latency or reduced admission.

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

The source receipt also binds the root `clippy.toml` policy. Its one named
duplicate-crate allowance covers the incompatible pinned `windows-sys` versions:
containment's `mxc-sdk`/`winreg` route requires 0.59, while native hashing and
other current consumers require 0.61.2. This does not replace dependencies or
disable other duplicate-crate diagnostics; changing or removing the policy
changes the admitted source identity.

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

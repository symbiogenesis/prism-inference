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

The explicit suffix drafter now visits historical match ends once, comparing
backwards against the current suffix with the existing 64-token maximum window.
It preserves overlapping matches, longest-match priority and the most recent
start on ties, while avoiding a separate history scan for every permitted match
length. Runtime generation copies at most eight continuation tokens into a fixed
stack buffer reused for the request; it no longer allocates a proposal vector for
each round. The public `SuffixDrafter::propose` API still returns an independently
owned, fallibly allocated vector. Invalid configurations reject before proposal
storage changes, even with zero remaining tokens.

This is bounded source-work reduction in the opt-in suffix path, not evidence
that serial greedy verification is faster than ordinary decoding. Decoder work,
state checkpoints, cancellation checks around proposal/verification, output limits
and exact greedy decisions retain their contracts. Exhaustive binary histories
through ten tokens and 8K maximum-window controls compare both proposal routes
against the retained original forward, length-first search. Owning hosted checks,
actual-model state/adapter parity and matched complete-request benefit remain
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

# CPU inference efficiency candidate

The current source combines the direct one-column F32 adapter projection,
canonical-empty rollback checkpoint, native-order top-20 sampler, redundant
empty-reset elimination and exact masked-attention padding pruning. The released
0.1.14 Runner pin predates these changes. Their source work and storage reductions
are not measured end-to-end gains or permission to lower admission limits.

The sampler now checks every logit while selecting the bounded heap, avoiding a
separate full-vocabulary validation traversal. A nonfinite value anywhere,
including an otherwise discarded tail candidate, rejects before drawing RNG.
The reusable allocation survives rejection, and ties, signed zeros, heap order,
probability arithmetic and the distribution's draws retain their existing contract.

## Exact hosted correctness gate

The owner-operated [Inference efficiency candidate workflow](../.github/workflows/inference-efficiency.yml)
runs only on hosted Windows ARM64, Windows x64 and Linux ARM64 against local main's
published source. It never queues an owner-device job. Its fixed inventory tests
direct projections against decoded rows, full-vocabulary selection against the
old full-candidate route, independent full-padding attention controls, empty and
nonempty snapshot ownership, adapter provenance, cancellation and seeded rollback.
The fused copy/decay test remains a synthetic control, not production selection.

The runner compiles its four test targets once and requires exactly one passed,
nonignored test for every exact name. Missing or renamed controls, zero matches,
ignored tests, duplicate executables and failures reject. Receipts retain the
revision, source hashes, executable digests and raw results. Strict Release
all-target Clippy and production executable linking are separate required steps;
a correctness receipt alone does not imply that later steps passed. The complete
workflow result is the model-free hosted gate.

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

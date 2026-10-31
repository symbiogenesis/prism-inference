# Explicit packed-projection row workers

The `parallel-rows` Cargo feature provides a private persistent Rayon 1.12.0
pool for PQ2/PTQ1 matrix rows. It is opt-in and all constructors retain serial
execution until configured. It does not promote a device profile or establish
a speedup. Full-model and target-device qualification remain separate.

The implementation independently adapts the borrowed-slice scheduling idea in
[OxiBonsai parallel.rs](https://github.com/cool-japan/oxibonsai/blob/1d10c48e14754ff6999158f8fa13c7c92b40b0dc/crates/oxibonsai-kernels/src/parallel.rs)
at `1d10c48e14754ff6999158f8fa13c7c92b40b0dc` (Apache-2.0, COOLJAPAN OU).
It retains Prism's original packed formats, quantizers, kernels and scale order.
No Oxi source text or serialized fixtures are copied.

## Resource and arithmetic contract

- `RowParallelism::new(n)` accepts 1 through 16 workers. One creates no threads;
  higher values require the feature. Pool creation failure is an error rather
  than a silent fallback. Environment/global Rayon settings do not choose the
  worker count.
- Policy clones share one persistent pool. Each worker requests a 512 KiB stack,
  charged in addition to ordinary graph/model storage by the process owner.
  Different independently created policies create independent pools; callers
  should share one policy rather than multiply worker budgets.
- One projection prepares activations once. Contiguous row tasks borrow the
  original immutable packed bytes and Q8_K/Q8_0 arrays, then write disjoint
  regions of the already reserved output. No matrix, activation or per-worker
  weight copy is created. At most the configured number of row tasks exists.
- Each row executes the same selected scalar/NEON dot function as serial mode.
  Integer grouping and every floating scale/add remain unchanged. Floating and
  other packed types retain their prior serial paths; single-row work is serial.
- Per-task failure storage is bounded by the worker count and reused. All tasks
  join before returning the first failure in row order. Failed matrix output is
  empty; graph/runtime transaction rollback retains its existing behavior.
- Cancellation/deadline checks remain cooperative between synchronous decoder
  operations. Row parallelism does not preempt a matrix operation or add a
  separate cancellation claim. Reset and checkpoint restore preserve the worker
  policy, while checkpoints contain no pool, scratch or mutable output aliases.

## Library, worker and private HTTP entry points

Library callers construct a policy before loading a model:

```rust,no_run
use bonsai_prism_runtime::kernels::RowParallelism;
let policy = RowParallelism::new(2)?;
// graph.set_row_parallelism(policy.clone());
// SwiftSession::load_with_row_parallelism(path, backend, seed, policy)?;
# Ok::<(), bonsai_prism_runtime::Error>(())
```

`Qwen35::row_workers()` reports the configured budget. Its existing
`backend_statistics().parallel_row_operations` counts actual packed attempts
using the pool, including provisional or failed operations. It is distinct
from scalar/NEON ISA selection and does not count committed tokens.

Build `prism-inference` or `prism-inference-server` with `--features parallel-rows`.
The prepared `0.1.9` GitHub release-bundle build includes this feature while
retaining the one-worker default. Earlier immutable bundles are unchanged.
The Swift JSONL worker and private `--native-engine-child` HTTP server accept
`--row-workers 1..16`, default 1. Values above one reject if built without the
feature. This is a trusted launch option, never a client-controlled HTTP field;
all existing authentication, model identity, process containment and memory
admission requirements still apply. Public request/readiness schemas are unchanged.

The source-only `swift_model_check` example accepts the same explicit budget for
`load`, `forward` and `trajectory`; `inspect` rejects worker settings. Prefix
order is representation, row workers, then adapter. For example, a hosted build
with `--features parallel-rows --example swift_model_check` can capture separate
serial and two-worker controls for the original PQ2 artifact:

```text
swift_model_check --row-workers 1 --rank-one-adapter <approved-adapter> 1.3 forward <original-PQ2> <new-serial-evidence> auto 248045
swift_model_check --row-workers 2 --rank-one-adapter <approved-adapter> 1.3 forward <original-PQ2> <new-parallel-evidence> auto 248045
```

All artifact/evidence paths must be absolute; evidence directories must already
exist and differ. These are capture interfaces, not instructions to run a long
matrix on the owner device. Preserve its five-minute total budget and leave
unrun gates pending. The checker keeps exact model/adapter identities, create-new
logits and complete logical KV/recurrent/convolution dumps and existing byte
bounds. Forward/trajectory step records also report the configured `row_workers`,
cumulative actual `parallel_row_operations` and `evaluation_milliseconds`.
Evaluation time covers only `Qwen35::evaluate_into`, including its state checks
and updates, and excludes model loading, evidence I/O, greedy selection and output.
It is a raw observation, not complete-request latency, service TPS or proof of a
speedup. The example uses the existing graph policy API directly so it can retain
complete state dumps; it does not expose the private Swift session's decoder.

## Validation

Focused controls cover explicit budgets, actual Swift column widths, awkward
row counts, both PQ2 activation contracts, PTQ1, prepared siblings, scalar/NEON
selection, pinned accumulation, malformed payloads, failed output cleanup,
floating fallback, borrowed row addresses and deterministic failure order.
Graph controls cover full-attention/recurrent logits, checkpoint restore,
reset and failed evaluation recovery. CI's feature-variant lane executes these
with the feature enabled; ordinary feature-disabled builds retain rejection
controls. Full suites and every benchmark remain CI-only.

The optional dependency is permissively licensed (`rayon`, `rayon-core` and
their enabled dependencies offer Apache-2.0/MIT terms). Distributors retain the
selected upstream licenses and notices from the exact locked crates. Adding a
feature does not qualify its thread scheduling, end-to-end latency, sustained
memory or energy on Snapdragon.

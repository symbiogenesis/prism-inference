# Packed-projection row workers

The opt-in `parallel-rows` feature provides a private persistent Rayon pool for
PQ2/PTQ1 matrix rows. Constructors remain serial until configured. Worker count
does not establish target-device throughput or energy benefit.

The scheduling design independently adapts borrowed-slice scheduling from
[OxiBonsai parallel.rs](https://github.com/cool-japan/oxibonsai/blob/1d10c48e14754ff6999158f8fa13c7c92b40b0dc/crates/oxibonsai-kernels/src/parallel.rs)
(Apache-2.0, COOLJAPAN OU). No source text or serialized fixtures are copied.

## Resource and arithmetic contract

- `RowParallelism::new(n)` accepts 1..16. One creates no threads; larger values
  require the feature. Pool-creation failure rejects. Environment/global Rayon
  settings do not select the count.
- Policy clones share a pool. Each worker requests a separately charged 512 KiB
  stack; independently created policies create independent pools.
- A projection prepares activations once. At most the configured number of tasks
  borrow immutable packed weights and activations and write disjoint existing
  output slices. No per-worker matrix or activation copy is retained.
- Every row uses the serial path's selected scalar/NEON dot, integer grouping and
  scale/add order. Floating/other formats and single-row projections remain serial.
- Bounded per-task error storage is reused. All tasks join before returning the
  first error in row order. Failed matrix output is empty.
- Cancellation is cooperative between synchronous operations. Reset/restore
  preserve worker policy; checkpoints contain no pool or scratch aliases.

## Library, worker and private HTTP entry points

Construct the policy before loading:

```rust,no_run
use bonsai_prism_runtime::kernels::RowParallelism;
let policy = RowParallelism::new(2)?;
// graph.set_row_parallelism(policy.clone());
// SwiftSession::load_with_row_parallelism(path, backend, seed, policy)?;
# Ok::<(), bonsai_prism_runtime::Error>(())
```

Build with `--features parallel-rows`. The Swift JSONL worker and private native
HTTP child accept trusted `--row-workers 1..16`, default 1. Values above one
reject without the feature. HTTP clients cannot override it. Readiness reports
the instantiated session's `row_workers`.

`Qwen35::row_workers()` reports policy.
`backend_statistics().parallel_row_operations` counts actual packed attempts,
including failed/provisional work; it is separate from ISA selection and committed
tokens. [CPU tiles](cpu-tiled-prefill.md) share this pool.

The source correctness tool accepts the same option before adapter selection:

```text
swift_model_check --row-workers 2 --rank-one-adapter <approved-adapter> 1.3 forward <original-PQ2> <new-evidence> auto 248045
```

Use absolute artifact paths and a distinct existing evidence directory.
`inspect` rejects worker settings. Captures include worker count, operation
count and `evaluation_milliseconds`; the last measures only
`Qwen35::evaluate_into`, excluding loading, evidence I/O and output. It does not
measure complete-request latency. Follow the [capture procedure](build-and-validation.md)
and [owner-device allowance](../AGENTS.md#owner-device-execution-budgets).

## Validation

Controls cover worker bounds, Swift and awkward row shapes, both PQ2 activation
contracts, PTQ1, prepared siblings, backend/pinned arithmetic, malformed input,
output clearing, borrowed storage, deterministic failure, graph continuation,
checkpoint/reset and recovery. Feature-disabled builds retain rejection controls.
See [building and validation](build-and-validation.md) for owning checks.

Distributors retain the exact locked Rayon's Apache-2.0/MIT licenses and notices.

# Experimental native Hexagon graph integration

This unpublished crate joins the Qwen3.5 graph to a caller-owned X2 projection
session. The main CPU runtime depends on neither it nor the
[native driver boundary](../hexagon-native/README.md).

## Executors

All executors implement `Pq2ProjectionExecutor`. The graph applies its original
Hadamard/sign/grouped transforms before projection and approved rank-one adapter
corrections afterward. Embedding, attention, GDN, convolution, sampling and
committed state remain in the Rust graph.

| Executor | Arithmetic |
| --- | --- |
| `GuardedQ2Executor` | Guarded HVX K=256/N=2 with prepared Q8_K codes |
| `HmxQ2Executor` | HMX K=128/N=8 bitplanes of one token's Q8_K codes; signed high bit has coefficient -128 |
| `HmxQ8Executor` | Explicit PTQ1/odd-block PQ2 with Q8_0 codes and per-32 scales; four sparse rows preserve DOTPROD lanes |
| `NativeQ2Executor` | Native resident Q2_0 preparation and projection |

Original scale bits and per-128 fused/nonfused reductions remain on CPU.
Pinned PTQ1 retains four FMA accumulators and adjacent-pair reduction; generic
PTQ1/odd-block PQ2 retain their per-32/per-128 nonfused order. PTQ1 byte aliases
keep decoder semantics. Its fourfold row duplication is an explicit compatibility
path, without an efficiency claim. Unsupported arithmetic stays on CPU before
submission. Any submitted error clears provisional output and permanently closes
the executor and scoped decoder; failed work is never replayed on CPU.

Use `Qwen35::with_projection_executor` inside the native driver's projection work
callback, then place the borrowed `ProjectedQwen35` in an ordinary `Runtime`.
Its borrow covers all tokens and projections in that one session. Request
checkpoints, cancellation and prefix policies remain active. Keep outputs
provisional until the native report's `completed_cycle()` succeeds.

## Placement and work admission

`PlacedProjectionExecutor` freezes a model-tensor subset before graph construction,
for example `blk.0.ffn_up.weight`. Unknown names, duplicates, unsupported arithmetic
and changed immutable storage/geometry reject. Identity binds the model, executor,
selected names and operation counts. Qualification must observe nonzero submissions
for the intended placement; selecting a model tensor does not prove graph use.

Executors report stable, nonzero operation counts for each selected projection.
`ProjectedQwen35` derives ingestion/evaluation costs from graph projections,
excluding embedding gathers and CPU placement. `Runtime::admit_work` reserves all
unreused prompt work and the maximum output before inference or sampling; early
stop is not assumed. Direct graph calls admit one complete token. Capacity
refusal leaves a healthy scope; errors after admitted evaluation starts are
terminal. Checkpoints/reset never replenish capacity.

At the retained Swift geometry and 17,408-row tile limit, all eligible projections
cost 22,656 submissions per ingestion token and another 600 for its vocabulary
head. One evaluation therefore fits the 24,000-write cap; ingestion plus evaluation
does not. Useful larger requests need explicit partial placement or a separately
qualified session contract. These are source-derived counts, not device timings.

## Resources and residency

`maximum_owned_payload_bytes` bounds provisional F32 output, one padded weight
tile, returned limb/plane results and activation payload. Model/prepared activation
storage, caller output, allocator/stack overhead, native mappings and VTCM are
additional charges. Insufficient host payload or unsupported geometry selects
CPU before submission. Native operation/deadline exhaustion fails the scope.
The [native session limits and ownership rules](../hexagon-native/README.md#projection-sessions)
and [owner-device policy](../../AGENTS.md#owner-device-execution-budgets) both apply.

For original-PQ2 HMX residency, first reserve a native arena with
`HmxQ2ProjectionLimits::with_resident_weights`, then enable executor
`with_resident_weights(maximum_cache_metadata_bytes)`. Metadata is reserved inside
the owned-payload cap for all supported projections. Strong `Arc` ownership retains
the original immutable bytes without a model copy. Cache identity uses shared
ownership, tensor kind and dimensions rather than raw addresses or per-token hashes.

All padded bytes and metadata for a projection must fit before preparation.
Capacity miss selects streaming for that entire projection; unexpected preparation
failure closes the executor. Entries are append-only, without eviction or renewed
limits. Statistics report preparations, cache hits, copied bytes and metadata
separately from kernel operations. Host caching proves neither DSP residency nor
reduced DMA/cache work.

Resident metadata uses a capacity-reserved sorted storage/geometry index and
append-only projection records. Native Q2 and HMX lookups check retained strong
`Arc` ownership and exact geometry, including different matrix views sharing one
immutable byte allocation; names do not determine cache identity. The index's
retained capacity is charged inside the existing metadata reservation, reducing
the number of entries that fit a given reservation. Native Q2 sibling preparation
reuses one extent for repeated identical tensors without changing operation
charges. HMX lookups use binary search instead of a linear metadata scan. These
are source work reductions; device timing, resources and complete-model parity
still require independent qualification.

`NativeQ2Executor` can pack a fresh resident extent directly through the submitter's
exact-length callback into its admitted mapped arena. Registry growth and handles
remain private until packing, scale checks and deadline checks pass. Error or
unwinding publishes nothing; existing extents stay untouched. Submitters without
this capability use scratch-and-copy preparation.

## Prefill packets

Native resident dispatch batches up to eight ordered operations per packet and
uses paired-token FFN chains when the submitter explicitly supports them.
Member/token order, arithmetic and lifetime operation charges remain unchanged.
Failure or nonfinite output clears the complete tile and closes admission.
Transient/custom submitters retain their declared scalar capabilities.

These are ordered `N=1` projections, not true multi-token matrix multiplication.
See the [native packet contract](../hexagon-native/README.md#resident-weights-and-fused-packets)
for supported chains, command/resource bounds and publication rules.

## Validation

Model-free controls compare fused/nonfused matvec bits, row tails, all i8 bitplanes,
PTQ1 captures and failure poisoning. Runtime `projection_scope` tests cover
attention/recurrent state, transforms, adapters, checkpoints, reset, CPU fallback
and terminal failure. Hosted Hexagon/Foundation CI owns complete tests and analyzers.
The [model probe](MODEL-PROBE.md) provides the bounded actual-model procedure.

Signed-device shapes, full logits/state/adapters, actual resource admission,
cancellation/device loss, persistent use and matched request benefit remain
unqualified. No automatic backend or application profile is enabled.

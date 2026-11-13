# Explicit CPU tiled prefill

The experimental graph API now consumes bounded packed multi-token projections
inside real state-only Qwen3.5 prompt ingestion. Ordinary graphs retain their
current policy. New local source also exposes an explicitly admitted FP16
private-service profile and an independently selected batched target verifier.
These consumers are not in Runner's published 0.1.25 bundle; they change no
runtime release, Runner pin or application default.

```rust,ignore
let graph = Qwen35::new(model, context)?;
let reservation = graph.tiled_prefill_scratch_bytes(8)?;
let graph = graph.with_tiled_prefill(8, reservation)?;
// Optional, separate numerical contract:
let graph = graph.with_chunkwise_gdn_prefill()?;
let runtime = Runtime::new(graph, limits)?;
// generate_with_prefill(... Prefill { block_tokens: 32 }, ...) calls real tiles.
```

Select only 8, 16 or 32 token tiles on an empty graph. Both APIs renew checkpoint
provenance and append a phase-specific identity. Ordinary token evaluation stays
unchanged. External projection-executor scopes retain their existing sequential
ingestion and do not dispatch these CPU tiles.

## Explicit private-service consumer

`swift_session::CpuPrefill::new(tokens, scratch_bytes)` binds 8, 16 or 32 tokens
to a caller-supplied additional reserve. Existing SwiftSession loaders preserve
ordinary ingestion; `load_profile_with_prefill` supplies the explicit selection.
The graph independently checks its geometry-derived minimum. Owners must add
the full supplied reserve to the existing context/process allowance using
`validate_prefill_capacity`, while retaining worker, allocator and host reserves.

The private server accepts paired trusted `--cpu-prefill-tokens` and
`--cpu-prefill-scratch-bytes` arguments only with explicit `--kv-cache f16` and
paired `--context-tokens` / `--memory-limit-bytes`. Invalid pairing, precision,
bounds or insufficient total allowance rejects before artifact I/O. Requests
cannot change this selection. Startup readiness reports `cpu_prefill_tokens`,
`cpu_prefill_scratch_bytes` and the exact `startup_graph_identity`.

This profile retains sequential GDN arithmetic and skips implicit Hexagon
discovery/installation. Explicit `--prompt-reuse latest` remains compatible;
reset and adapter scale changes retain the immutable tile policy. Rotated KV
and triangular GDN are separate profiles, not fallbacks. Full-model, long-prefix,
adapter, cancellation/recovery and matched whole-request benefit remain pending.

## Explicit target-verification consumer

After selecting exact FP16 tiles on an empty CPU graph, call
`Qwen35::tiled_verification_scratch_bytes` and charge that additional reservation
before `with_tiled_verification`. The separate `cpu-tiled-verify-v1` identity
admits at most eight candidate positions. It charges each full vocabulary row
and row metadata in addition to ordinary tile, state and checkpoint budgets.

`Decoder::evaluate_batch_into` then produces every position's complete target
logits while layer-major packed projections, including a packed output head,
share decoded weight blocks across candidates. Final-layer state-only narrowing
is disabled for verification. Causal attention, convolution and sequential GDN
remain complete. Triangular GDN, rotated KV and installed external projection
executors reject the exact CPU verifier.

`Runtime::verify_draft_block` uses this capability for eligible multi-token
drafts; ordinary graphs keep sequential verification. Candidate zero uses the
committed prefix's logits; later candidates use the preceding returned row.
Full acceptance keeps the computed state and evaluates one bonus token. A
rejection restores the complete checkpoint, replays the accepted prefix and
evaluates the correction. Work admission conservatively charges the proposed
block plus possible replay; failed restore closes admission without retry.

This is greedy block verification, not sampled serving. The suffix-generation
loop and SwiftSession speculative completion remain sequential; stochastic
acceptance/residual sampling and a useful drafter still require integration.
Focused synthetic controls do not establish actual-model parity or speed.

## Exact packed tiles

The graph runs layer-major over a tile, retains causal attention/cache updates,
and then computes FFN up, gate and down projections across all tile tokens.
For each packed output row, one 128-weight block is decoded once, followed by
all token dots. PQ2 retains the +2 fourth symbol, Q8_K activation preparation and
original per-128 scale/FMA order. Odd-block PQ2 retains Q8_0. PTQ1 retains its
quantizer, scale grouping and pinned four-lane FMA/reduction order. Floating and
other supported formats use the existing matvec contract. Each token is
transformed independently; adapters consume the original untransformed input.
No decoded matrix or persistent repacked weight copy is retained.

Full-attention decode and tiles borrow gate heads directly from the immutable
interleaved query/gate projection. Only queries are normalized into separate
scratch; the original gate values remain available until score/value work has
finished. This omits one gate materialization without changing sigmoid, rounded
probabilities, value accumulation, adapters or cancellation/rollback behavior.
For Swift's 24 heads of width 256 across 16 full-attention layers, the omitted
copy payload is 384 KiB per token (768 KiB of logical copy reads/writes), excluding
the unchanged later gate reads. Actual traffic, latency and peak RAM are
unmeasured. Recurrent/FFN gate scratch and conservative admission remain charged.

Without the separate chunkwise selection, full attention, recurrent attention,
convolution and FP32 GDN retain their previous per-token operations. The graph's
`ingest_block` path is the consumer; this is more than a standalone tiled helper.
The tile inherits the graph's requested matrix backend. On the existing eligible
Windows ARM64 MSVC build, Auto/NEON prepare sixteen baseline NEON vectors from
the one decoded block and reuse them across token dots. Signed products widen
before summation; the complete PQ2 dot is bounded by 32,768, and PTQ1 keeps its
four native integer lanes separate until the original floating FMAs. Explicit
scalar forcing and unsupported builds retain scalar reductions. The fixed
256-byte decoded payload is stack-scoped, with no new retained activation or
weight allocation. Optional DOTPROD/I8MM instructions are not selected.

Packed tiles inherit the graph's existing private row-worker pool, including
selection before or after enabling tiles. The pool is shared with ordinary
matvec work; projections create no additional workers. Each worker borrows
original weights and prepared activations, owns disjoint output row slices
and decodes each packed row block once across all tile tokens. One worker and
builds without `parallel-rows` retain sequential execution. Floating fallback
continues to use the ordinary sequential projection contract. This source path
still needs comparison against ordinary NEON/worker matvec before assuming that
shared decoding and explicit SIMD improve complete prompt time.

## Separate chunkwise GDN

The `gdn-wy-f32-v1` selection batches recurrent input projections, keeps causal
convolution and normalization, and solves the bounded lower-triangular delta
system for each recurrent value head. Every head uses `head % ssm_keys`.

With `P_t` the cumulative decay and `D[t,j]` the product of decays over `(j,t]`,
the implementation solves

`delta_t = beta_t * (v_t - P_t * S0*k_t - sum(j<t, D[t,j]*dot(k_j,k_t)*delta_j))`.

Outputs combine `P_t*S0*q_t` with the accepted causal deltas, and the final state
combines `P_last*S0` with all deltas. Initial recurrent rows are read row-major
and the final recurrent matrix is written once per chunk. Products are formed
without cumulative-product division, including zero and underflowed decay.
No inverse-state rollback, truncated series or reduced state precision is used.
Finite decays and sigmoid updates must lie in `[0,1]`; other gates reject.

This changes floating evaluation order. Its initial qualification tolerance is
`1e-3 + 1e-3 * abs(reference)` for complete logits and logical state. Full-model,
long-prefix, adapter and restored-continuation evidence remains required before
service selection. Different numerical identities cannot share prefix state.

## Direct GDN convolution input

The source candidate reads immutable original projection rows directly during
head-major pinned GDN convolution. Eligible multi-token width-four NEON groups
write only the convolved `SiLU` values into the existing output storage, rather
than copying raw projections there and reading them back first. The shared
kernel keeps the same FMA sequence, register-sliding history, finite-result
checks and failure publication. Unsupported shapes and targets retain complete
raw staging before their original scalar execution, including later token
contents on an error. Input rows remain immutable and the graph's callbacks,
request rollback and conservative scratch reservation are unchanged.

For Swift's 10,240 convolution channels, a 32-token tile and 48 recurrent layers,
the omitted copy accounts for 60 MiB of nominal raw staging writes and 60 MiB of
matching reads. These are source payload counts, not measured memory traffic or
latency. Independent scalar controls cover FMA/`SiLU`/history bits, signed zero
and subnormal inputs, history sliding, unsupported widths and tails, geometry
rejection and complete copied-route failure publication. Release code generation,
complete-model logits/state/adapter recovery and whole-prompt benefit remain
pending before release integration.

## Grouped GDN finite checks

The staged baseline NEON four/eight-row GDN helpers combine all updated-lane
finite masks before one horizontal predicate reduction at their existing
publication boundary. Every updated vector is still checked. Key-dot and
query-dot checks retain their order, and arithmetic, private staging capacities
and output bits are unchanged. Any failure leaves all grouped destinations
untouched, allowing the checked facade to replay the original row-ordered
scalar failure path. Single-row helpers retain their earlier blockwise checks
and exact partial-error publication.

At Swift's width 128, an eight-row decode group has one updated-lane horizontal
predicate reduction instead of 64. Across 48 recurrent layers, 48 value heads
and 16 groups per head, this is 36,864 reductions instead of 2,359,296 per token.
A 32-token exact tile combines 2,048 such predicates per group into one. These
are source work counts; every vector finite predicate and separate dot-result
check remains. They establish no instruction count or measured speedup. Release
ARM64 code generation and spill review, complete-model state/logit/adapter and
cancellation parity, and matched whole-request benefit remain required.

Direct NEON controls compare successful state/output bits to independent scalar
recurrence and reject early/late key, update and output failures without any
publication across both staging capacities. A separate checked-facade control
compares scalar replay's complete partial state and output bits after a late
update overflow.

## Bounds and recovery

Pinned Swift's six query heads per KV head now share key/value conversion as a
complete group during tiled prefill. Each token reads only its causal prefix
from the already-appended tile cache; the current token keeps its original F32
key/value arithmetic. Other geometries and source arithmetic retain their prior
path. The six-head path removes the old two single-head tails, while the old
four-head path also shared scans across tokens: total iterator sweeps are not
uniformly lower and no measured prompt benefit is established.

Six-head scores reserve two extra full-context rows per tile token:
`2 * tile_tokens * context_limit * sizeof(f32)`, at most 2 MiB for 32 tokens at
context 8192. The graph charges this even under serial execution. Complete-model
state/logit/adapter and recovery parity, ARM64 code generation/resources and
matched whole-prompt benefit remain required before release qualification.

The additional reservation conservatively charges all tile vectors, both packed
activation contracts, floating fallback high-water storage and small triangular
matrices at once. The graph reports this reservation separately. It includes a
fixed 64 KiB allowance for the maximum triangular
tables and temporary metadata, including eight-token or very narrow fixtures.
A further fixed 16 KiB covers the maximum sixteen row-task descriptors, their
32 token-output slices, errors, counters and cancellation synchronization.
The descriptors are stack-scoped; no row-major result matrix, transpose copy or
retained packed-weight expansion is introduced. This charge is present even
under one worker so a later row-policy change cannot exceed the reservation.
Owners must add it to ordinary live state, checkpoint, scratch and worker-stack charges; allocator
overhead remains separate. No existing admission limit is reduced.

Cancellation/deadline checks surround token preprocessing, causal attention,
recurrent heads, layers and projections. Serial projections check every packed
output row. During a parallel projection the original callback stays on its
calling thread, with nominal one-millisecond waits between checks; it needs
neither `Send` nor `Sync`. Workers observe cancellation before each next row.
This is a cooperative bound rather than a real-time scheduling guarantee.
All scoped tasks join before a callback cancellation or the earliest failing
row chunk is returned and all provisional outputs are cleared. Completion
guards also release the caller's wait if a worker unwinds. The implementation
does not initialize or use Rayon's global pool.
As allowed by the decoder block contract, failure can leave provisional
layer-major state. The runtime restores its entire request checkpoint before
reuse. Position is marked nonempty before the first mutation so explicit reset
clears a failed first tile as well. Successful tiles retain complete logical
state. Diagnostic counters include work later rolled back.

## Owning checks and pending acceptance

Pinned full-attention four-head value tiles fuse the original sigmoid gate into
their final NEON publication store, in both single-token decode and two-token
tiled prefill. The reduced vectors reuse the now-dead first temporal accumulator
lane; no additional output buffer or stack payload is reserved. Successful
grouped tiles omit the separate attended-output reread and rewrite. Scalar,
non-ARM and single-head tails retain their existing arithmetic paths.
Every reduction must pass the complete finite mask before any gate is applied.
On eligible NEON tiles, failed reductions publish the original ungated partial
output in increasing token/head/channel order. Scalar fallback retains atomic
staging and leaves all destinations untouched on reduction failure. Graph
rollback and each dispatch path's original error behavior are preserved.
The exponentials and separate binary32 add, divide and multiply operations match
the original gate. Independent per-channel FMA controls cover channel tails,
temporal padding, causal token boundaries and failed partial publication.
These source-level store reductions establish no measured request-time benefit;
actual-model state/logit/adapter, cancellation and matched performance acceptance
remain required before release integration.

Foundation already discovers the library unit tests and the `graph` integration
target on its hosted platforms. Focused commands for this source are:

```text
cargo test --release --locked --features parallel-rows --lib tiled_
cargo test --release --locked --features parallel-rows --lib chunkwise_gdn_
cargo test --release --locked --features parallel-rows --test graph tiled_prefill_
cargo test --release --locked --features parallel-rows --test graph chunkwise_gdn_prefill_
```

Independent existing matvec rows cover both quantizers and pinned FMA ordering;
tile controls compare all requested backends and repeated pinned/unpinned reuse
bit for bit against scalar matvec, plus extreme integer/native-lane reductions
and cancellation or an invalid final scale after a complete output row.
Parallel controls cover awkward row counts, two/four-worker exact PQ2/Q8_K,
odd PQ2/Q8_0 and PTQ1 tiles, invalid-scale clearing/retry, callback thread affinity,
joined cancellation, deterministic failure and worker unwind. A fixed-size
assertion checks task metadata against the explicit reservation. Graph controls
set workers before/after tile selection and preserve adapter/state bits through
continuation, policy change, checkpoint restoration and reset.
The chunk unit control uses ordinary recurrence, signed key correlations and zero
decay. Graph controls exercise packed FFN, nonzero/zero adapters, hybrid KV page
boundaries, continuation, checkpoint restore, cancellation/reset and runtime
request rollback. Prepared tests are not a compiled verdict. Owner-managed
hosted checks, complete-model numerical and resource acceptance and matched
target-device prompt-time measurements remain pending. No TPS gain is claimed.

# Explicit CPU tiled prefill

The experimental graph API now consumes bounded packed multi-token projections
inside real state-only Qwen3.5 prompt ingestion. Ordinary graphs and private
services keep their current sequential policy. No runtime release, Runner pin,
memory admission or application default changes here.

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

## Exact packed tiles

The graph runs layer-major over a tile, retains causal attention/cache updates,
and then computes FFN up, gate and down projections across all tile tokens.
For each packed output row, one 128-weight block is decoded once, followed by
all token dots. PQ2 retains the +2 fourth symbol, Q8_K activation preparation and
original per-128 scale/FMA order. Odd-block PQ2 retains Q8_0. PTQ1 retains its
quantizer, scale grouping and pinned four-lane FMA/reduction order. Floating and
other supported formats use the existing scalar matvec contract. Each token is
transformed independently; adapters consume the original untransformed input.
No decoded matrix or persistent repacked weight copy is retained.

Without the separate chunkwise selection, full attention, recurrent attention,
convolution and FP32 GDN retain their previous per-token operations. The graph's
`ingest_block` path is the consumer; this is more than a standalone tiled helper.
The tile kernel currently uses scalar integer reductions. It must be compared
against the existing NEON/worker path before assuming that reduced decoding work
improves complete prompt time.

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

## Bounds and recovery

The additional reservation conservatively charges all tile vectors, both packed
activation contracts, floating fallback high-water storage and small triangular
matrices at once. The graph reports this reservation separately. It includes a
fixed 64 KiB allowance for the maximum triangular
tables and temporary metadata, including eight-token or very narrow fixtures.
Owners must add it to ordinary live state, checkpoint, scratch and worker-stack charges; allocator
overhead remains separate. No existing admission limit is reduced.

Cancellation/deadline checks surround token preprocessing, causal attention,
recurrent heads, layers and projections, and occur for every packed output row.
As allowed by the decoder block contract, failure can leave provisional
layer-major state. The runtime restores its entire request checkpoint before
reuse. Position is marked nonempty before the first mutation so explicit reset
clears a failed first tile as well. Successful tiles retain complete logical
state. Diagnostic counters include work later rolled back.

## Owning checks and pending acceptance

Foundation already discovers the library unit tests and the `graph` integration
target on its hosted platforms. Focused commands for this source are:

```text
cargo test --release --locked --features parallel-rows --lib tiled_
cargo test --release --locked --features parallel-rows --lib chunkwise_gdn_
cargo test --release --locked --features parallel-rows --test graph tiled_prefill_
cargo test --release --locked --features parallel-rows --test graph chunkwise_gdn_prefill_
```

Independent existing matvec rows cover both quantizers and pinned FMA ordering;
the chunk unit control uses ordinary recurrence, signed key correlations and zero
decay. Graph controls exercise packed FFN, nonzero/zero adapters, hybrid KV page
boundaries, continuation, checkpoint restore, cancellation/reset and runtime
request rollback. Prepared tests are not a compiled verdict. Owner-managed
hosted checks, complete-model numerical and resource acceptance and matched
target-device prompt-time measurements remain pending. No TPS gain is claimed.

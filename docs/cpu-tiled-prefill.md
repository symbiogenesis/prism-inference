# CPU tiled prefill

The experimental Qwen35 graph supports explicit 8-, 16- or 32-token CPU tiles for
state-only prompt ingestion. Select on an empty graph:

```rust,ignore
let graph = Qwen35::new(model, context)?;
let reservation = graph.tiled_prefill_scratch_bytes(8)?;
let graph = graph.with_tiled_prefill(8, reservation)?;
// Optional, separate numerical contract:
let graph = graph.with_chunkwise_gdn_prefill()?;
let runtime = Runtime::new(graph, limits)?;
```

Selection renews checkpoint provenance and adds a phase identity. Runtime
`generate_with_prefill` calls the graph through `Decoder::ingest_block`.
Ordinary token evaluation and external projection-executor ingestion retain
their separate paths. Actual-model parity and prompt-time benefit require
qualification.

## Explicit private-service consumer

`CpuPrefill::new(tokens, scratch_bytes)` binds tile size to an additional reserve;
`SwiftSession::load_profile_with_prefill` selects it. The graph checks the
geometry-derived minimum. Charge the complete reserve with
`validate_prefill_capacity_with_kv` before artifact I/O, in addition to state,
checkpoint, worker, allocator and host allowances.

Trusted server startup requires paired `--cpu-prefill-tokens` and
`--cpu-prefill-scratch-bytes`, explicit `--kv-cache`, and paired
`--context-tokens` / `--memory-limit-bytes`. Invalid combinations reject.
Readiness binds `cpu_prefill_tokens`, `cpu_prefill_scratch_bytes` and
`startup_graph_identity`; requests cannot alter them.

FP16/K4/V4/K4/V3 retain their selected causal attention arithmetic, without
expanded packed history. This profile uses sequential GDN, skips implicit Hexagon
discovery/installation and supports explicit latest-prefix reuse. Reset/scale
changes retain tile policy. Triangular GDN remains a separate FP16 selection.

## Explicit target-verification consumer

After tile selection, charge `tiled_verification_scratch_bytes` before calling
`with_tiled_verification`. Identity `cpu-tiled-verify-v1` admits at most eight
positions and charges complete vocabulary rows/metadata beyond tile/state/
checkpoint storage.

`Decoder::evaluate_batch_into` returns every position's target logits while
sharing decoded packed blocks across candidates. Causal attention, convolution
and sequential GDN remain complete; final-layer state-only narrowing is disabled.
Triangular GDN and external executors reject verification.

`Runtime::verify_draft_block` uses prefix logits for candidate zero and the
preceding returned row for later candidates. Full acceptance keeps state and
evaluates a bonus token. Rejection restores the round checkpoint, replays accepted
tokens and evaluates the correction. Admission charges verification plus possible
replay; failed restore closes admission.

## Explicit sampled suffix consumer

`Runtime::generate_sampled_speculative_until` verifies deterministic history-suffix
proposals with the caller's sampler. Each position draws once from the target:
a matching draw accepts; a different draw supplies the conditional residual.
This preserves filter/RNG progression under identical positional logits. It
does not implement a probabilistic learned drafter.

Eligible proposals use tiled verification. Stop or rejection restores the round
checkpoint and replays accepted tokens before a non-stop correction. Stops never
enter state. Callback failure, cancellation or expiry restores the entire request
and sampler; failed restore closes admission. Rejected/replayed work remains
charged.

`CpuSpeculation::new(tokens, verification_scratch_bytes, checkpoint_bytes)`
accepts 1..8 positions and two extra reserves. Verification covers all output rows;
the extra checkpoint covers a round prefix beside request rollback, at least the
selected state budget with a 1 GiB floor. Full-context K4/V4 at 262,144 tokens
requires 4,644,798,464 bytes per prefix reserve. `validate_execution_capacity`
charges both before artifact I/O; `load_profile_with_execution` selects them.

Server startup requires all three `--cpu-speculative-tokens`,
`--cpu-verification-scratch-bytes` and `--cpu-speculative-checkpoint-bytes`,
admitted tiles and paired context/process limits. Completion uses Swift's
temperature-1/top-20/top-p-0.95 sampler. Readiness binds the full selection.
Phase-timing requests reject speculation before mutation because those counters
cannot represent replay. Greedy and standalone stochastic-verifier APIs remain
separate.

## Exact packed tiles

Layer-major execution retains causal state updates and batches FFN projections.
Each 128-weight block is decoded once per output row for all token dots.
PQ2 preserves its +2 fourth symbol, Q8_K preparation and per-128 scale/FMA order;
odd-group PQ2 uses Q8_0. PTQ1 preserves quantization and pinned four-lane
FMA/reduction order. Adapters consume original inputs. No expanded matrix persists.

Eligible Windows ARM64 baseline NEON reuses sixteen decoded vectors across dots;
signed products widen before summation. A stack-scoped 256-byte decoded block
adds no retained storage. Scalar forcing/unsupported targets use scalar reductions;
DOTPROD/I8MM are not selected. Packed tasks share the existing
[row-worker pool](parallel-rows.md), borrowing activations and writing disjoint
rows. Floating fallback remains sequential.

Full attention borrows immutable interleaved gate heads and normalizes queries
into separate scratch. [Attention locality](inference-efficiency.md#attention-locality)
defines grouped/paired cache traversal. Ordinary recurrent arithmetic remains
sequential unless chunkwise GDN is explicitly selected.

## Separate chunkwise GDN

`gdn-wy-f32-v1` batches recurrent projections, preserves causal convolution/
normalization, and solves a lower-triangular delta system per value head
(`head % ssm_keys`). With cumulative decay `P_t` and decay product `D[t,j]`
over `(j,t]`:

`delta_t = beta_t * (v_t - P_t * S0*k_t - sum(j<t, D[t,j]*dot(k_j,k_t)*delta_j))`

Outputs combine `P_t*S0*q_t` with causal deltas; final state combines
`P_last*S0` with all deltas. Products avoid cumulative-product division, including
zero/underflowed decay. Decays and sigmoid updates must lie in `[0,1]`.
No inverse-state rollback or reduced precision is used.

This changes floating order. Initial complete-logit/state tolerance is
`1e-3 + 1e-3 * abs(reference)`; long-prefix, adapter and restored-continuation
qualification remain required. Different numerical identities cannot share state.

## Bounds and recovery

Scratch conservatively reserves all tile vectors, both activation contracts,
floating fallback, every four/six-head score row per token, 64 KiB for triangular
tables/metadata and 16 KiB for row-task metadata. The latter covers at most
16 tasks and 32 output slices, including serial execution. Add it to ordinary
live/checkpoint/scratch/worker charges; allocator overhead is separate.

Cancellation checks surround preprocessing, causal attention, recurrent heads,
layers and projections. Serial packed work checks each output row. Parallel
callbacks stay on the caller thread with nominal one-millisecond waits; workers
check before each row and all tasks join before error/cancellation return.
Worker-unwind guards release waiting callers. This is cooperative scheduling.

Failures can leave provisional layer-major state; Runtime restores the request
checkpoint. Position becomes nonempty before mutation so reset clears a failed
first tile. Outputs clear on failure. Counters include rolled-back work.

## Validation

Focused commands, subject to [local-check policy](build-and-validation.md):

```text
cargo test --release --locked --features parallel-rows --lib tiled_
cargo test --release --locked --features parallel-rows --lib chunkwise_gdn_
cargo test --release --locked --features parallel-rows --test graph tiled_prefill_
cargo test --release --locked --features parallel-rows --test graph chunkwise_gdn_prefill_
```

Controls cover scalar/NEON and quantizer bits, packed extremes, awkward rows,
worker selection, deterministic errors, callback affinity, joined cancellation,
unwind, causal page boundaries, adapters, continuation, checkpoint/reset and
request rollback. Hosted checks, full-model correctness/resources and matched
target-device prompt measurements remain distinct gates.

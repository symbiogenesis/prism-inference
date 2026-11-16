# Roadmap

[Documentation index](index.md) · [Runtime contracts](runtime-contracts.md)

The implemented generation path targets the pinned Swift/Bonsai dense Qwen35
model. Broader Prism architectures and general model selection remain planned.
The [runtime contracts](runtime-contracts.md) describe current behavior; this page
tracks remaining work.

## Qualification limits

Model-free checks cover parsing, codecs, graph state, rollback, tokenization and
service protocols. Bounded numerical controls cover selected Swift workloads;
they do not establish arbitrary-input, long-context or device qualification.
Results apply to their exact source, artifact, profile and workload.
See [validation](build-and-validation.md) and [conformance](prism-upstream-tests.md)
for the required checks. No CPU optimization or accelerator path is qualified
for performance by its availability alone.

## Remaining work

| Area | Next work |
| --- | --- |
| Models | Add broader Prism architectures and general model selection |
| Numerics | Compare full-model logits and complete logical state across more prompts, contexts and generated trajectories |
| Tokenization and chat | Broaden template-byte, tokenizer-ID and turn-boundary coverage |
| Sampling | Broaden production sampler and RNG comparisons |
| Adapters | Qualify native residual writers, scale-zero/nonzero behavior, grouped SSM basis, scale changes and service lifecycle under the [adapter contract](rank-one-adapter.md) |
| CPU execution | Qualify zero-copy `MappedSlice` GGUF loading + parallel payload validation, spin-then-park `NativeRowPool`, bit-exact `G in {2, 3, 6}` grouped GQA packed-KV decode/tile attention, packed prefill tiles, attention scoring, bounded score-scratch growth, parallel transforms and [work/storage reductions](inference-efficiency.md) with full-model parity and matched whole-request measurements |
| Accelerators | Qualify the explicit Hexagon FFN/SwiGLU chains (including `adapter_affects_any` per-layer LoRA gating), rank-1 `ffn_down` side-branches, and investigate Adreno after defining execution and ownership contracts |
| Service ownership | Complete process/OS, cancellation, recovery, daily-service and device lifecycle acceptance |
| Devices | Establish correctness, quality, throughput, memory and security on each intended target |

Model weights and adapters follow their publishers' licenses. Code attribution
and upstream source identities are recorded in [NOTICE](../NOTICE).

Completed structural and efficiency implementations in source (awaiting a
reviewed Foundation bundle, Runner repin, full-model parity and X1/X2 device
qualification) cover Windows zero-copy mapped GGUF loading (`MappedReadOnlyFile`
/ `MappedSlice` with `PrefetchVirtualMemory`, chunked CNG SHA-256 and parallel
payload validation), `--broker-pipe` `KIND_VERIFIED_SECTION` 52-byte LE handle
intake, verified `"SB2HQ2C1"` `NativeQ2ArenaCache` sealing and
`Qwen35::release_executor_resident_host_weights` host weight release, zero-alloc
spin-then-park `NativeRowPool`, bit-exact `G in {2, 3, 6}` grouped GQA packed-KV
decode and tiled-prefill attention with `[token][head]` consumer-layout packed
probabilities and KV-group-aligned balanced tail partitions, `adapter_affects_any`
and full Hexagon `ffn_down` LoRA side-branch fusion (`supports_ffn_chain_lora` /
`submit_resident_ffn_chain_lora`), opt-in `--cpu-prefill-gdn wy-f32`
(`:gdn-wy-f32-v1`), mixed-phase CPU tiled prefill with Hexagon decode handoff,
bounded `1..=8` `BranchCheckpoints`, `GrammarConstrainedSampler` speculation,
atomic checkpoint/restore identity checks, checkpoint buffer recycling,
transient vs `PromptEndAndLatest` checkpoints, `HybridCheckpointBoundary`,
atomic conv shifts, mid-tile `state_consistent` and GDN decay/norm guards,
bounded speculative verification telemetry (`SpeculativeStats`),
`SwiftExecutionMemoryLedger`, worker EOF cancellation and server settle bounds.

- [ ] Publish and qualify uncapped ordinary native HTTP generation: source-bound
  bundle and Runner repin, explicit diagnostic deadlines, long-prefill Stop/retry
  and confirmed owner cleanup. Model-free virtual-time checks do not qualify
  actual-model long-prompt behavior or throughput.

- [ ] Qualify the local immutable-source guard in Foundation: native and fallback
  strict analysis, outgoing branch/tag and replacement-object isolation,
  committed-policy controls, receipt invalidation, process/output bounds and
  cleanup. Focused local checks supply no hosted acceptance verdict.

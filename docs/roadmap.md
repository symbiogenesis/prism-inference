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
| CPU execution | Qualify packed prefill tiles, attention scoring, parallel transforms and [work/storage reductions](inference-efficiency.md) with full-model parity and matched whole-request measurements |
| Accelerators | Qualify the explicit Hexagon FFN chain and investigate Adreno after defining execution and ownership contracts |
| Service ownership | Complete process/OS, cancellation, recovery, daily-service and device lifecycle acceptance |
| Devices | Establish correctness, quality, throughput, memory and security on each intended target |

Model weights and adapters follow their publishers' licenses. Code attribution
and upstream source identities are recorded in [NOTICE](../NOTICE).

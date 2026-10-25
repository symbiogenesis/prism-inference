# Status and roadmap

[Documentation index](index.md) · [Runtime contracts](runtime-contracts.md) · [Build and validation](build-and-validation.md)

## Current scope

Prism Inference is an independent Rust CPU runtime for Prism-compatible GGUF
models. The first implemented generation path is the dense Qwen35 architecture
used by the pinned Swift/Bonsai model. Broader Prism architectures and general
model selection are in scope and deferred.

The workspace includes the token driver, persistent tokenizer, Swift JSONL
worker, private HTTP server, GGUF tooling, ternary codecs and native storage
overlay. SwiftBonsai consumes a prebuilt server inside its guarded Windows Job;
ordinary SwiftBonsai builds do not compile this runtime.

## What the evidence establishes

The retained model-loading evidence covers the pinned 851-tensor artifact and
initialization of its 8K graph within the configured CPU budget. Scalar and NEON
state/logits match on the retained three-step trajectory. Differences from the
native reference exceed the declared numerical tolerance.

The pinned Swift chat path has native template-byte/tokenizer-ID captures and
seeded full-vocabulary MT19937 sampler comparisons. The exact verified Swift
identity selects the pinned Windows ARM64 binary's weighted RMS, PQ2 fused
accumulation and SiLU/SwiGLU arithmetic. Independent native captures cover those
operators; other model identities and public scalar APIs preserve their existing
source-order contracts. Decoder identity records the pinned arithmetic choice.
These operator repairs do not close the failed full-model numerical comparison.

The private server also cancels model work when an in-flight request future is
dropped. Focused recovery checks cover rollback, reset-required admission and a
successful request after explicit reset. Complete device lifecycle evidence
remains pending.

Synthetic checks cover bounded parsing, exact codec arithmetic, graph state,
checkpoint isolation, rollback, tokenization and service contracts. The
[Foundation workflow](../.github/workflows/foundation.yml) checks both Windows
targets and the Linux native storage overlay. Passing these checks does not
establish full-model parity, throughput or complete device qualification.

## Remaining work

| Area | Next work |
| --- | --- |
| Model support | Add broader Prism architectures and general model selection |
| Numerical qualification | Compare full-model logits, recurrent/attention states and generated tokens with the pinned native reference |
| Tokenization and chat | Broaden inputs/templates beyond the pinned Swift prompt-byte and tokenizer-ID captures |
| Sampling | Broaden qualification beyond the pinned Swift MT19937 sampler/RNG captures |
| Adapters | Implement adapter forward execution; immutable B-vector sharing alone does not apply an adapter |
| CPU execution | Profile allocation, prefill, checkpoint and packed-kernel behavior; qualify further SIMD paths |
| Other accelerators | Investigate Adreno and additional backends after their execution and ownership contracts are established |
| Application ownership | Complete broader process/OS, daily-service and device lifecycle qualification |
| Device acceptance | Retain correctness, tokens/second, memory, quality and security evidence on the intended hardware |

## Research and provenance

The original SwiftBonsai records remain useful historical context for the
extracted runtime:

- [MVP and upstream checklist](https://github.com/symbiogenesis/SwiftBonsai2/blob/370084c530e2271f10a1c55a5a10bba7c806aacb/docs/prism-upstream-todo.md)
- [Native engine R0–R7 plan](https://github.com/symbiogenesis/SwiftBonsai2/blob/370084c530e2271f10a1c55a5a10bba7c806aacb/docs/native-rust-engine.md)
- [CPU SIMD S0–S6 plan](https://github.com/symbiogenesis/SwiftBonsai2/blob/370084c530e2271f10a1c55a5a10bba7c806aacb/docs/simd-plan.md)
- [Shared upstream C++/Rust conformance baseline](https://github.com/symbiogenesis/SwiftBonsai2/blob/370084c530e2271f10a1c55a5a10bba7c806aacb/docs/prism-upstream-tests.md)
- [Partial guarded-device receipt](https://github.com/symbiogenesis/SwiftBonsai2/blob/370084c530e2271f10a1c55a5a10bba7c806aacb/config/swift-native-session.evidence.json)

The graph/quantizer contract references Prism
`73bfe6cc485554822f80c473a3c0c4f25e80f41b`. The
[native-overlay source lock](../tools/prism/source.lock.json) is a separate pin.
The [license notice](../NOTICE) records Prism/ggml and Camelid attribution;
model weights remain separate downloads with their publishers' terms.

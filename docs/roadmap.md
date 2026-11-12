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

The [0.1.24 registry delivery](release-0.1.24.md) records the newer published
runtime package and its verified checksum. Its binary bundle and Runner repin
remain pending owner source publication and source-bound hosted production.

The [0.1.23 delivery scope](release-0.1.23.md) records the earlier CPU crate and
Windows ARM64 development prerelease at `055c743`. Its bounded hosted operation
passed source/receipt controls, analyzers, portable compilation and production
linking. Complete Foundation, runtime conformance, platform/model/state/adapter
acceptance and sustained performance qualification remain pending. Runner's new
short-request observations do not close those gates. Registry delivery, binary
verification and application-default qualification remain distinct. Original PQ2, one row
worker and fresh per-request state remain the service defaults. PTQ1, approved
rank-one adapters, parallel row workers and latest-prefix reuse require explicit
selection; availability does not establish their model or device qualification.

The [rotated KV source candidate](quantized-kv.md) adds Rust 4-bit and
asymmetric 4/3-bit profiles, common X1/X2 NEON kernels and exact packed-state
rollback. K4/V4 (`rsq4`) is the owner-selected source development default for
generation, Swift sessions and workers, with explicit FP16 comparison/recovery.
Full-model quality, long-context benefit and separate X1/X2 device qualification
remain open. The default does not claim those passes; active bundle/profile
integration remains pending owner publication. K4/V3 remains opt-in.

## What the evidence establishes

The retained model-loading evidence covers the pinned 851-tensor artifact and
initialization of its 8K graph within the configured CPU budget. At source
revision `9899a25`, NEON consumes a 61-token Swift chat prefix and three generated
greedy tokens. All 166,078,464 checked logit and complete logical state elements
match the pinned native reference byte-for-byte, as do the greedy token IDs.
The retained scalar three-token run also matches native byte-for-byte. Earlier
full-prefix scalar/NEON equality is retained at its separate tested revision.
The [historical follow-up receipt](https://github.com/symbiogenesis/SwiftBonsai2Runner/blob/54d46ad660107c572769dac49c00f411f5c7a6c4/config/swift-native-mvp-followup.evidence.json)
records exact scopes, revisions, hashes and resource observations.

The pinned Swift chat path has native template-byte/tokenizer-ID captures and
seeded full-vocabulary MT19937 sampler comparisons. The exact verified Swift
identity selects the pinned Windows ARM64 binary's weighted RMS, fused Q8_K
preparation and PQ2 accumulation, GDN updates, convolution FMA, SiLU/SwiGLU and
nonflash attention arithmetic, including native in-place RoPE ordering.
Independent native captures cover those
operators; other model identities and public scalar APIs preserve their existing
source-order contracts. Decoder identity records the pinned arithmetic choice.
The operator controls and complete model comparison qualify these bounded Swift
CPU workloads. Other inputs, longer trajectories and full-context behavior
still require their own evidence. Runner's retained `0.1.11` binary/source
[release integration record](https://github.com/symbiogenesis/SwiftBonsai2Runner/blob/main/docs/evidence/prism-0111-release-integration.json)
describes that historical delivery separately.
That record keeps actual-model qualification pending. These historical numerical
results do not transfer automatically to a newer source or binary, and the
`0.1.13` release does not itself change Runner's pin or defaults.

The private server also cancels model work when an in-flight request future is
dropped. Focused recovery checks cover rollback, reset-required admission and a
successful request after explicit reset. The
[historical bounded owned HTTP device receipt](https://github.com/symbiogenesis/SwiftBonsai2Runner/blob/54d46ad660107c572769dac49c00f411f5c7a6c4/config/swift-owned-http-mvp.evidence.json)
also covers supplied-history streaming, seeded reset/replay, context rejection
and recovery, active abandonment with Job cleanup, and a successful request
after a fresh owned restart. Exact diagnostic binaries use explicit 13.65 GiB
startup patches in both owner and server; the 12 GiB Job cap and 4 GiB running
reserve remain enforced. These results do not change the production 16 GiB
policy or establish released-runtime packaging, full-context service behavior
or daily security. The later reviewed prebuilt integration above is a separate
delivery result and does not replace these device gates.

Synthetic checks cover bounded parsing, exact codec arithmetic, graph state,
checkpoint isolation, rollback, tokenization and service contracts. The
[Foundation workflow](../.github/workflows/foundation.yml) checks both Windows
targets and the Linux native storage overlay. Passing these checks does not
establish full-model parity, throughput or complete device qualification.

## Remaining work

| Area | Next work |
| --- | --- |
| Model support | Add broader Prism architectures and general model selection |
| Numerical qualification | Broaden paired full-model inputs, context lengths and generated trajectories beyond the retained pinned Swift workloads |
| Tokenization and chat | Broaden inputs/templates beyond the pinned Swift prompt-byte and tokenizer-ID captures |
| Sampling | Broaden qualification beyond the pinned Swift MT19937 sampler/RNG captures |
| Adapters | Qualify the explicit [rank-one execution and private-service path](rank-one-adapter.md) against native residual writers, scale-zero/nonzero logits/state, intended grouped SSM basis, scale changes and actual-target lifecycle; JSONL workers remain adapter-free |
| CPU execution | Profile allocation, prefill, checkpoint and packed-kernel behavior; qualify the `I8MM`/`SDOT` `4×4`/`8×8` `PQ2_0` prefill tiles, `4×4` `F16` attention scoring, parallel `RMSNorm + Sign*FWHT`, fused prefill `SwiGLU + Sign*FWHT`, and end-to-end intra-layer FFN VTCM chain (`project_f32_ffn_chain`) |
| Other accelerators | Investigate Adreno and additional backends after their execution and ownership contracts are established |
| Application ownership | Complete broader process/OS, daily-service and device lifecycle qualification |
| Device acceptance | Retain correctness, tokens/second, memory, quality and security evidence on the intended hardware |

Local CPU source also reuses detected SDOT/I8MM eligibility across prepared PQ2
prefill tiles, keeps bounded elementwise worker error slots on the stack, and
combines adapter-enabled FFN SwiGLU with down-projection basis preparation.
Original-basis activations remain available for the down adapter. These source
work reductions require final-source hosted checks, feature-disabled fallback,
strict Release analysis and production linking, ARM64 codegen/stack review,
complete-model state/logit/adapter/cancellation parity and independent matched
X1 request measurements. X2 development checks establish no X1 benefit. The new
opaque capability APIs require a new `bonsai-neon-dot` helper publication and
exact runtime dependency update before registry delivery; Runner continues to
consume its verified 0.1.23 binary bundle until a qualified release and repin.

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

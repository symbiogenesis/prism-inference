# Immutable LoRA B storage sharing

The [Rust Prism overlay](https://github.com/symbiogenesis/prism-inference/blob/main/tools/prism/README.md) reduces CPU allocation for the
approved adapter by sharing identical immutable B payloads. Each target retains
its tensor metadata/name over a same-context view of one allocation; aliases
skip loading. Model weights, adapter bytes, scales and graph arithmetic remain
unchanged. The released assessment runtime does not include this overlay.

## Sharing contract

Only ordinary CPU-buffer, contiguous F32 rank-one B tensors qualify. Context,
buffer type/device, shape, strides and full payload bytes must match. Signed
zero and NaN payload bits remain distinct. Candidate size is capped at 64 KiB;
the comparison pool at 64 entries and 1 MiB. Ineligible, unequal or uncached
payloads retain independent allocations. Independent adapter lifetimes/devices
never share storage.

The approved adapter has 129 identical 20,480-byte B vectors. The focused native
regression verifies **2,621,440 fewer backend allocation bytes (2.5 MiB)**,
distinct names/layouts, byte-identical matrix-operation outputs and fallback
cases. It checks allocation/arithmetic correctness, not full-model quality or
performance. This saving does not explain the roughly 560 MiB diagnostic
[Job-counter difference](https://github.com/symbiogenesis/SwiftBonsai2Runner/blob/main/docs/swift-lora.md#memory-and-execution-cost).

## Source and build boundary

The Rust crate owns metadata extraction, one-handle bounded reads, validation,
sharing policy, native views and loading. The policy forbids unsafe code;
documented C ABI pointer handling and native-test calls are confined to boundary
modules. There are no runtime dependencies. The external C/C++ engine retains
call sites and exception translation; optional cbindgen generates declarations
outside the repository. Full inference replacement is separate
[native Rust engine work](https://github.com/symbiogenesis/SwiftBonsai2Runner/blob/main/docs/native-rust-engine.md).

[source.lock.json](https://github.com/symbiogenesis/prism-inference/blob/main/tools/prism/source.lock.json) pins Prism commit
`b87ca7225456b06b0975b8239c66b5aecf07e37c`, archive size **39,429,253 bytes**,
SHA-256 `333ab1cc57bdb1be0b52f894fe8c6822eb7da667a239d7a587dce79283782e58`,
and before/after loader/CMake hashes. This source is seven commits beyond the
assessment package at `73bfe6cc`, including UTF-8, Q2-row, JSON-schema, grammar
and null-callback fixes. Those source fixes are absent from the pinned binary.
Prism also diverges from ggml upstream; the pinned source does not contain all
upstream fixes. See the [revision comparison](https://github.com/ggml-org/llama.cpp/compare/b87ca7225456b06b0975b8239c66b5aecf07e37c...1a24df163920a4744a95373d178d790b72a9bb2f).

The [applicator](https://github.com/symbiogenesis/prism-inference/blob/main/tools/prism/Apply-Overlay.ps1) rejects source drift,
ambiguous insertion points and repeat application. The
[builder](../scripts/Build-PrismEngine.ps1) verifies the archive, creates a fresh
external lane, applies the overlay and records source/compiler/configuration,
Rust static library and binary identities. It never installs a runtime,
loads a model or promotes a profile. `-PrepareOnly` still builds Rust and its
header but needs no C++ compiler. Full ARM CPU builds require Clang. See the
[crate's build commands](https://github.com/symbiogenesis/prism-inference/blob/main/tools/prism/README.md#build-and-checks).

Foundation CI builds the modified completion/server linked to Rust and the
focused native ggml regression with Clang on Ubuntu ARM64, then runs correctness
and strict Rust checks. This validates loader linkage and storage behavior;
it does not qualify a Windows binary or Snapdragon throughput. Keep the
publisher runtime until a verified Windows build passes model correctness,
throughput and quality gates. Initial graph reservation and rank-one fusion
remain unimplemented alternatives; no training, adapter quantization or dense
merge is part of this overlay.

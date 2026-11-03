# Prism Inference 0.1.14

This prepares the Rust library, source tools and Windows ARM64/x64 CPU bundles
for publication after exact-source release checks pass. It follows `v0.1.13`
(`cae16fa6998dac58a947eba9ae4398ed6a9fb863`); the earlier release and its retained
model/device observations keep their original identities.

## Changes since 0.1.13

Encoded prompt token-capacity overflow now preserves a value-free concrete
`TokenLimitExceeded` error through the existing boxed Rust result APIs. Native
Swift admission recognizes only that concrete error under its validated context
reservation. Matching display text does not classify an error. Tokenizer
corruption, schema failures, byte/piece safety bounds and allocation failures
retain their failure classification.

The authenticated private count endpoint returns `413 Content Too Large` with
exactly `{"error":{"code":"context_length_exceeded"}}` for a genuine context
rejection before output starts. Other failures retain the generic failure
response. A rejected count clears the previous prepared-prompt seal, starts no
generation and leaves a healthy owner available for a fresh count/generation
exchange. This does not recover a faulted generation or replay any request.

Model-free regression source covers typed output/insertion bounds, corruption
and safety-limit distinctions, malformed UTF-8 error classification, spoofed
error text, authenticated HTTP framing, stale-count rejection and same-owner
recovery without reset. Exact-source owning CI must establish those checks.
No new owner-device trial or completed full service gate is asserted here.

## Supported release scope

CPU arithmetic, model/adapter pins, sampling, context/output bounds, admission,
memory reserves, required containment and rollback remain unchanged. Original
PQ2, one row worker, no adapter and fresh private-service request state remain
the defaults. PTQ1 and approved adapters still require explicit selection and
independent qualification. This release does not qualify a new application
default, numerical parity, adapter provenance, performance, quality or comfort.

The Rust error marker is additive; existing result aliases and successful
count/generation schemas remain compatible. The private context-error response
is now precise. Worker names and the five CPU executable names are unchanged.
Applications must explicitly recognize the bounded private error contract and
verify the new release before selecting it.

## Installation and provenance

Publication requires successful Foundation, runtime conformance and source-package
checks on the exact release commit. Attach the unchanged ARM64 and x64 Foundation
archives and matching integrity receipts from that commit. Receipts bind source,
compiler, target, all five executable hashes, the four license/notice files and
the archive digest. Release notes must link the successful owning runs.

The [build guide](build-and-validation.md) documents versioned source installation
and bundles. Dependency notices and pins are unchanged: `bonsai-native-hash`
remains `0.1.0`, `bonsai-gguf` remains `0.1.1`, and the server retains its exact
Git credential-helper dependency and unpublished status. Registry publication
remains separate. A Runner repin must independently verify the source, archive
and file identities; passing hosted checks does not reopen a closed laptop
allowance or qualify this release on an actual model.

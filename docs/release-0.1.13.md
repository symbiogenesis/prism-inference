# Prism Inference 0.1.13

This CPU release rolls up the source changes since `v0.1.12`
(`c48687616f55eec9aff0de04860917de3e1047f2`). It prepares the Rust library,
source tools and Windows ARM64/x64 CPU bundles for publication after the exact
source gates below pass. Earlier releases and their evidence remain available.

## Changes since 0.1.12

| Area | Result |
| --- | --- |
| Graph construction | Immutable tensor payloads are validated once before graph publication, including allowed unused tensors. Shape binding no longer repeats that full payload scan. PTQ1 validation retains extent and finite-scale checks without decoding and discarding valid packed trits. |
| Request attribution | Optional runtime and Swift session timing APIs separate prefill, decoding, sampling, provisional callbacks, snapshots and recovery. Returned failures preserve completed lower bounds and unknown partial work. Ordinary calls read no additional attribution clocks. |
| Diagnostic delivery | A bounded source checker uses the production Swift sampler and an original absolute deadline covering input, loading and request work. A hosted main-only workflow prepares its standalone ARM64 executable, exact source/build receipt and focused model-free test logs. The checker is separate from the five executable CPU bundle. |
| Explicit projection integration | Source APIs support borrowed PQ2/PTQ1 projection executors and complete-request operation admission. Experimental Hexagon helpers add bounded guarded/HMX execution, immutable packed-weight arenas and explicit partial placement. These helpers remain unpublished and are not dependencies of the CPU executables. |

The construction change removes redundant work in source. No actual-model
startup, throughput or memory improvement is claimed. The existing CNG artifact
verifier was already present in 0.1.12 and is unchanged.

## Supported release scope

Original PQ2, one row worker, no adapter and fresh private-service request state
remain the defaults. PTQ1, approved adapters, extra row workers and exact-prefix
reuse still require explicit selection. CPU quantization, projection reduction
order and the production copy-then-scale recurrent arithmetic are unchanged.
The proposed fused recurrent update remains test-only.

The ordinary CPU graph supplies no external projection executor. The new
`Decoder::admit_work` hook defaults to accepting CPU work; explicitly wrapped
decoders use it to reject requests exceeding their remaining native operation
budget. The experimental projection APIs do not enable a service accelerator
profile or establish signed-device acceptance, full-model numerical parity or
accelerator performance. Native resources remain outside graph checkpoints.

Worker protocols, the private HTTP schema, Rust library import name and the
five CPU executable names are unchanged. The release enables no model or
adapter download and does not modify an application's runtime lock or defaults.

Model-free regressions cover invalid payload rejection, timed/untimed output,
state and sampler equivalence, stop draws, cancellation, callback failure,
rollback and incomplete prefill accounting. They do not transfer historical
actual-model evidence to these new binaries. Full-context trajectories, adapter
provenance, quality, sustained comfort, application isolation and installation
acceptance remain governed by the [roadmap](roadmap.md),
[profile contract](swift-session-profiles.md) and [adapter contract](rank-one-adapter.md).
This release does not qualify PTQ1 as a default.

## Installation and provenance

Publication requires successful Foundation, runtime conformance and source-package
checks on the exact release commit. Attach the unchanged ARM64 and x64 Foundation
archives and matching integrity receipts from that commit. Each receipt binds
the source, pinned compiler, target, executable inventory and archive/file
SHA-256 digests. The published release notes must link those successful owning
runs; a passing earlier source build does not satisfy this requirement.

The [build guide](build-and-validation.md) documents the versioned Git source and
five executable bundles. The standalone request-timing tool has its own hosted
preparation receipt and is not a substitute for released service acceptance.

Registry publication remains separate. This release changes no registry
dependency versions: `bonsai-native-hash` remains `0.1.0`, `bonsai-gguf` remains
`0.1.1`, and the private server retains its exact Git credential-helper dependency
and unpublished status. A Runner repin must independently verify the new source,
archive and file identities and retain earlier device observations at their
original identities.

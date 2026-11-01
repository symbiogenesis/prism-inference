# Prism Inference 0.1.12

This full release rolls up the improvements since the last full release,
`v0.1.6` (`02c8a24b31ac1c35f2ffd3530f6a753ce440bdc0`). It supplies the Rust library,
source tools and prebuilt Windows ARM64/x64 CPU bundles. A full release does not
make the experimental model profiles or accelerators qualified defaults.

## Improvements since 0.1.6

| Area | Result |
| --- | --- |
| Contained startup | Exact read paths are checked inside AppContainer without requiring broader directory access; bounded startup-stage diagnostics retain no artifact paths or values. |
| Model profiles | The Swift session and private HTTP service can explicitly select the verified current-weight PTQ1 export and approved rank-one adapter. Adapter scales are bounded and bound to prepared requests; changing a scale clears incompatible live state and checkpoints. |
| CPU workers | The optional `parallel-rows` feature provides a persistent, explicitly budgeted packed-row pool. Private readiness reports the instantiated session's effective worker budget, allowing the owner to reject a policy mismatch. |
| Artifact verification | GGUF verification uses Windows CNG SHA-256 with a portable Rust implementation elsewhere. Exact file length, digest and the retained read-only handle remain mandatory; existing payload-loading progress/cancellation behavior is unchanged. |
| State reuse | Library callers can explicitly select `PromptReuse::Latest`; ownership, exact-prefix and adapter-scale checks prevent incompatible reuse. The private service resets for every request by default; latest-prefix reuse requires explicit selection. |
| Correctness and delivery | Added regressions cover row-worker parity and failure recovery, sampled and hybrid-state reuse, adapter profile boundaries and hash streams. Source-only rollback and private-service capture tools support further bounded qualification. |
| Rust consumers | GGUF now has its own `0.1.1` source identity and the runtime pins it exactly. Hosted source-package checks exercise packaged dependencies and an external library consumer. Installation and library examples distinguish GitHub source/binaries from registry publication. |

The production recurrent update retains the prior copy-then-scale arithmetic.
The proposed fused copy/decay implementation is test-only pending its own
qualification. This release claims no measured startup, throughput or memory gain.

## Supported release scope

The first generation path remains the pinned Swift/Bonsai dense Qwen35 CPU
architecture. Original PQ2, one row worker, no adapter and fresh private-service
request state remain the defaults. The runtime has no C/C++ inference dependency;
the existing Rust import path remains `bonsai_prism_runtime`.

PTQ1, approved-adapter execution, extra row workers and exact-prefix reuse require
explicit selection. Their device, numerical, quality and resource qualification
remains scoped to the exact evidence already documented in the
[roadmap](roadmap.md), [worker contract](parallel-rows.md),
[adapter contract](rank-one-adapter.md) and
[profile contract](swift-session-profiles.md). Full-context trajectories, adapter
authoring provenance, sustained comfort, application isolation and installation
acceptance remain open. These limits also prevent automatic promotion in Runner.

The isolated, unpublished native Hexagon diagnostic crate is outside the five
bundled CPU executables. It enables no model offload or accelerator profile.
Broader model architectures and general model selection remain future work.

## Installation and provenance

Use the versioned Windows archives and matching integrity receipts attached to
the release, or the pinned Git source instructions in the
[build guide](build-and-validation.md). Models and adapters are separate artifacts.
Each binary receipt records its source revision, compiler, target, complete file
inventory and archive/file SHA-256 digests. Older releases remain available for
rollback; this release does not repin an application's existing runtime lock.

The release requires successful Foundation, runtime conformance and source-package
checks on its exact source revision. Binary attachments must be the unchanged
Foundation artifacts for that revision. The published release notes retain links
to those successful owning runs.

Registry publication is a separate delivery channel: publish `bonsai-native-hash`
`0.1.0`, then `bonsai-gguf` `0.1.1`, then `prism-inference` `0.1.12`. Source-package
checks exercise these packages before publication; they do not publish them.
The private server retains its exact Git credential-helper dependency and is not
registry-published. Use the documented Git source dependency until the matching
registry versions are available.

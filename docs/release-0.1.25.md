# Prism Inference 0.1.25 registry delivery

The runtime and all five public helpers were published to crates.io on November
12, 2026, from clean local source
`99bab71270b1e6a6fb6d47eec2e34b8097cdcd4c`. Normal Cargo package/publication
verification passed for every crate. The normalized archives retained that
source identity and exact registry pins, and all six public registry checksums
matched the uploaded bytes.

| Package | Published version |
| --- | --- |
| [`bonsai-native-hash`](https://crates.io/crates/bonsai-native-hash/0.1.3) | `0.1.3` |
| [`bonsai-ternary`](https://crates.io/crates/bonsai-ternary/0.1.3) | `0.1.3` |
| [`bonsai-prism`](https://crates.io/crates/bonsai-prism/0.1.5) | `0.1.5` |
| [`bonsai-neon-dot`](https://crates.io/crates/bonsai-neon-dot/0.1.2) | `0.1.2` |
| [`bonsai-gguf`](https://crates.io/crates/bonsai-gguf/0.1.5) | `0.1.5` |
| [`prism-inference`](https://crates.io/crates/prism-inference/0.1.25) | `0.1.25` |

The runtime archive contains 575 members and is 3,771,972 bytes, with SHA-256
`0a117db7589a821b27c224e41aaababa5ddd2ac9852344c0a8121a0849046363`.
Every versioned local dependency advances with its package. The server package
and its exact runtime dependency advance together to 0.1.25, but the server,
Hexagon crates and qualification tools retain `publish = false`.

Live crates.io metadata identified three outdated direct third-party pins:
`fancy-regex` 0.14.0 to 0.20.2, `fearless_simd` 1.1.0 to 1.2.0, and the
independent reference `tokenizers` 0.22.2 to stable 0.23.2. Other direct registry
dependencies already selected their latest stable versions. The `tokenizers`
1.0 prerelease is excluded. Its default features remain disabled; `fancy-regex`
stays selected without native regex or C++ trainer activation. The lockfile
also refreshes compatible transitive packages. Tokenizers retains its own
compatible `fancy-regex` 0.17 dependency alongside production 0.20.2.

The initial selected all-target/parallel-row compilation at `839c2e4` found
four tokenizers ownership API migrations. Commit `99bab71` passes owned tokens
and propagates the new fallible results. That failed preflight was not repeated.
The subsequently started scalar library check, seven GGUF tokenizer controls,
two focused Hugging Face reference controls and all six normal Cargo publication
verifications passed. Full all-target compilation, strict analyzers and hosted
source-package consumer variants remain pending; these focused checks do not
substitute for Foundation or runtime conformance.

The single operation reached its completed checksum audit in 266.6 seconds,
initial evidence retention in 324.222 seconds, and final retention after artifact
cleanup in 471.988 seconds. This fits its predeclared 600-second ceiling and
480-second conservative estimate, including the initial compiler failure, source
correction, publication, a temporary-registry archive lookup correction,
checksum verification, retention and cleanup. No publication was repeated.
Power snapshots before and after each workload reported AC disconnected and
battery saver off. Charge declined from 74% initially to 72% at the completed
audit and 71% at both retention observations. Snapshots do not prove uninterrupted
stability; no performance comparison is qualified. Owned child Jobs were closed
and direct processes reaped. Task-generated package source directories and both
staging sets of six crate archives were removed; compiled caches were retained.
Runner retains the concise commands, results, hashes, power, budget and cleanup
[registry-delivery record](https://github.com/symbiogenesis/SwiftBonsai2Runner/blob/main/docs/evidence/prism-0125-publish-20261112/summary.json).

This release contains the reviewed CPU/kernel and KV source changes after
0.1.24, including the new NEON helper APIs and the K4/V4 (`rsq4`) development
default. Explicit FP16 recovery remains available. Source availability and
package verification establish no full-model quality, device performance,
complete Foundation or cross-target qualification pass.

The last verified Windows ARM64 binary bundle remains 0.1.23. A crates.io
release does not supply Runner's source-bound five-executable bundle. Owner
source publication, a matching hosted bundle and source/archive/nine-file
verification remain prerequisites for Runner's active binary pin update.
Private server/Hexagon and model/state/adapter acceptance remain owner-operated
checks. Existing diagnostic deadlines stay closed. No Git push, tag, GitHub
release, CI operation, model or benchmark ran during this delivery.

# Prism Inference 0.1.25 registry release candidate

This candidate advances the runtime and every publishable Prism helper. Registry
publication and checksum verification are still pending; the last verified
Windows ARM64 binary bundle remains 0.1.23. A crates.io release does not supply
Runner's source-bound five-executable bundle or authorize a binary pin change.

| Package | New exact version |
| --- | --- |
| `bonsai-native-hash` | `0.1.3` |
| `bonsai-neon-dot` | `0.1.2` |
| `bonsai-ternary` | `0.1.3` |
| `bonsai-prism` | `0.1.5` |
| `bonsai-gguf` | `0.1.5` |
| `prism-inference` | `0.1.25` |

Publish the hash, NEON, ternary and overlay helpers before GGUF, then the runtime.
Every versioned local dependency advances with its package. The server package
and its exact runtime dependency advance together to 0.1.25, but the server,
Hexagon crates and qualification tools retain `publish = false`.

Live crates.io metadata on November 12, 2026, identified three outdated direct
third-party pins: `fancy-regex` 0.14.0 to 0.20.2, `fearless_simd` 1.1.0 to 1.2.0,
and the independent reference `tokenizers` 0.22.2 to stable 0.23.2. Other direct
registry dependencies already selected their latest stable versions. The
`tokenizers` 1.0 prerelease is excluded. Its default features remain disabled;
`fancy-regex` stays selected without native regex or C++ trainer activation.
The lockfile also refreshes compatible transitive packages. Tokenizers retains
its own compatible `fancy-regex` 0.17 dependency alongside production 0.20.2.

This release contains the reviewed CPU/kernel and KV source changes after
0.1.24, including the new NEON helper APIs and the K4/V4 (`rsq4`) development
default. Explicit FP16 recovery remains available. Source availability and
package verification establish no full-model quality, device performance,
complete Foundation or cross-target qualification pass.

Cargo package/publication verification, focused checks and public archive
identities must be recorded by the bounded delivery operation. Full Foundation,
runtime conformance, source-package consumer variants, private server/Hexagon
acceptance and model/state/adapter qualification remain owner-operated checks.
Existing diagnostic deadlines stay closed. No Git push, tag, GitHub release,
CI dispatch or new performance measurement is implied by this candidate.

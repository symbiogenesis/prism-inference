# Prism Inference 0.1.26 registry delivery

The runtime and changed public NEON helper were published to crates.io on
November 13, 2026, from clean source
`53f2fcbc89bbdd43ca4b83aea234096626f26040`. NEON 0.1.3 was published first;
runtime 0.1.26 pins it exactly. Ordinary Cargo publication verification passed
for both packages, and their registry checksums match the uploaded archives.

| Package | Archive bytes | SHA-256 |
| --- | ---: | --- |
| [`bonsai-neon-dot 0.1.3`](https://crates.io/crates/bonsai-neon-dot/0.1.3) | 30,416 | `a97d60b4dbcbbbab0b3f840f35c14c7213cd73190fa5a50d4027ed6b30abfe96` |
| [`prism-inference 0.1.26`](https://crates.io/crates/prism-inference/0.1.26) | 3,843,685 | `a72bb5d530c593603c962981364ae64640212fa1d823837ec7cb481f23807c0f` |

Native-hash/ternary remain 0.1.3 and GGUF/Prism remain 0.1.5: their source has
not changed since the [0.1.25 delivery](release-0.1.25.md). The private server
version, its exact runtime path dependency and workspace lockfile remain
coherent at 0.1.26. The server, Hexagon helpers and qualification tools retain
`publish = false`.

The NEON helper adds borrowed, prechecked finite queries for packed-key scans
and an owned packed-value accumulator. Each row still checks extent and finite
results; an invalid append permanently prevents publication of its accumulated
sum. Native and portable paths retain their existing arithmetic. The runtime
consumes those APIs and includes prepared PQ2 row validation and compatible
CPU sibling prefill projections. Source changes alone establish no speed gain.

The candidate includes
[selected-cache context accounting](runtime-contracts.md#memory-and-ownership)
for native sequential serving, cache-aware state/checkpoint budgets, and complete
FP16 tiled-score reservations. Its conservative source ceilings are 62,720 FP16,
241,664 K4/V4 and 262,144 K4/V3 tokens, subject to the selected memory allowance.
These bounds establish no larger-window quality, latency or measured memory
result. Explicit FP16 recovery and separate tile/speculation reserves remain.

Source preparation also includes trusted private-child `--placement auto|cpu`.
The CPU selection bypasses accelerator discovery/initialization, clears any
preinstalled projection executor and identifies CPU placement/backend in private
readiness. Automatic placement preserves existing behavior and readiness.
Memory allowances remain unchanged; this supplies no measured capacity or
latency result and requires separate exact-source Runner integration.

## Separate binary source and pending gates

The current published Windows ARM64 bundle remains
[0.1.25](https://github.com/symbiogenesis/prism-inference/releases/tag/v0.1.25),
from source `99bab71270b1e6a6fb6d47eec2e34b8097cdcd4c`. Its archive is
5,715,851 bytes with SHA-256
`22e1ba7e5cf3d19136bcf03a0e451dbfe58a9792bde457652c3f21841ffc5154`;
its 1,971-byte receipt has SHA-256
`ea54d93d9367122c372b5bc3396ccf112ecaca7900714ed19ed1f08a5ef42bbf`.
These are historical 0.1.25 identities, independent of the new registry source.

The selected Windows binary source is
`3d735627df43056d3bf569002af2c23af30a3e6a`. It follows the registry source with
two explicit raw-pointer fixture references and removal of two stale Clippy
expectations in three runtime source files. NEON helper source is unchanged.
These revisions must remain distinct even though both identify runtime 0.1.26;
registry archive verification does not validate a binary built from later source.

The first [Foundation bundle attempt](https://github.com/symbiogenesis/prism-inference/actions/runs/43477092273)
was cancelled without producing artifacts. The owner-authorized
[subsequent attempt](https://github.com/symbiogenesis/prism-inference/actions/runs/43478391022)
is pending. No 0.1.26 binary release, archive digest or source-bound receipt is
claimed here. Registry publication and the five-executable bundle remain
independent deliveries.

Complete Foundation, the independent source-package consumer matrix and
actual-model/state/resource qualification remain pending. Earlier focused checks
and closed context-check allowances do not qualify later source. Runner must
review a genuine published bundle's source, archive and nine-file identities
before updating its active runtime pin and activating this context protocol.
The selected published 0.1.25 CPU comparison and historical evidence retain their
existing versions, checksums and deadlines. No performance comparison or
larger-window acceptance is established by this registry delivery.

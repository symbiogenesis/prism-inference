# Prism Inference 0.1.26 registry and Windows ARM64 delivery

The runtime and changed public NEON helper were published to crates.io on
November 13, 2026, from clean source
`53f2fcbc89bbdd43ca4b83aea234096626f26040`. NEON 0.1.3 was published first;
runtime 0.1.26 pins it exactly. Ordinary Cargo publication verification passed
for both packages. Downloaded public archives match their registry checksums;
Cargo's retained extracted package metadata identifies the registry source above.

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

The release includes
[selected-cache context accounting](runtime-contracts.md#memory-and-ownership)
for native sequential serving, cache-aware state/checkpoint budgets, and complete
FP16 tiled-score reservations. Its conservative source ceilings are 62,720 FP16,
241,664 K4/V4 and 262,144 K4/V3 tokens, subject to the selected memory allowance.
These bounds establish no larger-window quality, latency or measured memory
result. Explicit FP16 recovery and separate tile/speculation reserves remain.

The release also includes trusted private-child `--placement auto|cpu`.
The CPU selection bypasses accelerator discovery/initialization, clears any
preinstalled projection executor and identifies CPU placement/backend in private
readiness. Automatic placement preserves existing behavior and readiness.
Memory allowances remain unchanged; this supplies no measured capacity or
latency result. Runner's atomic repin consumes the published bundle and its
matching exact source; actual-model qualification remains separate.

## Separate binary source and pending gates

Subsequent read-only PE inspection found `vcruntime140.dll` imports in all five
published executables. The archive contains no runtime DLL, so 0.1.26 requires
the compatible external ARM64 Visual C++ runtime. Its published assets remain
unchanged. The [0.1.27 preparation](release-0.1.27.md) targets a separately
verified static-runtime replacement and supplies no new release bytes yet.

The published experimental [0.1.26 Windows ARM64 bundle](https://github.com/symbiogenesis/prism-inference/releases/tag/v0.1.26)
comes from source `ac941f2668ae93dba9824002b4597081457aa09f` and successful
[Foundation bundle run 43480377743](https://github.com/symbiogenesis/prism-inference/actions/runs/43480377743).

| Published asset | Bytes | SHA-256 |
| --- | ---: | --- |
| Windows ARM64 runtime archive | 5,837,601 | `ed7663c178ee33903bdd2854d2bd5957f441bfce19fc64d4efa99239fc1ce602` |
| Source-bound bundle receipt | 1,979 | `422c4c3d95e4fe4d05b336956844a9e8c173490720a08f65a918aa9384750aa7` |
| Source ZIP | 5,103,161 | `1623bd210f8fcf1eb38c5db9747d18f5cb943ab36500ac413ecae1a98be7042a` |

The hosted run passed all nine receipt/version controls, its selected strict
Release analyzers and scalar library compilation, then linked and packaged all
five executables. Trusted `bonsai-tools` review verified source, all nine file
hashes and the five executables' Windows ARM64 PE targets. Published assets were
downloaded again and their hashes verified. The binary source follows registry source with the
`3d73562` fixture/lint corrections described below, then documentation and CI
policy changes; no additional runtime behavior change lies between those sources.

The release operation had elapsed 339.5 seconds before final offline review.
The operator's before/after observations showed AC connected, battery saver off
and 95 percent charge; they do not establish uninterrupted power stability.
Hosted build power is unknown. The owner's new instruction applied GitHub's
cap without a local release-build deadline. No performance result is inferred.

Runner's atomic runtime/source repin `bf19ea0b` consumes this exact binary and
source pair. Complete Foundation, the independent source-package consumer matrix,
and actual-model, state, resource, quality and performance acceptance remain
unqualified. The bundle-only run does not close those separate gates.

The previous [0.1.25 Windows ARM64 bundle](https://github.com/symbiogenesis/prism-inference/releases/tag/v0.1.25)
retains source `99bab71270b1e6a6fb6d47eec2e34b8097cdcd4c`. Its archive is
5,715,851 bytes with SHA-256
`22e1ba7e5cf3d19136bcf03a0e451dbfe58a9792bde457652c3f21841ffc5154`;
its 1,971-byte receipt has SHA-256
`ea54d93d9367122c372b5bc3396ccf112ecaca7900714ed19ed1f08a5ef42bbf`.
These are historical 0.1.25 identities, independent of the new registry source.

The source selected for the first two Windows bundle attempts was
`3d735627df43056d3bf569002af2c23af30a3e6a`. That revision follows the registry
source with two explicit raw-pointer fixture references and removal of two stale Clippy
expectations in three runtime source files. NEON helper source is unchanged.
These revisions must remain distinct even though both identify runtime 0.1.26;
registry archive verification does not validate a binary built from later source.

The first [Foundation bundle attempt](https://github.com/symbiogenesis/prism-inference/actions/runs/43477092273)
was cancelled without producing artifacts. The owner-authorized
[subsequent attempt](https://github.com/symbiogenesis/prism-inference/actions/runs/43478391022)
reached its five-minute job limit while compiling the five executables, without
producing artifacts. Both attempts remain closed. The owner's subsequent
instruction removes repository-imposed hosted release-build deadlines in favor
of GitHub's own cap. The workflow retains its exact-source and operation-ID
guards and all nine receipt/version controls. Owner-device diagnostic, model
and benchmark limits remain unchanged. The later successful run and published
assets above supply their own evidence. Registry publication and the
five-executable bundle remain independent deliveries.

Earlier focused checks and closed context-check allowances do not qualify later
source. The selected published 0.1.25 CPU comparison and historical evidence retain their
existing versions, checksums and deadlines. No performance comparison or
larger-window acceptance is established by this registry delivery.

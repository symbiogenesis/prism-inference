# Prism Inference 0.1.26 source preparation (unreleased)

The local runtime and private server identify the next development candidate as
0.1.26. The server's exact path dependency and workspace lockfile advance
together. The changed public NEON helper advances from 0.1.2 to 0.1.3, and the
runtime pins it exactly. Native-hash/ternary remain 0.1.3 and GGUF/Prism remain
0.1.5: their source has not changed since the 0.1.25 delivery. No package, tag,
GitHub release or binary bundle has been published by this preparation.

Publish `bonsai-neon-dot 0.1.3` before `prism-inference 0.1.26`, using ordinary
Cargo package/publication verification for both. The server, Hexagon helpers
and qualification tools remain private. Registry publication and the
five-executable bundle are independent deliveries; one cannot substitute for
the other.

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

The earlier version-only preparation received static manifest/lock consistency
review. Its prior context-check allowance remains closed. Earlier focused checks
do not qualify this candidate's later placement changes. Complete
Foundation, source-package verification, exact-source bundle production and
actual-model/state/resource qualification remain pending.

Runner's active 0.1.25 binary and exact-source pins remain unchanged at 8K. The
owner must publish reviewed source and obtain a genuine source-bound compatible
bundle before Runner can review its archive/file identities and activate this
context protocol. Registry installation and the selected published 0.1.25 CPU
comparison retain their existing versions and checksums. No future release URL,
asset digest or performance claim is implied.

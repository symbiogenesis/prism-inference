# Prism Inference 0.1.26 source preparation (unreleased)

The local runtime and private server identify the next development candidate as
0.1.26. The server's exact path dependency and workspace lockfile advance
together. Public helper versions remain unchanged: native-hash/ternary 0.1.3,
NEON 0.1.2 and GGUF/Prism 0.1.5. No package, tag, GitHub release or binary bundle
has been published by this preparation.

The candidate includes
[selected-cache context accounting](runtime-contracts.md#memory-and-ownership)
for native sequential serving, cache-aware state/checkpoint budgets, and complete
FP16 tiled-score reservations. Its conservative source ceilings are 62,720 FP16,
241,664 K4/V4 and 262,144 K4/V3 tokens, subject to the selected memory allowance.
These bounds establish no larger-window quality, latency or measured memory
result. Explicit FP16 recovery and separate tile/speculation reserves remain.

This version-only change received static manifest/lock consistency review. It
starts no new compiler, model or test workload; the prior context-check allowance
remains closed. Earlier focused checks do not qualify this version. Complete
Foundation, source-package verification, exact-source bundle production and
actual-model/state/resource qualification remain pending.

Runner's active 0.1.25 binary and exact-source pins remain unchanged at 8K. The
owner must publish reviewed source and obtain a genuine source-bound compatible
bundle before Runner can review its archive/file identities and activate this
context protocol. Registry installation and the selected published 0.1.25 CPU
comparison retain their existing versions and checksums. No future release URL,
asset digest or performance claim is implied.

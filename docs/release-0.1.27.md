# Prism Inference 0.1.27 registry and Windows ARM64 delivery

The registry runtime and experimental [Windows ARM64 bundle](https://github.com/symbiogenesis/prism-inference/releases/tag/v0.1.27)
were published on November 13, 2026, from the same source:
`1429a33772089dfd3a50a96c698b9dbe0b36836a`. The private server and its exact
runtime pin also identify 0.1.27. Public helpers remain unchanged:
NEON/native-hash/ternary 0.1.3 and GGUF/Prism 0.1.5.

| Published item | Bytes | SHA-256 |
| --- | ---: | --- |
| [`prism-inference 0.1.27`](https://crates.io/crates/prism-inference/0.1.27) | 3,859,075 | `5e90c1cf72edc0cb7143993092d494c11d33977442865fecff98e74696e13e29` |
| Windows ARM64 runtime ZIP | 6,152,272 | `52c7bb325b0f5e4af7b4cf176c65408f4b9612d55faebdac3c87aa5ddef8e015` |
| Source-bound nine-file receipt | 1,979 | `9003b30b7bfa5fa43b2d439c193ff9a9690933debbfe15c25d09d0360c086984` |

Ordinary locked Cargo publication verification passed. The public registry
archive's normalized manifest and source identity were verified separately from
the binary delivery. The private server and Hexagon helpers remain unpublished
registry packages.

## Verified delivery scope

[Foundation bundle run 43490214802](https://github.com/symbiogenesis/prism-inference/actions/runs/43490214802)
passed three version, six receipt and five data-only PE controls, the selected
strict Release analyzers and scalar library compilation. Two exact unwind/drain
controls and six exact context controls each passed once, then all five
executables linked and were packaged with four license/notice files.

The two unwind controls cover a tiled-worker panic's completion/join guard and
a failed worker group's drain followed by plan reuse. The six context controls
cover all three `graph::swift_state` geometry/identity/generic-cap fixtures,
selected-cache capacity, full-checkpoint reservation and private-child
capacity/cache argument ordering through 262,144 tokens. These selections do
not qualify complete Foundation or an actual model.

The bundle uses an explicit Windows ARM64 target and target-specific
`+crt-static`, preserving host-tool linkage and Rust panic unwinding. Its
`-RequireStaticRuntime` packaging gate validates normal and delay-load imports
in every executable before output is created and rejects separately installed
VC runtimes. Windows system CRT/API-set imports remain allowed. This statically
links Microsoft's runtime; it is not a pure Rust CRT/EH replacement.

The published 0.1.26 archive still imports `vcruntime140.dll` in all five
executables and contains no such DLL. Its assets and prerequisite remain
unchanged; Runner's own linkage cannot alter a downloaded runtime's imports.

## Scoped K4/V4 context accounting

The public `swift_session::state_limit_bytes` query reports the selected cache's
aggregate live-state/checkpoint ceiling. K4/V4 (`Rotated4`) advances from 241,664
tokens to the unchanged 262,144-token model ceiling through a private,
exact-identity/geometry/recurrence check. Its 16 attention and 48 recurrent
layers derive the 4,644,798,464-byte aggregate allowance. Generic constructors,
individual allocations and scratch retain their 4 GiB bounds; FP16/K4-V3 bounds,
default 8K library APIs and paired context/memory child arguments remain intact.
Speculation still funds a complete checkpoint. Runner must bind
`K4V4_FULL_CONTEXT_RELEASE` to the reviewed memory release through atomic source
and binary pins before activating this scope.

## Previous attempts and limits

| Hosted bundle run | Retained outcome |
| --- | --- |
| [43487080842](https://github.com/symbiogenesis/prism-inference/actions/runs/43487080842) | Failed formatting after the concurrent scoped-state change; no release artifact. |
| [43488012054](https://github.com/symbiogenesis/prism-inference/actions/runs/43488012054) | Failed strict Clippy on private-module constant visibility. The 3 version, 6 receipt and 5 PE controls passed; later Rust controls/linking/packaging were skipped. |
| [43489026858](https://github.com/symbiogenesis/prism-inference/actions/runs/43489026858) | Selected full strict Release analyzers passed, then scalar compilation rejected three unused CLI-only items; later Rust controls/linking/packaging were skipped. |

The final source preserves the owner's scoped-state behavior, corrects
formatting/visibility and gates CLI production items to their consumers while
retaining unit fixtures. The corrected scalar source passed one focused local
compatibility check in 7.349 seconds within its declared 90-second allowance;
power endpoints showed AC connected, saver off and 100 percent charge.

Normal publication verification completed in 21.726 seconds within its fixed
300-second allowance and 90-second estimate; its separate endpoints showed AC
connected, saver off and 99 percent charge. Endpoint snapshots do not establish
uninterrupted power stability. Hosted power is unknown. The release build used
GitHub's cap without a local cancellation deadline; device/model/diagnostic
limits and closed attempts remain unchanged.

Full Foundation, actual-model/state/resource behavior, larger-window quality,
performance and clean-machine acceptance remain unqualified. These source,
linkage, model-free and publication checks do not transfer earlier qualification
or establish measured memory or speed gains.

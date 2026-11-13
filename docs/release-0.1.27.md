# Prism Inference 0.1.27 static-runtime and full K4/V4 preparation (unreleased)

The runtime and private server advance together to 0.1.27, including the
server's exact runtime dependency and workspace lockfile. Public helpers remain
unchanged: NEON/native-hash/ternary 0.1.3 and GGUF/Prism 0.1.5. Their source
bytes remain unchanged. The combined candidate also includes the scoped Swift
state-budget API and full K4/V4 context admission described below.
No 0.1.27 registry package, tag or binary bundle is claimed by this preparation.

Read-only inspection of the published 0.1.26 bundle found `vcruntime140.dll`
imports in all five executables, while its nine-file archive contains no such
DLL. That release requires the compatible external ARM64 Visual C++ runtime.
Runner's own static linking cannot remove a downloaded runtime's dependency.
The 0.1.26 assets, source identities and historical evidence remain immutable;
the replacement uses a new version rather than changing those assets.

## Scoped K4/V4 context accounting

The public `swift_session::state_limit_bytes` query reports the selected Swift
cache's aggregate live-state/checkpoint ceiling. `Rotated4` (K4/V4) advances from
the earlier 241,664-token limit to the unchanged 262,144-token model ceiling.
The private constructor authorizes its 4,644,798,464-byte aggregate allowance
only after verifying the exact Swift artifact identity, model geometry and
recurrence layout: 16 attention layers and 48 recurrent layers. Generic graph
constructors, individual allocations and scratch retain their existing 4 GiB
bounds; FP16 and K4/V3 aggregate bounds and default 8K library APIs remain intact.
Speculation still reserves a complete checkpoint at the selected capacity.

The paired private-child `--context-tokens` / `--memory-limit-bytes` interface
is unchanged. Runner must bind `K4V4_FULL_CONTEXT_RELEASE` to the exact reviewed
memory release through its atomic source/binary pins before activating this
larger scope. Source accounting and model-free fixtures establish no larger-window
quality, actual memory, device latency or performance qualification.

The dedicated bundle lane selects six existing model-free context controls:
all three exact geometry/identity/generic-cap fixtures in `graph::swift_state`,
the selected-cache capacity and full-checkpoint session fixtures, and the private
child's paired capacity/cache argument-order fixture through 262,144 tokens.
Each exact selection must pass once. Their scope is recorded separately from
the two unwind/drain controls and complete Foundation remains unqualified.

## Prepared delivery contract

The dedicated Foundation bundle lane keeps its explicit Windows ARM64 Cargo
target and selects `+crt-static` through
`CARGO_TARGET_AARCH64_PC_WINDOWS_MSVC_RUSTFLAGS`. Host build tools and procedural
macros retain their ordinary host linkage. Its five executables still come from
the existing target-triple `release` directory. This linkage change accompanies
the separately scoped state-budget changes above.

This statically links Microsoft's runtime; it is not a pure Rust CRT/EH
replacement. Existing panic unwinding remains enabled. Two named model-free
Rust controls check that linkage: the tiled-worker panic
completion guard and failed-worker-group drain followed by plan reuse. Neither
control is a model, benchmark or complete correctness suite.

The lane passes `-RequireStaticRuntime` to the ordinary bundle producer. Before
creating output, a bounded data-only PE32+ check validates normal and delay-load
imports in all five images and rejects separately installed Visual C++ runtimes,
including `vcruntime`, `msvcp`, `concrt` and `vcomp`. Windows system CRT/API-set
imports remain permitted. Malformed headers, sections, import descriptors and
names fail closed. The check loads no image or DLL and uses no third-party parser.
Ordinary Foundation behavior remains unchanged; its independently produced
bundles receive no static-runtime claim from a matching package version alone.

First produce the genuine exact-source hosted bundle, independently inspect
every PE/import and verify all source/archive/file identities. After its selected
checks pass, publish `prism-inference 0.1.27` using ordinary Cargo verification;
the unchanged helpers are already published. The private server and Hexagon
helpers remain unpublished registry packages. Publish the reviewed bundle and
use its exact binary/source identities for Runner's atomic repin.

The hosted release build uses GitHub's cap without a repository job timeout or
local operator cancellation deadline. Owner-device, model and diagnostic limits
remain unchanged. Registry verification, hosted analyzers/linking and import
checks do not qualify full Foundation, model/state/resource behavior, quality
or performance; those gates remain pending with their own exact identities.

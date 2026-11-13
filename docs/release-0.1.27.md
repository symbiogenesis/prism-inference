# Prism Inference 0.1.27 static-runtime preparation (unreleased)

The runtime and private server advance together to 0.1.27, including the
server's exact runtime dependency and workspace lockfile. Public helpers remain
unchanged: NEON/native-hash/ternary 0.1.3 and GGUF/Prism 0.1.5. Runtime APIs,
inference behavior and helper source are unchanged from the 0.1.26 binary source.
No 0.1.27 registry package, tag or binary bundle is claimed by this preparation.

Read-only inspection of the published 0.1.26 bundle found `vcruntime140.dll`
imports in all five executables, while its nine-file archive contains no such
DLL. That release requires the compatible external ARM64 Visual C++ runtime.
Runner's own static linking cannot remove a downloaded runtime's dependency.
The 0.1.26 assets, source identities and historical evidence remain immutable;
the replacement uses a new version rather than changing those assets.

## Prepared delivery contract

The dedicated Foundation bundle lane keeps its explicit Windows ARM64 Cargo
target and selects `+crt-static` through
`CARGO_TARGET_AARCH64_PC_WINDOWS_MSVC_RUSTFLAGS`. Host build tools and procedural
macros retain their ordinary host linkage. Its five executables still come from
the existing target-triple `release` directory; no compiler/runtime API changes
or inference optimizations are introduced.

This statically links Microsoft's runtime; it is not a pure Rust CRT/EH
replacement. Existing panic unwinding remains enabled. The lane runs only two
named model-free Rust controls under that linkage: the tiled-worker panic
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

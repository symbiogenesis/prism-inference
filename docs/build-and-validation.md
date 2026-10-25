# Building and validation

[Documentation index](index.md) · [CLI and workers](cli-and-workers.md)

## Install from source

Install the published source package with the pinned nightly:

```powershell
cargo +nightly-2026-10-22 install prism-inference --version 0.1.1 --locked
prism-inference --help
```

`cargo install` compiles source. Consumers using a prebuilt binary bundle should
pin its version, target and SHA-256 rather than rebuild the runtime on startup.
Library consumers can retain the existing import path with this dependency:

```toml
bonsai-prism-runtime = { package = "prism-inference", version = "=0.1.1" }
```

## Cargo features

Default features are `cli` (JSON token driver) and `experimental-neon`.

| Cargo feature selection | Result |
| --- | --- |
| Defaults | Token driver/library with eligible NEON selection |
| `--no-default-features --features experimental-neon --lib` | NEON-enabled library without JSON driver |
| `--no-default-features --lib` | Scalar-only library |
| `--no-default-features --features cli` | Scalar-only token driver |

## Build a checkout

Use the pinned Rust nightly-2026-10-22 and an absolute manifest/target path from
a unique system-temp working directory. After fetching locked dependencies:

```powershell
$repo = 'C:\Users\symbi\source\repos\prism-inference'
$lane = Join-Path ([IO.Path]::GetTempPath()) ('PrismInference-' + [guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($lane) | Out-Null
$previous = [Environment]::CurrentDirectory
Push-Location $lane
try {
    [Environment]::CurrentDirectory = $lane
    cargo +nightly-2026-10-22-aarch64-pc-windows-msvc build --workspace --bins --locked --offline --release `
        --manifest-path (Join-Path $repo 'Cargo.toml') `
        --target-dir (Join-Path $lane 'build')
    if ($LASTEXITCODE) { throw 'Prism Inference build failed.' }
    & (Join-Path $lane 'build/release/prism-inference.exe') --help
} finally {
    [Environment]::CurrentDirectory = $previous
    Pop-Location
}
```

## Correctness checks

The current [Foundation workflow](../.github/workflows/foundation.yml) runs
complete workspace correctness tests, formatting, Release Clippy/rustdoc,
feature variants, loader-ABI Clippy/rustdoc and scalar fallback checks on Windows
ARM64 and x64. These workspace checks run once per target; the manual compiler
workflow does not repeat them.
Its Linux ARM64 gate separately builds the pinned Prism engine with the Rust
storage overlay and checks native linking and rejection behavior. Fixtures cover
bounded loading, exact codecs/rounding, graph state, output reuse, checkpoint
isolation and transactional recovery. They do not load the full model. Locally
run focused named regressions, from the same external working directory
convention, for example:

```powershell
cargo +nightly-2026-10-22-aarch64-pc-windows-msvc test --locked --offline --release `
    --manifest-path <repo>/Cargo.toml --target-dir <temp>/build `
    --test runtime edited_turns_restore_complete_recurrent_prefix_and_failed_stream_rolls_back -- --exact
```

The [R0–R7 roadmap](https://github.com/symbiogenesis/SwiftBonsai2/blob/370084c530e2271f10a1c55a5a10bba7c806aacb/docs/native-rust-engine.md) still requires profiling,
full-model logits/states/tokens, complete tokenizer/template and Prism sampler/RNG parity,
adapter execution, broader process/OS ownership qualification, SIMD/Adreno support and device
correctness/TPS/quality/security acceptance. Full suites and every benchmark
remain CI-only under [validation policy](https://github.com/symbiogenesis/SwiftBonsai2/blob/370084c530e2271f10a1c55a5a10bba7c806aacb/docs/validation.md).

This repository also owns the [complete upstream conformance guide](prism-upstream-tests.md).
Its [runtime workflow](../.github/workflows/runtime-conformance.yml) preserves
four ordinary source partitions, 16 planning partitions and 32 execution partitions,
plus native codec parity and a fail-closed aggregate. Measurements and gradients
remain explicit manual opt-ins. The [focused workflow](../.github/workflows/prism-focused.yml)
diagnoses one named source without replacing the complete gate.

The [standards workflow](../.github/workflows/standards.yml) owns PowerShell and
embedded-C# analysis for the migrated orchestration. The manual
[compiler acceptance workflow](../.github/workflows/rust-nightly.yml) compares
isolated stable/nightly language probes and preserves the fail-closed
`compatibility` check. Dispatch Foundation to rerun workspace validation. Runner
contains only app/provisioning/released-runtime integration gates; none of these
runtime suites bootstrap through its maintenance executable or released source.
All full suites and every benchmark remain hosted-only.

Automatic workflows skip pushes limited to root Markdown files and `docs/`;
manual dispatch remains available. Test fixtures and package READMEs outside
those paths still trigger validation.

## Bounded actual-model correctness captures

Build the `swift_model_check` example explicitly from the external working lane.
Its `forward` mode retains the existing 1–16 supplied-token captures. For a
verified chat prefix, `trajectory` reads an absolute JSON token-ID file and
reserves all requested 1–16 greedy steps within the 8,192-token context:

```text
swift_model_check trajectory <absolute-model> <existing-absolute-evidence-dir> <scalar|auto|neon> <absolute-prompt-ids.json> <greedy-steps>
```

The qualifier ingests intermediate prompt tokens without logits, then captures
logits and complete logical hybrid state for the final prefix and every
fed-back greedy token. It reports exact prompt/generated IDs and includes
special tokens in this fixed-step diagnostic. Service stop and sampler
qualification remain separate. Snapshots never overwrite files; their combined
payload is capped at 8 GiB. Retain absolute external outputs and the existing
process-memory/deadline supervision. No timings or throughput are measured.
The evidence-write guards run in the ordinary hosted correctness suite.

## Release bundles and compatibility

The Windows jobs build five executables and package them with the four root
license/notice files. Each bundle has a receipt identifying its exact source
revision, target, version, file hashes and archive hash. The aggregate Foundation
check requires both Windows targets and the Linux native overlay gate.

The checkout prepares version `0.1.2`; the published crates.io installation
examples above remain at `0.1.1`. Packaging rejects a changed or untracked source
checkout, a revision mismatch, inconsistent runtime/server versions or a
toolchain string that differs from the pinned channel and target. The receipt
records the resulting bytes; its source-to-binary trust still depends on the
checked Foundation build. The synthetic release-bundle regressions verify
identity rejection, PE target checks and every archive entry's recorded hash.

The main CLI, token worker and Swift worker come from the `prism-inference`
package. `prism-inference-server` and `bonsai-gguf` are separate packages in
the workspace; `cargo install prism-inference` alone installs the first three
executables. The [release bundles](https://github.com/symbiogenesis/prism-inference/releases)
include all five.

The worker executable names, Swift readiness identity `bonsai-prism-runtime`
and library import `bonsai_prism_runtime` preserve the existing SwiftBonsai
protocol/API. Their names do not restrict the repository's broader model scope.

Complete suites and every benchmark run only in CI. Local builds, formatting,
analyzers and focused named regressions are allowed. Avoid approximating a full
suite by batching filtered local runs. Build outputs, logs and native working
directories belong in unique system-temp lanes, outside checkouts.

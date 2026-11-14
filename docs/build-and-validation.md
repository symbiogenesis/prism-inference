# Building and validation

[Documentation index](index.md) · [CLI and workers](cli-and-workers.md)

## Install from source

Use the compiler pinned in [rust-toolchain.toml](../rust-toolchain.toml).
Its next-generation trait solver is enabled globally by default: there is no
per-trait opt-in. Workspace crates, dependencies, build scripts, procedural macros
and rustdoc use the selected compiler's solver without extra flags. Keep external
builds and consumers on the pinned compiler; `--manifest-path` alone does not
select a toolchain. The [compiler gates](../tools/rust-trial/tests/compiler.rs)
and `SupportsLogitsView` doctest check the default behavior. Stable and
coherence-only solver selections belong to the isolated
[compiler comparison controls](../tools/rust-trial/README.md).

Install the CLI and two JSONL workers from a checkout:

```powershell
cargo +nightly-2026-11-09 install --path <absolute-repo> --locked
prism-inference --help
```

The private server and GGUF tool are separate workspace packages. To build all
five executables, use [Build a checkout](#build-a-checkout). Prebuilt consumers
should verify source, target and file hashes from the bundle receipt.

## Use the Rust library

The package name is `prism-inference`; its Rust import path is
`bonsai_prism_runtime`. For a sibling source checkout:

```toml
[dependencies]
bonsai-prism-runtime = { package = "prism-inference", path = "../prism-inference", default-features = false }
```

Pin the same compiler and retain the consuming application's `Cargo.lock`.
Enable `experimental-neon` for eligible Windows ARM64 kernels. JSON chat and
`swift_session::SwiftSession` require `cli`.

This example verifies the original-PQ2 artifact and reuses its tokenizer without
loading weights or constructing a graph:

```rust
use std::path::Path;

use bonsai_prism_runtime::{
    Result,
    model::swift::{MODEL_BYTES, MODEL_SHA256},
    special_input::SpecialInputPolicy,
    tokenization::{
        SpecialTokenInsertion, TextTokenizationPolicy, TokenBudget, TokenizationSession,
    },
};

fn prompt_ids(model: &Path, text: &str) -> Result<Vec<u32>> {
    let tokenizer = TokenizationSession::load(
        model,
        MODEL_BYTES,
        MODEL_SHA256,
        256 * 1024 * 1024,
    )?;
    let policy = TextTokenizationPolicy {
        special_input: SpecialInputPolicy::Literal,
        insert_special: SpecialTokenInsertion::None,
        budget: TokenBudget { context: 8192, reserved_output_tokens: 512 },
    };
    Ok(tokenizer.encode(text, policy)?.token_ids)
}
```

Pass an absolute model path. Verification reads the entire file; the metadata
budget does not bound whole-process memory. Retain `TokenizationSession` for
repeated prompts. This API applies no chat template.

For generation, `SwiftSession::load` verifies the model, `prepare` renders and
reserves a complete chat request, and `complete` takes an absolute deadline,
shared cancellation and a fallible text callback. See
[session profiles](swift-session-profiles.md) for model/adapter selection and
[runtime contracts](runtime-contracts.md) for ownership and rollback.

## Cargo features

| Selection | Result |
| --- | --- |
| Defaults: `cli`, `experimental-neon` | JSON driver and eligible Windows ARM64 NEON |
| `--features parallel-rows` | Persistent packed-row pool; worker count still defaults to one |
| `--no-default-features --features experimental-neon --lib` | NEON library without JSON APIs |
| `--no-default-features --lib` | Scalar library |
| `--no-default-features --features cli` | Scalar JSON driver |

See [row workers](parallel-rows.md) for execution budgets and configuration.
[Cargo.toml](../Cargo.toml) is authoritative for package versions and dependencies.
[Native hashing](../tools/native-hash/README.md) documents artifact verification.

## Build a checkout

Run builds and checks from a unique system-temp directory, with absolute
manifest and target paths. Fetch locked dependencies before using `--offline`.

```powershell
$repo = 'C:\Users\symbi\source\repos\prism-inference'
$lane = Join-Path ([IO.Path]::GetTempPath()) ('PrismInference-' + [guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($lane) | Out-Null
$previous = [Environment]::CurrentDirectory
Push-Location $lane
try {
    [Environment]::CurrentDirectory = $lane
    cargo +nightly-2026-11-09-aarch64-pc-windows-msvc build --workspace --bins --locked --offline --release `
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

Foundation runs formatting, strict Release Clippy/rustdoc and selected workspace
correctness tests on Windows ARM64 and x64, plus applicable native-overlay,
codec and source-package checks. Synthetic fixtures cover loading, arithmetic,
state, checkpoints and recovery; they do not establish full-model parity.

Full suites and benchmarks run only in CI. Local checks should name the affected
regression, for example:

```powershell
cargo +nightly-2026-11-09-aarch64-pc-windows-msvc test --locked --offline --release `
    --manifest-path <repo>/Cargo.toml --target-dir <temp>/build `
    --test runtime edited_turns_restore_complete_recurrent_prefix_and_failed_stream_rolls_back -- --exact
```

### Owner-device budgets

Follow [AGENTS.md](../AGENTS.md#owner-device-execution-budgets) for aggregate
allowances and [CI ownership](../AGENTS.md#ci-ownership-and-completion).
Record AC power before and after each workload and each comparison arm, with
battery saver and charge when available. Unknown or mismatched power conditions
cannot qualify performance comparisons; equal endpoints do not prove stability
throughout a run. Existing narrower tool limits remain binding.

Prepare source-bound native capture helpers before the device allowance starts.
Retain only replay inputs, identities and actionable diagnostics required by the
gate; routine source checks need commands, results and limits in the delivery
message rather than a new evidence bundle.

### Owning workflows

The [impact map](../.github/ci/impact.json) combines changed paths with Cargo
dependencies. Shared or unclassified inputs select full validation. Planning
uses the latest successful ancestor; unavailable history falls back to full
selection. Aggregate gates require every selected job to succeed.

Foundation skips changes limited to root Markdown and `docs/`; Source checks
still validates Markdown. Manual dispatch selects the workflow's complete scope.
Caches store compiler outputs or verified immutable inputs, never verdicts.
Every selected check and generated receipt runs afresh.

### Workflow inventory

| Workflow | Responsibility |
| --- | --- |
| [Foundation](../.github/workflows/foundation.yml) | Selected workspace, cross-target, overlay, regression, codec and source-package checks |
| [Source checks](../.github/workflows/source-checks.yml) | Workflow preflight, Markdown, scripts, compiler probes and model-free parity |
| [Foundation bundle](../.github/workflows/foundation-bundle.yml) | Exact-source hosted ARM64 bundle preparation |
| [Runtime conformance](../.github/workflows/runtime-conformance.yml) | Complete upstream source/reference and backend gate |
| [Focused reference](../.github/workflows/prism-focused.yml) | One named source contract or backend plan |
| [Private-service acceptance](../.github/workflows/private-service-acceptance.yml) | Source-bound service lifecycle and receipt checks |
| [Request timing preparation](../.github/workflows/request-timing-preparation.yml) | Standalone actual-model diagnostic tool preparation |

Choose `model_free: efficiency` in Source checks for the
[exact efficiency controls](inference-efficiency.md); its default parity matrix
does not substitute for them. Neither matrix loads the full model.
See [runtime conformance](prism-upstream-tests.md) for complete reference coverage
and [server acceptance](../server/README.md) for the private-service protocol.

### Focused checks before local delivery

Before committing a workflow or composite action, run:

```powershell
pwsh -NoLogo -NoProfile -File .github/ci/Test-Workflow.ps1 -CheckFixtures
```

This compiler-free preflight runs checksum-pinned actionlint, validates action
references and local inputs, and exercises rejection and PowerShell exit-status
controls. Use `-ActionlintPath <absolute-path>` for a trusted offline binary.
A local preflight supplies no hosted verdict.

For cfg-sensitive Rust changes, follow
[the cross-target policy](../AGENTS.md#cross-target-lint-checks).
A dependency build does not lint a helper's tests. Select changed packages
explicitly with `--all-targets`, covering native and fallback branches:

```powershell
cargo +nightly-2026-11-09-aarch64-pc-windows-msvc clippy --locked --offline --release `
    --manifest-path <repo>/Cargo.toml --target-dir <temp>/build `
    --target aarch64-pc-windows-msvc -p bonsai-neon-dot -p bonsai-hexagon-native `
    -p bonsai-hexagon-runtime --all-targets -- -D warnings
cargo +nightly-2026-11-09-aarch64-pc-windows-msvc clippy --locked --offline --release `
    --manifest-path <repo>/Cargo.toml --target-dir <temp>/build `
    --target x86_64-pc-windows-msvc -p bonsai-neon-dot -p bonsai-hexagon-native `
    -p bonsai-hexagon-runtime --all-targets -- -D warnings
```

Cross-target Clippy checks types without executing the other architecture.
Report unavailable targets and pending native hosted checks.

## Bounded actual-model correctness captures

Build `swift_model_check` explicitly before a device capture. Use absolute
external paths and process/Job/deadline supervision. The
[checker](../examples/swift_model_check.rs) enforces artifact identities and
capture bounds; a successful source build alone qualifies no model behavior.

| Mode | Scope |
| --- | --- |
| `forward` | Logits and complete logical state for 1–16 supplied token IDs |
| `trajectory` | Final prompt state and 1–16 fed-back greedy steps within 8,192 tokens; 8 GiB snapshot cap |
| `rollback` | Committed state, cancelled provisional step, exact restoration and fresh one-token continuation; 2 GiB snapshot cap |
| `request-timing` | One production SwiftSession request, at most 128 rendered prompt tokens and 1–16 output tokens |

```text
swift_model_check trajectory <absolute-model> <existing-absolute-evidence-dir> <scalar|auto|neon> <absolute-prompt-ids.json> <greedy-steps>
swift_model_check --row-workers 4 --rank-one-adapter <absolute-approved-adapter> 1.3 rollback <absolute-original-PQ2> <new-existing-absolute-evidence-dir> auto 248045
```

Rollback must return `Interrupted`, preserve committed token IDs and restore
every captured logit/state bit before reset. Its stop-token probe inspects
existing logits without advancing the graph. Greedy captures include special
IDs and qualify neither production stop behavior nor sampling. Cancellation
occurs after a model step, not inside a kernel.

These numerical captures retain their five-minute aggregate owner allowance;
the general fifteen-minute ceiling does not extend them. Include setup,
loading, comparison, power observations, cleanup and retention in that clock.

### Model and adapter selection

Prefix a command with `--use-repacked-ptq1` to select the exact current-weight
PTQ1 export. [swift.rs](../src/model/swift.rs) defines its accepted size, hash and
inventory. The prefix applies to `inspect`, `load`, `forward`, `trajectory`,
`rollback` and `request-timing`; the ordinary loader selects original PQ2.

Compare PTQ1 against the same file in the pinned native engine using frozen
same-format tolerances and matched tokens, context, transforms and state types.
Scalar/NEON self-comparison is insufficient. PQ2 and PTQ1 use different activation
quantization, so cross-representation quality is a separate decision.

The optional `--rank-one-adapter <absolute-approved-adapter> <scale>` follows
the representation prefix. See [rank-one adapters](rank-one-adapter.md) for
verification, residual-writer basis and native numerical controls, and
[session profiles](swift-session-profiles.md) for private-service selection.
Loading an adapter or changing generated text does not qualify its arithmetic.

### Request timing

The [preparation workflow](../.github/workflows/request-timing-preparation.yml)
checks model-free controls and produces a Windows ARM64 checker with
`parallel-rows` and a source/run/compiler/file-hash receipt. It executes no model.

```text
swift_model_check --use-repacked-ptq1 --row-workers 4 --rank-one-adapter <absolute-approved-adapter> 1.3 request-timing <absolute-current-PTQ1> <new-existing-evidence-dir> auto <absolute-messages.json> 1 <original-work-deadline-unix-ms> <exact-hosted-source-commit>
```

The absolute deadline must leave 1–240 seconds at entry and maps once to a
monotonic clock before intake, hashing and loading. The external owner binds
the executable to its hosted receipt, supervises blocking load, and reserves
receipt writing and teardown inside the five-minute aggregate allowance.

The request uses SwiftSampler, seed 42, native stops, thinking disabled and no
prefix reuse. Create-new `request-timing-started.json` and `request-timing.json`
record input/executable hashes, selected profile and completed phase timings,
including returned failures. A missing final receipt leaves completion unknown.
They omit prompt/output text, token IDs, paths and raw errors.

Provisional callbacks differ from committed output, sampled stops and text
chunks. The no-op consumer measures neither HTTP nor backpressure. These
timings attribute session work; they do not qualify app throughput, model parity
or service lifecycle. For a controlled multi-arm experiment, use the
[CPU bundle comparison protocol](cpu-bundle-comparison.md).

## Binary bundles

A bundle contains `prism-inference`, `bonsai-token-worker`,
`bonsai-swift-worker`, `prism-inference-server` and `bonsai-gguf`, plus root
licenses and notices. The root package supplies the first three executables;
the server and GGUF tool are separate packages.

Packaging rejects dirty source, revision/target mismatches, inconsistent
runtime/server versions and an unexpected compiler. Receipts bind source,
target, features, executable hashes and archive bytes. PE checks enforce
architecture and static Microsoft runtime linkage. Source-to-binary trust
still depends on the reviewed hosted build.

The [Foundation bundle workflow](../.github/workflows/foundation-bundle.yml)
requires the exact main revision and a unique 1–48-character operation ID:
letters, digits, underscores or hyphens, starting with a letter or digit.
It runs packaging controls, strict analyzers, scalar compilation and named
unwind/context regressions before linking the five executables. Its
`bundle-scope.json` records that complete Foundation, model acceptance and
performance remain unqualified. Follow [CI ownership](../AGENTS.md#ci-ownership-and-completion)
for authorization and hosted execution limits.

Foundation's independent source-package matrix verifies normalized archives and
external scalar/default/parallel-row consumers on Windows ARM64, Windows x64 and
Linux ARM64. Archive hashes, source identity, consumer locks and commands are
retained in its receipt. Registry publication requires dependency helpers to be
available first; package and binary verification remain separate gates.

The server is not registry-publishable while it depends on unpublished Hexagon
helpers. Use a verified bundle or exact source checkout. Worker names, readiness
identity `bonsai-prism-runtime` and library import `bonsai_prism_runtime` are
the current protocol identifiers.

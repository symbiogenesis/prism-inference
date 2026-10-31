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
| `--features parallel-rows` | Opt-in persistent packed-row pool; trusted worker count still defaults to one |
| `--no-default-features --features experimental-neon --lib` | NEON-enabled library without JSON driver |
| `--no-default-features --lib` | Scalar-only library |
| `--no-default-features --features cli` | Scalar-only token driver |

See [explicit row workers](parallel-rows.md) for worker budgets, unchanged row
arithmetic, CLI/service configuration and qualification limits.

GGUF artifact verification uses the safe streaming API from
[`bonsai-native-hash`](../tools/native-hash/README.md). Windows delegates SHA-256
CPU dispatch to its CNG provider through a private owned-handle boundary; other
targets retain the portable Rust baseline. Every model byte is still checked
through the retained handle before metadata parsing. The GGUF crate remains
unsafe-free, and full-file reads, load progress and cancellation contracts are
unchanged. This implementation alone establishes no startup timing improvement.
Publish the new hash dependency before a registry GGUF release that requires it.

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
formatting, Release Clippy/rustdoc and complete correctness tests for affected
workspace crates and their transitive consumers on Windows ARM64 and x64.
Each Windows target runs analyzers, feature variants and four correctness shards
concurrently. Cargo metadata assigns every enabled integration-test target to
exactly one shard; same-named targets share a shard, while unit tests, doctests,
test-enabled examples and default binary/release packaging run on shard zero.
Every shard retains its full target inventory and executed command list. The
`windows` matrix remains required by the fail-closed Foundation aggregate.
Every Cargo test command keeps the complete selected package set to preserve
dependency feature unification. The unit-test invocation also selects the three
empty `test = false` library roots; dormant feature-gated native targets remain
disabled. Unsupported dependency activation of a gated target fails selection.
Compiler caches are isolated by architecture and lane; locked registry downloads
have their own shared cache, avoiding a complete workspace cache in every shard.
Feature variants, loader-ABI checks and mutation guards run only when their
packages or inputs are selected. Its Linux ARM64 gate builds the pinned Prism
engine with the Rust storage overlay only when that overlay's inputs change.
Fixtures cover
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
adapter numerical/service qualification, broader process/OS ownership qualification, SIMD/Adreno support and device
correctness/TPS/quality/security acceptance. Full suites and every benchmark
remain CI-only under [validation policy](https://github.com/symbiogenesis/SwiftBonsai2/blob/370084c530e2271f10a1c55a5a10bba7c806aacb/docs/validation.md).

This repository also owns the [complete upstream conformance guide](prism-upstream-tests.md).
Its [runtime workflow](../.github/workflows/runtime-conformance.yml) preserves
four ordinary source partitions, 16 planning partitions and 32 execution partitions,
plus native codec parity and a fail-closed aggregate. Measurements and gradients
remain explicit manual opt-ins. The [focused workflow](../.github/workflows/prism-focused.yml)
diagnoses one named source without replacing the complete gate.

The [standards workflow](../.github/workflows/standards.yml) owns PowerShell and
embedded-C# analysis; Rust-only pushes do not start it. The
[compiler acceptance workflow](../.github/workflows/rust-nightly.yml) compares
isolated stable/nightly language probes and preserves the fail-closed
`compatibility` check. It runs when probe inputs change or on manual dispatch.
Dispatch Foundation to rerun complete workspace validation. Runner
contains only app/provisioning/released-runtime integration gates; none of these
runtime suites bootstrap through its maintenance executable or released source.
All full suites and every benchmark remain hosted-only.

Automatic workflows skip pushes limited to root Markdown files and `docs/`.
For other pushes, the [impact map](../.github/ci/impact.json) combines changed
paths with Cargo's local dependency graph. A server-only change checks the
server; a ternary-codec change also checks its GGUF/runtime/server consumers and
the native codec lane. Runtime changes select upstream conformance; unrelated
server or policy-tool changes do not start its backend partitions. Shared
manifests, lockfiles, toolchains, CI policy and unclassified inputs select full
validation. Manual dispatch always selects every group in that workflow.
Known unrelated server, compiler-trial, policy-tool, packaging and C# analyzer
paths also bypass the conformance workflow at its push trigger. The native
codec lane builds its probe and checks parity; Foundation owns its Rust analyzers.

Each workflow compares with its most recent successful push revision on the
current ancestry, so changes from failed or cancelled pushes remain covered.
If that baseline cannot be established, selection falls back to full validation.
Automatic Foundation, standards and compiler-probe runs supersede older pushes
on the same branch, releasing runners for the newest revision. Successful-ancestor
selection carries changes from cancelled and failed pushes into Foundation's
next plan. Manual dispatches retain independent complete validation requests.
Aggregate gates require successful planning
and successful selected jobs; they permit skips only for unselected groups.

## Checkpoint restore storage

Checkpoint restoration stages every fallible KV page-table and mutable-tail copy
before changing live state. Convolution and recurrent matrices then restore their
exact binary32 bits into existing live allocations. Independently owned snapshots
and sealed KV-page sharing retain their previous contracts. For Swift, this avoids
another 156,893,184 bytes (149.625 MiB) of transient recurrent/convolution payload
during restore. It is a storage-lifetime bound, not a measured decrease in process
peak RAM or admission requirements. Focused ownership, staging-failure and frozen
hybrid-state recovery regressions cover the change.
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

PQ2 without an adapter remains the default service profile.
For the verified lossless export of those same current weights, prefix the
offline checker command with `--use-repacked-ptq1`:

```text
swift_model_check --use-repacked-ptq1 forward <absolute-current-repacked-PTQ1-model> <existing-absolute-evidence-dir> <scalar|auto|neon> <id,id,...>
```

The prefix also applies to `inspect`, `load` and `trajectory`. It admits only
the current-weight export with SHA-256
`e658f0f0736a03a4549535a439f9b07254ebf8ab88ae27cbe6e52d098ca5d1f5`,
5,946,648,928 complete file bytes and the strict 851-tensor PTQ1/F32/BF16
inventory. The older publisher PTQ1 checkpoint is not admitted. Receipts include
the selected representation and its actual artifact identity. The existing
`swift::load` and `swift::inspect` continue to select PQ2; the explicit
`load_representation`/`inspect_representation` APIs serve this correctness lane.
The private service separately admits explicit PTQ1/approved-adapter selections
through the [verified session profile](swift-session-profiles.md); selection is
not full-model or device qualification.

Compare PTQ1 snapshots against the same verified PTQ1 file executed by the pinned
publisher Prism CPU implementation, using the existing declared full-model
numerical tolerances. Scalar/NEON self-comparison alone is insufficient. Keep
the exact supplied token IDs, context, KV/state types, transforms and per-token
logical state extents matched. Equal PQ2/PTQ1 decoded weights do not establish
equal logits: PQ2 uses Q8_K activations on complete rows, while PTQ1 uses Q8_0.
A comparison between representations therefore has a separate quality decision;
do not loosen the trusted same-format correctness tolerance to make it pass.

The source graph implements an explicit [rank-one adapter correctness path](rank-one-adapter.md);
the offline checker accepts `--rank-one-adapter <absolute-approved-adapter> <scale>`
after the optional `--use-repacked-ptq1` prefix for `load`, `forward` and
`trajectory`. Both base selection and approved adapter verification remain exact;
the existing PQ2 constructor defaults do not admit the export implicitly.
The JSONL workers remain adapter-free. The explicit private service requires the
exact approved adapter and pinned-slot request scale. An approved LoRA's
load-at-zero logits identity, nonzero correction and Swift grouped-SSM input
ordering must be verified at native residual writers before adapter qualification.
Neither a successful PTQ1 load nor changed generated text settles the adapter's
authoring basis. These correctness captures contain no benchmarks and cannot
substitute for the separate target-device quality/performance gates.

The bounded 2026-10-26 capture at runtime revision `934fb13` is **not qualified**.
It used the exact current-weight PTQ1 export, pinned publisher runtime
`73bfe6cc`, supplied token IDs `[248045, 872, 198]`, an 8K context, FP16 KV,
F32 recurrent state and the existing `1e-3 + 1e-3 * abs(reference)` tolerance:

| Comparison | Bounded result |
| --- | --- |
| Rust scalar PTQ1 against native PTQ1 | 2,691,842 of 118,611,456 logits/state values exceed tolerance across 483 files; maximum absolute error 0.19140625; no file is byte-identical. |
| Native PTQ1 against compatible prior native PQ2 | 593,835 of 744,960 logits exceed the same bound; maximum absolute error 0.10386914014816284. Prior report controls and each reused logits hash were verified. |

Both comparisons retain the greedy IDs `[198, 198, 220]` for these three supplied
inputs. This small observation establishes neither output-quality equivalence
nor successful same-format numerical parity. The PTQ1 scalar/native mismatch
must be diagnosed without relaxing the frozen tolerance. The comparison between
representations remains a distinct question because activation arithmetic differs.
NEON, adapter correctness, longer contexts and quality/performance qualification
were not run in this quick local check. Successful native loading and the exact
weight repack remain separate, narrower results. No default, service selection
or production memory requirement was changed by this diagnostic evidence.

### PTQ1 arithmetic repair, 2026-10-26

The earlier capture exposed three implementation differences, repaired in the
code through `e2d8d6f`:

- The graph selected pinned arithmetic only for the PQ2 hash. The exact verified
  current-weight PTQ1 hash now selects the same pinned graph contract, including
  normalization, activations, attention and recurrent-state arithmetic.
- Pinned Q8_0 activation preparation now rounds binary32 products to nearest,
  ties to even, matching the ARM64 quantizer. Public reference encoding retains
  ties away from zero. Six independent native Q8_0/Q8_1 captures match completely.
- Pinned PTQ1 scalar and baseline NEON kernels now preserve the native DOTPROD
  binary's four partial integer lanes, per-group scale multiplication, fused
  accumulation across the row and final pairwise reduction. Forty independent
  native rows matched this order; the prior implementation differed in thirty.
  The checked-in fixtures distinguish these orders without loading a model.

The native dot captures use `nrc=1`; the type trait's two-row capability does
not establish 2x2-tile parity. Public source-order kernels and the PQ2 fallback
keep their prior behavior. Focused graph selection/recovery and native fixture
regressions accompany the repair; full suites remain hosted.

A single-token actual-model recheck was attempted with the existing supervisor,
16 GiB admission, 12 GiB committed limit and 4 GiB physical reserve. Admission
was rejected before the model process started: minimum available physical memory
was 14,668,292,096 bytes. The guard was preserved. Therefore these operator and
dispatch repairs do **not** establish new full-model parity or supersede the
retained failed capture. No new model snapshot, adapter/quality qualification,
performance measurement, service enablement, default or memory-policy change
is claimed. The prepared one-token check can reuse the existing pinned native
reference when the unchanged admission requirements are available.

### Conditional PTQ1 admission trial

The owner subsequently authorized a provisional reduction of 1,259,520,000 bytes
from the 16 GiB admission requirement, conditional on successful checks. At
15,920,349,184 bytes (about 14.827 GiB), the corrected automatic backend completed
one supplied token `[248045]` with the unchanged 12 GiB Job limit and 4 GiB
physical reserve. The checker was built from `45fdd85`; later repair commits
only changed a test assertion and fixture metadata. No further production
arithmetic changed between that build and `e2d8d6f`.

Against the retained pinned native PTQ1 reference, 430,526 of 39,504,384 complete
logits/live-state values exceeded `1e-3 + 1e-3 * abs(reference)` across 129 files.
Maximum absolute error was 0.06584501266479492. The greedy ID was 198 in both
implementations, and 28 state files were byte-identical. The first nonexact
stored state belongs to zero-based block 14; the first stored state outside
tolerance belongs to block 15. This localizes the next investigation boundary,
not the exact first divergent graph operation.

The resource checks passed, but numerical parity did not. The conditional
admission reduction was therefore reverted, with production admission retained
at 16 GiB. The provisional policy patch, snapshots, comparison hashes and failure
receipt remain outside Git in the admission-trial evidence lane. Runner's
[repair receipt](https://github.com/symbiogenesis/SwiftBonsai2Runner/blob/main/docs/ptq1-arithmetic-evidence.json)
records the bounded result. No benchmark, quality/adapter qualification, default
change or additional model capture followed this failure.

### Bounded floating-projection follow-up

Native operator captures identified a separate pinned F32/F16 projection
accumulation gap. Rows divisible by sixteen now use the native four-register
binary32 FMA reduction. Twelve checked native rows cover F32, F16 and BF16 at
widths 16, 32, 128 and 5120. Three controls distinguish the repair from the
previous generic binary64 sum. Generic arithmetic, BF16 and unqualified tail
widths retain their previous behavior. The fixture records the pinned native
runtime/DLL and source identities; it does not establish full-model PTQ1 parity.

A new one-token intermediate capture was rejected before model launch by the
current Runner development budget: 11,230,294,016 available physical bytes versus
15.5 GiB required. The 11.5 GiB Job ceiling and 4 GiB physical reserve remained
unchanged. The earlier failed full-model comparison therefore remains the latest
numerical verdict. No PTQ1 default, service selection or admission reduction is
qualified by these operator fixtures.
### PTQ1 activation preparation repair

A later one-token Rust recheck completed under a separate diagnostic envelope:
12 GiB available admission, 8 GiB Job commit and the same 4 GiB physical reserve.
Peak Job commit was 6,275,588,096 bytes; minimum available physical RAM was
7,574,892,544 bytes. All 129 outputs matched the earlier Rust capture, including
its numerical failure. This is bounded resource evidence, not a service profile.

A native-backed intermediate trace localized the first difference to block 13's
FFN down-projection. Its input and Hadamard transform matched exactly; one Q8_0
activation code differed. The pinned DLL uses amax multiplied by rounded
binary32 1/127 for scale and direct 127/amax division for inverse scale. The
previous division-then-reciprocal sequence rounded a halfway value differently.
The pinned PTQ1 path now reproduces the native sequence and has a distinct
activation-sharing contract. Generic/PQ2 behavior remains unchanged. Four small
synthetic native controls and the entire isolated 17,408-by-5,120 projection
match exactly. A new full-model comparison after this repair remains pending;
production defaults, service selection and admission are unqualified.

The unchanged DSpark real-eval case passed in isolated main CI run 40446869313
but exhausted its native session lifetime in concurrent ordinary partition 3.
That partition now runs adapters serially; the other ordinary partitions retain
their prior concurrency. Source coverage, inputs, assertions and deadlines stay
unchanged. CI on the final combined revision is handed to the owner for monitoring.
## Release bundles and compatibility

Manual Foundation dispatch and shared or packaging-input changes build five
executables and package them with the four root
license/notice files. Each bundle has a receipt identifying its exact source
revision, target, version, file hashes and archive hash. Isolated crate changes
check and build their affected packages without rebuilding a complete bundle.
The aggregate Foundation check requires both Windows targets when selected and
the Linux native overlay gate when its inputs are selected.

The checkout prepares version `0.1.9`; the published crates.io installation
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

The `0.1.9` server is prepared for a GitHub binary prerelease, preserving the
immutable `0.1.7` and `0.1.8` artifacts. Its Windows credential helper is pinned to Runner
revision `abe04b4a8eff18388af96f0e75c8a8d180b81989`; the Git source is retained in
the lockfile. The server temporarily sets `publish = false`, because a crates.io
package cannot preserve that Git-only dependency and substituting registry
`0.1.0` would omit the required read APIs. Checked GitHub bundles and exact source
checkouts are the supported delivery paths for this server candidate. A reviewed
helper registry release and server publication remain separate pending work;
the root runtime library's publication contract is unchanged. Source preparation
and passing CI alone do not publish the candidate or qualify model serving.

The candidate release-bundle build explicitly enables
`prism-inference/parallel-rows` and `prism-inference-server/parallel-rows` in the
retained Foundation build command. The receipt's exact source revision, binary
hashes and archive hash bind that build; its schema is unchanged. Library and
server constructors, including the private HTTP server, still default to one
row worker. Feature availability does not enable multiple workers in Runner or
establish model parity, speed, resource suitability or default-runtime promotion.

The worker executable names, Swift readiness identity `bonsai-prism-runtime`
and library import `bonsai_prism_runtime` preserve the existing SwiftBonsai
protocol/API. Their names do not restrict the repository's broader model scope.

Complete suites and every benchmark run only in CI. Local builds, formatting,
analyzers and focused named regressions are allowed. Avoid approximating a full
suite by batching filtered local runs. Build outputs, logs and native working
directories belong in unique system-temp lanes, outside checkouts.

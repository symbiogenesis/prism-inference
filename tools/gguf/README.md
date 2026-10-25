# GGUF tools

`bonsai-gguf` provides bounded little-endian GGUF v3 inspection, Prism PQ2 census
and experimental weight-lossless PTQ1 export using the shared Rust codecs.
Every operation verifies complete input size/SHA-256. Windows handles deny source
writes/deletion through verification and use. This is not a general validator
for every GGUF extension.

`Metadata`, `MetadataValue` and `MetadataArray` retain native scalar types and
compact homogeneous arrays; JSON conversion occurs only for reports.
`--no-default-features --lib` omits JSON/export tooling; the default `report`
feature includes the CLI, census and export APIs.

Runtime metadata loading shares the same parser and validation, but skips the
inspection report's tokenizer-record hashes, histograms and tensor-name lists.
Header reads use one bounded 64 KiB buffer, including any read-ahead into tensor
data; tensor payloads are not interpreted or materialized by header parsing.
Complete artifact size/SHA-256 verification still precedes parsing.

## PowerShell entry points

Run from the repository root with model/output files outside Git:

```powershell
# Defaults to the Qwen reference artifact identity:
./scripts/Read-Gguf.ps1 -ModelPath C:\Models\pinned.gguf
# Swift requires its separate identity and explicit Prism dialect:
$swift = Get-Content -Raw ./config/swift-assessment.lock.json | ConvertFrom-Json -DateKind String -NoEnumerate
./scripts/Read-Gguf.ps1 -ModelPath C:\Models\swift.gguf -ExpectedBytes $swift.model.bytes -ExpectedSha256 $swift.model.sha256 -Dialect Prism-73bfe6cc485554822f80c473a3c0c4f25e80f41b -CensusPq2
./scripts/Export-SwiftPtq1.ps1 -ModelPath C:\Models\swift.gguf -OutputPath C:\Models\swift-ptq1.gguf -WhatIf
```

Inspection accepts `-ExpectedBytes` and `-ExpectedSha256` together for other
pinned artifacts or fixtures. Export always uses the Swift assessment lock.
Paths must satisfy `Resolve-ExternalPath`; outputs must be new files in existing
external directories. Inspection skips tensor payloads after hashing; census
streams PQ2 payloads in bounded batches. Optional report output is exclusive.

Export supports `-WhatIf` and `-TimeoutSeconds` (1–3600, default 1800). It writes
a unique partial, validates it and publishes with an exclusive hard link; NTFS
or another hard-link-capable filesystem is required. Failures retain partials.
Receipts include the lock's `SourceRevision`; if partial-alias removal fails after
publication they also include `RetainedPartialPath` and `PartialCleanupError`.
The timeout bounds conversion/copy/padding/initial output hash; source verification
and final validation are additional work.

Both wrappers accept `-GgufToolPath C:\Tools\bonsai-gguf.exe`. Otherwise the shared
resolver builds pinned Rust sources with `--locked --release`, using external
working/target/log directories. Cargo, the pinned toolchain and locked dependencies
must be available or downloadable. Each call checks freshness; incremental output
is reused within the PowerShell session. Child environment is inherited and
native diagnostics stay outside the JSON report.

## Direct use

Build from an external native working directory:

```powershell
$manifest = [IO.Path]::GetFullPath('./tools/gguf/Cargo.toml')
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('SwiftBonsai-GgufBuild-' + [guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($scratch)
$previous = [Environment]::CurrentDirectory
Push-Location $scratch
try {
    [Environment]::CurrentDirectory = $scratch
    cargo +nightly-2026-10-22-aarch64-pc-windows-msvc build --locked --release --manifest-path $manifest --target-dir (Join-Path $scratch 'target')
    if ($LASTEXITCODE) { throw 'GGUF build failed.' }
} finally {
    [Environment]::CurrentDirectory = $previous
    Pop-Location
}
```

The executable is `target/release/bonsai-gguf.exe` under that lane:

```text
bonsai-gguf inspect --model PATH --expected-bytes N --expected-sha256 HASH [--dialect Stock|Prism-73bfe6cc485554822f80c473a3c0c4f25e80f41b] [--census-pq2] [--output PATH]
bonsai-gguf export-ptq1 --model PATH --output PATH --expected-bytes N --expected-sha256 HASH [--timeout-seconds N]
```

Success writes one JSON object; failure writes stderr and returns nonzero.
PTQ1 export preserves raw scales, ternary symbols, other tensor payloads and
metadata, and checks byte-identical PQ2 reconstruction. It does not qualify native
PTQ1 activation quantization, logits, quality, speed or an inference profile.
See [validation](../../docs/validation.md) for focused checks and CI coverage.

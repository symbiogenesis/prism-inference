#requires -Version 7.6
<#
.SYNOPSIS
Checks source pinning, deterministic overlay output, and rejection without edits.
.DESCRIPTION
Builds Rust and generates its C declarations, then checks source application.
RustTargetDirectory optionally reuses the engine loader's external Cargo artifacts.
Does not run native tests, load a model, or benchmark.
#>
[CmdletBinding()]
[OutputType([string])]
param(
    [Parameter(Mandatory)][string]$SourceArchive,
    [string]$RustTargetDirectory
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$overlay = Join-Path $repo 'tools/prism/Apply-Overlay.ps1'
$lock = Get-Content -LiteralPath (Join-Path $repo 'tools/prism/source.lock.json') -Raw | ConvertFrom-Json -DateKind String
$result = & (Join-Path $repo 'scripts/Build-PrismEngine.ps1') -SourceArchive $SourceArchive -RustTargetDirectory $RustTargetDirectory -PrepareOnly
$adapter = Join-Path $result.SourceDirectory 'src/llama-adapter.cpp'
$header = Join-Path $result.SourceDirectory 'src/bonsai_prism.h'
$cmake = Join-Path $result.SourceDirectory 'src/CMakeLists.txt'
if ((Get-FileHash -LiteralPath $adapter -Algorithm SHA256).Hash.ToLowerInvariant() -cne $lock.OverlaidAdapterSha256 -or
    (Get-FileHash -LiteralPath $cmake -Algorithm SHA256).Hash.ToLowerInvariant() -cne $lock.OverlaidCMakeSha256 -or
    (Get-FileHash -LiteralPath $header -Algorithm SHA256).Hash.ToLowerInvariant() -cne $result.Rust.HeaderSha256) {
    throw 'Overlay source output is not deterministic.'
}
$rejected = $false
try { $null = & $overlay -SourceDirectory $result.SourceDirectory -GeneratedHeader $result.Rust.GeneratedHeader }
catch { $rejected = $_.Exception.Message -eq 'Pinned llama-adapter.cpp source hash mismatch.' }
if (-not $rejected) { throw 'Repeat application was not rejected.' }
if ((Get-FileHash -LiteralPath $adapter -Algorithm SHA256).Hash.ToLowerInvariant() -cne $lock.OverlaidAdapterSha256 -or
    (Get-FileHash -LiteralPath $cmake -Algorithm SHA256).Hash.ToLowerInvariant() -cne $lock.OverlaidCMakeSha256) { throw 'Repeat rejection changed source.' }
$fixture = Join-Path $result.EvidenceDirectory 'drift-fixture'
[void][IO.Directory]::CreateDirectory((Join-Path $fixture 'src'))
$driftPath = Join-Path $fixture 'src/llama-adapter.cpp'
[IO.File]::WriteAllText($driftPath, '// wrong source revision')
$before = (Get-FileHash -LiteralPath $driftPath -Algorithm SHA256).Hash
$rejected = $false
try { $null = & $overlay -SourceDirectory $fixture -GeneratedHeader $result.Rust.GeneratedHeader }
catch { $rejected = $_.Exception.Message -eq 'Pinned llama-adapter.cpp source hash mismatch.' }
if (-not $rejected -or (Get-FileHash -LiteralPath $driftPath -Algorithm SHA256).Hash -cne $before -or
    [IO.File]::Exists((Join-Path $fixture 'src/bonsai_prism.h'))) { throw 'Source drift rejection changed the fixture.' }
Remove-Item -LiteralPath $driftPath -Force
Write-Output "Pinned Prism overlay source/rejection checks passed. Evidence: $($result.EvidenceDirectory)"

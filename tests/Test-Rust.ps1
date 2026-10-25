#requires -Version 7.6
<#
.SYNOPSIS
Runs registered Rust format, Release analyzer, documentation and optional CI suites.
.DESCRIPTION
Uses the same package/feature registry as Foundation and the nightly trial.
Packages share Cargo invocations by phase; restricted feature profiles remain
separate. Receipts retain package coverage and command durations for each batch.
Targets, command logs and receipts remain external. Complete suites require
main-branch GitHub Actions. No toolchain installation or benchmark is performed.
#>
[CmdletBinding()]
[OutputType([Collections.IDictionary])]
param(
    [string[]]$Package = @(),
    [ValidateSet('All', 'Runtime', 'Broker')][string]$Group = 'All',
    [string]$Toolchain = 'nightly-2026-10-22-aarch64-pc-windows-msvc',
    [Parameter(Mandatory)][string]$TargetDirectory,
    [Parameter(Mandatory)][string]$EvidenceDirectory,
    [ValidateSet('Foundation', 'Compatibility', 'Native')][string]$CheckProfile = 'Foundation',
    [ValidateSet('fmt', 'clippy', 'doc', 'test', 'build')][string[]]$Phase = @('fmt', 'clippy', 'doc'),
    [ValidateRange(1, 3600)][int]$TimeoutSeconds = 900,
    [switch]$FullTests,
    [switch]$Build,
    [switch]$Offline
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot '../scripts/Bonsai.Rust.psm1')
Invoke-BonsaiRustCheck @PSBoundParameters

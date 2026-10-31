#requires -Version 7.6
[CmdletBinding()]
[OutputType([string])]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot 'Intake.psm1') -Force
$root = Join-Path ([IO.Path]::GetTempPath()) ('prism-service-intake-' + [guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($root)
try {
    # Header-only fixtures establish machine/inventory checking; never executed.
    $bytes = [byte[]]::new(80)
    $bytes[0]=0x4d; $bytes[1]=0x5a; $bytes[0x3c]=64
    $bytes[64]=0x50; $bytes[65]=0x45; $bytes[68]=0x64; $bytes[69]=0xaa
    $names = @('private-service-check.exe', 'private-service-owner.exe')
    foreach ($name in $names) { [IO.File]::WriteAllBytes((Join-Path $root $name), $bytes) }
    $expected = @{
        repository='symbiogenesis/prism-inference'; revision=('a' * 40); gitRef='refs/heads/main'
        workflowPath='.github/workflows/private-service-acceptance.yml'; runId='123'; runAttempt='1'
        architecture='ARM64'; scope='harness-preparation-only'; runnerOwnerRevision=('b' * 40); ownerSourceSha256=('c' * 64)
    }
    $receipt = $expected.Clone()
    $receipt.schemaVersion=2; $receipt.runAttempt=1; $receipt.modelExecuted=$false; $receipt.compilerIdentity='fixture'
    $receipt.files = @($names | ForEach-Object {
        @{ name=$_; bytes=$bytes.Length; sha256=(Get-FileHash -LiteralPath (Join-Path $root $_) -Algorithm SHA256).Hash.ToLowerInvariant() }
    })
    $receiptPath = Join-Path $root 'receipt.json'
    function Write-Receipt {
        [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseShouldProcessForStateChangingFunctions', '', Justification = 'Writes only this test-owned temporary fixture.')]
        [CmdletBinding()]
        [OutputType([void])]
        param([Collections.IDictionary]$Value)
        $Value | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding utf8NoBOM
    }
    function Assert-Rejected {
        [CmdletBinding()]
        [OutputType([void])]
        param()
        $opened = $null
        try { $opened = Open-PrivateServicePackage -Package $root -Expected $expected } catch { return }
        finally { if ($null -ne $opened) { foreach ($guard in $opened.Guards) { $guard.Dispose() } } }
        throw 'A malformed package was accepted.'
    }
    Write-Receipt $receipt
    $opened = Open-PrivateServicePackage -Package $root -Expected $expected
    try {
        if ($opened.Guards.Count -ne 3) { throw 'The exact package was not retained.' }
        $refused = $false
        try { [IO.File]::WriteAllBytes((Join-Path $root $names[0]), $bytes) } catch { $refused = $true }
        if (-not $refused) { throw 'Executable guard permitted replacement.' }
    } finally { foreach ($guard in $opened.Guards) { $guard.Dispose() } }
    # Receipt ordering is irrelevant; exact unique names select the binaries.
    $changed = $receipt.Clone(); $changed.files = @($receipt.files[1], $receipt.files[0]); Write-Receipt $changed
    $opened = Open-PrivateServicePackage -Package $root -Expected $expected
    foreach ($guard in $opened.Guards) { $guard.Dispose() }
    foreach ($field in @('schemaVersion', 'repository', 'revision', 'gitRef', 'workflowPath', 'runId', 'runAttempt', 'architecture', 'scope', 'runnerOwnerRevision', 'ownerSourceSha256', 'modelExecuted')) {
        $changed = $receipt.Clone(); $changed[$field]='wrong'; Write-Receipt $changed; Assert-Rejected
    }
    $changed=$receipt.Clone(); $changed.extra='bad'; Write-Receipt $changed; Assert-Rejected
    $changed=$receipt.Clone(); $changed.files=@($receipt.files[0], $receipt.files[0]); Write-Receipt $changed; Assert-Rejected
    $changed=$receipt.Clone(); $changed.files=@($receipt.files[0]); Write-Receipt $changed; Assert-Rejected
    Write-Receipt $receipt
    [IO.File]::WriteAllText((Join-Path $root 'unexpected.txt'), 'extra'); Assert-Rejected
    Remove-Item -LiteralPath (Join-Path $root 'unexpected.txt')
    $bytes[68]=0x64; $bytes[69]=0x86
    [IO.File]::WriteAllBytes((Join-Path $root $names[0]), $bytes)
    Assert-Rejected
    $changed=$receipt.Clone(); $changed.files=@($receipt.files[0].Clone(), $receipt.files[1].Clone())
    $changed.files[0].sha256=(Get-FileHash -LiteralPath (Join-Path $root $names[0]) -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Receipt $changed; Assert-Rejected
    'Passed same-run/source/member binding, reordering, guarded-file, digest and non-ARM64 rejection controls.'
} finally {
    # The target is this test's single freshly created child, verified before deletion.
    $resolved = [IO.Path]::GetFullPath($root)
    $parent = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (-not $resolved.StartsWith($parent, [StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetFileName($resolved) -notmatch '^prism-service-intake-[a-f0-9]{32}$') { throw 'Unsafe fixture cleanup target.' }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}

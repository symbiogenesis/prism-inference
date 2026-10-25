#requires -Version 7.6
<#
.SYNOPSIS
Checks release source identity and bundle integrity with synthetic PE fixtures.
#>
[CmdletBinding()]
[OutputType([string])]
param([ValidateSet('Identity','Target','Archive','All')][string]$Case = 'All')
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
if ($Case -eq 'All' -and $env:GITHUB_ACTIONS -ne 'true') { throw 'The complete suite runs only in CI. Select one focused case locally.' }
Import-Module (Join-Path $PSScriptRoot '../scripts/Bonsai.Native.psm1')
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$scratch = [IO.Path]::GetFullPath((Join-Path ([IO.Path]::GetTempPath()) ('Prism-ReleaseBundle-' + [guid]::NewGuid().ToString('N'))))
if (-not [IO.Path]::IsPathFullyQualified($scratch) -or $scratch -match '%[^%]+%') { throw 'Use an expanded absolute temp directory.' }
[void][IO.Directory]::CreateDirectory($scratch)
$git = (Get-Command git -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
$fixture = Join-Path $scratch 'source'
$build = Join-Path $scratch 'build'
foreach ($directory in @($fixture, $build, (Join-Path $fixture 'scripts'), (Join-Path $fixture 'server'))) {
    [void][IO.Directory]::CreateDirectory($directory)
}
foreach ($name in @('New-ReleaseBundle.ps1','Bonsai.Common.psm1','Bonsai.Native.psm1')) {
    [IO.File]::Copy((Join-Path $repo "scripts/$name"), (Join-Path $fixture "scripts/$name"), $false)
}
foreach ($name in @('LICENSE','NOTICE','PRISM-LICENSE.txt','CAMELID-LICENSE.txt')) {
    [IO.File]::WriteAllText((Join-Path $fixture $name), "Synthetic $name fixture.")
}
[IO.File]::WriteAllText((Join-Path $fixture 'Cargo.toml'), "[package]`nname = `"prism-inference`"`nversion = `"0.1.2`"`n")
[IO.File]::WriteAllText((Join-Path $fixture 'server/Cargo.toml'), "[package]`nname = `"prism-inference-server`"`nversion = `"0.1.2`"`n[dependencies]`nbonsai-prism-runtime = { package = `"prism-inference`", path = `"..`", version = `"=0.1.2`" }`n")
[IO.File]::WriteAllText((Join-Path $fixture 'rust-toolchain.toml'), "[toolchain]`nchannel = `"nightly-2026-10-22`"`n")
foreach ($name in @('prism-inference.exe','bonsai-token-worker.exe','bonsai-swift-worker.exe','prism-inference-server.exe','bonsai-gguf.exe')) {
    $bytes = [byte[]]::new(128)
    [BitConverter]::GetBytes([uint16]0x5a4d).CopyTo($bytes, 0)
    [BitConverter]::GetBytes([uint32]0x40).CopyTo($bytes, 0x3c)
    [BitConverter]::GetBytes([uint32]0x4550).CopyTo($bytes, 0x40)
    [BitConverter]::GetBytes([uint16]0xaa64).CopyTo($bytes, 0x44)
    [IO.File]::WriteAllBytes((Join-Path $build $name), $bytes)
}
function Invoke-FixtureGit {
    [CmdletBinding()]
    [OutputType([string])]
    param([Parameter(Mandatory)][string[]]$ArgumentList)
    $result = Invoke-BonsaiNativeProcess -Executable $git -ArgumentList (@('-C',$fixture) + $ArgumentList) -WorkingDirectory $scratch -TimeoutSeconds 30
    if ($result.ExitCode) { throw "Release fixture Git failed: $($result.StandardError)" }
    return $result.StandardOutput.Trim()
}
function Save-FixtureRevision {
    [CmdletBinding()]
    [OutputType([string])]
    param()
    Invoke-FixtureGit @('add','--all') | Out-Null
    Invoke-FixtureGit @('-c','user.name=Release Test','-c','user.email=release-test@example.invalid','commit','-m','Synthetic packaging fixture') | Out-Null
    return Invoke-FixtureGit @('rev-parse','HEAD')
}
function Assert-BundleRejected {
    [CmdletBinding()]
    [OutputType([void])]
    param([Parameter(Mandatory)][Collections.IDictionary]$Parameters, [Parameter(Mandatory)][string]$ExpectedMessage)
    $caught = $false
    try { & (Join-Path $fixture 'scripts/New-ReleaseBundle.ps1') @Parameters | Out-Null }
    catch {
        if (-not $_.Exception.Message.Contains($ExpectedMessage, [StringComparison]::Ordinal)) { throw }
        $caught = $true
    }
    if (-not $caught) { throw "The release bundle accepted invalid input: $ExpectedMessage" }
    if ([IO.Directory]::Exists($Parameters.OutputDirectory) -and $ExpectedMessage -notlike 'Executable target*') {
        throw 'Source identity rejection must happen before output is created.'
    }
}
Invoke-FixtureGit @('init','--initial-branch=main') | Out-Null
$revision = Save-FixtureRevision
$parameters = @{
    BuildDirectory = $build
    OutputDirectory = Join-Path $scratch 'valid'
    Target = 'aarch64-pc-windows-msvc'
    SourceRevision = $revision
    Version = '0.1.2'
    Toolchain = 'nightly-2026-10-22-aarch64-pc-windows-msvc'
}
if ($Case -in @('Identity','Archive','All')) {
    $receipt = & (Join-Path $fixture 'scripts/New-ReleaseBundle.ps1') @parameters
    if ($receipt.SourceRevision -cne $revision -or $receipt.Version -cne '0.1.2' -or $receipt.Files.Count -ne 9) {
        throw 'The release receipt lost the source, version or complete file inventory.'
    }
}
if ($Case -in @('Identity','All')) {
    $parameters.OutputDirectory = Join-Path $scratch 'wrong-revision'
    $parameters.SourceRevision = '0000000000000000000000000000000000000000'
    Assert-BundleRejected $parameters 'The checked source revision differs'
    $parameters.SourceRevision = $revision
    $parameters.OutputDirectory = Join-Path $scratch 'dirty-source'
    [IO.File]::AppendAllText((Join-Path $fixture 'Cargo.toml'), "# Uncommitted source.`n")
    Assert-BundleRejected $parameters 'Release bundles require a clean source checkout.'
    Invoke-FixtureGit @('restore','--','Cargo.toml') | Out-Null
    $parameters.OutputDirectory = Join-Path $scratch 'untracked-source'
    [IO.File]::WriteAllText((Join-Path $fixture 'extra.rs'), '// Untracked source.')
    Assert-BundleRejected $parameters 'Release bundles require a clean source checkout.'
    $parameters.SourceRevision = Save-FixtureRevision
    $parameters.OutputDirectory = Join-Path $scratch 'wrong-version'
    $parameters.Version = '0.1.3'
    Assert-BundleRejected $parameters 'The package version differs'
    $parameters.Version = '0.1.2'
    $serverManifest = Join-Path $fixture 'server/Cargo.toml'
    $originalServer = [IO.File]::ReadAllText($serverManifest)
    [IO.File]::WriteAllText($serverManifest, $originalServer.Replace('version = "0.1.2"', 'version = "0.1.3"'))
    $parameters.SourceRevision = Save-FixtureRevision
    $parameters.OutputDirectory = Join-Path $scratch 'wrong-server-version'
    Assert-BundleRejected $parameters 'The package version differs'
    [IO.File]::WriteAllText($serverManifest, $originalServer.Replace('version = "=0.1.2"', 'version = "=0.1.3"'))
    $parameters.SourceRevision = Save-FixtureRevision
    $parameters.OutputDirectory = Join-Path $scratch 'wrong-dependency'
    Assert-BundleRejected $parameters 'The server must pin'
    [IO.File]::WriteAllText($serverManifest, $originalServer)
    $parameters.SourceRevision = Save-FixtureRevision
    $parameters.OutputDirectory = Join-Path $scratch 'wrong-toolchain'
    $parameters.Toolchain = 'nightly-2026-10-21-aarch64-pc-windows-msvc'
    Assert-BundleRejected $parameters 'The release toolchain must match'
}
if ($Case -in @('Target','All')) {
    $parameters.OutputDirectory = Join-Path $scratch 'wrong-target'
    $parameters.Target = 'x86_64-pc-windows-msvc'
    $parameters.Toolchain = 'nightly-2026-10-22-x86_64-pc-windows-msvc'
    Assert-BundleRejected $parameters 'Executable target differs'
}
if ($Case -in @('Archive','All')) {
    $archivePath = Join-Path $scratch "valid/$($receipt.Archive.Name)"
    if ((Get-Item -LiteralPath $archivePath).Length -ne $receipt.Archive.Bytes -or
        (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $receipt.Archive.Sha256) {
        throw 'The release archive differs from its receipt.'
    }
    $archive = [IO.Compression.ZipFile]::OpenRead($archivePath)
    try {
        if ($archive.Entries.Count -ne 9) { throw 'The release archive inventory is incomplete.' }
        foreach ($entry in $archive.Entries) {
            $file = @($receipt.Files | Where-Object { $_.Name -ceq $entry.FullName })
            if ($file.Count -ne 1 -or $file[0].Bytes -ne $entry.Length) { throw 'The archive file identity differs.' }
            $stream = $entry.Open()
            try { $hash = [Convert]::ToHexStringLower([Security.Cryptography.SHA256]::HashData($stream)) }
            finally { $stream.Dispose() }
            if ($hash -cne $file[0].Sha256) { throw 'The archive file hash differs.' }
        }
    } finally { $archive.Dispose() }
}
return "Release bundle $Case checks passed. Synthetic evidence: $scratch"

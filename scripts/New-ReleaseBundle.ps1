#requires -Version 7.6
<#
.SYNOPSIS
Packages checked Windows executables with licenses and an exact source receipt.
#>
[CmdletBinding()]
[OutputType([Collections.IDictionary])]
param(
    [Parameter(Mandatory)][string]$BuildDirectory,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [Parameter(Mandatory)][ValidateSet('aarch64-pc-windows-msvc','x86_64-pc-windows-msvc')][string]$Target,
    [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{40}$')][string]$SourceRevision,
    [Parameter(Mandatory)][ValidatePattern('^[0-9]+\.[0-9]+\.[0-9]+$')][string]$Version,
    [Parameter(Mandatory)][string]$Toolchain
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot 'Bonsai.Common.psm1')
Import-Module (Join-Path $PSScriptRoot 'Bonsai.Native.psm1')
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$build = Resolve-ExternalPath $BuildDirectory
$output = Resolve-ExternalPath $OutputDirectory
if (Test-Path -LiteralPath $output) { throw 'Use a new release bundle output directory.' }
[void][IO.Directory]::CreateDirectory($output)
$files = Join-Path $output 'files'
[void][IO.Directory]::CreateDirectory($files)
$git = (Get-Command git -CommandType Application -ErrorAction Stop).Source
$identity = Invoke-BonsaiNativeProcess -Executable $git -ArgumentList @('-C',$repo,'rev-parse','HEAD') -WorkingDirectory $output -TimeoutSeconds 30
if ($identity.ExitCode -or $identity.StandardOutput.Trim() -cne $SourceRevision) { throw 'The checked source revision differs from the release receipt.' }
$manifest = [IO.File]::ReadAllText((Join-Path $repo 'Cargo.toml'))
$declared = [regex]::Match($manifest, '(?m)^version = "(?<Version>[0-9]+\.[0-9]+\.[0-9]+)"\r?$')
if (-not $declared.Success -or $declared.Groups['Version'].Value -cne $Version) { throw 'The package version differs from the release receipt.' }
$executables = @('prism-inference.exe','bonsai-token-worker.exe','bonsai-swift-worker.exe','prism-inference-server.exe','bonsai-gguf.exe')
$machine = if ($Target -ceq 'aarch64-pc-windows-msvc') { 0xaa64 } else { 0x8664 }
foreach ($name in $executables) {
    $source = Join-Path $build $name
    $stream = [IO.File]::OpenRead($source)
    $reader = [IO.BinaryReader]::new($stream)
    try {
        if ($reader.ReadUInt16() -ne 0x5a4d) { throw "Expected a Windows executable: $name" }
        $stream.Position = 0x3c
        $offset = $reader.ReadUInt32()
        if ($offset -gt $stream.Length - 6) { throw "Invalid PE header: $name" }
        $stream.Position = $offset
        if ($reader.ReadUInt32() -ne 0x4550 -or $reader.ReadUInt16() -ne $machine) { throw "Executable target differs from bundle target: $name" }
    } finally { $reader.Dispose() }
    [IO.File]::Copy($source, (Join-Path $files $name), $false)
}
foreach ($name in @('LICENSE','NOTICE','PRISM-LICENSE.txt','CAMELID-LICENSE.txt')) {
    [IO.File]::Copy((Join-Path $repo $name), (Join-Path $files $name), $false)
}
$inventory = @(Get-ChildItem -LiteralPath $files -File | Sort-Object Name | ForEach-Object {
    [ordered]@{ Name = $_.Name; Bytes = $_.Length; Sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
})
$stem = "prism-inference-$Version-$Target"
$archive = Join-Path $output ($stem + '.zip')
Compress-Archive -LiteralPath @(Get-ChildItem -LiteralPath $files -File | ForEach-Object FullName) -DestinationPath $archive -CompressionLevel Fastest
$receipt = [ordered]@{
    SchemaVersion = 1
    Repository = 'https://github.com/symbiogenesis/prism-inference'
    SourceRevision = $SourceRevision
    Version = $Version
    Target = $Target
    Toolchain = $Toolchain
    Files = $inventory
    Archive = [ordered]@{ Name = [IO.Path]::GetFileName($archive); Bytes = (Get-Item -LiteralPath $archive).Length; Sha256 = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() }
    RunId = $env:GITHUB_RUN_ID
    RunAttempt = $env:GITHUB_RUN_ATTEMPT
}
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $output ($stem + '.receipt.json')) -Encoding utf8NoBOM
return $receipt

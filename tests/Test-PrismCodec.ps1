#requires -Version 7.6
<# .SYNOPSIS
Checks the preserved synthetic codec corpus against the exact released ARM64 ABI.
.DESCRIPTION
Verifies and holds every archive/installed file before a bounded owned child runs.
No Runner installer, maintenance CLI, model, or performance measurement is used.
#>
[CmdletBinding()]
[OutputType([string])]
param([Parameter(Mandatory)][string]$RuntimeArchive,
    [Parameter(Mandatory)][string]$ProbeToolPath,
    [Parameter(Mandatory)][string]$EvidenceDirectory)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
if (-not $IsWindows -or [Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture -ne 'Arm64') {
    throw 'The pinned native codec test requires Windows ARM64 PowerShell.'
}
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Import-Module (Join-Path $repo 'scripts/Bonsai.Common.psm1')
Import-Module (Join-Path $repo 'scripts/Bonsai.Native.psm1')
$archivePath = Resolve-ExternalPath $RuntimeArchive
$probe = Resolve-ExternalPath $ProbeToolPath
$evidence = Resolve-ExternalPath $EvidenceDirectory
if (Test-Path -LiteralPath $evidence) { throw 'Use a fresh codec evidence directory.' }
$lock = Get-Content -LiteralPath (Join-Path $repo 'tools/codec-probe/native.lock.json') -Raw | ConvertFrom-Json -AsHashtable
$runtime = $lock.Runtime
if ($lock.SchemaVersion -ne 1 -or $runtime.commit -cne '73bfe6cc485554822f80c473a3c0c4f25e80f41b' -or
    $runtime.asset.bytes -ne 12412126 -or $runtime.asset.sha256 -cne 'cc808b33d35c72a3bd5c69a0d609d422426feb9dce23868e77992b6c33b8ddb3') {
    throw 'Review the fixed native codec ABI before changing its released runtime pin.'
}
Assert-Arm64PE -Path $probe
$corpus = Join-Path $repo 'tools/codec-probe/tests/fixtures/random-pq2.bin'
if ((Get-FileHash -LiteralPath $corpus -Algorithm SHA256).Hash.ToLowerInvariant() -cne '0f0405ee9d335b5cb42df4c2ac95c1adb6bf272fb6380f1c9ba890e7c725b64f') {
    throw 'Preserved native codec corpus digest changed.'
}
$handles = [Collections.Generic.List[IDisposable]]::new()
$zip = $null
try {
    $archive = [IO.FileStream]::new($archivePath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
    $handles.Add($archive)
    if ($archive.Length -ne $runtime.asset.bytes -or
        [Convert]::ToHexStringLower([Security.Cryptography.SHA256]::HashData($archive)) -cne $runtime.asset.sha256) {
        throw 'Native codec runtime archive differs from its immutable pin.'
    }
    $archive.Position = 0
    $zip = [IO.Compression.ZipArchive]::new($archive, [IO.Compression.ZipArchiveMode]::Read, $true)
    if ($zip.Entries.Count -gt 4096) { throw 'Runtime archive entry count exceeds the fixed bound.' }
    $names = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $total = 0L
    foreach ($entry in $zip.Entries) {
        if ($entry.FullName -cnotmatch '^(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+/?$' -or
            @($entry.FullName.TrimEnd('/').Split('/') | Where-Object { $_ -in @('.', '..') -or $_.EndsWith('.') }).Count -or
            -not $names.Add($entry.FullName.TrimEnd('/')) -or
            (($entry.ExternalAttributes -shr 16) -band 0xF000) -eq 0xA000 -or
            $entry.Length -gt 256MB) { throw 'Unsupported runtime ZIP entry.' }
        $total += $entry.Length
        if ($total -gt 512MB) { throw 'Runtime archive expansion exceeds512MiB.' }
    }
    [void][IO.Directory]::CreateDirectory($evidence)
    $runtimeDirectory = Join-Path $evidence $runtime.commit
    [void][IO.Directory]::CreateDirectory($runtimeDirectory)
    $files = [Collections.Generic.List[object]]::new()
    foreach ($entry in $zip.Entries) {
        $path = Join-Path $runtimeDirectory $entry.FullName
        if ($entry.FullName.EndsWith('/')) { [void][IO.Directory]::CreateDirectory($path); continue }
        [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($path))
        $source = $entry.Open()
        $output = [IO.FileStream]::new($path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
        $hash = [Security.Cryptography.IncrementalHash]::CreateHash([Security.Cryptography.HashAlgorithmName]::SHA256)
        try {
            $buffer = [byte[]]::new(65536)
            $copied = 0L
            while (($count = $source.Read($buffer, 0, $buffer.Length)) -gt 0) {
                $copied += $count
                if ($copied -gt $entry.Length) { throw 'Runtime ZIP entry exceeded its declared size.' }
                $hash.AppendData($buffer, 0, $count)
                $output.Write($buffer, 0, $count)
            }
            if ($copied -ne $entry.Length) { throw 'Runtime ZIP entry was truncated.' }
            $expected = [Convert]::ToHexStringLower($hash.GetHashAndReset())
            $output.Flush($true)
        } finally { $hash.Dispose(); $output.Dispose(); $source.Dispose() }
        $guard = [IO.FileStream]::new($path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
        $handles.Add($guard)
        $digest = [Convert]::ToHexStringLower([Security.Cryptography.SHA256]::HashData($guard))
        if ($guard.Length -ne $entry.Length -or $digest -cne $expected) { throw 'Installed runtime file differs from its verified ZIP entry.' }
        if ([IO.Path]::GetExtension($path) -in @('.exe','.dll')) { Assert-Arm64PE -Path $path }
        $files.Add(@{ Name=$entry.FullName; Bytes=$guard.Length; Sha256=$digest })
    }
    foreach ($name in @('llama-server.exe','ggml-base.dll')) {
        if (-not [IO.File]::Exists((Join-Path $runtimeDirectory $name))) { throw "Missing required native runtime file: $name" }
    }
    [IO.File]::WriteAllText((Join-Path $evidence 'runtime-receipt.json'), (@{
        SchemaVersion=1; Commit=$runtime.commit; ArchiveSha256=$runtime.asset.sha256; Files=$files.ToArray()
    } | ConvertTo-Json -Depth 8))
    $resultPath = Join-Path $evidence 'result.json'
    $parent = [Diagnostics.Process]::GetCurrentProcess()
    try {
        $arguments = @('--runtime-directory',$runtimeDirectory,'--result',$resultPath,
            '--parent-process-id',[string]$PID,'--parent-creation-time-file-time',[string]$parent.StartTime.ToUniversalTime().ToFileTimeUtc())
        $run = Invoke-BonsaiNativeProcess -Executable $probe -ArgumentList $arguments -WorkingDirectory $evidence -TimeoutSeconds 90 -LogPath (Join-Path $evidence 'supervisor.log')
    } finally { $parent.Dispose() }
    if ($run.ExitCode -ne 0) { throw "Owned native codec test failed. Evidence: $evidence" }
    $supervision = Get-Content -LiteralPath (Join-Path $evidence 'supervision.json') -Raw | ConvertFrom-Json -DateKind String
    $result = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json -DateKind String
    if ($supervision.Status -cne 'owned-native-codec-complete' -or $supervision.MemoryLimitBytes -ne 1GB -or
        $supervision.ActiveProcessLimit -ne 1 -or $supervision.TimeoutSeconds -ne 30 -or $supervision.ModelLoaded -ne $false -or
        $supervision.NativeRuntimeCommit -cne $runtime.commit -or $supervision.CodecPeakJobMemoryBytes -gt 1GB -or
        $supervision.VersionPeakJobMemoryBytes -gt 1GB -or
        $result.Status -cne 'native-fp32-bit-parity-passed' -or $result.SyntheticBlocks -ne 6670 -or
        $result.Fp32Values -ne 853760 -or $result.NativePq2Type -ne 142 -or $result.NativePtq1Type -ne 143 -or
        $result.TraitsBytes -ne 56 -or $result.ToFloatOffset -ne 40 -or $result.ModelLoaded -ne $false -or
        $result.NativeRuntimeCommit -cne $runtime.commit) { throw "Native codec evidence differs from the fixed contract: $evidence" }
    Write-Output "Synthetic native PQ2/PTQ1 FP32 bit-parity passed. Evidence: $evidence"
} finally {
    if ($zip) { $zip.Dispose() }
    foreach ($handle in $handles) { $handle.Dispose() }
}

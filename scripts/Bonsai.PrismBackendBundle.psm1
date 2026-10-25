#requires -Version 7.6
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot 'Bonsai.Common.psm1')

function New-BonsaiPrismBackendBundle {
    <# .SYNOPSIS
    Copies the settled native ARM64 production tools and Rust harness for CI shards.
    #>
    [CmdletBinding()]
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseShouldProcessForStateChangingFunctions', '', Justification = 'Copies already authorized CI build outputs into a fresh external artifact directory.')]
    [OutputType([hashtable])]
    param([Parameter(Mandatory)][Collections.IDictionary]$Receipt,
          [Parameter(Mandatory)][string]$Directory)
    $directory = Resolve-ExternalPath $Directory
    if ([IO.Directory]::Exists($directory) -or [IO.File]::Exists($directory)) { throw 'Backend artifact bundle must be fresh.' }
    $common = @($Receipt.NativeBuilds | Where-Object { $_.Family -ceq 'common' })
    if ($Receipt.NativePlatform -cne 'ARM64' -or $common.Count -ne 1 -or
        $Receipt.ProjectRevision -cnotmatch '^[a-f0-9]{40}$') { throw 'Backend shards require a revision-bound native ARM64 build.' }
    $inputs = @(
        @{ Name='upstream_prism.exe'; Path=$Receipt.HarnessExecutable; Sha256=$Receipt.HarnessSha256 },
        @{ Name='prism-upstream-common-bridge.exe'; Path=$common[0].Executable; Sha256=$common[0].ExecutableSha256 }
    )
    $ggml = @($Receipt.NativeBuilds | Where-Object { $_.Family -ceq 'ggml' })
    $sharedTools = $ggml.Count -eq 1 -and $Receipt.Contains('LoraMergeTool')
    if ($sharedTools) {
        $inputs += @(
            @{ Name='prism-upstream-bridge.exe'; Path=$ggml[0].Executable; Sha256=$ggml[0].ExecutableSha256 },
            @{ Name='llama-export-lora.exe'; Path=$Receipt.LoraMergeTool.Executable; Sha256=$Receipt.LoraMergeTool.Sha256 }
        )
    }
    [void][IO.Directory]::CreateDirectory($directory)
    $files = @(foreach ($inputFile in $inputs) {
        Assert-Arm64PE -Path $inputFile.Path
        $digest = (Get-FileHash -LiteralPath $inputFile.Path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($digest -cne $inputFile.Sha256) { throw 'Backend build changed before artifact publication.' }
        $destination = Join-Path $directory $inputFile.Name
        [IO.File]::Copy($inputFile.Path, $destination, $false)
        @{ Name=$inputFile.Name; Bytes=([IO.FileInfo]$destination).Length; Sha256=$digest; PeMachine='0xAA64' }
    })
    $source = Join-Path $Receipt.SourceDirectory 'tests/test-backend-ops.cpp'
    $sourceHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($sourceHash -cne '8bf7a5cf29d3b3e88aa0c30611f5f1679f46a71ec00306908c44a73ec98ac425') { throw 'Backend source differs from the immutable catalog.' }
    $sourceDirectory = Join-Path $directory 'source/tests'
    [void][IO.Directory]::CreateDirectory($sourceDirectory)
    [IO.File]::Copy($source, (Join-Path $sourceDirectory 'test-backend-ops.cpp'), $false)
    $manifest = @{
        SchemaVersion=$(if ($sharedTools) { 2 } else { 1 }); ProjectRevision=$Receipt.ProjectRevision; SourceCommit=$Receipt.SourceCommit
        SourceArchiveSha256=$Receipt.SourceArchiveSha256; SourceSha256=$sourceHash
        Toolchain=$Receipt.Toolchain; NativePlatform='ARM64'; Files=$files
    }
    if ($sharedTools) {
        $manifest.NativeBuilds = @($Receipt.NativeBuilds)
        $manifest.LoraMergeTool = $Receipt.LoraMergeTool
        $manifest.ReferenceProductionFix = $Receipt.ReferenceProductionFix
    }
    [IO.File]::WriteAllText((Join-Path $directory 'manifest.json'), ($manifest | ConvertTo-Json -Depth 10))
    return $manifest
}

function Import-BonsaiPrismBackendBundle {
    <# .SYNOPSIS
    Verifies revision, immutable source, hashes and PE architecture before shard execution.
    #>
    [CmdletBinding()]
    [OutputType([hashtable])]
    param([Parameter(Mandatory)][string]$Directory,
          [Parameter(Mandatory)][string]$Repository,
          [string]$Toolchain,
          [switch]$RequireCommonTools)
    $directory = Resolve-ExternalPath $Directory
    $manifestPath = Join-Path $directory 'manifest.json'
    if ((Get-Item -LiteralPath $manifestPath).Length -gt 65536) { throw 'Backend manifest exceeds its bound.' }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json -AsHashtable
    $lock = Get-Content -LiteralPath (Join-Path $Repository 'tools/prism/source.lock.json') -Raw | ConvertFrom-Json -AsHashtable
    $expectedNames = @('upstream_prism.exe', 'prism-upstream-common-bridge.exe')
    if ($manifest.SchemaVersion -eq 2) { $expectedNames += @('prism-upstream-bridge.exe', 'llama-export-lora.exe') }
    if ($manifest.SchemaVersion -notin @(1, 2) -or $manifest.ProjectRevision -cnotmatch '^[a-f0-9]{40}$' -or
        $manifest.ProjectRevision -cne $env:GITHUB_SHA -or
        $manifest.SourceCommit -cne $lock.Commit -or $manifest.SourceArchiveSha256 -cne $lock.ArchiveSha256 -or
        $manifest.SourceSha256 -cne '8bf7a5cf29d3b3e88aa0c30611f5f1679f46a71ec00306908c44a73ec98ac425' -or
        $manifest.NativePlatform -cne 'ARM64' -or $manifest.Files.Count -ne $expectedNames.Count -or
        ($Toolchain -and $manifest.Toolchain -cne $Toolchain) -or
        ($RequireCommonTools -and $manifest.SchemaVersion -ne 2)) { throw 'Backend artifact provenance differs from this revision.' }
    $names = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($file in $manifest.Files) {
        if ($file.Name -cnotin $expectedNames -or
            -not $names.Add($file.Name) -or $file.Bytes -le 0 -or $file.Bytes -gt 268435456 -or
            $file.Sha256 -cnotmatch '^[a-f0-9]{64}$' -or $file.PeMachine -cne '0xAA64') { throw 'Invalid backend artifact file identity.' }
        $path = Resolve-ExternalPath (Join-Path $directory $file.Name)
        Test-LockedArtifact -Path $path -Bytes $file.Bytes -Sha256 $file.Sha256
        Assert-Arm64PE -Path $path
    }
    $source = Resolve-ExternalPath (Join-Path $directory 'source/tests/test-backend-ops.cpp')
    if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() -cne $manifest.SourceSha256) { throw 'Backend shard source differs.' }
    $harness = @($manifest.Files | Where-Object Name -CEQ 'upstream_prism.exe')[0]
    $bridge = @($manifest.Files | Where-Object Name -CEQ 'prism-upstream-common-bridge.exe')[0]
    $result = @{
        HarnessPath=(Join-Path $directory $harness.Name); HarnessSha256=$harness.Sha256
        CommonBridgePath=(Join-Path $directory $bridge.Name); CommonBridgeSha256=$bridge.Sha256
        SourceDirectory=(Join-Path $directory 'source')
    }
    if ($manifest.SchemaVersion -eq 2) {
        if ($manifest.NativeBuilds.Count -ne 2) { throw 'Shared native tool build provenance is incomplete.' }
        $builds = [Collections.Generic.List[object]]::new()
        foreach ($family in @('common', 'ggml')) {
            $build = @($manifest.NativeBuilds | Where-Object { $_.Family -ceq $family })
            $name = if ($family -ceq 'common') { 'prism-upstream-common-bridge.exe' } else { 'prism-upstream-bridge.exe' }
            $file = @($manifest.Files | Where-Object { $_.Name -ceq $name })[0]
            if ($build.Count -ne 1 -or $build[0].ProjectRevision -cne $manifest.ProjectRevision -or
                $build[0].ExecutableSha256 -cne $file.Sha256 -or
                $build[0].VerifiedPeMachine -cne '0xAA64' -or $build[0].CacheSha256 -cnotmatch '^[a-f0-9]{64}$') {
                throw 'Shared native tool build identity differs from its verified executable.'
            }
            $build[0].Executable = Join-Path $directory $name
            $builds.Add($build[0])
        }
        $lora = @($manifest.Files | Where-Object { $_.Name -ceq 'llama-export-lora.exe' })[0]
        if ($manifest.LoraMergeTool.ProjectRevision -cne $manifest.ProjectRevision -or
            $manifest.LoraMergeTool.Sha256 -cne $lora.Sha256 -or $manifest.LoraMergeTool.VerifiedPeMachine -cne '0xAA64') {
            throw 'Shared LoRA tool identity differs from its verified executable.'
        }
        $manifest.LoraMergeTool.Executable = Join-Path $directory $lora.Name
        $result.NativeBuilds = $builds.ToArray()
        $result.LoraMergeTool = $manifest.LoraMergeTool
        $patchLock = Get-Content -LiteralPath (Join-Path $Repository 'tools/prism/upstream/reference-patches.lock.json') -Raw | ConvertFrom-Json -AsHashtable
        $fix = $manifest.ReferenceProductionFix
        if ($patchLock.SourceCommit -cne $manifest.SourceCommit -or $fix.Variant -cne $patchLock.Variant -or
            $fix.Patches.Count -ne $patchLock.Patches.Count) { throw 'Shared production fix provenance differs from the reviewed lock.' }
        foreach ($expected in $patchLock.Patches) {
            $actual = @($fix.Patches | Where-Object { $_.RelativeSource -ceq $expected.RelativeSource })
            if ($actual.Count -ne 1 -or $actual[0].BeforeSha256 -cne $expected.BeforeSha256 -or
                $actual[0].AfterSha256 -cne $expected.AfterSha256 -or
                $actual[0].CanonicalPatchSha256 -cne $expected.CanonicalPatchSha256) {
                throw 'Shared production source substitution differs from its reviewed byte identity.'
            }
        }
        $result.ReferenceProductionFix = $fix
    }
    $result.ManifestSha256 = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
    return $result
}

Export-ModuleMember -Function New-BonsaiPrismBackendBundle, Import-BonsaiPrismBackendBundle

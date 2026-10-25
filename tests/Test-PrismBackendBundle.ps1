#requires -Version 7.6
<# .SYNOPSIS
Checks same-revision compiled tool handoff, rebasing and fail-closed provenance.
#>
[CmdletBinding()]
[OutputType([string])]
param([Parameter(Mandatory)][string]$SourceArchive)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$repo = Split-Path $PSScriptRoot -Parent
Import-Module (Join-Path $PSScriptRoot 'Bonsai.TestSupport.psm1')
Import-Module (Join-Path $repo 'scripts/Bonsai.Common.psm1')
Import-Module (Join-Path $repo 'scripts/Bonsai.PrismBackendBundle.psm1')
Import-Module (Join-Path $repo 'scripts/Bonsai.PrismSource.psm1')
$runtimeSource = Resolve-PrismSource
$lock = Get-Content -LiteralPath (Join-Path $runtimeSource 'tools/prism/source.lock.json') -Raw | ConvertFrom-Json -AsHashtable
$archive = Resolve-ExternalPath $SourceArchive
Test-LockedArtifact -Path $archive -Bytes $lock.ArchiveBytes -Sha256 $lock.ArchiveSha256
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('SwiftBonsai-PrismBundleTest-' + [guid]::NewGuid().ToString('N'))
$source = Join-Path $scratch 'verified-source'
[void][IO.Directory]::CreateDirectory((Join-Path $source 'tests'))
$zip = [IO.Compression.ZipFile]::OpenRead($archive)
try {
    $entry = $zip.GetEntry($lock.ArchiveRoot + '/tests/test-backend-ops.cpp')
    if ($null -eq $entry) { throw 'The pinned archive has no backend source.' }
    [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $source 'tests/test-backend-ops.cpp'))
} finally { $zip.Dispose() }
# These headers exercise the architecture verifier without executing any image.
$image = [byte[]]::new(128)
$image[0] = 0x4D; $image[1] = 0x5A; $image[60] = 64
$image[64] = 0x50; $image[65] = 0x45; $image[68] = 0x64; $image[69] = 0xAA
$paths = @{}
foreach ($name in @('harness', 'common', 'ggml', 'lora')) {
    $paths[$name] = Join-Path $scratch ($name + '.exe')
    [IO.File]::WriteAllBytes($paths[$name], $image)
}
$digest = (Get-FileHash -LiteralPath $paths.harness -Algorithm SHA256).Hash.ToLowerInvariant()
$revision = 'a' * 40
$toolchain = 'nightly-2026-10-22-aarch64-pc-windows-msvc'
$patchLock = Get-Content -LiteralPath (Join-Path $runtimeSource 'tools/prism/upstream/reference-patches.lock.json') -Raw | ConvertFrom-Json -AsHashtable
$receipt = [ordered]@{
    NativePlatform='ARM64'; ProjectRevision=$revision; SourceCommit=$lock.Commit
    SourceArchiveSha256=$lock.ArchiveSha256; SourceDirectory=$source; Toolchain=$toolchain
    HarnessExecutable=$paths.harness; HarnessSha256=$digest
    NativeBuilds=@(foreach ($family in @('common', 'ggml')) {
        @{ Family=$family; ProjectRevision=$revision; Executable=$paths[$family]; ExecutableSha256=$digest
           VerifiedPeMachine='0xAA64'; CacheSha256=('b' * 64); ConfigureArguments=@('-DPRISM_SOURCE_DIR=verified') }
    })
    LoraMergeTool=@{ Target='llama-export-lora'; ProjectRevision=$revision; Executable=$paths.lora; Sha256=$digest; VerifiedPeMachine='0xAA64' }
    ReferenceProductionFix=@{ Variant=$patchLock.Variant; Patches=$patchLock.Patches }
}
$bundlePath = Join-Path $scratch 'bundle'
$manifest = New-BonsaiPrismBackendBundle -Receipt $receipt -Directory $bundlePath
Assert ($manifest.SchemaVersion -eq 2 -and $manifest.Files.Count -eq 4) 'Prepared tools did not include both bridges and the LoRA exporter.'
$manifestPath = Join-Path $bundlePath 'manifest.json'
$originalManifest = [IO.File]::ReadAllText($manifestPath)
$previousRevision = $env:GITHUB_SHA
try {
    $env:GITHUB_SHA = $revision
    $bundle = Import-BonsaiPrismBackendBundle -Directory $bundlePath -Repository $runtimeSource -Toolchain $toolchain -RequireCommonTools
    Assert ($bundle.NativeBuilds.Count -eq 2 -and $bundle.ManifestSha256 -cmatch '^[a-f0-9]{64}$') 'Shared tool provenance was lost.'
    foreach ($build in $bundle.NativeBuilds) {
        Assert ($build.Executable.StartsWith($bundlePath + [IO.Path]::DirectorySeparatorChar, [StringComparison]::Ordinal)) 'A build path still points to the producer runner.'
    }
    Assert ($bundle.LoraMergeTool.Executable -ceq (Join-Path $bundlePath 'llama-export-lora.exe')) 'The LoRA executable was not rebased.'
    $env:GITHUB_SHA = 'c' * 40
    Assert-Failure { Import-BonsaiPrismBackendBundle -Directory $bundlePath -Repository $runtimeSource -RequireCommonTools } 'Foreign revision tools were accepted.'
    $env:GITHUB_SHA = $revision
    Assert-Failure { Import-BonsaiPrismBackendBundle -Directory $bundlePath -Repository $runtimeSource -Toolchain 'wrong' -RequireCommonTools } 'A foreign compiler toolchain was accepted.'
    $executable = Join-Path $bundlePath 'prism-upstream-common-bridge.exe'
    $changed = $image.Clone(); $changed[100] = 1
    [IO.File]::WriteAllBytes($executable, $changed)
    Assert-Failure { Import-BonsaiPrismBackendBundle -Directory $bundlePath -Repository $runtimeSource -RequireCommonTools } 'Modified same-length executable bytes were accepted.'
    [IO.File]::WriteAllBytes($executable, $image)
    foreach ($mutation in @('Duplicate', 'BuildDigest', 'Architecture', 'SourcePin', 'BuildRevision', 'ProductionFix')) {
        $record = $originalManifest | ConvertFrom-Json -AsHashtable
        switch -CaseSensitive ($mutation) {
            'Duplicate' { $record.Files[1].Name = $record.Files[0].Name }
            'BuildDigest' { $record.NativeBuilds[0].ExecutableSha256 = 'd' * 64 }
            'Architecture' { $record.Files[0].PeMachine = '0x8664' }
            'SourcePin' { $record.SourceArchiveSha256 = 'e' * 64 }
            'BuildRevision' { $record.NativeBuilds[0].ProjectRevision = 'f' * 40 }
            'ProductionFix' { $record.ReferenceProductionFix.Patches[0].AfterSha256 = '0' * 64 }
        }
        [IO.File]::WriteAllText($manifestPath, ($record | ConvertTo-Json -Depth 10))
        Assert-Failure { Import-BonsaiPrismBackendBundle -Directory $bundlePath -Repository $runtimeSource -RequireCommonTools } "Invalid $mutation provenance was accepted."
    }
    [IO.File]::WriteAllText($manifestPath, $originalManifest)
    $backendSource = Join-Path $bundlePath 'source/tests/test-backend-ops.cpp'
    [IO.File]::AppendAllText($backendSource, '// tampered')
    Assert-Failure { Import-BonsaiPrismBackendBundle -Directory $bundlePath -Repository $runtimeSource -RequireCommonTools } 'Modified pinned source was accepted.'
    $legacy = Join-Path $scratch 'legacy'
    $receipt.NativeBuilds = @($receipt.NativeBuilds | Where-Object { $_.Family -ceq 'common' })
    $receipt.Remove('LoraMergeTool')
    $legacyManifest = New-BonsaiPrismBackendBundle -Receipt $receipt -Directory $legacy
    Assert ($legacyManifest.SchemaVersion -eq 1) 'Backend-only tool publication changed schema.'
    $null = Import-BonsaiPrismBackendBundle -Directory $legacy -Repository $runtimeSource
    Assert-Failure { Import-BonsaiPrismBackendBundle -Directory $legacy -Repository $runtimeSource -RequireCommonTools } 'A backend-only bundle replaced ordinary source tools.'
} finally { $env:GITHUB_SHA = $previousRevision }
Write-Output "Prism compiled tool handoff and provenance checks passed. Evidence: $scratch"

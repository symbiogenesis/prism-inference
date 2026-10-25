#requires -Version 7.6
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot 'Bonsai.Common.psm1')
Import-Module (Join-Path $PSScriptRoot 'Bonsai.Native.psm1')
Import-Module (Join-Path $PSScriptRoot 'Bonsai.PrismNativeCache.psm1')

function Initialize-BonsaiPrismGrammarDependency {
    <#
    .SYNOPSIS
    Builds the pinned production grammar dependency in verified external storage.
    .DESCRIPTION
    Verifies the archive and freshly extracted sources against a stable source
    mirror, then builds only the LLGuidance production package with its unchanged
    Cargo.lock. Stable paths and timestamps preserve Cargo and native header
    reuse. It inherits the host environment and runs no tests or benchmarks.
    #>
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseShouldProcessForStateChangingFunctions', '', Justification = 'Explicit Prism test setup authorizes this bounded external dependency build.')]
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param(
        [Parameter(Mandatory)][string]$Repository,
        [Parameter(Mandatory)][string]$Directory,
        [Parameter(Mandatory)][ValidatePattern('^[A-Za-z0-9._-]+$')][string]$Toolchain,
        [Parameter(Mandatory)][ValidateSet('aarch64-pc-windows-msvc','x86_64-pc-windows-msvc')][string]$TargetTriple
    )
    if (-not [IO.Path]::IsPathFullyQualified($Repository) -or $Repository -match '%[^%]+%') {
        throw 'The dependency lock repository must be an expanded absolute path.'
    }
    $root = Resolve-ExternalPath $Directory
    [void][IO.Directory]::CreateDirectory($root)
    $lockPath = Join-Path $Repository 'tools/prism/upstream/llguidance.lock.json'
    $lock = Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json -AsHashtable
    $commit = 'd795912fedc7d393de740177ea9ea761e7905774'
    if ($lock.SchemaVersion -ne 1 -or $lock.Commit -cne $commit -or
        $lock.Repository -cne 'https://github.com/guidance-ai/llguidance' -or
        $lock.ArchiveRoot -cne "llguidance-$commit" -or
        $lock.ArchiveUrl -cne "https://codeload.github.com/guidance-ai/llguidance/zip/$commit" -or
        $lock.ArchiveBytes -ne 1130833 -or
        $lock.ArchiveSha256 -cne '01cbaa3ae908d9b164ec054bc3a1e2a3a0da97fa4e09e217e4017cab9b6d00c7' -or
        $lock.CargoLockSha256 -cnotmatch '^[a-f0-9]{64}$' -or $lock.LicenseSha256 -cnotmatch '^[a-f0-9]{64}$') {
        throw 'The production grammar dependency lock does not match the pinned upstream identity.'
    }
    $archive = Receive-LockedArtifact -Uri $lock.ArchiveUrl -Path (Join-Path $root "$commit.zip") `
        -Bytes $lock.ArchiveBytes -Sha256 $lock.ArchiveSha256
    # Retain fresh byte-verified source evidence for every invocation. Compilation
    # uses a separately verified mirror so fresh paths do not invalidate Cargo
    # fingerprints or CMake's production-header dependency paths.
    $invocation = Resolve-ExternalPath (Join-Path $root ('build-' + [Guid]::NewGuid().ToString('N')))
    [void][IO.Directory]::CreateDirectory($invocation)
    $stage = Join-Path $invocation 'source'
    [void][IO.Directory]::CreateDirectory($stage)
    $stagePrefix = $stage.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    $zip = [IO.Compression.ZipFile]::OpenRead($archive)
    try {
        if ($zip.Entries.Count -lt 1 -or $zip.Entries.Count -gt 10000) { throw 'Grammar archive entry count exceeds its bound.' }
        [long]$uncompressed = 0
        foreach ($entry in $zip.Entries) {
            $name = $entry.FullName
            if (-not $name.StartsWith($lock.ArchiveRoot + '/', [StringComparison]::Ordinal) -or
                $name.Contains('\') -or $name.Split('/') -contains '..' -or
                (($entry.ExternalAttributes -shr 16) -band 0xF000) -eq 0xA000) {
                throw 'Grammar archive has an unsafe source entry.'
            }
            $uncompressed += $entry.Length
            if ($uncompressed -gt 536870912) { throw 'Grammar archive expands beyond512MiB.' }
            $path = Resolve-ExternalPath ([IO.Path]::GetFullPath((Join-Path $stage $name)))
            if (-not $path.StartsWith($stagePrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Grammar archive entry escapes its extraction directory.' }
            if ($name.EndsWith('/', [StringComparison]::Ordinal)) {
                [void][IO.Directory]::CreateDirectory($path)
            } else {
                [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($path))
                $inputStream = $entry.Open()
                $outputStream = [IO.FileStream]::new($path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
                try { $inputStream.CopyTo($outputStream) } finally { $inputStream.Dispose(); $outputStream.Dispose() }
            }
        }
    } finally { $zip.Dispose() }
    $freshSource = Join-Path $stage $lock.ArchiveRoot
    $cargoLock = Join-Path $freshSource 'Cargo.lock'
    $license = Join-Path $freshSource 'LICENSE'
    if ((Get-FileHash -LiteralPath $cargoLock -Algorithm SHA256).Hash -ine $lock.CargoLockSha256 -or
        (Get-FileHash -LiteralPath $license -Algorithm SHA256).Hash -ine $lock.LicenseSha256) {
        throw 'Extracted grammar lock or license digest differs.'
    }
    # The verified subtree is separate from the archive, per-invocation evidence
    # and target directories. Every mirrored source byte is checked before use.
    $source = Initialize-BonsaiPrismNativeSource -SourceDirectory $freshSource -BuildDirectory (Join-Path $root 'verified')
    $cargoLock = Join-Path $source 'Cargo.lock'
    $cargo = (Get-Command cargo -CommandType Application -ErrorAction Stop).Source
    $version = Invoke-BonsaiNativeProcess -Executable $cargo -ArgumentList @("+$Toolchain", '--version') `
        -WorkingDirectory $invocation -TimeoutSeconds 120 -LogPath (Join-Path $invocation 'cargo-version.log')
    if ($version.ExitCode -ne 0) { throw "Cannot use the requested Rust toolchain; see $invocation" }
    # MSVC's linker still encounters legacy path limits for nested Cargo rlibs.
    # Keep this cache component short; Cargo fingerprints the actual toolchain.
    $target = Resolve-ExternalPath (Join-Path $root 't')
    $temporary = Join-Path $invocation 'temp'
    [void][IO.Directory]::CreateDirectory($temporary)
    $build = Invoke-BonsaiNativeProcess -Executable $cargo -ArgumentList @("+$Toolchain", 'build', '--locked',
        '--release', '--package', 'llguidance', '--manifest-path', (Join-Path $source 'Cargo.toml'),
        '--target', $TargetTriple, '--target-dir', $target) -WorkingDirectory $invocation `
        -Environment @{ TEMP=$temporary; TMP=$temporary } -TimeoutSeconds 1800 -LogPath (Join-Path $invocation 'build.log')
    if ($build.ExitCode -ne 0) { throw "Pinned production grammar dependency build failed; see $invocation" }
    if ((Get-FileHash -LiteralPath $cargoLock -Algorithm SHA256).Hash -ine $lock.CargoLockSha256) {
        throw 'The locked grammar dependency build changed Cargo.lock.'
    }
    $library = Resolve-ExternalPath (Join-Path $target "$TargetTriple/release/llguidance.lib")
    $includeDirectory = Join-Path $source 'parser'
    $header = Join-Path $includeDirectory 'llguidance.h'
    if (-not (Test-Path -LiteralPath $library -PathType Leaf) -or (Get-Item -LiteralPath $library).Length -le 0 -or
        -not (Test-Path -LiteralPath $header -PathType Leaf)) { throw 'The pinned grammar static library or production header is missing.' }
    $receipt = Join-Path $invocation 'receipt.json'
    $evidence = [ordered]@{
        SchemaVersion=1; Commit=$commit; Archive=$archive; ArchiveSha256=$lock.ArchiveSha256
        CargoLockSha256=$lock.CargoLockSha256; Toolchain=$Toolchain; TargetTriple=$TargetTriple
        CargoVersion=$version.StandardOutput.Trim(); SourceDirectory=$source; FreshSourceDirectory=$freshSource; Library=$library
        LibraryBytes=(Get-Item -LiteralPath $library).Length
        LibrarySha256=(Get-FileHash -LiteralPath $library -Algorithm SHA256).Hash.ToLowerInvariant()
        Header=$header; HeaderSha256=(Get-FileHash -LiteralPath $header -Algorithm SHA256).Hash.ToLowerInvariant()
        BuildLog=(Join-Path $invocation 'build.log'); ProductionPackageOnly=$true
    }
    [IO.File]::WriteAllText($receipt, ($evidence | ConvertTo-Json -Depth 5))
    return [pscustomobject]@{ Library=$library; IncludeDirectory=$includeDirectory; Receipt=$receipt }
}

Export-ModuleMember -Function Initialize-BonsaiPrismGrammarDependency

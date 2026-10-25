#requires -Version 7.6
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot 'Bonsai.Common.psm1')
Import-Module (Join-Path $PSScriptRoot 'Bonsai.Native.psm1')

function Get-BonsaiRustPackage {
    <#
    .SYNOPSIS
    Selects registered Rust packages without invoking any native tools.
    #>
    [CmdletBinding()]
    [OutputType([Collections.IDictionary])]
    param(
        [ValidateSet('All', 'Runtime', 'Broker')][string]$Group = 'All',
        [string[]]$Name = @()
    )
    $registry = Import-PowerShellDataFile (Join-Path $PSScriptRoot '../config/rust-packages.psd1')
    $packages = @($registry.Packages | Where-Object { $Group -eq 'All' -or $_.Group -eq $Group })
    foreach ($selected in $Name) {
        if ($selected -cnotin $packages.Name) { throw "Unknown Rust package in group ${Group}: $selected" }
    }
    $packages | Where-Object { $Name.Count -eq 0 -or $_.Name -cin $Name }
}

function Get-BonsaiRustFeature {
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory)][Collections.IDictionary]$Package,
        [Parameter(Mandatory)][ValidateSet('fmt', 'clippy', 'doc', 'test', 'build')][string]$Phase,
        [ValidateSet('Foundation', 'Compatibility', 'Native')][string]$CheckProfile = 'Foundation'
    )
    if ($CheckProfile -eq 'Native' -and $Package.Contains('NativeFeatures')) { return $Package.NativeFeatures }
    if ($CheckProfile -eq 'Compatibility' -and $Phase -ne 'test' -and $Package.Contains('CompatibilityCompileFeatures')) {
        return $Package.CompatibilityCompileFeatures
    }
    return $Package.Features
}

function Get-BonsaiRustArgument {
    <#
    .SYNOPSIS
    Builds a literal Cargo command without widening registered feature selection.
    #>
    [CmdletBinding()]
    [OutputType([string], [array])]
    param(
        [Parameter(Mandatory)][ValidateNotNullOrEmpty()][Collections.IDictionary[]]$Package,
        [string]$Toolchain = 'nightly-2026-10-22-aarch64-pc-windows-msvc',
        [Parameter(Mandatory)][ValidateSet('fmt', 'clippy', 'doc', 'test', 'build')][string]$Phase,
        [ValidateSet('Foundation', 'Compatibility', 'Native')][string]$CheckProfile = 'Foundation',
        [switch]$Offline
    )
    $manifest = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../Cargo.toml'))
    $arguments = @(('+' + $Toolchain), $Phase, '--manifest-path', $manifest)
    foreach ($selected in $Package) { $arguments += @('--package', $selected.Name) }
    if ($Phase -eq 'fmt') { return $arguments + '--check' }
    # --all-features covers selected packages only. Restricted feature profiles
    # need separate Cargo invocations, including when they are path dependencies.
    if ($Package.Count -gt 1) {
        foreach ($selected in $Package) {
            $selectedFeatures = @(Get-BonsaiRustFeature -Package $selected -Phase $Phase -CheckProfile $CheckProfile)
            if ($selectedFeatures.Count -ne 1 -or $selectedFeatures[0] -cne '*') {
                throw 'Only all-feature packages can share a Cargo compile command.'
            }
        }
    }
    $arguments += @('--locked', '--release')
    if ($Offline) { $arguments += '--offline' }
    $features = @(Get-BonsaiRustFeature -Package $Package[0] -Phase $Phase -CheckProfile $CheckProfile)
    if ('*' -in $features) { $arguments += '--all-features' }
    elseif ($features.Count) { $arguments += @('--features', ($features -join ',')) }
    if ($Phase -eq 'clippy') { $arguments += @('--all-targets', '--', '-D', 'warnings') }
    elseif ($Phase -eq 'doc') { $arguments += '--no-deps' }
    elseif ($Phase -eq 'test') { $arguments += '--no-fail-fast' }
    return $arguments
}

function Get-BonsaiRustCheckPlan {
    <#
    .SYNOPSIS
    Batches selected packages by phase while keeping restricted features isolated.
    #>
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param(
        [Parameter(Mandatory)][ValidateNotNullOrEmpty()][Collections.IDictionary[]]$Package,
        [string]$Toolchain = 'nightly-2026-10-22-aarch64-pc-windows-msvc',
        [Parameter(Mandatory)][ValidateSet('fmt', 'clippy', 'doc', 'test', 'build')][string[]]$Phase,
        [ValidateSet('Foundation', 'Compatibility', 'Native')][string]$CheckProfile = 'Foundation',
        [switch]$Offline
    )
    $seenPhases = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($step in $Phase) {
        if (-not $seenPhases.Add($step)) { continue }
        $batches = [Collections.Generic.List[object]]::new()
        if ($step -eq 'fmt') {
            $batches.Add([pscustomobject]@{ Packages = $Package })
        } else {
            $allFeatures = [Collections.Generic.List[Collections.IDictionary]]::new()
            foreach ($selected in $Package) {
                $features = @(Get-BonsaiRustFeature -Package $selected -Phase $step -CheckProfile $CheckProfile)
                if ($features.Count -eq 1 -and $features[0] -ceq '*') { $allFeatures.Add($selected) }
                else { $batches.Add([pscustomobject]@{ Packages = @($selected) }) }
            }
            if ($allFeatures.Count) { $batches.Insert(0, [pscustomobject]@{ Packages = $allFeatures.ToArray() }) }
        }
        foreach ($batch in $batches) {
            $name = if ($batch.Packages.Count -eq 1) { $batch.Packages[0].Name } else { 'packages' }
            [pscustomobject]@{
                Name = $name + '-' + $step
                Phase = $step
                Packages = @($batch.Packages.Name)
                Arguments = @(Get-BonsaiRustArgument -Package $batch.Packages -Toolchain $Toolchain -Phase $step `
                    -CheckProfile $CheckProfile -Offline:$Offline)
            }
        }
    }
}

function Invoke-BonsaiRustCheck {
    <#
    .SYNOPSIS
    Runs selected Rust phases with external targets and a retained check receipt.
    #>
    [CmdletBinding()]
    [OutputType([Collections.IDictionary], [Collections.Specialized.OrderedDictionary])]
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
    $phases = [Collections.Generic.List[string]]::new()
    foreach ($item in $Phase) { if ($item -cnotin $phases) { $phases.Add($item) } }
    if ($FullTests -and 'test' -cnotin $phases) { $phases.Add('test') }
    if ($Build -and 'build' -cnotin $phases) { $phases.Add('build') }
    if ('test' -in $phases -and ($env:GITHUB_ACTIONS -cne 'true' -or $env:GITHUB_REF -cne 'refs/heads/main')) {
        throw 'Complete Rust suites run only in GitHub Actions on main.'
    }
    $packages = @(Get-BonsaiRustPackage -Group $Group -Name $Package)
    if (-not $packages.Count -or -not $phases.Count) { throw 'Select at least one registered package and one phase.' }
    $plan = @(Get-BonsaiRustCheckPlan -Package $packages -Toolchain $Toolchain -Phase $phases.ToArray() `
        -CheckProfile $CheckProfile -Offline:$Offline)
    $target = Resolve-ExternalPath -Path $TargetDirectory
    $evidence = Resolve-ExternalPath -Path $EvidenceDirectory
    if ($target -eq $evidence -or $target.StartsWith($evidence + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
        $evidence.StartsWith($target + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Cargo targets must be separate from uploaded evidence.'
    }
    if (Test-Path -LiteralPath $evidence) { throw 'Use a new Rust check evidence directory.' }
    [void][IO.Directory]::CreateDirectory($target)
    [void][IO.Directory]::CreateDirectory($evidence)
    # Synthetic test scratch can contain credentials and large files; upload
    # command evidence separately and preserve every inherited system variable.
    $work = Resolve-ExternalPath -Path (Join-Path ([IO.Path]::GetTempPath()) ('SwiftBonsai-RustCheck-' + [guid]::NewGuid().ToString('N')))
    [void][IO.Directory]::CreateDirectory($work)
    $environment = @{ CARGO_TARGET_DIR = $target; TEMP = $work; TMP = $work }
    $cargo = (Get-Command -Name cargo -CommandType Application -ErrorAction Stop).Source
    $rustc = (Get-Command -Name rustc -CommandType Application -ErrorAction Stop).Source
    $records = [Collections.Generic.List[object]]::new()
    $receipt = [ordered]@{
        SchemaVersion = 2; Revision = $env:GITHUB_SHA; Toolchain = $Toolchain; CheckProfile = $CheckProfile
        FingerprintMode = if ($env:CARGO_BUILD_FINGERPRINT) { $env:CARGO_BUILD_FINGERPRINT } else { 'mtime' }
        Packages = @($packages.Name); TargetDirectory = $target; WorkingDirectory = $work
        CompilerIdentity = $null; Checks = $records; Success = $false; Error = $null
        PerformanceMeasured = $false
        PlannedCargoInvocations = $plan.Count
        UnbatchedCargoInvocations = $packages.Count * $phases.Count
        WorkspaceSha256 = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot '../Cargo.toml') -Algorithm SHA256).Hash.ToLowerInvariant()
        LockSha256 = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot '../Cargo.lock') -Algorithm SHA256).Hash.ToLowerInvariant()
        RegistrySha256 = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot '../config/rust-packages.psd1') -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    try {
        $identity = Invoke-BonsaiNativeProcess -Executable $rustc -ArgumentList @(('+' + $Toolchain), '-vV') `
            -WorkingDirectory $work -TimeoutSeconds 60 -LogPath (Join-Path $evidence 'rustc.log')
        if ($identity.ExitCode -ne 0) { throw 'Rust compiler identification failed.' }
        $receipt.CompilerIdentity = $identity.StandardOutput.Trim()
        foreach ($check in $plan) {
            $log = Join-Path $evidence ($check.Name + '.log')
            $record = [pscustomobject]@{
                Name = $check.Name; Phase = $check.Phase; Packages = $check.Packages
                Executable = $cargo; Arguments = $check.Arguments; ExitCode = $null
                Log = $log; Passed = $false; ElapsedMilliseconds = $null
            }
            $records.Add($record)
            $timer = [Diagnostics.Stopwatch]::StartNew()
            try {
                $result = Invoke-BonsaiNativeProcess -Executable $cargo -ArgumentList $check.Arguments `
                    -WorkingDirectory $work -TimeoutSeconds $TimeoutSeconds -Environment $environment -LogPath $log
            } finally {
                $timer.Stop()
                $record.ElapsedMilliseconds = $timer.ElapsedMilliseconds
            }
            $record.ExitCode = $result.ExitCode
            $record.Passed = $result.ExitCode -eq 0
            Write-Information -MessageData "$($check.Name) ($($check.Packages.Count) packages) exit $($result.ExitCode)" -InformationAction Continue
        }
        if ($records.Where({ -not $_.Passed }).Count) { throw 'Rust checks failed; inspect the retained command logs.' }
        $receipt.Success = $true
    } catch {
        $receipt.Error = $_.Exception.Message
        throw
    } finally {
        $receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $evidence 'rust-checks.json') -Encoding utf8NoBOM
    }
    return $receipt
}

Export-ModuleMember -Function Get-BonsaiRustPackage, Get-BonsaiRustArgument, Get-BonsaiRustCheckPlan, Invoke-BonsaiRustCheck

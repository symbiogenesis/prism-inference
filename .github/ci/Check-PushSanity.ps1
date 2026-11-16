#Requires -Version 7.6
<# Admit a source/digest-bound native checker and immutable committed checking policy. #>
[CmdletBinding()]
[OutputType([string])]
param(
    [string]$Root = (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent),
    [string]$TargetDirectory = (Join-Path ([IO.Path]::GetTempPath()) 'Prism-Push-Sanity-Cargo'),
    [switch]$Prepare,
    [string]$Commit
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
function Test-PushCheckerReceipt {
    [CmdletBinding()]
    [OutputType([bool])]
    param([Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)][string]$Image, [Parameter(Mandatory)][string]$SourceIdentity)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf) -or -not (Test-Path -LiteralPath $Image -PathType Leaf)) { return $false }
    try {
        $receipt = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -AsHashtable
        return $receipt -is [Collections.IDictionary] -and $receipt.Count -eq 3 -and
            $receipt.Contains('SchemaVersion') -and $receipt.Contains('SourceIdentity') -and $receipt.Contains('Sha256') -and
            $receipt.SchemaVersion -is [long] -and $receipt.SchemaVersion -eq 1 -and
            $receipt.SourceIdentity -is [string] -and $receipt.Sha256 -is [string] -and
            $receipt.SourceIdentity -ceq $SourceIdentity -and
            $receipt.Sha256 -ceq (Get-FileHash -LiteralPath $Image -Algorithm SHA256).Hash.ToLowerInvariant()
    } catch { return $false }
}
function Get-PushPowerObservation {
    [CmdletBinding()]
    [OutputType([string])]
    param()
    try {
        [void][Reflection.Assembly]::Load('System.Windows.Forms')
        $power = [Windows.Forms.SystemInformation]::PowerStatus
        return "AC=$($power.PowerLineStatus); charge=$($power.BatteryLifePercent); flags=$($power.BatteryChargeStatus); battery saver unknown"
    } catch { return 'AC=unknown; charge=unknown; flags=unknown; battery saver unknown' }
}
$started = [Diagnostics.Stopwatch]::StartNew()
$updates = if (-not $Prepare -and -not $Commit -and [Console]::IsInputRedirected) { [Console]::In.ReadToEnd() } else { '' }
if (-not $updates.Trim() -and -not $Prepare -and -not $Commit) { return }
if (-not $IsWindows -or [Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture -ne 'Arm64') { throw 'Push sanity requires native Windows ARM64 PowerShell.' }
if ($Prepare -and $Commit) { throw 'Choose preparation or a commit check.' }
$originalPwsh = $env:PRISM_PUSH_PWSH_PATH
$originalChecksum = $env:CARGO_UNSTABLE_CHECKSUM_FRESHNESS
$originalFingerprint = $env:CARGO_BUILD_FINGERPRINT
$originalIncremental = $env:CARGO_INCREMENTAL
$originalReplacements = $env:GIT_NO_REPLACE_OBJECTS
$gitEnvironment = @{}
foreach ($name in @('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_COMMON_DIR', 'GIT_OBJECT_DIRECTORY', 'GIT_ALTERNATE_OBJECT_DIRECTORIES', 'GIT_NAMESPACE', 'GIT_CONFIG', 'GIT_CONFIG_COUNT', 'GIT_CONFIG_PARAMETERS', 'GIT_SHALLOW_FILE')) {
    $gitEnvironment[$name] = [Environment]::GetEnvironmentVariable($name)
    [Environment]::SetEnvironmentVariable($name, [NullString]::Value)
}
$env:GIT_NO_REPLACE_OBJECTS = '1'
$env:PRISM_PUSH_PWSH_PATH = [Environment]::ProcessPath
Write-Information -MessageData "Push sanity power before: $(Get-PushPowerObservation)." -InformationAction Continue
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('Prism-Push-Bootstrap-' + [guid]::NewGuid().ToString('N'))
try {
    [void][IO.Directory]::CreateDirectory($scratch)
    $target = [IO.Path]::GetFullPath($TargetDirectory)
    $tool = Join-Path $target 'aarch64-pc-windows-msvc/debug/prism-source-sanity.exe'
    $receipt = Join-Path $target 'push-checker.json'
    if ($Prepare) {
        [void][IO.Directory]::CreateDirectory($target)
        [IO.File]::Delete($receipt)
    }
    $head = (git -C $Root rev-parse --verify HEAD).Trim()
    $inputs = @('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'tools/source-sanity')
    $objects = @(git -C $Root rev-parse @($inputs | ForEach-Object { '{0}:{1}' -f $head, $_ }))
    if ($objects.Count -ne $inputs.Count -or @($objects | Where-Object { $_ -cnotmatch '^(?:[0-9a-f]{40}|[0-9a-f]{64})$' }).Count) { throw 'Push blocked: committed checker input identities are incomplete.' }
    $configuration = @(git -C $Root ls-tree $head -- .cargo)
    if ($configuration.Count -gt 1 -or ($configuration.Count -eq 1 -and $configuration[0] -cnotmatch '^[0-9]+ tree [0-9a-f]{40}(?:[0-9a-f]{24})?\t\.cargo$')) { throw 'Push blocked: unexpected committed Cargo configuration.' }
    $objects += if ($configuration.Count) { ($configuration[0] -split '\s+')[2] } else { 'absent:.cargo' }
    $sourceId = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes(($objects -join "`n")))).ToLowerInvariant()
    if (-not $Prepare -and -not (Test-PushCheckerReceipt -Path $receipt -Image $tool -SourceIdentity $sourceId)) {
        throw 'Push blocked: native checker is missing, stale or corrupt. Run .github/ci/Check-PushSanity.ps1 -Prepare in a declared source-build allowance under native process-tree supervision.'
    }
    $archive = Join-Path $scratch 'policy.zip'
    $policy = Join-Path $scratch 'source'
    $arguments = @('-C', $Root, 'archive', '--format=zip', "--output=$archive", $head)
    if (-not $Prepare) { $arguments += @('--', '.github/ci/Check-SourceSnapshot.ps1', '.github/ci/Test-Workflow.ps1') }
    git @arguments
    Expand-Archive -LiteralPath $archive -DestinationPath $policy
    if ($Prepare) {
        $pin = @(Get-Content -LiteralPath (Join-Path $policy 'rust-toolchain.toml') | Where-Object { $_ -cmatch '^channel\s*=\s*"nightly-[0-9]{4}-[0-9]{2}-[0-9]{2}"\s*$' })
        if ($pin.Count -ne 1) { throw 'Preparation requires one dated nightly compiler pin.' }
        $toolchain = ($pin[0] -split '"')[1] + '-aarch64-pc-windows-msvc'
        $env:CARGO_UNSTABLE_CHECKSUM_FRESHNESS = 'true'
        $env:CARGO_BUILD_FINGERPRINT = 'content'
        $env:CARGO_INCREMENTAL = '1'
        Push-Location -LiteralPath $policy
        try {
            $metadata = 'prism_push_prepare_' + [guid]::NewGuid().ToString('N')
            cargo "+$toolchain" rustc --quiet --locked --offline --manifest-path (Join-Path $policy 'Cargo.toml') --package prism-source-sanity --bin prism-source-sanity --target aarch64-pc-windows-msvc --target-dir $target -- -C "metadata=$metadata"
        } finally { Pop-Location }
        @{ SchemaVersion = 1; SourceIdentity = $sourceId; Sha256 = (Get-FileHash -LiteralPath $tool -Algorithm SHA256).Hash.ToLowerInvariant() } | ConvertTo-Json | Set-Content -LiteralPath $receipt -Encoding utf8NoBOM
        if (-not (Test-PushCheckerReceipt -Path $receipt -Image $tool -SourceIdentity $sourceId)) {
            [IO.File]::Delete($receipt)
            throw 'Preparation failed native checker receipt verification.'
        }
        'Prepared committed native source checker; outgoing-tip validation remains separate.'
        return
    }
    $arguments = @('--root', $Root, '--policy-root', $policy, '--bootstrap-milliseconds', $started.ElapsedMilliseconds)
    if ($Commit) { $arguments += @('--check-commit', $Commit) }
    $updates | & $tool @arguments
    if ($LASTEXITCODE) { throw 'Push blocked: immutable source sanity did not pass.' }
} finally {
    try {
        $temporary = [IO.Path]::TrimEndingDirectorySeparator([IO.Path]::GetFullPath([IO.Path]::GetTempPath())) + [IO.Path]::DirectorySeparatorChar
        $absolute = [IO.Path]::GetFullPath($scratch)
        if (-not $absolute.StartsWith($temporary, [StringComparison]::OrdinalIgnoreCase) -or -not [IO.Path]::GetFileName($absolute).StartsWith('Prism-Push-Bootstrap-', [StringComparison]::Ordinal)) { throw 'Push bootstrap cleanup escaped temporary storage.' }
        if (Test-Path -LiteralPath $absolute -PathType Container) { Remove-Item -LiteralPath $absolute -Recurse -Force }
    } finally {
        Write-Information -MessageData "Push sanity power after: $(Get-PushPowerObservation); aggregate elapsed=$($started.Elapsed)." -InformationAction Continue
        foreach ($entry in @(@{ Name = 'PRISM_PUSH_PWSH_PATH'; Value = $originalPwsh }, @{ Name = 'CARGO_UNSTABLE_CHECKSUM_FRESHNESS'; Value = $originalChecksum }, @{ Name = 'CARGO_BUILD_FINGERPRINT'; Value = $originalFingerprint }, @{ Name = 'CARGO_INCREMENTAL'; Value = $originalIncremental }, @{ Name = 'GIT_NO_REPLACE_OBJECTS'; Value = $originalReplacements })) {
            if ($null -eq $entry.Value) { [Environment]::SetEnvironmentVariable($entry.Name, [NullString]::Value) }
            else { [Environment]::SetEnvironmentVariable($entry.Name, $entry.Value) }
        }
        foreach ($name in $gitEnvironment.Keys) {
            if ($null -eq $gitEnvironment[$name]) { [Environment]::SetEnvironmentVariable($name, [NullString]::Value) }
            else { [Environment]::SetEnvironmentVariable($name, $gitEnvironment[$name]) }
        }
    }
}

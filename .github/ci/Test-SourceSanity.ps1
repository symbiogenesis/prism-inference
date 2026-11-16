#Requires -Version 7.6
<# Focused syntax, checking-policy isolation and fail-closed checker receipt controls. #>
[CmdletBinding()]
[OutputType([string])]
param([ValidateSet('all', 'syntax', 'receipts')][string]$Case = 'all')
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('Prism-Source-Sanity-Control-' + [guid]::NewGuid().ToString('N'))
function Write-ControlSource {
    [CmdletBinding()]
    [OutputType([void])]
    param([Parameter(Mandatory)][string]$Directory, [Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)][string]$Text)
    $destination = Join-Path $Directory $Path
    [void][IO.Directory]::CreateDirectory((Split-Path $destination -Parent))
    [IO.File]::WriteAllText($destination, $Text)
}
function Assert-ControlRejection {
    [CmdletBinding()]
    [OutputType([void])]
    param([Parameter(Mandatory)][scriptblock]$Action, [Parameter(Mandatory)][string]$Expected)
    $failure = $null
    try { & $Action | Out-Null } catch { $failure = $_.Exception.Message }
    if ($null -eq $failure -or -not $failure.Contains($Expected, [StringComparison]::Ordinal)) { throw "Expected rejection containing '$Expected'; received '$failure'." }
}
try {
    [void][IO.Directory]::CreateDirectory($scratch)
    if ($Case -in @('all', 'syntax')) {
        $candidate = Join-Path $scratch 'candidate'
        $paths = Join-Path $scratch 'paths.json'
        $checker = Join-Path $PSScriptRoot 'Check-SourceSnapshot.ps1'
        Write-ControlSource -Directory $candidate -Path 'scripts/file with spaces.ps1' -Text "throw 'Candidate bodies must never execute.'`n"
        [IO.File]::WriteAllText($paths, '["scripts/file with spaces.ps1","scripts/deleted.ps1"]')
        & $checker -Root $candidate -PathsFile $paths
        Write-ControlSource -Directory $candidate -Path 'scripts/file with spaces.ps1' -Text 'function Broken {'
        Assert-ControlRejection { & $checker -Root $candidate -PathsFile $paths } 'PowerShell syntax failed'
        [IO.File]::WriteAllText($paths, '["../escaped.ps1"]')
        Assert-ControlRejection { & $checker -Root $candidate -PathsFile $paths } 'escaped its root'
        $policy = Join-Path $scratch 'policy/.github/ci'
        [void][IO.Directory]::CreateDirectory($policy)
        [IO.File]::Copy($checker, (Join-Path $policy 'Check-SourceSnapshot.ps1'))
        Write-ControlSource -Directory $candidate -Path '.github/ci/Test-Workflow.ps1' -Text "throw 'Candidate checking policy must never execute.'`n"
        Write-ControlSource -Directory $policy -Path 'Test-Workflow.ps1' -Text 'param([string]$Root); [IO.File]::WriteAllText((Join-Path $PSScriptRoot "observed-root.txt"), $Root); $global:LASTEXITCODE = 0'
        [IO.File]::WriteAllText($paths, '[".github/workflows/deleted.yml"]')
        $previousLinter = $env:PRISM_ACTIONLINT_PATH
        try {
            [Environment]::SetEnvironmentVariable('PRISM_ACTIONLINT_PATH', [NullString]::Value)
            & (Join-Path $policy 'Check-SourceSnapshot.ps1') -Root $candidate -PathsFile $paths
        } finally {
            if ($null -eq $previousLinter) { [Environment]::SetEnvironmentVariable('PRISM_ACTIONLINT_PATH', [NullString]::Value) }
            else { $env:PRISM_ACTIONLINT_PATH = $previousLinter }
        }
        if ([IO.File]::ReadAllText((Join-Path $policy 'observed-root.txt')) -cne [IO.Path]::GetFullPath($candidate)) { throw 'Trusted workflow checker did not receive the candidate snapshot.' }
        'Snapshot controls passed: parse-only intake, syntax rejection, deleted sources, path containment and trusted policy selection.'
    }
    if ($Case -in @('all', 'receipts')) {
        $repository = Join-Path $scratch 'repository'
        $target = Join-Path $scratch 'target'
        foreach ($path in @('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'tools/source-sanity/fixture.txt')) { Write-ControlSource -Directory $repository -Path $path -Text 'Inert committed identity fixture; no compiler consumer.' }
        git -C $repository init --quiet
        git -C $repository add --all
        git -C $repository -c user.name=Fixture -c user.email=fixture@example.invalid -c core.hooksPath= -c commit.gpgsign=false commit --quiet -m fixture
        $head = (git -C $repository rev-parse HEAD).Trim()
        $objects = @(@('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'tools/source-sanity') | ForEach-Object { (git -C $repository rev-parse --verify ('{0}:{1}' -f $head, $_)).Trim() })
        $objects += 'absent:.cargo'
        $identity = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes(($objects -join "`n")))).ToLowerInvariant()
        $image = Join-Path $target 'aarch64-pc-windows-msvc/debug/prism-source-sanity.exe'
        Write-ControlSource -Directory $target -Path 'aarch64-pc-windows-msvc/debug/prism-source-sanity.exe' -Text 'Inert checker must never execute.'
        $digest = (Get-FileHash -LiteralPath $image -Algorithm SHA256).Hash.ToLowerInvariant()
        $receipt = Join-Path $target 'push-checker.json'
        $adapter = Join-Path $PSScriptRoot 'Check-PushSanity.ps1'
        foreach ($state in @('missing', 'malformed', 'stale', 'collection-source', 'collection-digest', 'corrupt')) {
            [IO.File]::Delete($receipt)
            $value = @{ SchemaVersion = 1; SourceIdentity = $identity; Sha256 = $digest }
            switch ($state) {
                'stale' { $value.SourceIdentity = '0' * 64 }
                'collection-source' { $value.SourceIdentity = @($identity, ('0' * 64)) }
                'collection-digest' { $value.Sha256 = @($digest, ('0' * 64)) }
                'corrupt' { $value.Sha256 = '0' * 64 }
            }
            if ($state -eq 'malformed') { [IO.File]::WriteAllText($receipt, '{"unfinished":') }
            elseif ($state -ne 'missing') { $value | ConvertTo-Json | Set-Content -LiteralPath $receipt -Encoding utf8NoBOM }
            Assert-ControlRejection { & $adapter -Root $repository -TargetDirectory $target -Commit $head } '-Prepare'
        }
        'Receipt controls passed: missing, malformed, stale, collection-valued and corrupt helpers fail before execution.'
    }
} finally {
    $absolute = [IO.Path]::GetFullPath($scratch)
    $temporary = [IO.Path]::TrimEndingDirectorySeparator([IO.Path]::GetFullPath([IO.Path]::GetTempPath())) + [IO.Path]::DirectorySeparatorChar
    if (-not $absolute.StartsWith($temporary, [StringComparison]::OrdinalIgnoreCase) -or -not [IO.Path]::GetFileName($absolute).StartsWith('Prism-Source-Sanity-Control-', [StringComparison]::Ordinal)) { throw 'Source sanity control cleanup escaped temporary storage.' }
    Remove-Item -LiteralPath $absolute -Recurse -Force
}
exit 0

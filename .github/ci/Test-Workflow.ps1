#Requires -Version 7.6
<# Validate workflow syntax, expressions, dependencies, local action inputs and immutable action references. #>
[CmdletBinding()]
param(
    [string]$Root = (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent),
    [string]$ActionlintPath,
    [switch]$CheckFixtures
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

function Test-CiActionReferences([string]$RepositoryRoot) {
    $errors = [Collections.Generic.List[string]]::new()
    $sources = @(Get-ChildItem -LiteralPath (Join-Path $RepositoryRoot '.github/workflows') -File |
        Where-Object Extension -cin @('.yml', '.yaml'))
    $sources += @(Get-ChildItem -LiteralPath (Join-Path $RepositoryRoot '.github/actions') -File -Recurse |
        Where-Object Name -cin @('action.yml', 'action.yaml'))
    foreach ($source in $sources) {
        $blockIndent = -1
        $number = 0
        foreach ($line in [IO.File]::ReadAllLines($source.FullName)) {
            $number++
            $indent = $line.Length - $line.TrimStart(' ').Length
            if ($blockIndent -ge 0) {
                if (-not $line.Trim() -or $indent -gt $blockIndent) { continue }
                $blockIndent = -1
            }
            if ($line -match ':\s*[|>][+-]?\s*(?:#.*)?$') { $blockIndent = $indent; continue }
            if ($line -notmatch '^\s*(?:-\s*)?(?:uses|''uses''|"uses"):\s*(.+?)\s*$') { continue }
            $reference = ($Matches[1] -split '\s+#', 2)[0].Trim().Trim('"', "'")
            $location = [IO.Path]::GetRelativePath($RepositoryRoot, $source.FullName) + ':' + $number
            if ($reference.StartsWith('./', [StringComparison]::Ordinal)) {
                $relative = $reference.Substring(2)
                if ($relative.Contains('@') -or $relative.Contains('\') -or '..' -cin ($relative -split '/')) {
                    $errors.Add("${location}: invalid local action path $reference")
                    continue
                }
                $target = [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $relative))
                $prefix = [IO.Path]::TrimEndingDirectorySeparator([IO.Path]::GetFullPath($RepositoryRoot)) + [IO.Path]::DirectorySeparatorChar
                if (-not $target.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
                    $errors.Add("${location}: local action escaped the repository")
                    continue
                }
                $manifests = @('action.yml', 'action.yaml') | ForEach-Object { Join-Path $target $_ } |
                    Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }
                if (@($manifests).Count -ne 1 -and -not ($relative.StartsWith('.github/workflows/') -and (Test-Path -LiteralPath $target -PathType Leaf))) {
                    $errors.Add("${location}: local action or reusable workflow is missing or ambiguous: $reference")
                }
            } elseif ($reference -cnotmatch '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*@[a-fA-F0-9]{40}$') {
                $errors.Add("${location}: external actions require a full commit SHA: $reference")
            }
        }
    }
    return ,$errors.ToArray()
}

function Invoke-CiWorkflowLint([string]$RepositoryRoot, [string]$Linter) {
    $errors = Test-CiActionReferences $RepositoryRoot
    if ($errors.Count) { throw ($errors -join "`n") }
    $workflows = @(Get-ChildItem -LiteralPath (Join-Path $RepositoryRoot '.github/workflows') -File |
        Where-Object Extension -cin @('.yml', '.yaml') | Sort-Object Name | ForEach-Object FullName)
    if (-not $workflows.Count) { throw 'No workflow manifests found.' }
    Push-Location -LiteralPath $RepositoryRoot
    try {
        # Keep optional script runtimes out of the preflight. actionlint still
        # validates YAML, expressions, job graphs and local composite inputs.
        & $Linter -shellcheck= -pyflakes= -no-color @workflows
        if ($LASTEXITCODE) { throw "Workflow syntax failed (exit $LASTEXITCODE)." }
    } finally { Pop-Location }
}

if ($MyInvocation.InvocationName -ne '.') {
    $version = '1.7.12'
    # Official release assets: https://github.com/rhysd/actionlint/releases/tag/v1.7.12
    $assets = @{
        'Linux-Arm64' = @('linux_arm64.tar.gz', '325e971b6ba9bfa504672e29be93c24981eeb1c07576d730e9f7c8805afff0c6')
        'Linux-X64' = @('linux_amd64.tar.gz', '8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8')
        'Windows-Arm64' = @('windows_arm64.zip', 'cadcf7ea4efe3a68728893813643cebe1185e5b1d4be5b96245f65c9a4d5ea41')
        'Windows-X64' = @('windows_amd64.zip', '6e7241b51e6817ea6a047693d8e6fed13b31819c9a0dd6c5a726e1592d22f6e9')
    }
    $scratch = $null
    try {
        if (-not $ActionlintPath) {
            $platform = if ($IsWindows) { 'Windows' } elseif ($IsLinux) { 'Linux' } else { throw 'Supply -ActionlintPath on this platform.' }
            $key = "$platform-$([Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture)"
            if (-not $assets.ContainsKey($key)) { throw "Supply -ActionlintPath on $key." }
            $asset, $digest = $assets[$key]
            $scratch = Join-Path ([IO.Path]::GetTempPath()) ('Prism-Workflow-' + [guid]::NewGuid().ToString('N'))
            [void][IO.Directory]::CreateDirectory($scratch)
            $archive = Join-Path $scratch $asset
            Invoke-WebRequest -Uri "https://github.com/rhysd/actionlint/releases/download/v$version/actionlint_${version}_$asset" -OutFile $archive -ConnectionTimeoutSeconds 10 -OperationTimeoutSeconds 20 -MaximumRetryCount 2 -RetryIntervalSec 1
            if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -cne $digest) { throw 'Pinned actionlint checksum differs.' }
            $executable = if ($IsWindows) { 'actionlint.exe' } else { 'actionlint' }
            if ($IsWindows) { Expand-Archive -LiteralPath $archive -DestinationPath $scratch }
            else { tar -xzf $archive -C $scratch $executable }
            $ActionlintPath = Join-Path $scratch $executable
        }
        $ActionlintPath = (Get-Command $ActionlintPath -CommandType Application).Source
        $identity = @(& $ActionlintPath -version)
        if ($identity[0] -cne $version) { throw "Expected actionlint $version; received $($identity[0])." }
        Invoke-CiWorkflowLint $Root $ActionlintPath
        if ($CheckFixtures) { & (Join-Path $PSScriptRoot 'Test-WorkflowControls.ps1') -ActionlintPath $ActionlintPath }
        'Workflow preflight passed: syntax, job graph, local action inputs and immutable action references.'
    } finally {
        if ($scratch) {
            $absolute = [IO.Path]::GetFullPath($scratch)
            $temporary = [IO.Path]::TrimEndingDirectorySeparator([IO.Path]::GetFullPath([IO.Path]::GetTempPath())) + [IO.Path]::DirectorySeparatorChar
            if (-not $absolute.StartsWith($temporary, [StringComparison]::OrdinalIgnoreCase) -or -not [IO.Path]::GetFileName($absolute).StartsWith('Prism-Workflow-')) { throw 'Workflow scratch cleanup escaped its temporary directory.' }
            Remove-Item -LiteralPath $absolute -Recurse -Force
        }
    }
}

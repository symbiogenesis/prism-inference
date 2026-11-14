#Requires -Version 7.6
<# Parse selected candidate scripts; run workflow checks from committed checking policy. #>
[CmdletBinding()]
[OutputType([string])]
param(
    [Parameter(Mandatory)][string]$Root,
    [Parameter(Mandatory)][string]$PathsFile
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$repository = [IO.Path]::GetFullPath($Root)
$prefix = [IO.Path]::TrimEndingDirectorySeparator($repository) + [IO.Path]::DirectorySeparatorChar
$paths = @(Get-Content -LiteralPath $PathsFile -Raw | ConvertFrom-Json)
$failed = $false
$workflows = $false
foreach ($path in $paths) {
    if ($path -isnot [string] -or -not $path -or [IO.Path]::IsPathRooted($path)) { throw 'Invalid snapshot source path.' }
    $source = [IO.Path]::GetFullPath((Join-Path $repository $path))
    if (-not $source.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Snapshot source path escaped its root.' }
    if ($path -match '\.(?:ya?ml)$' -or $path -ceq '.github/ci/Test-Workflow.ps1' -or $path -ceq '.github/ci/Check-SourceSnapshot.ps1') { $workflows = $true }
    if ([IO.Path]::GetExtension($path) -notin @('.ps1', '.psm1', '.psd1') -or -not (Test-Path -LiteralPath $source -PathType Leaf)) { continue }
    $tokens = $null
    $parseErrors = $null
    [void][Management.Automation.Language.Parser]::ParseFile($source, [ref]$tokens, [ref]$parseErrors)
    foreach ($problem in $parseErrors) {
        [Console]::Error.WriteLine("${path}:$($problem.Extent.StartLineNumber): $($problem.Message)")
        $failed = $true
    }
}
if ($failed) { throw 'Selected PowerShell syntax failed.' }
if ($workflows) {
    $parameters = @{ Root = $repository }
    if ($env:PRISM_ACTIONLINT_PATH) { $parameters.ActionlintPath = $env:PRISM_ACTIONLINT_PATH }
    $global:LASTEXITCODE = 0
    & (Join-Path $PSScriptRoot 'Test-Workflow.ps1') @parameters
    if ($LASTEXITCODE) { throw 'Snapshot workflow syntax failed.' }
}
'Selected snapshot PowerShell and workflow syntax passed.'

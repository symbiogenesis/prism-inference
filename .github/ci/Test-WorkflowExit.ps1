#Requires -Version 7.6
<# Exercise the actual PowerShell process boundary used by GitHub Actions. #>
[CmdletBinding()]
[OutputType([string])]
param([Parameter(Mandatory)][string]$ActionlintPath)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

function Invoke-WorkflowWrapper {
    [CmdletBinding()]
    [OutputType([void])]
    param(
        [Parameter(Mandatory)][string]$Body,
        [Parameter(Mandatory)][int]$ExpectedExitCode,
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Directory
    )
    $script = Join-Path $Directory ($Name + '.ps1')
    [IO.File]::WriteAllText($script, $Body)
    $quoted = $script.Replace("'", "''")
    # These are GitHub runner's PowerShell prepend/append commands. A direct
    # -File check misses the stale LASTEXITCODE failure at this boundary.
    $wrapper = "`$ErrorActionPreference = 'stop'`n. '$quoted'`nif ((Test-Path -LiteralPath variable:\LASTEXITCODE)) { exit `$LASTEXITCODE }"
    $start = [Diagnostics.ProcessStartInfo]::new((Join-Path $PSHOME 'pwsh'))
    if ($IsWindows) { $start.FileName += '.exe' }
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    foreach ($argument in @('-NoLogo', '-NoProfile', '-NonInteractive', '-Command', $wrapper)) { $start.ArgumentList.Add($argument) }
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $start
    try {
        if (-not $process.Start()) { throw "Could not start wrapper control: $Name" }
        $output = $process.StandardOutput.ReadToEndAsync()
        $errors = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(30000)) {
            $process.Kill($true)
            $process.WaitForExit()
            throw "Workflow wrapper control timed out: $Name"
        }
        $stdout = $output.GetAwaiter().GetResult()
        $stderr = $errors.GetAwaiter().GetResult()
        if ($process.ExitCode -ne $ExpectedExitCode) {
            throw "Workflow wrapper control $Name exited $($process.ExitCode), expected ${ExpectedExitCode}.`n$stdout`n$stderr"
        }
    } finally { $process.Dispose() }
}

$scratch = Join-Path ([IO.Path]::GetTempPath()) ('Prism-Workflow-Exit-' + [guid]::NewGuid().ToString('N'))
try {
    $directory = Join-Path $scratch '.github/workflows'
    [void][IO.Directory]::CreateDirectory($directory)
    [void][IO.Directory]::CreateDirectory((Join-Path $scratch '.github/actions'))
    [void][IO.Directory]::CreateDirectory((Join-Path $scratch '.git'))
    $workflow = Join-Path $directory 'check.yml'
    $valid = "name: Control`non: push`njobs:`n  check:`n    runs-on: ubuntu-24.04`n    steps:`n      - run: echo control`n"
    [IO.File]::WriteAllText($workflow, $valid)
    $checker = (Join-Path $PSScriptRoot 'Test-Workflow.ps1').Replace("'", "''")
    $controls = (Join-Path $PSScriptRoot 'Test-WorkflowControls.ps1').Replace("'", "''")
    $linter = [IO.Path]::GetFullPath($ActionlintPath).Replace("'", "''")
    $root = $scratch.Replace("'", "''")
    $arguments = @{ Directory = $scratch; ExpectedExitCode = 0 }
    Invoke-WorkflowWrapper @arguments -Name success -Body "`$global:LASTEXITCODE = 37`n& '$checker' -Root '$root' -ActionlintPath '$linter'"
    Invoke-WorkflowWrapper @arguments -Name preserved -Body "`$global:LASTEXITCODE = 37`n& '$controls' -ActionlintPath '$linter'`nif (`$LASTEXITCODE -ne 37) { throw 'Negative controls changed caller status.' }`n`$global:LASTEXITCODE = 0"
    [IO.File]::WriteAllText($workflow, $valid.Replace('    runs-on:', "    needs: absent`n    runs-on:"))
    Invoke-WorkflowWrapper -Directory $scratch -ExpectedExitCode 1 -Name rejection -Body "`$global:LASTEXITCODE = 0`n& '$checker' -Root '$root' -ActionlintPath '$linter'"
    'Workflow process controls passed: successful preflight, preserved caller status and real rejection.'
} finally {
    $absolute = [IO.Path]::GetFullPath($scratch)
    $temporary = [IO.Path]::TrimEndingDirectorySeparator([IO.Path]::GetFullPath([IO.Path]::GetTempPath())) + [IO.Path]::DirectorySeparatorChar
    if (-not $absolute.StartsWith($temporary, [StringComparison]::OrdinalIgnoreCase) -or -not [IO.Path]::GetFileName($absolute).StartsWith('Prism-Workflow-Exit-')) { throw 'Workflow process cleanup escaped temporary storage.' }
    Remove-Item -LiteralPath $absolute -Recurse -Force
}

#requires -Version 7.6
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

function Invoke-BonsaiNativeProcess {
    <#
    .SYNOPSIS
    Captures an ordinary tool's separate output streams with owned process cleanup.
    .DESCRIPTION
    Inherits the host environment and passes arguments without a shell. A zero
    timeout leaves execution unbounded; output drainage is bounded after exit.
    Nonzero exit codes are returned for the caller to interpret. Optional logs
    contain stdout followed by stderr, including when execution times out.
    Resource-limited model processes must use the separate Windows Job launcher.
    #>
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param(
        [Parameter(Mandatory)][string]$Executable,
        [AllowEmptyCollection()][AllowEmptyString()][string[]]$ArgumentList = @(),
        [Parameter(Mandatory)][string]$WorkingDirectory,
        [ValidateRange(0,86400)][int]$TimeoutSeconds = 0,
        [Collections.IDictionary]$Environment,
        [string]$LogPath
    )
    if (-not [IO.Path]::IsPathFullyQualified($WorkingDirectory) -or $WorkingDirectory -match '%[^%]+%' -or
        -not [IO.Directory]::Exists($WorkingDirectory)) {
        throw 'Native working directory must be an existing expanded absolute directory.'
    }
    if ($LogPath -and (-not [IO.Path]::IsPathFullyQualified($LogPath) -or $LogPath -match '%[^%]+%')) {
        throw 'Native log path must be an expanded absolute path.'
    }
    $start = [Diagnostics.ProcessStartInfo]::new($Executable)
    $start.WorkingDirectory = $WorkingDirectory
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    foreach ($argument in $ArgumentList) { $start.ArgumentList.Add($argument) }
    if ($Environment) {
        foreach ($key in $Environment.Keys) { $start.Environment[$key] = [string]$Environment[$key] }
    }
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $start
    $started = $false
    try {
        if (-not $process.Start()) { throw "Native process failed to start: $Executable" }
        $started = $true
        # Drain both pipes concurrently so compiler output cannot block the child.
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        $timedOut = $false
        if ($TimeoutSeconds -eq 0) { $process.WaitForExit() }
        elseif (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            $timedOut = $true
            $process.Kill($true)
            if (-not $process.WaitForExit(10000)) { throw "Native process did not terminate: $Executable" }
        }
        # A descendant may keep a redirected pipe open after the root exits.
        if (-not $stdout.Wait(10000) -or -not $stderr.Wait(10000)) { throw "Native process output drain timed out: $Executable" }
        $result = [pscustomobject]@{
            ExitCode = $process.ExitCode
            StandardOutput = $stdout.GetAwaiter().GetResult()
            StandardError = $stderr.GetAwaiter().GetResult()
        }
        if ($LogPath) { [IO.File]::WriteAllText($LogPath, $result.StandardOutput + $result.StandardError) }
        if ($timedOut) {
            $evidence = if ($LogPath) { "; see $LogPath" } else { '' }
            throw "Native process exceeded $TimeoutSeconds seconds: $Executable$evidence"
        }
        return $result
    } finally {
        try {
            if ($started -and -not $process.HasExited) {
                $process.Kill($true)
                if (-not $process.WaitForExit(10000)) { throw "Native process did not terminate during cleanup: $Executable" }
            }
        } finally { $process.Dispose() }
    }
}

Export-ModuleMember -Function Invoke-BonsaiNativeProcess

#requires -Version 7.6
<#
.SYNOPSIS
Checks native process timeouts, Stop-BonsaiProcessTree edge cases, and Rust scratch cleanup.
#>
[CmdletBinding()]
[OutputType([string])]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module (Join-Path $PSScriptRoot 'Bonsai.TestSupport.psm1')
$nativeModule = Import-Module (Join-Path $PSScriptRoot '../scripts/Bonsai.Native.psm1') -PassThru
Import-Module (Join-Path $PSScriptRoot '../scripts/Bonsai.Rust.psm1')

$scratch = Join-Path ([IO.Path]::GetTempPath()) ('Prism-FaultToleranceTest-' + [guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($scratch)

try {
    $pwshExe = (Get-Process -Id $PID).Path

    # 1. Invoke-BonsaiNativeProcess -TimeoutSeconds 1 terminates child and retains log streams.
    $timeoutLog = Join-Path $scratch 'timeout.log'
    $timer = [Diagnostics.Stopwatch]::StartNew()
    $timeoutMsg = Assert-Failure -PassThru {
        Invoke-BonsaiNativeProcess -Executable $pwshExe -ArgumentList @(
            '-NoLogo', '-NoProfile', '-Command',
            'Write-Output "before-timeout"; [Console]::Error.WriteLine("err-before-timeout"); Start-Sleep -Seconds 30'
        ) -WorkingDirectory $scratch -TimeoutSeconds 1 -LogPath $timeoutLog
    } 'Hung child process must trigger timeout failure.'
    $timer.Stop()
    Assert ($timeoutMsg -like '*exceeded 1 seconds*') "Unexpected timeout error: $timeoutMsg"
    Assert ($timer.Elapsed.TotalSeconds -lt 15) "Timeout took too long: $($timer.Elapsed.TotalSeconds)s"
    $logText = [IO.File]::ReadAllText($timeoutLog)
    Assert ($logText -match 'before-timeout') 'Timeout log omitted stdout captured before kill.'
    Assert ($logText -match 'err-before-timeout') 'Timeout log omitted stderr captured before kill.'

    # 2. Stop-BonsaiProcessTree handles already-exited process races and -WhatIf cleanly.
    $exitedProc = Start-Process -FilePath $pwshExe -ArgumentList @('-NoLogo', '-NoProfile', '-Command', 'exit 0') -PassThru -WindowStyle Hidden
    try {
        Assert ($exitedProc.WaitForExit(10000)) 'Short test child did not exit in time.'
        & $nativeModule {
            param($p)
            Stop-BonsaiProcessTree -Process $p -ErrorMessage 'whatif' -WhatIf
            Stop-BonsaiProcessTree -Process $p -ErrorMessage 'exited process race'
        } $exitedProc
    } finally {
        $exitedProc.Dispose()
    }

    $sleepProc = Start-Process -FilePath $pwshExe -ArgumentList @('-NoLogo', '-NoProfile', '-Command', 'Start-Sleep -Seconds 30') -PassThru -WindowStyle Hidden
    try {
        & $nativeModule {
            param($p)
            Stop-BonsaiProcessTree -Process $p -ErrorMessage 'whatif live' -WhatIf
        } $sleepProc
        Assert (-not $sleepProc.HasExited) 'Stop-BonsaiProcessTree -WhatIf terminated a live process.'
        & $nativeModule {
            param($p)
            Stop-BonsaiProcessTree -Process $p -ErrorMessage 'failed to terminate sleep child'
        } $sleepProc
        Assert ($sleepProc.HasExited) 'Stop-BonsaiProcessTree did not wait for exited process.'
    } finally {
        if (-not $sleepProc.HasExited) { $sleepProc.Kill($true) }
        $sleepProc.Dispose()
    }

    # 3. Invoke-BonsaiRustCheck writes error receipt and cleans up $work temp directory on failure.
    $targetDir = Join-Path $scratch 'cargo-target'
    $evidenceDir = Join-Path $scratch 'rust-evidence'
    $null = Assert-Failure -PassThru {
        Invoke-BonsaiRustCheck -Package @('prism-inference') -Phase @('clippy') `
            -Toolchain 'nonexistent-toolchain-ft-test' -TargetDirectory $targetDir -EvidenceDirectory $evidenceDir
    } 'Unknown toolchain must fail Invoke-BonsaiRustCheck.'
    $receiptPath = Join-Path $evidenceDir 'rust-checks.json'
    Assert ([IO.File]::Exists($receiptPath)) 'Failed Invoke-BonsaiRustCheck must still write rust-checks.json.'
    $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
    Assert (-not $receipt.Success) 'Failed Rust check recorded Success=true.'
    Assert (-not [string]::IsNullOrWhiteSpace($receipt.WorkingDirectory)) 'Receipt omitted WorkingDirectory.'
    Assert (-not [IO.Directory]::Exists($receipt.WorkingDirectory)) "Invoke-BonsaiRustCheck leaked work dir: $($receipt.WorkingDirectory)"

    return "Prism PowerShell fault tolerance checks passed. Evidence: $scratch"
} finally {
    Remove-Item -LiteralPath $scratch -Recurse -Force -ErrorAction SilentlyContinue
}

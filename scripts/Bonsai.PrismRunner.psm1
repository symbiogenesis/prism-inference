#requires -Version 7.6
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

function Invoke-BonsaiPrismCaseBatch {
    <#
    .SYNOPSIS
    Runs isolated Prism adapters with bounded concurrency and ordered results.
    .DESCRIPTION
    Each adapter owns its output and bridge evidence paths. Its C++ and Rust
    controls remain sequential, while independent adapters can overlap. Native
    timeout and process-tree cleanup stay owned by Invoke-BonsaiNativeProcess.
    #>
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param(
        [Parameter(Mandatory)][object[]]$Cases,
        [Parameter(Mandatory)][string]$Executable,
        [Parameter(Mandatory)][string]$EvidenceDirectory,
        [Parameter(Mandatory)][string]$UpstreamCommit,
        [Collections.IDictionary]$Environment,
        [ValidateRange(1, 12)][int]$Parallel = 2
    )
    $nativeModule = Join-Path $PSScriptRoot 'Bonsai.Native.psm1'
    # Start the adapters with the largest existing timeout budgets first. This
    # keeps the exhaustive tokenizer/architecture cases out of the queue tail.
    $completed = @($Cases | Sort-Object -Property @{ Expression = 'TimeoutSeconds'; Descending = $true }, Ordinal |
        ForEach-Object -Parallel {
            Set-StrictMode -Version Latest
            $ErrorActionPreference = 'Stop'
            $PSNativeCommandUseErrorActionPreference = $true
            Import-Module $using:nativeModule
            $adapter = $_
            $reference = $null
            $rust = $null
            $observations = [Collections.Generic.List[object]]::new()
            foreach ($command in $adapter.Commands) {
                $timer = [Diagnostics.Stopwatch]::StartNew()
                $result = Invoke-BonsaiNativeProcess -Executable $using:Executable -ArgumentList $command.Arguments -WorkingDirectory $using:EvidenceDirectory -TimeoutSeconds $adapter.TimeoutSeconds -Environment $using:Environment -LogPath $command.LogPath
                $timer.Stop()
                if ($result.ExitCode -ne 0) {
                    $detail = if ($result.StandardError.Trim()) { $result.StandardError.Trim() } else { $result.StandardOutput.Trim() }
                    $detail = $detail.Substring([Math]::Max(0, $detail.Length - 4096))
                    throw "Prism adapter $($adapter.Id) ($($command.Target)) exited $($result.ExitCode): $detail; see $($command.LogPath)"
                }
                $report = Get-Content -LiteralPath $command.OutputPath -Raw | ConvertFrom-Json -AsHashtable
                if ($report.SchemaVersion -ne 1 -or $report.UpstreamCommit -cne $using:UpstreamCommit -or
                    $report.Target -cne $command.Target -or @($report.Cases).Count -ne 1 -or
                    $report.Cases[0].Id -cne $adapter.Id) {
                    throw "Shared assertion harness returned an unexpected case identity: $($adapter.Id) ($($command.Target))"
                }
                if ($command.Target -ceq 'cpp') { $reference = $report.Cases[0] }
                else { $rust = $report.Cases[0] }
                $observations.Add([ordered]@{ Target = $command.Target; ElapsedMilliseconds = $timer.ElapsedMilliseconds; LogPath = $command.LogPath })
            }
            [pscustomobject]@{ Ordinal = $adapter.Ordinal; Id = $adapter.Id; Reference = $reference; Rust = $rust; Commands = $observations.ToArray() }
        } -ThrottleLimit $Parallel)
    # Scheduling must not change the catalog's stable presentation order.
    return $completed | Sort-Object Ordinal
}

function Invoke-BonsaiPrismTaskBatch {
    <#
    .SYNOPSIS
    Runs independent module commands with bounded concurrency and ordered results.
    .DESCRIPTION
    Each task passes explicit parameters to a fresh module session. The invoked
    command owns its absolute native working directories, evidence paths,
    validation and native process cleanup.
    #>
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param(
        [Parameter(Mandatory)][object[]]$Tasks,
        [Parameter(Mandatory)][string]$ModulePath,
        [Parameter(Mandatory)][string]$CommandName,
        [ValidateRange(1, 2)][int]$Parallel = 2
    )
    $completed = @($Tasks | ForEach-Object -Parallel {
        Set-StrictMode -Version Latest
        $ErrorActionPreference = 'Stop'
        $PSNativeCommandUseErrorActionPreference = $true
        $task = $_
        $timer = [Diagnostics.Stopwatch]::StartNew()
        $module = Import-Module -Name $using:ModulePath -PassThru -Force
        $command = Get-Command -Name $using:CommandName -Module $module.Name -CommandType Function
        $parameters = $task.Parameters
        $value = & $command @parameters
        $timer.Stop()
        [pscustomobject]@{ Ordinal = $task.Ordinal; Id = $task.Id; Value = $value; ElapsedMilliseconds = $timer.ElapsedMilliseconds }
    } -ThrottleLimit $Parallel)
    if ($completed.Count -ne $Tasks.Count) { throw 'An independent Prism task did not complete.' }
    return $completed | Sort-Object Ordinal
}

Export-ModuleMember -Function Invoke-BonsaiPrismCaseBatch, Invoke-BonsaiPrismTaskBatch

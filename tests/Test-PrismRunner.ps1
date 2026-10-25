#requires -Version 7.6
<#
.SYNOPSIS
Checks one focused Prism scheduler regression using isolated synthetic adapters.
#>
[CmdletBinding()]
[OutputType([string])]
param([Parameter(Mandatory)][ValidateSet('Scheduling', 'Identity', 'ProcessFailure')][string]$Case)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot 'Bonsai.TestSupport.psm1')
Import-Module (Join-Path $PSScriptRoot '../scripts/Bonsai.PrismRunner.psm1')
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('SwiftBonsai-PrismRunnerTest-' + [guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($scratch)
$probe = Join-Path $scratch 'adapter.ps1'
$commit = 'a' * 40
# File rendezvous proves adapters overlap without using elapsed-time assertions.
# Each adapter's Rust command also checks that its C++ control has completed.
[IO.File]::WriteAllText($probe, @'
param([string]$Id, [string]$Target, [string]$Output, [string]$Directory, [string]$Commit, [string]$Mode)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($env:BONSAI_RUNNER_SENTINEL -cne 'inherited') { throw 'The child environment was lost.' }
if ($Mode -ceq 'ProcessFailure') { [Console]::Error.WriteLine('synthetic adapter failure detail'); exit 17 }
if ($Mode -ceq 'Scheduling') {
    if ($Target -ceq 'cpp') {
        [IO.File]::WriteAllText((Join-Path $Directory "$Id-started"), '')
        $deadline = [DateTime]::UtcNow.AddSeconds(10)
        while (@([IO.Directory]::GetFiles($Directory, '*-started')).Count -lt 2) {
            if ([DateTime]::UtcNow -ge $deadline) { throw 'Independent adapters did not overlap.' }
            Start-Sleep -Milliseconds 10
        }
        [IO.File]::WriteAllText((Join-Path $Directory "$Id-cpp-complete"), '')
    } elseif (-not [IO.File]::Exists((Join-Path $Directory "$Id-cpp-complete"))) {
        throw 'A Rust command overtook its C++ control.'
    }
}
$reportedId = if ($Mode -ceq 'Identity') { 'tests/test-unexpected.cpp' } else { "tests/test-$Id.cpp" }
$report = @{ SchemaVersion = 1; UpstreamCommit = $Commit; Target = $Target; Cases = @(@{ Id = $reportedId; Status = 'pass'; Detail = 'synthetic adapter' }) }
[IO.File]::WriteAllText($Output, ($report | ConvertTo-Json -Depth 5))
'@)
$work = @(
    foreach ($ordinal in 0..1) {
        $commands = @(
            foreach ($target in @('cpp', 'rust')) {
                $output = Join-Path $scratch "$ordinal-$target.json"
                [pscustomobject]@{
                    Target = $target; OutputPath = $output; LogPath = Join-Path $scratch "$ordinal-$target.log"
                    Arguments = @('-NoLogo', '-NoProfile', '-File', $probe, '-Id', [string]$ordinal, '-Target', $target,
                        '-Output', $output, '-Directory', $scratch, '-Commit', $commit, '-Mode', $Case)
                }
            }
        )
        [pscustomobject]@{ Ordinal = $ordinal; Id = "tests/test-$ordinal.cpp"; TimeoutSeconds = 20 + $ordinal; Commands = $commands }
    }
)
$parameters = @{
    Cases = $work; Executable = (Get-Command pwsh -CommandType Application).Source
    EvidenceDirectory = $scratch; UpstreamCommit = $commit; Parallel = 2
    Environment = @{ BONSAI_RUNNER_SENTINEL = 'inherited' }
}
if ($Case -ceq 'Scheduling') {
    $results = @(Invoke-BonsaiPrismCaseBatch @parameters)
    Assert ($results.Count -eq 2 -and $results[0].Ordinal -eq 0 -and $results[1].Ordinal -eq 1) 'Concurrent completion must preserve original adapter order.'
    foreach ($ordinal in 0..1) {
        Assert ($results[$ordinal].Reference.Id -ceq "tests/test-$ordinal.cpp" -and $results[$ordinal].Rust.Id -ceq "tests/test-$ordinal.cpp") 'Both independent control identities must survive scheduling.'
    }
} else {
    $diagnostic = Assert-Failure { Invoke-BonsaiPrismCaseBatch @parameters } -PassThru
    $expected = if ($Case -ceq 'Identity') { 'unexpected case identity' } else { 'exited 17' }
    Assert ($diagnostic.Contains($expected, [StringComparison]::Ordinal)) "The scheduler did not retain its $Case failure diagnostic."
    if ($Case -ceq 'ProcessFailure') {
        Assert ($diagnostic.Contains('synthetic adapter failure detail', [StringComparison]::Ordinal)) 'The scheduler must surface the adapter stderr without downloading a large evidence archive.'
    }
}
return "Prism runner $Case passed; evidence: $scratch"

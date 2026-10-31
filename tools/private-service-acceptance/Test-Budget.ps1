#requires -Version 7.6
[CmdletBinding()]
[OutputType([string])]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot 'Budget.psm1') -Force
$now = [DateTime]::new(2026, 10, 31, 12, 0, 0, [DateTimeKind]::Utc)
$revision = 'a' * 40
$owner = @{ RunId='123'; Revision=$revision; StartedTick='1000'; StartedUtc=$now.AddSeconds(-30).ToString('o'); WorkSeconds='210' }
$base = @{ JobStartedUtc=$now.AddSeconds(-20); JobStartedTick=2000L; NowUtc=$now; NowTick=4000L; TickFrequency=100L; RunId='123'; Revision=$revision; Owner=$owner }
$result = Get-PrivateServiceBudget @base
if ($result.HardSeconds -ne 165 -or $result.SoftSeconds -ne 155 -or $result.OwnerElapsedSeconds -ne 30) { throw 'Intake was not charged against the original owner clock.' }
$cases = @(
    @{ Owner=@{} },
    @{ Owner=@{ RunId='123' } },
    @{ Owner=($owner + @{ Extra='bad' }) },
    @{ JobStartedTick=5000L },
    @{ NowTick=30000L },
    @{ NowUtc=$now.AddSeconds(-25) }
)
foreach ($field in @('RunId', 'Revision', 'StartedTick', 'StartedUtc', 'WorkSeconds')) {
    $changed = $owner.Clone(); $changed[$field] = 'wrong'
    $cases += @{ Owner=$changed }
}
foreach ($change in $cases) {
    $arguments = $base.Clone()
    foreach ($key in $change.Keys) { $arguments[$key] = $change[$key] }
    $rejected = $false
    try { $null = Get-PrivateServiceBudget @arguments } catch { $rejected = $true }
    if (-not $rejected) { throw 'Malformed, partial, stale or exhausted owner budget was admitted.' }
}
'Passed exact-owner deadline, setup charging and malformed/partial/exhausted rejection controls.'

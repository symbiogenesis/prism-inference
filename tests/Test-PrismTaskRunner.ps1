#requires -Version 7.6
<# .SYNOPSIS
Checks one focused independent-target scheduler contract with synthetic modules.
#>
[CmdletBinding()]
[OutputType([string])]
param([Parameter(Mandatory)][ValidateSet('Scheduling', 'ProcessFailure')][string]$Case)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot 'Bonsai.TestSupport.psm1')
Import-Module (Join-Path $PSScriptRoot '../scripts/Bonsai.PrismRunner.psm1')
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('SwiftBonsai-PrismTaskTest-' + [guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($scratch)
$module = Join-Path $scratch 'SyntheticPrismTask.psm1'
# The rendezvous proves overlap independently of timing. Each target's second
# mode must observe its first mode, and its module state must remain isolated.
[IO.File]::WriteAllText($module, @'
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$script:owner = $null
function Invoke-SyntheticPrismTarget {
    [CmdletBinding()]
    [OutputType([Collections.IDictionary])]
    param([string]$Id, [string]$Directory, [string]$Mode)
    if ($Mode -ceq 'ProcessFailure') { throw 'synthetic target process failure' }
    if ($script:owner) { throw 'Independent target module state was reused.' }
    $script:owner = $Id
    $results = @{}
    foreach ($phase in @('eval','support')) {
        if ($phase -ceq 'eval') {
            [IO.File]::WriteAllText((Join-Path $Directory "$Id-started"), '')
            $deadline = [DateTime]::UtcNow.AddSeconds(10)
            while (@([IO.Directory]::GetFiles($Directory, '*-started')).Count -lt 2) {
                if ([DateTime]::UtcNow -ge $deadline) { throw 'Independent target chains did not overlap.' }
                Start-Sleep -Milliseconds 10
            }
            [IO.File]::WriteAllText((Join-Path $Directory "$Id-eval-complete"), '')
        } elseif (-not [IO.File]::Exists((Join-Path $Directory "$Id-eval-complete"))) {
            throw 'A support mode overtook its eval mode.'
        }
        $results[$phase] = @{ Target=$Id; Phase=$phase; Owner=$script:owner }
    }
    return $results
}
Export-ModuleMember -Function Invoke-SyntheticPrismTarget
'@)
$tasks = @(foreach ($ordinal in 0..1) {
    $target = @('cpp','rust')[$ordinal]
    [pscustomobject]@{ Ordinal=$ordinal; Id=$target; Parameters=@{ Id=$target; Directory=$scratch; Mode=$Case } }
})
$parameters = @{ Tasks=$tasks; ModulePath=$module; CommandName='Invoke-SyntheticPrismTarget'; Parallel=2 }
if ($Case -ceq 'Scheduling') {
    $results = @(Invoke-BonsaiPrismTaskBatch @parameters)
    Assert ($results.Count -eq 2 -and $results[0].Id -ceq 'cpp' -and $results[1].Id -ceq 'rust') 'Independent target completion must retain target order.'
    foreach ($result in $results) {
        foreach ($phase in @('eval','support')) {
            Assert ($result.Value[$phase].Target -ceq $result.Id -and $result.Value[$phase].Owner -ceq $result.Id) 'Every mode must retain its target identity and isolated module state.'
        }
    }
} else {
    $diagnostic = Assert-Failure { Invoke-BonsaiPrismTaskBatch @parameters } -PassThru
    Assert ($diagnostic.Contains('synthetic target process failure', [StringComparison]::Ordinal)) 'A target failure must reach the stage caller.'
}
return "Prism task runner $Case passed; evidence: $scratch"

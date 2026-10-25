#requires -Version 7.6
<# .SYNOPSIS
Runs one revision-bound CI stage of the complete backend source matrix.
#>
[CmdletBinding()]
[OutputType([Collections.IDictionary])]
param(
    [Parameter(Mandatory)][ValidateSet('Plan','JoinPlan','Run','Gate')][string]$Phase,
    [Parameter(Mandatory)][string]$BundleDirectory,
    [Parameter(Mandatory)][string]$EvidenceDirectory,
    [string]$InputDirectory,
    [string]$PlanPath,
    [string]$OrdinaryEvidenceDirectory,
    [switch]$PartitionedOrdinary,
    [ValidateRange(0,4095)][int]$Index = 0,
    [ValidateRange(1,4096)][int]$ExecutionCount = 16,
    [switch]$Benchmarks,
    [switch]$Gradients
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
if ($env:GITHUB_ACTIONS -cne 'true' -or $env:GITHUB_REF -cne 'refs/heads/main') {
    throw 'Complete backend source stages run only in main GitHub Actions.'
}
$partitionCount = if ($Phase -cin @('Run','Gate')) { $ExecutionCount } else { 16 }
if ($Index -ge $partitionCount) { throw 'Backend stage index must be below its partition count.' }
$repo = Split-Path $PSScriptRoot -Parent
Import-Module (Join-Path $repo 'scripts/Bonsai.Common.psm1')
Import-Module (Join-Path $repo 'scripts/Bonsai.PrismBackend.psm1')
Import-Module (Join-Path $repo 'scripts/Bonsai.PrismRunner.psm1')
Import-Module (Join-Path $repo 'scripts/Bonsai.PrismBackendBundle.psm1')
Import-Module (Join-Path $repo 'scripts/Bonsai.PrismBaseline.psm1')
Import-Module (Join-Path $repo 'scripts/Bonsai.PrismSource.psm1')
$runtimeSource = Resolve-PrismSource
$evidence = Resolve-ExternalPath $EvidenceDirectory
if (Test-Path -LiteralPath $evidence) { throw 'Backend stage evidence must be fresh.' }
[void][IO.Directory]::CreateDirectory($evidence)
$bundle = Import-BonsaiPrismBackendBundle -Directory $BundleDirectory -Repository $runtimeSource
$correctnessModes = @('eval','support')
if ($Gradients) { $correctnessModes += 'grad' }
$modes = $correctnessModes
if ($Benchmarks) { $modes += 'perf' }
$contexts = @{}
foreach ($target in @('cpp','rust')) {
    $parameters = @{
        HarnessPath=$bundle.HarnessPath; HarnessSha256=$bundle.HarnessSha256
        SourceDirectory=$bundle.SourceDirectory; EvidenceDirectory=(Join-Path $evidence $target)
        Target=$target; TimeoutSeconds=18000
    }
    if ($target -ceq 'cpp') {
        $parameters.CommonBridgePath=$bundle.CommonBridgePath
        $parameters.CommonBridgeSha256=$bundle.CommonBridgeSha256
    }
    $contexts[$target] = New-BonsaiPrismBackendContext @parameters
}
function Read-BackendStage {
    <# .SYNOPSIS
    Reads one bounded same-revision stage artifact.
    #>
    [CmdletBinding()]
    [OutputType([Collections.IDictionary])]
    param([string]$Path, [string]$ExpectedPhase)
    $expectedCount = if ($ExpectedPhase -ceq 'Run') { $ExecutionCount } else { 16 }
    $path = Resolve-ExternalPath $Path
    if ((Get-Item -LiteralPath $path).Length -gt 128MB) { throw 'Backend stage artifact exceeds128MiB.' }
    $record = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json -AsHashtable -Depth 100 -DateKind String
    if ($record.SchemaVersion -ne 1 -or $record.ProjectRevision -cne $env:GITHUB_SHA -or
        $record.Phase -cne $ExpectedPhase -or $record.Count -ne $expectedCount -or
        $record.Benchmarks -isnot [bool] -or $record.Benchmarks -ne $Benchmarks.IsPresent -or
        $record.Gradients -isnot [bool] -or $record.Gradients -ne $Gradients.IsPresent -or
        ($record.Modes -join ',') -cne ($modes -join ',')) { throw 'Backend stage identity or workload selection differs.' }
    return $record
}
function Write-BackendStage {
    <# .SYNOPSIS
    Retains one bounded external stage result.
    #>
    [CmdletBinding()]
    [OutputType([void])]
    param([string]$Name, [Collections.IDictionary]$Value)
    $bytes = [Text.Encoding]::UTF8.GetBytes(($Value | ConvertTo-Json -Depth 100 -Compress -WarningAction Stop))
    if ($bytes.LongLength -gt 128MB) { throw 'Backend stage result exceeds128MiB.' }
    [IO.File]::WriteAllBytes((Join-Path $evidence $Name), $bytes)
}
$record = @{
    SchemaVersion=1; ProjectRevision=$env:GITHUB_SHA; Phase=$Phase; Count=$partitionCount
    Index=$Index; Benchmarks=$Benchmarks.IsPresent; Gradients=$Gradients.IsPresent; Modes=$modes; Targets=@{}
}
if ($Phase -cne 'Gate') {
    $parts = $null
    $plan = $null
    if ($Phase -ceq 'JoinPlan') {
        $parts = @(foreach ($part in 0..15) {
            $item = Read-BackendStage -Path (Join-Path $InputDirectory "prism-backend-plan-$part/record.json") -ExpectedPhase 'Plan'
            if ($item.Index -ne $part) { throw 'Backend plan artifact index differs.' }
            $item
        })
    } elseif ($Phase -ceq 'Run') {
        $plan = Read-BackendStage -Path $PlanPath -ExpectedPhase 'JoinPlan'
    }
    # Each target owns its context/evidence and single-threaded native graphs.
    # Overlap the independent target chains while preserving each mode's order,
    # source-order CRT offsets and all cross-job provenance checks.
    $parallelModes = if ($Phase -ceq 'Run') { $correctnessModes } else { $modes }
    $tasks = @(foreach ($ordinal in 0..1) {
        $target = @('cpp','rust')[$ordinal]
        [pscustomobject]@{
            Ordinal=$ordinal; Id=$target
            Parameters=@{ Context=$contexts[$target]; Phase=$Phase; Modes=$parallelModes; Index=$Index; Count=$partitionCount; Parts=$parts; Plan=$plan }
        }
    })
    $results = @(Invoke-BonsaiPrismTaskBatch -Tasks $tasks -ModulePath (Join-Path $repo 'scripts/Bonsai.PrismBackend.psm1') -CommandName Invoke-BonsaiPrismBackendTarget)
    $record.TargetElapsedMilliseconds = @{}
    foreach ($result in $results) {
        $record.Targets[$result.Id] = $result.Value
        $record.TargetElapsedMilliseconds[$result.Id] = $result.ElapsedMilliseconds
    }
    # Requested measurements retain exclusive use of the runner after all
    # correctness tasks finish; concurrency must not contaminate benchmark data.
    if ($Phase -ceq 'Run' -and $Benchmarks) {
        foreach ($target in @('cpp','rust')) {
            $performance = Invoke-BonsaiPrismBackendTarget -Context $contexts[$target] -Phase Run -Modes @('perf') -Index $Index -Count $partitionCount -Plan $plan
            $record.Targets[$target]['perf'] = $performance.perf
        }
    }
    $name = if ($Phase -ceq 'JoinPlan') { 'plans.json' } else { 'record.json' }
    Write-BackendStage -Name $name -Value $record
    return $record
}
$plan = Read-BackendStage -Path $PlanPath -ExpectedPhase 'JoinPlan'
$parts = @(foreach ($part in 0..($ExecutionCount - 1)) {
    $item = Read-BackendStage -Path (Join-Path $InputDirectory "prism-backend-run-$part/record.json") -ExpectedPhase 'Run'
    if ($item.Index -ne $part) { throw 'Backend execution artifact index differs.' }
    $item
})
$sources = @{}
foreach ($target in @('cpp','rust')) {
    $record.Targets[$target] = @{}
    $correctness = @(foreach ($mode in $correctnessModes) {
        $reports = @($parts | ForEach-Object { $_.Targets[$target][$mode] })
        $modeResult = Merge-BonsaiPrismBackendPartition -Context $contexts[$target] -Reports $reports -Plan $plan.Targets[$target][$mode]
        $record.Targets[$target][$mode] = $modeResult
        $modeResult
    })
    $sources[$target] = Merge-BonsaiPrismBackendSource -Context $contexts[$target] -Modes $correctness
    if ($Benchmarks) {
        $reports = @($parts | ForEach-Object { $_.Targets[$target]['perf'] })
        $record.Targets[$target]['perf'] = Merge-BonsaiPrismBackendBenchmark -Context $contexts[$target] -Reports $reports -Plan $plan.Targets[$target]['perf']
    }
}
Write-BackendStage -Name 'record.json' -Value $record
$combined = @{}
$catalog = Import-BonsaiPrismCatalog -Path (Join-Path $runtimeSource 'tools/prism/upstream/catalog.json')
$ordinarySources = $null
if ($PartitionedOrdinary) {
    Import-Module (Join-Path $repo 'scripts/Bonsai.PrismSourceResults.psm1')
    $ordinarySources = Merge-BonsaiPrismSourceResult -InputDirectory $OrdinaryEvidenceDirectory -Catalog $catalog -ProjectRevision $env:GITHUB_SHA -HarnessSha256 $bundle.HarnessSha256 -CommonBridgeSha256 $bundle.CommonBridgeSha256
    $ordinaryReceipt = $ordinarySources.Receipt
    Write-BackendStage -Name 'ordinary-partitions.json' -Value $ordinaryReceipt
} else {
    $ordinaryReceiptPath = Resolve-ExternalPath (Join-Path $OrdinaryEvidenceDirectory 'receipt.json')
    if ((Get-Item -LiteralPath $ordinaryReceiptPath).Length -gt 1MB) { throw 'Ordinary source receipt exceeds1MiB.' }
    $ordinaryReceipt = Get-Content -LiteralPath $ordinaryReceiptPath -Raw | ConvertFrom-Json -AsHashtable -Depth 100 -DateKind String
}
$ordinaryCommon = @($ordinaryReceipt.NativeBuilds | Where-Object { $_.Family -ceq 'common' })
if ($ordinaryReceipt.ProjectRevision -cne $env:GITHUB_SHA -or $ordinaryReceipt.Status -cne 'awaiting-backend' -or
    $ordinaryReceipt.DeferredSource -cne 'tests/test-backend-ops.cpp' -or
    $ordinaryReceipt.HarnessSha256 -cne $bundle.HarnessSha256 -or $ordinaryCommon.Count -ne 1 -or
    $ordinaryCommon[0].ExecutableSha256 -cne $bundle.CommonBridgeSha256) { throw 'Ordinary results are not the matching deferred source handoff.' }
foreach ($target in @('cpp','rust')) {
    if ($PartitionedOrdinary) { $ordinary = $ordinarySources[$target] }
    else {
        $path = Resolve-ExternalPath (Join-Path $OrdinaryEvidenceDirectory "$target-results.json")
        if ((Get-Item -LiteralPath $path).Length -gt 128MB) { throw 'Ordinary source result exceeds128MiB.' }
        $ordinary = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json -AsHashtable -Depth 100 -DateKind String
    }
    if (@($ordinary.Cases | Where-Object Id -CEQ 'tests/test-backend-ops.cpp').Count) { throw 'Backend source was already counted.' }
    $ordinary.Cases = @($ordinary.Cases) + @($sources[$target].Payload)
    $combined[$target] = $ordinary
    Write-BackendStage -Name "$target-results.json" -Value $ordinary
}
$report = New-BonsaiPrismReport -Catalog $catalog -TargetResults $combined.rust -ReferenceResults $combined.cpp
Write-BackendStage -Name 'report.json' -Value $report
$gate = Test-BonsaiPrismBaseline -Catalog $catalog -Report $report -BaselinePath (Join-Path $runtimeSource 'tools/prism/upstream/baseline.json')
Write-BackendStage -Name 'gate.json' -Value $gate
if (-not $gate.Passed) { throw 'The complete source gate failed; every reference and target outcome is retained.' }
if ($Benchmarks -and $record.Targets.cpp.perf.Payload.Status -cne 'measured') { throw 'The requested complete C++ backend benchmark workload did not complete.' }
return $gate

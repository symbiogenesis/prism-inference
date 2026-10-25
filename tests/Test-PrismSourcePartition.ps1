#requires -Version 7.6
<#
.SYNOPSIS
Checks ordinary Prism shard completeness and stable source identities without running workloads.
#>
[CmdletBinding()]
[OutputType([string])]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot 'Bonsai.TestSupport.psm1')
Import-Module (Join-Path $PSScriptRoot '../scripts/Bonsai.PrismSourcePartition.psm1')
Import-Module (Join-Path $PSScriptRoot '../scripts/Bonsai.PrismSource.psm1')
$runtimeSource = Resolve-PrismSource
$catalog = Get-Content -LiteralPath (Join-Path $runtimeSource 'tools/prism/upstream/catalog.json') -Raw | ConvertFrom-Json
$sources = @(
    foreach ($entry in $catalog.Cases) {
        if ($entry.Id -ceq 'tests/test-backend-ops.cpp') { continue }
        [pscustomobject]@{ Id = $entry.Id; Ordinal = 0; Selection = $entry.SourceSha256 }
    }
)
for ($ordinal = 0; $ordinal -lt $sources.Count; $ordinal++) { $sources[$ordinal].Ordinal = $ordinal }
$seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
$all = [Collections.Generic.List[object]]::new()
foreach ($index in 0..3) {
    $partition = @(Get-BonsaiPrismSourcePartition -Cases $sources -Index $index)
    Assert ($partition.Count -gt 0) 'Every hosted ordinary shard must contain catalog sources.'
    $reversed = @($sources)
    [array]::Reverse($reversed)
    $reordered = @(Get-BonsaiPrismSourcePartition -Cases $reversed -Index $index)
    Assert (($partition.Id -join ',') -ceq ($reordered.Id -join ',')) 'Input enumeration order changed a shard or its catalog presentation.'
    $previousOrdinal = -1
    foreach ($entry in $partition) {
        Assert ($seen.Add($entry.Id)) "Source $($entry.Id) appeared in more than one shard."
        Assert ([object]::ReferenceEquals($entry, $sources[$entry.Ordinal])) 'Partitioning replaced or changed the original source object.'
        Assert ($entry.Ordinal -gt $previousOrdinal) 'Shard output lost original catalog order.'
        $previousOrdinal = $entry.Ordinal
        $all.Add($entry)
    }
    $expectedHeavy = switch ($index) {
        0 { @('tests/test-tokenizer-random.py') }
        1 { @('tests/test-thread-safety.cpp') }
        2 { @('tests/test-quantize-stats.cpp', 'tests/test-grammar-llguidance.cpp') }
        3 { @() }
    }
    foreach ($id in $expectedHeavy) { Assert ($id -cin $partition.Id) "Heavy source $id moved from its measured shard." }
    if ($index -lt 3) { Assert ($partition.Count -eq @($expectedHeavy).Count) 'An unrelated source entered a dedicated heavy shard.' }
}
Assert ($seen.Count -eq $sources.Count -and $all.Count -eq $sources.Count) 'Ordinary partitioning omitted catalog sources.'
foreach ($entry in $sources) { Assert ($seen.Contains($entry.Id)) "Ordinary source $($entry.Id) was omitted." }
$future = [pscustomobject]@{ Id = 'tests/test-future-source.cpp'; Ordinal = $sources.Count; Selection = 'future' }
Assert (@(Get-BonsaiPrismSourcePartition -Cases @($sources + $future) -Index 3).Count -eq
    @(Get-BonsaiPrismSourcePartition -Cases $sources -Index 3).Count + 1) 'A newly cataloged source did not enter the remaining-source shard.'
Assert (@(Get-BonsaiPrismSourcePartition -Cases @() -Index 0).Count -eq 0) 'An empty selection invented source work.'
Assert-Failure { Get-BonsaiPrismSourcePartition -Cases $sources -Index 4 } 'An out-of-range shard index was accepted.'
Assert-Failure { Get-BonsaiPrismSourcePartition -Cases @($sources[0], $sources[0]) -Index 3 } 'Duplicate source identity was accepted.'
$duplicateOrdinal = [pscustomobject]@{ Id = 'tests/test-other.cpp'; Ordinal = $sources[0].Ordinal }
Assert-Failure { Get-BonsaiPrismSourcePartition -Cases @($sources[0], $duplicateOrdinal) -Index 3 } 'Duplicate catalog ordinal was accepted.'
$badOrdinal = [pscustomobject]@{ Id = 'tests/test-other.cpp'; Ordinal = -1 }
Assert-Failure { Get-BonsaiPrismSourcePartition -Cases @($badOrdinal) -Index 3 } 'A negative catalog ordinal was accepted.'
$blankId = [pscustomobject]@{ Id = ''; Ordinal = 0 }
Assert-Failure { Get-BonsaiPrismSourcePartition -Cases @($blankId) -Index 3 } 'An empty source identity was accepted.'
foreach ($ordinal in @('0', 0.0, [long]0)) {
    $mistypedOrdinal = [pscustomobject]@{ Id = 'tests/test-other.cpp'; Ordinal = $ordinal }
    Assert-Failure { Get-BonsaiPrismSourcePartition -Cases @($mistypedOrdinal) -Index 3 } 'A non-Int32 PowerShell ordinal was accepted.'
}
$numericId = [pscustomobject]@{ Id = 42; Ordinal = 0 }
Assert-Failure { Get-BonsaiPrismSourcePartition -Cases @($numericId) -Index 3 } 'A numeric source identity was accepted.'
$characterId = [pscustomobject]@{ Id = [char]'x'; Ordinal = 0 }
Assert-Failure { Get-BonsaiPrismSourcePartition -Cases @($characterId) -Index 3 } 'A character source identity was accepted as JSON text.'
Assert-Failure { Get-BonsaiPrismSourcePartition -Cases @($null) -Index 3 } 'A null source object was accepted.'
return "Prism ordinary source partitions cover all $($sources.Count) catalog sources exactly once in stable order."

#requires -Version 7.6
<# .SYNOPSIS
Checks complete, revision-bound fan-in using synthetic ordinary source outcomes.
#>
[CmdletBinding()]
[OutputType([string])]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot 'Bonsai.TestSupport.psm1')
Import-Module (Join-Path $PSScriptRoot '../scripts/Bonsai.PrismSourceResults.psm1')
Import-Module (Join-Path $PSScriptRoot '../scripts/Bonsai.PrismBaseline.psm1')
Import-Module (Join-Path $PSScriptRoot '../scripts/Bonsai.PrismSourcePartition.psm1')
Import-Module (Join-Path $PSScriptRoot '../scripts/Bonsai.PrismSource.psm1')
$runtimeSource = Resolve-PrismSource
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('SwiftBonsai-PrismSourceResults-' + [guid]::NewGuid().ToString('N'))
$catalog = Import-BonsaiPrismCatalog -Path (Join-Path $runtimeSource 'tools/prism/upstream/catalog.json')
$cases = @(for ($ordinal = 0; $ordinal -lt $catalog.AdapterIds.Count; $ordinal++) {
    if ($catalog.AdapterIds[$ordinal] -cne 'tests/test-backend-ops.cpp') {
        [pscustomobject]@{ Id=$catalog.AdapterIds[$ordinal]; Ordinal=$ordinal }
    }
})
$revision = 'a' * 40
$harness = 'b' * 64
$bridge = 'c' * 64
foreach ($index in 0..3) {
    $directory = Join-Path $scratch "prism-ordinary-results-$index"
    [void][IO.Directory]::CreateDirectory($directory)
    $receipt = @{
        SchemaVersion=1; ProjectRevision=$revision; SourceCommit=$catalog.UpstreamCommit
        SourceArchiveSha256=$catalog.ArchiveSha256; Status='awaiting-backend'; Selection='All'
        SourcePartition=$index; SourcePartitionCount=4; NativePlatform='ARM64'; Architecture='Arm64'
        DeferredSource='tests/test-backend-ops.cpp'; HarnessSha256=$harness
        NativeBuilds=@(@{ Family='common'; ExecutableSha256=$bridge }); CaseRuns=@(@{ Partition=$index })
        AdditionalReceipt=@{ Retained=$true }
    }
    [IO.File]::WriteAllText((Join-Path $directory 'receipt.json'), ($receipt | ConvertTo-Json -Depth 10))
    foreach ($target in @('cpp','rust')) {
        $report = @{
            SchemaVersion=1; UpstreamCommit=$catalog.UpstreamCommit
            Cases=@(Get-BonsaiPrismSourcePartition -Cases $cases -Index $index | ForEach-Object {
                @{ Id=$_.Id; Status=if ($target -ceq 'cpp') { 'pass' } else { 'fail' }; Detail='Synthetic source result.'; ExtraCase=@{ Retained=$true } }
            })
        }
        [IO.File]::WriteAllText((Join-Path $directory "$target-results.json"), ($report | ConvertTo-Json -Depth 10))
    }
}
$parameters = @{ InputDirectory=$scratch; Catalog=$catalog; ProjectRevision=$revision; HarnessSha256=$harness; CommonBridgeSha256=$bridge }
$merged = Merge-BonsaiPrismSourceResult @parameters
foreach ($target in @('cpp','rust')) {
    Assert ($merged[$target].Cases.Count -eq $cases.Count) 'The source fan-in dropped an ordinary source.'
    Assert (($merged[$target].Cases.Id -join ',') -ceq ($cases.Id -join ',')) 'The source fan-in lost catalog order.'
    Assert (@($merged[$target].Cases | Where-Object { -not $_.ExtraCase.Retained }).Count -eq 0) 'The source fan-in lost additional case evidence.'
}
Assert (@($merged.rust.Cases | Where-Object Status -CEQ 'fail').Count -eq $cases.Count) 'Fan-in changed Rust failures into passes.'
Assert ($merged.Receipt.SourcePartitions.Count -eq 4) 'The merged receipt omitted partition provenance.'
Assert ($merged.Receipt.AdditionalReceipt.Retained) 'The merged receipt lost partition zero evidence.'
Assert (($merged.Receipt.CaseRuns.Partition -join ',') -ceq '0,1,2,3') 'The merged receipt lost or nested the case runs.'
foreach ($index in 0..3) {
    $receiptDigest = (Get-FileHash -LiteralPath (Join-Path $scratch "prism-ordinary-results-$index/receipt.json") -Algorithm SHA256).Hash.ToLowerInvariant()
    Assert ($merged.Receipt.SourcePartitions[$index].ReceiptSha256 -ceq $receiptDigest) 'The merged receipt hash differs from the parsed artifact bytes.'
}
$otherReceiptPath = Join-Path $scratch 'prism-ordinary-results-1/receipt.json'
$otherOriginal = [IO.File]::ReadAllText($otherReceiptPath)
$otherReceipt = $otherOriginal | ConvertFrom-Json -AsHashtable
$otherReceipt.HarnessSha256 = 'd' * 64
$otherReceipt.NativeBuilds[0].ExecutableSha256 = 'e' * 64
[IO.File]::WriteAllText($otherReceiptPath, ($otherReceipt | ConvertTo-Json -Depth 10))
[void](Merge-BonsaiPrismSourceResult @parameters)
[IO.File]::WriteAllText($otherReceiptPath, $otherOriginal)
$receiptPath = Join-Path $scratch 'prism-ordinary-results-0/receipt.json'
$original = [IO.File]::ReadAllText($receiptPath)
$deep = 'survives'
for ($level = 0; $level -lt 97; $level++) { $deep = @{ Nested=$deep } }
$changed = $original | ConvertFrom-Json -AsHashtable
$changed.CaseRuns[0]['Deep'] = $deep
[IO.File]::WriteAllText($receiptPath, ($changed | ConvertTo-Json -Depth 100))
$deepMerged = Merge-BonsaiPrismSourceResult @parameters
$retained = $deepMerged.Receipt.SourcePartitions[0].CaseRuns[0].Deep
for ($level = 0; $level -lt 97; $level++) { $retained = $retained.Nested }
Assert ($retained -ceq 'survives') 'Fan-in lost evidence at the supported source artifact depth.'
[IO.File]::WriteAllText($receiptPath, $original)
foreach ($mutation in @(
    @{ Field='SchemaVersion'; Value='1' },
    @{ Field='SchemaVersion'; Value=1.0 },
    @{ Field='ProjectRevision'; Value=('d' * 40) },
    @{ Field='SourcePartition'; Value=1 },
    @{ Field='SourcePartition'; Value='0' },
    @{ Field='SourcePartitionCount'; Value=3 },
    @{ Field='SourcePartitionCount'; Value='4' },
    @{ Field='SourcePartitionCount'; Value=4.0 },
    @{ Field='Status'; Value='building' },
    @{ Field='SourceCommit'; Value=('d' * 40) },
    @{ Field='SourceArchiveSha256'; Value=('d' * 64) },
    @{ Field='HarnessSha256'; Value=('d' * 64) },
    @{ Field='CaseRuns'; Value=@{} },
    @{ Field='NativeBuilds'; Value=@() }
)) {
    $changed = $original | ConvertFrom-Json -AsHashtable
    $changed[$mutation.Field] = $mutation.Value
    [IO.File]::WriteAllText($receiptPath, ($changed | ConvertTo-Json -Depth 10))
    Assert-Failure { Merge-BonsaiPrismSourceResult @parameters } "Source fan-in accepted invalid $($mutation.Field)."
}
[IO.File]::WriteAllText($receiptPath, $original)
$changed = $original | ConvertFrom-Json -AsHashtable
$changed.NativeBuilds[0].ExecutableSha256 = 'd' * 64
[IO.File]::WriteAllText($receiptPath, ($changed | ConvertTo-Json -Depth 10))
Assert-Failure { Merge-BonsaiPrismSourceResult @parameters } 'Source fan-in accepted a different partition zero backend bridge.'
[IO.File]::WriteAllText($receiptPath, $original)
$resultPath = Join-Path $scratch 'prism-ordinary-results-3/rust-results.json'
$original = [IO.File]::ReadAllText($resultPath)
foreach ($mode in @('missing','duplicate','misplaced','foreign-revision','mistyped-schema','nonarray-cases')) {
    $changed = $original | ConvertFrom-Json -AsHashtable
    switch ($mode) {
        'missing' { $changed.Cases = @($changed.Cases | Select-Object -Skip 1) }
        'duplicate' { $changed.Cases[1] = $changed.Cases[0] }
        'misplaced' { $changed.Cases[0].Id = 'tests/test-tokenizer-random.py' }
        'foreign-revision' { $changed.UpstreamCommit = 'd' * 40 }
        'mistyped-schema' { $changed.SchemaVersion = '1' }
        'nonarray-cases' { $changed.Cases = @{} }
    }
    [IO.File]::WriteAllText($resultPath, ($changed | ConvertTo-Json -Depth 10))
    Assert-Failure { Merge-BonsaiPrismSourceResult @parameters } "Source fan-in accepted $mode results."
}
[IO.File]::WriteAllText($resultPath, $original)
[IO.File]::Move($receiptPath, $receiptPath + '.missing')
Assert-Failure { Merge-BonsaiPrismSourceResult @parameters } 'Source fan-in accepted a missing partition.'
return "Prism source fan-in passed; synthetic evidence: $scratch"

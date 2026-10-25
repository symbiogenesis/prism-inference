#requires -Version 7.6
<#
.SYNOPSIS
Runs one focused regression group for the imported Prism test baseline gate.
#>
[CmdletBinding()]
[OutputType([string])]
param([Parameter(Mandatory)][ValidateSet('Catalog', 'Ratchet', 'Evidence')][string]$Case)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot 'Bonsai.TestSupport.psm1')
Import-Module (Join-Path $PSScriptRoot '../scripts/Bonsai.PrismBaseline.psm1') -Force
Import-Module (Join-Path $PSScriptRoot '../scripts/Bonsai.PrismSource.psm1')
$scratch = [IO.Path]::GetFullPath((Join-Path ([IO.Path]::GetTempPath()) ('SwiftBonsai-PrismBaselineTest-' + [guid]::NewGuid().ToString('N'))))
[void][IO.Directory]::CreateDirectory($scratch)
$catalogPath = Join-Path $scratch 'catalog.json'
$baselinePath = Join-Path $scratch 'baseline.json'
$commit = 'a' * 40
$ids = @('tests/test-one.cpp', 'tests/test-two.cpp', 'tests/nested/test-one.cpp')
$catalog = [ordered]@{
    SchemaVersion = 1; UpstreamCommit = $commit; ArchiveSha256 = 'b' * 64; TestCMakeSourceSha256 = 'd' * 64
    Measure = 'upstream-source-files'; AdapterIds = @($ids[0])
    Cases = @($ids | ForEach-Object { [ordered]@{ Id = $_; SourceSha256 = 'c' * 64 } })
}
$baseline = [ordered]@{
    SchemaVersion = 1; UpstreamCommit = $commit; Measure = 'upstream-source-files'; Total = 3
    MinimumPassed = 1; RequiredPassedIds = @($ids[0]); RequiredReferencePassedIds = @($ids[0], $ids[2])
}
$catalog | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $catalogPath -Encoding utf8
$catalog = Import-BonsaiPrismCatalog -Path $catalogPath
$baseline.CatalogSha256 = $catalog.CatalogSha256
$baseline | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $baselinePath -Encoding utf8

function New-TestResult {
    <# .SYNOPSIS
    Creates a sparse runner report for the selected synthetic source IDs.
    #>
    [CmdletBinding()]
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseShouldProcessForStateChangingFunctions', '', Justification = 'Constructs only an in-memory synthetic runner report; no external state changes.')]
    [OutputType([Collections.Specialized.OrderedDictionary])]
    param([string[]]$Ids = @(), [string[]]$Statuses = @())
    return [ordered]@{
        SchemaVersion = 1; UpstreamCommit = $commit
        Cases = @(for ($i = 0; $i -lt $Ids.Count; $i++) { [ordered]@{ Id = $Ids[$i]; Status = $Statuses[$i]; Detail = 'Synthetic gate regression.' } })
    }
}

switch ($Case) {
    'Catalog' {
        $imported = Import-BonsaiPrismCatalog -Path $catalogPath
        Assert ($imported.Cases.Count -eq 3) 'Catalog IDs must preserve nested paths.'
        $metadata = @{ Leaf='Preserve metadata beyond the bridge serialization depth.' }
        for ($depth = 0; $depth -lt 70; $depth++) { $metadata = @{ Nested=$metadata } }
        $extended = Get-Content -LiteralPath $catalogPath -Raw | ConvertFrom-Json -AsHashtable -DateKind String
        $extended.Extra = $metadata
        $extended.Cases[0].Extra = $metadata
        $extendedPath = Join-Path $scratch 'extended-catalog.json'
        $extended | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $extendedPath -Encoding utf8
        $preserved = Import-BonsaiPrismCatalog -Path $extendedPath
        foreach ($extra in @($preserved.Extra, $preserved.Cases[0].Extra)) {
            $leaf = $extra
            for ($depth = 0; $depth -lt 70; $depth++) { $leaf = $leaf.Nested }
            Assert ($leaf.Leaf -ceq 'Preserve metadata beyond the bridge serialization depth.') 'Catalog metadata was truncated by the bridge.'
        }
        Assert ($preserved.CatalogSha256 -ceq $imported.CatalogSha256) 'Unrelated metadata changed the canonical catalog identity.'
        $target = New-TestResult -Ids @($ids[0]) -Statuses @('pass')
        $reference = New-TestResult -Ids $ids -Statuses @('pass', 'pass', 'pass')
        $target.Cases[0].Coverage = 'x' * (17 * 1024 * 1024)
        $report = New-BonsaiPrismReport -Catalog $preserved -TargetResults $target -ReferenceResults $reference
        $report.Summary.Extra = $target.Cases[0].Coverage
        Assert ((Test-BonsaiPrismBaseline -Catalog $preserved -Report $report -BaselinePath $baselinePath).Passed) 'Ignored diagnostic metadata must not exceed the bridge policy-request limit.'
        $invalid = New-TestResult -Ids @($ids[0]) -Statuses @('pass')
        $invalid.Cases = @([pscustomobject]@{ Id=$ids[0]; Status='pass'; Detail='A non-dictionary input must remain invalid.' })
        Assert-Failure { New-BonsaiPrismReport -Catalog $preserved -TargetResults $invalid -ReferenceResults $reference } 'Serialization repaired an invalid non-dictionary result.'
        $invalid.Cases = [Collections.Generic.List[object]]::new()
        $invalid.Cases.Add(@{ Id=$ids[0]; Status='pass'; Detail='A list must not become a valid array.' })
        Assert-Failure { New-BonsaiPrismReport -Catalog $preserved -TargetResults $invalid -ReferenceResults $reference } 'Serialization repaired an invalid Cases collection.'
        $invalid = New-TestResult -Ids @($ids[0]) -Statuses @('pass')
        $invalid.Cases[0].Detail = [DateTime]::UtcNow
        Assert-Failure { New-BonsaiPrismReport -Catalog $preserved -TargetResults $invalid -ReferenceResults $reference } 'Serialization repaired a non-string result detail.'
        $runtimeSource = Resolve-PrismSource
        $real = Import-BonsaiPrismCatalog -Path (Join-Path $runtimeSource 'tools/prism/upstream/catalog.json')
        Assert ($real.Cases.Count -eq 71) 'Pinned upstream inventory must include all 71 test source files.'
        Assert (@($real.Cases | Where-Object { $_.Id -ceq 'tests/test-tokenizer-0.cpp' }).Count -eq 1) 'C++ tokenizer ID is absent.'
        Assert (@($real.Cases | Where-Object { $_.Id -ceq 'tests/test-tokenizer-0.py' }).Count -eq 1) 'Python tokenizer ID must remain separate.'
        Assert (@($real.Cases | Where-Object { $_.Id -ceq 'tests/peg-parser/test-unicode.cpp' }).Count -eq 1) 'Nested parser source was omitted.'
        $migration = Get-Content -Raw -LiteralPath (Join-Path $runtimeSource 'tools/prism/upstream/migration.json') | ConvertFrom-Json -AsHashtable
        $required = Get-Content -Raw -LiteralPath (Join-Path $runtimeSource 'tools/prism/upstream/baseline.json') | ConvertFrom-Json -AsHashtable
        Assert ($migration.SchemaVersion -eq 1 -and $migration.UpstreamCommit -ceq $real.UpstreamCommit -and
            $migration.Total -eq $real.Cases.Count -and $migration.Cases.Count -eq $real.Cases.Count) 'Migration inventory must cover the exact pinned catalog.'
        $migrationIds = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
        foreach ($entry in $migration.Cases) {
            Assert ($migrationIds.Add($entry.Id)) 'Migration inventory contains a duplicate source.'
            $original = @($real.Cases | Where-Object { $_.Id -ceq $entry.Id -and $_.SourceSha256 -ceq $entry.SourceSha256 })
            Assert ($original.Count -eq 1) 'Migration source identity or digest drifted.'
            Assert ($entry.SafeRustPort -is [bool] -and $entry.SafeRustPort -eq ($real.AdapterIds -ccontains $entry.Id)) 'Migration port status must match the executable adapters.'
            Assert ($entry.LegacyReferenceRequired -is [bool] -and $entry.LegacyReferenceRequired -eq ($required.RequiredReferencePassedIds -ccontains $entry.Id)) 'Migration inventory lost a required reference control.'
            Assert ($entry.Needs -is [array] -and $entry.Needs.Count -gt 0 -and -not [string]::IsNullOrWhiteSpace($entry.AssertionsToPort)) 'Migration scope and prerequisites must remain explicit.'
        }
        Assert ($migration.Target.Platform -ceq 'Windows ARM64' -and $migration.Target.Hardware -ceq 'Snapdragon X Elite') 'Migration acceptance must retain the selected target.'
        Assert ($migration.Target.ExcludedSources.Count -eq 1 -and $migration.Target.ExcludedSources[0].Id -ceq 'tests/test-rset-release.cpp') 'Only the explicitly excluded Apple wired-memory source is outside the target.'
        Assert ($migrationIds.Contains($migration.Target.ExcludedSources[0].Id) -and $real.AdapterIds -cnotcontains $migration.Target.ExcludedSources[0].Id) 'The excluded source must remain in inventory and cannot become a passing adapter.'
        $sourceRoot = Join-Path $scratch 'source'
        [void][IO.Directory]::CreateDirectory((Join-Path $sourceRoot 'tests/nested'))
        foreach ($id in $ids) {
            [IO.File]::WriteAllText((Join-Path $sourceRoot $id), 'synthetic upstream source')
        }
        $cmakePath = Join-Path $sourceRoot 'tests/CMakeLists.txt'
        [IO.File]::WriteAllText($cmakePath, 'synthetic upstream targets')
        $catalog.TestCMakeSourceSha256 = (Get-FileHash -LiteralPath $cmakePath -Algorithm SHA256).Hash.ToLowerInvariant()
        foreach ($source in $catalog.Cases) { $source.SourceSha256 = (Get-FileHash -LiteralPath (Join-Path $sourceRoot $source.Id) -Algorithm SHA256).Hash.ToLowerInvariant() }
        $catalog | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $catalogPath -Encoding utf8
        [void](Import-BonsaiPrismCatalog -Path $catalogPath -SourceDirectory $sourceRoot)
        $extraPath = Join-Path $sourceRoot 'tests/test-unlisted.cpp'
        [IO.File]::WriteAllText($extraPath, 'unlisted')
        Assert-Failure { Import-BonsaiPrismCatalog -Path $catalogPath -SourceDirectory $sourceRoot } 'A source omitted from the inventory was accepted.'
        [IO.File]::Delete($extraPath)
        [IO.File]::Delete((Join-Path $sourceRoot $ids[2]))
        Assert-Failure { Import-BonsaiPrismCatalog -Path $catalogPath -SourceDirectory $sourceRoot } 'A missing upstream source was accepted.'
        [IO.File]::WriteAllText((Join-Path $sourceRoot $ids[2]), 'changed upstream source')
        Assert-Failure { Import-BonsaiPrismCatalog -Path $catalogPath -SourceDirectory $sourceRoot } 'A modified upstream source was accepted.'
        [IO.File]::WriteAllText((Join-Path $sourceRoot $ids[2]), 'synthetic upstream source')
        [IO.File]::WriteAllText($cmakePath, 'changed upstream targets')
        Assert-Failure { Import-BonsaiPrismCatalog -Path $catalogPath -SourceDirectory $sourceRoot } 'Changed upstream test registrations were accepted.'
        $catalog.Cases += $catalog.Cases[0]
        $catalog | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $catalogPath -Encoding utf8
        Assert-Failure { Import-BonsaiPrismCatalog -Path $catalogPath } 'Duplicate catalog ID was accepted.'
        $catalog.Cases = $catalog.Cases[0..2]
        $catalog.Cases[0].Id = '../tests/test-escape.cpp'
        $catalog | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $catalogPath -Encoding utf8
        Assert-Failure { Import-BonsaiPrismCatalog -Path $catalogPath } 'Escaping source path was accepted.'
    }
    'Ratchet' {
        $target = New-TestResult -Ids @($ids[0]) -Statuses @('pass')
        $reference = New-TestResult -Ids $ids -Statuses @('pass', 'pass', 'pass')
        $report = New-BonsaiPrismReport -Catalog $catalog -TargetResults $target -ReferenceResults $reference
        $gate = Test-BonsaiPrismBaseline -Catalog $catalog -Report $report -BaselinePath $baselinePath
        Assert ($gate.Passed -and $gate.Summary.Total -eq 3 -and $gate.Summary.Passed -eq 1 -and $gate.Summary.Unsupported -eq 2) 'Sparse adapters must retain unported entries as red.'
        Assert ($gate.Summary.GreenRatio -eq (1 / 3)) 'The denominator must include all upstream files.'
        $reducedReference = New-TestResult -Ids @($ids[0], $ids[1]) -Statuses @('pass', 'pass')
        $reducedReport = New-BonsaiPrismReport -Catalog $catalog -TargetResults $target -ReferenceResults $reducedReference
        $reducedGate = Test-BonsaiPrismBaseline -Catalog $catalog -Report $reducedReport -BaselinePath $baselinePath
        Assert (-not $reducedGate.Passed -and $reducedGate.Summary.Passed -eq 1) 'Removing a C++ reference-only control must fail even when Rust coverage is unchanged.'
        Assert (@($reducedGate.Issues | Where-Object { $_ -like 'Previously passing C++ reference regressed:*' }).Count -eq 1) 'Missing previous C++ green coverage must have a separate diagnostic.'
        foreach ($status in @('fail', 'skip', 'unsupported')) {
            $redReference = New-TestResult -Ids $ids -Statuses @('pass', 'pass', $status)
            $redReport = New-BonsaiPrismReport -Catalog $catalog -TargetResults $target -ReferenceResults $redReference
            Assert (-not (Test-BonsaiPrismBaseline -Catalog $catalog -Report $redReport -BaselinePath $baselinePath).Passed) "Previously passing C++ reference status $status bypassed the reference floor."
        }
        $target = New-TestResult -Ids @($ids[0], $ids[1]) -Statuses @('fail', 'pass')
        $report = New-BonsaiPrismReport -Catalog $catalog -TargetResults $target -ReferenceResults $reference
        $gate = Test-BonsaiPrismBaseline -Catalog $catalog -Report $report -BaselinePath $baselinePath
        Assert (-not $gate.Passed) 'A different new green case must not replace a previously green case.'
        foreach ($status in @('fail', 'skip')) {
            $target = New-TestResult -Ids @($ids[0]) -Statuses @($status)
            $report = New-BonsaiPrismReport -Catalog $catalog -TargetResults $target -ReferenceResults $reference
            $report.Summary.Passed = 3
            $gate = Test-BonsaiPrismBaseline -Catalog $catalog -Report $report -BaselinePath $baselinePath
            Assert (-not $gate.Passed -and $gate.Summary.Passed -eq 0) "Red status $status or a forged presentation summary bypassed the minimum."
            Assert (@($gate.Issues | Where-Object { $_ -like 'Green ratio fell below*' }).Count -eq 1) 'The ratio floor must reject a regression independently.'
        }
        $target = New-TestResult -Ids @($ids[0], $ids[1]) -Statuses @('pass', 'pass')
        $report = New-BonsaiPrismReport -Catalog $catalog -TargetResults $target -ReferenceResults $reference
        Assert ((Test-BonsaiPrismBaseline -Catalog $catalog -Report $report -BaselinePath $baselinePath).Passed) 'New green cases must be accepted without rewriting the policy.'
        $reference = New-TestResult -Ids @($ids[0]) -Statuses @('fail')
        $report = New-BonsaiPrismReport -Catalog $catalog -TargetResults $target -ReferenceResults $reference
        $gate = Test-BonsaiPrismBaseline -Catalog $catalog -Report $report -BaselinePath $baselinePath
        Assert (-not $gate.Passed -and $gate.Summary.ReferenceFailureIds.Count -eq 2) 'Failed and missing C++ reference results must both reject a green target.'
    }
    'Evidence' {
        $valid = New-TestResult -Ids @($ids[0]) -Statuses @('pass')
        $empty = New-TestResult
        Assert-Failure { New-BonsaiPrismReport -Catalog $catalog -TargetResults $empty -ReferenceResults $valid } 'A missing implemented target adapter was accepted.'
        Assert-Failure { New-BonsaiPrismReport -Catalog $catalog -TargetResults $valid -ReferenceResults $empty } 'A missing implemented C++ adapter was accepted.'
        $invalid = New-TestResult -Ids @($ids[0]) -Statuses @('unsupported')
        Assert-Failure { New-BonsaiPrismReport -Catalog $catalog -TargetResults $invalid -ReferenceResults $valid } 'An implemented adapter was allowed to disappear as unsupported.'
        $invalid = New-TestResult -Ids @($ids[0], $ids[0]) -Statuses @('pass', 'fail')
        Assert-Failure { New-BonsaiPrismReport -Catalog $catalog -TargetResults $invalid -ReferenceResults $valid } 'Duplicate target IDs were accepted.'
        Assert-Failure { New-BonsaiPrismReport -Catalog $catalog -TargetResults $valid -ReferenceResults $invalid } 'Duplicate reference IDs were accepted.'
        $invalid = New-TestResult -Ids @('tests/test-unknown.cpp') -Statuses @('pass')
        Assert-Failure { New-BonsaiPrismReport -Catalog $catalog -TargetResults $invalid -ReferenceResults $valid } 'Unknown target ID was accepted.'
        Assert-Failure { New-BonsaiPrismReport -Catalog $catalog -TargetResults $valid -ReferenceResults $invalid } 'Unknown reference ID was accepted.'
        $invalid = New-TestResult -Ids @($ids[0]) -Statuses @('PASS')
        Assert-Failure { New-BonsaiPrismReport -Catalog $catalog -TargetResults $invalid -ReferenceResults $valid } 'Wrong-case status was accepted.'
        $invalid = New-TestResult -Ids @($ids[0]) -Statuses @('pass')
        $invalid.UpstreamCommit = 'd' * 40
        Assert-Failure { New-BonsaiPrismReport -Catalog $catalog -TargetResults $invalid -ReferenceResults $valid } 'Mismatched upstream pin was accepted.'
        $invalid.UpstreamCommit = @($commit)
        Assert-Failure { New-BonsaiPrismReport -Catalog $catalog -TargetResults $invalid -ReferenceResults $valid } 'An array coerced into the upstream pin was accepted.'
        $invalid = New-TestResult -Ids @($ids[0]) -Statuses @('pass')
        $invalid.SchemaVersion = '1'
        Assert-Failure { New-BonsaiPrismReport -Catalog $catalog -TargetResults $invalid -ReferenceResults $valid } 'A string schema version was accepted.'
        $invalid = New-TestResult -Ids @($ids[0]) -Statuses @('pass')
        $invalid.Remove('Cases')
        Assert-Failure { New-BonsaiPrismReport -Catalog $catalog -TargetResults $invalid -ReferenceResults $valid } 'A missing Cases array was accepted.'
        $report = New-BonsaiPrismReport -Catalog $catalog -TargetResults $valid -ReferenceResults $valid
        $report.Cases = @($report.Cases[0], $report.Cases[0], $report.Cases[2])
        Assert-Failure { Test-BonsaiPrismBaseline -Catalog $catalog -Report $report -BaselinePath $baselinePath } 'Duplicate full-report records were accepted.'
        $report = New-BonsaiPrismReport -Catalog $catalog -TargetResults $valid -ReferenceResults $valid
        $report.Cases = @($report.Cases[0])
        Assert-Failure { Test-BonsaiPrismBaseline -Catalog $catalog -Report $report -BaselinePath $baselinePath } 'Missing full-report records were accepted.'
        $report = New-BonsaiPrismReport -Catalog $catalog -TargetResults $valid -ReferenceResults $valid
        $baseline.RequiredPassedIds = @($ids[0], $ids[0])
        $baseline | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $baselinePath -Encoding utf8
        Assert-Failure { Test-BonsaiPrismBaseline -Catalog $catalog -Report $report -BaselinePath $baselinePath } 'Duplicate baseline green IDs were accepted.'
        $baseline.RequiredPassedIds = @($ids[0])
        $baseline.Total = 2
        $baseline | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $baselinePath -Encoding utf8
        Assert-Failure { Test-BonsaiPrismBaseline -Catalog $catalog -Report $report -BaselinePath $baselinePath } 'An altered denominator was accepted.'
        $baseline.Total = 3
        $baseline.RequiredReferencePassedIds = @($ids[0], $ids[0])
        $baseline | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $baselinePath -Encoding utf8
        Assert-Failure { Test-BonsaiPrismBaseline -Catalog $catalog -Report $report -BaselinePath $baselinePath } 'Duplicate baseline C++ reference green IDs were accepted.'
        $baseline.RequiredReferencePassedIds = @('tests/test-unknown.cpp')
        $baseline | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $baselinePath -Encoding utf8
        Assert-Failure { Test-BonsaiPrismBaseline -Catalog $catalog -Report $report -BaselinePath $baselinePath } 'Unknown baseline C++ reference green IDs were accepted.'
        $baseline.RequiredReferencePassedIds = $ids[0]
        $baseline | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $baselinePath -Encoding utf8
        Assert-Failure { Test-BonsaiPrismBaseline -Catalog $catalog -Report $report -BaselinePath $baselinePath } 'A scalar C++ reference baseline ID list was accepted.'
        $baseline.RequiredReferencePassedIds = @($ids[0], $ids[2])
        $baseline.CatalogSha256 = '0' * 64
        $baseline | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $baselinePath -Encoding utf8
        Assert-Failure { Test-BonsaiPrismBaseline -Catalog $catalog -Report $report -BaselinePath $baselinePath } 'Baseline policy for a different catalog was accepted.'
    }
}
Write-Output "Focused Prism baseline $Case checks passed. Evidence: $scratch"

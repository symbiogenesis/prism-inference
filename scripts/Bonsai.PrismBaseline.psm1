#requires -Version 7.6
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot 'Bonsai.PrismPolicy.psm1')

function Get-BonsaiPrismPolicyInput {
    <# .SYNOPSIS
    Projects policy fields without serializing unrelated caller metadata.
    #>
    [CmdletBinding()]
    [OutputType([hashtable])]
    param([AllowNull()][object]$Value, [string[]]$Fields, [string[]]$CaseFields = @())
    if ($Value -isnot [Collections.IDictionary]) { return $null }
    $inputValue = @{}
    foreach ($key in $Fields) {
        if (-not $Value.Contains($key)) { continue }
        $field = $Value[$key]
        # Preserve the CLR input contract where JSON serialization could repair
        # a list, date, enum or arbitrary object into an otherwise valid value.
        $inputValue[$key] = $null
        if ($key -cin @('Cases', 'AdapterIds', 'RequiredPassedIds', 'RequiredReferencePassedIds')) {
            if ($field -is [array]) {
                $inputValue[$key] = if ($key -ceq 'Cases') { ,$field } else {
                    ,@(foreach ($item in $field) { if ($item -is [string]) { $item } else { ,$null } })
                }
            }
        } elseif ($key -cin @('SchemaVersion', 'Total', 'MinimumPassed')) {
            if ($field -is [int] -or $field -is [long]) { $inputValue[$key] = $field }
        } elseif ($field -is [string]) { $inputValue[$key] = $field }
    }
    if ($CaseFields.Count -and $inputValue.ContainsKey('Cases') -and $inputValue.Cases -is [array]) {
        $inputValue.Cases = @(foreach ($case in $inputValue.Cases) {
            ,(Get-BonsaiPrismPolicyInput -Value $case -Fields $CaseFields)
        })
    }
    return $inputValue
}

function Import-BonsaiPrismCatalog {
    <# .SYNOPSIS
    Reads the fixed inventory and delegates its identity and source checks to Rust.
    #>
    [CmdletBinding()]
    [OutputType([Collections.IDictionary])]
    param([Parameter(Mandatory)][string]$Path, [string]$SourceDirectory)
    $catalog = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -AsHashtable -DateKind String
    $request = @{
        Operation='catalog'
        Catalog=(Get-BonsaiPrismPolicyInput -Value $catalog -Fields @('SchemaVersion', 'UpstreamCommit', 'ArchiveSha256', 'TestCMakeSourceSha256', 'Measure', 'AdapterIds', 'Cases') -CaseFields @('Id', 'SourceSha256'))
    }
    if ($SourceDirectory) {
        if (-not [IO.Path]::IsPathFullyQualified($SourceDirectory) -or $SourceDirectory -match '%[^%]+%') {
            throw 'SourceDirectory must be an expanded absolute path.'
        }
        $sourceRoot = [IO.Path]::GetFullPath($SourceDirectory)
        $request.SourceInventory = @(foreach ($file in (Get-ChildItem -LiteralPath (Join-Path $sourceRoot 'tests') -Recurse -File)) {
            if ($file.Name -cnotmatch '^test-.*\.(?:cpp|c|py|sh)$') { continue }
            @{
                Id=[IO.Path]::GetRelativePath($sourceRoot, $file.FullName).Replace('\', '/')
                Sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            }
        })
        $request.TestCMakeSourceSha256 = (Get-FileHash -LiteralPath (Join-Path $sourceRoot 'tests/CMakeLists.txt') -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $verified = Invoke-PrismPolicy -Command prism-baseline -Request $request
    $catalog.CatalogSha256 = $verified.CatalogSha256
    return $catalog
}

function New-BonsaiPrismReport {
    <# .SYNOPSIS
    Expands sparse target/reference results through the Rust baseline policy.
    #>
    [CmdletBinding()]
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseShouldProcessForStateChangingFunctions', '', Justification = 'Constructs an evidence report through the bounded maintenance bridge.')]
    [OutputType([Collections.Specialized.OrderedDictionary])]
    param([Parameter(Mandatory)][Collections.IDictionary]$Catalog,
          [Parameter(Mandatory)][Collections.IDictionary]$TargetResults,
          [Parameter(Mandatory)][Collections.IDictionary]$ReferenceResults)
    $report = Invoke-PrismPolicy -Command prism-baseline -Request @{
        Operation='report'
        Catalog=(Get-BonsaiPrismPolicyInput -Value $Catalog -Fields @('SchemaVersion', 'UpstreamCommit', 'ArchiveSha256', 'TestCMakeSourceSha256', 'Measure', 'AdapterIds', 'Cases') -CaseFields @('Id', 'SourceSha256'))
        TargetResults=(Get-BonsaiPrismPolicyInput -Value $TargetResults -Fields @('SchemaVersion', 'UpstreamCommit', 'Cases') -CaseFields @('Id', 'Status', 'Detail'))
        ReferenceResults=(Get-BonsaiPrismPolicyInput -Value $ReferenceResults -Fields @('SchemaVersion', 'UpstreamCommit', 'Cases') -CaseFields @('Id', 'Status', 'Detail'))
    }
    return [ordered]@{
        SchemaVersion=$report.SchemaVersion; UpstreamCommit=$report.UpstreamCommit; CatalogSha256=$report.CatalogSha256
        Measure=$report.Measure; Cases=$report.Cases; Summary=$report.Summary
    }
}

function Test-BonsaiPrismBaseline {
    <# .SYNOPSIS
    Checks the recorded target/reference floor through Rust without changing it.
    .DESCRIPTION
    Malformed evidence throws. Regressions return Passed=false and diagnostics.
    Rust recomputes every count from source records, ignoring presentation totals.
    #>
    [CmdletBinding()]
    [OutputType([Collections.Specialized.OrderedDictionary])]
    param([Parameter(Mandatory)][Collections.IDictionary]$Catalog,
          [Parameter(Mandatory)][Collections.IDictionary]$Report,
          [Parameter(Mandatory)][string]$BaselinePath)
    $gate = Invoke-PrismPolicy -Command prism-baseline -Request @{
        Operation='baseline'
        Catalog=(Get-BonsaiPrismPolicyInput -Value $Catalog -Fields @('SchemaVersion', 'UpstreamCommit', 'ArchiveSha256', 'TestCMakeSourceSha256', 'Measure', 'AdapterIds', 'Cases') -CaseFields @('Id', 'SourceSha256'))
        Report=(Get-BonsaiPrismPolicyInput -Value $Report -Fields @('SchemaVersion', 'UpstreamCommit', 'CatalogSha256', 'Measure', 'Cases') -CaseFields @('Id', 'SourceSha256', 'Status', 'Detail', 'ReferenceStatus', 'ReferenceDetail'))
        Baseline=(Get-BonsaiPrismPolicyInput -Value (Get-Content -LiteralPath $BaselinePath -Raw | ConvertFrom-Json -AsHashtable -DateKind String) -Fields @('SchemaVersion', 'UpstreamCommit', 'CatalogSha256', 'Measure', 'Total', 'MinimumPassed', 'RequiredPassedIds', 'RequiredReferencePassedIds'))
    }
    return [ordered]@{
        Passed=$gate.Passed; Issues=$gate.Issues; Summary=$gate.Summary
        MinimumPassed=$gate.MinimumPassed; BaselineTotal=$gate.BaselineTotal
    }
}

Export-ModuleMember -Function Import-BonsaiPrismCatalog, New-BonsaiPrismReport, Test-BonsaiPrismBaseline

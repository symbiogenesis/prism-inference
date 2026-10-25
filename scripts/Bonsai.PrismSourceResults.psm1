#requires -Version 7.6
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot 'Bonsai.Common.psm1')
Import-Module (Join-Path $PSScriptRoot 'Bonsai.PrismPolicy.psm1')

function Merge-BonsaiPrismSourceResult {
    <# .SYNOPSIS
    Requires all four same-revision ordinary partitions exactly once on both targets.
    .DESCRIPTION
    Rust verifies bounded artifacts, provenance and complete partition membership.
    Partition zero supplies the backend tools; other partitions may have different
    native feature selections and executable digests. Every original source result
    and the receipt hashes of the exact parsed bytes survive the merge.
    #>
    [CmdletBinding()]
    [OutputType([Collections.IDictionary])]
    param(
        [Parameter(Mandatory)][string]$InputDirectory,
        [Parameter(Mandatory)][Collections.IDictionary]$Catalog,
        [Parameter(Mandatory)][string]$ProjectRevision,
        [Parameter(Mandatory)][string]$HarnessSha256,
        [Parameter(Mandatory)][string]$CommonBridgeSha256
    )
    return Invoke-PrismPolicy -Command prism-source -Request @{
        Operation='merge'; InputDirectory=(Resolve-ExternalPath $InputDirectory); Catalog=$Catalog
        ProjectRevision=$ProjectRevision; HarnessSha256=$HarnessSha256; CommonBridgeSha256=$CommonBridgeSha256
    }
}

Export-ModuleMember -Function Merge-BonsaiPrismSourceResult

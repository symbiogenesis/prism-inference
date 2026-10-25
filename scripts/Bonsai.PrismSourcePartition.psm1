#requires -Version 7.6
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot 'Bonsai.PrismPolicy.psm1')

function Get-BonsaiPrismSourcePartition {
    <#
    .SYNOPSIS
    Selects one deterministic shard of the complete ordinary Prism catalog.
    .DESCRIPTION
    Rust owns shard membership and validates unique source IDs and ordinals.
    Returning the selected original PowerShell objects preserves arbitrary caller
    properties and object identity. Unknown future sources enter the final shard.
    #>
    [CmdletBinding()]
    [OutputType([object])]
    param(
        [Parameter(Mandatory)][AllowEmptyCollection()][object[]]$Cases,
        [Parameter(Mandatory)][ValidateRange(0, 3)][int]$Index
    )
    $originals = [Collections.Generic.Dictionary[int,object]]::new()
    $identities = @(foreach ($source in $Cases) {
        # Preserve the caller's string/Int32 contract across the JSON boundary.
        if ($null -eq $source -or $source.Id -isnot [string] -or $source.Ordinal -isnot [int]) {
            throw 'Each ordinary source requires a string Id and integer Ordinal.'
        }
        if (-not $originals.TryAdd($source.Ordinal, $source)) { throw 'Ordinary source Ordinals must be unique.' }
        @{ Id=$source.Id; Ordinal=$source.Ordinal }
    })
    $selection = Invoke-PrismPolicy -Command prism-source -Request @{
        Operation='partition'; Cases=$identities; Index=$Index
    }
    foreach ($ordinal in $selection.Ordinals) { $originals[[int]$ordinal] }
}

Export-ModuleMember -Function Get-BonsaiPrismSourcePartition

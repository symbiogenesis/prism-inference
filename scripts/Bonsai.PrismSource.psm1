#requires -Version 7.6
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

function Resolve-PrismSource {
    <# .SYNOPSIS
    Returns this checked-out Prism revision for runtime-owned conformance tests.
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param()
    $root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
    foreach ($relative in @('Cargo.toml','Cargo.lock','tools/prism/source.lock.json','tools/prism/upstream/catalog.json')) {
        if (-not [IO.File]::Exists((Join-Path $root $relative))) { throw "Incomplete local Prism source checkout: $relative" }
    }
    return $root
}

Export-ModuleMember -Function Resolve-PrismSource
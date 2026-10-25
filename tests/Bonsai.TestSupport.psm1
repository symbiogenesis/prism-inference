#requires -Version 7.6
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

function Assert {
    <# .SYNOPSIS
    Fails a focused regression with its caller-supplied explanation.
    #>
    [CmdletBinding()]
    [OutputType([void])]
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}

function Assert-Failure {
    <# .SYNOPSIS
    Requires an invalid-input action to throw, optionally returning its diagnostic.
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param([scriptblock]$Action, [string]$Message = 'Expected invalid input to fail.', [switch]$PassThru)
    try { & $Action | Out-Null } catch {
        if ($PassThru) { return $_.Exception.Message }
        return
    }
    throw $Message
}

Export-ModuleMember -Function Assert, Assert-Failure

#requires -Version 7.6
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

function Get-PrivateServiceBudget {
    <# .SYNOPSIS
    Pure deadline calculation; the complete trusted owner binding is mandatory.
    #>
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param(
        [Parameter(Mandatory)][DateTime]$JobStartedUtc,
        [Parameter(Mandatory)][long]$JobStartedTick,
        [Parameter(Mandatory)][DateTime]$NowUtc,
        [Parameter(Mandatory)][long]$NowTick,
        [Parameter(Mandatory)][long]$TickFrequency,
        [Parameter(Mandatory)][string]$RunId,
        [Parameter(Mandatory)][string]$Revision,
        [Collections.IDictionary]$Owner = @{}
    )
    if ($RunId -cnotmatch '^[1-9][0-9]{0,19}$' -or $Revision -cnotmatch '^[a-f0-9]{40}$' -or
        $JobStartedUtc.Kind -ne [DateTimeKind]::Utc -or $NowUtc.Kind -ne [DateTimeKind]::Utc -or
        $JobStartedUtc -gt $NowUtc -or $JobStartedTick -le 0 -or $NowTick -lt $JobStartedTick -or $TickFrequency -le 0) {
        throw 'Invalid private-service clock or owning execution identity.'
    }
    $fields = @('RunId', 'Revision', 'StartedTick', 'StartedUtc', 'WorkSeconds')
    if ($null -eq $Owner -or $Owner.Count -ne $fields.Count) { throw 'The complete trusted owner deadline is required.' }
    foreach ($field in $fields) {
        if (-not $Owner.Contains($field) -or $Owner[$field] -isnot [string] -or [string]::IsNullOrEmpty($Owner[$field])) {
            throw 'The complete trusted owner deadline is required.'
        }
    }
    if ($Owner.RunId -cne $RunId -or $Owner.Revision -cne $Revision -or $Owner.WorkSeconds -cne '210') {
        throw 'Trusted owner deadline does not bind this exact main run and fixed allowance.'
    }
    $ownerTick = 0L
    $ownerUtc = [DateTime]::MinValue
    if ($Owner.StartedTick -cnotmatch '^[1-9][0-9]{0,18}$' -or
        -not [long]::TryParse($Owner.StartedTick, [ref]$ownerTick) -or $ownerTick -gt $JobStartedTick -or
        $Owner.StartedUtc -cnotmatch '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{7}Z$' -or
        -not [DateTime]::TryParseExact($Owner.StartedUtc, 'o', [Globalization.CultureInfo]::InvariantCulture,
            [Globalization.DateTimeStyles]::RoundtripKind, [ref]$ownerUtc) -or
        $ownerUtc.Kind -ne [DateTimeKind]::Utc -or $ownerUtc -gt $JobStartedUtc) {
        throw 'Trusted owner start is malformed, future or later than job setup.'
    }
    $jobElapsed = ($NowTick - $JobStartedTick) / [double]$TickFrequency
    $ownerElapsed = ($NowTick - $ownerTick) / [double]$TickFrequency
    # Fifteen seconds remain before the external 210-second work cutoff. The
    # outer owner retains thirty seconds for process teardown within its 240-
    # second lifetime, plus at most sixty seconds for receipt intake: 300 total.
    $hard = [Math]::Floor([Math]::Min(270 - $jobElapsed, 210 - $ownerElapsed - 15))
    $soft = [Math]::Min(240, $hard - 10)
    if ($soft -lt 21) { throw 'Setup exhausted the bounded service allowance; qualification remains pending.' }
    [pscustomobject]@{ JobElapsedSeconds=$jobElapsed; OwnerElapsedSeconds=$ownerElapsed; HardSeconds=[int]$hard; SoftSeconds=[int]$soft }
}

Export-ModuleMember -Function Get-PrivateServiceBudget

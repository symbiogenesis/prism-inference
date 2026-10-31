#requires -Version 7.6
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

function Open-PrivateServicePackage {
    <# .SYNOPSIS
    Verify a same-run hosted package and retain write/delete-denying file guards.
    #>
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseShouldProcessForStateChangingFunctions', '', Justification = 'Opens read-only verification handles without filesystem mutation.')]
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param(
        [Parameter(Mandatory)][string]$Package,
        [Parameter(Mandatory)][Collections.IDictionary]$Expected
    )
    $guards = [Collections.Generic.List[IO.FileStream]]::new()
    try {
        $members = @(Get-ChildItem -LiteralPath $Package -Force)
        $names = @('private-service-check.exe', 'private-service-owner.exe', 'receipt.json')
        if ($members.Count -ne $names.Count -or (Get-Item -LiteralPath $Package).Attributes.HasFlag([IO.FileAttributes]::ReparsePoint)) {
            throw 'Hosted service package has an unexpected inventory or redirected directory.'
        }
        foreach ($name in $names) {
            $member = @($members | Where-Object Name -CEQ $name)
            if ($member.Count -ne 1 -or $member[0].PSIsContainer -or $member[0].Attributes.HasFlag([IO.FileAttributes]::ReparsePoint) -or
                $member[0].Length -le 0 -or $member[0].Length -gt 64MB) { throw 'Hosted service member is invalid.' }
            $guards.Add([IO.FileStream]::new($member[0].FullName, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read))
        }
        if (($members | Measure-Object Length -Sum).Sum -gt 64MB -or $guards[2].Length -gt 64KB) { throw 'Hosted service package exceeds its bound.' }
        $reader = [IO.StreamReader]::new($guards[2], [Text.UTF8Encoding]::new($false, $true), $true, 4096, $true)
        try { $receiptText = $reader.ReadToEnd() } finally { $reader.Dispose() }
        $receipt = $receiptText | ConvertFrom-Json -AsHashtable
        $fields = @('schemaVersion', 'repository', 'revision', 'gitRef', 'workflowPath', 'runId', 'runAttempt',
            'architecture', 'modelExecuted', 'scope', 'runnerOwnerRevision', 'ownerSourceSha256', 'compilerIdentity', 'files')
        if ($receipt.Count -ne $fields.Count -or @($receipt.Keys | Where-Object { $_ -cnotin $fields }).Count) {
            throw 'Hosted service receipt fields differ.'
        }
        foreach ($field in @('repository', 'revision', 'gitRef', 'workflowPath', 'runId', 'architecture', 'scope', 'runnerOwnerRevision', 'ownerSourceSha256')) {
            if ($receipt[$field] -isnot [string] -or $receipt[$field] -cne $Expected[$field]) { throw 'Hosted service source, owner or run identity differs.' }
        }
        if ($receipt.schemaVersion -ne 2 -or [string]$receipt.runAttempt -cne $Expected.runAttempt -or
            $receipt.modelExecuted -isnot [bool] -or $receipt.modelExecuted -ne $false -or
            $receipt.compilerIdentity -isnot [string] -or [string]::IsNullOrWhiteSpace($receipt.compilerIdentity) -or @($receipt.files).Count -ne 2) {
            throw 'Hosted service source, owner or run identity differs.'
        }
        for ($index = 0; $index -lt 2; $index++) {
            $name = $names[$index]
            $file = @($receipt.files | Where-Object { $_.name -ceq $name })
            $stream = $guards[$index]
            if ($file.Count -ne 1 -or $file[0].Count -ne 3 -or @($file[0].Keys | Where-Object { $_ -cnotin @('name', 'bytes', 'sha256') }).Count -or
                $file[0].bytes -ne $stream.Length -or $file[0].sha256 -cnotmatch '^[a-f0-9]{64}$' -or
                [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($stream)).ToLowerInvariant() -cne $file[0].sha256) {
                throw 'Hosted service executable digest differs.'
            }
            $stream.Position = 0
            $binary = [IO.BinaryReader]::new($stream, [Text.Encoding]::UTF8, $true)
            try {
                if ($binary.ReadUInt16() -ne 0x5a4d -or $stream.Length -lt 64) { throw 'Hosted service executable lacks its PE header.' }
                $stream.Position = 0x3c
                $offset = $binary.ReadUInt32()
                if ($offset -lt 64 -or $offset -gt $stream.Length - 6) { throw 'Hosted service PE header is out of range.' }
                $stream.Position = $offset
                if ($binary.ReadUInt32() -ne 0x4550 -or $binary.ReadUInt16() -ne 0xaa64) { throw 'Hosted service executable must be native ARM64.' }
            } finally { $binary.Dispose() }
        }
        return [pscustomobject]@{ Guards=$guards; ReceiptText=$receiptText }
    } catch {
        foreach ($guard in $guards) { $guard.Dispose() }
        throw
    }
}

Export-ModuleMember -Function Open-PrivateServicePackage

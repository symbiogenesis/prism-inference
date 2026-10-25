#requires -Version 7.6
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot 'Bonsai.Common.psm1')

function Initialize-BonsaiPrismReferencePatch {
    <#
    .SYNOPSIS
    Creates explicit, byte-verified production fixes outside the original tree.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([hashtable])]
    param([Parameter(Mandatory)][string]$Repository,
          [Parameter(Mandatory)][string]$SourceDirectory,
          [Parameter(Mandatory)][string]$Directory)
    $source = Resolve-ExternalPath $SourceDirectory
    $output = Resolve-ExternalPath $Directory
    if ($output.Equals($source, [StringComparison]::OrdinalIgnoreCase) -or
        $output.StartsWith($source.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Production source substitutions must remain outside the verified source tree.'
    }
    $lock = Get-Content -LiteralPath (Join-Path $Repository 'tools/prism/upstream/reference-patches.lock.json') -Raw | ConvertFrom-Json -AsHashtable
    if (($lock.SchemaVersion -isnot [int] -and $lock.SchemaVersion -isnot [long]) -or $lock.SchemaVersion -ne 2 -or
        $lock.SourceCommit -isnot [string] -or $lock.SourceCommit -cne 'b87ca7225456b06b0975b8239c66b5aecf07e37c' -or
        $lock.Variant -isnot [string] -or
        $lock.Variant -cne 'pinned-upstream-with-documented-production-fixes' -or
        $lock.Patches -isnot [array] -or $lock.Patches.Count -ne 2) {
        throw 'Unexpected reference patch identity.'
    }
    $identities = [Collections.Generic.Dictionary[string, string]]::new([StringComparer]::Ordinal)
    $identities.Add('src/llama-model-saver.cpp', 'tools/prism/upstream/patches/dspark-saver.patch')
    $identities.Add('src/llama-vocab.cpp', 'tools/prism/upstream/patches/llama3-cleanup.patch')
    $prepared = [Collections.Generic.List[object]]::new()
    $newline = [string][char]10
    $crlf = [string][char]13 + $newline
    foreach ($entry in $lock.Patches) {
        if ($entry.RelativeSource -isnot [string] -or $entry.PatchPath -isnot [string] -or
            -not $identities.ContainsKey($entry.RelativeSource) -or
            $entry.PatchPath -cne $identities[$entry.RelativeSource]) {
            throw 'Unexpected or duplicate reference patch source.'
        }
        [void]$identities.Remove($entry.RelativeSource)
        foreach ($field in @('BeforeSha256', 'AfterSha256', 'CanonicalPatchSha256')) {
            if ($entry[$field] -isnot [string] -or $entry[$field] -cnotmatch '^[0-9a-f]{64}$') {
                throw 'Reference patch digest must be a lowercase SHA-256 string.'
            }
        }
        $original = Join-Path $source $entry.RelativeSource
        $before = (Get-FileHash -LiteralPath $original -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($before -cne $entry.BeforeSha256) { throw 'Reference patch input differs from its exact upstream pin.' }
        $patch = [IO.File]::ReadAllText((Join-Path $Repository $entry.PatchPath)).Replace($crlf, $newline)
        $patchHash = [Convert]::ToHexStringLower([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($patch)))
        if ($patchHash -cne $entry.CanonicalPatchSha256) { throw 'Reference patch content differs from its explicit pin.' }
        $lines = $patch.Split($newline)
        if ($lines.Count -lt 4 -or $lines[0] -cne ('--- a/' + $entry.RelativeSource) -or
            $lines[1] -cne ('+++ b/' + $entry.RelativeSource) -or $lines[-1].Length -ne 0) {
            throw 'Reference patch file identity or final newline differs.'
        }
        $originalLines = [IO.File]::ReadAllText($original).Replace($crlf, $newline).Split($newline)
        $result = [Collections.Generic.List[string]]::new()
        $oldIndex = 0
        $patchIndex = 2
        $hunks = 0
        while ($patchIndex -lt $lines.Count - 1) {
            $header = [regex]::Match($lines[$patchIndex], '^@@ -([1-9][0-9]*),([0-9]+) \+([1-9][0-9]*),([0-9]+) @@(?: .*)?$')
            if (-not $header.Success) { throw 'Invalid reference patch hunk header.' }
            $start = [int]$header.Groups[1].Value - 1
            $expectedOld = [int]$header.Groups[2].Value
            $newStart = [int]$header.Groups[3].Value - 1
            $expectedNew = [int]$header.Groups[4].Value
            if ($start -lt $oldIndex -or $start -ge $originalLines.Count) { throw 'Reference patch hunk order or extent differs.' }
            while ($oldIndex -lt $start) { $result.Add($originalLines[$oldIndex]); $oldIndex++ }
            if ($result.Count -ne $newStart) { throw 'Reference patch new hunk position differs.' }
            $patchIndex++
            $oldCount = 0
            $newCount = 0
            while ($patchIndex -lt $lines.Count - 1 -and -not $lines[$patchIndex].StartsWith('@@', [StringComparison]::Ordinal)) {
                $line = $lines[$patchIndex]
                if ($line.Length -lt 1 -or $line[0] -cnotin @(' ', '+', '-')) { throw 'Invalid reference patch hunk line.' }
                $value = $line.Substring(1)
                if ($line[0] -ne '+') {
                    if ($oldIndex -ge $originalLines.Count -or $originalLines[$oldIndex] -cne $value) { throw 'Reference patch context differs.' }
                    $oldIndex++
                    $oldCount++
                }
                if ($line[0] -ne '-') { $result.Add($value); $newCount++ }
                $patchIndex++
            }
            if ($oldCount -ne $expectedOld -or $newCount -ne $expectedNew) { throw 'Reference patch hunk extent differs.' }
            $hunks++
        }
        if ($hunks -eq 0) { throw 'Reference patch has no hunks.' }
        for ($i = $oldIndex; $i -lt $originalLines.Count; $i++) { $result.Add($originalLines[$i]) }
        $bytes = [Text.Encoding]::UTF8.GetBytes($result -join $newline)
        $after = [Convert]::ToHexStringLower([Security.Cryptography.SHA256]::HashData($bytes))
        if ($after -cne $entry.AfterSha256) { throw 'Patched production source differs from its reviewed byte identity.' }
        $destination = Join-Path $output ([IO.Path]::GetFileName($entry.RelativeSource))
        $prepared.Add(@{
            Bytes=$bytes
            Receipt=@{ Path=$destination; RelativeSource=$entry.RelativeSource; BeforeSha256=$before; AfterSha256=$after; CanonicalPatchSha256=$patchHash }
        })
    }
    foreach ($item in $prepared) {
        $destination = $item.Receipt.Path
        if (Test-Path -LiteralPath $destination) {
            Test-LockedArtifact -Path $destination -Bytes $item.Bytes.Length -Sha256 $item.Receipt.AfterSha256
        } else {
            if (-not $PSCmdlet.ShouldProcess($destination, 'Write verified production fix outside the unchanged source tree')) { return }
            [void][IO.Directory]::CreateDirectory($output)
            [IO.File]::WriteAllBytes($destination, $item.Bytes)
        }
    }
    return @{ Variant=$lock.Variant; Patches=@($prepared | ForEach-Object { $_.Receipt }) }
}

Export-ModuleMember -Function Initialize-BonsaiPrismReferencePatch

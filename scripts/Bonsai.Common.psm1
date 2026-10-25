#requires -Version 7.6
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

function Resolve-ExternalPath {
    <#
    .SYNOPSIS
    Validates expanded absolute external storage without Git or reparse ancestors.
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param([Parameter(Mandatory)][string]$Path)
    # Windows normalizes forward slashes before recognizing device namespaces.
    # Check the normalized spelling before probing any filesystem path.
    $namespacePath = $Path.Replace('/', '\')
    if ($Path -match '%[^%]+%' -or -not [IO.Path]::IsPathFullyQualified($Path) -or
        $namespacePath.StartsWith('\\?\') -or $namespacePath.StartsWith('\\.\')) {
        throw 'Use an absolute path with all environment variables already expanded.'
    }
    $full = [IO.Path]::GetFullPath($Path)
    if ($full.Substring([IO.Path]::GetPathRoot($full).Length).Contains(':')) { throw 'Alternate data streams are unsupported.' }
    # Reject repository storage and aliases that could redirect a write into it.
    $cursor = $full
    while ($cursor) {
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Reparse-point storage is unsupported: $cursor"
            }
            if ($item.PSIsContainer -and (Test-Path -LiteralPath (Join-Path $cursor '.git'))) {
                throw 'Artifact, report, and scratch paths must be outside Git repositories.'
            }
        }
        $parent = [IO.Path]::GetDirectoryName($cursor.TrimEnd([IO.Path]::DirectorySeparatorChar))
        if ($parent -eq $cursor) { break }
        $cursor = $parent
    }
    return $full
}

function Read-BonsaiLock {
    <#
    .SYNOPSIS
    Reads and validates immutable model and runtime artifact identities.
    #>
    [CmdletBinding()]
    [OutputType([Collections.IDictionary])]
    param([string]$Path = (Join-Path $PSScriptRoot '../config/runtime.lock.json'))
    $lock = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -DateKind String -AsHashtable -NoEnumerate
    if ($lock -isnot [Collections.IDictionary]) { throw 'The lock file must contain a JSON object.' }
    if ($lock.schemaVersion -ne 1 -or $lock.model.revision -notmatch '^[a-f0-9]{40}$' -or
        $lock.model.repository -notmatch '^[\w.-]+/[\w.-]+$' -or
        $lock.runtime.commit -notmatch '^[a-f0-9]{40}$' -or
        $lock.runtime.repository -notmatch '^https://github\.com/[\w.-]+/[\w.-]+\.git$') {
        throw 'Invalid or mutable model/runtime identity in lock file.'
    }
    foreach ($artifact in @($lock.model, $lock.runtime.asset)) {
        if ($artifact.filename -notmatch '^[\w.-]+$' -or
            $artifact.sha256 -notmatch '^[a-f0-9]{64}$' -or
            ($artifact.bytes -isnot [long] -and $artifact.bytes -isnot [int]) -or $artifact.bytes -le 0) {
            throw 'Each locked artifact needs a simple filename, exact byte length, and SHA-256.'
        }
    }
    $assetUri = [Uri]$lock.runtime.asset.url
    if (-not $assetUri.IsAbsoluteUri -or $assetUri.Scheme -ne 'https' -or
        $assetUri.Host -ne 'github.com' -or $assetUri.UserInfo -or $assetUri.Query -or
        $assetUri.AbsolutePath -notmatch '^/[\w.-]+/[\w.-]+/releases/download/[^/]+/[\w.-]+$') {
        throw 'Runtime asset must be a pinned HTTPS GitHub release download.'
    }
    if ($lock.runtime.release -notmatch '^[\w.-]+$') { throw 'Invalid runtime release name.' }
    $expectedUrl = $lock.runtime.repository -replace '\.git$', ''
    $expectedUrl += "/releases/download/$($lock.runtime.release)/$($lock.runtime.asset.filename)"
    if ($assetUri.AbsoluteUri -cne $expectedUrl) { throw 'Runtime repository, release, and asset URL disagree.' }
    return $lock
}

function Assert-LockedArtifactStream {
    [CmdletBinding()]
    [OutputType([void])]
    param([Parameter(Mandatory)][IO.FileStream]$Stream, [Parameter(Mandatory)][string]$Path,
          [Parameter(Mandatory)][long]$Bytes, [Parameter(Mandatory)][string]$Sha256)
    # Length and digest must describe the same open file, with no writer admitted.
    if ($Stream.Length -ne $Bytes) { throw "Artifact byte length mismatch: $Path" }
    $Stream.Position = 0
    $actualHash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($Stream))
    if ($actualHash -ine $Sha256) {
        throw "Artifact SHA-256 mismatch; remove the invalid file before retrying: $Path"
    }
}

function Test-LockedArtifact {
    <#
    .SYNOPSIS
    Rejects an artifact whose exact byte length or SHA-256 differs from its lock.
    #>
    [CmdletBinding()]
    [OutputType([void])]
    param([Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)][long]$Bytes,
          [Parameter(Mandatory)][string]$Sha256)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Artifact is missing: $Path" }
    # Windows share modes reject existing or new writers and renames for the
    # entire verification. SequentialScan helps cache a multi-gigabyte digest.
    $stream = [IO.FileStream]::new($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read,
        [IO.FileShare]::Read, 65536, [IO.FileOptions]::SequentialScan)
    try { Assert-LockedArtifactStream -Stream $stream -Path $Path -Bytes $Bytes -Sha256 $Sha256 }
    finally { $stream.Dispose() }
}

function Publish-LockedArtifact {
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseShouldProcessForStateChangingFunctions', '', Justification = 'Private publication step called only for an explicitly requested artifact download.')]
    [CmdletBinding()]
    [OutputType([void])]
    param([Parameter(Mandatory)][string]$PartialPath, [Parameter(Mandatory)][string]$Path)
    # A hard link publishes without replacement while the verified source is
    # still open exclusively. No close/reopen/rename window admits a writer.
    # Unsupported filesystems fail closed and retain the resumable partial.
    $link = New-Item -ItemType HardLink -Path $Path -Value $PartialPath -ErrorAction Stop
    if ($null -eq $link) { throw 'Artifact publication did not create a hard link; partial retained.' }
}

function New-ArtifactHttpClient {
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseShouldProcessForStateChangingFunctions', '', Justification = 'Constructs an in-memory client; no request or persistent change occurs.')]
    [CmdletBinding()]
    [OutputType([Net.Http.HttpClient])]
    param([Net.Http.HttpClientHandler]$Handler)
    return [Net.Http.HttpClient]::new($Handler)
}

function Receive-LockedArtifact {
    <#
    .SYNOPSIS
    Downloads a bounded HTTPS artifact and publishes it only after lock verification.
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param([Parameter(Mandatory)][Uri]$Uri, [Parameter(Mandatory)][string]$Path,
          [Parameter(Mandatory)][long]$Bytes, [Parameter(Mandatory)][string]$Sha256,
          [ValidateRange(1,86400)][int]$TimeoutSeconds = 7200)
    $Path = Resolve-ExternalPath $Path
    if ($Uri.Scheme -ne 'https' -or $Uri.UserInfo -or $Bytes -le 0 -or $Sha256 -notmatch '^[a-fA-F0-9]{64}$') {
        throw 'A public HTTPS URL, positive byte limit, and expected SHA-256 are required.'
    }
    if (Test-Path -LiteralPath $Path) {
        Test-LockedArtifact -Path $Path -Bytes $Bytes -Sha256 $Sha256
        return $Path
    }
    [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Path))
    $partial = Resolve-ExternalPath ($Path + '.partial')
    $output = $client = $handler = $response = $inputStream = $request = $cancellation = $null
    try {
        # Exclusive ownership also prevents two setup processes appending to one partial.
        $output = [IO.FileStream]::new($partial, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite,
            [IO.FileShare]::None, 65536, [IO.FileOptions]::SequentialScan)
        $offset = $output.Length
        if ($offset -gt $Bytes) { throw "Partial file exceeds expected size; remove it before retrying: $partial" }
        if ($offset -lt $Bytes) {
            $handler = [Net.Http.HttpClientHandler]::new()
            $handler.AllowAutoRedirect = $false
            $handler.UseCookies = $false
            $client = New-ArtifactHttpClient $handler
            $client.Timeout = [TimeSpan]::FromSeconds(60)
            $cancellation = [Threading.CancellationTokenSource]::new([TimeSpan]::FromSeconds($TimeoutSeconds))
            $currentUri = $Uri
            for ($redirect = 0; ; $redirect++) {
                if ($redirect -gt 10) { throw 'Too many download redirects.' }
                $request = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Get, $currentUri)
                $request.Headers.AcceptEncoding.ParseAdd('identity')
                if ($offset -gt 0) { $request.Headers.Range = [Net.Http.Headers.RangeHeaderValue]::new($offset, $null) }
                $response = $client.SendAsync($request, [Net.Http.HttpCompletionOption]::ResponseHeadersRead,
                    $cancellation.Token).GetAwaiter().GetResult()
                if ([int]$response.StatusCode -notin @(301,302,303,307,308)) { break }
                if (-not $response.Headers.Location) { throw 'Redirect has no destination.' }
                $nextUri = [Uri]::new($currentUri, $response.Headers.Location)
                if ($nextUri.Scheme -ne 'https' -or $nextUri.UserInfo) { throw 'Unsafe download redirect.' }
                $response.Dispose(); $response = $null
                $request.Dispose(); $request = $null
                $currentUri = $nextUri
            }
            if ([int]$response.StatusCode -eq 206) {
                $range = $response.Content.Headers.ContentRange
                if (-not $range -or $range.Unit -ne 'bytes' -or $range.From -ne $offset -or
                    $range.To -ne ($Bytes - 1) -or $range.Length -ne $Bytes) { throw 'Unexpected resume range.' }
            } elseif ([int]$response.StatusCode -eq 200) {
                # A server may ignore Range. Start over instead of appending a full file.
                $offset = 0
                $output.SetLength(0)
            } else { throw "Download failed with HTTP $([int]$response.StatusCode)." }
            $expectedRemaining = $Bytes - $offset
            $contentLength = $response.Content.Headers.ContentLength
            if ($null -ne $contentLength -and $contentLength -ne $expectedRemaining) { throw 'Unexpected response byte length.' }
            if ($response.Content.Headers.ContentEncoding.Count -ne 0) { throw 'Encoded artifact response is unsupported.' }
            $output.Position = $offset
            $inputStream = $response.Content.ReadAsStreamAsync($cancellation.Token).GetAwaiter().GetResult()
            $buffer = [byte[]]::new(1048576)
            while (($count = $inputStream.ReadAsync($buffer, 0, $buffer.Length, $cancellation.Token).GetAwaiter().GetResult()) -gt 0) {
                if ($output.Position + $count -gt $Bytes) { throw 'Download exceeded locked byte limit.' }
                $output.Write($buffer, 0, $count)
            }
            $output.Flush($true)
        }
        Assert-LockedArtifactStream -Stream $output -Path $partial -Bytes $Bytes -Sha256 $Sha256
        Publish-LockedArtifact -PartialPath $partial -Path $Path
        $output.Dispose(); $output = $null
        try { [IO.File]::Delete($partial) }
        catch { Write-Warning "Artifact published successfully; partial alias retained at ${partial}: $_" }
        return $Path
    } finally {
        foreach ($resource in @($inputStream, $response, $request, $client, $handler, $output, $cancellation)) {
            if ($null -ne $resource) { $resource.Dispose() }
        }
    }
}

function Assert-Arm64PE {
    <#
    .SYNOPSIS
    Validates a native ARM64 PE signature without executing the file.
    #>
    [CmdletBinding()]
    [OutputType([void])]
    param([Parameter(Mandatory)][string]$Path)
    $reader = [IO.BinaryReader]::new([IO.File]::OpenRead($Path))
    try {
        if ($reader.BaseStream.Length -lt 64 -or $reader.ReadUInt16() -ne 0x5A4D) { throw 'Invalid DOS header.' }
        $reader.BaseStream.Position = 0x3C
        $peOffset = $reader.ReadUInt32()
        if ($peOffset -gt $reader.BaseStream.Length - 24) { throw 'Invalid PE offset.' }
        $reader.BaseStream.Position = $peOffset
        if ($reader.ReadUInt32() -ne 0x00004550 -or $reader.ReadUInt16() -ne 0xAA64) {
            throw 'Expected a native ARM64 PE image.'
        }
    } catch { throw "ARM64 verification failed for ${Path}: $_" }
    finally { $reader.Dispose() }
}

Export-ModuleMember -Function Resolve-ExternalPath, Read-BonsaiLock, Test-LockedArtifact, Receive-LockedArtifact, Assert-Arm64PE

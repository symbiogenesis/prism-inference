#requires -Version 7.6
<#
.SYNOPSIS
Checks the bounded GGML bridge protocol, optionally against its previous executable.
.DESCRIPTION
Exercises only the PQ2/PTQ1 process adapter. Deterministic supplied inputs must
produce byte-identical responses when ReferenceBridge is supplied. Invalid
requests must fail before touching an existing output. No timings are measured.
#>
[CmdletBinding()]
[OutputType([string])]
param(
    [Parameter(Mandatory)][string]$Bridge,
    [string]$ReferenceBridge,
    [string]$EvidenceDirectory = (Join-Path ([IO.Path]::GetTempPath()) ('Prism-GgmlBridge-' + [guid]::NewGuid().ToString('N')))
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot 'Bonsai.TestSupport.psm1')
Import-Module (Join-Path $PSScriptRoot '../scripts/Bonsai.Common.psm1')
Import-Module (Join-Path $PSScriptRoot '../scripts/Bonsai.Native.psm1')
$evidence = Resolve-ExternalPath $EvidenceDirectory
if (Test-Path -LiteralPath $evidence) { throw 'Use a fresh GGML bridge evidence directory.' }
$executables = [ordered]@{ Rust = $Bridge }
if ($ReferenceBridge) { $executables.Reference = $ReferenceBridge }
foreach ($entry in $executables.GetEnumerator()) {
    if (-not [IO.Path]::IsPathFullyQualified($entry.Value) -or $entry.Value -match '%[^%]+%' -or
        -not [IO.File]::Exists($entry.Value)) {
        throw "$($entry.Key) bridge must be an existing expanded absolute executable."
    }
}
[void][IO.Directory]::CreateDirectory($evidence)
$observations = [Collections.Generic.List[object]]::new()

function Get-GgmlHeader {
    <# .SYNOPSIS
    Writes an explicitly supplied request header in the protocol's byte order.
    #>
    [CmdletBinding()]
    [OutputType([byte[]])]
    param([uint32]$Operation, [uint32]$K, [uint32]$M, [uint32]$N, [uint32]$PackedCount, [uint32]$ActivationCount)
    $stream = [IO.MemoryStream]::new()
    $writer = [IO.BinaryWriter]::new($stream)
    try {
        foreach ($value in @($Operation, $K, $M, $N, $PackedCount, $ActivationCount)) { $writer.Write([uint32]$value) }
        $PSCmdlet.WriteObject($stream.ToArray(), $false)
    } finally { $writer.Dispose(); $stream.Dispose() }
}

function Get-GgmlPacket {
    <# .SYNOPSIS
    Frames supplied request bytes without deriving expected production results.
    #>
    [CmdletBinding()]
    [OutputType([byte[]])]
    param([uint32]$Count, [AllowEmptyCollection()][byte[]]$Payload = @())
    $stream = [IO.MemoryStream]::new()
    $writer = [IO.BinaryWriter]::new($stream)
    try {
        $writer.Write([Text.Encoding]::ASCII.GetBytes("BSPREQ1`n"))
        $writer.Write($Count)
        $writer.Write($Payload)
        $PSCmdlet.WriteObject($stream.ToArray(), $false)
    } finally { $writer.Dispose(); $stream.Dispose() }
}

function Get-GgmlRequest {
    <# .SYNOPSIS
    Supplies deterministic finite-scale packed weights and activation bit patterns.
    #>
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param([uint32]$Operation, [uint32]$K, [uint32]$M, [uint32]$N, [int]$Seed)
    $blockBytes = if ($Operation -eq 1) { 34 } else { 28 }
    $blocks = [int]($K / 128 * $M)
    $packed = [byte[]]::new($blocks * $blockBytes)
    $random = [Random]::new($Seed)
    $random.NextBytes($packed)
    for ($block = 0; $block -lt $blocks; $block++) {
        # PQ2 starts with its f16 scale; PTQ1 stores it after 26 code bytes.
        # Preserve arbitrary, including noncanonical, PTQ1 packed codes.
        $scaleOffset = $block * $blockBytes + $(if ($Operation -eq 1) { 0 } else { 26 })
        $packed[$scaleOffset] = 0
        $packed[$scaleOffset + 1] = 0x3c
        if ($Operation -eq 1) {
            for ($offset = 2; $offset -lt $blockBytes; $offset++) {
                $packed[$block * $blockBytes + $offset] = [byte]($random.Next(3) -bor ($random.Next(3) -shl 2) -bor
                    ($random.Next(3) -shl 4) -bor ($random.Next(3) -shl 6))
            }
        }
    }
    $activationCount = if ($Operation -eq 1) { $K * $N } else { 0 }
    $stream = [IO.MemoryStream]::new()
    $writer = [IO.BinaryWriter]::new($stream)
    try {
        $writer.Write((Get-GgmlHeader -Operation $Operation -K $K -M $M -N $N -PackedCount $packed.Length -ActivationCount $activationCount))
        $writer.Write($packed)
        for ($index = 0; $index -lt $activationCount; $index++) { $writer.Write([single]($random.Next(-4096, 4097) / 1024.0)) }
        $outputCount = if ($Operation -eq 1) { $M * $N } else { $K * $M }
        return [pscustomobject]@{ Bytes = $stream.ToArray(); OutputCount = $outputCount }
    } finally { $writer.Dispose(); $stream.Dispose() }
}

function Assert-GgmlResponse {
    <# .SYNOPSIS
    Verifies response framing, supplied shapes, and absence of trailing bytes.
    #>
    [CmdletBinding()]
    [OutputType([void])]
    param([string]$Path, [uint32[]]$OutputCounts)
    $stream = [IO.File]::OpenRead($Path)
    $reader = [IO.BinaryReader]::new($stream)
    try {
        Assert ([Text.Encoding]::ASCII.GetString($reader.ReadBytes(8)) -ceq "BSPRES1`n") 'Wrong response magic.'
        Assert ($reader.ReadUInt32() -eq $OutputCounts.Length) 'Wrong response count.'
        foreach ($count in $OutputCounts) {
            Assert ($reader.ReadUInt32() -eq $count) 'Wrong response shape.'
            $values = $reader.ReadBytes([int]($count * 4))
            Assert ($values.Length -eq $count * 4) 'Truncated response values.'
        }
        Assert ($stream.Position -eq $stream.Length) 'Trailing response data.'
    } finally { $reader.Dispose(); $stream.Dispose() }
}

function Invoke-GgmlCase {
    <# .SYNOPSIS
    Retains each process result and checks rejection or exact cross-language parity.
    #>
    [CmdletBinding()]
    [OutputType([void])]
    param(
        [string]$Name,
        [byte[]]$Packet,
        [uint32[]]$OutputCounts = @(),
        [string]$ExpectedError,
        [ValidateSet('ordinary', 'usage', 'relative-input', 'relative-output', 'same-path', 'normalized-alias', 'missing-input', 'missing-output-parent', 'output-directory', 'oversized-file')]
        [string]$Mode = 'ordinary'
    )
    $directory = Join-Path $evidence $Name
    [void][IO.Directory]::CreateDirectory($directory)
    $inputPath = Join-Path $directory 'request.bin'
    [IO.File]::WriteAllBytes($inputPath, $Packet)
    if ($Mode -eq 'oversized-file') {
        $stream = [IO.File]::OpenWrite($inputPath)
        try { $stream.SetLength(128MB + 1) } finally { $stream.Dispose() }
    }
    $inputDigest = (Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash
    $responseDigest = $null
    foreach ($entry in $executables.GetEnumerator()) {
        $outputPath = Join-Path $directory ($entry.Key + '-response.bin')
        $sentinel = [Text.Encoding]::ASCII.GetBytes('existing output must survive input rejection')
        [IO.File]::WriteAllBytes($outputPath, $sentinel)
        $requestArgument = $inputPath
        $outputArgument = $outputPath
        switch ($Mode) {
            'relative-input' { $requestArgument = 'request.bin' }
            'relative-output' { $outputArgument = 'response.bin' }
            'same-path' { $outputArgument = $inputPath }
            'normalized-alias' {
                [void][IO.Directory]::CreateDirectory((Join-Path $directory 'nested'))
                $outputArgument = $directory + [IO.Path]::DirectorySeparatorChar + 'nested' + [IO.Path]::DirectorySeparatorChar + '..' + [IO.Path]::DirectorySeparatorChar + 'request.bin'
            }
            'missing-input' { $requestArgument = Join-Path $directory 'absent.bin' }
            'missing-output-parent' { $outputArgument = Join-Path $directory 'absent/response.bin' }
            'output-directory' { $outputArgument = $directory }
        }
        $arguments = if ($Mode -eq 'usage') { @('--output', $outputArgument, '--input', $requestArgument) }
            else { @('--input', $requestArgument, '--output', $outputArgument) }
        $result = Invoke-BonsaiNativeProcess -Executable $entry.Value -ArgumentList $arguments -WorkingDirectory $directory -TimeoutSeconds 60 -LogPath (Join-Path $directory ($entry.Key + '.log'))
        if ($OutputCounts.Length) {
            Assert ($result.ExitCode -eq 0) "$Name/$($entry.Key) failed: $($result.StandardError)"
            Assert-GgmlResponse -Path $outputPath -OutputCounts $OutputCounts
            $digest = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash
            if ($null -eq $responseDigest) { $responseDigest = $digest }
            else { Assert ($digest -ceq $responseDigest) "$Name produced different response bytes before and after the Rust port." }
        } else {
            Assert ($result.ExitCode -eq 1) "$Name/$($entry.Key) must reject with exit one."
            Assert ($result.StandardError.Contains('prism-upstream-bridge: ')) "$Name/$($entry.Key) omitted its process error prefix."
            if ($ExpectedError) { Assert ($result.StandardError.Contains($ExpectedError)) "$Name/$($entry.Key) changed its rejection: $($result.StandardError)" }
            Assert ([Convert]::ToHexString([IO.File]::ReadAllBytes($outputPath)) -ceq [Convert]::ToHexString($sentinel)) "$Name/$($entry.Key) touched the response before input validation completed."
            Assert ((Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash -ceq $inputDigest) "$Name/$($entry.Key) modified its input."
        }
        $observations.Add([ordered]@{ Case = $Name; Engine = $entry.Key; ExitCode = $result.ExitCode; ResponseSha256 = $responseDigest })
    }
}

# A focused numeric sample spans both formats, multiple rows/columns and mixed
# request order. Both executables receive exactly the same immutable input file.
$requests = @(
    (Get-GgmlRequest -Operation 1 -K 128 -M 1 -N 1 -Seed 11),
    (Get-GgmlRequest -Operation 2 -K 128 -M 257 -N 0 -Seed 12),
    (Get-GgmlRequest -Operation 1 -K 384 -M 4 -N 4 -Seed 13),
    (Get-GgmlRequest -Operation 1 -K 640 -M 4 -N 16 -Seed 14),
    (Get-GgmlRequest -Operation 2 -K 128 -M 1 -N 0 -Seed 15),
    (Get-GgmlRequest -Operation 1 -K 8192 -M 1 -N 1 -Seed 16)
)
$payload = [Collections.Generic.List[byte]]::new()
foreach ($request in $requests) { $payload.AddRange([byte[]]$request.Bytes) }
Invoke-GgmlCase -Name 'mixed-numerics' -Packet (Get-GgmlPacket -Count $requests.Length -Payload $payload.ToArray()) -OutputCounts @($requests | ForEach-Object { [uint32]$_.OutputCount })
$tiny = Get-GgmlRequest -Operation 2 -K 128 -M 1 -N 0 -Seed 123
$payload.Clear()
for ($index = 0; $index -lt 64; $index++) { $payload.AddRange([byte[]]$tiny.Bytes) }
Invoke-GgmlCase -Name 'maximum-request-count' -Packet (Get-GgmlPacket -Count 64 -Payload $payload.ToArray()) -OutputCounts ([uint32[]]::new(64) | ForEach-Object { [uint32]128 })
$valid = Get-GgmlPacket -Count 1 -Payload $tiny.Bytes

$invalid = @(
    @{ Name = 'empty'; Packet = [byte[]]::new(0); Error = 'invalid request magic' },
    @{ Name = 'bad-magic'; Packet = [Text.Encoding]::ASCII.GetBytes('BADREQ1!'); Error = 'invalid request magic' },
    @{ Name = 'truncated-count'; Packet = [Text.Encoding]::ASCII.GetBytes("BSPREQ1`n"); Error = 'truncated request' },
    @{ Name = 'zero-count'; Packet = (Get-GgmlPacket -Count 0); Error = 'request count outside 1..64' },
    @{ Name = 'too-many-requests'; Packet = (Get-GgmlPacket -Count 65); Error = 'request count outside 1..64' },
    @{ Name = 'truncated-header'; Packet = (Get-GgmlPacket -Count 1); Error = 'truncated request' },
    @{ Name = 'unknown-operation'; Packet = (Get-GgmlPacket -Count 1 -Payload (Get-GgmlHeader -Operation 3 -K 128 -M 1 -N 0 -PackedCount 0 -ActivationCount 0)); Error = 'unknown operation' },
    @{ Name = 'packed-limit'; Packet = (Get-GgmlPacket -Count 1 -Payload (Get-GgmlHeader -Operation 2 -K 128 -M 1 -N 0 -PackedCount (16MB + 1) -ActivationCount 0)); Error = 'request payload exceeds size limit' },
    @{ Name = 'activation-limit'; Packet = (Get-GgmlPacket -Count 1 -Payload (Get-GgmlHeader -Operation 1 -K 128 -M 1 -N 1 -PackedCount 34 -ActivationCount (4MB + 1))); Error = 'request payload exceeds size limit' },
    @{ Name = 'pq2-zero-k'; Packet = (Get-GgmlPacket -Count 1 -Payload (Get-GgmlHeader -Operation 1 -K 0 -M 1 -N 1 -PackedCount 0 -ActivationCount 0)); Error = 'invalid PQ2 matrix shape' },
    @{ Name = 'pq2-unblocked-k'; Packet = (Get-GgmlPacket -Count 1 -Payload (Get-GgmlHeader -Operation 1 -K 129 -M 1 -N 1 -PackedCount 34 -ActivationCount 129)); Error = 'invalid PQ2 matrix shape' },
    @{ Name = 'pq2-large-k'; Packet = (Get-GgmlPacket -Count 1 -Payload (Get-GgmlHeader -Operation 1 -K 8320 -M 1 -N 1 -PackedCount 0 -ActivationCount 0)); Error = 'invalid PQ2 matrix shape' },
    @{ Name = 'pq2-zero-m'; Packet = (Get-GgmlPacket -Count 1 -Payload (Get-GgmlHeader -Operation 1 -K 128 -M 0 -N 1 -PackedCount 0 -ActivationCount 128)); Error = 'invalid PQ2 matrix shape' },
    @{ Name = 'pq2-large-m'; Packet = (Get-GgmlPacket -Count 1 -Payload (Get-GgmlHeader -Operation 1 -K 128 -M 4097 -N 1 -PackedCount 0 -ActivationCount 128)); Error = 'invalid PQ2 matrix shape' },
    @{ Name = 'pq2-zero-n'; Packet = (Get-GgmlPacket -Count 1 -Payload (Get-GgmlHeader -Operation 1 -K 128 -M 1 -N 0 -PackedCount 34 -ActivationCount 0)); Error = 'invalid PQ2 matrix shape' },
    @{ Name = 'pq2-large-n'; Packet = (Get-GgmlPacket -Count 1 -Payload (Get-GgmlHeader -Operation 1 -K 128 -M 1 -N 257 -PackedCount 34 -ActivationCount 0)); Error = 'invalid PQ2 matrix shape' },
    @{ Name = 'pq2-weight-elements'; Packet = (Get-GgmlPacket -Count 1 -Payload (Get-GgmlHeader -Operation 1 -K 8192 -M 1025 -N 1 -PackedCount 0 -ActivationCount 0)); Error = 'PQ2 weights exceed element limit' },
    @{ Name = 'pq2-packed-length'; Packet = (Get-GgmlPacket -Count 1 -Payload (Get-GgmlHeader -Operation 1 -K 128 -M 1 -N 1 -PackedCount 33 -ActivationCount 128)); Error = 'PQ2 payload does not match shape' },
    @{ Name = 'pq2-activation-length'; Packet = (Get-GgmlPacket -Count 1 -Payload (Get-GgmlHeader -Operation 1 -K 128 -M 1 -N 1 -PackedCount 34 -ActivationCount 127)); Error = 'PQ2 payload does not match shape' },
    @{ Name = 'ptq1-wrong-k'; Packet = (Get-GgmlPacket -Count 1 -Payload (Get-GgmlHeader -Operation 2 -K 256 -M 1 -N 0 -PackedCount 28 -ActivationCount 0)); Error = 'invalid PTQ1 batch shape' },
    @{ Name = 'ptq1-zero-blocks'; Packet = (Get-GgmlPacket -Count 1 -Payload (Get-GgmlHeader -Operation 2 -K 128 -M 0 -N 0 -PackedCount 0 -ActivationCount 0)); Error = 'invalid PTQ1 batch shape' },
    @{ Name = 'ptq1-large-batch'; Packet = (Get-GgmlPacket -Count 1 -Payload (Get-GgmlHeader -Operation 2 -K 128 -M 20001 -N 0 -PackedCount 0 -ActivationCount 0)); Error = 'invalid PTQ1 batch shape' },
    @{ Name = 'ptq1-wrong-n'; Packet = (Get-GgmlPacket -Count 1 -Payload (Get-GgmlHeader -Operation 2 -K 128 -M 1 -N 1 -PackedCount 28 -ActivationCount 0)); Error = 'invalid PTQ1 batch shape' },
    @{ Name = 'ptq1-packed-length'; Packet = (Get-GgmlPacket -Count 1 -Payload (Get-GgmlHeader -Operation 2 -K 128 -M 1 -N 0 -PackedCount 27 -ActivationCount 0)); Error = 'PTQ1 payload does not match shape' },
    @{ Name = 'ptq1-activation-length'; Packet = (Get-GgmlPacket -Count 1 -Payload (Get-GgmlHeader -Operation 2 -K 128 -M 1 -N 0 -PackedCount 28 -ActivationCount 1)); Error = 'PTQ1 payload does not match shape' },
    @{ Name = 'truncated-packed'; Packet = (Get-GgmlPacket -Count 1 -Payload (Get-GgmlHeader -Operation 2 -K 128 -M 1 -N 0 -PackedCount 28 -ActivationCount 0)); Error = 'truncated packed payload' },
    @{ Name = 'truncated-activation'; Packet = (Get-GgmlPacket -Count 1 -Payload ((Get-GgmlHeader -Operation 1 -K 128 -M 1 -N 1 -PackedCount 34 -ActivationCount 128) + [byte[]]::new(34))); Error = 'truncated request' },
    @{ Name = 'trailing-data'; Packet = [byte[]]($valid + [byte]0); Error = 'trailing request data' }
)
foreach ($case in $invalid) { Invoke-GgmlCase -Name $case.Name -Packet $case.Packet -ExpectedError $case.Error }

# Six maximum-sized PTQ1 responses fit the total; the seventh header must reject
# before reading its packed payload or opening an output. This is not executed.
$payload.Clear()
$largeRequest = [byte[]]((Get-GgmlHeader -Operation 2 -K 128 -M 20000 -N 0 -PackedCount 560000 -ActivationCount 0) + [byte[]]::new(560000))
for ($index = 0; $index -lt 6; $index++) { $payload.AddRange($largeRequest) }
$payload.AddRange((Get-GgmlHeader -Operation 2 -K 128 -M 20000 -N 0 -PackedCount 560000 -ActivationCount 0))
Invoke-GgmlCase -Name 'total-output-limit' -Packet (Get-GgmlPacket -Count 7 -Payload $payload.ToArray()) -ExpectedError 'responses exceed total element limit'
foreach ($case in @(
    @{ Mode = 'usage'; Error = 'usage: prism-upstream-bridge' },
    @{ Mode = 'relative-input'; Error = 'protocol paths must be absolute' },
    @{ Mode = 'relative-output'; Error = 'protocol paths must be absolute' },
    @{ Mode = 'same-path'; Error = 'input and output paths must differ' },
    @{ Mode = 'normalized-alias'; Error = 'input and output paths must differ' },
    @{ Mode = 'missing-input'; Error = '' },
    @{ Mode = 'missing-output-parent'; Error = 'cannot open response file' },
    @{ Mode = 'output-directory'; Error = 'cannot open response file' },
    @{ Mode = 'oversized-file'; Error = 'request file exceeds size limit' }
)) { Invoke-GgmlCase -Name $case.Mode -Packet $valid -Mode $case.Mode -ExpectedError $case.Error }

$identities = @($executables.GetEnumerator() | ForEach-Object {
    [ordered]@{ Engine = $_.Key; Path = $_.Value; Sha256 = (Get-FileHash -LiteralPath $_.Value -Algorithm SHA256).Hash.ToLowerInvariant() }
})
[IO.File]::WriteAllText((Join-Path $evidence 'receipt.json'), ([ordered]@{
    SchemaVersion = 1; ByteParityCompared = [bool]$ReferenceBridge; Executables = $identities; Cases = $observations.ToArray()
} | ConvertTo-Json -Depth 8))
Write-Output "GGML bridge numeric/protocol checks passed; before/after byte parity: $([bool]$ReferenceBridge). Evidence: $evidence"

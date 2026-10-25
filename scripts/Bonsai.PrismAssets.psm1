#requires -Version 7.6
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot 'Bonsai.PrismSource.psm1')
Import-Module (Join-Path $PSScriptRoot 'Bonsai.Common.psm1')

function Save-BonsaiPrismMetadataAsset {
    <#
    .SYNOPSIS
    Materializes immutable, bounded metadata inputs outside the checkout.
    #>
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseShouldProcessForStateChangingFunctions', '', Justification = 'Called only by explicitly requested Prism test execution, writing fresh external evidence.')]
    [CmdletBinding()]
    [OutputType([hashtable])]
    param([Parameter(Mandatory)][string]$Directory)
    $root = Resolve-ExternalPath $Directory
    [void][IO.Directory]::CreateDirectory($root)
    $runtimeSource = Resolve-PrismSource
    $lockPath = Join-Path $runtimeSource 'tools/prism/upstream/metadata-assets.lock.json'
    $lock = Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json -AsHashtable
    if ($lock.SchemaVersion -ne 1 -or $lock.Models.Count -ne 11) { throw 'Unexpected metadata fixture lock schema.' }
    $client = [Net.Http.HttpClient]::new()
    $client.Timeout = [TimeSpan]::FromSeconds(120)
    $receipts = [Collections.Generic.List[object]]::new()
    try {
        foreach ($model in $lock.Models) {
            if ($model.Repo -cnotmatch '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$' -or
                $model.Revision -cnotmatch '^[0-9a-f]{40}$' -or $model.Files.Count -lt 1 -or $model.Files.Count -gt 256) {
                throw 'Invalid immutable metadata model identity.'
            }
            $provider = if ($model.ContainsKey('ProviderRepo')) { $model.ProviderRepo } else { $model.Repo }
            if ($provider -cnotmatch '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$') { throw 'Invalid metadata provider repository.' }
            foreach ($asset in @($model.Api) + @($model.Files)) {
                if ($asset.Path -cnotmatch '^metadata/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$' -or
                    ($asset.Bytes -isnot [int] -and $asset.Bytes -isnot [long]) -or $asset.Bytes -lt 1 -or $asset.Bytes -gt 67108864 -or
                    $asset.Sha256 -cnotmatch '^[0-9a-f]{64}$') {
                    throw 'Invalid bounded metadata asset identity.'
                }
                $destination = Resolve-ExternalPath (Join-Path $root $asset.Path)
                [void][IO.Directory]::CreateDirectory((Split-Path $destination -Parent))
                if (Test-Path -LiteralPath $destination) { throw 'Metadata fixture destination must be fresh.' }
                if ($asset.ContainsKey('CorpusPath')) {
                    if ($asset.CorpusPath -cnotmatch '^tools/prism/upstream/metadata-api/[A-Za-z0-9_.-]+\.json$' -or
                        $asset.Url -cne "https://huggingface.co/api/models/$provider/revision/$($model.Revision)") {
                        throw 'Invalid frozen API corpus provenance.'
                    }
                    # API listings include changing activity fields even at a revision URL.
                    # Use the original raw response corpus, never a decoded expected schema.
                    Copy-Item -LiteralPath (Join-Path $runtimeSource $asset.CorpusPath) -Destination $destination
                } else {
                    if ($asset.Filename -cnotmatch '^[A-Za-z0-9_./-]+\.gguf$' -or
                        $asset.Filename.Split('/') -ccontains '..' -or
                        $asset.Status -ne 206 -or $asset.RangeBytes -ne $asset.Bytes -or
                        $asset.Url -cne "https://huggingface.co/$provider/resolve/$($model.Revision)/$($asset.Filename)") {
                        throw 'Invalid immutable GGUF range provenance.'
                    }
                    $request = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Get, $asset.Url)
                    $request.Headers.Range = [Net.Http.Headers.RangeHeaderValue]::new(0, $asset.RangeBytes - 1)
                    $response = $null
                    try {
                        $response = $client.Send($request, [Net.Http.HttpCompletionOption]::ResponseHeadersRead)
                        $null = $response.EnsureSuccessStatusCode()
                        $range = $response.Content.Headers.ContentRange
                        if ([int]$response.StatusCode -ne 206 -or $null -eq $range -or $range.From -ne 0 -or
                            $range.To -ne $asset.RangeBytes - 1 -or $response.Content.Headers.ContentLength -gt $asset.Bytes) {
                            throw 'Server did not honor the exact bounded GGUF prefix request.'
                        }
                        $inputStream = $response.Content.ReadAsStream()
                        $outputStream = [IO.FileStream]::new($destination, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
                        try {
                            $buffer = [byte[]]::new(65536)
                            while (($count = $inputStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
                                if ($outputStream.Length + $count -gt $asset.Bytes) { throw 'GGUF prefix exceeds its pinned size.' }
                                $outputStream.Write($buffer, 0, $count)
                            }
                        } finally { $inputStream.Dispose(); $outputStream.Dispose() }
                    } finally { if ($response) { $response.Dispose() }; $request.Dispose() }
                }
                Test-LockedArtifact -Path $destination -Bytes $asset.Bytes -Sha256 $asset.Sha256
                $receipts.Add(@{ Path=$destination; Bytes=$asset.Bytes; Sha256=$asset.Sha256; Url=$asset.Url })
            }
        }
        # The same verified manifest is consumed by both safe Rust target runs.
        $manifestDestination = Join-Path $root 'metadata/manifest.json'
        Copy-Item -LiteralPath $lockPath -Destination $manifestDestination
        return @{ Assets=$receipts.ToArray(); Manifest=$manifestDestination; ManifestSha256=(Get-FileHash -LiteralPath $manifestDestination -Algorithm SHA256).Hash.ToLowerInvariant() }
    } finally { $client.Dispose() }
}

function Save-BonsaiPrismFixture {
    <# .SYNOPSIS
    Preflights a complete fixture lock before receiving its verified external files.
    #>
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseShouldProcessForStateChangingFunctions', '', Justification = 'Explicit Prism setup writes only hash-verified external input artifacts.')]
    [CmdletBinding()]
    [OutputType([hashtable])]
    param([Parameter(Mandatory)][hashtable]$Lock, [Parameter(Mandatory)][string]$Directory,
          [Parameter(Mandatory)][long]$TotalBytes, [Parameter(Mandatory)][long]$MaximumFileBytes,
          [Parameter(Mandatory)][scriptblock]$ValidateEntry, [scriptblock]$ValidatePaths,
          [hashtable]$Receipt = @{})
    $root = Resolve-ExternalPath $Directory
    $paths = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    [long]$total = 0
    # Family predicates own the allowed relative paths and immutable source URLs.
    # Validate the entire collection before allowing the first download.
    foreach ($entry in $Lock.Files) {
        if ($entry.Path -isnot [string] -or -not $paths.Add($entry.Path) -or
            $entry.Sha256 -cnotmatch '^[a-f0-9]{64}$' -or
            $entry.Bytes -le 0 -or $entry.Bytes -gt $MaximumFileBytes -or -not (& $ValidateEntry $entry)) {
            throw 'A Prism fixture input has an invalid or unpinned identity.'
        }
        $total += $entry.Bytes
        if ($total -gt $TotalBytes) { throw 'Prism fixture extent exceeds its lock.' }
    }
    if ($total -ne $TotalBytes) { throw 'Prism fixture extent is incomplete.' }
    if ($ValidatePaths) { $null = & $ValidatePaths $paths }
    $Receipt.Files = @(foreach ($entry in $Lock.Files) {
        $path = Resolve-ExternalPath (Join-Path $root ($Lock.AssetPrefix + $entry.Path))
        $null = Receive-LockedArtifact -Uri $entry.Url -Path $path -Bytes $entry.Bytes -Sha256 $entry.Sha256 -TimeoutSeconds 600
        @{ Path=$path; Bytes=$entry.Bytes; Sha256=$entry.Sha256; Url=$entry.Url }
    })
    $Receipt.TotalBytes = $total
    return $Receipt
}

function Save-BonsaiPrismTokenizerRepository {
    <#
    .SYNOPSIS
    Materializes every vocabulary/corpus triple in the compatible publisher snapshot.
    #>
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseShouldProcessForStateChangingFunctions', '', Justification = 'Explicit Prism setup writes only hash-verified external input artifacts.')]
    [CmdletBinding()]
    [OutputType([hashtable])]
    param([Parameter(Mandatory)][string]$Directory)
    $runtimeSource = Resolve-PrismSource
    $lockPath = Join-Path $runtimeSource 'tests/upstream_prism/tokenizer-repository-assets.json'
    $lock = Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json -AsHashtable
    $revision = 'cee4b7a68518c579bb758fca46bb149cd109348e'
    if ($lock.Repository -cne 'ggml-org/vocabs' -or $lock.Revision -cne $revision -or
        $lock.AssetPrefix -cne 'tokenizer-repository/' -or $lock.Files.Count -ne 18) {
        throw 'The tokenizer corpus repository lock differs from its source identity.'
    }
    return Save-BonsaiPrismFixture -Lock $lock -Directory $Directory -TotalBytes 23804987 -MaximumFileBytes 20000000 -Receipt @{ Revision=$revision; CorpusCount=6 } -ValidateEntry {
        param($entry)
        $entry.Path -cmatch '^(BPE|SPM|HybridDNA|PLaMo2|PLaMo3|RWKV|UGM|WPM)/ggml-vocab-[a-z0-9-]+\.gguf(?:\.(?:inp|out))?$' -and
        $entry.GitBlobSha1 -cmatch '^[a-f0-9]{40}$' -and
        $entry.Url -ceq "https://huggingface.co/ggml-org/vocabs/resolve/$revision/$($entry.Path)"
    }.GetNewClosure() -ValidatePaths {
        param($paths)
        foreach ($relative in $paths) {
            if ($relative.EndsWith('.gguf', [StringComparison]::Ordinal) -and
                (-not $paths.Contains($relative + '.inp') -or -not $paths.Contains($relative + '.out'))) {
                throw 'An external vocabulary has no complete input/output corpus.'
            }
        }
    }
}

function Save-BonsaiPrismTokenizerReference {
    <# .SYNOPSIS
    Materializes all immutable inputs of the original 26 tokenizer families.
    #>
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseShouldProcessForStateChangingFunctions', '', Justification = 'Explicit Prism setup writes only hash-verified external publisher inputs.')]
    [CmdletBinding()]
    [OutputType([hashtable])]
    param([Parameter(Mandatory)][string]$Directory)
    $runtimeSource = Resolve-PrismSource
    $lockPath = Join-Path $runtimeSource 'tests/upstream_prism/tokenizer-reference-assets.json'
    $lock = Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json -AsHashtable
    if ($lock.SchemaVersion -ne 1 -or $lock.UpstreamRevision -cne 'b87ca7225456b06b0975b8239c66b5aecf07e37c' -or
        $lock.AssetPrefix -cne 'tokenizer-reference/' -or $lock.Files.Count -ne 122 -or $lock.Cases.Count -ne 26) {
        throw 'The full tokenizer reference lock differs from its original source identity.'
    }
    $names = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($case in $lock.Cases) {
        if ($case.Name -cnotmatch '^[a-z0-9-]+$' -or -not $names.Add($case.Name)) { throw 'Invalid tokenizer family identity.' }
    }
    return Save-BonsaiPrismFixture -Lock $lock -Directory $Directory -TotalBytes 150188454 -MaximumFileBytes 33554432 -Receipt @{ Families=26; GeneratedVocabularyOwner='safe-rust' } -ValidateEntry {
        param($entry)
        $entry.Path -cmatch '^[a-z0-9-]+/(?:config\.json|tokenizer\.json|tokenizer_config(?:\.source)?\.json|special_tokens_map\.json|added_tokens\.json|tokenizer\.model|vocab\.(?:json|txt)|merges\.txt)$' -and
        $names.Contains($entry.Path.Split('/')[0]) -and
        $entry.Repository -cmatch '^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$' -and
        $entry.Revision -cmatch '^[a-f0-9]{40}$' -and
        $entry.Url.StartsWith("https://huggingface.co/$($entry.Repository)/resolve/$($entry.Revision)/", [StringComparison]::Ordinal)
    }.GetNewClosure()
}

function Save-BonsaiPrismLoraAsset {
    <#
    .SYNOPSIS
    Materializes the exact original model, adapter, tokenizer and text inputs.
    #>
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseShouldProcessForStateChangingFunctions', '', Justification = 'Explicit Prism test setup writes only verified external input artifacts.')]
    [CmdletBinding()]
    [OutputType([hashtable])]
    param([Parameter(Mandatory)][string]$Directory)
    $runtimeSource = Resolve-PrismSource
    $lockPath = Join-Path $runtimeSource 'tests/upstream_prism/lora-assets.json'
    $lock = Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json -AsHashtable
    $revision = 'c26d5fb85b4070a9e9c4e65d132c783b98086890'
    if ($lock.Repository -cne 'ggml-org/lora-tests' -or $lock.Revision -cne $revision -or
        $lock.Source -cne 'tests/test-lora-conversion-inference.sh' -or
        $lock.SourceSha256 -cne '69584616380bdaa7f6ee1e53d165408118fb0897e1280f964eb7bc3b932b9c9a' -or
        $lock.AssetPrefix -cne 'lora-tests/' -or $lock.Files.Count -ne 28) {
        throw 'The original LoRA fixture lock differs from its source identity.'
    }
    return Save-BonsaiPrismFixture -Lock $lock -Directory $Directory -TotalBytes 269404497 -MaximumFileBytes 100000000 -Receipt @{ Revision=$revision; ConvertedInputs=$false } -ValidateEntry {
        param($entry)
        $relative = $entry.Path
        -not ([string]::IsNullOrEmpty($relative) -or [IO.Path]::IsPathRooted($relative) -or $relative.Contains('\') -or
            @($relative.Split('/') | Where-Object { $_ -in @('', '.', '..') }).Count) -and
        $entry.Url -ceq "https://huggingface.co/ggml-org/lora-tests/resolve/$revision/$relative"
    }.GetNewClosure()
}

Export-ModuleMember -Function Save-BonsaiPrismMetadataAsset, Save-BonsaiPrismTokenizerRepository, Save-BonsaiPrismTokenizerReference, Save-BonsaiPrismLoraAsset

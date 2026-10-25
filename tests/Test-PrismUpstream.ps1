#requires -Version 7.6
<#
.SYNOPSIS
Runs safe Rust assertions against pinned Prism and the Rust runtime.
.DESCRIPTION
All source, build, request and result files remain in fresh external evidence.
The full catalog runs only in main-branch GitHub Actions. A named adapter may
run locally. Expected Rust conformance failures are data; compiler, reference,
timeout, malformed-report and baseline regressions fail the check.
#>
[CmdletBinding()]
[OutputType([Collections.IDictionary])]
param(
    [Parameter(Mandatory)][string]$SourceArchive,
    [Parameter(Mandatory)][string]$EvidenceDirectory,
    [Parameter(Mandatory)][string]$TargetDirectory,
    [ValidateSet('All', 'Pq2RowShapes', 'Ptq1ElementMap', 'Unicode', 'MtpCatchupBatch', 'Jinja', 'MtmdImpl', 'GrammarParser', 'ReasoningBudget', 'Col2im', 'Dfly', 'Rope', 'LlamaGrammar', 'LlGuidance', 'GrammarIntegration', 'Quantize', 'QuantizePerf', 'QuantizeStats', 'PtqCuda', 'Log', 'MoeReduce', 'PegBasic', 'PegGbnf', 'PegJson', 'PegSerialization', 'PegPythonDict', 'PegUnicode', 'PegParser', 'BatchAllocator', 'Allocator', 'Barrier', 'BackendSampler', 'BackendOps', 'ExportGraph', 'LoraConversion', 'TokenizersRepo', 'TokenizerPython', 'TokenizerShell', 'TokenizerRandom', 'ChatTemplate', 'MtmdCApi', 'Sampling', 'ChatPeg', 'Gguf', 'ChatAuto', 'Tokenizer0', 'ChatAnalysis', 'PublicC', 'DsparkLogSnr', 'DsparkForward', 'DsparkLoop', 'DsparkRealEval', 'GgufMetadata', 'QuantSelection', 'Architectures', 'RecurrentRollback', 'KvMeanCenter', 'Optimizer', 'DoubleFloat', 'GbnfValidator', 'ArgParser', 'Chat', 'ModelResolution', 'TokenizerSpm', 'TokenizerBpe', 'JsonSchema', 'Autorelease', 'ModelLoadCancel', 'SaveLoadState', 'StateRestoreFragmented', 'ThreadSafety')][string]$Case = 'All',
    [string]$CMakePath,
    [ValidateSet('Ninja', 'Visual Studio 18 2026', 'Visual Studio 17 2022')][string]$Generator = 'Ninja',
    [ValidateSet('ARM64', 'x64')][string]$NativePlatform = 'ARM64',
    [string]$Toolchain = 'nightly-2026-10-22',
    [string]$CompilerLauncher,
    [ValidateRange(1, 12)][int]$Parallel = 2,
    [ValidateRange(1, 12)][int]$CaseParallel = 2,
    [ValidateRange(-1, 3)][int]$SourcePartition = -1,
    [string]$NativeBuildDirectory,
    [string]$CommonToolsBundleDirectory,
    [switch]$DeferBackend,
    [switch]$PrepareBackendToolsOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
if ($Case -eq 'All' -and ($env:GITHUB_ACTIONS -cne 'true' -or $env:GITHUB_REF -cne 'refs/heads/main')) {
    throw 'The complete Prism catalog runs only in GitHub Actions on main; select one named local adapter.'
}
if ($PrepareBackendToolsOnly -and ($Case -cne 'BackendOps' -or $DeferBackend -or
    $env:GITHUB_ACTIONS -cne 'true' -or $env:GITHUB_REF -cne 'refs/heads/main' -or
    -not $IsWindows -or $NativePlatform -cne 'ARM64' -or
    [Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture -ne [Runtime.InteropServices.Architecture]::Arm64)) {
    throw 'Backend tool preparation requires exact Case BackendOps without DeferBackend on native Windows ARM64 main GitHub Actions.'
}
if ($DeferBackend -and $Case -cne 'All') { throw 'Only the complete CI catalog can defer its backend source to checked shards.' }
if ($SourcePartition -ge 0 -and ($Case -cne 'All' -or -not $DeferBackend)) {
    throw 'Ordinary source partitions require the complete deferred main CI catalog.'
}
if ($CommonToolsBundleDirectory -and ($Case -cne 'All' -or -not $DeferBackend -or
    $SourcePartition -notin @(0, 1, 3) -or $PrepareBackendToolsOnly -or $NativePlatform -cne 'ARM64' -or
    $env:GITHUB_ACTIONS -cne 'true' -or $env:GITHUB_REF -cne 'refs/heads/main' -or -not $IsWindows -or
    [Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture -ne [Runtime.InteropServices.Architecture]::Arm64)) {
    throw 'Shared native tools require ordinary partition 0, 1 or 3 on native Windows ARM64 main GitHub Actions.'
}
$repo = Split-Path $PSScriptRoot -Parent
Import-Module (Join-Path $repo 'scripts/Bonsai.Common.psm1')
Import-Module (Join-Path $repo 'scripts/Bonsai.PrismAssets.psm1')
Import-Module (Join-Path $repo 'scripts/Bonsai.Native.psm1')
Import-Module (Join-Path $repo 'scripts/Bonsai.PrismBaseline.psm1')
Import-Module (Join-Path $repo 'scripts/Bonsai.PrismNativeCache.psm1')
Import-Module (Join-Path $repo 'scripts/Bonsai.PrismRunner.psm1')
Import-Module (Join-Path $repo 'scripts/Bonsai.PrismSource.psm1')
$runtimeSource = Resolve-PrismSource
$archive = Resolve-ExternalPath $SourceArchive
$evidence = Resolve-ExternalPath $EvidenceDirectory
$target = Resolve-ExternalPath $TargetDirectory
if ($CompilerLauncher -and ($Generator -ne 'Ninja' -or
    -not [IO.Path]::IsPathFullyQualified($CompilerLauncher) -or $CompilerLauncher -match '%[^%]+%' -or
    $CompilerLauncher.Contains(';') -or -not [IO.File]::Exists($CompilerLauncher))) {
    throw 'CompilerLauncher requires Ninja and one existing expanded absolute executable without CMake list separators.'
}
if (Test-Path -LiteralPath $evidence) { throw 'Use a fresh evidence directory for each Prism import.' }
[void][IO.Directory]::CreateDirectory($evidence)
$lock = Get-Content -LiteralPath (Join-Path $runtimeSource 'tools/prism/source.lock.json') -Raw | ConvertFrom-Json -AsHashtable
$guard = [IO.FileStream]::new($archive, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
try {
    if ($guard.Length -ne $lock.ArchiveBytes -or
        [Convert]::ToHexStringLower([Security.Cryptography.SHA256]::HashData($guard)) -cne $lock.ArchiveSha256) {
        throw 'Prism source archive differs from its immutable pin.'
    }
    Expand-Archive -LiteralPath $archive -DestinationPath (Join-Path $evidence 'source')
} finally { $guard.Dispose() }
$source = Join-Path (Join-Path $evidence 'source') $lock.ArchiveRoot
$nativeBuildRoot = $evidence
$nativeSource = $source
$sharedTools = $null
if ($CommonToolsBundleDirectory) {
    Import-Module (Join-Path $repo 'scripts/Bonsai.PrismBackendBundle.psm1')
    $sharedTools = Import-BonsaiPrismBackendBundle -Directory $CommonToolsBundleDirectory -Repository $runtimeSource -Toolchain $Toolchain -RequireCommonTools
}
if ($NativeBuildDirectory -and -not $sharedTools) {
    $nativeBuildRoot = Resolve-ExternalPath $NativeBuildDirectory
    foreach ($other in @($evidence, $target)) {
        $prefix = $other.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
        $nativePrefix = $nativeBuildRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
        if ($nativeBuildRoot -eq $other -or $nativeBuildRoot.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) -or
            $other.StartsWith($nativePrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Native build cache, Rust targets and uploaded evidence must use separate directories.'
        }
    }
    $nativeSource = Initialize-BonsaiPrismNativeSource -SourceDirectory $source -BuildDirectory $nativeBuildRoot
}
$upstream = Join-Path $runtimeSource 'tools/prism/upstream'
$nativeAdapter = $upstream
if ($NativeBuildDirectory -and -not $sharedTools) {
    $nativeAdapter = Initialize-BonsaiPrismNativeAdapter -AdapterDirectory $upstream -BuildDirectory $nativeBuildRoot
}
$catalog = Import-BonsaiPrismCatalog -Path (Join-Path $upstream 'catalog.json') -SourceDirectory $source
$receipt = [ordered]@{
    SchemaVersion = 1
    SourceCommit = $lock.Commit
    SourceArchiveSha256 = $lock.ArchiveSha256
    ProjectRevision = $env:GITHUB_SHA
    RecordedAtUtc = [DateTime]::UtcNow.ToString('o')
    Platform = [Runtime.InteropServices.RuntimeInformation]::OSDescription
    Architecture = [Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString()
    ProcessorIdentifier = $env:PROCESSOR_IDENTIFIER
    LogicalProcessors = [Environment]::ProcessorCount
    Toolchain = $Toolchain
    Selection = $Case
    SourcePartition = $SourcePartition
    SourcePartitionCount = if ($SourcePartition -ge 0) { 4 } else { 1 }
    Generator = $Generator
    NativePlatform = $NativePlatform
    SourceDirectory = $source
    RuntimeSourceDirectory = $runtimeSource
    RuntimeManifestSha256 = (Get-FileHash -LiteralPath (Join-Path $runtimeSource 'Cargo.toml') -Algorithm SHA256).Hash.ToLowerInvariant()
    NativeSourceDirectory = $nativeSource
    NativeAdapterDirectory = $nativeAdapter
    NativeBuildDirectory = $nativeBuildRoot
    NativeToolset = if ($Generator -ne 'Ninja' -and $NativePlatform -eq 'ARM64') { 'ClangCL' } else { $null }
    CompilerLauncher = $CompilerLauncher
    ParallelJobs = $Parallel
    CaseParallelJobs = $CaseParallel
    Status = 'building'
}
if ($sharedTools) {
    $receipt.SharedTools = @{ Directory=$CommonToolsBundleDirectory; ManifestSha256=$sharedTools.ManifestSha256; SameRevision=$true }
    $receipt.ReferenceProductionFix = $sharedTools.ReferenceProductionFix
}
$receiptPath = Join-Path $evidence 'receipt.json'
[IO.File]::WriteAllText($receiptPath, ($receipt | ConvertTo-Json -Depth 8))
if (-not $sharedTools) {
    if (-not $CMakePath) { $CMakePath = (Get-Command cmake -CommandType Application -ErrorAction Stop).Source }
    if (-not [IO.Path]::IsPathFullyQualified($CMakePath) -or $CMakePath -match '%[^%]+%' -or -not [IO.File]::Exists($CMakePath)) {
        throw 'CMakePath must name an existing expanded absolute executable.'
    }
    $cargo = (Get-Command cargo -CommandType Application -ErrorAction Stop).Source
}
if ($Generator -ne 'Ninja' -and -not $IsWindows) { throw 'Visual Studio generators require Windows.' }
$script:upstreamCommandNumber = 0
function Invoke-UpstreamCommand {
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param([string]$Executable, [string[]]$Arguments, [int]$TimeoutSeconds = 900)
    $script:upstreamCommandNumber++
    $log = Join-Path $evidence ('command-{0:D2}.log' -f $script:upstreamCommandNumber)
    $result = Invoke-BonsaiNativeProcess -Executable $Executable -ArgumentList $Arguments -WorkingDirectory $evidence -TimeoutSeconds $TimeoutSeconds -Environment $compilerEnvironment -LogPath $log
    if ($result.ExitCode -ne 0) { throw "Prism import command failed with exit $($result.ExitCode); see $log" }
    return $result
}
$compilerEnvironment = @{}
$generatorArguments = @('-G', $Generator)
if ($Generator -ne 'Ninja') {
    $generatorArguments += @('-A', $NativePlatform)
    if ($NativePlatform -eq 'ARM64') { $generatorArguments += @('-T', 'ClangCL') }
}
if ($CompilerLauncher) {
    $generatorArguments += @("-DCMAKE_C_COMPILER_LAUNCHER=$CompilerLauncher", "-DCMAKE_CXX_COMPILER_LAUNCHER=$CompilerLauncher")
    $compilerEnvironment['CCACHE_BASEDIR'] = $evidence
}
$adapters = @{
    Pq2RowShapes = @{ Name = 'test-pq2-row-shapes'; Family = 'ggml' }
    Ptq1ElementMap = @{ Name = 'test-ptq1_0-element-map'; Family = 'ggml' }
    Unicode = @{ Name = 'test-unicode'; Family = 'common' }
    MtpCatchupBatch = @{ Name = 'test-mtp-catchup-batch'; Family = 'common' }
    Jinja = @{ Name = 'test-jinja'; Family = 'common' }
    MtmdImpl = @{ Name = 'test-mtmd-impl'; Family = 'common' }
    GrammarParser = @{ Name = 'test-grammar-parser'; Family = 'common' }
    ReasoningBudget = @{ Name = 'test-reasoning-budget'; Family = 'common' }
    Col2im = @{ Name = 'test-col2im-1d'; Family = 'common' }
    Dfly = @{ Name = 'test-dfly-fusion'; Family = 'common' }
    Rope = @{ Name = 'test-rope'; Family = 'common' }
    LlamaGrammar = @{ Name = 'test-llama-grammar'; Family = 'common' }
    LlGuidance = @{ Name = 'test-grammar-llguidance'; Family = 'common' }
    GrammarIntegration = @{ Name = 'test-grammar-integration'; Family = 'common' }
    Quantize = @{ Name = 'test-quantize-fns'; Family = 'common' }
    QuantizePerf = @{ Name = 'test-quantize-perf'; Family = 'common' }
    QuantizeStats = @{ Name = 'test-quantize-stats'; Family = 'common' }
    PtqCuda = @{ Name = 'test-ptq1_0-cuda-dot'; Family = 'common' }
    Log = @{ Name = 'test-log'; Family = 'common' }
    MoeReduce = @{ Name = 'test-moe-weighted-reduce'; Family = 'common' }
    PegBasic = @{ Name = 'peg-basic'; Id = 'tests/peg-parser/test-basic.cpp'; Family = 'common' }
    PegGbnf = @{ Name = 'peg-gbnf'; Id = 'tests/peg-parser/test-gbnf-generation.cpp'; Family = 'common' }
    PegJson = @{ Name = 'peg-json'; Id = 'tests/peg-parser/test-json-parser.cpp'; Family = 'common' }
    PegSerialization = @{ Name = 'peg-serialization'; Id = 'tests/peg-parser/test-json-serialization.cpp'; Family = 'common' }
    PegPythonDict = @{ Name = 'peg-python-dict'; Id = 'tests/peg-parser/test-python-dict-parser.cpp'; Family = 'common' }
    PegUnicode = @{ Name = 'peg-unicode'; Id = 'tests/peg-parser/test-unicode.cpp'; Family = 'common' }
    PegParser = @{ Name = 'test-peg-parser'; Family = 'common' }
    BatchAllocator = @{ Name = 'test-batch-alloc'; Family = 'common' }
    Allocator = @{ Name = 'test-alloc'; Family = 'common' }
    Barrier = @{ Name = 'test-barrier'; Family = 'common' }
    BackendSampler = @{ Name = 'test-backend-sampler'; Family = 'common' }
    BackendOps = @{ Name = 'test-backend-ops'; Family = 'common' }
    ExportGraph = @{ Name = 'test-export-graph-ops'; Family = 'common' }
    LoraConversion = @{ Name = 'test-lora-conversion-inference'; Id = 'tests/test-lora-conversion-inference.sh'; Family = 'common' }
    TokenizersRepo = @{ Name = 'test-tokenizers-repo'; Id = 'tests/test-tokenizers-repo.sh'; Family = 'common' }
    TokenizerPython = @{ Name = 'test-tokenizer-0-python'; Id = 'tests/test-tokenizer-0.py'; Family = 'independent' }
    TokenizerShell = @{ Name = 'test-tokenizer-0-shell'; Id = 'tests/test-tokenizer-0.sh'; Family = 'common' }
    TokenizerRandom = @{ Name = 'test-tokenizer-random'; Id = 'tests/test-tokenizer-random.py'; Family = 'common' }
    ChatTemplate = @{ Name = 'test-chat-template'; Family = 'common' }
    MtmdCApi = @{ Name = 'test-mtmd-c-api'; Id = 'tests/test-mtmd-c-api.c'; Family = 'common' }
    Sampling = @{ Name = 'test-sampling'; Family = 'common' }
    ChatPeg = @{ Name = 'test-chat-peg-parser'; Family = 'common' }
    Gguf = @{ Name = 'test-gguf'; Family = 'common' }
    ChatAuto = @{ Name = 'test-chat-auto-parser'; Family = 'common' }
    Tokenizer0 = @{ Name = 'test-tokenizer-0'; Family = 'common' }
    JsonSchema = @{ Name = 'test-json-schema-to-grammar'; Family = 'common' }
    ChatAnalysis = @{ Name = 'test-chat-analysis'; Family = 'common' }
    PublicC = @{ Name = 'test-c'; Id = 'tests/test-c.c'; Family = 'common' }
    DsparkLogSnr = @{ Name = 'test-dspark-logsnr-meta'; Family = 'common' }
    DsparkForward = @{ Name = 'test-dspark-forward'; Family = 'common' }
    DsparkLoop = @{ Name = 'test-dspark-loop'; Family = 'common' }
    DsparkRealEval = @{ Name = 'test-dspark-real-eval'; Family = 'common' }
    GgufMetadata = @{ Name = 'test-gguf-model-data'; Family = 'independent' }
    QuantSelection = @{ Name = 'test-quant-type-selection'; Family = 'common' }
    Architectures = @{ Name = 'test-llama-archs'; Family = 'common' }
    RecurrentRollback = @{ Name = 'test-recurrent-state-rollback'; Family = 'common' }
    KvMeanCenter = @{ Name = 'test-kv-mean-center'; Family = 'common' }
    Optimizer = @{ Name = 'test-opt'; Family = 'common' }
    DoubleFloat = @{ Name = 'test-double-float'; Family = 'independent' }
    GbnfValidator = @{ Name = 'test-gbnf-validator'; Family = 'common' }
    ArgParser = @{ Name = 'test-arg-parser'; Family = 'common' }
    Chat = @{ Name = 'test-chat'; Family = 'common' }
    ModelResolution = @{ Name = 'test-model-resolution'; Family = 'common' }
    TokenizerSpm = @{ Name = 'test-tokenizer-1-spm'; Family = 'common' }
    TokenizerBpe = @{ Name = 'test-tokenizer-1-bpe'; Family = 'common' }
    Autorelease = @{ Name = 'test-autorelease'; Family = 'common' }
    ModelLoadCancel = @{ Name = 'test-model-load-cancel'; Family = 'common' }
    SaveLoadState = @{ Name = 'test-save-load-state'; Family = 'common' }
    StateRestoreFragmented = @{ Name = 'test-state-restore-fragmented'; Family = 'common' }
    ThreadSafety = @{ Name = 'test-thread-safety'; Family = 'common' }
}
$selected = if ($Case -eq 'All') { @('Pq2RowShapes', 'Ptq1ElementMap', 'Unicode', 'MtpCatchupBatch', 'Jinja', 'MtmdImpl', 'GrammarParser', 'ReasoningBudget', 'Col2im', 'Dfly', 'Rope', 'LlamaGrammar', 'LlGuidance', 'GrammarIntegration', 'Quantize', 'QuantizePerf', 'QuantizeStats', 'PtqCuda', 'Log', 'MoeReduce', 'PegBasic', 'PegGbnf', 'PegJson', 'PegSerialization', 'PegPythonDict', 'PegUnicode', 'PegParser', 'BatchAllocator', 'Allocator', 'Barrier', 'BackendSampler', 'BackendOps', 'ExportGraph', 'LoraConversion', 'TokenizersRepo', 'TokenizerPython', 'TokenizerShell', 'TokenizerRandom', 'ChatTemplate', 'MtmdCApi', 'Sampling', 'ChatPeg', 'Gguf', 'ChatAuto', 'Tokenizer0', 'ChatAnalysis', 'PublicC', 'DsparkLogSnr', 'DsparkForward', 'DsparkLoop', 'DsparkRealEval', 'GgufMetadata', 'QuantSelection', 'Architectures', 'RecurrentRollback', 'KvMeanCenter', 'Optimizer', 'DoubleFloat', 'GbnfValidator', 'ArgParser', 'Chat', 'ModelResolution', 'TokenizerSpm', 'TokenizerBpe', 'JsonSchema', 'Autorelease', 'ModelLoadCancel', 'SaveLoadState', 'StateRestoreFragmented', 'ThreadSafety') } else { @($Case) }
if ($DeferBackend) { $selected = @($selected | Where-Object { $_ -cne 'BackendOps' }) }
if ($SourcePartition -ge 0) {
    Import-Module (Join-Path $repo 'scripts/Bonsai.PrismSourcePartition.psm1')
    $partitionCases = @(for ($ordinal = 0; $ordinal -lt $selected.Count; $ordinal++) {
        $selection = $selected[$ordinal]
        $adapter = $adapters[$selection]
        [pscustomobject]@{
            Ordinal = $ordinal; Selection = $selection
            Id = if ($adapter.ContainsKey('Id')) { $adapter.Id } else { "tests/$($adapter.Name).cpp" }
        }
    })
    $selected = @((Get-BonsaiPrismSourcePartition -Cases $partitionCases -Index $SourcePartition).Selection)
}
$modelSelections = @('ExportGraph', 'BackendSampler', 'Autorelease', 'ModelLoadCancel', 'SaveLoadState', 'StateRestoreFragmented', 'ThreadSafety')
$assetDirectory = $null
if (@($selected | Where-Object { $_ -in $modelSelections }).Count) {
    $assetDirectory = Join-Path $evidence 'model-assets'
    $modelLock = Get-Content -LiteralPath (Join-Path $upstream 'model-assets.lock.json') -Raw | ConvertFrom-Json -AsHashtable
    if ($modelLock.SchemaVersion -ne 1 -or $modelLock.Assets.Count -ne 1) { throw 'Unexpected model asset lock schema.' }
    $modelReceipts = [Collections.Generic.List[object]]::new()
    foreach ($asset in $modelLock.Assets) {
        if ($asset.Path -cne 'tinyllamas/stories15M-q4_0.gguf' -or
            $asset.Url -notlike 'https://huggingface.co/*' -or $asset.Sha256 -cnotmatch '^[0-9a-f]{64}$') {
            throw 'Unexpected or unpinned model asset.'
        }
        $destination = Join-Path $assetDirectory $asset.Path
        [void][IO.Directory]::CreateDirectory((Split-Path $destination -Parent))
        Invoke-WebRequest -Uri $asset.Url -OutFile $destination -TimeoutSec 180
        $modelGuard = [IO.FileStream]::new($destination, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
        try {
            $digest = [Convert]::ToHexStringLower([Security.Cryptography.SHA256]::HashData($modelGuard))
            if ($modelGuard.Length -ne $asset.Bytes -or $digest -cne $asset.Sha256) { throw 'Model asset differs from immutable byte identity.' }
            $modelReceipts.Add(@{ Path = $destination; Bytes = $modelGuard.Length; Sha256 = $digest; Url = $asset.Url })
        } finally { $modelGuard.Dispose() }
    }
    $receipt.ModelAssets = $modelReceipts.ToArray()
}
if (@($selected | Where-Object { $_ -in @('GgufMetadata', 'QuantSelection', 'ExportGraph') }).Count) {
    if (-not $assetDirectory) { $assetDirectory = Join-Path $evidence 'model-assets' }
    $receipt.MetadataAssets = Save-BonsaiPrismMetadataAsset -Directory $assetDirectory
    [IO.File]::WriteAllText($receiptPath, ($receipt | ConvertTo-Json -Depth 12))
}
$bridgeFamilies = @{
    ggml = @{ Source = $nativeAdapter; Target = 'prism-upstream-bridge'; Option = '--bridge' }
    common = @{ Source = (Join-Path $nativeAdapter 'common'); Target = 'prism-upstream-common-bridge'; Option = '--common-bridge' }
}
if ($selected -contains 'LoraConversion') {
    if (-not $assetDirectory) { $assetDirectory = Join-Path $evidence 'model-assets' }
    $receipt.LoraAssets = Save-BonsaiPrismLoraAsset -Directory $assetDirectory
    [IO.File]::WriteAllText($receiptPath, ($receipt | ConvertTo-Json -Depth 12))
}
if ($selected -contains 'TokenizersRepo') {
    if (-not $assetDirectory) { $assetDirectory = Join-Path $evidence 'model-assets' }
    $receipt.TokenizerRepository = Save-BonsaiPrismTokenizerRepository -Directory $assetDirectory
    [IO.File]::WriteAllText($receiptPath, ($receipt | ConvertTo-Json -Depth 12))
}
if (@($selected | Where-Object { $_ -in @('TokenizerPython','TokenizerShell','TokenizerRandom') }).Count) {
    if (-not $assetDirectory) { $assetDirectory = Join-Path $evidence 'model-assets' }
    $receipt.TokenizerReference = Save-BonsaiPrismTokenizerReference -Directory $assetDirectory
    [IO.File]::WriteAllText($receiptPath, ($receipt | ConvertTo-Json -Depth 12))
}
if ($selected -contains 'QuantizeStats') {
    if (-not $assetDirectory) { $assetDirectory = Join-Path $evidence 'model-assets' }
    $inputLock = Get-Content -LiteralPath (Join-Path $upstream 'quantize-stats-model.lock.json') -Raw | ConvertFrom-Json -AsHashtable
    $inputUrl = 'https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories42M.bin?download=true'
    if ($inputLock.SchemaVersion -ne 1 -or $inputLock.AssetPath -cne 'tinyllamas/stories42M.bin' -or
        $inputLock.Url -cne $inputUrl -or $inputLock.Bytes -ne 167020572 -or
        $inputLock.Sha256 -cne '9f65a1000e17d0bc167dd6332e0ce5119a0222a3d920cead5bce413bfab2ee7b') {
        throw 'The official unquantized input differs from its immutable identity.'
    }
    $path = Join-Path $assetDirectory $inputLock.AssetPath
    $null = Receive-LockedArtifact -Uri $inputLock.Url -Path $path -Bytes $inputLock.Bytes -Sha256 $inputLock.Sha256 -TimeoutSeconds 600
    $receipt.QuantizationInput = @{ Path=$path; Bytes=$inputLock.Bytes; Sha256=$inputLock.Sha256; Url=$inputLock.Url }
    [IO.File]::WriteAllText($receiptPath, ($receipt | ConvertTo-Json -Depth 12))
}
$bridges = @{}
$nativeBuilds = [Collections.Generic.List[object]]::new()
$extension = if ($IsWindows) { '.exe' } else { '' }
$grammarDependency = $null
if ($selected -contains 'LlGuidance') {
    if (-not $IsWindows) { throw 'The pinned grammar dependency acceptance lane targets Windows.' }
    Import-Module (Join-Path $repo 'scripts/Bonsai.PrismGrammar.psm1')
    $grammarTarget = if ($NativePlatform -eq 'ARM64') { 'aarch64-pc-windows-msvc' } else { 'x86_64-pc-windows-msvc' }
    $grammarDependency = Initialize-BonsaiPrismGrammarDependency -Repository $runtimeSource -Directory (Join-Path $nativeBuildRoot 'grammar') -Toolchain $Toolchain -TargetTriple $grammarTarget
    $receipt.GrammarDependency = $grammarDependency
    [IO.File]::WriteAllText($receiptPath, ($receipt | ConvertTo-Json -Depth 12))
}
$referencePatch = $null
if (-not $sharedTools -and @($selected | Where-Object { $adapters[$_].Family -eq 'common' }).Count) {
    Import-Module (Join-Path $repo 'scripts/Bonsai.PrismReferencePatch.psm1')
    $referencePatch = Initialize-BonsaiPrismReferencePatch -Repository $runtimeSource -SourceDirectory $nativeSource -Directory (Join-Path $nativeBuildRoot 'reference-patches')
    $receipt.ReferenceProductionFix = $referencePatch
    [IO.File]::WriteAllText($receiptPath, ($receipt | ConvertTo-Json -Depth 12))
}
$families = @($selected | ForEach-Object { $adapters[$_].Family } | Select-Object -Unique)
if ($PrepareBackendToolsOnly) { $families = @('common', 'ggml') }
foreach ($family in $families) {
    if ($family -eq 'independent') { continue }
    $definition = $bridgeFamilies[$family]
    if ($sharedTools) {
        $prepared = @($sharedTools.NativeBuilds | Where-Object { $_.Family -ceq $family })
        if ($prepared.Count -ne 1) { throw "Shared tool bundle lacks the $family production-call bridge." }
        $bridges[$family] = $prepared[0].Executable
        $nativeBuilds.Add($prepared[0])
        if ($family -ceq 'common' -and $selected -contains 'LoraConversion') { $receipt.LoraMergeTool = $sharedTools.LoraMergeTool }
        continue
    }
    $buildFamily = if ($PrepareBackendToolsOnly) { 'common' } else { $family }
    $build = Join-Path $nativeBuildRoot ("build-$buildFamily")
    $null = Resolve-ExternalPath $build
    $buildSource = $bridgeFamilies[$buildFamily].Source
    $configureArguments = @('-S', $buildSource, '-B', $build, "-DPRISM_SOURCE_DIR=$nativeSource", '-DCMAKE_BUILD_TYPE=Release',
        '-DLLAMA_BUILD_NUMBER=0', "-DLLAMA_BUILD_COMMIT=$($lock.Commit)") + $generatorArguments
    if ($buildFamily -eq 'common' -and $referencePatch) {
        foreach ($patch in $referencePatch.Patches) {
            $setting = switch -CaseSensitive ($patch.RelativeSource) {
                'src/llama-model-saver.cpp' { 'PRISM_PATCHED_SAVER' }
                'src/llama-vocab.cpp' { 'PRISM_PATCHED_VOCAB' }
                default { throw 'Unexpected verified production source substitution.' }
            }
            $configureArguments += "-D$setting=$($patch.Path)"
        }
    }
    if ($family -eq 'common' -and $grammarDependency) {
        $configureArguments += @("-DPRISM_LLGUIDANCE_LIBRARY=$($grammarDependency.Library)", "-DPRISM_LLGUIDANCE_INCLUDE_DIR=$($grammarDependency.IncludeDirectory)")
    }
    try {
        if (-not $PrepareBackendToolsOnly -or $family -ceq 'common') {
            $null = Invoke-UpstreamCommand -Executable $CMakePath -Arguments $configureArguments
            $targets = @($definition.Target)
            if ($PrepareBackendToolsOnly) { $targets += @('prism-upstream-bridge', 'llama-export-lora') }
            $null = Invoke-UpstreamCommand -Executable $CMakePath -Arguments (@('--build', $build, '--config', 'Release', '--parallel', [string]$Parallel, '--target') + $targets)
        }
    } finally {
        if ($NativeBuildDirectory) {
            $diagnostics = Join-Path $evidence ("native-$family")
            [void][IO.Directory]::CreateDirectory($diagnostics)
            foreach ($diagnostic in @('CMakeCache.txt', 'CMakeFiles/CMakeConfigureLog.yaml')) {
                $path = Join-Path $build $diagnostic
                if ([IO.File]::Exists($path)) { [IO.File]::Copy($path, (Join-Path $diagnostics ([IO.Path]::GetFileName($path))), $false) }
            }
        }
    }
    $executables = @(Get-ChildItem -LiteralPath $build -Recurse -File -Filter ($definition.Target + $extension))
    if ($executables.Count -ne 1) { throw "Expected exactly one $family production-call bridge." }
    $bridges[$family] = $executables[0].FullName
    if ($IsWindows -and $NativePlatform -eq 'ARM64') { Assert-Arm64PE -Path $bridges[$family] }
    if ($family -eq 'common' -and ($selected -contains 'LoraConversion' -or $PrepareBackendToolsOnly)) {
        if (-not $PrepareBackendToolsOnly) {
            $null = Invoke-UpstreamCommand -Executable $CMakePath -Arguments @('--build', $build, '--config', 'Release', '--parallel', [string]$Parallel, '--target', 'llama-export-lora')
        }
        $tools = @(Get-ChildItem -LiteralPath $build -Recurse -File -Filter ('llama-export-lora' + $extension))
        if ($tools.Count -ne 1 -or $tools[0].DirectoryName -cne [IO.Path]::GetDirectoryName($bridges[$family])) {
            throw 'Expected exactly one production LoRA export tool next to its common bridge.'
        }
        if ($IsWindows -and $NativePlatform -eq 'ARM64') { Assert-Arm64PE -Path $tools[0].FullName }
        $receipt.LoraMergeTool = @{
            ProjectRevision = $env:GITHUB_SHA
            Target = 'llama-export-lora'; Executable = $tools[0].FullName
            Sha256 = (Get-FileHash -LiteralPath $tools[0].FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            VerifiedPeMachine = if ($IsWindows -and $NativePlatform -eq 'ARM64') { '0xAA64' } else { $null }
        }
    }
    $nativeBuilds.Add([ordered]@{
        ProjectRevision = $env:GITHUB_SHA
        Family = $family
        ConfigureArguments = $configureArguments
        Target = $definition.Target
        CacheSha256 = (Get-FileHash -LiteralPath (Join-Path $build 'CMakeCache.txt') -Algorithm SHA256).Hash.ToLowerInvariant()
        Executable = $bridges[$family]
        ExecutableSha256 = (Get-FileHash -LiteralPath $bridges[$family] -Algorithm SHA256).Hash.ToLowerInvariant()
        VerifiedPeMachine = if ($IsWindows -and $NativePlatform -eq 'ARM64') { '0xAA64' } else { $null }
    })
}
$receipt.NativeBuilds = $nativeBuilds.ToArray()
[IO.File]::WriteAllText($receiptPath, ($receipt | ConvertTo-Json -Depth 10))
if ($sharedTools) {
    $harness = $sharedTools.HarnessPath
} else {
    $compiled = Invoke-UpstreamCommand -Executable $cargo -Arguments @("+$Toolchain", 'build', '--manifest-path', (Join-Path $runtimeSource 'Cargo.toml'), '--package', 'prism-inference', '--example', 'upstream_prism', '--locked', '--offline', '--release', '--target-dir', $target, '--message-format=json')
    $executables = @($compiled.StandardOutput -split "`n" | Where-Object { $_.StartsWith('{') } | ForEach-Object {
        $record = $_ | ConvertFrom-Json -AsHashtable
        if ($record.reason -eq 'compiler-artifact' -and $record.target.name -eq 'upstream_prism' -and $record.executable) { $record.executable }
    })
    if ($executables.Count -ne 1) { throw 'Expected exactly one compiled shared-assertion Rust harness.' }
    $harness = $executables[0]
}
if ($IsWindows -and $NativePlatform -eq 'ARM64') { Assert-Arm64PE -Path $harness }
$receipt.HarnessExecutable = $harness
$receipt.HarnessSha256 = (Get-FileHash -LiteralPath $harness -Algorithm SHA256).Hash.ToLowerInvariant()
$receipt.HarnessVerifiedPeMachine = if ($IsWindows -and $NativePlatform -eq 'ARM64') { '0xAA64' } else { $null }
if (($DeferBackend -and $SourcePartition -le 0 -and -not $sharedTools) -or $PrepareBackendToolsOnly) {
    Import-Module (Join-Path $repo 'scripts/Bonsai.PrismBackendBundle.psm1')
    $receipt.BackendBundleDirectory = Join-Path $evidence 'backend-tools'
    $receipt.BackendBundle = New-BonsaiPrismBackendBundle -Receipt $receipt -Directory $receipt.BackendBundleDirectory
    [IO.File]::WriteAllText($receiptPath, ($receipt | ConvertTo-Json -Depth 12))
}
if ($PrepareBackendToolsOnly) {
    $receipt.Status = 'backend-tools-ready'
    $receipt.Measure = 'backend-tools-preparation'
    $receipt.FullSourceGate = $false
    $receipt.SourceCount = 0
    $receipt.Validation = 'Verified build tools only; no source workload, correctness verdict or benchmark measurement was produced.'
    [IO.File]::WriteAllText($receiptPath, ($receipt | ConvertTo-Json -Depth 12))
    return $receipt
}
$caseWork = [Collections.Generic.List[object]]::new()
# Rust owns each selected test's inputs, reference calculations and assertions.
# The native bridge dispatches production engine calls without importing tests.
foreach ($selection in $selected) {
    $adapter = $adapters[$selection]
    $caseName = $adapter.Name
    $caseId = if ($adapter.ContainsKey('Id')) { $adapter.Id } else { "tests/$caseName.cpp" }
    $commands = [Collections.Generic.List[object]]::new()
    foreach ($engine in @('cpp', 'rust')) {
        $output = Join-Path $evidence "$caseName-$engine.json"
        $arguments = @('--target', $engine, '--case', $caseId, '--source-directory', $source, '--output', $output,
            '--evidence-directory', (Join-Path $evidence "$caseName-$engine-bridge"))
        if ($assetDirectory) { $arguments += @('--asset-directory', $assetDirectory) }
        if ($engine -eq 'cpp' -and $adapter.Family -ne 'independent') {
            $arguments += @($bridgeFamilies[$adapter.Family].Option, $bridges[$adapter.Family])
        }
        $timeout = if ($selection -eq 'TokenizerRandom') { 10800 } elseif ($selection -in @('TokenizerPython','TokenizerShell')) { 900 } elseif ($selection -eq 'QuantizeStats') { 14400 } elseif ($selection -eq 'QuantizePerf') { 900 } elseif ($selection -eq 'BackendOps') { 18000 } elseif ($selection -eq 'TokenizersRepo') { 900 } elseif ($selection -eq 'LoraConversion') { 1800 } elseif ($selection -eq 'ExportGraph') { 600 } elseif ($selection -eq 'DsparkForward') { 1800 } elseif ($selection -eq 'DsparkRealEval') { 960 } elseif ($selection -eq 'LlGuidance') { 1800 } elseif ($selection -eq 'Barrier') { 360 } elseif ($selection -eq 'Architectures') { 1800 } elseif ($selection -eq 'RecurrentRollback') { 600 } elseif ($selection -in @('KvMeanCenter','Optimizer')) { 600 } elseif ($selection -eq 'Chat') { 1200 } elseif ($selection -eq 'ModelResolution') { 360 } elseif ($selection -eq 'DoubleFloat') { 900 } elseif ($selection -eq 'TokenizerBpe') { 2100 } elseif ($selection -eq 'TokenizerSpm') { 600 } elseif ($selection -eq 'SaveLoadState') { 660 } elseif ($selection -eq 'ThreadSafety') { 1860 } elseif ($selection -in $modelSelections) { 360 } elseif ($selection -in @('Jinja', 'GrammarIntegration', 'Quantize', 'QuantizePerf', 'QuantizeStats', 'PtqCuda', 'ChatPeg', 'Gguf', 'ChatAuto', 'Tokenizer0', 'ChatAnalysis', 'PublicC', 'DsparkLogSnr', 'DsparkForward', 'DsparkLoop', 'DsparkRealEval', 'GgufMetadata', 'QuantSelection', 'Architectures', 'RecurrentRollback', 'KvMeanCenter', 'Optimizer', 'DoubleFloat', 'GbnfValidator', 'ArgParser', 'Chat', 'ModelResolution', 'TokenizerSpm', 'TokenizerBpe', 'JsonSchema', 'Autorelease', 'ModelLoadCancel', 'SaveLoadState', 'StateRestoreFragmented', 'ThreadSafety')) { 300 } else { 120 }
        $commands.Add([pscustomobject]@{
            Target = $engine; Arguments = $arguments; OutputPath = $output
            LogPath = Join-Path $evidence "$caseName-$engine.log"
        })
    }
    $caseWork.Add([pscustomobject]@{
        Ordinal = $caseWork.Count; Id = $caseId; TimeoutSeconds = $timeout; Commands = $commands.ToArray()
    })
}
$completed = @(Invoke-BonsaiPrismCaseBatch -Cases $caseWork.ToArray() -Executable $harness -EvidenceDirectory $evidence -UpstreamCommit $lock.Commit -Environment $compilerEnvironment -Parallel $CaseParallel)
$receipt.CaseRuns = @($completed | ForEach-Object { [ordered]@{ Id = $_.Id; Commands = $_.Commands } })
$reference = @{ SchemaVersion = 1; UpstreamCommit = $lock.Commit; Cases = @($completed.Reference) }
$rust = @{ SchemaVersion = 1; UpstreamCommit = $lock.Commit; Cases = @($completed.Rust) }
[IO.File]::WriteAllText((Join-Path $evidence 'cpp-results.json'), ($reference | ConvertTo-Json -Depth 10))
[IO.File]::WriteAllText((Join-Path $evidence 'rust-results.json'), ($rust | ConvertTo-Json -Depth 10))
if ($DeferBackend) {
    $receipt.DeferredSource = 'tests/test-backend-ops.cpp'
    $receipt.Validation = 'Ordinary results are retained; the complete source gate runs after every backend partition is combined.'
} elseif ($Case -eq 'All') {
    $report = New-BonsaiPrismReport -Catalog $catalog -TargetResults $rust -ReferenceResults $reference
    [IO.File]::WriteAllText((Join-Path $evidence 'report.json'), ($report | ConvertTo-Json -Depth 12))
    $gate = Test-BonsaiPrismBaseline -Catalog $catalog -Report $report -BaselinePath (Join-Path $upstream 'baseline.json')
    [IO.File]::WriteAllText((Join-Path $evidence 'gate.json'), ($gate | ConvertTo-Json -Depth 12))
    $receipt.Summary = $gate.Summary
    if (-not $gate.Passed) {
        $receipt.Status = 'failed'
        $receipt.Issues = $gate.Issues
        [IO.File]::WriteAllText($receiptPath, ($receipt | ConvertTo-Json -Depth 12))
        throw "Prism baseline regressed; all source contract failures are retained in $(Join-Path $evidence 'gate.json')"
    }
} else {
    $receipt.Results = $rust
    if (@($reference.Cases | Where-Object { $_.Status -cne 'pass' }).Count) {
        $receipt.Status = 'failed'
        [IO.File]::WriteAllText($receiptPath, ($receipt | ConvertTo-Json -Depth 12))
        throw 'The focused C++ reference contract failed; both target result files are retained.'
    }
}
$receipt.Status = if ($DeferBackend) { 'awaiting-backend' } else { 'complete' }
[IO.File]::WriteAllText($receiptPath, ($receipt | ConvertTo-Json -Depth 12))
return $receipt

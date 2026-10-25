#requires -Version 7.6
<#
.SYNOPSIS
Builds the pinned Prism storage overlay in a fresh external source/build lane.
.DESCRIPTION
Requires the exact source ZIP identified by tools/prism/source.lock.json. No
downloads, tests, benchmarks, model loading, installation, or profile promotion.
PrepareOnly builds Rust and its generated ABI header, then returns verified patched source.
Probe builds ggml and the Rust native alias test; Engine builds completion and server.
The pinned Rust toolchain builds the storage policy as a static library.
Parallel defaults to two compiler jobs; CI can explicitly choose a bounded budget.
CompilerLauncher optionally enables a compiler cache with the Ninja generator.
RustTargetDirectory optionally reuses Cargo build artifacts in external storage;
engine and native-test targets remain separate to preserve the policy archive.
#>
[CmdletBinding()]
[OutputType([Collections.IDictionary])]
param(
    [Parameter(Mandatory)][string]$SourceArchive,
    [string]$CMakePath,
    [ValidateSet('Probe','Engine','All')][string]$BuildMode = 'Probe',
    [ValidateSet('Ninja','Visual Studio 18 2026','Visual Studio 17 2022')][string]$Generator = 'Ninja',
    [ValidateRange(1,12)][int]$Parallel = 2,
    [string]$CompilerLauncher,
    [string]$RustTargetDirectory,
    [switch]$PrepareOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot 'Bonsai.Common.psm1') -Force
Import-Module (Join-Path $PSScriptRoot 'Bonsai.Native.psm1')
$archive = Resolve-ExternalPath -Path $SourceArchive
$overlay = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../tools/prism'))
$lock = Get-Content -LiteralPath (Join-Path $overlay 'source.lock.json') -Raw | ConvertFrom-Json -DateKind String
$scratch = Resolve-ExternalPath -Path (Join-Path ([IO.Path]::GetTempPath()) ('SwiftBonsai-PrismBuild-' + [guid]::NewGuid().ToString('N')))
[void][IO.Directory]::CreateDirectory($scratch)
if ($env:GITHUB_ACTIONS -eq 'true' -and $env:GITHUB_ENV) {
    $environmentPath = Resolve-ExternalPath -Path $env:GITHUB_ENV
    if ($scratch.IndexOfAny([char[]]"`r`n") -ge 0) { throw 'The evidence directory cannot contain line breaks.' }
    [IO.File]::AppendAllText($environmentPath, "BONSAI_PRISM_EVIDENCE=$scratch`n")
}
$guard = [IO.FileStream]::new($archive, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
try {
    if ($guard.Length -ne $lock.ArchiveBytes) { throw 'Pinned Prism source ZIP size mismatch.' }
    $archiveHash = [Convert]::ToHexStringLower([Security.Cryptography.SHA256]::HashData($guard))
    if ($archiveHash -cne $lock.ArchiveSha256) { throw 'Pinned Prism source ZIP hash mismatch.' }
    Expand-Archive -LiteralPath $archive -DestinationPath (Join-Path $scratch 'source')
} finally { $guard.Dispose() }
$script:prismCommandNumber = 0
function Invoke-PrismBuildCommand {
    [CmdletBinding()]
    [OutputType([string])]
    param([string[]]$Arguments, [string]$FilePath = $CMakePath, [Collections.IDictionary]$Environment)
    $script:prismCommandNumber++
    # Each build still uses fresh verified sources. ccache can normalize the
    # random lane prefix while retaining its compiler/header/option checks.
    $overrides = @{}
    if ($CompilerLauncher) { $overrides['CCACHE_BASEDIR'] = $scratch }
    if ($Environment) {
        foreach ($key in $Environment.Keys) { $overrides[$key] = $Environment[$key] }
    }
    $log = Join-Path $scratch ('command-{0:D2}.log' -f $script:prismCommandNumber)
    $result = Invoke-BonsaiNativeProcess -Executable $FilePath -ArgumentList $Arguments -WorkingDirectory $scratch -TimeoutSeconds 900 -Environment $overrides -LogPath $log
    if ($result.ExitCode) { throw "Prism build failed; see $log" }
    return ($result.StandardOutput + $result.StandardError).Trim()
}

$cargo = (Get-Command -Name cargo -CommandType Application -ErrorAction Stop).Source
$manifest = Join-Path $overlay 'Cargo.toml'
$rustRoot = if ($RustTargetDirectory) { Resolve-ExternalPath -Path $RustTargetDirectory } else { $scratch }
$rustBuild = Join-Path $rustRoot 'engine-loader'
# Opt-in native-test linking must never overwrite the policy archive used by
# the engine or invalidate its already recorded hash.
$rustNativeBuild = Join-Path $rustRoot 'native-tests'
$rustVersion = Invoke-PrismBuildCommand -FilePath $cargo -Arguments @('+nightly-2026-10-22','--version')
$cargoOutput = Invoke-PrismBuildCommand -FilePath $cargo -Arguments @('+nightly-2026-10-22','build','--locked','--offline','--release','--features','engine-loader','--message-format=json','--manifest-path',$manifest,'--target-dir',$rustBuild)
$records = @($cargoOutput -split "`n" | Where-Object { $_.StartsWith('{') } | ConvertFrom-Json -DateKind String)
$packages = @($records | Where-Object { $_.reason -eq 'compiler-artifact' -and $_.target.name -eq 'bonsai_prism' } | Select-Object -ExpandProperty package_id -Unique)
if ($packages.Count -ne 1) { throw 'Expected one Rust library package identity.' }
$buildScripts = @($records | Where-Object { $_.reason -eq 'build-script-executed' -and $_.package_id -ceq $packages[0] })
if ($buildScripts.Count -ne 1) { throw 'Expected one Rust header-generation build output.' }
$generatedHeader = Join-Path $buildScripts[0].out_dir 'bonsai_prism.h'
if (-not [IO.File]::Exists($generatedHeader)) { throw 'Rust did not generate its C ABI declarations.' }
$libraryName = if ($IsWindows) { 'bonsai_prism.lib' } else { 'libbonsai_prism.a' }
$rustLibrary = Join-Path $rustBuild ('release/' + $libraryName)
if (-not [IO.File]::Exists($rustLibrary)) { throw 'The Rust static library was not produced.' }
$rustReceipt = [ordered]@{
    Version = $rustVersion
    GeneratedHeader = $generatedHeader
    HeaderSha256 = (Get-FileHash -LiteralPath $generatedHeader -Algorithm SHA256).Hash.ToLowerInvariant()
    Library = $rustLibrary
    LibrarySha256 = (Get-FileHash -LiteralPath $rustLibrary -Algorithm SHA256).Hash.ToLowerInvariant()
    ManifestSha256 = (Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash.ToLowerInvariant()
    WorkspaceManifestSha256 = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot '../Cargo.toml') -Algorithm SHA256).Hash.ToLowerInvariant()
    LockSha256 = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot '../Cargo.lock') -Algorithm SHA256).Hash.ToLowerInvariant()
}

$source = Join-Path $scratch ('source/' + $lock.ArchiveRoot)
$overlayReceipt = & (Join-Path $overlay 'Apply-Overlay.ps1') -SourceDirectory $source -GeneratedHeader $generatedHeader
$receipt = [ordered]@{
    SchemaVersion = 1
    RecordedAtUtc = [DateTime]::UtcNow.ToString('o')
    SourceCommit = $lock.Commit
    SourceArchiveSha256 = $archiveHash
    SourceArchiveBytes = $lock.ArchiveBytes
    Overlay = $overlayReceipt
    SourceDirectory = $source
    EvidenceDirectory = $scratch
    ProbeSourceDirectory = $overlay
    BuildMode = $BuildMode
    Generator = $Generator
    ParallelJobs = $Parallel
    CompilerLauncher = $null
    Rust = $rustReceipt
    Binaries = @()
    BuildConfigurations = @()
    Status = 'prepared'
    Qualification = 'source/build candidate only; no TPS, output-quality, or model qualification'
}
[IO.File]::WriteAllText((Join-Path $scratch 'build-receipt.json'), ($receipt | ConvertTo-Json -Depth 7))
if (-not $PrepareOnly) {
    if (-not $CMakePath) { $CMakePath = (Get-Command -Name cmake -CommandType Application -ErrorAction Stop).Source }
    if (-not [IO.Path]::IsPathFullyQualified($CMakePath) -or $CMakePath -match '%[^%]+%' -or -not [IO.File]::Exists($CMakePath)) { throw 'CMakePath must name an existing expanded absolute executable.' }
    if ($Generator -ne 'Ninja' -and -not $IsWindows) { throw 'Visual Studio generators require Windows.' }
    if ($CompilerLauncher) {
        if ($Generator -ne 'Ninja') { throw 'CompilerLauncher requires the Ninja generator.' }
        if (-not [IO.Path]::IsPathFullyQualified($CompilerLauncher) -or $CompilerLauncher -match '%[^%]+%' -or
            $CompilerLauncher.Contains(';') -or -not [IO.File]::Exists($CompilerLauncher)) {
            throw 'CompilerLauncher must name one existing expanded absolute executable without CMake list separators.'
        }
        $CompilerLauncher = [IO.Path]::GetFullPath($CompilerLauncher)
        $receipt.CompilerLauncher = [ordered]@{
            Path = $CompilerLauncher
            Sha256 = (Get-FileHash -LiteralPath $CompilerLauncher -Algorithm SHA256).Hash.ToLowerInvariant()
            CacheBaseDirectory = $scratch
        }
        [IO.File]::WriteAllText((Join-Path $scratch 'build-receipt.json'), ($receipt | ConvertTo-Json -Depth 7))
    }
    $receipt.CMakeVersion = Invoke-PrismBuildCommand -Arguments @('--version')
    $generatorArguments = @('-G', $Generator)
    if ($Generator -ne 'Ninja') { $generatorArguments += @('-A','ARM64','-T','ClangCL') }
    if ($CompilerLauncher) {
        # Use exactly one explicit launcher instead of ggml's auto-detected cache.
        $generatorArguments += @('-DGGML_CCACHE=OFF', "-DCMAKE_C_COMPILER_LAUNCHER=$CompilerLauncher", "-DCMAKE_CXX_COMPILER_LAUNCHER=$CompilerLauncher")
    }
    $modes = if ($BuildMode -eq 'All') { @('Probe','Engine') } else { @($BuildMode) }
    $binaries = [Collections.Generic.List[object]]::new()
    $configurations = [Collections.Generic.List[object]]::new()
    foreach ($mode in $modes) {
        $build = Join-Path $scratch ('build-' + $mode.ToLowerInvariant())
        if ($mode -eq 'Probe') {
            $arguments = @('-S',$overlay,'-B',$build) + $generatorArguments + @("-DPRISM_SOURCE_DIR=$source",'-DCMAKE_BUILD_TYPE=Release','-DFETCHCONTENT_FULLY_DISCONNECTED=ON')
            $targets = @('ggml','ggml-cpu')
        } else {
            $arguments = @('-S',$source,'-B',$build) + $generatorArguments + @(
                '-DCMAKE_BUILD_TYPE=Release','-DBUILD_SHARED_LIBS=OFF','-DFETCHCONTENT_FULLY_DISCONNECTED=ON',
                '-DLLAMA_BUILD_NUMBER=0',"-DLLAMA_BUILD_COMMIT=$($lock.Commit)+swift-lora-b-share-rust-v3",
                "-DBONSAI_PRISM_LIBRARY=$rustLibrary",
                '-DGGML_NATIVE=OFF','-DGGML_BACKEND_DL=OFF',
                '-DGGML_CPU_ALL_VARIANTS=OFF','-DGGML_OPENMP=OFF','-DGGML_CUDA=OFF',
                '-DGGML_METAL=OFF','-DGGML_VULKAN=OFF','-DGGML_OPENCL=OFF','-DGGML_BLAS=OFF',
                '-DGGML_RPC=OFF','-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_APP=OFF',
                '-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_BUILD_TOOLS=ON','-DLLAMA_BUILD_SERVER=ON',
                '-DLLAMA_BUILD_UI=OFF','-DLLAMA_USE_PREBUILT_UI=OFF','-DLLAMA_OPENSSL=OFF',
                '-DLLAMA_BUILD_BORINGSSL=OFF','-DLLAMA_BUILD_LIBRESSL=OFF','-DLLAMA_SUBPROCESS=OFF','-DLLAMA_LLGUIDANCE=OFF')
            $targets = @('llama-completion','llama-server')
        }
        $null = Invoke-PrismBuildCommand -Arguments $arguments
        $null = Invoke-PrismBuildCommand -Arguments (@('--build',$build,'--config','Release','--parallel',[string]$Parallel,'--target') + $targets)
        if ($mode -eq 'Probe') {
            $nativeLibraries = @(Get-ChildItem -LiteralPath $build -Recurse -File | Where-Object { $_.Name -in @('ggml.lib','ggml-cpu.lib','ggml-base.lib','libggml.a','libggml-cpu.a','libggml-base.a') })
            if ($nativeLibraries.Count -ne 3) { throw 'Expected the three pinned ggml native libraries.' }
            $libraryDirectories = @($nativeLibraries.DirectoryName | Sort-Object -Unique)
            $nativeEnvironment = @{ BONSAI_PRISM_NATIVE_LIB_DIR = $libraryDirectories -join [IO.Path]::PathSeparator }
            $cargoOutput = Invoke-PrismBuildCommand -FilePath $cargo -Environment $nativeEnvironment -Arguments @(
                '+nightly-2026-10-22','test','--locked','--offline','--release','--manifest-path',$manifest,'--target-dir',$rustNativeBuild,
                '--features','native-tests','--test','native','--no-run','--message-format=json')
            $executables = @($cargoOutput -split "`n" | Where-Object { $_.StartsWith('{') } | ForEach-Object {
                $record = $_ | ConvertFrom-Json -DateKind String
                if ($record.reason -eq 'compiler-artifact' -and $record.target.name -eq 'native' -and $record.executable) { $record.executable }
            })
            if ($executables.Count -ne 1) { throw 'Expected one Rust native test executable.' }
            $nativeTest = $executables[0]
            $null = Invoke-PrismBuildCommand -Arguments ($arguments + @("-DBONSAI_PRISM_NATIVE_TEST=$nativeTest"))
            $binaries.Add([ordered]@{ Target = 'prism-rust-native-test'; Path = $nativeTest; Sha256 = (Get-FileHash -LiteralPath $nativeTest -Algorithm SHA256).Hash.ToLowerInvariant() })
        }
        $cache = Join-Path $build 'CMakeCache.txt'
        $configurations.Add([ordered]@{
            Mode = $mode
            ConfigureArguments = $arguments
            CacheSha256 = (Get-FileHash -LiteralPath $cache -Algorithm SHA256).Hash.ToLowerInvariant()
            Compiler = @(Get-ChildItem -Path (Join-Path $build 'CMakeFiles/*/CMakeCXXCompiler.cmake') | ForEach-Object {
                @(Get-Content -LiteralPath $_.FullName | Where-Object { $_ -match '^set\(CMAKE_CXX_COMPILER(_ID|_VERSION|_ARCHITECTURE_ID|_FRONTEND_VARIANT|_TARGET)? ' })
            })
            Options = @(Get-Content -LiteralPath $cache | Where-Object { $_ -match '^(CMAKE_BUILD_TYPE|BUILD_SHARED_LIBS|FETCHCONTENT_FULLY_DISCONNECTED|GGML_\w+|LLAMA_\w+):BOOL=' })
        })
        foreach ($target in @($targets | Where-Object { $_ -like 'llama-*' })) {
            $extension = if ($IsWindows) { '.exe' } else { '' }
            $builtFiles = @(Get-ChildItem -LiteralPath $build -Recurse -File -Filter ($target + $extension))
            if ($builtFiles.Count -ne 1) { throw "Expected one built $target executable." }
            $binary = $builtFiles[0].FullName
            $binaries.Add([ordered]@{ Target = $target; Path = $binary; Sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant() })
        }
    }
    $receipt.Binaries = $binaries.ToArray()
    $receipt.BuildConfigurations = $configurations.ToArray()
    $receipt.Status = 'built'
}
[IO.File]::WriteAllText((Join-Path $scratch 'build-receipt.json'), ($receipt | ConvertTo-Json -Depth 7))
return $receipt

#requires -Version 7.6
<#
.SYNOPSIS
Applies the exact CPU LoRA B storage overlay to a verified pinned Prism tree.
.DESCRIPTION
Rejects source drift and repeat application. Changes only src/llama-adapter.cpp
and src/CMakeLists.txt, and adds the Rust library's C ABI declarations.
It does not build, download, or run inference.
#>
[CmdletBinding()]
[OutputType([Collections.IDictionary])]
param(
    [Parameter(Mandatory)][string]$SourceDirectory,
    [Parameter(Mandatory)][string]$GeneratedHeader
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
if (-not [IO.Path]::IsPathFullyQualified($SourceDirectory) -or $SourceDirectory -match '%[^%]+%') { throw 'SourceDirectory must be expanded and absolute.' }
Import-Module (Join-Path $PSScriptRoot '../../scripts/Bonsai.Common.psm1') -Force
$source = Resolve-ExternalPath -Path $SourceDirectory
$generated = Resolve-ExternalPath -Path $GeneratedHeader
if (-not [IO.File]::Exists($generated)) { throw 'GeneratedHeader must name the Rust-generated C ABI header.' }
$adapterPath = Resolve-ExternalPath -Path (Join-Path $source 'src/llama-adapter.cpp')
$headerPath = Resolve-ExternalPath -Path (Join-Path $source 'src/bonsai_prism.h')
$cmakePath = Resolve-ExternalPath -Path (Join-Path $source 'src/CMakeLists.txt')
$lock = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'source.lock.json') -Raw | ConvertFrom-Json -DateKind String
if ((Get-FileHash -LiteralPath $adapterPath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $lock.AdapterSourceSha256) { throw 'Pinned llama-adapter.cpp source hash mismatch.' }
if ((Get-FileHash -LiteralPath $cmakePath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $lock.CMakeSourceSha256) { throw 'Pinned src/CMakeLists.txt source hash mismatch.' }
if ([IO.File]::Exists($headerPath)) { throw 'Overlay header already exists; use a new extracted source tree.' }
$text = [IO.File]::ReadAllText($adapterPath)
function Edit-ExactText {
    [CmdletBinding()]
    [OutputType([string])]
    param([string]$Text, [string]$Before, [string]$After)
    # Git may check this script out with CRLF on Windows; the hash-pinned
    # upstream source and deterministic output use LF on every platform.
    $Before = $Before.Replace("`r`n", "`n")
    $After = $After.Replace("`r`n", "`n")
    $first = $Text.IndexOf($Before, [StringComparison]::Ordinal)
    if ($first -lt 0 -or $Text.IndexOf($Before, $first + $Before.Length, [StringComparison]::Ordinal) -ge 0) { throw 'Expected a unique exact overlay insertion point.' }
    return $Text.Replace($Before, $After, [StringComparison]::Ordinal)
}
function Edit-ExactRegion {
    [CmdletBinding()]
    [OutputType([string])]
    param([string]$Text, [string]$Start, [string]$End, [string]$After)
    $first = $Text.IndexOf($Start, [StringComparison]::Ordinal)
    $last = $Text.IndexOf($End, [StringComparison]::Ordinal)
    if ($first -lt 0 -or $last -le $first -or
        $Text.IndexOf($Start, $first + $Start.Length, [StringComparison]::Ordinal) -ge 0 -or
        $Text.IndexOf($End, $last + $End.Length, [StringComparison]::Ordinal) -ge 0) {
        throw 'Expected unique ordered overlay region markers.'
    }
    return $Text.Substring(0, $first) + $After.Replace("`r`n", "`n") + "`n" + $Text.Substring($last)
}
# These are only ownership, call and exception adapters for the external engine.
# Tensor metadata, file I/O, bounds checks, view creation and loading live in Rust.
$text = Edit-ExactText -Text $text -Before '#include "llama-adapter.h"' -After "#include `"llama-adapter.h`"`n#include `"bonsai_prism.h`"`n#include <memory>"
$text = Edit-ExactText -Text $text -Before '    // add tensors' -After @'
    std::unique_ptr<NativeLoader, decltype(&bonsai_prism_loader_free)> rust_loader(
        bonsai_prism_loader_new(path_lora), &bonsai_prism_loader_free);
    if (!rust_loader) { throw std::runtime_error("Rust LoRA loader could not open adapter"); }

    // add tensors
'@
$text = Edit-ExactText -Text $text -Before '        ggml_tensor * tensor_b = ggml_dup_tensor(dev_ctx, w.b);' -After @'
        auto * tensor_b = static_cast<ggml_tensor *>(bonsai_prism_loader_make(rust_loader.get(), ctx_gguf.get(), dev_ctx, w.b, buft));
        if (!tensor_b) { throw std::runtime_error(bonsai_prism_loader_error(rust_loader.get())); }
'@
$text = Edit-ExactRegion -Text $text -Start '        llama_file gguf_file(path_lora, "rb");' -End '        for (auto & it : adapter.ab_map)' -After @'
        auto set_tensor = [&](ggml_tensor * orig, ggml_tensor * dev) {
            if (!bonsai_prism_loader_load(rust_loader.get(), ctx_gguf.get(), orig, dev)) {
                throw std::runtime_error(bonsai_prism_loader_error(rust_loader.get()));
            }
        };
'@
$bytes = [Text.UTF8Encoding]::new($false).GetBytes($text)
$outputHash = [Convert]::ToHexStringLower([Security.Cryptography.SHA256]::HashData($bytes))
if ($outputHash -cne $lock.OverlaidAdapterSha256) { throw "Overlay output hash mismatch ($outputHash); no source file was changed." }
$cmake = [IO.File]::ReadAllText($cmakePath)
$cmake = Edit-ExactText -Text $cmake -Before 'target_link_libraries(llama PUBLIC ggml)' -After @'
target_link_libraries(llama PUBLIC ggml)
if(NOT BONSAI_PRISM_LIBRARY OR NOT EXISTS "${BONSAI_PRISM_LIBRARY}")
    message(FATAL_ERROR "BONSAI_PRISM_LIBRARY must name the built Rust static library")
endif()
add_library(bonsai-prism STATIC IMPORTED)
set_target_properties(bonsai-prism PROPERTIES IMPORTED_LOCATION "${BONSAI_PRISM_LIBRARY}")
target_link_libraries(bonsai-prism INTERFACE ggml ggml-cpu ggml-base)
target_link_libraries(llama PRIVATE bonsai-prism)
if(WIN32)
    target_link_libraries(llama PRIVATE ws2_32 userenv ntdll dbghelp)
else()
    target_link_libraries(llama PRIVATE ${CMAKE_DL_LIBS} pthread m)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_libraries(llama PRIVATE rt util)
    endif()
endif()
'@
$cmakeBytes = [Text.UTF8Encoding]::new($false).GetBytes($cmake)
$cmakeHash = [Convert]::ToHexStringLower([Security.Cryptography.SHA256]::HashData($cmakeBytes))
if ($cmakeHash -cne $lock.OverlaidCMakeSha256) { throw "Overlay CMake output hash mismatch ($cmakeHash); no source file was changed." }
# Validate both deterministic outputs before publishing any source changes.
[IO.File]::Copy($generated, $headerPath, $false)
[IO.File]::WriteAllBytes($adapterPath, $bytes)
[IO.File]::WriteAllBytes($cmakePath, $cmakeBytes)
return [ordered]@{
    UpstreamCommit = $lock.Commit
    AdapterSourceSha256 = $lock.AdapterSourceSha256
    OverlaidAdapterSha256 = $outputHash
    CMakeSourceSha256 = $lock.CMakeSourceSha256
    OverlaidCMakeSha256 = $cmakeHash
    GeneratedHeader = $generated
    HeaderSha256 = (Get-FileHash -LiteralPath $headerPath -Algorithm SHA256).Hash.ToLowerInvariant()
    SourceDirectory = $source
    Qualification = 'CPU storage-only overlay; throughput and output-quality qualification is still required'
}

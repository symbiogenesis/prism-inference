#requires -Version 7.6
<#
.SYNOPSIS
Checks native source reuse and rejects changed or incomplete cached source trees.
#>
[CmdletBinding()]
[OutputType([string])]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot 'Bonsai.TestSupport.psm1')
Import-Module (Join-Path $PSScriptRoot '../scripts/Bonsai.PrismNativeCache.psm1')
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('SwiftBonsai-PrismCacheTest-' + [guid]::NewGuid().ToString('N'))
$source = Join-Path $scratch 'verified'
$cache = Join-Path $scratch 'native'
[void][IO.Directory]::CreateDirectory((Join-Path $source 'nested'))
[IO.File]::WriteAllText((Join-Path $source 'CMakeLists.txt'), 'verified build configuration')
[IO.File]::WriteAllText((Join-Path $source 'nested/kernel.cpp'), 'verified native source')
[void][IO.Directory]::CreateDirectory((Join-Path $source 'empty'))
$hidden = Join-Path $source '.hidden.h'
[IO.File]::WriteAllText($hidden, 'hidden native input')
if ($IsWindows) { [IO.File]::SetAttributes($hidden, [IO.FileAttributes]::Hidden) }
$preview = Join-Path $scratch 'preview'
Assert ($null -eq (Initialize-BonsaiPrismNativeSource -SourceDirectory $source -BuildDirectory $preview -WhatIf)) 'Source WhatIf returned an uncreated cache.'
Assert (-not (Test-Path -LiteralPath $preview)) 'Source WhatIf created a cache.'
$cached = Initialize-BonsaiPrismNativeSource -SourceDirectory $source -BuildDirectory $cache
Assert ([IO.Directory]::Exists((Join-Path $cached 'empty'))) 'Source caching discarded an empty directory.'
Assert ([IO.File]::ReadAllText((Join-Path $cached '.hidden.h')) -ceq 'hidden native input') 'Source caching discarded a hidden input.'
$kernel = Join-Path $cached 'nested/kernel.cpp'
$timestamp = [IO.File]::GetLastWriteTimeUtc($kernel)
[IO.File]::SetLastWriteTimeUtc((Join-Path $source 'nested/kernel.cpp'), $timestamp.AddHours(1))
Assert ((Initialize-BonsaiPrismNativeSource -SourceDirectory $source -BuildDirectory $cache) -ceq $cached) 'An intact source tree was not reusable.'
Assert ([IO.File]::GetLastWriteTimeUtc($kernel) -eq $timestamp) 'Fresh extraction timestamps invalidated native objects.'
[IO.File]::WriteAllText($kernel, 'modified native source')
Assert-Failure { Initialize-BonsaiPrismNativeSource -SourceDirectory $source -BuildDirectory $cache } 'A modified cached source was accepted.'
[IO.File]::Copy((Join-Path $source 'nested/kernel.cpp'), $kernel, $true)
$extra = Join-Path $cached 'injected.cpp'
[IO.File]::WriteAllText($extra, 'unlisted')
Assert-Failure { Initialize-BonsaiPrismNativeSource -SourceDirectory $source -BuildDirectory $cache } 'An unlisted cached source was accepted.'
[IO.File]::Delete($extra)
[IO.File]::Delete($kernel)
Assert-Failure { Initialize-BonsaiPrismNativeSource -SourceDirectory $source -BuildDirectory $cache } 'An incomplete cached tree was accepted.'
Assert-Failure { Initialize-BonsaiPrismNativeSource -SourceDirectory $source -BuildDirectory (Join-Path $source 'cache') } 'Overlapping source and cache paths were accepted.'
$orphaned = Join-Path $scratch 'orphaned-native'
[void][IO.Directory]::CreateDirectory((Join-Path $orphaned 'build-ggml'))
[IO.File]::WriteAllText((Join-Path $orphaned 'build-ggml/kernel.obj'), 'stale native output')
Assert-Failure { Initialize-BonsaiPrismNativeSource -SourceDirectory $source -BuildDirectory $orphaned } 'Native outputs without their verified source tree were accepted.'
Assert (-not [IO.Directory]::Exists((Join-Path $orphaned 'source'))) 'Orphaned native outputs were rehydrated with old source timestamps.'
# Grammar dependency extraction paths change on every invocation. Cargo must see
# the same verified source path and header timestamps across those extractions.
$grammarCache = Join-Path $scratch 'grammar/verified'
$stableGrammar = Initialize-BonsaiPrismNativeSource -SourceDirectory $source -BuildDirectory $grammarCache
$stableGrammarKernel = Join-Path $stableGrammar 'nested/kernel.cpp'
$stableGrammarTimestamp = [IO.File]::GetLastWriteTimeUtc($stableGrammarKernel)
$nextExtraction = Join-Path $scratch 'next-extraction'
[void][IO.Directory]::CreateDirectory((Join-Path $nextExtraction 'nested'))
[void][IO.Directory]::CreateDirectory((Join-Path $nextExtraction 'empty'))
[IO.File]::Copy($hidden, (Join-Path $nextExtraction '.hidden.h'), $false)
[IO.File]::Copy((Join-Path $source 'CMakeLists.txt'), (Join-Path $nextExtraction 'CMakeLists.txt'), $false)
[IO.File]::Copy((Join-Path $source 'nested/kernel.cpp'), (Join-Path $nextExtraction 'nested/kernel.cpp'), $false)
[IO.File]::SetLastWriteTimeUtc((Join-Path $nextExtraction 'nested/kernel.cpp'), $stableGrammarTimestamp.AddHours(2))
Assert ((Initialize-BonsaiPrismNativeSource -SourceDirectory $nextExtraction -BuildDirectory $grammarCache) -ceq $stableGrammar) 'A fresh extraction changed the verified grammar source path.'
Assert ([IO.File]::GetLastWriteTimeUtc($stableGrammarKernel) -eq $stableGrammarTimestamp) 'A fresh extraction changed a verified grammar header timestamp.'
[IO.File]::WriteAllText($stableGrammarKernel, 'modified grammar source')
Assert-Failure { Initialize-BonsaiPrismNativeSource -SourceDirectory $nextExtraction -BuildDirectory $grammarCache } 'A fresh extraction accepted altered grammar mirror bytes.'
$adapter = Join-Path $scratch 'checkout-adapter'
[void][IO.Directory]::CreateDirectory((Join-Path $adapter 'common'))
[IO.File]::WriteAllText((Join-Path $adapter 'CMakeLists.txt'), 'adapter build configuration')
[IO.File]::WriteAllText((Join-Path $adapter 'common/bridge.cpp'), 'adapter production calls')
[IO.File]::WriteAllText((Join-Path $adapter 'common/operations.hpp'), 'adapter production operations')
[IO.File]::WriteAllText((Join-Path $adapter 'catalog.json'), 'non-native evidence policy')
Assert ($null -eq (Initialize-BonsaiPrismNativeAdapter -AdapterDirectory $adapter -BuildDirectory $preview -WhatIf)) 'Adapter WhatIf returned an uncreated cache.'
Assert (-not (Test-Path -LiteralPath $preview)) 'Adapter WhatIf created a cache.'
$cachedAdapter = Initialize-BonsaiPrismNativeAdapter -AdapterDirectory $adapter -BuildDirectory $cache
Assert (-not [IO.File]::Exists((Join-Path $cachedAdapter 'catalog.json'))) 'Native caches must not bind unrelated evidence policy files.'
$bridge = Join-Path $cachedAdapter 'common/bridge.cpp'
$bridgeTimestamp = [IO.File]::GetLastWriteTimeUtc($bridge)
[IO.File]::SetLastWriteTimeUtc((Join-Path $adapter 'common/bridge.cpp'), $bridgeTimestamp.AddHours(1))
Assert ((Initialize-BonsaiPrismNativeAdapter -AdapterDirectory $adapter -BuildDirectory $cache) -ceq $cachedAdapter) 'A verified native adapter was not reusable.'
Assert ([IO.File]::GetLastWriteTimeUtc($bridge) -eq $bridgeTimestamp) 'Fresh checkout timestamps invalidated the native bridge.'
[IO.File]::WriteAllText((Join-Path $adapter 'catalog.json'), 'changed non-native evidence policy')
[void](Initialize-BonsaiPrismNativeAdapter -AdapterDirectory $adapter -BuildDirectory $cache)
[IO.File]::WriteAllText($bridge, 'modified adapter production calls')
Assert-Failure { Initialize-BonsaiPrismNativeAdapter -AdapterDirectory $adapter -BuildDirectory $cache } 'A modified cached native adapter was accepted.'
[IO.File]::Copy((Join-Path $adapter 'common/bridge.cpp'), $bridge, $true)
$injected = Join-Path $cachedAdapter 'unlisted.json'
[IO.File]::WriteAllText($injected, 'unlisted cache file')
Assert-Failure { Initialize-BonsaiPrismNativeAdapter -AdapterDirectory $adapter -BuildDirectory $cache } 'An unlisted cached adapter file was accepted.'
[IO.File]::Delete($injected)
[IO.File]::Delete($bridge)
Assert-Failure { Initialize-BonsaiPrismNativeAdapter -AdapterDirectory $adapter -BuildDirectory $cache } 'An incomplete native adapter cache was accepted.'
Assert-Failure { Initialize-BonsaiPrismNativeAdapter -AdapterDirectory $adapter -BuildDirectory (Join-Path $adapter 'cache') } 'Overlapping adapter and cache paths were accepted.'
Write-Output "Prism native cache source and adapter identity/timestamp checks passed. Evidence: $scratch"

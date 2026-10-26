#requires -Version 7.6
<# .SYNOPSIS
Checks verified immutable fixture reuse with fresh, exclusive evidence copies.
#>
[CmdletBinding()]
[OutputType([string])]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot 'Bonsai.TestSupport.psm1')
$assets = Import-Module (Join-Path $PSScriptRoot '../scripts/Bonsai.PrismAssets.psm1') -PassThru
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('Prism-FixtureCache-' + [guid]::NewGuid().ToString('N'))
$cache = Join-Path $scratch 'cache'
$cached = Join-Path $cache 'fixtures/input.bin'
[void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($cached))
$bytes = [byte[]]@(0, 255, 128, 13, 10, 0)
[IO.File]::WriteAllBytes($cached, $bytes)
$digest = [Convert]::ToHexStringLower([Security.Cryptography.SHA256]::HashData($bytes))
# The unreachable URL makes accidental network access fail. Every successful
# materialization must use the already verified raw bytes in the input cache.
$lock = @{ AssetPrefix='fixtures/'; Files=@(@{ Path='input.bin'; Bytes=$bytes.Length; Sha256=$digest; Url='https://127.0.0.1:1/input' }) }
$materialize = {
    param($module, $inputLock, $destination, $downloadCache)
    & $module {
        param($inputLock, $destination, $downloadCache)
        Save-BonsaiPrismFixture -Lock $inputLock -Directory $destination -CacheDirectory $downloadCache -TotalBytes 6 -MaximumFileBytes 6 -ValidateEntry { param($entry) $entry.Path -ceq 'input.bin' }
    } $inputLock $destination $downloadCache
}
$first = Join-Path $scratch 'first'
$receipt = & $materialize $assets $lock $first $cache
Assert ($receipt.Files.Count -eq 1 -and $receipt.Files[0].Sha256 -ceq $digest) 'Cached input lost its exact locked identity.'
$firstFile = Join-Path $first 'fixtures/input.bin'
Assert ([Convert]::ToBase64String([IO.File]::ReadAllBytes($firstFile)) -ceq [Convert]::ToBase64String($bytes)) 'A cache hit changed fixture bytes.'
Assert-Failure { & $materialize $assets $lock $first $cache } 'Cached input overwrote existing evidence.'
[IO.File]::WriteAllBytes($firstFile, [byte[]]@(1, 1, 1, 1, 1, 1))
Assert ([Convert]::ToBase64String([IO.File]::ReadAllBytes($cached)) -ceq [Convert]::ToBase64String($bytes)) 'Evidence mutations changed the cached input.'
$second = Join-Path $scratch 'second'
$null = & $materialize $assets $lock $second $cache
Assert ([Convert]::ToBase64String([IO.File]::ReadAllBytes((Join-Path $second 'fixtures/input.bin'))) -ceq [Convert]::ToBase64String($bytes)) 'A later run reused prior observations instead of immutable inputs.'
[IO.File]::WriteAllBytes($cached, [byte[]]@(1, 255, 128, 13, 10, 0))
$rejected = Join-Path $scratch 'rejected'
Assert-Failure { & $materialize $assets $lock $rejected $cache } 'Same-length corrupt cached input was accepted.'
Assert (-not [IO.Directory]::Exists($rejected)) 'Corrupt input was copied before verification.'
[IO.File]::WriteAllBytes($cached, $bytes)
$invalid = @{ AssetPrefix='fixtures/'; Files=@($lock.Files[0], @{ Path='other.bin'; Bytes=1; Sha256=$digest; Url='https://127.0.0.1:1/other' }) }
$preflight = Join-Path $scratch 'preflight'
Assert-Failure { & $materialize $assets $invalid $preflight $cache } 'An invalid complete fixture lock was accepted.'
Assert (-not [IO.Directory]::Exists($preflight)) 'A cached file bypassed complete lock preflight.'
return "Prism fixture cache reuse, corruption and exclusive evidence checks passed. Evidence: $scratch"

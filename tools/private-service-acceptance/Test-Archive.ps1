#requires -Version 7.6
[CmdletBinding()]
[OutputType([string])]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot 'Archive.psm1') -Force
$root = Join-Path ([IO.Path]::GetTempPath()) ('prism-service-archive-' + [guid]::NewGuid().ToString('N'))
$source = Join-Path $root 'transient/source'
$archive = Join-Path $root 'durable'
$transient = Join-Path $root 'transient'
$revision = 'a' * 40
$arguments = @{ SourceDirectory=$source; ArchiveRoot=$archive; TransientRoot=$transient; Revision=$revision; RunId='123'; RunAttempt='1' }
$hostValue = @{ schemaVersion=1; status='passed'; repository='symbiogenesis/prism-inference'; harnessRevision=$revision;
    gitRef='refs/heads/main'; runId='123'; runAttempt='1'; defaultPromotionQualified=$false }
$ownerValue = @{ schemaVersion=1; status='failed'; harnessRevision=$revision; defaultPromotionQualified=$false }
$serviceValue = @{ schemaVersion=1; status='pending'; completedSteps=@('exact-pq2-adapter-readiness'); readiness=$null; defaultPromotionQualified=$false }
$buildValue = @{ schemaVersion=2; repository='symbiogenesis/prism-inference'; revision=$revision; gitRef='refs/heads/main';
    runId='123'; runAttempt=1; modelExecuted=$false; files=@(@{ name='private-service-check.exe' }, @{ name='private-service-owner.exe' }) }

function Write-TestReceipt {
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseShouldProcessForStateChangingFunctions', '', Justification = 'Writes only this test-owned temporary fixture.')]
    [CmdletBinding()]
    [OutputType([void])]
    param([Parameter(Mandatory)][string]$Name, [Parameter(Mandatory)][Collections.IDictionary]$Value)
    $Value | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $source $Name) -Encoding utf8NoBOM
}

function Assert-ArchiveRejected {
    [CmdletBinding()]
    [OutputType([void])]
    param([Parameter(Mandatory)][Collections.IDictionary]$Changes)
    $options = $arguments.Clone()
    foreach ($key in $Changes.Keys) { $options[$key] = $Changes[$key] }
    $refused = $false
    try { $null = Save-PrivateServiceReceipts @options } catch { $refused = $true }
    if (-not $refused) { throw 'An unsafe archive path or identity was admitted.' }
}

try {
    # The owner-managed qualification leaf may be absent; only its already
    # validated parent can be used to create it.
    [void][IO.Directory]::CreateDirectory($source)
    Write-TestReceipt -Name 'host.json' -Value $hostValue
    Write-TestReceipt -Name 'owner.json' -Value $ownerValue
    Write-TestReceipt -Name 'service.json' -Value $serviceValue
    Write-TestReceipt -Name 'hosted-build.json' -Value $buildValue
    [IO.File]::WriteAllText((Join-Path $source 'frontend.key'), 'DO-NOT-ARCHIVE')
    [void][IO.Directory]::CreateDirectory((Join-Path $source 'protected'))
    [IO.File]::WriteAllText((Join-Path $source 'protected/backend.key'), 'DO-NOT-ARCHIVE')
    $complete = Save-PrivateServiceReceipts @arguments
    if ($complete.Files.Count -ne 4 -or $complete.Rejected.Count -or $complete.Missing.Count -or
        @(Get-ChildItem -LiteralPath $complete.Directory -Force).Count -ne 4 -or
        [IO.Path]::GetFileName($complete.Directory) -cnotmatch ('^prism-private-service-' + $revision + '-123-1-[a-f0-9]{32}$')) {
        throw 'Exact public-only source/run/attempt-bound archival failed.'
    }
    foreach ($name in $complete.Files) {
        if ((Get-FileHash -LiteralPath (Join-Path $source $name)).Hash -cne (Get-FileHash -LiteralPath (Join-Path $complete.Directory $name)).Hash) {
            throw 'Receipt archival changed bytes.'
        }
    }
    $originalHash = (Get-FileHash -LiteralPath (Join-Path $complete.Directory 'host.json')).Hash
    $failedHost = $hostValue.Clone(); $failedHost.status='failed'
    Write-TestReceipt -Name 'host.json' -Value $failedHost
    Remove-Item -LiteralPath (Join-Path $source 'hosted-build.json')
    $partial = Save-PrivateServiceReceipts @arguments
    if ($partial.Files.Count -ne 3 -or $partial.Missing.Count -ne 1 -or $partial.Rejected.Count -or $partial.Directory -ceq $complete.Directory -or
        (Get-FileHash -LiteralPath (Join-Path $complete.Directory 'host.json')).Hash -cne $originalHash -or
        (Get-Content -LiteralPath (Join-Path $partial.Directory 'owner.json') -Raw | ConvertFrom-Json).status -cne 'failed') {
        throw 'Partial failed receipts must survive without overwriting the earlier selection.'
    }
    [IO.File]::WriteAllText((Join-Path $source 'owner.json'), (' ' * (64KB + 1)))
    [IO.File]::WriteAllText((Join-Path $source 'service.json'), '{broken')
    $bad = Save-PrivateServiceReceipts @arguments
    if ($bad.Files.Count -ne 1 -or $bad.Files[0] -cne 'host.json' -or $bad.Rejected.Count -ne 2 -or
        @(Get-ChildItem -LiteralPath $bad.Directory -Force).Count -ne 1) { throw 'Invalid or oversized receipts were copied or valid partial evidence was lost.' }
    Write-TestReceipt -Name 'owner.json' -Value $ownerValue
    $secret = $serviceValue.Clone(); $secret.readiness=@{ apiKey='DO-NOT-ARCHIVE' }
    Write-TestReceipt -Name 'service.json' -Value $secret
    $bad = Save-PrivateServiceReceipts @arguments
    if ($bad.Rejected -cnotcontains 'service.json' -or (Test-Path -LiteralPath (Join-Path $bad.Directory 'service.json'))) { throw 'A nested credential field entered public archival.' }
    $secret=$hostValue.Clone(); $secret.apiKey='DO-NOT-ARCHIVE'
    Write-TestReceipt -Name 'host.json' -Value $secret
    $bad = Save-PrivateServiceReceipts @arguments
    if ($bad.Rejected -cnotcontains 'host.json') { throw 'An extra credential field entered public archival.' }
    Write-TestReceipt -Name 'host.json' -Value $hostValue
    $bad = Save-PrivateServiceReceipts @arguments -Revision ('b' * 40)
    if ($bad.Files.Count -or $bad.Rejected.Count -ne 3) { throw 'Stale-source receipts were admitted.' }
    Assert-ArchiveRejected -Changes @{ ArchiveRoot=$transient }
    Assert-ArchiveRejected -Changes @{ ArchiveRoot=$source }
    Assert-ArchiveRejected -Changes @{ ArchiveRoot=(Join-Path $archive '..\durable') }
    Assert-ArchiveRejected -Changes @{ ArchiveRoot='\\server\share' }
    Assert-ArchiveRejected -Changes @{ Revision='../bad' }
    [void][IO.Directory]::CreateDirectory((Join-Path $archive '.git'))
    Assert-ArchiveRejected -Changes @{}
    Remove-Item -LiteralPath (Join-Path $archive '.git')
    $junction = Join-Path $root 'redirected-source'
    New-Item -ItemType Junction -Path $junction -Target $source | Out-Null
    Assert-ArchiveRejected -Changes @{ SourceDirectory=$junction }
    Remove-Item -LiteralPath $junction
    # Reproduce the runner's temp cleanup after an independent upload failure.
    # The verified durable selection must remain byte-identical and available.
    $transientTarget = [IO.Path]::GetFullPath($transient)
    if ($transientTarget -cne (Join-Path $root 'transient')) { throw 'Unexpected transient fixture cleanup target.' }
    Remove-Item -LiteralPath $transientTarget -Recurse -Force
    if ((Get-FileHash -LiteralPath (Join-Path $complete.Directory 'host.json')).Hash -cne $originalHash -or
        @(Get-ChildItem -LiteralPath $complete.Directory -Force).Count -ne 4) { throw 'Runner temp cleanup destroyed the durable selection.' }
    'Passed public-only immutable archival, byte identity, partial failures, malformed/oversized/credential/stale receipt rejection and unsafe path/reparse controls.'
} finally {
    $resolved = [IO.Path]::GetFullPath($root)
    $parent = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (-not $resolved.StartsWith($parent, [StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetFileName($resolved) -notmatch '^prism-service-archive-[a-f0-9]{32}$') { throw 'Unsafe archive fixture cleanup target.' }
    # Remove the test-owned junction itself before recursive fixture cleanup.
    $junction = Join-Path $resolved 'redirected-source'
    if (Test-Path -LiteralPath $junction) { Remove-Item -LiteralPath $junction }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}

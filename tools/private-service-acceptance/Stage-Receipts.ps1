#requires -Version 7.6
[CmdletBinding()]
[OutputType([string])]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot 'Archive.psm1')

if ($env:GITHUB_REPOSITORY -cne 'symbiogenesis/prism-inference' -or $env:GITHUB_REF -cne 'refs/heads/main') {
    throw 'Durable service receipts require the exact owning main execution.'
}
$archive = Join-Path $env:LOCALAPPDATA 'SwiftBonsai2/qualification'
$result = Save-PrivateServiceReceipts -SourceDirectory $env:PRISM_SERVICE_EVIDENCE -ArchiveRoot $archive `
    -TransientRoot $env:RUNNER_TEMP -Revision $env:GITHUB_SHA -RunId $env:GITHUB_RUN_ID -RunAttempt $env:GITHUB_RUN_ATTEMPT
# Publish only the explicit validated names below, including when another receipt
# was rejected. The durable directory survives the runner's automatic temp cleanup.
if ($result.Files.Count) {
    $env:PRISM_SERVICE_PUBLIC_EVIDENCE = $result.Directory
    "PRISM_SERVICE_PUBLIC_EVIDENCE=$($result.Directory)" >> $env:GITHUB_ENV
}
if ($result.Rejected.Count -or -not $result.Files.Count) { throw 'Durable public receipt staging was incomplete; rejected files were not copied.' }
"Durable public receipts retained: $($result.Files.Count); missing: $($result.Missing.Count)."

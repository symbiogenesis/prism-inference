#requires -Version 7.6
[CmdletBinding()]
[OutputType([void])]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot 'Budget.psm1')
Import-Module (Join-Path $PSScriptRoot 'Intake.psm1')

$hostReport = [ordered]@{
    schemaVersion=1; status='pending'; stage='host-validation'; repository=$env:GITHUB_REPOSITORY
    harnessRevision=$env:GITHUB_SHA; gitRef=$env:GITHUB_REF; runId=$env:GITHUB_RUN_ID; runAttempt=$env:GITHUB_RUN_ATTEMPT
    workflowPath='.github/workflows/private-service-acceptance.yml'; artifactId=$env:PRISM_SERVICE_ARTIFACT_ID
    artifactName="private-service-check-$env:GITHUB_SHA-$env:GITHUB_RUN_ID-$env:GITHUB_RUN_ATTEMPT"
    artifactDigest=$env:PRISM_SERVICE_ARTIFACT_DIGEST; runnerOwnerRevision='1914330f8a31769bfe8368ca9728fe7a418ed84e'
    jobStartedUtc=$env:PRISM_SERVICE_JOB_STARTED_UTC; wholeOwnerMaximumSeconds=300; ownerWorkSeconds=210
    ownerProcessTotalSeconds=240; postWorkProcessReserveSeconds=30; retainedIntakeReserveSeconds=60; cpu=$null; architecture=[Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString()
    os=[Environment]::OSVersion.VersionString; ownerElapsedSeconds=$null; jobElapsedSeconds=$null
    ownerSoftSeconds=$null; ownerHardSeconds=$null; ownerExitCode=$null; defaultPromotionQualified=$false
}
$guards = [Collections.Generic.List[IO.FileStream]]::new()
$process = $null
try {
    if (-not $IsWindows -or $hostReport.architecture -cne 'Arm64' -or $env:GITHUB_ACTIONS -cne 'true' -or
        $env:GITHUB_REPOSITORY -cne 'symbiogenesis/prism-inference' -or $env:GITHUB_REF -cne 'refs/heads/main' -or
        $env:PRISM_SERVICE_ARTIFACT_ID -cnotmatch '^[1-9][0-9]+$' -or
        $env:PRISM_SERVICE_ARTIFACT_DIGEST -cnotmatch '^[a-f0-9]{64}$') {
        throw 'The service capture requires the exact owning main ARM64 execution.'
    }
    $cpu = Get-CimInstance Win32_Processor -OperationTimeoutSec 10 | Select-Object -First 1
    if ($cpu.Name -notmatch 'Snapdragon.*X.*Elite') { throw 'A native Snapdragon X/X2 Elite owner is required.' }
    $hostReport.cpu = $cpu.Name
    $hostReport.stage = 'same-run-package-verification'
    $package = Join-Path $env:RUNNER_TEMP 'Private-Service-Check-Package'
    $opened = Open-PrivateServicePackage -Package $package -Expected @{
        repository=$env:GITHUB_REPOSITORY; revision=$env:GITHUB_SHA; gitRef='refs/heads/main'
        workflowPath='.github/workflows/private-service-acceptance.yml'; runId=$env:GITHUB_RUN_ID
        runAttempt=$env:GITHUB_RUN_ATTEMPT; architecture='ARM64'; scope='harness-preparation-only'
        runnerOwnerRevision=$hostReport.runnerOwnerRevision
        ownerSourceSha256=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'owner.rs') -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $guards = $opened.Guards
    [IO.File]::WriteAllText((Join-Path $env:PRISM_SERVICE_EVIDENCE 'hosted-build.json'), $opened.ReceiptText, [Text.UTF8Encoding]::new($false))
    $hostReport.stage = 'remaining-owner-budget'
    $owner = @{
        RunId=$env:BONSAI_NATIVE_OWNER_RUN_ID; Revision=$env:BONSAI_NATIVE_OWNER_REVISION
        StartedTick=$env:BONSAI_NATIVE_OWNER_STARTED_TICK; StartedUtc=$env:BONSAI_NATIVE_OWNER_STARTED_UTC
        WorkSeconds=$env:BONSAI_NATIVE_OWNER_WORK_SECONDS
    }
    $limits = Get-PrivateServiceBudget -JobStartedUtc ([DateTime]::Parse($env:PRISM_SERVICE_JOB_STARTED_UTC, [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::RoundtripKind)) `
        -JobStartedTick ([long]$env:PRISM_SERVICE_JOB_STARTED_TICK) -NowUtc ([DateTime]::UtcNow) `
        -NowTick ([Diagnostics.Stopwatch]::GetTimestamp()) -TickFrequency ([Diagnostics.Stopwatch]::Frequency) `
        -RunId $env:GITHUB_RUN_ID -Revision $env:GITHUB_SHA -Owner $owner
    $hostReport.ownerElapsedSeconds = $limits.OwnerElapsedSeconds
    $hostReport.jobElapsedSeconds = $limits.JobElapsedSeconds
    $hostReport.ownerSoftSeconds = $limits.SoftSeconds
    $hostReport.ownerHardSeconds = $limits.HardSeconds
    $start = [Diagnostics.ProcessStartInfo]::new((Join-Path $package 'private-service-owner.exe'))
    $start.WorkingDirectory = $env:PRISM_SERVICE_WORK
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    foreach ($argument in @((Join-Path $env:LOCALAPPDATA 'SwiftBonsai2'), $env:PRISM_SERVICE_EVIDENCE,
        (Join-Path $package 'private-service-check.exe'), [string]$limits.SoftSeconds)) { $start.ArgumentList.Add($argument) }
    foreach ($name in @('ACTIONS_RUNTIME_TOKEN', 'ACTIONS_ID_TOKEN_REQUEST_TOKEN', 'ACTIONS_ID_TOKEN_REQUEST_URL',
        'GITHUB_TOKEN', 'GH_TOKEN', 'ACTIONS_RUNNER_INPUT_TOKEN')) { [void]$start.Environment.Remove($name) }
    $hostReport.stage = 'owned-model-capture'
    $process = [Diagnostics.Process]::Start($start)
    if (-not $process.WaitForExit($limits.HardSeconds * 1000)) { throw 'The service owner exceeded its remaining aggregate allowance.' }
    $hostReport.ownerExitCode = $process.ExitCode
    if ($process.ExitCode -ne 0) { throw 'The bounded service capture failed; review only the retained credential-free receipts.' }
    foreach ($name in @('owner.json', 'service.json')) {
        $result = Get-Content -LiteralPath (Join-Path $env:PRISM_SERVICE_EVIDENCE $name) -Raw | ConvertFrom-Json
        if ($result.status -cne 'passed' -or $result.defaultPromotionQualified -ne $false) { throw 'The owner/checker receipt is incomplete.' }
    }
    $hostReport.status = 'passed'
    $hostReport.stage = 'complete'
} finally {
    try {
        if ($null -ne $process) {
            try {
                if (-not $process.HasExited) {
                    $process.Kill($true)
                    if (-not $process.WaitForExit(10000)) { throw 'The bounded owner did not exit during forced cleanup.' }
                }
            } finally { $process.Dispose() }
        }
    } finally {
        foreach ($guard in $guards) { $guard.Dispose() }
        if ($hostReport.status -cne 'passed') { $hostReport.status = 'failed' }
        $hostReport | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $env:PRISM_SERVICE_EVIDENCE 'host.json') -Encoding utf8NoBOM
    }
}

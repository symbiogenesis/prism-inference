#requires -Version 7.6
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

function Assert-ReceiptDirectory {
    [CmdletBinding()]
    [OutputType([string])]
    param([Parameter(Mandatory)][string]$Path, [switch]$AllowMissingLeaf)
    if ($Path -cnotmatch '^[A-Za-z]:[\\/]' -or $Path -match '%[^%]+%' -or $Path.Substring(2).Contains(':')) {
        throw 'Receipt storage must use an expanded local absolute path.'
    }
    $full = [IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    if ($full.Length -le 3 -or $full -cne $Path.TrimEnd('\', '/')) { throw 'Receipt storage must use its canonical non-root path.' }
    $cursor = $full
    while ($cursor.Length -gt 3) {
        if ($AllowMissingLeaf -and $cursor -ceq $full -and -not (Test-Path -LiteralPath $cursor)) {
            $cursor = [IO.Path]::GetDirectoryName($cursor)
            continue
        }
        $item = Get-Item -LiteralPath $cursor -Force
        if (-not $item.PSIsContainer -or $item.Attributes.HasFlag([IO.FileAttributes]::ReparsePoint) -or
            [IO.File]::Exists((Join-Path $cursor '.git')) -or [IO.Directory]::Exists((Join-Path $cursor '.git'))) {
            throw 'Receipt storage cannot traverse reparse points or a Git checkout.'
        }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
    return $full
}

function Assert-ReceiptFields {
    [CmdletBinding()]
    [OutputType([void])]
    param([Parameter(Mandatory)][Collections.IDictionary]$Value, [Parameter(Mandatory)][string[]]$Fields)
    if (@($Value.Keys | Where-Object { $_ -cnotin $Fields }).Count) { throw 'A receipt contains an unexpected field.' }
}

function Assert-PublicReceipt {
    [CmdletBinding()]
    [OutputType([void])]
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][Collections.IDictionary]$Value,
        [Parameter(Mandatory)][string]$Revision,
        [Parameter(Mandatory)][string]$RunId,
        [Parameter(Mandatory)][string]$RunAttempt
    )
    if ($Name -ceq 'hosted-build.json') {
        Assert-ReceiptFields -Value $Value -Fields @('schemaVersion', 'repository', 'revision', 'gitRef', 'workflowPath', 'runId', 'runAttempt',
            'architecture', 'modelExecuted', 'scope', 'runnerOwnerRevision', 'ownerSourceSha256', 'compilerIdentity', 'files')
        if ($Value.schemaVersion -ne 2 -or $Value.repository -cne 'symbiogenesis/prism-inference' -or
            $Value.revision -cne $Revision -or $Value.gitRef -cne 'refs/heads/main' -or
            [string]$Value.runId -cne $RunId -or [string]$Value.runAttempt -cne $RunAttempt -or
            $Value.modelExecuted -isnot [bool] -or $Value.modelExecuted -ne $false -or @($Value.files).Count -ne 2) {
            throw 'The hosted receipt does not bind this execution.'
        }
        foreach ($file in $Value.files) {
            Assert-ReceiptFields -Value $file -Fields @('name', 'bytes', 'sha256')
            if ($file.name -cnotin @('private-service-check.exe', 'private-service-owner.exe')) { throw 'Unexpected hosted file.' }
        }
        return
    }
    if ($Value.schemaVersion -ne 1 -or $Value.status -cnotin @('pending', 'failed', 'passed') -or
        $Value.defaultPromotionQualified -isnot [bool] -or $Value.defaultPromotionQualified -ne $false) {
        throw 'The capture receipt schema or qualification boundary differs.'
    }
    switch -CaseSensitive ($Name) {
        'host.json' {
            Assert-ReceiptFields -Value $Value -Fields @('schemaVersion', 'status', 'stage', 'repository', 'harnessRevision', 'gitRef', 'runId', 'runAttempt',
                'workflowPath', 'artifactId', 'artifactName', 'artifactDigest', 'runnerOwnerRevision', 'jobStartedUtc', 'wholeOwnerMaximumSeconds',
                'ownerWorkSeconds', 'ownerProcessTotalSeconds', 'postWorkProcessReserveSeconds', 'retainedIntakeReserveSeconds', 'cpu', 'architecture',
                'os', 'ownerElapsedSeconds', 'jobElapsedSeconds', 'ownerSoftSeconds', 'ownerHardSeconds', 'ownerExitCode', 'defaultPromotionQualified')
            if ($Value.repository -cne 'symbiogenesis/prism-inference' -or $Value.harnessRevision -cne $Revision -or
                $Value.gitRef -cne 'refs/heads/main' -or [string]$Value.runId -cne $RunId -or [string]$Value.runAttempt -cne $RunAttempt) {
                throw 'The host receipt does not bind this execution.'
            }
        }
        'owner.json' {
            Assert-ReceiptFields -Value $Value -Fields @('schemaVersion', 'status', 'harnessRevision', 'runnerOwnerRevision', 'aggregateOwnerAllowanceSeconds',
                'elapsedMilliseconds', 'profile', 'rowWorkersRequested', 'releasedFilesAndArchiveVerified', 'authenticatedOwnedReadiness', 'ownedJobEmpty',
                'listenersReleased', 'checkerSucceeded', 'separateJsonlWorkerExecuted', 'activeGraphCancellationQualified', 'completeStateRollbackQualified',
                'defaultPromotionQualified', 'testedRuntime')
            if ($Value.harnessRevision -cne $Revision) { throw 'The owner receipt does not bind this source.' }
            if ($Value.Contains('testedRuntime')) {
                Assert-ReceiptFields -Value $Value.testedRuntime -Fields @('revision', 'version', 'archiveSha256', 'serverSha256', 'workerSha256')
            }
        }
        'service.json' {
            Assert-ReceiptFields -Value $Value -Fields @('schemaVersion', 'status', 'completedSteps', 'readiness', 'scope', 'fullStateNumericalParityQualified',
                'performanceQualified', 'activeGraphCancellationQualified', 'completeStateRollbackQualified', 'processOwnershipQualified', 'defaultPromotionQualified')
            if ($null -ne $Value.readiness) {
                Assert-ReceiptFields -Value $Value.readiness -Fields @('modelSha256', 'templateSha256', 'adapterSha256', 'seed', 'contextTokens',
                    'weightRepresentation', 'servingStatePolicy')
            }
        }
        default { throw 'Only four fixed public receipt filenames may be archived.' }
    }
}

function Save-PrivateServiceReceipts {
    <# .SYNOPSIS
    Stage only bounded, schema-checked public receipts outside ephemeral runner storage.
    .DESCRIPTION
    Missing receipts remain missing. Rejected receipts are never copied; validated
    partial failures are retained. Every call uses a fresh source/run/attempt-bound
    child and create-new files. The caller must treat rejected files as a failure.
    #>
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseShouldProcessForStateChangingFunctions', '', Justification = 'Explicitly requested immutable qualification receipt archival.')]
    [CmdletBinding()]
    [OutputType([pscustomobject])]
    param(
        [Parameter(Mandatory)][string]$SourceDirectory,
        [Parameter(Mandatory)][string]$ArchiveRoot,
        [Parameter(Mandatory)][string]$TransientRoot,
        [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{40}$')][string]$Revision,
        [Parameter(Mandatory)][ValidatePattern('^[1-9][0-9]{0,19}$')][string]$RunId,
        [Parameter(Mandatory)][ValidatePattern('^[1-9][0-9]{0,9}$')][string]$RunAttempt
    )
    $source = Assert-ReceiptDirectory -Path $SourceDirectory
    $archive = Assert-ReceiptDirectory -Path $ArchiveRoot -AllowMissingLeaf
    $transient = Assert-ReceiptDirectory -Path $TransientRoot
    foreach ($excluded in @($source, $transient)) {
        if ($archive.Equals($excluded, [StringComparison]::OrdinalIgnoreCase) -or
            $archive.StartsWith($excluded + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Durable receipt storage must be outside transient/source storage.' }
    }
    $destination = Join-Path $archive ("prism-private-service-$Revision-$RunId-$RunAttempt-" + [guid]::NewGuid().ToString('N'))
    if (Test-Path -LiteralPath $destination) { throw 'Receipt destination already exists.' }
    [void][IO.Directory]::CreateDirectory($destination)
    $destination = Assert-ReceiptDirectory -Path $destination
    $copied = [Collections.Generic.List[string]]::new()
    $missing = [Collections.Generic.List[string]]::new()
    $rejected = [Collections.Generic.List[string]]::new()
    foreach ($name in @('host.json', 'hosted-build.json', 'owner.json', 'service.json')) {
        $inputPath = Join-Path $source $name
        if (-not (Test-Path -LiteralPath $inputPath)) { $missing.Add($name); continue }
        $inputFile = $null
        $outputPath = $null
        $created = $false
        try {
            [void](Assert-ReceiptDirectory -Path $source)
            $item = Get-Item -LiteralPath $inputPath -Force
            if ($item.Name -cne $name -or $item.PSIsContainer -or $item.Attributes.HasFlag([IO.FileAttributes]::ReparsePoint) -or $item.Length -le 0 -or $item.Length -gt 64KB) {
                throw 'A public receipt is not a bounded regular file.'
            }
            $inputFile = [IO.FileStream]::new($inputPath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
            if ($inputFile.Length -le 0 -or $inputFile.Length -gt 64KB) { throw 'Public receipt length changed.' }
            $bytes = [byte[]]::new([int]$inputFile.Length)
            $inputFile.ReadExactly($bytes)
            $text = [Text.UTF8Encoding]::new($false, $true).GetString($bytes)
            $value = ConvertFrom-Json -InputObject $text -AsHashtable -Depth 16
            if ($value -isnot [Collections.IDictionary]) { throw 'A public receipt must be a JSON object.' }
            Assert-PublicReceipt -Name $name -Value $value -Revision $Revision -RunId $RunId -RunAttempt $RunAttempt
            [void](Assert-ReceiptDirectory -Path $destination)
            $outputPath = Join-Path $destination $name
            $output = [IO.FileStream]::new($outputPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
            $created = $true
            try { $output.Write($bytes); $output.Flush($true) } finally { $output.Dispose() }
            $copied.Add($name)
        } catch {
            if ($created) { Remove-Item -LiteralPath $outputPath -Force }
            $rejected.Add($name)
        } finally {
            if ($null -ne $inputFile) { $inputFile.Dispose() }
        }
    }
    [pscustomobject]@{ Directory=$destination; Files=$copied.ToArray(); Missing=$missing.ToArray(); Rejected=$rejected.ToArray() }
}

Export-ModuleMember -Function Save-PrivateServiceReceipts

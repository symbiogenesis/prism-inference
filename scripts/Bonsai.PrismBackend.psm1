#requires -Version 7.6
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot 'Bonsai.Common.psm1')
Import-Module (Join-Path $PSScriptRoot 'Bonsai.Native.psm1')

$script:BackendSourceSha256 = '8bf7a5cf29d3b3e88aa0c30611f5f1679f46a71ec00306908c44a73ec98ac425'
$script:BackendRequestLimit = 128MB
$script:BackendReportLimit = 128MB
$script:BackendLogLimit = 1MB

function Assert-BonsaiPrismBackendCI {
    [CmdletBinding()]
    [OutputType([void])]
    param()
    if ($env:GITHUB_ACTIONS -cne 'true' -or $env:GITHUB_REF -cne 'refs/heads/main') {
        throw 'Backend source matrices and performance workloads run only on main in GitHub Actions.'
    }
    if (-not $IsWindows -or [Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne [Runtime.InteropServices.Architecture]::Arm64) {
        throw 'Backend orchestration requires a native Windows ARM64 host.'
    }
}

function Assert-BonsaiPrismBackendBinary {
    [CmdletBinding()]
    [OutputType([string])]
    param([string]$Path, [string]$Sha256)
    $full = Resolve-ExternalPath $Path
    if ($Sha256 -cnotmatch '^[a-f0-9]{64}$' -or -not [IO.File]::Exists($full)) {
        throw 'A prebuilt binary and its lowercase SHA256 are required.'
    }
    Assert-Arm64PE $full
    if ((Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash.ToLowerInvariant() -cne $Sha256) {
        throw "Prebuilt backend binary SHA256 changed: $full"
    }
    return $full
}

function Write-BonsaiPrismBackendJson {
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseShouldProcessForStateChangingFunctions', '', Justification = 'Private publication helper writes only fresh authorized external CI evidence.')]
    [CmdletBinding()]
    [OutputType([void])]
    param([string]$Path, [object]$Value, [long]$Limit = $script:BackendRequestLimit)
    $text = ConvertTo-Json -InputObject $Value -Depth 100 -Compress -WarningAction Stop
    $bytes = [Text.UTF8Encoding]::new($false, $true).GetBytes($text)
    if ($bytes.LongLength -gt $Limit) { throw "Backend JSON exceeds its $Limit byte bound." }
    $stream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try { $stream.Write($bytes) } finally { $stream.Dispose() }
}

function Read-BonsaiPrismBackendJson {
    [CmdletBinding()]
    [OutputType([Collections.IDictionary])]
    param([string]$Path)
    $stream = [IO.File]::OpenRead($Path)
    try {
        if ($stream.Length -gt $script:BackendReportLimit) { throw 'Backend report exceeds 128MiB.' }
        # OpenRead retains FILE_SHARE_READ, so writers cannot change this length.
        $bytes = [byte[]]::new($stream.Length)
        $stream.ReadExactly($bytes)
        $text = [Text.UTF8Encoding]::new($false, $true).GetString($bytes)
    } finally { $stream.Dispose() }
    return ConvertFrom-Json -InputObject $text -Depth 100 -AsHashtable -NoEnumerate -DateKind String
}

function New-BonsaiPrismBackendContext {
    <#
    .SYNOPSIS
    Verifies prebuilt ARM64 executables and creates a fresh external evidence root.
    .DESCRIPTION
    SHA256 values come from the binary-build artifact receipt. No executable is
    launched here. Each phase rechecks the identities before launch. SourceDirectory
    is the already archive-verified pinned upstream extraction. ProjectRevision
    defaults to GITHUB_SHA and must match it when any phase executes.
    #>
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseShouldProcessForStateChangingFunctions', '', Justification = 'Explicitly requested CI preparation creates only fresh external evidence after verifying prebuilt identities.')]
    [CmdletBinding()]
    [OutputType([Collections.Specialized.OrderedDictionary])]
    param(
        [Parameter(Mandatory)][string]$HarnessPath,
        [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$HarnessSha256,
        [string]$CommonBridgePath,
        [ValidatePattern('^[a-f0-9]{64}$')][string]$CommonBridgeSha256,
        [Parameter(Mandatory)][string]$SourceDirectory,
        [Parameter(Mandatory)][string]$EvidenceDirectory,
        [Parameter(Mandatory)][ValidateSet('cpp', 'rust')][string]$Target,
        [ValidatePattern('^[a-f0-9]{40}$')][string]$ProjectRevision = $env:GITHUB_SHA,
        [ValidateRange(1,86400)][int]$TimeoutSeconds = 14400
    )
    if ($ProjectRevision -cnotmatch '^[a-f0-9]{40}$') { throw 'A complete project revision is required.' }
    $harness = Assert-BonsaiPrismBackendBinary -Path $HarnessPath -Sha256 $HarnessSha256
    $bridge = $null
    if ($Target -ceq 'cpp') {
        $bridge = Assert-BonsaiPrismBackendBinary -Path $CommonBridgePath -Sha256 $CommonBridgeSha256
    } elseif ($CommonBridgePath -or $CommonBridgeSha256) {
        throw 'The Rust target does not accept a C++ common bridge.'
    }
    $source = Resolve-ExternalPath $SourceDirectory
    if (-not [IO.Directory]::Exists($source)) { throw 'Verified upstream source directory is missing.' }
    $sourceFile = Resolve-ExternalPath (Join-Path $source 'tests/test-backend-ops.cpp')
    if (-not [IO.File]::Exists($sourceFile) -or
        (Get-FileHash -LiteralPath $sourceFile -Algorithm SHA256).Hash.ToLowerInvariant() -cne $script:BackendSourceSha256) {
        throw 'Pinned backend source provenance differs.'
    }
    $root = Resolve-ExternalPath $EvidenceDirectory
    $comparison = [StringComparison]::OrdinalIgnoreCase
    $separator = [IO.Path]::DirectorySeparatorChar
    if ($root.Equals($source, $comparison) -or $root.StartsWith($source.TrimEnd($separator) + $separator, $comparison) -or
        $source.StartsWith($root.TrimEnd($separator) + $separator, $comparison)) {
        throw 'Backend evidence and verified source must use separate directory trees.'
    }
    if (Test-Path -LiteralPath $root) { throw 'Use a fresh external backend evidence directory.' }
    [void][IO.Directory]::CreateDirectory($root)
    $context = [ordered]@{
        SchemaVersion = 1; Kind = 'prism-backend-context'; Target = $Target; ProjectRevision = $ProjectRevision
        HarnessPath = $harness; HarnessSha256 = $HarnessSha256
        CommonBridgePath = $bridge; CommonBridgeSha256 = $(if ($Target -ceq 'cpp') { $CommonBridgeSha256 } else { $null })
        SourceDirectory = $source; SourceSha256 = $script:BackendSourceSha256
        EvidenceDirectory = $root; TimeoutSeconds = $TimeoutSeconds
    }
    Write-BonsaiPrismBackendJson -Path (Join-Path $root 'context.json') -Value $context
    return $context
}

function Assert-BonsaiPrismBackendContext {
    [CmdletBinding()]
    [OutputType([void])]
    param([Collections.IDictionary]$Context)
    Assert-BonsaiPrismBackendCI
    if ($Context.SchemaVersion -ne 1 -or $Context.Kind -cne 'prism-backend-context' -or
        $Context.Target -cnotin @('cpp', 'rust') -or $Context.ProjectRevision -cnotmatch '^[a-f0-9]{40}$' -or
        $Context.ProjectRevision -cne $env:GITHUB_SHA -or $Context.SourceSha256 -cne $script:BackendSourceSha256 -or
        $Context.TimeoutSeconds -lt 1 -or $Context.TimeoutSeconds -gt 86400) {
        throw 'Invalid backend orchestration context.'
    }
    $null = Assert-BonsaiPrismBackendBinary -Path $Context.HarnessPath -Sha256 $Context.HarnessSha256
    if ($Context.Target -ceq 'cpp') { $null = Assert-BonsaiPrismBackendBinary -Path $Context.CommonBridgePath -Sha256 $Context.CommonBridgeSha256 }
    $null = Resolve-ExternalPath $Context.EvidenceDirectory
    if (-not [IO.Directory]::Exists($Context.EvidenceDirectory)) { throw 'Backend evidence root disappeared.' }
}

function New-BonsaiPrismBackendEnvelope {
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseShouldProcessForStateChangingFunctions', '', Justification = 'Constructs an in-memory provenance record without external state changes.')]
    [CmdletBinding()]
    [OutputType([Collections.Specialized.OrderedDictionary])]
    param([Collections.IDictionary]$Context, [Collections.IDictionary]$Payload)
    return [ordered]@{
        SchemaVersion=1; Target=$Context.Target; ProjectRevision=$Context.ProjectRevision
        HarnessSha256=$Context.HarnessSha256; CommonBridgeSha256=$Context.CommonBridgeSha256; Payload=$Payload
    }
}

function Get-BonsaiPrismBackendPayload {
    [CmdletBinding()]
    [OutputType([Collections.IDictionary])]
    param([Collections.IDictionary]$Context, [object]$Envelope,
        [string[]]$Statuses, [string]$Mode, [int]$Index = -1, [int]$Count = -1)
    if ($Envelope -isnot [Collections.IDictionary] -or $Envelope.Count -ne 6 -or
        @('SchemaVersion','Target','ProjectRevision','HarnessSha256','CommonBridgeSha256','Payload').Where({-not $Envelope.Contains($_)}).Count -or
        $Envelope.SchemaVersion -ne 1 -or $Envelope.Target -cne $Context.Target -or
        $Envelope.ProjectRevision -cne $Context.ProjectRevision -or $Envelope.HarnessSha256 -cne $Context.HarnessSha256 -or
        $Envelope.CommonBridgeSha256 -cne $Context.CommonBridgeSha256 -or $Envelope.Payload -isnot [Collections.IDictionary]) {
        throw 'Backend cross-job record target, revision, binary identity or envelope differs.'
    }
    $record = $Envelope.Payload
    if ($Statuses) {
        if ($record.SchemaVersion -ne 1 -or $record.SourceId -cne 'test-backend-ops' -or
            $record.SourceSha256 -cne $script:BackendSourceSha256 -or $record.Status -cnotin $Statuses) {
            throw 'Backend phase record has invalid provenance or status.'
        }
        if ($Mode -and $record.Mode -cne $Mode) { throw 'Backend phase mode differs from the request.' }
        if ($Index -ge 0 -and ($record.Partition.Index -ne $Index -or $record.Partition.Count -ne $Count)) {
            throw 'Backend phase partition differs from the request.'
        }
    }
    return $record
}

function Invoke-BonsaiPrismBackendDiagnostic {
    [CmdletBinding()]
    [OutputType([Collections.IDictionary])]
    param([Collections.IDictionary]$Context, [Collections.IDictionary]$Request, [string]$Phase,
        [string[]]$Statuses, [string]$Mode, [int]$Index = -1, [int]$Count = -1)
    Assert-BonsaiPrismBackendContext -Context $Context
    $stage = Join-Path $Context.EvidenceDirectory ($Phase + '-' + [Guid]::NewGuid().ToString('N'))
    [void][IO.Directory]::CreateDirectory($stage)
    $config = Join-Path $stage 'request.json'
    $output = Join-Path $stage 'harness-report.json'
    $log = Join-Path $stage 'harness.log'
    try {
        Write-BonsaiPrismBackendJson -Path $config -Value $Request
        $arguments = @('--target', $Context.Target, '--diagnostic', $config, '--output', $output,
            '--source-directory', $Context.SourceDirectory, '--evidence-directory', (Join-Path $stage 'bridge'))
        if ($Context.Target -ceq 'cpp') { $arguments += @('--common-bridge', $Context.CommonBridgePath) }
        $process = Invoke-BonsaiNativeProcess -Executable $Context.HarnessPath -ArgumentList $arguments -WorkingDirectory $stage -TimeoutSeconds $Context.TimeoutSeconds -LogPath $log
        if ([Text.Encoding]::UTF8.GetByteCount($process.StandardOutput) -gt $script:BackendLogLimit -or
            [Text.Encoding]::UTF8.GetByteCount($process.StandardError) -gt $script:BackendLogLimit) {
            throw 'Backend harness output exceeds 1MiB per stream.'
        }
        if ($process.ExitCode -ne 0) {
            $detail = if ($process.StandardError.Trim()) { $process.StandardError.Trim() } else { $process.StandardOutput.Trim() }
            $detail = $detail.Substring([Math]::Max(0, $detail.Length - 4096))
            throw "Backend harness infrastructure exit $($process.ExitCode): $detail; see $log"
        }
        $report = Read-BonsaiPrismBackendJson -Path $output
        if ($report -isnot [Collections.IDictionary] -or $report.SchemaVersion -ne 1 -or
            $report.Target -cne $Context.Target -or $report.UpstreamCommit -cne 'b87ca7225456b06b0975b8239c66b5aecf07e37c' -or
            $report.Measure -cne 'diagnostic-workloads' -or $report.Cases -isnot [array] -or $report.Cases.Count -ne 1 -or
            $report.Cases[0] -isnot [Collections.IDictionary]) { throw 'Backend harness diagnostic envelope differs.' }
        $envelope = New-BonsaiPrismBackendEnvelope -Context $Context -Payload $report.Cases[0]
        Write-BonsaiPrismBackendJson -Path (Join-Path $stage 'phase-record.json') -Value $envelope
    } catch {
        Write-BonsaiPrismBackendJson -Path (Join-Path $stage 'transport-failure.json') -Value ([ordered]@{
            SchemaVersion = 1; Status = 'infrastructure-failure'; Phase = $Phase; Target = $Context.Target; Reason = $_.Exception.Message
        })
        throw
    }
    if ($Statuses) { $null = Get-BonsaiPrismBackendPayload -Context $Context -Envelope $envelope -Statuses $Statuses -Mode $Mode -Index $Index -Count $Count }
    return $envelope
}

function New-BonsaiPrismBackendIncomplete {
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseShouldProcessForStateChangingFunctions', '', Justification = 'Explicitly requested CI orchestration retains a fresh external failure receipt.')]
    [CmdletBinding()]
    [OutputType([Collections.IDictionary])]
    param([Collections.IDictionary]$Context, [string]$Phase, [string]$Mode, [array]$Inputs)
    Assert-BonsaiPrismBackendContext -Context $Context
    $value = [ordered]@{
        SchemaVersion = 1; SourceId = 'test-backend-ops'; SourceSha256 = $script:BackendSourceSha256
        Status = 'unsupported'; Mode = $Mode; Target = $Context.Target; Incomplete = $true
        OrchestrationPhase = $Phase; Reason = 'Required production API is unsupported; complete matrix receipts do not exist.'
        Inputs = $Inputs; CorrectnessSourceCount = 0
    }
    $envelope = New-BonsaiPrismBackendEnvelope -Context $Context -Payload $value
    Write-BonsaiPrismBackendJson -Path (Join-Path $Context.EvidenceDirectory ($Phase + '-incomplete-' + [Guid]::NewGuid().ToString('N') + '.json')) -Value $envelope
    return $envelope
}

function Invoke-BonsaiPrismBackendPlanPartition {
    <#
    .SYNOPSIS
    Plans one ordered geometry partition without executing its numerical graphs.
    #>
    [CmdletBinding()]
    [OutputType([Collections.IDictionary])]
    param([Parameter(Mandatory)][Collections.IDictionary]$Context,
        [Parameter(Mandatory)][ValidateRange(0,4095)][int]$Index,
        [Parameter(Mandatory)][ValidateRange(1,4096)][int]$Count,
        [Parameter(Mandatory)][ValidateSet('eval','grad','support','perf')][string]$Mode)
    if ($Index -ge $Count) { throw 'Partition index must be below its count.' }
    $request = [ordered]@{Kind='backend-plan';Index=$Index;Count=$Count;Mode=$Mode}
    return Invoke-BonsaiPrismBackendDiagnostic -Context $Context -Request $request -Phase "plan-$Mode-$Index" -Statuses @('planned','unsupported') -Mode $Mode -Index $Index -Count $Count
}

function Merge-BonsaiPrismBackendPlan {
    <#
    .SYNOPSIS
    Verifies cross-job identity and requests the Rust source-order/CRT-prefix proof.
    #>
    [CmdletBinding()]
    [OutputType([Collections.IDictionary])]
    param([Parameter(Mandatory)][Collections.IDictionary]$Context,
        [Parameter(Mandatory)][ValidateNotNullOrEmpty()][array]$Plans)
    $raw = @(foreach ($record in $Plans) { Get-BonsaiPrismBackendPayload -Context $Context -Envelope $record -Statuses @('planned','unsupported') })
    $mode = [string]$raw[0].Mode
    if ($mode -and $raw.Where({ $_.Mode -cne $mode }, 'First').Count) { throw 'Backend phase mode differs from the request.' }
    if ($raw.Where({ $_.Status -ceq 'unsupported' }, 'First').Count) {
        return New-BonsaiPrismBackendIncomplete -Context $Context -Phase 'aggregate-plans' -Mode $mode -Inputs $raw
    }
    $request = [ordered]@{Kind='backend-aggregate-plans';Plans=$raw}
    return Invoke-BonsaiPrismBackendDiagnostic -Context $Context -Request $request -Phase "aggregate-plans-$mode" -Statuses @('planned-complete') -Mode $mode
}

function Invoke-BonsaiPrismBackendPartition {
    <#
    .SYNOPSIS
    Executes an exact source partition from a complete Rust-verified input plan.
    #>
    [CmdletBinding()]
    [OutputType([Collections.IDictionary])]
    param([Parameter(Mandatory)][Collections.IDictionary]$Context,
        [Parameter(Mandatory)][ValidateRange(0,4095)][int]$Index,
        [Parameter(Mandatory)][ValidateRange(1,4096)][int]$Count,
        [Parameter(Mandatory)][ValidateSet('eval','grad','support','perf')][string]$Mode,
        [Parameter(Mandatory)][Collections.IDictionary]$Plan)
    if ($Index -ge $Count) { throw 'Partition index must be below its count.' }
    $raw = Get-BonsaiPrismBackendPayload -Context $Context -Envelope $Plan -Statuses @('planned-complete','unsupported') -Mode $Mode
    if ($raw.Status -ceq 'unsupported') { return New-BonsaiPrismBackendIncomplete -Context $Context -Phase 'run' -Mode $Mode -Inputs @($raw) }
    $request = [ordered]@{Kind='backend-run';Index=$Index;Count=$Count;Mode=$Mode;Plan=$raw}
    return Invoke-BonsaiPrismBackendDiagnostic -Context $Context -Request $request -Phase "run-$Mode-$Index" -Statuses @('pass','fail','unsupported') -Mode $Mode -Index $Index -Count $Count
}

function Merge-BonsaiPrismBackendPartition {
    <#
    .SYNOPSIS
    Proves correctness partition completeness while retaining source failures as data.
    #>
    [CmdletBinding()]
    [OutputType([Collections.IDictionary])]
    param([Parameter(Mandatory)][Collections.IDictionary]$Context,
        [Parameter(Mandatory)][ValidateNotNullOrEmpty()][array]$Reports,
        [Parameter(Mandatory)][Collections.IDictionary]$Plan)
    $rawPlan = Get-BonsaiPrismBackendPayload -Context $Context -Envelope $Plan -Statuses @('planned-complete','unsupported')
    $mode = [string]$rawPlan.Mode
    if ($mode -ceq 'perf') { throw 'Use Merge-BonsaiPrismBackendBenchmark for performance reports.' }
    $raw = @(foreach ($record in $Reports) { Get-BonsaiPrismBackendPayload -Context $Context -Envelope $record -Statuses @('pass','fail','unsupported') -Mode $mode })
    if ($rawPlan.Status -ceq 'unsupported' -or $raw.Where({ $_.Status -ceq 'unsupported' }, 'First').Count) {
        return New-BonsaiPrismBackendIncomplete -Context $Context -Phase 'aggregate' -Mode $mode -Inputs @($rawPlan; $raw)
    }
    $request = [ordered]@{Kind='backend-aggregate';Reports=$raw;Plan=$rawPlan}
    return Invoke-BonsaiPrismBackendDiagnostic -Context $Context -Request $request -Phase "aggregate-$mode" -Statuses @('pass','fail') -Mode $mode
}

function Merge-BonsaiPrismBackendSource {
    <#
    .SYNOPSIS
    Requires complete eval and support, plus grad when explicitly selected; excludes performance.
    #>
    [CmdletBinding()]
    [OutputType([Collections.IDictionary])]
    param([Parameter(Mandatory)][Collections.IDictionary]$Context,
        [Parameter(Mandatory)][ValidateNotNullOrEmpty()][array]$Modes)
    $raw = @(foreach ($record in $Modes) { Get-BonsaiPrismBackendPayload -Context $Context -Envelope $record -Statuses @('pass','fail','unsupported') })
    if ($raw.Count -notin @(2,3) -or
        $raw.Where({ $_.Mode -ceq 'eval' }).Count -ne 1 -or
        $raw.Where({ $_.Mode -ceq 'support' }).Count -ne 1 -or
        $raw.Where({ $_.Mode -ceq 'grad' }).Count -ne ($raw.Count - 2)) {
        throw 'Backend source requires eval and support exactly once, with grad only when selected.'
    }
    if ($raw.Where({ $_.Status -ceq 'unsupported' }, 'First').Count) {
        $coverage = New-BonsaiPrismBackendIncomplete -Context $Context -Phase 'source-aggregate' -Mode '' -Inputs $raw
        $value = [ordered]@{Id='tests/test-backend-ops.cpp';Status='fail';Detail=$coverage.Payload.Reason;SourceCount=1;Coverage=$coverage.Payload}
        $record = New-BonsaiPrismBackendEnvelope -Context $Context -Payload $value
        Write-BonsaiPrismBackendJson -Path (Join-Path $Context.EvidenceDirectory ('source-incomplete-' + [Guid]::NewGuid().ToString('N') + '.json')) -Value $record
        return $record
    }
    $record = Invoke-BonsaiPrismBackendDiagnostic -Context $Context -Request ([ordered]@{Kind='backend-source-aggregate';Modes=$raw}) -Phase 'source-aggregate'
    $value = Get-BonsaiPrismBackendPayload -Context $Context -Envelope $record
    if ($value.Id -cne 'tests/test-backend-ops.cpp' -or $value.Status -cnotin @('pass','fail') -or $value.SourceCount -ne 1) {
        throw 'Backend source aggregation returned an invalid verdict.'
    }
    return $record
}

function Merge-BonsaiPrismBackendBenchmark {
    <#
    .SYNOPSIS
    Aggregates CI-only measurements outside the correctness denominator.
    #>
    [CmdletBinding()]
    [OutputType([Collections.IDictionary])]
    param([Parameter(Mandatory)][Collections.IDictionary]$Context,
        [Parameter(Mandatory)][ValidateNotNullOrEmpty()][array]$Reports,
        [Parameter(Mandatory)][Collections.IDictionary]$Plan)
    $rawPlan = Get-BonsaiPrismBackendPayload -Context $Context -Envelope $Plan -Statuses @('planned-complete','unsupported') -Mode 'perf'
    $raw = @(foreach ($record in $Reports) { Get-BonsaiPrismBackendPayload -Context $Context -Envelope $record -Statuses @('pass','fail','unsupported') -Mode 'perf' })
    if ($rawPlan.Status -ceq 'unsupported' -or $raw.Where({ $_.Status -ceq 'unsupported' }, 'First').Count) {
        $coverage = New-BonsaiPrismBackendIncomplete -Context $Context -Phase 'benchmark-aggregate' -Mode 'perf' -Inputs @($rawPlan; $raw)
        return New-BonsaiPrismBackendEnvelope -Context $Context -Payload ([ordered]@{SchemaVersion=1;Benchmark='backend-ops-performance';Status='unsupported';CorrectnessSourceCount=0;Coverage=$coverage.Payload})
    }
    $record = Invoke-BonsaiPrismBackendDiagnostic -Context $Context -Request ([ordered]@{Kind='backend-benchmark';Phase='aggregate';Reports=$raw;Plan=$rawPlan}) -Phase 'benchmark-aggregate'
    $value = Get-BonsaiPrismBackendPayload -Context $Context -Envelope $record
    if ($value.SchemaVersion -ne 1 -or $value.Benchmark -cne 'backend-ops-performance' -or
        $value.Status -cnotin @('measured','fail','unsupported') -or $value.CorrectnessSourceCount -ne 0) {
        throw 'Backend benchmark aggregation returned an invalid measurement report.'
    }
    return $record
}

function Invoke-BonsaiPrismBackendTarget {
    <#
    .SYNOPSIS
    Runs one target's backend stage, preserving its sequential mode order.
    #>
    [CmdletBinding()]
    [OutputType([hashtable])]
    param(
        [Parameter(Mandatory)][Collections.IDictionary]$Context,
        [Parameter(Mandatory)][ValidateSet('Plan', 'JoinPlan', 'Run')][string]$Phase,
        [Parameter(Mandatory)][string[]]$Modes,
        [ValidateRange(0,4095)][int]$Index = 0,
        [ValidateRange(1,4096)][int]$Count = 16,
        [object[]]$Parts,
        [Collections.IDictionary]$Plan
    )
    $results = @{}
    foreach ($mode in $Modes) {
        $results[$mode] = switch ($Phase) {
            'Plan' { Invoke-BonsaiPrismBackendPlanPartition -Context $Context -Index $Index -Count $Count -Mode $mode }
            'JoinPlan' {
                $plans = @($Parts | ForEach-Object { $_.Targets[$Context.Target][$mode] })
                Merge-BonsaiPrismBackendPlan -Context $Context -Plans $plans
            }
            'Run' { Invoke-BonsaiPrismBackendPartition -Context $Context -Index $Index -Count $Count -Mode $mode -Plan $Plan.Targets[$Context.Target][$mode] }
        }
    }
    return $results
}

Export-ModuleMember -Function New-BonsaiPrismBackendContext, Invoke-BonsaiPrismBackendPlanPartition,
    Merge-BonsaiPrismBackendPlan, Invoke-BonsaiPrismBackendPartition, Merge-BonsaiPrismBackendPartition,
    Merge-BonsaiPrismBackendSource, Merge-BonsaiPrismBackendBenchmark, Invoke-BonsaiPrismBackendTarget

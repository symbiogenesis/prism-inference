#requires -Version 7.6
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot 'Bonsai.Common.psm1')
Import-Module (Join-Path $PSScriptRoot 'Bonsai.Native.psm1')

$script:ToolScratch = @{}
$script:CheckedTools = @{}

function Get-PrismToolScratch {
    <# .SYNOPSIS
    Returns the retained external incremental build lane for one tool in this session.
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param([Parameter(Mandatory)][ValidatePattern('^[a-z-]+$')][string]$Name)
    if (-not $script:ToolScratch.ContainsKey($Name)) {
        $path = Resolve-ExternalPath (Join-Path ([IO.Path]::GetTempPath()) ('SwiftBonsai-' + $Name + '-' + [guid]::NewGuid().ToString('N')))
        [void][IO.Directory]::CreateDirectory($path)
        $script:ToolScratch[$Name] = $path
    }
    return $script:ToolScratch[$Name]
}

function Resolve-PrismRustTool {
    <# .SYNOPSIS
    Resolves a packaged tool or builds reviewed sources in an external incremental lane.
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param([Parameter(Mandatory)][ValidatePattern('^[a-z-]+$')][string]$Name,
          [Parameter(Mandatory)][string]$ManifestRelativePath, [string]$ToolPath)
    if ($ToolPath) {
        if (-not [IO.Path]::IsPathFullyQualified($ToolPath) -or $ToolPath -match '%[^%]+%' -or -not [IO.File]::Exists($ToolPath)) {
            throw 'ToolPath must name an existing expanded absolute executable.'
        }
        return [IO.Path]::GetFullPath($ToolPath)
    }
    $scratch = Get-PrismToolScratch -Name $Name
    $cargo = (Get-Command -Name cargo -CommandType Application -ErrorAction Stop).Source
    $manifest = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot $ManifestRelativePath))
    $target = Join-Path $scratch 'target'
    $arguments = @('+nightly-2026-10-22-aarch64-pc-windows-msvc', 'build', '--locked', '--release', '--manifest-path', $manifest, '--target-dir', $target)
    $log = Join-Path $scratch ('cargo-build-' + [guid]::NewGuid().ToString('N') + '.log')
    $result = Invoke-BonsaiNativeProcess -Executable $cargo -ArgumentList $arguments -WorkingDirectory $scratch -LogPath $log
    if ($result.ExitCode -ne 0) { throw "Tool build failed; inspect $log`n$($result.StandardError)" }
    $tool = Join-Path $target ('release/' + $Name + '.exe')
    if (-not [IO.File]::Exists($tool)) { throw "Tool build produced no executable; inspect $log" }
    return $tool
}

function Invoke-PrismPolicy {
    <# .SYNOPSIS
    Runs one bounded JSON maintenance operation with a packaged or source-built tool.
    #>
    [CmdletBinding()]
    [OutputType([Collections.IDictionary])]
    param([Parameter(Mandatory)][ValidateSet('prism-source','prism-baseline')][string]$Command,
          [Collections.IDictionary]$Request = @{}, [string]$ToolPath,
          [ValidateRange(0,86400)][int]$TimeoutSeconds = 60)
    if (-not $ToolPath -and $env:PRISM_CONFORMANCE_POLICY_PATH) { $ToolPath = $env:PRISM_CONFORMANCE_POLICY_PATH }
    $tool = Resolve-PrismRustTool -Name 'prism-conformance-policy' -ManifestRelativePath '../tools/conformance-policy/Cargo.toml' -ToolPath $ToolPath
    $scratch = Get-PrismToolScratch -Name 'prism-conformance-policy'
    $file = Get-Item -LiteralPath $tool
    $identity = $tool + ':' + $file.Length + ':' + $file.LastWriteTimeUtc.Ticks
    if (-not $script:CheckedTools.ContainsKey($identity)) {
        $probe = Invoke-BonsaiNativeProcess -Executable $tool -ArgumentList @('--version') -WorkingDirectory $scratch -TimeoutSeconds 10
        if ($probe.ExitCode -ne 0) { throw 'The maintenance tool version probe failed.' }
        $version = $probe.StandardOutput | ConvertFrom-Json -AsHashtable -NoEnumerate -DateKind String
        if ($version -isnot [Collections.IDictionary] -or $version.Name -cne 'prism-conformance-policy' -or $version.Version -cne '0.1.0' -or $version.ProtocolVersion -ne 1 -or
            -not $version.Contains('Commands') -or $version.Commands -isnot [array] -or
            @($version.Commands | Where-Object { $_ -isnot [string] }).Count) {
            throw 'Use a compatible prism-conformance-policy executable (version 0.1.0, protocol 1).'
        }
        $script:CheckedTools[$identity] = $version.Commands
    }
    if ($Command -cnotin $script:CheckedTools[$identity]) { throw "The selected maintenance tool does not support '$Command'; use a current build." }
    $lane = Join-Path $scratch ([guid]::NewGuid().ToString('N'))
    [void][IO.Directory]::CreateDirectory($lane)
    $requestPath = Join-Path $lane 'request.json'
    $payload = [ordered]@{}
    foreach ($key in $Request.Keys) { $payload[$key] = $Request[$key] }

    $requestDepth = if ($Command -eq 'prism-source') { 100 } else { 64 }
    # Fan-in wraps a source artifact in the receipt and partition envelopes.
    $responseDepth = if ($Command -eq 'prism-source') { 104 } else { 64 }
    $json = ConvertTo-Json -InputObject $payload -Depth $requestDepth -Compress
    [IO.File]::WriteAllText($requestPath, $json, [Text.UTF8Encoding]::new($false))
    $result = Invoke-BonsaiNativeProcess -Executable $tool -ArgumentList @($Command, '--request', $requestPath) -WorkingDirectory $lane -TimeoutSeconds $TimeoutSeconds
    if ($result.ExitCode -ne 0) { throw "Maintenance command '$Command' failed: $($result.StandardError.Trim()) Evidence: $lane" }
    if ($result.StandardError) { Write-Verbose -Message $result.StandardError.TrimEnd() }
    $value = ConvertFrom-Json -InputObject $result.StandardOutput -AsHashtable -NoEnumerate -DateKind String -Depth $responseDepth
    if ($value -isnot [Collections.IDictionary]) { throw 'The maintenance tool returned no JSON object.' }
    [IO.File]::Delete($requestPath)
    [IO.Directory]::Delete($lane, $false)
    return $value
}

Export-ModuleMember -Function Get-PrismToolScratch, Resolve-PrismRustTool, Invoke-PrismPolicy

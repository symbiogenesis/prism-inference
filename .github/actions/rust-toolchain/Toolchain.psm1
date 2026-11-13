#Requires -Version 7.6
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

function Install-VerifiedRustToolchain {
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory)][string]$Toolchain,
        [Parameter(Mandatory)][string]$ExpectedHost,
        [Parameter(Mandatory)][string]$ToolchainPath,
        [string[]]$Components = @(),
        [switch]$CacheRestored,
        [switch]$HostedRunner,
        [scriptblock]$RunRustup = {
            param([string[]]$Arguments)
            & rustup @Arguments
            if ($LASTEXITCODE -ne 0) { throw "rustup $($Arguments -join ' ') exited with code $LASTEXITCODE." }
        }
    )
    if ($Toolchain -cnotmatch '^(?:(?:stable|beta|nightly)-[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]+\.[0-9]+\.[0-9]+)-(?<host>[a-z0-9_]+(?:-[a-z0-9_]+){2,})$' -or $Matches.host -cne $ExpectedHost) {
        throw 'Expected an exact toolchain with its matching host triple.'
    }
    if (-not [IO.Path]::IsPathFullyQualified($ToolchainPath) -or
        [IO.Path]::GetFileName([IO.Path]::TrimEndingDirectorySeparator($ToolchainPath)) -cne $Toolchain -or
        [IO.Path]::GetFileName([IO.Path]::GetDirectoryName([IO.Path]::TrimEndingDirectorySeparator($ToolchainPath))) -cne 'toolchains') {
        throw 'Expected the exact pinned rustup toolchain directory.'
    }
    foreach ($component in $Components) {
        if ($component -cnotmatch '^[a-z0-9][a-z0-9-]*$') { throw 'Invalid Rust component name.' }
    }
    $install = {
        $arguments = @('toolchain', 'install', $Toolchain, '--profile', 'minimal', '--no-self-update')
        if ($Components.Count) { $arguments += @('--component', ($Components -join ',')) }
        & $RunRustup -Arguments $arguments | ForEach-Object { Write-Information -MessageData $_ -InformationAction Continue }
    }
    $verify = {
        # rustup run never installs a missing compiler implicitly.
        $identity = @(& $RunRustup -Arguments @('run', $Toolchain, 'rustc', '-vV'))
        if ($identity -cnotcontains "host: $ExpectedHost") { throw 'Rust compiler has an unexpected host.' }
        $sysroot = ((& $RunRustup -Arguments @('run', $Toolchain, 'rustc', '--print', 'sysroot')) | Out-String).Trim()
        if ([IO.Path]::TrimEndingDirectorySeparator([IO.Path]::GetFullPath($sysroot)) -cne [IO.Path]::TrimEndingDirectorySeparator([IO.Path]::GetFullPath($ToolchainPath))) {
            throw 'Rust compiler resolved outside the exact toolchain directory.'
        }
        $installed = @(& $RunRustup -Arguments @('component', 'list', '--toolchain', $Toolchain, '--installed', '--quiet'))
        foreach ($component in $Components) {
            if ($installed -cnotcontains $component -and $installed -cnotcontains "$component-$ExpectedHost") {
                throw "Required Rust component is missing: $component"
            }
        }
        return ,$identity
    }
    if (-not $CacheRestored) { & $install }
    try {
        $identity = & $verify
    } catch {
        # A bad optional cache may be discarded once on an ephemeral hosted
        # runner. Never uninstall an owner-managed installation or hide a bad
        # fresh install, and require the replacement to pass the same checks.
        if (-not ($CacheRestored -and $HostedRunner)) { throw }
        Write-Warning "Discarding the invalid restored compiler: $_"
        & $RunRustup -Arguments @('toolchain', 'uninstall', $Toolchain) | ForEach-Object { Write-Information -MessageData $_ -InformationAction Continue }
        & $install
        $identity = & $verify
    }
    foreach ($line in $identity) { Write-Information -MessageData $line -InformationAction Continue }
    return [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes(($identity -join "`n")))).ToLowerInvariant()
}

Export-ModuleMember -Function Install-VerifiedRustToolchain

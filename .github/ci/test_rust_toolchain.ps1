#Requires -Version 7.6
[CmdletBinding()]
[OutputType([void])]
param([string]$Case = '*')
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot '../actions/rust-toolchain/Toolchain.psm1') -Force

function Assert-Condition {
    [CmdletBinding()]
    [OutputType([void])]
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}
function Get-RustupFixture {
    [CmdletBinding()]
    [OutputType([hashtable])]
    param()
    $toolchain = 'nightly-2026-11-09-aarch64-pc-windows-msvc'
    $path = Join-Path ([IO.Path]::GetTempPath()) "fixture-rustup/toolchains/$toolchain"
    $state = @{
        Installs = 0; Uninstalls = 0; Calls = 0; Repaired = $false
        Broken = ''; ReplacementBroken = $false
        Host = 'aarch64-pc-windows-msvc'; Path = $path
    }
    $runner = {
        param([string[]]$Arguments)
        $state.Calls++
        switch ($Arguments[0]) {
            'toolchain' {
                if ($Arguments[1] -ceq 'uninstall') { $state.Uninstalls++ }
                elseif ($Arguments[1] -ceq 'install') { $state.Installs++; $state.Repaired = $true }
                else { throw 'Unexpected toolchain operation.' }
            }
            'run' {
                $broken = -not $state.Repaired -or $state.ReplacementBroken
                if ($Arguments[-1] -ceq '-vV') {
                    if ($broken -and $state.Broken -ceq 'unreadable') { throw 'Unusable cached rustc.' }
                    'rustc 1.94.0-nightly (fixture)'
                    if ($broken -and $state.Broken -ceq 'host') { 'host: x86_64-pc-windows-msvc' }
                    else { "host: $($state.Host)" }
                } elseif ($Arguments[-1] -ceq 'sysroot') {
                    if ($broken -and $state.Broken -ceq 'sysroot') { Join-Path ([IO.Path]::GetTempPath()) 'unrelated-toolchain' }
                    else { $state.Path }
                } else { throw 'Unexpected compiler command.' }
            }
            'component' {
                "rustfmt-$($state.Host)"
                if (-not ((-not $state.Repaired -or $state.ReplacementBroken) -and $state.Broken -ceq 'component')) { "clippy-$($state.Host)" }
            }
            default { throw 'Unexpected rustup command.' }
        }
    }.GetNewClosure()
    return @{
        State = $state
        Options = @{
            Toolchain = $toolchain; ExpectedHost = $state.Host
            ToolchainPath = $path; Components = @('rustfmt', 'clippy')
            CacheRestored = $true; HostedRunner = $true; RunRustup = $runner
        }
    }
}

$cases = [ordered]@{
    cold_install_is_verified = {
        $fixture = Get-RustupFixture
        $fixture.Options.CacheRestored = $false
        $options = $fixture.Options
        $digest = Install-VerifiedRustToolchain @options
        Assert-Condition ($digest -cmatch '^[a-f0-9]{64}$') 'Expected the verified compiler digest.'
        Assert-Condition ($fixture.State.Installs -eq 1 -and $fixture.State.Uninstalls -eq 0) 'Cold install must not uninstall anything.'
    }
    exact_hit_skips_install = {
        $fixture = Get-RustupFixture
        $options = $fixture.Options
        $digest = Install-VerifiedRustToolchain @options
        Assert-Condition ($digest -cmatch '^[a-f0-9]{64}$') 'Expected the verified compiler digest.'
        Assert-Condition ($fixture.State.Installs -eq 0 -and $fixture.State.Uninstalls -eq 0) 'An intact exact hit must not install anything.'
    }
    invalid_hosted_cache_repairs_once = {
        foreach ($broken in @('component', 'host', 'sysroot', 'unreadable')) {
            $fixture = Get-RustupFixture
            $fixture.State.Broken = $broken
            $options = $fixture.Options
            $digest = Install-VerifiedRustToolchain @options
            Assert-Condition ($digest -cmatch '^[a-f0-9]{64}$') "Expected verified replacement after $broken."
            Assert-Condition ($fixture.State.Installs -eq 1 -and $fixture.State.Uninstalls -eq 1) "Expected one exact-toolchain repair after $broken."
        }
    }
    replacement_failure_stays_fatal = {
        $fixture = Get-RustupFixture
        $fixture.State.Broken = 'component'
        $fixture.State.ReplacementBroken = $true
        $options = $fixture.Options
        $failed = $false
        try { Install-VerifiedRustToolchain @options | Out-Null } catch { $failed = $true }
        Assert-Condition $failed 'An invalid replacement must fail.'
        Assert-Condition ($fixture.State.Installs -eq 1 -and $fixture.State.Uninstalls -eq 1) 'Invalid replacement must not retry.'
    }
    owner_install_is_never_removed = {
        $fixture = Get-RustupFixture
        $fixture.State.Broken = 'component'
        $fixture.Options.HostedRunner = $false
        $options = $fixture.Options
        $failed = $false
        try { Install-VerifiedRustToolchain @options | Out-Null } catch { $failed = $true }
        Assert-Condition $failed 'An invalid owner installation must fail verification.'
        Assert-Condition ($fixture.State.Installs -eq 0 -and $fixture.State.Uninstalls -eq 0) 'Never remove an owner installation.'
    }
    fresh_failure_is_not_retried = {
        $fixture = Get-RustupFixture
        $fixture.State.Broken = 'host'
        $fixture.State.ReplacementBroken = $true
        $fixture.Options.CacheRestored = $false
        $options = $fixture.Options
        $failed = $false
        try { Install-VerifiedRustToolchain @options | Out-Null } catch { $failed = $true }
        Assert-Condition $failed 'An invalid fresh compiler must fail verification.'
        Assert-Condition ($fixture.State.Installs -eq 1 -and $fixture.State.Uninstalls -eq 0) 'Fresh installation failure must not retry or uninstall.'
    }
    unsafe_identity_is_rejected_before_rustup = {
        $fixture = Get-RustupFixture
        $fixture.Options.Toolchain = '../nightly'
        $options = $fixture.Options
        $failed = $false
        try { Install-VerifiedRustToolchain @options | Out-Null } catch { $failed = $true }
        Assert-Condition $failed 'An unpinned toolchain must fail.'
        Assert-Condition ($fixture.State.Calls -eq 0) 'Reject unsafe input before any rustup command.'
    }
}
$selected = @($cases.Keys | Where-Object { $_ -like $Case })
if (-not $selected.Count) { throw "Unknown toolchain fixture: $Case" }
foreach ($name in $selected) {
    & $cases[$name]
    Write-Information -MessageData "PASS $name" -InformationAction Continue
}
Write-Information -MessageData "Passed $($selected.Count) toolchain fixture(s)." -InformationAction Continue

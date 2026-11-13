#Requires -Version 7.6
[CmdletBinding()]
[OutputType([string])]
param([Parameter(Mandatory)][string]$ActionlintPath)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
. (Join-Path $PSScriptRoot 'Test-Workflow.ps1') -ActionlintPath $ActionlintPath
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('Prism-Workflow-Control-' + [guid]::NewGuid().ToString('N'))
try {
    # actionlint enables local action-input validation only within a repository.
    [void][IO.Directory]::CreateDirectory((Join-Path $scratch '.git'))
    $directory = Join-Path $scratch '.github/workflows'
    [void][IO.Directory]::CreateDirectory($directory)
    $actionDirectory = Join-Path $scratch '.github/actions/control'
    [void][IO.Directory]::CreateDirectory($actionDirectory)
    [IO.File]::WriteAllText((Join-Path $actionDirectory 'action.yml'), "name: Control`ndescription: Control`ninputs:`n  label:`n    description: Control label`n    required: true`nruns:`n  using: composite`n  steps:`n    - shell: bash`n      run: echo control`n")
    $fixture = Join-Path $directory 'check.yml'
    $valid = "name: Control`non: push`njobs:`n  check:`n    runs-on: ubuntu-24.04`n    steps:`n      - uses: actions/checkout@$('a' * 40)`n      - run: echo control`n"
    $valid += "      - uses: ./.github/actions/control`n        with:`n          label: control`n"
    [IO.File]::WriteAllText($fixture, $valid)
    Invoke-CiWorkflowLint $scratch $ActionlintPath
    $mutations = [ordered]@{
        mutable_pin = $valid.Replace(('a' * 40), 'main')
        spaced_pin = $valid.Replace(('uses: actions/checkout@' + ('a' * 40)), '"uses" : actions/checkout@main')
        block_pin = $valid.Replace(('actions/checkout@' + ('a' * 40)), "|`n          actions/checkout@main")
        flow_pin = $valid.Replace(('uses: actions/checkout@' + ('a' * 40)), '{uses: actions/checkout@main}')
        flow_steps = "name: Control`non: push`njobs:`n  check:`n    runs-on: ubuntu-24.04`n    steps: [{uses: actions/checkout@main}]`n"
        unchecked_action_path = $valid.Replace(('actions/checkout@' + ('a' * 40)), './tools/custom')
        invalid_local_input = $valid.Replace('label: control', 'unknown: control')
        missing_action = $valid.Replace(('actions/checkout@' + ('a' * 40)), './.github/actions/missing')
        invalid_dependency = $valid.Replace('    runs-on:', "    needs: absent`n    runs-on:")
        invalid_expression = $valid.Replace('echo control', 'echo ${{ needs.absent.outputs.value }}')
        duplicate_job = $valid + "  check:`n    runs-on: ubuntu-24.04`n    steps:`n      - run: echo duplicate`n"
    }
    foreach ($case in $mutations.Keys) {
        [IO.File]::WriteAllText($fixture, $mutations[$case])
        $rejected = $false
        try { Invoke-CiWorkflowLint $scratch $ActionlintPath *> $null } catch { $rejected = $true }
        if (-not $rejected) { throw "Workflow preflight accepted mutation: $case" }
    }
    "Workflow negative controls passed: $($mutations.Count)."
} finally {
    $absolute = [IO.Path]::GetFullPath($scratch)
    $temporary = [IO.Path]::TrimEndingDirectorySeparator([IO.Path]::GetFullPath([IO.Path]::GetTempPath())) + [IO.Path]::DirectorySeparatorChar
    if (-not $absolute.StartsWith($temporary, [StringComparison]::OrdinalIgnoreCase) -or -not [IO.Path]::GetFileName($absolute).StartsWith('Prism-Workflow-Control-')) { throw 'Workflow control cleanup escaped temporary storage.' }
    Remove-Item -LiteralPath $absolute -Recurse -Force
}

#Requires -Version 7.6
param([Parameter(Mandatory)][string]$ActionlintPath)
. (Join-Path $PSScriptRoot 'Test-Workflow.ps1') -ActionlintPath $ActionlintPath
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('Prism-Workflow-Control-' + [guid]::NewGuid().ToString('N'))
try {
    $directory = Join-Path $scratch '.github/workflows'
    [void][IO.Directory]::CreateDirectory($directory)
    [void][IO.Directory]::CreateDirectory((Join-Path $scratch '.github/actions'))
    $fixture = Join-Path $directory 'check.yml'
    $valid = "name: Control`non: push`njobs:`n  check:`n    runs-on: ubuntu-24.04`n    steps:`n      - uses: actions/checkout@$('a' * 40)`n      - run: echo control`n"
    [IO.File]::WriteAllText($fixture, $valid)
    Invoke-CiWorkflowLint $scratch $ActionlintPath
    $mutations = [ordered]@{
        mutable_pin = $valid.Replace(('a' * 40), 'main')
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

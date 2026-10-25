#requires -Version 7.6
<#
.SYNOPSIS
Checks all repository PowerShell contracts and compiles its C# with strict analyzers.
.DESCRIPTION
Requires PSScriptAnalyzer 1.25.0 and the repository's .NET SDK. No dependencies are
installed. AnalyzerModulePath may select an isolated pinned module manifest.
C# extraction, build outputs, and logs are retained in a unique system-temp lane.
This validation does not run test suites, model inference, or benchmarks.
#>
[CmdletBinding()]
[OutputType([string])]
param(
    [string]$AnalyzerModulePath,
    [string]$DotNetPath
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot '../scripts/Bonsai.Native.psm1')

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$settings = Join-Path $repoRoot 'PSScriptAnalyzerSettings.psd1'
$requiredAnalyzerVersion = [version]'1.25.0'
Write-Information -MessageData "PowerShell standards: loading PSScriptAnalyzer $requiredAnalyzerVersion." -InformationAction Continue
if ($AnalyzerModulePath) {
    if (-not [IO.Path]::IsPathFullyQualified($AnalyzerModulePath) -or $AnalyzerModulePath -match '%[^%]+%') {
        throw 'AnalyzerModulePath must be an expanded absolute module manifest path.'
    }
    $analyzer = Import-Module -Name $AnalyzerModulePath -PassThru -Force
} else {
    $analyzer = Import-Module -Name PSScriptAnalyzer -RequiredVersion $requiredAnalyzerVersion -PassThru -Force
}
if ($analyzer.Name -ne 'PSScriptAnalyzer' -or $analyzer.Version -ne $requiredAnalyzerVersion) {
    throw "Standards validation requires exactly PSScriptAnalyzer $requiredAnalyzerVersion."
}
function Assert-ParameterContract {
    [CmdletBinding()]
    [OutputType([void])]
    param(
        [Parameter(Mandatory)][Management.Automation.Language.ParamBlockAst]$ParamBlock,
        [Parameter(Mandatory)][string]$Context,
        [switch]$RequireOutputType
    )
    foreach ($parameter in $ParamBlock.Parameters) {
        if (-not $parameter.Attributes.Where({ $_ -is [Management.Automation.Language.TypeConstraintAst] }, 'First').Count) {
            throw "${Context}: parameter $($parameter.Name) must declare a type."
        }
    }
    $attributes = @($ParamBlock.Attributes.TypeName.Name)
    if ('CmdletBinding' -notin $attributes) { throw "${Context}: declare CmdletBinding. Paste a param() block for parameterless functions." }
    if ($RequireOutputType -and 'OutputType' -notin $attributes) { throw "${Context}: declare an explicit OutputType, including void for commands with no output." }
}

$scratch = [IO.Path]::GetFullPath((Join-Path ([IO.Path]::GetTempPath()) ('Prism-Standards-' + [guid]::NewGuid().ToString('N'))))
if (-not [IO.Path]::IsPathFullyQualified($scratch) -or $scratch -match '%[^%]+%') { throw 'The system temp path must be expanded and absolute.' }
$embeddedDirectory = Join-Path $scratch 'embedded-csharp'
[void][IO.Directory]::CreateDirectory($embeddedDirectory)
$files = [Collections.Generic.List[IO.FileInfo]]::new()
$directories = [Collections.Generic.Stack[string]]::new()
$directories.Push($repoRoot)
while ($directories.Count) {
    foreach ($entry in (Get-ChildItem -LiteralPath $directories.Pop() -Force)) {
        if ($entry.PSIsContainer) {
            if ($entry.Name -notin @('.git', '.agents', '.codex', 'target', 'bin', 'obj', 'artifacts', 'node_modules', '.venv')) {
                if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Do not hide repository source behind a directory link: $($entry.FullName)" }
                $directories.Push($entry.FullName)
            }
        } elseif ($entry.Extension -in @('.ps1', '.psm1', '.psd1')) {
            $files.Add($entry)
        }
    }
}
Write-Information -MessageData "PowerShell standards: discovered $($files.Count) source files." -InformationAction Continue
$compilerOptions = @('/nullable:enable', '/checked+', '/unsafe-', '/warn:9999', '/warnaserror+', '/langversion:14.0')
$embeddedCount = 0
$diagnosticCount = 0
foreach ($file in ($files | Sort-Object -Property FullName -Unique)) {
    $tokens = $null
    $parseErrors = $null
    $ast = [Management.Automation.Language.Parser]::ParseFile($file.FullName, [ref]$tokens, [ref]$parseErrors)
    if ($parseErrors.Count) { throw "Parse errors in $($file.FullName): $($parseErrors.Message -join '; ')" }
    if ($file.Extension -ne '.psd1') {
        if ($null -eq $ast.ScriptRequirements -or $ast.ScriptRequirements.RequiredPSVersion -lt [version]'7.6') {
            throw "$($file.FullName): declare #requires -Version 7.6 or later."
        }
        if ($file.Extension -eq '.ps1') {
            if ($null -eq $ast.ParamBlock) { throw "$($file.FullName): declare a typed script param block." }
            Assert-ParameterContract -ParamBlock $ast.ParamBlock -Context $file.FullName -RequireOutputType
        }
        $statements = $ast.EndBlock.Statements
        if (-not $statements.Where({ $_.Extent.Text -match '^Set-StrictMode\s+-Version\s+Latest$' }, 'First').Count) {
            throw "$($file.FullName): enable Set-StrictMode -Version Latest at script scope."
        }
        foreach ($preference in @('ErrorActionPreference', 'PSNativeCommandUseErrorActionPreference')) {
            $expected = if ($preference -eq 'ErrorActionPreference') { "'Stop'" } else { '$true' }
            if (-not $statements.Where({
                $_ -is [Management.Automation.Language.AssignmentStatementAst] -and
                $_.Left.Extent.Text -ceq ('$' + $preference) -and $_.Right.Extent.Text -ceq $expected
            }, 'First').Count) { throw "$($file.FullName): set `$$preference = $expected at script scope." }
        }
        $functions = $ast.FindAll({ param([Management.Automation.Language.Ast]$Node) $Node -is [Management.Automation.Language.FunctionDefinitionAst] }, $true)
        foreach ($function in $functions) {
            if ($null -eq $function.Body.ParamBlock -or ($null -ne $function.Parameters -and $function.Parameters.Count)) {
                throw "$($file.FullName): $($function.Name) must use a typed advanced-function param block."
            }
            Assert-ParameterContract -ParamBlock $function.Body.ParamBlock -Context "$($file.Name)/$($function.Name)" -RequireOutputType
        }
        $addTypeCommands = $ast.FindAll({
            param([Management.Automation.Language.Ast]$Node)
            $Node -is [Management.Automation.Language.CommandAst] -and $Node.GetCommandName() -eq 'Add-Type'
        }, $true)
        foreach ($command in $addTypeCommands) {
            $elements = $command.CommandElements
            $optionsIndex = -1
            for ($index = 1; $index -lt $elements.Count; $index++) {
                if ($elements[$index] -is [Management.Automation.Language.CommandParameterAst] -and $elements[$index].ParameterName -eq 'CompilerOptions') { $optionsIndex = $index + 1 }
                if ($elements[$index] -is [Management.Automation.Language.CommandParameterAst] -and $elements[$index].ParameterName -eq 'TypeDefinition') {
                    if ($index + 1 -ge $elements.Count -or $elements[$index + 1] -isnot [Management.Automation.Language.StringConstantExpressionAst]) {
                        throw "$($file.FullName): Add-Type TypeDefinition must use a literal source so it can be analyzed."
                    }
                    $embeddedCount++
                    [IO.File]::WriteAllText((Join-Path $embeddedDirectory ('Embedded{0:D3}.cs' -f $embeddedCount)), $elements[$index + 1].Value)
                }
            }
            if ($optionsIndex -lt 0 -or $optionsIndex -ge $elements.Count) { throw "$($file.FullName): Add-Type must declare strict compiler options." }
            $declaredOptions = @($elements[$optionsIndex].SafeGetValue())
            if ($declaredOptions.Count -ne $compilerOptions.Count -or $compilerOptions.Where({ $_ -cnotin $declaredOptions }, 'First').Count) {
                throw "$($file.FullName): Add-Type compiler options must match the strict C# policy."
            }
        }
    }
    Write-Information -MessageData "PowerShell standards: analyzing $($file.FullName)." -InformationAction Continue
    $diagnostics = @(Invoke-ScriptAnalyzer -Path $file.FullName -Settings $settings -ErrorAction Stop)
    foreach ($diagnostic in $diagnostics) {
        Write-Information -MessageData ('{0}:{1}: {2} {3}: {4}' -f $file.FullName, $diagnostic.Line, $diagnostic.Severity, $diagnostic.RuleName, $diagnostic.Message) -InformationAction Continue
    }
    $diagnosticCount += $diagnostics.Count
}
if ($diagnosticCount) { throw "PowerShell standards failed with $diagnosticCount diagnostic(s)." }

# Prism currently has no embedded C# helpers. Continue extracting and validating
# every Add-Type source above, and enable the strict build as soon as one exists.
if ($embeddedCount -eq 0) {
    return "PowerShell standards passed for $($files.Count) files; no embedded C# helpers were found. Evidence: $scratch"
}
if (-not $DotNetPath) { $DotNetPath = (Get-Command -Name dotnet -CommandType Application -ErrorAction Stop).Source }
if (-not [IO.Path]::IsPathFullyQualified($DotNetPath) -or $DotNetPath -match '%[^%]+%' -or
    -not [IO.File]::Exists($DotNetPath)) { throw 'DotNetPath must name an existing expanded absolute executable.' }

$project = Join-Path $repoRoot 'tools/validation/CSharp.Strict.csproj'
if (-not [IO.File]::Exists($project)) { throw 'The strict C# validation project is missing.' }
# SDK selection follows the native working directory, which deliberately stays
# outside the checkout. Carry the repository pin into this owned build lane.
$sdkPin = Join-Path $repoRoot 'global.json'
if (-not [IO.File]::Exists($sdkPin)) { throw 'The repository .NET SDK pin is missing.' }
[IO.File]::Copy($sdkPin, (Join-Path $scratch 'global.json'), $false)
$arguments = @('build', $project, '--configuration', 'Release', '--artifacts-path', (Join-Path $scratch 'artifacts'), "-p:EmbeddedSourcesDirectory=$embeddedDirectory", '--nologo')
Write-Information -MessageData 'C# standards: starting the strict Release build.' -InformationAction Continue
$log = Join-Path $scratch 'csharp-build.log'
$result = Invoke-BonsaiNativeProcess -Executable $DotNetPath -ArgumentList $arguments -WorkingDirectory $scratch -TimeoutSeconds 120 -LogPath $log
if ($result.ExitCode -ne 0) { throw "Strict C# validation failed. See $log`n$($result.StandardOutput)$($result.StandardError)" }
Write-Output "PowerShell standards passed for $($files.Count) files; strict C# build passed including $embeddedCount extracted helper(s). Evidence: $scratch"

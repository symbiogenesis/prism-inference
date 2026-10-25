#requires -Version 7.6
<# .SYNOPSIS
Checks focused archive selection, byte retention and exclusive publication.
#>
[CmdletBinding()]
[OutputType([string])]
param([Parameter(Mandatory)][ValidateSet('Ordinary', 'Backend')][string]$Case)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot 'Bonsai.TestSupport.psm1')
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('SwiftBonsai-PrismEvidence-' + [guid]::NewGuid().ToString('N'))
$evidence = Join-Path $scratch 'evidence'
$names = @('report.json', 'command.log', '.hidden.log', 'ignored.txt', 'source/code.cpp', 'build/engine.exe',
    'native-case/input.bin', 'native-case/response.json', 'case-bridge/nested/output.json',
    'cpp/phase-record.json', 'cpp/case/actions-0001.json', 'cpp/case/checkpoint-0001.json',
    'cpp/case/input-state.json', 'cpp/case/result.bin', 'cpp/case/frame.jsonl', 'cpp/case/command.log', 'cpp/case/unselected.json')
foreach ($name in $names) {
    $path = Join-Path $evidence $name
    [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($path))
    [IO.File]::WriteAllBytes($path, [Text.Encoding]::UTF8.GetBytes("retained bytes: $name`n"))
}
[IO.File]::WriteAllBytes((Join-Path $evidence 'native-case/input.bin'), [byte[]]@(0, 255, 128, 13, 10, 0))
if ($IsWindows) { [IO.File]::SetAttributes((Join-Path $evidence '.hidden.log'), [IO.FileAttributes]::Hidden) }
$expected = if ($Case -ceq 'Ordinary') {
    @('report.json', 'command.log', '.hidden.log', 'native-case/input.bin', 'native-case/response.json', 'case-bridge/nested/output.json')
} else {
    @('report.json', 'command.log', '.hidden.log', 'native-case/input.bin', 'cpp/phase-record.json', 'cpp/case/actions-0001.json',
        'cpp/case/checkpoint-0001.json', 'cpp/case/input-state.json', 'cpp/case/result.bin', 'cpp/case/frame.jsonl', 'cpp/case/command.log')
}
$scriptPath = Join-Path $PSScriptRoot '../scripts/Compress-PrismEvidence.ps1'
$archivePath = Join-Path $scratch 'evidence.zip'
$result = & $scriptPath -EvidenceDirectory $evidence -ArchivePath $archivePath -EvidenceKind $Case
Assert ($result.FileCount -eq $expected.Count) 'Archive receipt must count every selected file exactly once.'
$archive = [IO.Compression.ZipFile]::OpenRead($archivePath)
try {
    Assert ($archive.Entries.Count -eq $expected.Count) 'Archive must contain exactly the selected diagnostics.'
    foreach ($name in $expected) {
        $entry = $archive.GetEntry($name)
        Assert ($null -ne $entry) "Missing diagnostic: $name"
        $inputStream = $entry.Open()
        $buffer = [IO.MemoryStream]::new()
        try {
            $inputStream.CopyTo($buffer)
            $actual = [Convert]::ToBase64String($buffer.ToArray())
        } finally { $buffer.Dispose(); $inputStream.Dispose() }
        Assert ($actual -ceq [Convert]::ToBase64String([IO.File]::ReadAllBytes((Join-Path $evidence $name)))) "Diagnostic bytes changed: $name"
    }
} finally { $archive.Dispose() }
Assert-Failure { & $scriptPath -EvidenceDirectory $evidence -ArchivePath $archivePath -EvidenceKind $Case } 'An existing evidence archive must not be overwritten.'
Assert-Failure { & $scriptPath -EvidenceDirectory $evidence -ArchivePath (Join-Path $evidence 'recursive.zip') -EvidenceKind $Case } 'An archive must not recursively select itself.'
if ($IsWindows) {
    $target = Join-Path $scratch 'link-target'
    [void][IO.Directory]::CreateDirectory($target)
    $link = Join-Path $evidence 'native-hidden-link'
    [void](New-Item -ItemType Junction -Path $link -Value $target)
    [IO.File]::SetAttributes($link, [IO.File]::GetAttributes($link) -bor [IO.FileAttributes]::Hidden)
    $rejectedArchive = Join-Path $scratch 'hidden-link.zip'
    Assert-Failure { & $scriptPath -EvidenceDirectory $evidence -ArchivePath $rejectedArchive -EvidenceKind $Case } 'Hidden filesystem links must be rejected.'
    Assert (-not [IO.File]::Exists($rejectedArchive)) 'Link rejection must happen before archive publication.'
}
return "Prism evidence $Case passed; evidence: $scratch"

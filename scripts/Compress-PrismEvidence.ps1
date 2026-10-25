#requires -Version 7.6
<# .SYNOPSIS
Streams the existing Prism diagnostic selection into one fast-compressed artifact.
#>
[CmdletBinding()]
[OutputType([Collections.IDictionary])]
param(
    [Parameter(Mandatory)][string]$EvidenceDirectory,
    [Parameter(Mandatory)][string]$ArchivePath,
    [Parameter(Mandatory)][ValidateSet('Ordinary', 'Backend')][string]$EvidenceKind
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Import-Module (Join-Path $PSScriptRoot 'Bonsai.Common.psm1')
$source = Resolve-ExternalPath $EvidenceDirectory
$destination = Resolve-ExternalPath $ArchivePath
$prefix = $source.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
if (-not [IO.Directory]::Exists($source) -or $destination.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) -or
    [IO.Path]::GetExtension($destination) -cne '.zip') {
    throw 'Use an existing evidence directory and a separate external .zip destination.'
}
$files = [Collections.Generic.List[IO.FileInfo]]::new()
$directories = [Collections.Generic.Stack[string]]::new()
$directories.Push($source)
while ($directories.Count) {
    $directory = $directories.Pop()
    foreach ($item in (Get-ChildItem -LiteralPath $directory -Force)) {
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Prism evidence must not contain filesystem links.' }
        $rootItem = $directory -ceq $source
        if ($item.PSIsContainer) {
            if ($EvidenceKind -ceq 'Backend' -or -not $rootItem -or $item.Name -like 'native-*' -or $item.Name -like '*-bridge') {
                $directories.Push($item.FullName)
            }
            continue
        }
        $selected = if ($EvidenceKind -ceq 'Ordinary') {
            -not $rootItem -or $item.Extension -in @('.json', '.log')
        } else {
            ($rootItem -and $item.Extension -ceq '.json') -or
            $item.Extension -in @('.jsonl', '.bin', '.log') -or
            $item.Name -in @('phase-record.json', 'transport-failure.json', 'case-receipt.json', 'retention-receipt.json', 'input-state.json') -or
            $item.Name -like 'checkpoint-*.json' -or $item.Name -like 'actions-*.json'
        }
        if ($selected) { $files.Add($item) }
    }
}
if (-not $files.Count) { throw 'No selected Prism evidence files were produced.' }
[void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($destination))
$stream = [IO.FileStream]::new($destination, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
try {
    $archive = [IO.Compression.ZipArchive]::new($stream, [IO.Compression.ZipArchiveMode]::Create, $true)
    try {
        [long]$inputBytes = 0
        foreach ($file in $files) {
            $relative = [IO.Path]::GetRelativePath($source, $file.FullName).Replace('\', '/')
            [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive, $file.FullName, $relative, [IO.Compression.CompressionLevel]::Fastest)
            $inputBytes += $file.Length
        }
    } finally { $archive.Dispose() }
} finally { $stream.Dispose() }
return @{ ArchivePath=$destination; EvidenceKind=$EvidenceKind; FileCount=$files.Count; InputBytes=$inputBytes; ArchiveBytes=([IO.FileInfo]::new($destination)).Length }

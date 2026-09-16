param(
    [Parameter(Mandatory=$true)][string]$Source,
    [Parameter(Mandatory=$true)][string]$OutputPdf
)
$ErrorActionPreference = 'Stop'
$sourcePath = (Resolve-Path -LiteralPath $Source).Path
$outputPath = [IO.Path]::GetFullPath($OutputPdf)
if (Test-Path -LiteralPath $outputPath) { throw 'Refusing to overwrite an existing QA PDF.' }
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($outputPath)) | Out-Null
$before = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $initialCount = $word.Documents.Count
    $document = $word.Documents.Open($sourcePath, $false, $true, $false)
    $document.ExportAsFixedFormat($outputPath, 17)
    Write-Output "PDF exported: $outputPath"
} finally {
    if ($null -ne $document) { $document.Close(0) }
    if ($null -ne $word -and $initialCount -eq 0) { $word.Quit(0) }
}
$after = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash
if ($before -ne $after) { throw 'Source report changed during read-only export.' }
Write-Output "Source SHA256 unchanged: $after"

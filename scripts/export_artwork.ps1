<# Export an external editable Figure 1 or Figure 5a slide using installed PowerPoint.
   This opens the input read-only and never saves changes to it. #>
param(
    [Parameter(Mandatory=$true)][string]$InputPptx,
    [Parameter(Mandatory=$true)][string]$OutputPdf
)
$ErrorActionPreference = 'Stop'
$sourcePath = (Resolve-Path -LiteralPath $InputPptx).Path
$outputPath = [IO.Path]::GetFullPath($OutputPdf)
if ([IO.Path]::GetExtension($outputPath) -ne '.pdf') { throw 'Output must be a PDF' }
if (Test-Path -LiteralPath $outputPath) { throw 'Output already exists; choose a new path' }
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($outputPath)) | Out-Null
$powerpoint = New-Object -ComObject PowerPoint.Application
$presentation = $null
try {
    $presentation = $powerpoint.Presentations.Open($sourcePath, -1, 0, 0)
    if ($presentation.Slides.Count -ne 1) { throw 'Expected a single artwork slide' }
    $presentation.SaveCopyAs($outputPath, 32, -1)
} finally {
    if ($null -ne $presentation) {
        $presentation.Close()
        [Runtime.InteropServices.Marshal]::ReleaseComObject($presentation) | Out-Null
    }
    # Do not quit a possibly pre-existing user PowerPoint session.
    [Runtime.InteropServices.Marshal]::ReleaseComObject($powerpoint) | Out-Null
}

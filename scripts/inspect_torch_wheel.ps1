# Download and inspect Torch 2.6.2 wheels locally (PowerShell)
# Usage: .\.venv\Scripts\Activate.ps1; pwsh ./scripts/inspect_torch_wheel.ps1

param(
    [string]$TorchVersion = "2.6.2",
    [string]$TorchVisionVersion = "0.16.2",
    [string]$DownloadDir = "$env:TEMP\torch-wheels"
)

Set-StrictMode -Version Latest

if (!(Test-Path -Path $DownloadDir)) { New-Item -ItemType Directory -Path $DownloadDir | Out-Null }

Write-Host "Downloading wheels to: $DownloadDir"
python -m pip download "torch==$TorchVersion" "torchvision==$TorchVisionVersion" --no-deps -d $DownloadDir

Get-ChildItem -Path $DownloadDir -Filter *.whl | ForEach-Object {
    $f = $_.FullName
    Write-Host "---- $f ----"
    # List the top-level contents
    & 7z l $f | Select-Object -First 40
    # Try to unzip RECORD if present
    try {
        $record = & 7z e -so $f "*.dist-info\RECORD" 2>$null
        if ($record) { $record | Select-Object -First 200 }
    } catch { }

    # Search for likely native lib references inside the wheel
    try {
        & 7z e -so $f | Select-String -Pattern "libuv|tensorpipe|uv_|libcurel|libssl|libcrypto" -SimpleMatch -AllMatches | ForEach-Object { $_.Matches }
    } catch { }
}

Write-Host "Optional: run pip-audit against an extracted wheel install"
Write-Host "To perform a deeper SCA on the wheel contents, install 'pip-audit' and run it against an extracted install folder." 
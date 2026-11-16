# Windows-only Face Access runner. Do not use on Ventuno — use run_access_app.sh there.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (Test-Path ".\faceenv\Scripts\Activate.ps1") {
  . .\faceenv\Scripts\Activate.ps1
}

$env:CAMERA_BACKEND = if ($env:CAMERA_BACKEND) { $env:CAMERA_BACKEND } else { "opencv" }
$env:CAMERA_INDEX = if ($env:CAMERA_INDEX) { $env:CAMERA_INDEX } else { "0" }
$env:MOORCHEH_BASE_URL = if ($env:MOORCHEH_BASE_URL) { $env:MOORCHEH_BASE_URL } else { "http://localhost:8083" }
$env:FACE_ACCESS_HOST = if ($env:FACE_ACCESS_HOST) { $env:FACE_ACCESS_HOST } else { "127.0.0.1" }
$env:FACE_ACCESS_PORT = if ($env:FACE_ACCESS_PORT) { $env:FACE_ACCESS_PORT } else { "8501" }

Write-Host "[face-access-win] Edge: $($env:MOORCHEH_BASE_URL)"
Write-Host "[face-access-win] Camera index: $($env:CAMERA_INDEX)"
Write-Host "[face-access-win] UI: http://127.0.0.1:$($env:FACE_ACCESS_PORT)/"

python -m access_app.server

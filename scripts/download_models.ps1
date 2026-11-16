# Download YuNet + SFace ONNX models into ../models
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Models = Join-Path $Root "models"
New-Item -ItemType Directory -Force -Path $Models | Out-Null

$files = @(
  @{
    Name = "face_detection_yunet_2023mar.onnx"
    Url  = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
  },
  @{
    Name = "face_recognition_sface_2021dec.onnx"
    Url  = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
  }
)

foreach ($f in $files) {
  $out = Join-Path $Models $f.Name
  if (Test-Path $out) {
    Write-Host "[models] Already have $($f.Name)"
    continue
  }
  Write-Host "[models] Downloading $($f.Name) ..."
  Invoke-WebRequest -Uri $f.Url -OutFile $out -UseBasicParsing
}

Write-Host "[models] Done -> $Models"
Get-ChildItem $Models -Filter *.onnx | Select-Object Name, Length

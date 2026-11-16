#!/usr/bin/env bash
# Download YuNet + SFace ONNX models into ../models
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/models"
mkdir -p "$MODELS"

download() {
  local name="$1" url="$2"
  local out="$MODELS/$name"
  if [[ -f "$out" ]]; then
    echo "[models] Already have $name"
    return 0
  fi
  echo "[models] Downloading $name ..."
  curl -fL -o "$out" "$url"
}

download face_detection_yunet_2023mar.onnx \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
download face_recognition_sface_2021dec.onnx \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

echo "[models] Done -> $MODELS"
ls -lh "$MODELS"/*.onnx

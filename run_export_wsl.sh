#!/usr/bin/env bash
set -euo pipefail

VENV="$HOME/yolo-export-venv"
PROJ="/mnt/c/Users/patel/Downloads/Edge AI/moorcheh-face-access"

# Fresh CPU-only env (CUDA torch is huge and unnecessary for TFLite export)
rm -rf "$VENV"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install -U pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install ultralytics

cd "$PROJ"
python export_yolo_litert.py

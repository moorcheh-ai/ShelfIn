#!/usr/bin/env python3
"""Export YOLOv8n to LiteRT (.tflite) for Face Access object boxes.

Ultralytics LiteRT export only works on Mac or Linux x86_64 — not Windows,
not aarch64 boards. On Windows, use WSL Ubuntu:

  wsl
  cd /mnt/c/Users/patel/Downloads/Edge\\ AI/moorcheh-face-access
  python3 -m venv .wsl-export && source .wsl-export/bin/activate
  pip install ultralytics
  python export_yolo_litert.py

Then place yolov8n.tflite in the board project root (OBJECT_DETECTOR=auto|litert).
Until then, the board uses Torch yolov8n.pt automatically.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
PT = ROOT / "yolov8n.pt"
OUT = ROOT / "yolov8n.tflite"
IMGSZ = 256


def main() -> None:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "windows" or "aarch64" in machine or "arm64" in machine:
        print(
            "LiteRT export is not supported on this OS/CPU "
            f"({platform.system()} {platform.machine()}).\n"
            "Use WSL Ubuntu or a Mac/Linux x86_64 machine, then put "
            "yolov8n.tflite in the project root on the board.\n"
            "Meanwhile Face Access falls back to Torch yolov8n.pt "
            "(OBJECT_DETECTOR=auto).",
            file=sys.stderr,
        )
        raise SystemExit(2)

    model = YOLO(str(PT) if PT.is_file() else "yolov8n.pt")
    # Creates yolov8n.tflite (LiteRT / TFLite). imgsz must match OBJECT_IMGSZ (256).
    path = model.export(format="litert", imgsz=IMGSZ)
    exported = Path(str(path))
    if exported.resolve() != OUT.resolve() and exported.is_file():
        OUT.write_bytes(exported.read_bytes())
        print(f"Copied to: {OUT}")
    print(f"Exported: {path}")
    print("Put yolov8n.tflite in the board project root (or set LITERT_YOLO_MODEL).")


if __name__ == "__main__":
    main()

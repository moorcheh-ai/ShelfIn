"""Camera open helper: Picamera2 on Raspberry Pi CSI, OpenCV on Mac/USB/Ventuno.

Force a backend with:
  CAMERA_BACKEND=picamera2
  CAMERA_BACKEND=opencv

Pick a device index (Ventuno USB webcam is often not video0):
  CAMERA_INDEX=1
"""

from __future__ import annotations

import os
import sys
from typing import Any

import cv2
import numpy as np


def _want_backend() -> str | None:
    raw = (os.environ.get("CAMERA_BACKEND") or "").strip().lower()
    if raw in ("picamera2", "pi", "csi"):
        return "picamera2"
    if raw in ("opencv", "cv2", "usb", "webcam"):
        return "opencv"
    return None


def _is_raspberry_pi() -> bool:
    try:
        with open("/proc/device-tree/model", encoding="utf-8", errors="ignore") as f:
            return "Raspberry Pi" in f.read()
    except OSError:
        return False


def _camera_index_from_env(default: int = 0) -> int:
    raw = (os.environ.get("CAMERA_INDEX") or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"CAMERA_INDEX must be an int, got {raw!r}") from exc


def _opencv_index_candidates(preferred: int) -> list[int]:
    """Prefer CAMERA_INDEX, then scan a few V4L2 indices (Ventuno USB ≠ video0)."""
    ordered: list[int] = []
    for idx in (preferred, *range(0, 10)):
        if idx not in ordered:
            ordered.append(idx)
    return ordered


class Camera:
    """Minimal read()/release() API shared by live_*.py scripts."""

    def __init__(self, width: int = 640, height: int = 480, index: int | None = None):
        self.width = width
        self.height = height
        # CAMERA_INDEX always wins so callers that hardcode index=0 still work on Ventuno.
        if (os.environ.get("CAMERA_INDEX") or "").strip():
            self.index = _camera_index_from_env(0)
        elif index is None:
            self.index = 0
        else:
            self.index = index
        self.backend = "none"
        self._picam: Any = None
        self._cap: cv2.VideoCapture | None = None
        self._open()

    def _open(self) -> None:
        prefer = _want_backend()
        on_pi = _is_raspberry_pi()
        # Face Access uses a USB webcam — OpenCV is the default unless forced.
        if prefer is None:
            prefer = "opencv"

        errors: list[str] = []

        if prefer == "picamera2":
            try:
                self._open_picamera2()
                return
            except Exception as exc:  # noqa: BLE001 — try OpenCV next if not forced-only
                errors.append(f"picamera2: {exc}")
                # On Pi, Picamera2 missing from venv is the usual failure mode.
                hint = ""
                if on_pi and isinstance(exc, ImportError):
                    hint = (
                        "\nPicamera2 not visible in this venv. Recreate with:\n"
                        "  deactivate\n"
                        "  rm -rf faceenv\n"
                        "  sudo apt install -y python3-picamera2\n"
                        "  python3 -m venv --system-site-packages faceenv\n"
                        "  source faceenv/bin/activate\n"
                        "  pip install 'setuptools<81' opencv-python numpy "
                        "moorcheh-edge ultralytics torchvision pillow\n"
                    )
                # Fall through to OpenCV only when not on Pi (or user may have USB cam).
                if on_pi and _want_backend() == "picamera2":
                    raise RuntimeError(
                        "CAMERA_BACKEND=picamera2 but CSI camera failed.\n"
                        + "\n".join(errors)
                        + hint
                    ) from exc
                if on_pi:
                    # Still try OpenCV (USB webcam on Pi), but keep the hint.
                    errors.append(hint.strip())

        if prefer in ("opencv", "picamera2"):
            try:
                self._open_opencv()
                return
            except Exception as exc:  # noqa: BLE001
                errors.append(f"opencv: {exc}")

        raise RuntimeError(
            "Could not open a camera.\n"
            "Pi Camera Module 3: need Picamera2 "
            "(sudo apt install -y python3-picamera2) and a venv with "
            "--system-site-packages.\n"
            "Mac/USB: CAMERA_BACKEND=opencv (default on Mac); grant camera access.\n"
            + "\n".join(e for e in errors if e)
        )

    def _open_picamera2(self) -> None:
        from picamera2 import Picamera2  # type: ignore

        cam = Picamera2()
        config = cam.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"}
        )
        cam.configure(config)
        cam.start()
        # Warm up one frame so the first read is less likely empty.
        cam.capture_array("main")
        self._picam = cam
        self.backend = "picamera2"

    def _opencv_backends(self) -> list[int | None]:
        """V4L2 on Linux/Ventuno only — CAP_V4L2 fails on Windows and breaks local testing."""
        if sys.platform.startswith("linux"):
            return [cv2.CAP_V4L2, None]
        if sys.platform == "darwin":
            return [cv2.CAP_AVFOUNDATION, None]
        # Windows: DirectShow / Media Foundation via default, then DSHOW
        return [None, cv2.CAP_DSHOW]

    def _open_opencv(self) -> None:
        # On Qualcomm/Ventuno, /dev/video0 is often a platform/CSI node. Opening it
        # yields "camera index out of range" / UVW stream errors. Prefer V4L2 on
        # Linux and probe indices until a real capture device returns a frame.
        # On Windows/macOS never force V4L2.
        errors: list[str] = []
        for idx in _opencv_index_candidates(self.index):
            for api in self._opencv_backends():
                cap = None
                try:
                    cap = (
                        cv2.VideoCapture(idx)
                        if api is None
                        else cv2.VideoCapture(idx, api)
                    )
                    if not cap.isOpened():
                        cap.release()
                        errors.append(f"index {idx} api={api}: open failed")
                        continue
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        cap.release()
                        errors.append(
                            f"index {idx} api={api}: opened but first frame read failed"
                        )
                        continue
                    self.index = idx
                    self._cap = cap
                    self.backend = "opencv"
                    return
                except Exception as exc:  # noqa: BLE001
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:  # noqa: BLE001
                            pass
                    errors.append(f"index {idx} api={api}: {exc}")

        raise RuntimeError(
            "No working OpenCV camera found. Plug in a webcam and set CAMERA_INDEX.\n"
            "  Windows: CAMERA_BACKEND=opencv CAMERA_INDEX=0\n"
            "  Ventuno: CAMERA_BACKEND=opencv CAMERA_INDEX=2\n"
            "Tried:\n  " + "\n  ".join(errors)
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._picam is not None:
            rgb = self._picam.capture_array("main")
            if rgb is None or rgb.size == 0:
                return False, None
            return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if self._cap is not None:
            return self._cap.read()
        return False, None

    def release(self) -> None:
        if self._picam is not None:
            try:
                self._picam.stop()
            except Exception:  # noqa: BLE001
                pass
            self._picam = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

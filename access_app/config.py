"""Runtime config for the Face Access web app."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KNOWN_FACES_DIR = ROOT / "known_faces"
KNOWN_OBJECTS_DIR = ROOT / "known_objects"

BASE_URL = os.environ.get("MOORCHEH_BASE_URL", "http://localhost:8083").rstrip("/")
OBJECTS_URL = os.environ.get("MOORCHEH_OBJECTS_URL", "http://localhost:8081").rstrip("/")
HOST = os.environ.get("FACE_ACCESS_HOST", "0.0.0.0")
PORT = int(os.environ.get("FACE_ACCESS_PORT", "8501"))

# Identity match threshold for ACCESS GRANTED
MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", "0.28"))
DETECT_SCORE_THRESHOLD = float(os.environ.get("FACE_DETECT_THRESHOLD", "0.5"))
SEARCH_TOP_K = int(os.environ.get("FACE_SEARCH_TOP_K", "3"))
# Object match: YOLO crops need a solid Edge hit; center-crop fallback is stricter.
# Defaults aligned with Moorcheh-Face to cut left/right false names.
OBJECT_MATCH_THRESHOLD = float(os.environ.get("OBJECT_MATCH_THRESHOLD", "0.38"))
OBJECT_FALLBACK_THRESHOLD = float(os.environ.get("OBJECT_FALLBACK_THRESHOLD", "0.55"))
OBJECT_MARGIN = float(os.environ.get("OBJECT_MARGIN", "0.03"))
OBJECT_SEARCH_TOP_K = int(os.environ.get("OBJECT_SEARCH_TOP_K", "3"))

# Face / object cadence — same N so boxes update on a similar rhythm.
FACE_EVERY_N_FRAMES = int(os.environ.get("FACE_EVERY_N_FRAMES", "3"))
FACE_SCALE = float(os.environ.get("FACE_SCALE", "0.35"))
OBJECT_PUSH_EVERY_N = int(os.environ.get("OBJECT_PUSH_EVERY_N", "3"))
OBJECT_SCAN_SLEEP_S = float(os.environ.get("OBJECT_SCAN_SLEEP_S", "0.20"))
# Object box proposer: auto (LiteRT .tflite if present, else Torch .pt), litert, or torch.
_OBJECT_DETECTOR_RAW = (os.environ.get("OBJECT_DETECTOR") or "auto").strip().lower()
OBJECT_DETECTOR = (
    _OBJECT_DETECTOR_RAW
    if _OBJECT_DETECTOR_RAW in {"auto", "litert", "torch", "pt", "tflite"}
    else "auto"
)
if OBJECT_DETECTOR == "pt":
    OBJECT_DETECTOR = "torch"
if OBJECT_DETECTOR == "tflite":
    OBJECT_DETECTOR = "litert"
LITERT_YOLO_MODEL = (os.environ.get("LITERT_YOLO_MODEL") or "").strip() or None
TORCH_YOLO_MODEL = (os.environ.get("TORCH_YOLO_MODEL") or "yolov8n.pt").strip() or "yolov8n.pt"
OBJECT_CONF = float(os.environ.get("OBJECT_CONF", "0.25"))
OBJECT_IMGSZ = int(os.environ.get("OBJECT_IMGSZ", "256"))

# Spoken line: "template" (default) or "llm" via an external chat LLM (facts in the prompt).
_GUIDANCE_MODE_RAW = (os.environ.get("GUIDANCE_MODE") or "template").strip().lower()
GUIDANCE_MODE = "llm" if _GUIDANCE_MODE_RAW in {"llm", "external", "ollama", "openai"} else "template"
GUIDANCE_TEMPERATURE = float(os.environ.get("GUIDANCE_TEMPERATURE", "0.2"))
GUIDANCE_HEADER = (os.environ.get("GUIDANCE_HEADER") or "").strip() or None
# External LLM — Ollama by default; set LLM_API=openai for OpenAI-compatible /v1/chat/completions.
_LLM_API_RAW = (os.environ.get("LLM_API") or "ollama").strip().lower()
LLM_API = "openai" if _LLM_API_RAW in {"openai", "v1", "compatible"} else "ollama"
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
LLM_MODEL = (os.environ.get("LLM_MODEL") or "qwen2.5:0.5b-instruct").strip()
LLM_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "20"))
# Standalone Piper (optional). If unset, try common Ventuno paths, else espeak / SAPI.
def _resolve_piper_defaults() -> tuple[str, str | None]:
    home = Path.home()
    candidates = [
        (home / "piper" / "piper" / "piper", home / "piper" / "piper" / "en_US-lessac-medium.onnx"),
        (home / "piper" / "piper", home / "piper" / "en_US-lessac-medium.onnx"),
        (ROOT / "piper" / "piper", ROOT / "piper" / "en_US-lessac-medium.onnx"),
    ]
    env_bin = (os.environ.get("PIPER_BIN") or "").strip()
    env_model = (os.environ.get("PIPER_MODEL") or "").strip()
    if env_bin and env_model and Path(env_model).is_file():
        return env_bin, env_model
    if env_model and Path(env_model).is_file():
        return (env_bin or "piper"), env_model
    for binary, model in candidates:
        if binary.is_file() and model.is_file():
            return str(binary), str(model)
    return (env_bin or "piper"), (env_model or None)


PIPER_BIN, PIPER_MODEL = _resolve_piper_defaults()
if PIPER_MODEL:
    os.environ.setdefault("PIPER_BIN", PIPER_BIN)
    os.environ.setdefault("PIPER_MODEL", PIPER_MODEL)
PIPER_SPEAKER = (os.environ.get("PIPER_SPEAKER") or "").strip() or None

CAMERA_WIDTH = int(os.environ.get("CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.environ.get("CAMERA_HEIGHT", "480"))
# USB webcam via OpenCV only (not the board CSI / Picamera2 path).
os.environ.setdefault("CAMERA_BACKEND", "opencv")
CAMERA_BACKEND = os.environ.get("CAMERA_BACKEND", "opencv").strip().lower() or "opencv"
# Ventuno USB webcam is usually index 0. Override with CAMERA_INDEX if needed.
CAMERA_INDEX_RAW = (os.environ.get("CAMERA_INDEX") or "0").strip()
CAMERA_INDEX = int(CAMERA_INDEX_RAW) if CAMERA_INDEX_RAW else 0

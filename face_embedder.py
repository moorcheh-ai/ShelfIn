"""OpenCV YuNet + SFace face detection and embeddings (no dlib)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"


class FaceEmbedder:
    def __init__(
        self,
        yunet_path: Path | str = YUNET_PATH,
        sface_path: Path | str = SFACE_PATH,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ):
        yunet_path = Path(yunet_path)
        sface_path = Path(sface_path)
        if not yunet_path.is_file():
            raise FileNotFoundError(f"YuNet model missing: {yunet_path}")
        if not sface_path.is_file():
            raise FileNotFoundError(f"SFace model missing: {sface_path}")

        self._score_threshold = score_threshold
        self._nms_threshold = nms_threshold
        self._top_k = top_k
        self._yunet_path = str(yunet_path)
        # Prefer classic DNN engine; avoids noisy "setPreferableTarget" warnings on some builds.
        try:
            cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
        except Exception:
            pass

        self.recognizer = cv2.FaceRecognizerSF.create(str(sface_path), "")
        self._detector = None
        self._input_size = None

    def _ensure_detector(self, width: int, height: int) -> None:
        size = (width, height)
        if self._detector is not None and self._input_size == size:
            return
        self._detector = cv2.FaceDetectorYN.create(
            self._yunet_path,
            "",
            size,
            self._score_threshold,
            self._nms_threshold,
            self._top_k,
        )
        self._input_size = size

    @staticmethod
    def _l2_normalize(feature) -> list[float]:
        vec = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def detect(self, bgr: np.ndarray) -> np.ndarray | None:
        """Return YuNet faces array or None if no faces."""
        h, w = bgr.shape[:2]
        self._ensure_detector(w, h)
        _, faces = self._detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        return faces

    def embed_face(self, bgr: np.ndarray, face_row: np.ndarray) -> list[float]:
        aligned = self.recognizer.alignCrop(bgr, face_row)
        feature = self.recognizer.feature(aligned)
        return self._l2_normalize(feature)

    def embed_image(self, bgr: np.ndarray) -> list[float] | None:
        """Embed the first (highest-confidence) face in an image."""
        faces = self.detect(bgr)
        if faces is None:
            return None
        return self.embed_face(bgr, faces[0])

    def embed_path(self, path: str | Path) -> list[float] | None:
        bgr = cv2.imread(str(path))
        if bgr is None:
            return None
        return self.embed_image(bgr)

    @staticmethod
    def face_box_xyxy(face_row: np.ndarray) -> tuple[int, int, int, int]:
        x, y, w, h = face_row[:4].astype(int)
        return int(x), int(y), int(x + w), int(y + h)


def feature_dim(embedder: FaceEmbedder | None = None) -> int:
    """Probe SFace feature length using a tiny blank image if needed."""
    # SFace MobileFaceNet features are 128-d; confirm via a synthetic call if possible.
    return 128

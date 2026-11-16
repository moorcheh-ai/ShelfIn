"""Background camera + Face → Edge search loop for the access UI."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from moorcheh_edge import MoorchehEdgeApiClient, MoorchehEdgeApiError

from camera_capture import Camera
from face_embedder import FaceEmbedder

from access_app.config import (
    BASE_URL,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    DETECT_SCORE_THRESHOLD,
    FACE_EVERY_N_FRAMES,
    FACE_SCALE,
    MATCH_THRESHOLD,
    OBJECT_FALLBACK_THRESHOLD,
    OBJECT_MATCH_THRESHOLD,
    OBJECT_PUSH_EVERY_N,
    SEARCH_TOP_K,
)
from access_app.guidance import build_guidance
from access_app.object_watch import ObjectWatch
from retail_guide import RetailVoice


@dataclass
class AccessState:
    authorized: bool = False
    label: str = "Unknown"
    score: float | None = None
    faces: int = 0
    message: str = "Looking for a face…"
    camera_ok: bool = False
    camera_backend: str = "none"
    camera_index: int | None = None
    edge_ok: bool = False
    error: str | None = None
    guide: str = ""
    updated_at: float = field(default_factory=time.time)


class RecognitionLoop:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = AccessState()
        self._frame_jpeg: bytes | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._embedder: FaceEmbedder | None = None
        self._client: MoorchehEdgeApiClient | None = None
        self._camera: Camera | None = None
        self._objects = ObjectWatch(
            match_threshold=OBJECT_MATCH_THRESHOLD,
            fallback_threshold=OBJECT_FALLBACK_THRESHOLD,
        )
        self._voice = RetailVoice()
        self._last_guide = ""
        self._frame_i = 0
        # (left, top, right, bottom, label, score|None, authorized)
        self._last_face_hits: list[tuple[int, int, int, int, str, float | None, bool]] = []
        self._last_best_label = "Unknown"
        self._last_best_score: float | None = None
        self._last_authorized = False
        self._last_face_count = 0
        self._last_object_hits: list[dict[str, Any]] = []
        self._last_object_gen = -1

    @property
    def threshold(self) -> float:
        return MATCH_THRESHOLD

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._objects.start()
        self._thread = threading.Thread(target=self._run, name="face-access", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._objects.stop()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._voice.close()
        if self._camera:
            self._camera.release()
            self._camera = None

    def snapshot(self) -> AccessState:
        with self._lock:
            return AccessState(**self._state.__dict__)

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._frame_jpeg

    def _set_state(self, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self._state, key, value)
            self._state.updated_at = time.time()

    def _set_frame(self, bgr: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        with self._lock:
            self._frame_jpeg = buf.tobytes()

    def _ensure_clients(self) -> None:
        if self._embedder is None:
            self._embedder = FaceEmbedder(score_threshold=DETECT_SCORE_THRESHOLD)
        if self._client is None:
            self._client = MoorchehEdgeApiClient(BASE_URL)
            self._client.health()
            self._set_state(edge_ok=True, error=None)

    def _ensure_camera(self) -> None:
        if self._camera is not None:
            return
        kwargs: dict[str, Any] = {
            "width": CAMERA_WIDTH,
            "height": CAMERA_HEIGHT,
        }
        if CAMERA_INDEX is not None:
            kwargs["index"] = CAMERA_INDEX
        self._camera = Camera(**kwargs)
        self._set_state(
            camera_ok=True,
            camera_backend=self._camera.backend,
            camera_index=self._camera.index,
            error=None,
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._ensure_clients()
                self._ensure_camera()
                assert self._camera is not None
                assert self._embedder is not None
                assert self._client is not None

                ok, frame = self._camera.read()
                if not ok or frame is None:
                    self._set_state(
                        authorized=False,
                        label="Unknown",
                        score=None,
                        faces=0,
                        message="Camera frame unavailable",
                        camera_ok=False,
                    )
                    time.sleep(0.2)
                    continue

                self._frame_i += 1
                if self._frame_i % OBJECT_PUSH_EVERY_N == 0:
                    self._objects.push(frame.copy())

                display = frame.copy()
                inv_scale = 1.0 / FACE_SCALE if FACE_SCALE > 0 else 1.0

                # Face search every N frames (reuse boxes between ticks).
                if self._frame_i % FACE_EVERY_N_FRAMES == 0:
                    small = cv2.resize(frame, (0, 0), fx=FACE_SCALE, fy=FACE_SCALE)
                    faces = self._embedder.detect(small)
                    face_hits: list[
                        tuple[int, int, int, int, str, float | None, bool]
                    ] = []
                    best_label = "Unknown"
                    best_score: float | None = None
                    authorized = False
                    face_count = 0

                    if faces is not None:
                        face_count = len(faces)
                        for face_row in faces:
                            emb = self._embedder.embed_face(small, face_row)
                            result = self._client.search(
                                {
                                    "query": emb,
                                    "top_k": SEARCH_TOP_K,
                                }
                            )
                            label = "Unknown"
                            score = None
                            matches = result.get("results", [])
                            if matches:
                                best = matches[0]
                                score = float(best.get("score", 0.0) or 0.0)
                                if score >= MATCH_THRESHOLD:
                                    label = (
                                        best.get("text")
                                        or best.get("label")
                                        or "Unknown"
                                    )

                            left, top, right, bottom = FaceEmbedder.face_box_xyxy(
                                face_row
                            )
                            left = int(left * inv_scale)
                            top = int(top * inv_scale)
                            right = int(right * inv_scale)
                            bottom = int(bottom * inv_scale)
                            is_auth = label != "Unknown"
                            face_hits.append(
                                (left, top, right, bottom, label, score, is_auth)
                            )
                            if is_auth and (
                                best_score is None or (score or 0) > best_score
                            ):
                                best_label = label
                                best_score = score
                                authorized = True
                            elif best_score is None and score is not None:
                                best_label = label
                                best_score = score

                    self._last_face_hits = face_hits
                    self._last_best_label = best_label
                    self._last_best_score = best_score
                    self._last_authorized = authorized
                    self._last_face_count = face_count

                best_label = self._last_best_label
                best_score = self._last_best_score
                authorized = self._last_authorized
                face_count = self._last_face_count

                for left, top, right, bottom, label, score, is_auth in self._last_face_hits:
                    color = (40, 180, 80) if is_auth else (40, 40, 220)
                    cv2.rectangle(display, (left, top), (right, bottom), color, 2)
                    text = label if score is None else f"{label} {score:.3f}"
                    cv2.putText(
                        display,
                        text,
                        (left, max(top - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )

                # Sticky object boxes between scans (same idea as face hits).
                fresh, gen = self._objects.snapshot()
                if gen != self._last_object_gen:
                    self._last_object_hits = fresh
                    self._last_object_gen = gen
                hits = self._last_object_hits
                for hit in hits:
                    x1, y1, x2, y2 = hit["box"]
                    matched = bool(hit.get("matched", True))
                    if matched:
                        caption = hit["name"]
                        if hit.get("score") is not None:
                            caption = f"{caption} {float(hit['score']):.3f}"
                        if hit.get("shelf"):
                            caption = f"{caption} · {hit['shelf']}"
                        color = (40, 180, 80)
                    else:
                        score = hit.get("score")
                        caption = (
                            f"Unknown {float(score):.3f}"
                            if score is not None
                            else "Unknown"
                        )
                        color = (40, 40, 220)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        display,
                        caption,
                        (x1, max(y1 - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )

                matched_hits = [h for h in hits if h.get("matched", True)]
                guide = ""
                if matched_hits:
                    guide = build_guidance(
                        [
                            (
                                hit["name"],
                                hit.get("shelf") or "",
                                hit.get("description") or "",
                            )
                            for hit in matched_hits
                        ],
                        best_label if authorized else None,
                    )
                    if guide and guide != self._last_guide:
                        self._voice.say(guide)
                        self._last_guide = guide
                elif self._last_guide:
                    self._last_guide = ""

                if face_count == 0:
                    message = guide or "No face detected — access locked"
                    banner = (50, 50, 50)
                    banner_text = "WAITING"
                elif authorized:
                    message = guide or f"Authorized — welcome, {best_label}"
                    banner = (40, 140, 70)
                    banner_text = "ACCESS GRANTED"
                else:
                    message = guide or "Unknown face — access denied"
                    banner = (40, 40, 180)
                    banner_text = "ACCESS DENIED"

                cv2.rectangle(display, (0, 0), (display.shape[1], 42), banner, -1)
                cv2.putText(
                    display,
                    banner_text,
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                self._set_frame(display)
                self._set_state(
                    authorized=authorized,
                    label=best_label,
                    score=best_score,
                    faces=face_count,
                    message=message,
                    guide=guide,
                    camera_ok=True,
                    edge_ok=True,
                    error=None,
                )
            except Exception as exc:  # noqa: BLE001
                # Camera failures are RuntimeError — do NOT mark Edge down for those.
                is_edge = isinstance(exc, MoorchehEdgeApiError) or (
                    "moorcheh" in str(exc).lower() and "edge" in str(exc).lower()
                )
                is_camera = (not is_edge) and (
                    "camera" in str(exc).lower()
                    or "videocapture" in str(exc).lower()
                    or "v4l2" in str(exc).lower()
                )
                self._set_state(
                    authorized=False,
                    label="Unknown",
                    score=None,
                    faces=0,
                    message="Camera error" if is_camera else (
                        "Edge error" if is_edge else "Recognition error"
                    ),
                    error=str(exc).split("\n")[0],
                    camera_ok=False if is_camera else self.snapshot().camera_ok,
                    edge_ok=False if is_edge else True,
                )
                blank = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
                tip = "Camera error — check USB webcam" if is_camera else (
                    "Edge error — is Moorcheh Edge on :8083?"
                    if is_edge
                    else "See status message"
                )
                cv2.putText(
                    blank,
                    tip,
                    (24, CAMERA_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 200),
                    2,
                )
                self._set_frame(blank)
                time.sleep(1.0)
                if self._camera is not None and is_camera:
                    try:
                        self._camera.release()
                    except Exception:  # noqa: BLE001
                        pass
                    self._camera = None
                if is_edge:
                    self._client = None

            time.sleep(0.03)

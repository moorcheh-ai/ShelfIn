"""Slow object pass aligned with Moorcheh-Face live_face_and_objects rules.

YOLO proposes boxes (LiteRT .tflite preferred, Torch .pt fallback).
Naming comes only from Moorcheh objects Edge search.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from core.retail_guide import load_catalog

# Same skip set spirit as Moorcheh-Face (person / vehicles / furniture).
_SKIP_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "chair",
    "couch",
    "bed",
    "dining table",
    "toilet",
    "tv",
}

MAX_CROP_AREA_FRAC = 0.45
ROOT = Path(__file__).resolve().parent.parent


class ObjectWatch:
    def __init__(
        self,
        match_threshold: float = 0.35,
        fallback_threshold: float = 0.45,
    ) -> None:
        self._match_threshold = match_threshold
        self._fallback_threshold = fallback_threshold
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._hits: list[dict[str, Any]] = []
        self._generation = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._yolo: Any = None
        self._yolo_backend: str | None = None
        self._embedder: Any = None
        self._client: Any = None
        self._catalog: dict = {}
        self._catalog_at = 0.0
        self._last_error: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="object-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def push(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame

    def hits(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._hits)

    def snapshot(self) -> tuple[list[dict[str, Any]], int]:
        with self._lock:
            return list(self._hits), self._generation

    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def _run(self) -> None:
        time.sleep(1.5)
        while not self._stop.is_set():
            with self._lock:
                frame = None if self._frame is None else self._frame.copy()
            if frame is None:
                time.sleep(0.2)
                continue
            try:
                found = self._scan(frame)
                with self._lock:
                    self._hits = found
                    self._generation += 1
                    self._last_error = None
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).split("\n")[0]
                print(f"[object-watch] {msg}")
                with self._lock:
                    self._hits = []
                    self._generation += 1
                    self._last_error = msg
            from access_app.config import OBJECT_SCAN_SLEEP_S

            time.sleep(max(0.05, OBJECT_SCAN_SLEEP_S))

    def _catalog_now(self) -> dict:
        now = time.time()
        if now - self._catalog_at > 2.0:
            self._catalog = load_catalog()
            self._catalog_at = now
        return self._catalog

    def _litert_candidates(self) -> list[Path]:
        from access_app.config import LITERT_YOLO_MODEL

        paths: list[Path] = []
        if LITERT_YOLO_MODEL:
            paths.append(Path(LITERT_YOLO_MODEL).expanduser())
        paths.extend(
            [
                ROOT / "yolov8n.tflite",
                ROOT / "yolo11n.tflite",
                ROOT / "yolo26n.tflite",
            ]
        )
        return paths

    def _torch_candidates(self) -> list[Path | str]:
        from access_app.config import TORCH_YOLO_MODEL

        raw = TORCH_YOLO_MODEL
        path = Path(raw).expanduser()
        if path.is_file():
            return [path]
        local = ROOT / raw
        if local.is_file():
            return [local]
        # Ultralytics will download yolov8n.pt when given the bare name.
        return [raw]

    def _load_yolo(self, model_ref: Path | str, backend: str) -> Any:
        from ultralytics import YOLO

        label = str(model_ref)
        print(f"[object-watch] loading YOLO ({backend}): {label}")
        model = YOLO(label)
        self._yolo = model
        self._yolo_backend = backend
        print(f"[object-watch] YOLO ready ({backend})")
        return model

    def _ensure_yolo(self) -> None:
        if self._yolo is not None:
            return

        from access_app.config import OBJECT_DETECTOR

        mode = OBJECT_DETECTOR
        litert_paths = [p for p in self._litert_candidates() if Path(p).is_file()]
        errors: list[str] = []

        if mode in {"auto", "litert"}:
            if not litert_paths and mode == "litert":
                raise FileNotFoundError(
                    "OBJECT_DETECTOR=litert but no .tflite found. "
                    "Run: python export_yolo_litert.py on x86/Mac, copy yolov8n.tflite here, "
                    "or set LITERT_YOLO_MODEL=/path/to/model.tflite"
                )
            for path in litert_paths:
                try:
                    self._load_yolo(path, "litert")
                    return
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"litert {path}: {exc}")
                    print(f"[object-watch] LiteRT load failed ({path}): {exc}")
            if mode == "litert":
                raise RuntimeError(
                    "LiteRT YOLO failed to load. "
                    + ("; ".join(errors) if errors else "no candidates")
                )

        if mode in {"auto", "torch"}:
            for ref in self._torch_candidates():
                try:
                    self._load_yolo(ref, "torch")
                    if mode == "auto" and litert_paths:
                        print(
                            "[object-watch] note: LiteRT was preferred but failed; "
                            "using Torch .pt fallback"
                        )
                    elif mode == "auto" and not litert_paths:
                        print(
                            "[object-watch] no yolov8n.tflite found — using Torch .pt "
                            "(run python export_yolo_litert.py on x86/Mac to enable LiteRT)"
                        )
                    return
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"torch {ref}: {exc}")
                    print(f"[object-watch] Torch load failed ({ref}): {exc}")

        raise RuntimeError(
            "Could not load any YOLO detector. "
            + ("; ".join(errors) if errors else "no candidates")
        )

    def _ensure_embed(self) -> bool:
        try:
            if self._embedder is None:
                from core.object_embedder import ObjectEmbedder

                print("[object-watch] loading object embedder …")
                self._embedder = ObjectEmbedder()
            if self._client is None:
                from moorcheh_edge import MoorchehEdgeApiClient

                from access_app.config import OBJECTS_URL

                client = MoorchehEdgeApiClient(OBJECTS_URL)
                client.health()
                self._client = client
                print(f"[object-watch] objects Edge OK ({OBJECTS_URL})")
        except Exception as exc:  # noqa: BLE001
            print(f"[object-watch] embed/edge not ready: {exc}")
            return False
        return self._embedder is not None and self._client is not None

    def _search_crop(
        self,
        crop: np.ndarray,
        threshold: float,
    ) -> tuple[str, float]:
        assert self._embedder is not None
        assert self._client is not None
        emb = self._embedder.embed_image(crop)
        if emb is None:
            return "Unknown", 0.0
        try:
            from access_app.config import OBJECT_MARGIN as MARGIN
            from access_app.config import OBJECT_SEARCH_TOP_K

            result = self._client.search(
                {"query": emb, "top_k": OBJECT_SEARCH_TOP_K}
            )
        except Exception as exc:  # noqa: BLE001
            self._client = None
            raise RuntimeError(f"objects search failed: {exc}") from exc
        matches = result.get("results", [])
        if not matches:
            return "Unknown", 0.0
        best = matches[0]
        score = float(best.get("score", 0.0) or 0.0)
        name = best.get("text") or best.get("label") or "Unknown"
        if score < threshold:
            return "Unknown", score
        # Compare against the next of the top-3 so weak near-ties stay Unknown.
        if len(matches) > 1:
            second = float(matches[1].get("score", 0.0) or 0.0)
            if score - second < MARGIN:
                return "Unknown", score
        return str(name), score

    def _scan(self, frame: np.ndarray) -> list[dict[str, Any]]:
        if not self._ensure_embed():
            return []
        self._ensure_yolo()
        assert self._yolo is not None

        catalog = self._catalog_now()
        h, w = frame.shape[:2]
        frame_area = float(h * w)
        # (x1, y1, x2, y2, proposal_name)
        crops: list[tuple[int, int, int, int, str]] = []

        from access_app.config import OBJECT_CONF, OBJECT_IMGSZ

        results = self._yolo.predict(
            frame,
            conf=OBJECT_CONF,
            imgsz=OBJECT_IMGSZ,
            verbose=False,
        )
        if results:
            r0 = results[0]
            names = r0.names
            if r0.boxes is not None:
                for box in r0.boxes:
                    cls_id = int(box.cls[0])
                    cname = str(names.get(cls_id, cls_id)).lower()
                    if cname in _SKIP_CLASSES:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    bw, bh = x2 - x1, y2 - y1
                    if bw < 20 or bh < 20:
                        continue
                    if (bw * bh) / frame_area > MAX_CROP_AREA_FRAC:
                        continue
                    crops.append((x1, y1, x2, y2, cname))

        used_fallback = False
        if not crops:
            side = min(h, w) // 3
            cx, cy = w // 2, h // 2
            crops.append(
                (
                    max(0, cx - side // 2),
                    max(0, cy - side // 2),
                    min(w - 1, cx + side // 2),
                    min(h - 1, cy + side // 2),
                    "center",
                )
            )
            used_fallback = True

        thr = self._fallback_threshold if used_fallback else self._match_threshold
        found: list[dict[str, Any]] = []
        seen: set[str] = set()
        for x1, y1, x2, y2, proposal in crops:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            label, score = self._search_crop(crop, threshold=thr)
            matched = label != "Unknown"
            # Fallback center crop: only keep a real enrolled match (same as Moorcheh-Face).
            if used_fallback and not matched:
                continue
            if matched:
                key = label.lower()
                if key in seen:
                    continue
                seen.add(key)
                spec = catalog.get(key, {})
                hit = {
                    "name": str(spec.get("say") or label),
                    "shelf": str(spec.get("shelf") or ""),
                    "description": str(spec.get("description") or ""),
                    "score": score,
                    "matched": True,
                    "box": (x1, y1, x2, y2),
                }
                print(f"[object-watch] match {hit['name']} score={score:.3f}")
            else:
                hit = {
                    "name": "Unknown",
                    "shelf": "",
                    "description": "",
                    "score": score,
                    "matched": False,
                    "proposal": proposal,
                    "box": (x1, y1, x2, y2),
                }
            found.append(hit)
            if sum(1 for h in found if h.get("matched")) >= 2:
                break
        return found

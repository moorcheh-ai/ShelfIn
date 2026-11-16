"""Retail guide: face recognition + object detection + spoken shelf guidance.

Says which item is in view and which shelf it belongs on. A known face is
greeted in the same line. Shelf notes come from known_objects/<Name>/shelf.txt
or retail_catalog.json.

  ./start_faces_edge.sh
  ./start_objects_edge.sh
  python enroll_faces.py
  python enroll_objects.py
  python live_retail_guide.py

RETAIL_VOICE=0 prints the line and does not speak.
"""

from __future__ import annotations

import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
from moorcheh_edge import MoorchehEdgeApiClient, MoorchehEdgeApiError
from ultralytics import YOLO

from camera_capture import Camera
from face_embedder import FaceEmbedder
from object_embedder import ObjectEmbedder

CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
FACE_SCALE = 0.35
FACE_EVERY_N_FRAMES = 3
FACE_THRESHOLD = 0.28
OBJECT_EVERY_N_FRAMES = 4
OBJECT_CONF = 0.25
OBJECT_IMGSZ = 256
OBJECT_THRESHOLD = 0.38
OBJECT_MARGIN = 0.03
OBJECT_FALLBACK_THRESHOLD = 0.55
MAX_CROP_AREA_FRAC = 0.45
SKIP_YOLO_CLASSES = {
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
from retail_guide import RetailVoice, guidance_line, load_catalog, lookup, shelf_phrase

FACES_URL = os.environ.get("MOORCHEH_BASE_URL", "http://localhost:8083")
OBJECTS_URL = os.environ.get("MOORCHEH_OBJECTS_URL", "http://localhost:8081")
GUIDE_PORT = int(os.environ.get("RETAIL_GUIDE_PORT", "8502"))

PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Retail guide</title>
<style>
body { margin: 0; background: #111; color: #fff; font-family: sans-serif; }
img { width: 100%; max-height: 82vh; object-fit: contain; background: #000; }
#line { padding: 16px 20px; font-size: 1.5rem; }
</style>
</head>
<body>
<img src="/stream" alt="camera">
<div id="line">Starting camera</div>
<script>
async function tick() {
  const res = await fetch("/line");
  document.getElementById("line").textContent = await res.text();
}
setInterval(tick, 400);
tick();
</script>
</body>
</html>
""".encode()


class _View:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jpeg = b""
        self.banner = "Point the camera at an item"


_view = _View()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            body = PAGE
            content_type = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/line":
            with _view.lock:
                body = _view.banner.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    with _view.lock:
                        jpeg = _view.jpeg
                    if jpeg:
                        chunk = (
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                            + jpeg
                            + b"\r\n"
                        )
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
            return
        self.send_error(404)


def _serve() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", GUIDE_PORT), _Handler)
    server.serve_forever()


def _connect(url: str, label: str) -> MoorchehEdgeApiClient:
    try:
        client = MoorchehEdgeApiClient(url)
        client.health()
        return client
    except MoorchehEdgeApiError as exc:
        raise RuntimeError(
            f"Could not reach {label} Edge at {url}. "
            "Start faces with ./start_faces_edge.sh and objects with "
            "./start_objects_edge.sh"
        ) from exc


def search_object(client, embedder, crop, threshold: float) -> tuple[str, float]:
    emb = embedder.embed_bgr(crop)
    result = client.search({"query": emb, "top_k": 3})
    matches = result.get("results", [])
    if not matches:
        return "Unknown", 0.0
    best = matches[0]
    score = float(best.get("score", 0.0) or 0.0)
    name = str(best.get("text") or best.get("label") or "Unknown")
    if score < threshold:
        return "Unknown", score
    if len(matches) > 1:
        second = float(matches[1].get("score", 0.0) or 0.0)
        if score - second < OBJECT_MARGIN:
            return "Unknown", score
    return name, score


def main() -> None:
    faces_client = _connect(FACES_URL, "faces")
    objects_client = _connect(OBJECTS_URL, "objects")
    catalog = load_catalog()
    print(f"Retail catalog: {len(catalog)} item(s)")
    for key, info in sorted(catalog.items()):
        shelf = info.get("shelf") or "(no shelf yet)"
        print(f"  {info.get('say') or key} -> {shelf}")

    face_embedder = FaceEmbedder(score_threshold=0.5)
    object_embedder = ObjectEmbedder()
    yolo = YOLO("yolov8n.pt")
    cap = Camera(CAPTURE_WIDTH, CAPTURE_HEIGHT)
    voice = RetailVoice()
    threading.Thread(target=_serve, daemon=True).start()
    print(f"Camera backend: {cap.backend}")
    print(f"Open http://10.0.0.30:{GUIDE_PORT}/")
    print(f"On the board: http://127.0.0.1:{GUIDE_PORT}/")

    frame_i = 0
    last_objects: list[tuple[int, int, int, int, str, str]] = []
    last_faces: list[tuple[int, int, int, int, str]] = []
    banner = "Point the camera at an item"
    last_said = ""

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            frame_i += 1
            height, width = frame.shape[:2]
            frame_area = float(height * width)

            if frame_i % OBJECT_EVERY_N_FRAMES == 0:
                last_objects = []
                results = yolo.predict(
                    frame, conf=OBJECT_CONF, imgsz=OBJECT_IMGSZ, verbose=False
                )
                crops: list[tuple[int, int, int, int, str]] = []
                if results and results[0].boxes is not None:
                    names = results[0].names
                    for box in results[0].boxes:
                        class_name = names.get(int(box.cls[0]), "")
                        if class_name in SKIP_YOLO_CLASSES:
                            continue
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width - 1, x2), min(height - 1, y2)
                        box_w, box_h = x2 - x1, y2 - y1
                        if box_w < 20 or box_h < 20:
                            continue
                        if (box_w * box_h) / frame_area > MAX_CROP_AREA_FRAC:
                            continue
                        crops.append((x1, y1, x2, y2, class_name))

                used_fallback = False
                if not crops:
                    side = min(height, width) // 3
                    cx, cy = width // 2, height // 2
                    crops.append(
                        (cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2, "center")
                    )
                    used_fallback = True

                threshold = (
                    OBJECT_FALLBACK_THRESHOLD if used_fallback else OBJECT_THRESHOLD
                )
                for x1, y1, x2, y2, proposal in crops:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    label, _score = search_object(
                        objects_client, object_embedder, crop, threshold
                    )
                    if label == "Unknown":
                        if used_fallback:
                            continue
                        info = lookup(catalog, proposal)
                        if info is None:
                            continue
                        spoken = info["say"] or proposal
                        shelf = shelf_phrase(info.get("shelf", ""))
                    else:
                        info = lookup(catalog, label) or {}
                        spoken = info.get("say") or label
                        shelf = shelf_phrase(info.get("shelf", ""))
                    last_objects.append((x1, y1, x2, y2, spoken, shelf))

            if frame_i % FACE_EVERY_N_FRAMES == 0:
                last_faces = []
                small = cv2.resize(frame, (0, 0), fx=FACE_SCALE, fy=FACE_SCALE)
                inv = 1.0 / FACE_SCALE
                faces = face_embedder.detect(small)
                if faces is not None:
                    for face_row in faces:
                        emb = face_embedder.embed_face(small, face_row)
                        result = faces_client.search({"query": emb, "top_k": 1})
                        label = "Unknown"
                        matches = result.get("results", [])
                        if matches:
                            best = matches[0]
                            score = float(best.get("score", 0.0) or 0.0)
                            if score >= FACE_THRESHOLD:
                                label = str(
                                    best.get("text") or best.get("label") or "Unknown"
                                )
                        left, top, right, bottom = FaceEmbedder.face_box_xyxy(face_row)
                        last_faces.append(
                            (
                                max(0, min(width - 1, int(left * inv))),
                                max(0, min(height - 1, int(top * inv))),
                                max(0, min(width - 1, int(right * inv))),
                                max(0, min(height - 1, int(bottom * inv))),
                                label,
                            )
                        )

            person = next(
                (label for *_, label in last_faces if label != "Unknown"),
                None,
            )
            guided = [(name, shelf) for *_, name, shelf in last_objects]
            line = guidance_line(guided, person)
            if line and line != last_said:
                banner = line
                last_said = line
                voice.say(line)
            elif not line:
                banner = "Point the camera at an item"
                last_said = ""

            for x1, y1, x2, y2, name, shelf in last_objects:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)
                caption = f"{name} | {shelf}" if shelf else name
                cv2.putText(
                    frame,
                    caption,
                    (x1, max(y1 - 12, 28)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 180, 0),
                    2,
                )
            for left, top, right, bottom, label in last_faces:
                color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (left, max(top - 12, 28)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

            cv2.rectangle(frame, (0, height - 64), (width, height), (20, 24, 28), -1)
            cv2.putText(
                frame,
                banner[:90],
                (12, height - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            ok_jpg, encoded = cv2.imencode(".jpg", frame)
            if ok_jpg:
                with _view.lock:
                    _view.jpeg = encoded.tobytes()
                    _view.banner = banner
    finally:
        voice.close()
        cap.release()


if __name__ == "__main__":
    main()

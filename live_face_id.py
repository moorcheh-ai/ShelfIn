import os

import cv2
from moorcheh_edge import MoorchehEdgeApiClient, MoorchehEdgeApiError

from camera_capture import Camera
from face_embedder import FaceEmbedder

BASE_URL = os.environ.get("MOORCHEH_BASE_URL", "http://localhost:8083")

try:
    client = MoorchehEdgeApiClient(BASE_URL)
    client.health()
except MoorchehEdgeApiError as exc:
    raise RuntimeError(
        f"Could not reach Moorcheh Edge at {BASE_URL}. "
        "Start it with: moorcheh-edge up"
    ) from exc

# Slightly lower detect threshold helps webcam faces.
embedder = FaceEmbedder(score_threshold=0.5)

# Ventuno USB webcam is often /dev/video2 (not 0). Override with CAMERA_INDEX=2.
cap = Camera(640, 480)
print(f"Camera backend: {cap.backend} index={cap.index}")

# Client-side match cutoff on Moorcheh score (0–1).
# Same-person still photos often land ~0.45–1.0; live webcam is lower.
THRESHOLD = 0.28

while True:
    ok, frame = cap.read()
    if not ok:
        break

    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    faces = embedder.detect(small)

    if faces is not None:
        for face_row in faces:
            emb = embedder.embed_face(small, face_row)

            # Do not use kiosk_mode here — it can hide all hits below threshold
            # and leave you with Unknown and no score to tune against.
            result = client.search({
                "query": emb,
                "top_k": 3,
            })

            label = "Unknown"
            score = None

            matches = result.get("results", [])
            if matches:
                best = matches[0]
                score = float(best.get("score", 0.0) or 0.0)
                if score >= THRESHOLD:
                    label = best.get("text") or best.get("label") or "Unknown"

            left, top, right, bottom = FaceEmbedder.face_box_xyxy(face_row)
            left *= 2
            top *= 2
            right *= 2
            bottom *= 2

            color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            if score is None:
                text = label
            else:
                text = f"{label} {score:.3f}"
            cv2.putText(
                frame, text, (left, max(top - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

    cv2.imshow("Moorcheh Face ID (SFace)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

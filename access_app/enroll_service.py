"""Save onboarding photos and upload embeddings to Moorcheh Edge."""

from __future__ import annotations

import re
import time
from pathlib import Path

import cv2
import numpy as np
from moorcheh_edge import MoorchehEdgeApiClient, MoorchehEdgeApiError

from core.face_embedder import FaceEmbedder

from access_app.config import BASE_URL, FACE_STORE, KNOWN_FACES_DIR

_SAFE_NAME = re.compile(r"[^A-Za-z0-9 _.-]+")


def sanitize_person_name(name: str) -> str:
    cleaned = _SAFE_NAME.sub("", (name or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ._-" )
    if not cleaned:
        raise ValueError("Name is required")
    if len(cleaned) > 64:
        cleaned = cleaned[:64]
    return cleaned


def list_known_people() -> list[dict]:
    KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
    people: list[dict] = []
    for person_dir in sorted(KNOWN_FACES_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        photos = sorted(
            p
            for p in person_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        people.append(
            {
                "name": person_dir.name,
                "photo_count": len(photos),
                "photos": [p.name for p in photos],
            }
        )
    return people


def _edge_client() -> MoorchehEdgeApiClient:
    client = MoorchehEdgeApiClient(BASE_URL, store=FACE_STORE)
    try:
        client.health()
    except MoorchehEdgeApiError as exc:
        raise RuntimeError(
            f"Could not reach Moorcheh Edge at {BASE_URL}. "
            "Start it with: ./start_faces_edge.sh  or  "
            "moorcheh-edge up --skip-ollama --server-port 8083 -y"
        ) from exc
    return client


def enroll_person_images(
    name: str,
    files: list[tuple[str, bytes]],
    embedder: FaceEmbedder | None = None,
) -> dict:
    """
    Save photos under known_faces/<name>/ and upload vectors to Edge.

    files: list of (original_filename, raw_bytes)
    """
    person = sanitize_person_name(name)
    if not files:
        raise ValueError("Upload at least one photo")

    person_dir = KNOWN_FACES_DIR / person
    person_dir.mkdir(parents=True, exist_ok=True)

    embedder = embedder or FaceEmbedder()
    client = _edge_client()
    items: list[dict] = []
    saved: list[str] = []
    skipped: list[str] = []

    stamp = int(time.time())
    for i, (fname, raw) in enumerate(files, start=1):
        ext = Path(fname).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            ext = ".jpg"
        out_name = f"{stamp}_{i}{ext}"
        out_path = person_dir / out_name

        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            skipped.append(f"{fname} (could not decode)")
            continue

        ok, encoded = cv2.imencode(ext if ext != ".webp" else ".jpg", bgr)
        if not ok:
            skipped.append(f"{fname} (encode failed)")
            continue
        if ext == ".webp":
            out_name = f"{stamp}_{i}.jpg"
            out_path = person_dir / out_name
        out_path.write_bytes(encoded.tobytes())
        saved.append(out_name)

        emb = embedder.embed_image(bgr)
        if emb is None:
            skipped.append(f"{out_name} (no face found)")
            continue

        items.append(
            {
                "id": f"{person.lower()}/{out_name}",
                "vector": emb,
                "text": person,
            }
        )

    upload_result = None
    if items:
        upload_result = client.upload(
            {
                "store_mode": "vector",
                "items": items,
            }
        )

    return {
        "name": person,
        "saved": saved,
        "enrolled": len(items),
        "skipped": skipped,
        "upload": upload_result,
        "people": list_known_people(),
    }

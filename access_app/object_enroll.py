"""Save item photos and upload embeddings to the objects Edge."""

from __future__ import annotations

import re
import time
from pathlib import Path

import cv2
import numpy as np
from moorcheh_edge import MoorchehEdgeApiClient, MoorchehEdgeApiError

from access_app.config import KNOWN_OBJECTS_DIR, OBJECTS_URL
from retail_guide import load_catalog, save_catalog_entry

_SAFE = re.compile(r"[^A-Za-z0-9 _.-]+")
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def sanitize_label(value: str, field: str) -> str:
    cleaned = _SAFE.sub("", (value or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ._-" )
    if not cleaned:
        raise ValueError(f"{field} is required")
    if len(cleaned) > 64:
        cleaned = cleaned[:64]
    return cleaned


def list_known_items() -> list[dict]:
    KNOWN_OBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog()
    items: list[dict] = []
    for item_dir in sorted(KNOWN_OBJECTS_DIR.iterdir()):
        if not item_dir.is_dir() or item_dir.name.startswith("."):
            continue
        photos = sorted(
            p for p in item_dir.iterdir() if p.suffix.lower() in _IMAGE_EXT
        )
        shelf_file = item_dir / "shelf.txt"
        shelf = ""
        if shelf_file.is_file():
            shelf = shelf_file.read_text(encoding="utf-8").splitlines()[0].strip()
        desc_file = item_dir / "description.txt"
        description = ""
        if desc_file.is_file():
            description = desc_file.read_text(encoding="utf-8").splitlines()[0].strip()
        spec = catalog.get(item_dir.name.lower(), {})
        items.append(
            {
                "name": item_dir.name,
                "shelf": shelf or str(spec.get("shelf") or ""),
                "description": description or str(spec.get("description") or ""),
                "photo_count": len(photos),
            }
        )
    return items


def enroll_item_images(
    name: str,
    shelf: str,
    files: list[tuple[str, bytes]],
    description: str = "",
) -> dict:
    item = sanitize_label(name, "Item name")
    place = sanitize_label(shelf, "Shelf")
    note = (description or "").strip()
    if note:
        note = sanitize_label(note, "Description")
    if not files:
        raise ValueError("Upload at least one photo")

    item_dir = KNOWN_OBJECTS_DIR / item
    item_dir.mkdir(parents=True, exist_ok=True)
    (item_dir / "shelf.txt").write_text(place + "\n", encoding="utf-8")
    if note:
        (item_dir / "description.txt").write_text(note + "\n", encoding="utf-8")
    save_catalog_entry(item, place, note)

    try:
        from object_embedder import ObjectEmbedder
    except ImportError as exc:
        raise RuntimeError(
            "Object embedding is not installed in this environment. "
            "From the repo folder: pip install -r requirements-retail.txt"
        ) from exc

    client = MoorchehEdgeApiClient(OBJECTS_URL)
    try:
        client.health()
    except MoorchehEdgeApiError as exc:
        raise RuntimeError(
            f"Could not reach the objects Edge at {OBJECTS_URL}. "
            "Start it with: ./start_objects_edge.sh"
        ) from exc

    embedder = ObjectEmbedder()
    items: list[dict] = []
    saved: list[str] = []
    skipped: list[str] = []
    stamp = int(time.time())

    for i, (fname, raw) in enumerate(files, start=1):
        ext = Path(fname).suffix.lower()
        if ext not in _IMAGE_EXT:
            ext = ".jpg"
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            skipped.append(f"{fname} (could not decode)")
            continue
        out_name = f"{stamp}_{i}.jpg"
        out_path = item_dir / out_name
        ok, encoded = cv2.imencode(".jpg", bgr)
        if not ok:
            skipped.append(f"{fname} (encode failed)")
            continue
        out_path.write_bytes(encoded.tobytes())
        saved.append(out_name)
        emb = embedder.embed_image(bgr)
        if emb is None:
            skipped.append(f"{out_name} (no embedding)")
            continue
        items.append(
            {
                "id": f"{item.lower()}/{out_name}",
                "vector": emb,
                "text": item,
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
        "name": item,
        "shelf": place,
        "description": note,
        "saved": saved,
        "enrolled": len(items),
        "skipped": skipped,
        "upload": upload_result,
        "items": list_known_items(),
    }

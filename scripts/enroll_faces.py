"""Bulk-enroll faces from known_faces/ into Moorcheh Edge (:8083).

Run from repo root:
  python scripts/enroll_faces.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moorcheh_edge import MoorchehEdgeApiClient, MoorchehEdgeApiError

from core.face_embedder import FaceEmbedder

BASE_URL = os.environ.get("MOORCHEH_BASE_URL", "http://localhost:8083")
STORE = (os.environ.get("MOORCHEH_FACE_STORE") or "faces").strip() or "faces"

client = MoorchehEdgeApiClient(BASE_URL, store=STORE)

try:
    health = client.health()
except MoorchehEdgeApiError as exc:
    raise RuntimeError(
        f"Could not reach Moorcheh Edge at {BASE_URL}. "
        "Start it with: bash ./start_faces_edge.sh"
    ) from exc

print(
    f"Moorcheh Edge: {health.get('items', 0)} items, "
    f"store_mode={health.get('store_mode')}, dimension={health.get('dimension')}"
)

embedder = FaceEmbedder()

items = []
root = ROOT / "known_faces"

for person_name in os.listdir(root):
    person_dir = root / person_name
    if not person_dir.is_dir():
        continue

    for fname in os.listdir(person_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = person_dir / fname
        emb = embedder.embed_path(str(path))
        if emb is None:
            print(f"Skipping {path} (no face found)")
            continue

        items.append({
            "id": f"{person_name.lower()}/{fname}",
            "vector": emb,
            "text": person_name,
        })
        print(f"Enrolled {path} (dim={len(emb)})")

if not items:
    raise RuntimeError("No face embeddings found to upload")

result = client.upload({
    "store_mode": "vector",
    "items": items,
})
print(result)

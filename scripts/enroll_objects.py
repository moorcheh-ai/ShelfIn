"""Bulk-enroll objects from known_objects/ into Moorcheh Edge (:8081).

Run from repo root:
  python scripts/enroll_objects.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moorcheh_edge import MoorchehEdgeApiClient, MoorchehEdgeApiError

from core.object_embedder import EMBED_DIM, ObjectEmbedder
from core.retail_guide import save_catalog_entry

BASE_URL = os.environ.get("MOORCHEH_OBJECTS_URL", "http://localhost:8081")

client = MoorchehEdgeApiClient(BASE_URL)

try:
    health = client.health()
except MoorchehEdgeApiError as exc:
    raise RuntimeError(
        f"Could not reach objects Edge at {BASE_URL}. "
        "Start it with: bash ./start_objects_edge.sh"
    ) from exc

print(
    f"Objects Edge: {health.get('items', 0)} items, "
    f"store_mode={health.get('store_mode')}, dimension={health.get('dimension')}"
)
if health.get("dimension") not in (None, EMBED_DIM):
    raise RuntimeError(
        f"Objects store dimension is {health.get('dimension')}, expected {EMBED_DIM}. "
        "Clear it first: moorcheh-edge clear-store -y --base-url http://localhost:8081"
    )

embedder = ObjectEmbedder()
items = []
root = ROOT / "known_objects"

for object_name in sorted(os.listdir(root)):
    object_dir = root / object_name
    if not object_dir.is_dir():
        continue

    shelf_note = object_dir / "shelf.txt"
    shelf = ""
    if shelf_note.is_file():
        with open(shelf_note, encoding="utf-8") as handle:
            shelf = handle.readline().strip()
    desc_note = object_dir / "description.txt"
    description = ""
    if desc_note.is_file():
        with open(desc_note, encoding="utf-8") as handle:
            description = handle.readline().strip()
    if shelf or description:
        save_catalog_entry(object_name, shelf, description)

    for fname in os.listdir(object_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = object_dir / fname
        emb = embedder.embed_path(str(path))
        if emb is None:
            print(f"Skipping {path} (could not read image)")
            continue

        items.append(
            {
                "id": f"{object_name.lower()}/{fname}",
                "vector": emb,
                "text": object_name,
            }
        )
        extra = []
        if shelf:
            extra.append(f"shelf={shelf}")
        if description:
            extra.append(f"desc={description}")
        where = f" ({', '.join(extra)})" if extra else ""
        print(f"Enrolled {path} (dim={len(emb)}{where})")

if not items:
    raise RuntimeError("No object embeddings found to upload")

result = client.upload(
    {
        "store_mode": "vector",
        "items": items,
    }
)
print(result)

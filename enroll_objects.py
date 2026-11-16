"""Enroll known objects into Moorcheh Edge objects store (port 8081)."""

import os

from moorcheh_edge import MoorchehEdgeApiClient, MoorchehEdgeApiError

from object_embedder import EMBED_DIM, ObjectEmbedder
from retail_guide import save_catalog_entry

BASE_URL = os.environ.get("MOORCHEH_OBJECTS_URL", "http://localhost:8081")

client = MoorchehEdgeApiClient(BASE_URL)

try:
    health = client.health()
except MoorchehEdgeApiError as exc:
    raise RuntimeError(
        f"Could not reach objects Edge at {BASE_URL}. "
        "Start it with: ./start_objects_edge.sh"
    ) from exc

print(
    f"Objects Edge: {health.get('items', 0)} items, "
    f"store_mode={health.get('store_mode')}, dimension={health.get('dimension')}"
)
if health.get("dimension") not in (None, EMBED_DIM):
    raise RuntimeError(
        f"Objects store dimension is {health.get('dimension')}, expected {EMBED_DIM}. "
        "Clear it first: curl -X DELETE http://localhost:8081/store"
    )

embedder = ObjectEmbedder()
items = []
root = "known_objects"

for object_name in sorted(os.listdir(root)):
    object_dir = os.path.join(root, object_name)
    if not os.path.isdir(object_dir):
        continue

    shelf_note = os.path.join(object_dir, "shelf.txt")
    shelf = ""
    if os.path.isfile(shelf_note):
        with open(shelf_note, encoding="utf-8") as handle:
            shelf = handle.readline().strip()
    desc_note = os.path.join(object_dir, "description.txt")
    description = ""
    if os.path.isfile(desc_note):
        with open(desc_note, encoding="utf-8") as handle:
            description = handle.readline().strip()
    if shelf or description:
        save_catalog_entry(object_name, shelf, description)

    for fname in os.listdir(object_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(object_dir, fname)
        emb = embedder.embed_path(path)
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

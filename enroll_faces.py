import os

from moorcheh_edge import MoorchehEdgeApiClient, MoorchehEdgeApiError

from face_embedder import FaceEmbedder

BASE_URL = os.environ.get("MOORCHEH_BASE_URL", "http://localhost:8083")

client = MoorchehEdgeApiClient(BASE_URL)

try:
    health = client.health()
except MoorchehEdgeApiError as exc:
    raise RuntimeError(
        f"Could not reach Moorcheh Edge at {BASE_URL}. "
        "Start it with: moorcheh-edge up"
    ) from exc

print(
    f"Moorcheh Edge: {health.get('items', 0)} items, "
    f"store_mode={health.get('store_mode')}, dimension={health.get('dimension')}"
)

embedder = FaceEmbedder()

items = []
root = "known_faces"

for person_name in os.listdir(root):
    person_dir = os.path.join(root, person_name)
    if not os.path.isdir(person_dir):
        continue

    for fname in os.listdir(person_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(person_dir, fname)
        emb = embedder.embed_path(path)
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

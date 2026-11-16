"""Local face gallery: NumPy arrays + cosine nearest-neighbor (no Moorcheh)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_GALLERY_PATH = Path(__file__).resolve().parent / "data" / "faces_gallery.npz"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def save_gallery(
    path: Path | str,
    vectors: list[list[float]],
    names: list[str],
    ids: list[str],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(vectors, dtype=np.float32)
    np.savez_compressed(
        path,
        vectors=arr,
        names=np.asarray(names, dtype=object),
        ids=np.asarray(ids, dtype=object),
    )
    return path


def load_gallery(path: Path | str = DEFAULT_GALLERY_PATH) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Gallery not found: {path}. Run enroll_faces_traditional.py first."
        )
    data = np.load(path, allow_pickle=True)
    return {
        "vectors": np.asarray(data["vectors"], dtype=np.float32),
        "names": [str(x) for x in data["names"].tolist()],
        "ids": [str(x) for x in data["ids"].tolist()],
        "path": path,
    }


def match_cosine(
    query: list[float] | np.ndarray,
    gallery_vectors: np.ndarray,
    gallery_names: list[str],
    threshold: float,
) -> tuple[str, float]:
    """
    Brute-force 1:N search with an explicit for-loop + cosine similarity.

    Returns (label, best_score). Label is "Unknown" if best_score < threshold.
    """
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    best_score = -1.0
    best_name = "Unknown"

    for i in range(len(gallery_vectors)):
        score = cosine_similarity(query, gallery_vectors[i])
        if score > best_score:
            best_score = score
            best_name = gallery_names[i]

    if best_score < threshold:
        return "Unknown", best_score
    return best_name, best_score

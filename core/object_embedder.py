"""Image embeddings for enrolled objects (ResNet18 → 256-d, L2-normalized).

Moorcheh Edge allowlist: 128, 256, 384, 768, 1024, 1536 (not 512).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

EMBED_DIM = 256


class ObjectEmbedder:
    """General image embedder for object gallery matching (not face identity)."""

    def __init__(self):
        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)
        backbone.fc = nn.Identity()
        backbone.eval()
        self.model = backbone
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.preprocess = weights.transforms()

        # Deterministic 512 → 256 projection (Edge does not allow 512-d).
        rng = np.random.default_rng(42)
        proj = rng.standard_normal((512, EMBED_DIM)).astype(np.float32)
        proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8
        self.proj = proj

    @torch.inference_mode()
    def embed_bgr(self, bgr: np.ndarray) -> list[float]:
        if bgr is None or bgr.size == 0:
            raise ValueError("empty image")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
        feat = self.model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
        reduced = feat @ self.proj
        return self._l2_normalize(reduced)

    def embed_image(self, bgr: np.ndarray) -> list[float] | None:
        """Same as embed_bgr; returns None on empty input (UI / watch callers)."""
        try:
            return self.embed_bgr(bgr)
        except ValueError:
            return None

    def embed_path(self, path: str | Path) -> list[float] | None:
        bgr = cv2.imread(str(path))
        if bgr is None:
            return None
        return self.embed_bgr(bgr)

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> list[float]:
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

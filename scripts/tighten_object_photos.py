"""Crop known_objects so the item fills almost the whole image."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "known_objects"
SKIP = {
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
OUT_SIZE = 512


def fill_square(crop: np.ndarray, size: int = OUT_SIZE) -> np.ndarray:
    """Scale crop so the longer side fills `size` (object covers the frame)."""
    h, w = crop.shape[:2]
    if h < 1 or w < 1:
        return crop
    scale = size / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    # Match edge color roughly so letterbox is less noisy for ResNet.
    canvas[:] = np.median(resized.reshape(-1, 3), axis=0).astype(np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def main() -> None:
    try:
        from ultralytics import YOLO

        yolo = YOLO("yolov8n.pt")
        use_yolo = True
        print("YOLO loaded")
    except Exception as exc:  # noqa: BLE001
        print("YOLO unavailable, using center crop:", exc)
        yolo = None
        use_yolo = False

    def tight_crop(bgr: np.ndarray):
        h, w = bgr.shape[:2]
        if use_yolo:
            result = yolo.predict(bgr, conf=0.15, imgsz=640, verbose=False)[0]
            best = None
            best_area = 0
            if result.boxes is not None:
                for box in result.boxes:
                    cname = str(result.names.get(int(box.cls[0]), "")).lower()
                    if cname in SKIP:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    area = max(0, x2 - x1) * max(0, y2 - y1)
                    if area > best_area:
                        best_area = area
                        best = (x1, y1, x2, y2)
            if best is not None:
                x1, y1, x2, y2 = best
                bw, bh = max(1, x2 - x1), max(1, y2 - y1)
                # Tiny pad only — object should dominate the frame.
                pad = int(0.04 * max(bw, bh))
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w - 1, x2 + pad)
                y2 = min(h - 1, y2 + pad)
                return bgr[y1:y2, x1:x2], "yolo"
        # Aggressive center crop (~92% of min side).
        side = int(min(h, w) * 0.92)
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)
        return bgr[y1:y2, x1:x2], "center"

    count = 0
    for folder in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        imgs = [
            *sorted(folder.glob("*.jpg")),
            *sorted(folder.glob("*.jpeg")),
            *sorted(folder.glob("*.png")),
        ]
        staged: list[tuple[Path, str]] = []
        for i, path in enumerate(imgs, start=1):
            bgr = cv2.imread(str(path))
            if bgr is None:
                print("skip unreadable", path)
                continue
            crop, how = tight_crop(bgr)
            if crop is None or crop.size == 0:
                print("skip empty", path)
                continue
            filled = fill_square(crop)
            tmp = folder / f"__tight_{i}.jpg"
            cv2.imwrite(str(tmp), filled, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
            staged.append((tmp, how))
            if path.resolve() != tmp.resolve():
                path.unlink(missing_ok=True)
            print(
                f"{folder.name}: staged {tmp.name} ({how}) "
                f"{filled.shape[1]}x{filled.shape[0]}"
            )

        for i, (tmp, how) in enumerate(staged, start=1):
            out = folder / f"{folder.name.lower()} {i}.jpg"
            if out.exists() and out.resolve() != tmp.resolve():
                out.unlink(missing_ok=True)
            tmp.rename(out)
            print(f"{folder.name}: -> {out.name} ({how})")
            count += 1

    print("tightened", count, "images")


if __name__ == "__main__":
    main()

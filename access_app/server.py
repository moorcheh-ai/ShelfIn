"""Visual Memory web app - live access + shelf guidance + onboarding."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from access_app.config import BASE_URL, HOST, MATCH_THRESHOLD, PORT
from access_app.enroll_service import enroll_person_images, list_known_people
from access_app.object_enroll import enroll_item_images, list_known_items
from access_app.recognition_loop import RecognitionLoop

STATIC_DIR = Path(__file__).resolve().parent / "static"
loop = RecognitionLoop()


def _edge_reachable() -> bool:
    """Live check of Moorcheh Edge — independent of camera loop state."""
    try:
        from urllib.request import urlopen

        with urlopen(f"{BASE_URL}/health", timeout=2.0) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except Exception:  # noqa: BLE001
        return False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    loop.start()
    yield
    loop.stop()


app = FastAPI(title="Visual Memory", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict:
    state = loop.snapshot()
    edge_ok = _edge_reachable()
    return {
        "ok": True,
        "edge_url": BASE_URL,
        "match_threshold": MATCH_THRESHOLD,
        "camera_ok": state.camera_ok,
        "camera_backend": state.camera_backend,
        "camera_index": state.camera_index,
        "edge_ok": edge_ok,
        "error": state.error,
    }


@app.get("/api/status")
def status() -> dict:
    state = loop.snapshot()
    edge_ok = _edge_reachable()
    return {
        "authorized": state.authorized,
        "label": state.label,
        "score": state.score,
        "faces": state.faces,
        "message": state.message,
        "guide": state.guide,
        "threshold": MATCH_THRESHOLD,
        "camera_ok": state.camera_ok,
        "edge_ok": edge_ok,
        "error": None if edge_ok and state.camera_ok else state.error,
        "updated_at": state.updated_at,
    }


@app.get("/api/stream")
def stream() -> StreamingResponse:
    boundary = "frame"

    def gen():
        import time

        while True:
            jpeg = loop.latest_jpeg()
            if jpeg:
                yield (
                    b"--" + boundary.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            time.sleep(0.05)

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
    )


@app.get("/api/faces")
def faces() -> dict:
    return {"people": list_known_people(), "threshold": MATCH_THRESHOLD}


@app.post("/api/faces/enroll")
async def faces_enroll(
    name: str = Form(...),
    photos: list[UploadFile] = File(...),
) -> JSONResponse:
    if not photos:
        raise HTTPException(status_code=400, detail="Upload at least one photo")

    files: list[tuple[str, bytes]] = []
    for photo in photos:
        raw = await photo.read()
        if not raw:
            continue
        files.append((photo.filename or "photo.jpg", raw))

    try:
        result = enroll_person_images(name, files)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if result["enrolled"] == 0:
        skipped = "; ".join(result["skipped"][:5]) or "no faces detected"
        raise HTTPException(
            status_code=400,
            detail=f"No faces could be enrolled ({skipped})",
        )

    return JSONResponse(result)


@app.get("/api/objects")
def objects() -> dict:
    return {"items": list_known_items()}


@app.post("/api/objects/enroll")
async def objects_enroll(
    name: str = Form(...),
    shelf: str = Form(...),
    photos: list[UploadFile] = File(...),
    description: str = Form(""),
) -> JSONResponse:
    if not photos:
        raise HTTPException(status_code=400, detail="Upload at least one photo")

    files: list[tuple[str, bytes]] = []
    for photo in photos:
        raw = await photo.read()
        if not raw:
            continue
        files.append((photo.filename or "photo.jpg", raw))

    try:
        result = enroll_item_images(name, shelf, files, description=description)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if result["enrolled"] == 0:
        skipped = "; ".join(result["skipped"][:5]) or "no photo could be embedded"
        raise HTTPException(
            status_code=400,
            detail=f"No item photos could be enrolled ({skipped})",
        )

    return JSONResponse(result)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "access_app.server:app",
        host=HOST,
        port=PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()

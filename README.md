# ShelfIn

Access control and shelf guidance on **Ventuno Q** (USB webcam + speaker + browser UI).

- **YuNet / SFace** → faces Edge `:8083` → ACCESS GRANTED / DENIED
- **YOLO / ResNet** → objects Edge `:8081` → known item + shelf line
- Optional external LLM for spoken rewrite; **Piper / espeak** on the speaker
- Web UI on `:8501`

Product story: **[PROJECT_HUB.md](./PROJECT_HUB.md)**

## Layout

```text
shelfin/
  access_app/     # FastAPI dashboard + enroll APIs
  core/           # camera, face/object embed, TTS helpers
  scripts/        # bulk enroll, model download, LiteRT export
  known_faces/    # local face photos (gitignored)
  known_objects/  # local item photos (gitignored)
  models/         # YuNet / SFace ONNX (gitignored weights)
  docs/           # figures + intro slide
  start.sh        # UI
  start_faces_edge.sh
  start_objects_edge.sh
```

## One-time setup

```bash
cd ~
git clone https://github.com/moorcheh-ai/ShelfIn.git
cd ShelfIn

sudo apt update && sudo apt install -y docker.io
sudo usermod -aG docker "$USER"   # then log out/in
sudo systemctl enable --now docker

python3 -m venv faceenv
source faceenv/bin/activate
pip install -U pip
pip install -r requirements-retail.txt

bash ./scripts/download_models.sh
# place yolov8n.pt in project root
```

Optional Piper (auto-detected): `~/piper/piper/piper` + `en_US-lessac-medium.onnx`

## Every boot

```bash
cd ~/ShelfIn   # or ~/shelfin
source faceenv/bin/activate
bash ./start_faces_edge.sh
bash ./start_objects_edge.sh
bash ./start.sh
```

Open **http://127.0.0.1:8501/** or **http://\<board-ip\>:8501/**

Defaults (no exports needed): OpenCV camera index `0`, object thresholds `0.38` / `0.55` / `0.03`, `XDG_RUNTIME_DIR`, Piper when installed.

## Enroll

**UI:** person photos → Insert & authorize; item name / shelf / description / photos → Add item.

**Bulk:**

```bash
python scripts/enroll_faces.py
python scripts/enroll_objects.py
```

## Optional

```bash
# LiteRT YOLO (export on Mac/x86 only)
python scripts/export_yolo_litert.py

# External LLM guidance
export GUIDANCE_MODE=llm
bash ./start.sh

# Windows local UI
.\scripts\run_access_app.ps1
```

## Ports

| Port | Service |
|------|---------|
| 8083 | Faces Edge |
| 8081 | Objects Edge |
| 8501 | ShelfIn UI |

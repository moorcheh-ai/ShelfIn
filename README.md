# Visual Memory

**Visual Memory** remembers what the camera sees on **Ventuno Q**: known people for access, known products for shelf guidance - one camera, dual Moorcheh Edge stores, spoken lines on the speaker.

- **YuNet / SFace** → faces Edge `:8083` → ACCESS GRANTED / DENIED
- **YOLO / ResNet** → objects Edge `:8081` → known item + shelf line
- Optional external LLM for spoken rewrite; **Piper / espeak** on the speaker
- Web UI on `:8501`

Product story: **[PROJECT_HUB.md](./PROJECT_HUB.md)**

## Layout

```text
ShelfIn/          # GitHub repo folder (product: Visual Memory)
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

**Disk:** check `df -h` first. On Ventuno Q, `pip install torch` from default PyPI pulls NVIDIA CUDA / triton wheels, fills the disk, and stops halfway — then `start.sh` fails with `No module named 'fastapi'` or `moorcheh_edge`. Install **CPU Torch** from the PyTorch index instead.

If that already happened:

```bash
docker system prune -af
rm -rf ~/.cache/pip ~/ShelfIn/faceenv
```

Then recreate the venv and follow the CPU install below.

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
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics

python -c "import fastapi, moorcheh_edge, torch; print('ok', torch.__version__)"

# Free offline Edge license (required before start_faces_edge.sh / start_objects_edge.sh)
# https://moorcheh.ai/products/edge/license
# Then either:
#   moorcheh-edge license install path/to/license.key
# or:
#   moorcheh-edge up    # paste token when prompted; Ctrl+C after it starts
test -f ~/.moorcheh-edge/license.key

bash ./scripts/download_models.sh
# place yolov8n.pt in project root
```

If a `.sh` script errors with `$'\r'` / `set: -`, fix CRLF then retry: `sed -i 's/\r$//' scripts/*.sh start*.sh`

Optional Piper (auto-detected): `~/piper/piper/piper` + `~/piper/piper/en_US-lessac-medium.onnx`. Otherwise `sudo apt-get install -y espeak-ng`.

## Every boot

```bash
cd ~/ShelfIn
source faceenv/bin/activate
bash ./start_faces_edge.sh
bash ./start_objects_edge.sh
bash ./start.sh
```

Open **http://127.0.0.1:8501/** or **http://\<board-ip\>:8501/**

**License:** Moorcheh Edge needs a free offline license at `~/.moorcheh-edge/license.key`. Get one at [moorcheh.ai/products/edge/license](https://moorcheh.ai/products/edge/license) (one per verified email). Visual Memory uses two containers with Edge store names **`faces`** and **`objects`** (not `moorcheh-faces`). Each store holds up to **1,000** items. Docs: [Licensing](https://docs.moorcheh.ai/on-edge/licensing).

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
| 8501 | Visual Memory UI |

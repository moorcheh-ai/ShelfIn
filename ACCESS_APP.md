# ShelfIn app (Face Access UI)

Web UI for **live gate authorization**, **person onboarding**, and **item enroll** on top of Moorcheh Edge + OpenCV YuNet/SFace + YOLO/ResNet.

## What you get

| Area | Behavior |
|------|----------|
| Live gate | Camera stream; **ACCESS GRANTED** if face score ≥ threshold |
| Person onboard | Name + several photos → **Insert & authorize** → Edge `:8083` |
| Add item | Name, shelf, description + photos → Edge `:8081` |
| Known lists | Enrolled people and items |

Defaults: face match **0.28**, object match **0.38** / fallback **0.55** / margin **0.03**. Camera **opencv** index **0**. Piper auto if under `~/piper/`.

## Fresh setup

Full guide: **[README.md](./README.md)**. Product story: **[PROJECT_HUB.md](./PROJECT_HUB.md)**.

Short version on Ventuno / Linux:

```bash
git clone https://github.com/moorcheh-ai/moorcheh-face-access.git
cd ~/moorcheh-face-access
python3 -m venv faceenv && source faceenv/bin/activate
pip install -r requirements-retail.txt
bash ./scripts/download_models.sh
# place yolov8n.pt in project root

bash ./start_faces_edge.sh
bash ./start_objects_edge.sh
bash ./start.sh
```

Open `http://<board-ip>:8501/`.

Enroll from the UI, or bulk:

```bash
python enroll_faces.py
python enroll_objects.py
```

`start.sh` does **not** start Edge. If faces Edge is down it exits and tells you to run `start_faces_edge.sh`.

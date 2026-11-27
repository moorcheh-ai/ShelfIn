#!/usr/bin/env bash
# Start Visual Memory UI (assumes faces Edge is already up on :8083)
# Usage:
#   bash ./start_faces_edge.sh    # faces store :8083
#   bash ./start_objects_edge.sh  # objects store :8081
#   bash ./start.sh               # Visual Memory UI
set -eu

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
sed -i 's/\r$//' "$ROOT/start.sh" 2>/dev/null || true

# Defaults so day-to-day is: start Edge stores, then bash ./start.sh
export CAMERA_BACKEND="${CAMERA_BACKEND:-opencv}"
export CAMERA_INDEX="${CAMERA_INDEX:-0}"
export MOORCHEH_BASE_URL="${MOORCHEH_BASE_URL:-http://localhost:8083}"
export MOORCHEH_OBJECTS_URL="${MOORCHEH_OBJECTS_URL:-http://localhost:8081}"
export MOORCHEH_FACE_STORE="${MOORCHEH_FACE_STORE:-faces}"
export MOORCHEH_OBJECT_STORE="${MOORCHEH_OBJECT_STORE:-objects}"
# Object thresholds (aligned with Moorcheh-Face)
export OBJECT_MATCH_THRESHOLD="${OBJECT_MATCH_THRESHOLD:-0.38}"
export OBJECT_FALLBACK_THRESHOLD="${OBJECT_FALLBACK_THRESHOLD:-0.55}"
export OBJECT_MARGIN="${OBJECT_MARGIN:-0.03}"
export OBJECT_CONF="${OBJECT_CONF:-0.25}"
export OBJECT_DETECTOR="${OBJECT_DETECTOR:-auto}"
# template (default) or llm — llm calls an external chat model with facts in the prompt
export GUIDANCE_MODE="${GUIDANCE_MODE:-template}"
export LLM_API="${LLM_API:-ollama}"
export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:11434}"
export LLM_MODEL="${LLM_MODEL:-qwen2.5:0.5b-instruct}"
export FACE_ACCESS_HOST="${FACE_ACCESS_HOST:-0.0.0.0}"
export FACE_ACCESS_PORT="${FACE_ACCESS_PORT:-8501}"

# SSH / headless: audio needs a runtime dir + PipeWire/Pulse.
if [[ -z "${XDG_RUNTIME_DIR:-}" && -d "/run/user/$(id -u)" ]]; then
  export XDG_RUNTIME_DIR="/run/user/$(id -u)"
fi
systemctl --user start pipewire pipewire-pulse 2>/dev/null \
  || systemctl --user start pulseaudio 2>/dev/null \
  || true

# Piper: use env if set; else pick a common Ventuno install path when present.
if [[ -z "${PIPER_MODEL:-}" || ! -f "${PIPER_MODEL}" ]]; then
  for pair in \
    "${HOME}/piper/piper/piper|${HOME}/piper/piper/en_US-lessac-medium.onnx" \
    "${HOME}/piper/piper|${HOME}/piper/en_US-lessac-medium.onnx" \
    "${ROOT}/piper/piper|${ROOT}/piper/en_US-lessac-medium.onnx"
  do
    bin="${pair%%|*}"
    model="${pair##*|}"
    if [[ -f "$bin" && -f "$model" ]]; then
      export PIPER_BIN="$bin"
      export PIPER_MODEL="$model"
      break
    fi
  done
fi
if [[ -n "${PIPER_MODEL:-}" && -f "${PIPER_MODEL}" ]]; then
  export PIPER_BIN="${PIPER_BIN:-piper}"
  echo "[start] Piper: ${PIPER_BIN} + ${PIPER_MODEL}"
else
  echo "[start] Piper not found — espeak / print fallback"
fi

if [[ ! -f faceenv/bin/activate ]]; then
  echo "[start] Missing faceenv/. Create once:"
  echo "  python3 -m venv faceenv && source faceenv/bin/activate"
  echo "  pip install -r requirements.txt"
  echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
  echo "  pip install ultralytics"
  echo "  bash ./scripts/download_models.sh"
  exit 1
fi

# shellcheck disable=SC1091
source faceenv/bin/activate

if ! curl -sf "${MOORCHEH_BASE_URL}/health" >/dev/null 2>&1; then
  echo "[start] Moorcheh Edge is not up at ${MOORCHEH_BASE_URL}"
  echo "  Start it first, then re-run this script:"
  echo "    bash ./start_faces_edge.sh"
  echo "  Or with venv + CLI:"
  echo "    source faceenv/bin/activate"
  echo "    moorcheh-edge up --skip-ollama --server-port 8083 -y"
  exit 1
fi

echo "[start] Edge OK: ${MOORCHEH_BASE_URL}"
echo "[start] Guidance mode: ${GUIDANCE_MODE}"
if [[ "${GUIDANCE_MODE}" == "llm" ]]; then
  echo "[start] External LLM: ${LLM_API} ${LLM_BASE_URL} model=${LLM_MODEL}"
  if [[ "${LLM_API}" == "ollama" ]] && ! curl -sf "${LLM_BASE_URL}/api/tags" >/dev/null 2>&1; then
    echo "[start] Warning: Ollama not reachable at ${LLM_BASE_URL}"
    echo "  Start Ollama (or set LLM_BASE_URL), or use GUIDANCE_MODE=template"
  fi
fi

echo "[start] Camera: backend=${CAMERA_BACKEND} index=${CAMERA_INDEX}"
echo "[start] Object thresholds: match=${OBJECT_MATCH_THRESHOLD} fallback=${OBJECT_FALLBACK_THRESHOLD} margin=${OBJECT_MARGIN}"

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "[start] Visual Memory:"
echo "        http://127.0.0.1:${FACE_ACCESS_PORT}/"
if [[ -n "${IP}" ]]; then
  echo "        http://${IP}:${FACE_ACCESS_PORT}/"
fi

exec python -m access_app.server

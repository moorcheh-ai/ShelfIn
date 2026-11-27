#!/usr/bin/env bash
# Start faces Moorcheh Edge on host port 8083.
# Container name matches the objects one: moorcheh-edge-faces / moorcheh-edge-objects.
set -euo pipefail

NAME=moorcheh-edge-faces
LEGACY_NAME=moorcheh-edge
PORT=8083
STORE="${MOORCHEH_FACE_STORE:-faces}"
DATA_DIR="${HOME}/.moorcheh-edge-faces/data"
LICENSE_FILE="${HOME}/.moorcheh-edge/license.key"
IMAGE=moorcheh/moorcheh-edge:latest

if [[ ! -f "$LICENSE_FILE" ]]; then
  echo "[faces-edge] Missing offline license: $LICENSE_FILE"
  echo "  Get a free license: https://moorcheh.ai/products/edge/license"
  echo "  Then: moorcheh-edge license install path/to/license.key"
  echo "  Or run: moorcheh-edge up   # prompts once, then Ctrl+C after it starts"
  exit 1
fi

mkdir -p "$DATA_DIR"
chmod 777 "$DATA_DIR" 2>/dev/null || true

# Backup store file before recreate
if [[ -f "$DATA_DIR/moorcheh_edge_store.json" ]]; then
  cp -a "$DATA_DIR/moorcheh_edge_store.json" \
    "$DATA_DIR/moorcheh_edge_store.json.bak.$(date +%Y%m%d%H%M%S)"
fi

if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  CUR_PORTS=$(docker port "$NAME" 2>/dev/null || true)
  if echo "$CUR_PORTS" | grep -q "8083"; then
    echo "[faces-edge] '$NAME' already on :8083"
    curl -s "http://localhost:${PORT}/health" || true
    echo
    curl -s -X POST "http://localhost:${PORT}/stores" \
      -H "Content-Type: application/json" \
      -d "{\"name\":\"${STORE}\"}" >/dev/null || true
    echo "[faces-edge] Store: ${STORE}"
    exit 0
  fi
fi

# Drop the old generic name if it still holds :8083.
docker rm -f "$LEGACY_NAME" >/dev/null 2>&1 || true

echo "[faces-edge] Recreating '$NAME' on host port ${PORT} (data: $DATA_DIR)"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d \
  --name "$NAME" \
  -p "${PORT}:8080" \
  -v "${DATA_DIR}:/app/data" \
  -v "${LICENSE_FILE}:/app/license.key:ro" \
  "$IMAGE"

sleep 2
echo "[faces-edge] API: http://localhost:${PORT}"
curl -s "http://localhost:${PORT}/health" || true
echo
curl -s -X POST "http://localhost:${PORT}/stores" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"${STORE}\"}" >/dev/null || true
echo "[faces-edge] Store: ${STORE}"

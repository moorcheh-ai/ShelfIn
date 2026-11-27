#!/usr/bin/env bash
# Second Moorcheh Edge container for enrolled objects (faces stay on :8083).
set -euo pipefail

NAME=moorcheh-edge-objects
PORT=8081
STORE="${MOORCHEH_OBJECT_STORE:-objects}"
DATA_DIR="${HOME}/.moorcheh-edge-objects/data"
LICENSE_FILE="${HOME}/.moorcheh-edge/license.key"
IMAGE=moorcheh/moorcheh-edge:latest

if [[ ! -f "$LICENSE_FILE" ]]; then
  echo "[objects-edge] Missing offline license: $LICENSE_FILE"
  echo "  Get a free license: https://moorcheh.ai/products/edge/license"
  echo "  Then: moorcheh-edge license install path/to/license.key"
  echo "  Or run: moorcheh-edge up   # prompts once, then Ctrl+C after it starts"
  exit 1
fi

mkdir -p "$DATA_DIR"
chmod 777 "$DATA_DIR" 2>/dev/null || true

if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "[objects-edge] '$NAME' already running on port $PORT"
else
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  echo "[objects-edge] Starting $NAME on port $PORT …"
  docker run -d \
    --name "$NAME" \
    -p "${PORT}:8080" \
    -v "${DATA_DIR}:/app/data" \
    -v "${LICENSE_FILE}:/app/license.key:ro" \
    "$IMAGE"
fi

echo "[objects-edge] Data: $DATA_DIR"
echo "[objects-edge] API:  http://localhost:${PORT}"
curl -s "http://localhost:${PORT}/health" || true
echo
curl -s -X POST "http://localhost:${PORT}/stores" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"${STORE}\"}" >/dev/null || true
echo "[objects-edge] Store: ${STORE}"

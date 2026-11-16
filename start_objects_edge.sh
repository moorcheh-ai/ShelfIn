#!/usr/bin/env bash
# Second Moorcheh Edge container for enrolled objects (faces stay on :8083).
set -euo pipefail

NAME=moorcheh-edge-objects
PORT=8081
DATA_DIR="${HOME}/.moorcheh-edge-objects/data"
IMAGE=moorcheh/moorcheh-edge:latest

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
    "$IMAGE"
fi

echo "[objects-edge] Data: $DATA_DIR"
echo "[objects-edge] API:  http://localhost:${PORT}"
curl -s "http://localhost:${PORT}/health" || true
echo

#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="acoustics-dev"
WORKDIR="/home/acoustics"
PARAMS_HOST="${1:-params.txt}"
PARAMS_BASENAME="$(basename "$PARAMS_HOST")"
PARAMS_IN_CONTAINER="$WORKDIR/$PARAMS_BASENAME"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "Container not running: $CONTAINER_NAME" >&2
  exit 1
fi

if [[ ! -f "$PARAMS_HOST" ]]; then
  echo "Params file not found on host: $PARAMS_HOST" >&2
  exit 1
fi

# Copy params into container workspace so it can be parsed inside
docker cp "$PARAMS_HOST" "$CONTAINER_NAME:$PARAMS_IN_CONTAINER"

docker exec -i "$CONTAINER_NAME" bash -lc "export PYTHONPATH=/usr/local/dolfinx-complex/lib/python3.12/dist-packages:\$PYTHONPATH; export LD_LIBRARY_PATH=/usr/local/dolfinx-complex/lib:\$LD_LIBRARY_PATH; cd $WORKDIR && bash run_all.sh '$PARAMS_BASENAME'"

#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="acoustics-dev"
WORKDIR="/home/acoustics"
PARAMS_HOST="${1:-params.txt}"
PARAMS_BASENAME="$(basename "$PARAMS_HOST")"
PARAMS_IN_CONTAINER="$WORKDIR/$PARAMS_BASENAME"
PARAMS_DIR_HOST="$(dirname "$PARAMS_HOST")"
RESULTS_SPHERES_IN_CONTAINER="$WORKDIR/results/$(basename "$PARAMS_BASENAME" .txt)/spheres"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "Container not running: $CONTAINER_NAME" >&2
  exit 1
fi

if [[ ! -f "$PARAMS_HOST" ]]; then
  echo "Params file not found on host: $PARAMS_HOST" >&2
  exit 1
fi

docker cp "$PARAMS_HOST" "$CONTAINER_NAME:$PARAMS_IN_CONTAINER"

docker exec -i -w "$WORKDIR" "$CONTAINER_NAME" bash -lc "
  set -e
  source /usr/local/bin/dolfinx-complex-mode
  bash run_all.sh '$PARAMS_BASENAME'
"

# Copy spheres output back to the host next to the params file
docker cp "$CONTAINER_NAME:$RESULTS_SPHERES_IN_CONTAINER" "$PARAMS_DIR_HOST/"
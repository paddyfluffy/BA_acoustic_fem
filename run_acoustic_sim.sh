#!/usr/bin/env bash
set -Eeuo pipefail

CONTAINER_NAME="acoustics-dev"
WORKDIR="/home/acoustics"
DEFAULT_PARAMS_FILE="params.txt"

DIR_WAV_PATTERN="*_hoa_ir_dir.wav"

die() {
  echo "Error: $*" >&2
  exit 1
}

container_running() {
  docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"
}

stop_solver() {
  if container_running; then
    docker exec -i "$CONTAINER_NAME" bash -lc \
      "pkill -f 'mpiexec -n .*run_solver_from_params.py' || true" \
      >/dev/null 2>&1 || true
  fi
}

cleanup() {
  stop_solver
}

main() {

  if [[ "${1:-}" == "--stop" ]]; then
    stop_solver
    exit 0
  fi

  local params_host="${1:-$DEFAULT_PARAMS_FILE}"

  [[ -f "$params_host" ]] || die "Params file not found: $params_host"
  container_running || die "Container not running: $CONTAINER_NAME"

  local params_basename
  params_basename="$(basename "$params_host")"

  local params_dir_host
  params_dir_host="$(cd "$(dirname "$params_host")" && pwd)"

  local params_host_abs="$params_dir_host/$params_basename"
  local params_stem="${params_basename%.txt}"

  local params_in_container="$WORKDIR/$params_basename"
  local spheres_dir_in_container="$WORKDIR/results/$params_stem/spheres"
  local spheres_dir_host="$params_dir_host/spheres"

  trap cleanup INT TERM

  docker cp "$params_host_abs" "$CONTAINER_NAME:$params_in_container"

  docker exec -i -w "$WORKDIR" "$CONTAINER_NAME" bash -lc "
    set -Eeuo pipefail
    source /usr/local/bin/dolfinx-complex-mode
    bash run_all.sh '$params_basename'
  "

  trap - INT TERM

  rm -rf "$spheres_dir_host"
  docker cp "$CONTAINER_NAME:$spheres_dir_in_container" "$params_dir_host/"

  # Return directional HOA IR wav files
  find "$spheres_dir_host" -type f -name "$DIR_WAV_PATTERN" | sort
}

main "$@"
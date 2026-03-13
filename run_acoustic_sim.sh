#!/usr/bin/env bash
set -Eeuo pipefail

CONTAINER_NAME="acoustics-dev"
WORKDIR="/home/acoustics"
DEFAULT_PARAMS_FILE="params.txt"
MIC_IR_FILENAME="${MIC_IR_FILENAME:-hoa_ir_dir.wav}"

die() {
  echo "Error: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
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

abs_path() {
  python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

cleanup() {
  stop_solver
}

main() {
  require_cmd docker
  require_cmd python3

  if [[ "${1:-}" == "--stop" ]]; then
    stop_solver
    exit 0
  fi

  local params_host="${1:-$DEFAULT_PARAMS_FILE}"
  local params_host_abs
  params_host_abs="$(abs_path "$params_host")"

  [[ -f "$params_host_abs" ]] || die "Params file not found on host: $params_host"
  container_running || die "Container not running: $CONTAINER_NAME"

  local params_basename
  params_basename="$(basename "$params_host_abs")"

  local params_stem
  params_stem="${params_basename%.txt}"

  local params_in_container="$WORKDIR/$params_basename"
  local params_dir_host
  params_dir_host="$(dirname "$params_host_abs")"

  local spheres_dir_in_container="$WORKDIR/results/$params_stem/spheres"
  local spheres_dir_host="$params_dir_host/spheres"

  trap cleanup INT TERM EXIT

  docker cp "$params_host_abs" "$CONTAINER_NAME:$params_in_container"

  docker exec -i -w "$WORKDIR" "$CONTAINER_NAME" bash -lc "
    set -Eeuo pipefail
    source /usr/local/bin/dolfinx-complex-mode
    bash run_all.sh '$params_basename'
  "

  rm -rf "$spheres_dir_host"
  docker cp "$CONTAINER_NAME:$spheres_dir_in_container" "$spheres_dir_host"
  find "$spheres_dir_host" -type f -name "$MIC_IR_FILENAME" -print

  trap - INT TERM EXIT
}

main "$@"
#!/usr/bin/env bash
set -euo pipefail

PARAMS_FILE="params.txt"

BASE_DIR="results/$(basename "$PARAMS_FILE" .txt)/meshes"
MESH_PATH_FILE="$BASE_DIR/mesh_pkl_path.txt"

# create directory only if it doesn't exist
mkdir -p "$BASE_DIR"

/dolfinx-env/bin/python3 code/run_mesh_from_params.py --params "$PARAMS_FILE"

if [[ ! -f "$MESH_PATH_FILE" ]]; then
  echo "Mesh path file not found: $MESH_PATH_FILE" >&2
  exit 1
fi

MESH_PKL=$(cat "$MESH_PATH_FILE")

mpiexec -n 32 /dolfinx-env/bin/python3 code/msh_to_xdmf.py \
  --mesh-pkl "$MESH_PKL"


mpiexec -n 32 /dolfinx-env/bin/python3 code/run_solver_from_params.py \
  --params "$PARAMS_FILE" \
  --mesh-pkl "$MESH_PKL"

/dolfinx-env/bin/python3 code/build_hoa_from_pkls.py --config "$PARAMS_FILE"
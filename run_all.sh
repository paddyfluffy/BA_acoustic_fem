#!/usr/bin/env bash
set -euo pipefail

PARAMS_FILE="test_params.txt"
MESH_PATH_FILE="results/$(basename "$PARAMS_FILE" .txt)/meshes/mesh_pkl_path.txt"

/dolfinx-env/bin/python3 code/run_mesh_from_params.py --params "$PARAMS_FILE"

if [[ ! -f "$MESH_PATH_FILE" ]]; then
  echo "Mesh path file not found: $MESH_PATH_FILE" >&2
  exit 1
fi

MESH_PKL=$(cat "$MESH_PATH_FILE")

mpiexec -n 32 /dolfinx-env/bin/python3 code/run_solver_from_params.py \
  --params "$PARAMS_FILE" \
  --mesh-pkl "$MESH_PKL"

/dolfinx-env/bin/python3 code/build_hoa_from_pkls.py --config "$PARAMS_FILE"

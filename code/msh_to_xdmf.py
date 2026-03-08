import argparse
import os
import pickle
from mpi4py import MPI

from utils.mesh_utils import create_mesh


def _xdmf_path(msh_path, out_dir=None):
    base = os.path.splitext(os.path.basename(msh_path))[0] + ".xdmf"
    if out_dir:
        return os.path.join(out_dir, base)
    return os.path.join(os.path.dirname(msh_path), base)


def _load_msh_paths(mesh_pkl):
    with open(mesh_pkl, "rb") as f:
        data = pickle.load(f)
    msh_paths = data.get("msh_paths")
    if msh_paths is None:
        raise ValueError("mesh_pkl does not contain 'msh_paths'")
    return data, msh_paths


def main():
    parser = argparse.ArgumentParser(description="Convert .msh to .xdmf using all MPI ranks per mesh.")
    parser.add_argument("--mesh-pkl", required=True, help="Mesh metadata .pkl containing 'msh_paths'")
    parser.add_argument("--out-dir", help="Output directory for .xdmf files")
    parser.add_argument("--gdim", type=int, default=3, help="Geometry dimension")
    parser.add_argument("--name", default="mesh", help="Mesh name in XDMF")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    data, msh_paths = _load_msh_paths(args.mesh_pkl)

    if args.out_dir and comm.rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
    comm.barrier()

    if isinstance(msh_paths, dict):
        items = sorted(msh_paths.items(), key=lambda kv: kv[0])
    else:
        items = list(enumerate(msh_paths))

    xdmf_paths = {}
    for key, msh_path in items:
        if comm.rank == 0:
            xdmf_path = _xdmf_path(msh_path, args.out_dir)
            exists = os.path.exists(xdmf_path)
        else:
            xdmf_path = None
            exists = None
        xdmf_path = comm.bcast(xdmf_path, root=0)
        exists = comm.bcast(exists, root=0)

        if exists:
            if comm.rank == 0:
                print(f"XDMF exists, skipping: {xdmf_path}")
            xdmf_paths[key] = xdmf_path
            continue

        ok = create_mesh(MPI.COMM_WORLD, meshPath=msh_path, xdmfPath=xdmf_path, gdim=args.gdim, mode="w", name=args.name)
        if comm.rank == 0:
            if ok:
                print(f"Wrote {xdmf_path}")
            else:
                print(f"Failed to write {xdmf_path}")
        xdmf_paths[key] = xdmf_path

    if comm.rank == 0:
        data["xdmf_paths"] = xdmf_paths
        with open(args.mesh_pkl, "wb") as f:
            pickle.dump(data, f)
        print(f"Updated mesh metadata with xdmf_paths: {args.mesh_pkl}")


if __name__ == "__main__":
    main()

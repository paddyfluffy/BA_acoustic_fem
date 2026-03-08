import argparse
import os
import re
import numpy as np

from utils.gmsh_step_mesher import mesh_range_from_planes

TOKEN_RE = re.compile(r"\(|\)|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _parse_sexpr(text):
    tokens = TOKEN_RE.findall(text.replace(",", " "))
    idx = 0

    def parse_list():
        nonlocal idx
        items = []
        if tokens[idx] != "(":
            raise ValueError("Expected '(' in list")
        idx += 1
        while idx < len(tokens) and tokens[idx] != ")":
            tok = tokens[idx]
            if tok == "(":
                items.append(parse_list())
            else:
                items.append(float(tok))
                idx += 1
        if idx >= len(tokens) or tokens[idx] != ")":
            raise ValueError("Unmatched '(' in list")
        idx += 1
        return items

    if not tokens:
        return None
    result = parse_list()
    if idx != len(tokens):
        raise ValueError("Extra tokens after parsing")
    return result


def _parse_sections(path):
    sections = {}
    current = None
    buf = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith(":"):
                if current:
                    sections[current] = " ".join(buf)
                current = line[:-1].strip().lower().replace(" ", "_")
                buf = []
            else:
                buf.append(line)
    if current:
        sections[current] = " ".join(buf)
    return sections


def main():
    parser = argparse.ArgumentParser(description="Mesh from params.txt (no CLI config for mesh options).")
    parser.add_argument("--params", default="params.txt", help="Path to params .txt file")
    args = parser.parse_args()

    params_path = args.params
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Params file not found: {params_path}")

    sections = _parse_sections(params_path)
    if "planes" not in sections:
        raise ValueError("Params file must include a 'planes:' section")

    planes = _parse_sexpr(sections["planes"])

    if "meshing_freqs" in sections:
        freqs = _parse_sexpr(sections["meshing_freqs"])
    else:
        raise ValueError("Params file must include a 'meshing_freqs:' section")

    base_name = os.path.splitext(os.path.basename(params_path))[0]
    outdir = os.path.join("results", base_name, "meshes")
    os.makedirs(outdir, exist_ok=True)

    mesh_pkl = mesh_range_from_planes(
        planes=planes,
        frequencies=freqs,
        show_meshing_info=True,
        data_pkl_path=os.path.join(outdir, "planes_mesh_data.pkl"),
        outpath=outdir,
        c=343.0,
        div=10.0,
        num_threads=os.cpu_count() or 1,
        algorithm3d="hxt",
    )

    if mesh_pkl is None:
        raise RuntimeError("Mesh generation failed")

    mesh_pkl_path = os.path.join(outdir, "mesh_pkl_path.txt")
    with open(mesh_pkl_path, "w", encoding="utf-8") as f:
        f.write(mesh_pkl)

    print(f"Mesh metadata saved to: {mesh_pkl}")
    print(f"Mesh metadata path file: {mesh_pkl_path}")


if __name__ == "__main__":
    main()

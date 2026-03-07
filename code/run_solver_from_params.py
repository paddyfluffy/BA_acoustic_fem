import argparse
import os
import re
import numpy as np
from mpi4py import MPI

import code.dolfinx_computational_acoustics as solver


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


def _compute_freqs(sections, default_min=20, default_max=3000, default_step=5):
    if "freqs" in sections:
        freqs = _parse_sexpr(sections["freqs"])
        return [float(f) for f in freqs]

    freq_min = default_min
    freq_max = default_max
    freq_step = default_step

    if "freq_range" in sections:
        vals = _parse_sexpr(sections["freq_range"])
        if len(vals) >= 2:
            freq_min = float(vals[0])
            freq_max = float(vals[1])
        if len(vals) >= 3:
            freq_step = float(vals[2])
    if "freq_min" in sections:
        freq_min = float(_parse_sexpr(sections["freq_min"])[0])
    if "freq_max" in sections:
        freq_max = float(_parse_sexpr(sections["freq_max"])[0])
    if "freq_step" in sections:
        freq_step = float(_parse_sexpr(sections["freq_step"])[0])

    freqs = np.arange(freq_min, freq_max + freq_step, freq_step)
    return [int(f) if float(f).is_integer() else float(f) for f in freqs]


def main():
    parser = argparse.ArgumentParser(description="Run Helmholtz solve from params file (no meshing).")
    parser.add_argument("--params", required=True, help="Path to params .txt file")
    parser.add_argument("--mesh-pkl", required=True, help="Path to mesh metadata .pkl")
    parser.add_argument("--outdir", default=None, help="Output directory for results")
    args = parser.parse_args()

    sections = _parse_sections(args.params)

    if "mic_positions" not in sections and "mic_positions_list" not in sections:
        raise ValueError("Params file must include a 'mic_positions:' section")

    if "mic_positions" in sections:
        mic_positions = _parse_sexpr(sections["mic_positions"])
    else:
        mic_positions = _parse_sexpr(sections["mic_positions_list"])

    mic_angles = None
    if "mic_angles" in sections:
        mic_angles = _parse_sexpr(sections["mic_angles"])
    elif "mic_angles_list" in sections:
        mic_angles = _parse_sexpr(sections["mic_angles_list"])

    mic_amplitudes = None
    if "mic_amplitudes" in sections:
        mic_amplitudes = _parse_sexpr(sections["mic_amplitudes"])
    elif "mic_amplitude_list" in sections:
        mic_amplitudes = _parse_sexpr(sections["mic_amplitude_list"])

    mic_patterns = None
    if "mic_patterns" in sections:
        mic_patterns = _parse_sexpr(sections["mic_patterns"])
    elif "mic_patterns_list" in sections:
        mic_patterns = _parse_sexpr(sections["mic_patterns_list"])

    wall_abs = None
    if "wall_abs" in sections:
        wall_abs = _parse_sexpr(sections["wall_abs"])
    elif "wall_abs_list" in sections:
        wall_abs = _parse_sexpr(sections["wall_abs_list"])

    source_position = None
    if "source_position" in sections:
        source_position = _parse_sexpr(sections["source_position"])
    elif "source_pos" in sections:
        source_position = _parse_sexpr(sections["source_pos"])

    freqs = _compute_freqs(sections)

    # Update solver config
    solver.CONFIG["mic_positions"] = mic_positions
    solver.CONFIG["mesh_pkl"] = args.mesh_pkl
    solver.CONFIG["freqs"] = freqs
    solver.CONFIG["freq_range"] = (min(freqs), max(freqs))
    solver.CONFIG["freq_step"] = freqs[1] - freqs[0] if len(freqs) > 1 else 1

    base_name = os.path.splitext(os.path.basename(args.params))[0]
    if args.outdir is None:
        mesh_dir = os.path.dirname(os.path.abspath(args.mesh_pkl))
        parent_dir = os.path.dirname(mesh_dir)
        solver.CONFIG["results_folder"] = parent_dir
    else:
        solver.CONFIG["results_folder"] = args.outdir

    if mic_angles is not None:
        solver.CONFIG["mic_angles"] = mic_angles
    if mic_amplitudes is not None:
        solver.CONFIG["mic_amplitudes"] = mic_amplitudes
    if mic_patterns is not None:
        solver.CONFIG["mic_patterns"] = mic_patterns
    if wall_abs is not None:
        solver.CONFIG["wall_abs"] = wall_abs
    if source_position is not None:
        if isinstance(source_position, list) and len(source_position) == 1:
            source_position = source_position[0]
        solver.CONFIG["source_position"] = source_position

    solver.main()


if __name__ == "__main__":
    main()

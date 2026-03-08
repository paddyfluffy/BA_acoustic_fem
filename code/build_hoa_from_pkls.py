import argparse
import os
import re
import numpy as np
from pathlib import Path
from scipy.io.wavfile import write

import pyfar as pf
import spharpy.samplings as sps
import spharpy.spherical as sph

from utils.gmsh_step_mesher import load_pickle


# -----------------------------
# Settings (match  FEM run)
# -----------------------------
ROOT = Path("results/medium_room_larger_spheres/spheres")  # mic_0/, mic_1/, ...
FS = 44100

FREQ_MIN = 50
FREQ_MAX = 3000
FREQ_STEP = 5 # must match  saved sweep

HOA_ORDER_OUT = 1  # fixed HOA output order -> (N+1)^2 channels
T_DESIGN_NMAX = 8  # must match spharpy_dual_sphere(... t_design_nmax=10)

OUT_NAME = "hoa_ir.wav"
OUT_SPECTRUM = "hoa_spectrum.npz"

DIR_OUT_NAME = "hoa_ir_dir.wav"
DIR_OUT_SPECTRUM = "hoa_spectrum_dir.npz"
MIC_ANGLES_DEG = None
MIC_ELEV_DEG = 0.0


# -----------------------------
# Helpers
# -----------------------------
def acn_num_channels(N: int) -> int:
    return (N + 1) ** 2

def parse_freq_hz(filename: str) -> int:
    m = re.search(r"_([0-9]+(?:\.[0-9]+)?)Hz\.pkl$", filename)
    if not m:
        raise ValueError(f"Cannot parse frequency from: {filename}")
    val = float(m.group(1))
    return int(val) if val.is_integer() else val


def _parse_sections(path: str) -> dict:
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


def _parse_sexpr(text: str):
    tokens = re.findall(r"\(|\)|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text.replace(",", " "))
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


def _parse_float(value: str) -> float:
    return float(value.strip())


def _parse_int(value: str) -> int:
    return int(float(value.strip()))


def _apply_config(path: str) -> None:
    global ROOT, FREQ_MIN, FREQ_MAX, FREQ_STEP, OUT_NAME, FS, HOA_ORDER_OUT
    global DIR_OUT_NAME, DIR_OUT_SPECTRUM, MIC_ANGLES_DEG, MIC_ELEV_DEG
    sections = _parse_sections(path)
    base_name = os.path.splitext(os.path.basename(path))[0]

    if "spheres_root" in sections:
        ROOT = Path(sections["spheres_root"])
    elif "root" in sections:
        ROOT = Path(sections["root"])
    else:
        ROOT = Path("results") / base_name / "spheres"

    if "freq_range" in sections:
        vals = _parse_sexpr(sections["freq_range"])
        if len(vals) >= 2:
            FREQ_MIN = float(vals[0])
            FREQ_MAX = float(vals[1])
        if len(vals) >= 3:
            FREQ_STEP = float(vals[2])

    if "freq_min" in sections:
        FREQ_MIN = _parse_float(sections["freq_min"])
    if "freq_max" in sections:
        FREQ_MAX = _parse_float(sections["freq_max"])
    if "freq_step" in sections:
        FREQ_STEP = _parse_float(sections["freq_step"])

    if "out_name" in sections:
        OUT_NAME = sections["out_name"].strip()

    if "sampling_rate" in sections:
        FS = _parse_int(sections["sampling_rate"])
    if "hoa_order" in sections:
        HOA_ORDER_OUT = _parse_int(sections["hoa_order"])

    if "dir_out_name" in sections:
        DIR_OUT_NAME = sections["dir_out_name"].strip()
    if "dir_out_spectrum" in sections:
        DIR_OUT_SPECTRUM = sections["dir_out_spectrum"].strip()
    if "mic_angles" in sections:
        vals = _parse_sexpr(sections["mic_angles"])
        if vals is not None:
            MIC_ANGLES_DEG = [float(v) for v in vals]
    if "mic_elev" in sections:
        MIC_ELEV_DEG = float(sections["mic_elev"].strip())

def choose_nfft(fs: int, df_target: float) -> int:
    """
    Pick NFFT so that df = fs/NFFT equals df_target exactly (or extremely close).
    For df_target=1 Hz at fs=44100 -> NFFT=44100.
    """
    nfft = int(round(fs / df_target))
    df = fs / nfft
    if abs(df - df_target) > 1e-12:
        raise RuntimeError(
            f"df mismatch: wanted {df_target} Hz, got {df} Hz with NFFT={nfft}. "
            "Change FS, df_target, or allow interpolation."
        )
    return nfft

def weighted_lstsq(Y: np.ndarray, p: np.ndarray, w: np.ndarray | None) -> np.ndarray:
    """
    Weighted least squares for complex p.
    Y: (n_pts, n_coeff) real (real SH basis)
    p: (n_pts,) complex
    w: (n_pts,) real weights (sum ~ 4π) or None
    Returns a: (n_coeff,) complex
    """
    if w is None:
        a, *_ = np.linalg.lstsq(Y, p, rcond=None)
        return a

    ws = np.sqrt(np.asarray(w, dtype=float))
    A = Y * ws[:, None]
    b = p * ws
    a, *_ = np.linalg.lstsq(A, b, rcond=None)
    return a

def reconstruct_sampling():
    """
    Reconstruct unit directions and SH basis using spharpy t-design.
    Assumes ordering matches spharpy_dual_sphere generation.
    """
    coords = sps.spherical_t_design(T_DESIGN_NMAX)
    Y = sph.spherical_harmonic_basis_real(HOA_ORDER_OUT, coords)
    n_pts = Y.shape[0]
    return Y, n_pts

def _dir_to_xyz(azim_deg: float, elev_deg: float) -> tuple[float, float, float]:
    az = np.deg2rad(azim_deg)
    el = np.deg2rad(elev_deg)
    x = np.cos(az) * np.cos(el)
    y = np.sin(az) * np.cos(el)
    z = np.sin(el)
    return float(x), float(y), float(z)

class _Coord:
    def __init__(self, x: float, y: float, z: float):
        self.x = np.array([x], dtype=float)
        self.y = np.array([y], dtype=float)
        self.z = np.array([z], dtype=float)
        self.n_points = 1
        self.azimuth = np.array([np.arctan2(y, x)], dtype=float)
        self.elevation = np.array([np.arctan2(z, np.hypot(x, y))], dtype=float)

def steering_weights(order: int, azim_deg: float, elev_deg: float) -> np.ndarray:
    x, y, z = _dir_to_xyz(azim_deg, elev_deg)
    coords = _Coord(x, y, z)
    Y_dir = sph.spherical_harmonic_basis_real(order, coords)
    return np.asarray(Y_dir, dtype=float).reshape(-1)


# -----------------------------
# Processing per mic folder
# -----------------------------
def process_mic_folder(mic_dir: Path):
    pkls = sorted(mic_dir.glob("spherical_cardioid_signal_*Hz.pkl"))
    if not pkls:
        print(f"[{mic_dir.name}] no pkl files found")
        return

    # reconstruct sampling
    Y, n_pts = reconstruct_sampling()

    # FFT / spectrum grid: df must match FREQ_STEP (no interpolation)
    nfft = choose_nfft(FS, float(FREQ_STEP))
    df = FS / nfft
    n_bins = nfft // 2 + 1

    n_ch = acn_num_channels(HOA_ORDER_OUT)
    H = np.zeros((n_ch, n_bins), dtype=np.complex128)

    used = 0
    order_energy_accum = np.zeros(HOA_ORDER_OUT + 1, dtype=np.float64)
    for pkl_path in pkls:
        f_hz = parse_freq_hz(pkl_path.name)

        if f_hz < FREQ_MIN or f_hz > FREQ_MAX:
            continue
        if (f_hz - FREQ_MIN) % FREQ_STEP != 0:
            continue

        data = load_pickle(str(pkl_path))
        p_vals = np.asarray(data["values"], dtype=np.complex128)

        if p_vals.shape[0] != n_pts:
            raise RuntimeError(
                f"{pkl_path.name}: values length={p_vals.shape[0]} does not match "
                f"sampling n_pts={n_pts}. This indicates sampling mismatch."
            )

        w = data.get("weights", None)
        if w is not None:
            w = np.asarray(w, dtype=float)
            if w.shape[0] != n_pts:
                raise RuntimeError(
                    f"{pkl_path.name}: weights length={w.shape[0]} does not match n_pts={n_pts}."
                )

        # SH coefficients (ACN order), complex
        a = weighted_lstsq(Y, p_vals, w)

        # Accumulate per-order energy for diagnostics
        for l in range(HOA_ORDER_OUT + 1):
            start = l * l
            end = (l + 1) * (l + 1)
            order_energy_accum[l] += np.sum(np.abs(a[start:end]) ** 2)

        # place into correct FFT bin
        bin_idx = int(round(f_hz / df))
        if bin_idx < 0 or bin_idx >= n_bins:
            continue
        H[:, bin_idx] = a
        used += 1

    print(f"[{mic_dir.name}] filled {used} frequency bins (df={df} Hz, nfft={nfft})")
    if used > 0:
        order_energy_norm = order_energy_accum / np.max(order_energy_accum)
        order_energy_db = 10 * np.log10(np.maximum(order_energy_norm, 1e-16))
        print(f"[{mic_dir.name}] HOA order energy (dB, normalized): {order_energy_db}")

    # Explicit DC/Nyquist
    H[:, 0] = 0.0
    if nfft % 2 == 0:
        H[:, -1] = 0.0

    # IFFT per channel using pyfar
    ir = np.zeros((n_ch, nfft), dtype=np.float64)
    for ch in range(n_ch):
        ir_signal = pf.dsp.fft.irfft(H[ch], n_samples=nfft, sampling_rate=FS, fft_norm='none')
        ir[ch] = np.array(ir_signal)

    # Normalize
    peak = np.max(np.abs(ir))
    if peak > 0:
        ir = 0.9 * ir / peak

    wav = ir.T.astype(np.float32)  # (samples, channels)
    out_path = mic_dir / OUT_NAME
    write(str(out_path), FS, wav)
    print(f"[{mic_dir.name}] wrote {out_path}  shape={wav.shape}")

    # Save frequency-domain HOA spectrum for direct convolution
    spec_path = mic_dir / OUT_SPECTRUM
    np.savez(
        spec_path,
        H=H,
        fs=FS,
        nfft=nfft,
        df=df,
        freq_min=FREQ_MIN,
        freq_max=FREQ_MAX,
        freq_step=FREQ_STEP,
    )
    print(f"[{mic_dir.name}] wrote {spec_path}")

    if MIC_ANGLES_DEG is not None:
        m = re.search(r"mic_(\d+)$", mic_dir.name)
        if m:
            mic_idx = int(m.group(1))
            if mic_idx < len(MIC_ANGLES_DEG):
                azim = MIC_ANGLES_DEG[mic_idx]
                w = steering_weights(HOA_ORDER_OUT, azim, MIC_ELEV_DEG)
                dir_ir = w @ ir

                peak_dir = np.max(np.abs(dir_ir))
                if peak_dir > 0:
                    dir_ir = 0.9 * dir_ir / peak_dir

                dir_wav = dir_ir.astype(np.float32)
                dir_out_path = mic_dir / DIR_OUT_NAME
                write(str(dir_out_path), FS, dir_wav)
                print(f"[{mic_dir.name}] wrote {dir_out_path}  shape={dir_wav.shape}")

                dir_spec_path = mic_dir / DIR_OUT_SPECTRUM
                H_dir = np.fft.rfft(dir_ir, n=nfft)
                np.savez(
                    dir_spec_path,
                    H_dir=H_dir,
                    fs=FS,
                    nfft=nfft,
                    df=df,
                    azim_deg=azim,
                    elev_deg=MIC_ELEV_DEG,
                )
                print(f"[{mic_dir.name}] wrote {dir_spec_path}")


def main():
    parser = argparse.ArgumentParser(description="Build HOA IRs from sphere PKLs.")
    parser.add_argument("--config", type=str, default=None, help="Path to config .txt")
    args = parser.parse_args()

    if args.config:
        _apply_config(args.config)

    mic_dirs = [d for d in sorted(ROOT.glob("mic_*")) if d.is_dir()]
    for mic_dir in mic_dirs:
        process_mic_folder(mic_dir)


if __name__ == "__main__":
    main()
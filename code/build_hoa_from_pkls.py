import argparse
import os
import re
import numpy as np
from pathlib import Path
from scipy.io.wavfile import write

import pyfar as pf
import spharpy.samplings as sps
import spharpy.spherical as sph

# Suppress SSL warnings from urllib3 (used by spharpy for downloading t-design data)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from utils.gmsh_step_mesher import load_pickle


# -----------------------------
# Settings (match FEM run)
# -----------------------------
ROOT = Path("results/params/spheres")  # mic_0/, mic_1/, ...
FS = 44100

FREQ_MIN = 20
FREQ_MAX = 1000
FREQ_STEP = 5  # must match saved sweep

HOA_ORDER_OUT = 1  # fixed HOA output order -> (N+1)^2 channels
T_DESIGN_NMAX = 8  # must match spharpy_dual_sphere(... t_design_nmax=10)

ZERO_PAD_FACTOR = 1.0
FADE_OUT_MS = 0  # optional safety fade only
LISTEN_GAIN_DB = 0
TARGET_PEAK = 0.9

# New: smooth band edges in frequency domain
FREQ_TAPER_LOW_HZ = 20.0
FREQ_TAPER_HIGH_HZ = 100.0

OUT_NAME = "hoa_ir.wav"
OUT_SPECTRUM = "hoa_spectrum.npz"

DIR_OUT_NAME = "hoa_ir_dir.wav"
DIR_OUT_SPECTRUM = "hoa_spectrum_dir.npz"
MIC_ANGLES_DEG = None
MIC_AMPLITUDES = None
MIC_ELEV_DEG = 0.0


# -----------------------------
# Helpers
# -----------------------------
def acn_num_channels(N: int) -> int:
    return (N + 1) ** 2


def parse_freq_hz(filename: str) -> int | float:
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
    global DIR_OUT_NAME, DIR_OUT_SPECTRUM, MIC_ANGLES_DEG, MIC_AMPLITUDES, MIC_ELEV_DEG
    global ZERO_PAD_FACTOR, FADE_OUT_MS, LISTEN_GAIN_DB
    global FREQ_TAPER_LOW_HZ, FREQ_TAPER_HIGH_HZ

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

    if "zero_pad_factor" in sections:
        ZERO_PAD_FACTOR = max(1.0, _parse_float(sections["zero_pad_factor"]))
    if "fade_out_ms" in sections:
        FADE_OUT_MS = max(0, _parse_int(sections["fade_out_ms"]))
    if "listen_gain_db" in sections:
        LISTEN_GAIN_DB = _parse_float(sections["listen_gain_db"])

    if "freq_taper_low_hz" in sections:
        FREQ_TAPER_LOW_HZ = max(0.0, _parse_float(sections["freq_taper_low_hz"]))
    if "freq_taper_high_hz" in sections:
        FREQ_TAPER_HIGH_HZ = max(0.0, _parse_float(sections["freq_taper_high_hz"]))

    if "dir_out_name" in sections:
        DIR_OUT_NAME = sections["dir_out_name"].strip()
    if "dir_out_spectrum" in sections:
        DIR_OUT_SPECTRUM = sections["dir_out_spectrum"].strip()
    if "mic_angles" in sections:
        vals = _parse_sexpr(sections["mic_angles"])
        if vals is not None:
            MIC_ANGLES_DEG = [float(v) for v in vals]
    if "mic_amplitudes" in sections:
        vals = _parse_sexpr(sections["mic_amplitudes"])
        if vals is not None:
            MIC_AMPLITUDES = [float(v) for v in vals]
    if "mic_elev" in sections:
        MIC_ELEV_DEG = float(sections["mic_elev"].strip())


def choose_nfft(fs: int, df_target: float) -> int:
    nfft = int(round(fs / df_target))
    df = fs / nfft
    if abs(df - df_target) > 1e-12:
        raise RuntimeError(
            f"df mismatch: wanted {df_target} Hz, got {df} Hz with NFFT={nfft}. "
            "Change FS, df_target, or allow interpolation."
        )
    return nfft


def weighted_lstsq(Y: np.ndarray, p: np.ndarray, w: np.ndarray | None) -> np.ndarray:
    if w is None:
        a, *_ = np.linalg.lstsq(Y, p, rcond=None)
        return a

    ws = np.sqrt(np.asarray(w, dtype=float))
    A = Y * ws[:, None]
    b = p * ws
    a, *_ = np.linalg.lstsq(A, b, rcond=None)
    return a


def reconstruct_sampling():
    coords = sps.spherical_t_design(T_DESIGN_NMAX)
    Y = sph.spherical_harmonic_basis_real(HOA_ORDER_OUT, coords)
    n_pts = Y.shape[0]
    return Y, n_pts


def _apply_fade(ir: np.ndarray, fs: int, fade_ms: int) -> np.ndarray:
    if fade_ms <= 0:
        return ir
    fade_len = int(round(fs * (fade_ms / 1000.0)))
    if fade_len <= 0 or fade_len >= ir.shape[-1]:
        return ir
    window = np.linspace(1.0, 0.0, fade_len, dtype=ir.dtype)
    ir[..., -fade_len:] *= window
    return ir


def _apply_gain_db(x: np.ndarray, gain_db: float) -> np.ndarray:
    if gain_db == 0:
        return x
    scale = 10 ** (gain_db / 20.0)
    return x * scale


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


def _mic_index_from_dirname(mic_dir: Path) -> int | None:
    m = re.search(r"mic_(\d+)$", mic_dir.name)
    if not m:
        return None
    return int(m.group(1))


def _cosine_taper_1d(x: np.ndarray, x0: float, x1: float, rising: bool) -> np.ndarray:
    """
    Smooth cosine ramp from 0->1 (rising=True) or 1->0 (rising=False)
    between x0 and x1.
    """
    y = np.zeros_like(x, dtype=float) if rising else np.ones_like(x, dtype=float)

    if x1 <= x0:
        if rising:
            y[x >= x1] = 1.0
        else:
            y[x >= x0] = 0.0
        return y

    mask_mid = (x >= x0) & (x <= x1)
    t = (x[mask_mid] - x0) / (x1 - x0)

    if rising:
        y[x > x1] = 1.0
        y[mask_mid] = 0.5 - 0.5 * np.cos(np.pi * t)
    else:
        y[x > x1] = 0.0
        y[mask_mid] = 0.5 + 0.5 * np.cos(np.pi * t)

    return y


def _build_band_taper(freqs_hz: np.ndarray, fmin: float, fmax: float,
                      taper_low_hz: float, taper_high_hz: float) -> np.ndarray:
    """
    Create frequency-domain window:
    - 0 below fmin - taper_low_hz
    - smooth rise to 1 at fmin
    - 1 in passband
    - smooth fall from fmax to fmax + taper_high_hz
    - 0 above that
    """
    if fmax <= fmin:
        raise ValueError("fmax must be > fmin")

    low_start = max(0.0, fmin - taper_low_hz)
    low_end = fmin
    high_start = fmax
    high_end = fmax + taper_high_hz

    low_win = _cosine_taper_1d(freqs_hz, low_start, low_end, rising=True)
    high_win = _cosine_taper_1d(freqs_hz, high_start, high_end, rising=False)
    return low_win * high_win


# -----------------------------
# Build raw data per mic
# -----------------------------
def build_mic_data(mic_dir: Path) -> dict | None:
    pkls = sorted(mic_dir.glob("spherical_cardioid_signal_*Hz.pkl"))
    if not pkls:
        print(f"[{mic_dir.name}] no pkl files found")
        return None

    Y, n_pts = reconstruct_sampling()

    nfft = choose_nfft(FS, float(FREQ_STEP))
    df = FS / nfft
    n_bins = nfft // 2 + 1
    freq_axis = np.arange(n_bins, dtype=float) * df

    n_ch = acn_num_channels(HOA_ORDER_OUT)
    H = np.zeros((n_ch, n_bins), dtype=np.complex128)

    used = 0
    order_energy_accum = np.zeros(HOA_ORDER_OUT + 1, dtype=np.float64)
    printed_stats = False

    for pkl_path in pkls:
        f_hz = parse_freq_hz(pkl_path.name)

        if f_hz < FREQ_MIN or f_hz > FREQ_MAX:
            continue
        if (f_hz - FREQ_MIN) % FREQ_STEP != 0:
            continue

        data = load_pickle(str(pkl_path))
        p_vals = np.asarray(data["values"], dtype=np.complex128)

        if not printed_stats:
            abs_vals = np.abs(p_vals)
            print(
                f"[{mic_dir.name}] {pkl_path.name} |p| min={abs_vals.min():.3e} "
                f"max={abs_vals.max():.3e} mean={abs_vals.mean():.3e}"
            )
            printed_stats = True

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

        a = weighted_lstsq(Y, p_vals, w)

        for l in range(HOA_ORDER_OUT + 1):
            start = l * l
            end = (l + 1) * (l + 1)
            order_energy_accum[l] += np.sum(np.abs(a[start:end]) ** 2)

        bin_idx = int(round(f_hz / df))
        if 0 <= bin_idx < n_bins:
            H[:, bin_idx] = a
            used += 1

    print(f"[{mic_dir.name}] filled {used} frequency bins (df={df} Hz, nfft={nfft})")
    if used > 0 and np.max(order_energy_accum) > 0:
        order_energy_norm = order_energy_accum / np.max(order_energy_accum)
        order_energy_db = 10 * np.log10(np.maximum(order_energy_norm, 1e-16))
        print(f"[{mic_dir.name}] HOA order energy (dB, normalized): {order_energy_db}")

    H[:, 0] = 0.0
    if nfft % 2 == 0:
        H[:, -1] = 0.0

    # Smooth frequency-domain taper to reduce ringing / abrupt time-domain tail
    band_taper = _build_band_taper(
        freq_axis,
        fmin=FREQ_MIN,
        fmax=FREQ_MAX,
        taper_low_hz=FREQ_TAPER_LOW_HZ,
        taper_high_hz=FREQ_TAPER_HIGH_HZ,
    )
    H *= band_taper[None, :]

    nfft_pad = int(round(nfft * max(1.0, ZERO_PAD_FACTOR)))
    ir_raw = np.zeros((n_ch, nfft_pad), dtype=np.float64)
    for ch in range(n_ch):
        ir_signal = pf.dsp.fft.irfft(
            H[ch],
            n_samples=nfft_pad,
            sampling_rate=FS,
            fft_norm="none",
        )
        ir_raw[ch] = np.array(ir_signal)

    # Optional tiny safety fade only
    ir_raw = _apply_fade(ir_raw, FS, FADE_OUT_MS)

    mic_idx = _mic_index_from_dirname(mic_dir)
    dir_ir_raw = None
    azim = None

    if (
        MIC_ANGLES_DEG is not None
        and mic_idx is not None
        and mic_idx < len(MIC_ANGLES_DEG)
    ):
        azim = MIC_ANGLES_DEG[mic_idx]
        w_dir = steering_weights(HOA_ORDER_OUT, azim, MIC_ELEV_DEG)
        dir_ir_raw = w_dir @ ir_raw

        mic_gain = 1.0
        if MIC_AMPLITUDES is not None and mic_idx < len(MIC_AMPLITUDES):
            mic_gain = float(MIC_AMPLITUDES[mic_idx])
        if mic_gain != 1.0:
            dir_ir_raw = dir_ir_raw * mic_gain

        dir_ir_raw = _apply_fade(dir_ir_raw, FS, FADE_OUT_MS)

    print(f"[{mic_dir.name}] raw Max |H|={np.max(np.abs(H)):.3e}")
    print(f"[{mic_dir.name}] raw Max |ir|={np.max(np.abs(ir_raw)):.3e}")
    if dir_ir_raw is not None:
        print(f"[{mic_dir.name}] raw Max |dir_ir|={np.max(np.abs(dir_ir_raw)):.3e}")

    return {
        "mic_dir": mic_dir,
        "mic_idx": mic_idx,
        "H": H,
        "ir_raw": ir_raw,
        "dir_ir_raw": dir_ir_raw,
        "azim_deg": azim,
        "nfft": nfft,
        "nfft_pad": nfft_pad,
        "df": df,
        "band_taper": band_taper,
    }


def write_mic_outputs(mic_data: dict, global_scale: float) -> None:
    mic_dir = mic_data["mic_dir"]
    H = mic_data["H"]
    ir_raw = mic_data["ir_raw"]
    dir_ir_raw = mic_data["dir_ir_raw"]
    azim = mic_data["azim_deg"]
    nfft = mic_data["nfft"]
    nfft_pad = mic_data["nfft_pad"]
    df = mic_data["df"]
    band_taper = mic_data["band_taper"]

    ir = ir_raw * global_scale
    ir = _apply_gain_db(ir, LISTEN_GAIN_DB)

    wav = ir.T.astype(np.float32)
    out_path = mic_dir / f"{mic_dir.name}_{OUT_NAME}"
    write(str(out_path), FS, wav)
    print(f"[{mic_dir.name}] wrote {out_path}  shape={wav.shape}")

    spec_path = mic_dir / OUT_SPECTRUM
    np.savez(
        spec_path,
        H=H,
        ir_raw=ir_raw,
        band_taper=band_taper,
        global_scale=global_scale,
        listen_gain_db=LISTEN_GAIN_DB,
        fs=FS,
        nfft=nfft,
        nfft_pad=ir_raw.shape[-1],
        df=df,
        freq_min=FREQ_MIN,
        freq_max=FREQ_MAX,
        freq_step=FREQ_STEP,
        freq_taper_low_hz=FREQ_TAPER_LOW_HZ,
        freq_taper_high_hz=FREQ_TAPER_HIGH_HZ,
    )
    print(f"[{mic_dir.name}] wrote {spec_path}")

    if dir_ir_raw is not None:
        dir_ir = dir_ir_raw * global_scale
        dir_ir = _apply_gain_db(dir_ir, LISTEN_GAIN_DB)

        dir_wav = dir_ir.astype(np.float32)
        dir_out_path = mic_dir / f"{mic_dir.name}_{DIR_OUT_NAME}"
        write(str(dir_out_path), FS, dir_wav)
        print(f"[{mic_dir.name}] wrote {dir_out_path}  shape={dir_wav.shape}")

        dir_spec_path = mic_dir / DIR_OUT_SPECTRUM
        H_dir = np.fft.rfft(dir_ir_raw)
        np.savez(
            dir_spec_path,
            H_dir=H_dir,
            dir_ir_raw=dir_ir_raw,
            global_scale=global_scale,
            listen_gain_db=LISTEN_GAIN_DB,
            fs=FS,
            nfft=dir_ir_raw.shape[-1],
            nfft_pad=dir_ir_raw.shape[-1],
            df=FS / dir_ir_raw.shape[-1],
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
    if not mic_dirs:
        raise RuntimeError(f"No mic_* directories found under {ROOT}")

    all_mic_data = []
    global_peak = 0.0

    for mic_dir in mic_dirs:
        mic_data = build_mic_data(mic_dir)
        if mic_data is None:
            continue

        all_mic_data.append(mic_data)

        ir_peak = float(np.max(np.abs(mic_data["ir_raw"])))
        global_peak = max(global_peak, ir_peak)

        if mic_data["dir_ir_raw"] is not None:
            dir_peak = float(np.max(np.abs(mic_data["dir_ir_raw"])))
            global_peak = max(global_peak, dir_peak)

    if not all_mic_data:
        raise RuntimeError("No valid mic data found.")

    global_scale = TARGET_PEAK / global_peak if global_peak > 0 else 1.0

    print(f"[global] peak before normalization = {global_peak:.6e}")
    print(f"[global] normalization scale      = {global_scale:.6e}")
    print(f"[global] listen gain (dB)         = {LISTEN_GAIN_DB:.2f}")

    for mic_data in all_mic_data:
        write_mic_outputs(mic_data, global_scale)


if __name__ == "__main__":
    main()
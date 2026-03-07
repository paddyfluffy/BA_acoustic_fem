import numpy as np

def air_absorption_db_per_m_iso9613(f_hz, T_C=20.0, RH=50.0, p_kPa=101.325):
    """
    ISO 9613-1:1993 pure-tone atmospheric absorption coefficient alpha [dB/m].

    Parameters
    ----------
    f_hz : float or array-like
        Frequency in Hz.
    T_C : float
        Air temperature in deg C.
    RH : float
        Relative humidity in percent (0..100).
    p_kPa : float
        Ambient pressure in kPa.

    Returns
    -------
    alpha_db_per_m : float or np.ndarray
        Atmospheric absorption coefficient in dB/m.
    """
    f = np.asarray(f_hz, dtype=np.float64)
    T = T_C + 273.15  # K

    # ISO reference values
    T0 = 293.15       # K (20C)
    T01 = 273.16      # K (triple point)
    pr = 101.325      # kPa

    # Clamp RH for sanity
    RH = float(np.clip(RH, 0.0, 100.0))

    # Saturation vapour pressure (ISO 9613-1 Annex B closed form):
    # psat = pr * 10^C,  C = -6.8346*(T01/T)^1.261 + 4.6151
    C = -6.8346 * (T01 / T) ** 1.261 + 4.6151
    psat_kPa = pr * (10.0 ** C)

    # Molar concentration of water vapour h in percent:
    # h = RH * psat / pa   (with RH in %, psat and pa in same units)
    h = RH * (psat_kPa / p_kPa)

    # Relaxation frequencies (Hz), ISO eqs (3) and (4)
    frO = (p_kPa / pr) * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frN = (p_kPa / pr) * (T / T0) ** (-0.5) * (
        9.0 + 280.0 * h * np.exp(-4.170 * ((T / T0) ** (-1.0 / 3.0) - 1.0))
    )

    # Attenuation coefficient alpha (dB/m), ISO eq (5)
    term_classical = 1.84e-11 * (pr / p_kPa) * (T / T0) ** 0.5

    term_O = 0.01275 * np.exp(-2239.1 / T) / (frO + (f * f) / frO)
    term_N = 0.1068 * np.exp(-3352.0 / T) / (frN + (f * f) / frN)

    alpha = 8.686 * (f * f) * (
        term_classical + (T / T0) ** (-2.5) * (term_O + term_N)
    )

    return alpha
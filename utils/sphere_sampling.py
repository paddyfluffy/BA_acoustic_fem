import numpy as np

import spharpy.samplings as sps

def spharpy_dual_sphere(center_xyz, radius, dr, t_design_nmax=8):
    """
    Returns inner/outer sphere point clouds using spharpy.

    Choose either:
      - t_design_nmax: relates to the max SH order you want to represent robustly
    """
    center_xyz = np.asarray(center_xyz, dtype=float)

    # 1) Create sampling on unit sphere
    # t-design: pick based on SH order. A safe rule:
    #   t ≳ 2 * N_out  (often ok)
    # But spharpy uses n_max as a parameter; its exact meaning depends on sampler.
    # Start simple and increase if needed.

    coords = sps.spherical_t_design(t_design_nmax)  
    # coords likely has x/y/z on unit sphere
    xyz = np.column_stack([coords.x, coords.y, coords.z]).astype(float)

    # 2) Compute sampling weights (Voronoi cell areas) for integration / SH projection
    # Use scipy's SphericalVoronoi areas; falls back to uniform weights if unavailable.
    try:
        vor = sps.spherical_voronoi(coords)
        if hasattr(vor, "calculate_areas"):
            weights = np.asarray(vor.calculate_areas(), dtype=float)
        else:
            weights = None
    except Exception:
        weights = None

    if weights is None:
        # Fallback: uniform weights that sum to 4π
        weights = np.full(xyz.shape[0], 4 * np.pi / xyz.shape[0], dtype=float)

    # 3) Scale to desired radii and shift to center
    inner = center_xyz[None, :] + radius * xyz
    outer = center_xyz[None, :] + (radius + dr) * xyz

    return inner, outer, weights
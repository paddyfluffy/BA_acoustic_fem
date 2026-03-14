# Acoustic FEM Pipeline

Acoustic FEM simulation pipeline using DOLFINx and Gmsh. Builds meshes from room geometry, runs the Helmholtz solver, and post-processes spherical sampling into HOA impulse responses.

## Docker Setup

1) Build the image (from repo root):

```bash
docker build -f Dockerfile.dolfinx -t acoustics-dev .
```

2) Run the container (example):

```bash
docker run --name acoustics-dev --rm -it \
  -v "$(pwd)":/home/acoustics \
  acoustics-dev
```

Tip: increase Docker memory/CPU limits; FEM simulations can be resource intensive.

## Run From Outside the Container

Use `run_acoustic_sim.sh` on the host. It copies the params file into the container, runs the full pipeline, then copies the spheres output back next to the params file.

```bash
bash run_acoustic_sim.sh params.txt
```

If no argument is passed, it defaults to `params.txt`.

## params.txt Reference

The `params.txt` file is a simple section-based config. Each section name ends with `:` and the values follow on the next lines.

sections:

- `planes:` List of triangle planes defining the room. Each plane is 3 XYZ points.
- `meshing_freqs:` Frequencies used to build meshes (Hz). Include the highest frequency you will solve.
- `source_position:` XYZ location of the point source.
- `mic_positions:` List of mic XYZ positions.
- `mic_angles:` Azimuth angles (deg) for directional IR steering. One per mic.
- `mic_amplitudes:` Per-mic gain multiplier applied to directional IR only.
- `mic_patterns:` Per-mic pattern weights used by downstream processing (optional).
- `wall_abs:` Per-wall absorber settings. Use `0` for rigid. Otherwise `(<sigma> <thickness>)` for Delany-Bazley, where:
  - `sigma` is flow resistivity (Pa*s/m^2). Typical ranges: 5e3 to 5e5.
  - `thickness` is absorber thickness (m). Typical ranges: 0.01 to 0.2.
- `freq_range:` `(min max step)` in Hz for the solver sweep.
- `sampling_rate:` Output sampling rate for IRs (Hz).
- `hoa_order:` HOA order for output (1 -> 4 channels).


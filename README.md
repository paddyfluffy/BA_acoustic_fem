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

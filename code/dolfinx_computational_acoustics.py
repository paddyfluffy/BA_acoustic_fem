import os
import numpy as np
import logging 
import time
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import default_scalar_type

# Suppress SSL warnings from urllib3 (used by spharpy for downloading t-design data)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) # ignore SSL warnings pyfar
from dolfinx.io import XDMFFile
from dolfinx.fem import (
    functionspace, Function, Constant, form
)
from dolfinx.fem.petsc import assemble_matrix
from ufl import (
    TrialFunction, TestFunction, Measure, inner, dx, grad
)
from scifem import PointSource

from utils.locate_points import PointLocator
from utils.mesh_utils import read_xdmf_data
from utils.air_absorption import air_absorption_db_per_m_iso9613
from utils.gmsh_step_mesher import load_pickle, save_pickle

from utils.sphere_sampling import spharpy_dual_sphere 

2
# Initialize logging
if MPI.COMM_WORLD.rank == 0:
    logging.basicConfig(level=logging.INFO)
else:
    logging.basicConfig(level=logging.WARNING)

CONFIG = {
    "rho0": 1.225,
    "c": 343,
    "sigma": 1.5e4,
    "d": 0.02,
    "Q": 1e-5, 
    "source_position": [0.5, 2.5, 1.0],
    "mic_positions": [
        [2.0, 1, 1.6],
        [2.0, 4, 1.6],
    ],
    "deg": 1,
    "freq_range": (20, 3000),
    "freq_step": 5,
    "shift_beta": 0.5,
    "results_folder": "results/medium_room_larger_spheres/",
    "mesh_pkl": "4x5x2p2_div10_hxt/4x5x2p2_mesh_data.pkl",

}

def delany_bazley_layer(f, rho0, c, sigma, d):
    X = rho0 * f / sigma
    Zc = rho0 * c * (1 + 0.0571 * X**-0.754 - 1j * 0.087 * X**-0.732)
    kc = 2 * np.pi * f / c * (1 + 0.0978 * X**-0.700 - 1j * 0.189 * X**-0.595)
    return -1j * Zc * (1 / np.tan(kc * d))

def find_mesh_for_freq(freq, freqs, xdmf_paths):
    for f in freqs:
        if freq <= f:
            return f, xdmf_paths[f]
    return freqs[-1], xdmf_paths[freqs[-1]]


def main():
    os.makedirs(CONFIG["results_folder"], exist_ok=True)
    data_dict = load_pickle(CONFIG["mesh_pkl"])


    if "freqs" in CONFIG and CONFIG["freqs"] is not None:
        freqs = np.array(CONFIG["freqs"], dtype=float)
    else:
        freqs = np.arange(CONFIG["freq_range"][0], CONFIG["freq_range"][1] + 1, CONFIG["freq_step"])    
    z_vals = delany_bazley_layer(freqs, CONFIG["rho0"], CONFIG["c"], CONFIG["sigma"], CONFIG["d"])
    wall_abs_cfg = CONFIG.get("wall_abs")

    mic_positions = CONFIG.get("mic_positions")
    if not mic_positions:
        mic_positions = [CONFIG["mic_position"]]
    mic_positions = [np.array(p, dtype=float) for p in mic_positions]

    # source_positions = CONFIG["source_positions"]
    # source_results_subfolders = []
    # for s_idx, source_pos in enumerate(source_positions):
    #     source_subfolder = os.path.join(CONFIG["results_folder"], f"source_{s_idx}")
    #     os.makedirs(source_subfolder, exist_ok=True)
    #     source_results_subfolders.append(source_subfolder) 


    current_mesh_freq = None

    deg = CONFIG["deg"]
    FREQUENCIES =  data_dict["frequencies"][:]

    # Generate spherical sampling points once (independent of frequency)
    radius = 0.35
    dr = 0.05 # 50mm spacing for radial velocity approximation
    mic_sphere_data = []
    
    for mic_idx, mic_pos in enumerate(mic_positions):
        mic_pos = np.array(mic_pos, dtype=float)
        
        sphere_pts_inner, sphere_pts_outer, sph_weights = spharpy_dual_sphere(
            center_xyz=mic_pos,
            radius=radius,
            dr=dr,
            t_design_nmax=8,
        )
        # logging.info(f" sphere weights sum to: {np.sum(sph_weights)} (should be ~4π={4*np.pi})")
        
        mic_sphere_data.append({
            "idx": mic_idx,
            "pos": mic_pos,
            "sphere_pts_inner": sphere_pts_inner,
            "sphere_pts_outer": sphere_pts_outer,
            "sph_weights": sph_weights,
        })

    for i, f in enumerate(freqs):

        mesh_freq, mesh_path = find_mesh_for_freq(f, FREQUENCIES, data_dict["xdmf_paths"])
        # logging.info(f"Using mesh for {mesh_freq} Hz: {mesh_path}")

        if current_mesh_freq != mesh_freq:
            logging.info(f"Switching to finer mesh: {mesh_freq} Hz at {f} Hz")
            domain, ct, ft = read_xdmf_data(mesh_path, MPI.COMM_WORLD, gdim=3)
            domain.topology.create_connectivity(0, domain.topology.dim)

            omega = Constant(domain, default_scalar_type(0))
            k = Constant(domain, default_scalar_type(0))
            Z = Constant(domain, default_scalar_type(0))

            V = functionspace(domain, ("Lagrange", deg))
            ds = Measure("ds", domain=domain, subdomain_data=ft)

            p = TrialFunction(V)
            v = TestFunction(V)

            # Optional per-wall impedance setup
            wall_group_tags = sorted(data_dict.get("wall_groups", {}).keys())
            use_wall_abs = False
            wall_specs = {}
            wall_Z = {}
            if wall_abs_cfg and wall_group_tags:
                if len(wall_abs_cfg) != len(wall_group_tags):
                    raise ValueError(
                        f"wall_abs length {len(wall_abs_cfg)} does not match wall groups {len(wall_group_tags)}"
                    )
                for tag, spec in zip(wall_group_tags, wall_abs_cfg):
                    if isinstance(spec, (list, tuple)) and len(spec) == 2:
                        wall_specs[tag] = spec
                        wall_Z[tag] = Constant(domain, default_scalar_type(0))
                    else:
                        wall_Z[tag] = None
                use_wall_abs = True

            ### Point Source Setup ###
            source_pos = np.array(CONFIG["source_position"], dtype=domain.geometry.x.dtype)
            source_locator = PointLocator(domain, source_pos)
            source_local_cells, _, _ = source_locator.get()

            found_source_point = domain.comm.allgather(len(source_local_cells) > 0)
            if not any(found_source_point):
                raise RuntimeError(f"Source point {source_pos} not found at {mesh_freq}hz")

            if "point_source" in locals():
                # logging.info("Destroying previous point source")
                del point_source
            if len(source_local_cells) > 0:
                source_points = source_pos
            else:
                source_points = np.zeros((0, 3), dtype=domain.geometry.x.dtype)

            point_source = PointSource(V, source_points, magnitude=CONFIG["Q"])

            # Create locators for the pre-computed spherical points
            mic_eval = []
            for sphere_data in mic_sphere_data:
                mic_pos = np.array(sphere_data["pos"], dtype=domain.geometry.x.dtype)
                mic_locator = PointLocator(domain, mic_pos)
                inner_locator = PointLocator(domain, sphere_data["sphere_pts_inner"])
                outer_locator = PointLocator(domain, sphere_data["sphere_pts_outer"])

                mic_eval.append({
                    "idx": sphere_data["idx"],
                    "pos": mic_pos,
                    "mic_locator": mic_locator,
                    "sphere_pts_inner": sphere_data["sphere_pts_inner"],
                    "sphere_pts_outer": sphere_data["sphere_pts_outer"],
                    "sph_weights": sphere_data["sph_weights"],
                    "inner_locator": inner_locator,
                    "outer_locator": outer_locator,
                })

            # Create per-mic output folders once per mesh
            if domain.comm.rank == 0:
                spheres_root = os.path.join(CONFIG["results_folder"], "spheres")
                os.makedirs(spheres_root, exist_ok=True)
                for m in mic_eval:
                    os.makedirs(os.path.join(spheres_root, f"mic_{m['idx']}"), exist_ok=True)
            domain.comm.barrier()

            p_a = Function(V)
            p_a.name = "pressure"
            b = Function(V)
            b.name = "source_vector"

            ### Solver Setup ###
            if 'solver' in locals():
                logging.info("Destroying previous solver")
                solver.destroy()

            # solver = PETSc.KSP().create(domain.comm)
            # solver.setType("preonly")
            # solver.getPC().setType("lu")
            # solver.getPC().setFactorSolverType("mumps")

            solver = PETSc.KSP().create(domain.comm)
            # solver.setOperators(A)  # you'll reset this each freq anyway

            # fgmres with hypre AMG preconditioner
            opt_prefix = "helm_"
            solver.setOptionsPrefix(opt_prefix)
            opts = PETSc.Options()

            opts[f"{opt_prefix}ksp_type"] = "fgmres"
            opts[f"{opt_prefix}ksp_rtol"] = 1e-4  # tighter tolerance for shifted preconditioning
            opts[f"{opt_prefix}ksp_atol"] = 1e-8
            opts[f"{opt_prefix}ksp_max_it"] = 500  # more iterations may be needed

            # GAMG works well on shifted operator (more elliptic-like than original)
            opts[f"{opt_prefix}pc_type"] = "gamg"
            opts[f"{opt_prefix}pc_gamg_type"] = "agg"

            # Better smoothing for complex/shifted matrices
            opts[f"{opt_prefix}mg_levels_ksp_type"] = "chebyshev" # "richardson"
            opts[f"{opt_prefix}mg_levels_pc_type"] = "jacobi" # "sor"

            # More aggressive coarsening helps with shifted operator
            opts[f"{opt_prefix}pc_gamg_threshold"] = 0.005


            # opts[f"{opt_prefix}ksp_type"] = "fgmres"          # or "fgmres"
            # opts[f"{opt_prefix}ksp_rtol"] = 1e-5             # loosen if you want faster
            # opts[f"{opt_prefix}ksp_atol"] = 0.0
            # opts[f"{opt_prefix}ksp_max_it"] = 400            # cap iterations

            # opts[f"{opt_prefix}pc_type"] = "asm"            # additive Schwarz
            # opts[f"{opt_prefix}sub_pc_type"] = "lu"
            # # opts[f"{opt_prefix}pc_gamg_type"] = "agg"
            # # opts[f"{opt_prefix}pc_gamg_threshold"] = 0.02    # coarsening aggressiveness

            # # smoother on multigrid levels
            # opts[f"{opt_prefix}mg_levels_ksp_type"] = "chebyshev"
            # opts[f"{opt_prefix}mg_levels_pc_type"] = "jacobi"

            solver.setFromOptions()
            current_mesh_freq = mesh_freq

        if "A" in locals():
            A.destroy()
            del aA
        if "P" in locals():
            P.destroy()
            del aP

        omega.value = 2 * np.pi * f
        # k.value = omega.value / CONFIG["c"]
        alpha_db_per_m = air_absorption_db_per_m_iso9613(f, T_C=20.0, RH=50.0, p_kPa=101.325)
        alpha_p = alpha_db_per_m / 8.686
        logging.info(f"Frequency {f} Hz: air absorption alpha = {alpha_db_per_m:.4f} dB/m, alpha_p = {alpha_p:.6f} 1/m")
        k.value = (omega.value / CONFIG["c"]) + 1j * alpha_p
        Z.value = z_vals[i]
        if use_wall_abs:
            for tag, spec in wall_specs.items():
                sigma, d = spec
                wall_Z[tag].value = delany_bazley_layer(f, CONFIG["rho0"], CONFIG["c"], sigma, d)

        b.x.array[:] = 0.0

        # if owning_rank:
        point_source.apply_to_vector(b)

        if use_wall_abs:
            wall_z_info = {tag: (wall_Z[tag].value if wall_Z[tag] is not None else None) for tag in wall_Z}
            logging.info(
                f"Solving for frequency {f} Hz with omega={omega.value:.2f}, k={k.value:.2f}, wall Z values: {wall_z_info}"
            )
        else:
            logging.info(f"Solving for frequency {f} Hz with omega={omega.value}, k={k.value}, Z={Z.value}")

        # a = (
        #     inner(grad(p), grad(v)) * dx
        #     # + 1j * CONFIG["rho0"] * omega / Z * inner(p, v) * ds(100002)
        #     # + sum(1j * CONFIG["rho0"] * omega / Z * inner(p, v) * ds(fid) for fid in vertical_facets)
        #     + sum(1j * CONFIG["rho0"] * omega / Z * inner(p, v) * ds(fid) for fid in data_dict["volume"][1])
        #     - k**2 * inner(p, v) * dx
        # )
        # A = assemble_matrix(form(a))
        # A.assemble()
        # solver.setOperators(A)
        beta = CONFIG.get("shift_beta", 0.5)

        if use_wall_abs:
            boundary_term = 0
            for tag, zc in wall_Z.items():
                if zc is None:
                    continue
                boundary_term += 1j * CONFIG["rho0"] * omega / zc * inner(p, v) * ds(tag)
        else:
            boundary_term = sum(
                1j * CONFIG["rho0"] * omega / Z * inner(p, v) * ds(fid)
                for fid in data_dict["volume"][1]
            )

        # True operator A
        aA = (
            inner(grad(p), grad(v)) * dx
            + boundary_term
            - k**2 * inner(p, v) * dx
        )

        # Shifted operator P = K - (1+i*beta) k^2 M + i*B
        aP = (
            inner(grad(p), grad(v)) * dx
            + boundary_term
            - (1 + 1j * beta) * k**2 * inner(p, v) * dx
        )

        # Assemble both matrices
        A = assemble_matrix(form(aA))
        A.assemble()

        P = assemble_matrix(form(aP))
        P.assemble()

        # IMPORTANT: setOperators(A, P)  (A=system, P=preconditioner matrix)
        solver.setOperators(A, P)
        
        # Start timer for solve
        logging.info(f"Starting solve for {f} Hz")
        solve_start_time = time.time()
        
        solver.solve(b.x.petsc_vec, p_a.x.petsc_vec)
        p_a.x.scatter_forward()

        with XDMFFile(domain.comm, os.path.join(CONFIG["results_folder"], f"solution_{f}Hz.xdmf"), "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(p_a)
        
        # Calculate and log solve time
        solve_elapsed_time = time.time() - solve_start_time
        logging.info(f"Frequency {f} Hz: Solve completed in {solve_elapsed_time:.3f} seconds")

        # Evaluate pressure on both layers for each mic and save
        for m in mic_eval:
            sphere_values_inner = np.full((m["sphere_pts_inner"].shape[0],), np.nan, dtype=complex)
            sphere_values_outer = np.full((m["sphere_pts_outer"].shape[0],), np.nan, dtype=complex)

            vals_inner = m["inner_locator"].evaluate(p_a)
            vals_outer = m["outer_locator"].evaluate(p_a)

            # Fill in valid entries
            sphere_values_inner[m["inner_locator"]._local_indices] = vals_inner.ravel()
            sphere_values_outer[m["outer_locator"]._local_indices] = vals_outer.ravel()

            # Gather values from all ranks
            all_inner = domain.comm.gather(sphere_values_inner, root=0)
            all_outer = domain.comm.gather(sphere_values_outer, root=0)

            if domain.comm.rank == 0:
                stacked_inner = np.stack(all_inner)
                stacked_outer = np.stack(all_outer)

                sphere_values_inner_g = np.nanmean(stacked_inner, axis=0)
                sphere_values_outer_g = np.nanmean(stacked_outer, axis=0)

                valid = (~np.isnan(sphere_values_inner_g)) & (~np.isnan(sphere_values_outer_g))
                pressure_diff = sphere_values_outer_g[valid] - sphere_values_inner_g[valid]

                v_radial = np.zeros_like(sphere_values_inner_g)
                v_radial[valid] = -(pressure_diff / (2 * dr * 1j * omega.value * CONFIG["rho0"]))

                gamma = 1 / (1j * omega.value * CONFIG["c"])
                s_combined = np.full_like(sphere_values_inner_g, np.nan)
                s_combined[valid] = sphere_values_inner_g[valid] + gamma * v_radial[valid]

                # logging.info(
                #     f"Mic {m['idx']}: cardioid signal for {f} Hz computed, valid points: {np.sum(valid)}"
                # )

                sphere_data = {
                    "values": s_combined,
                    "weights": m["sph_weights"], 
                    "mic_position": np.array(m["pos"], dtype=float),
                    "radius": float(radius),
                    "dr": float(dr),
                }

                spheres_root = os.path.join(CONFIG["results_folder"], "spheres", f"mic_{m['idx']}")
                out_path = os.path.join(spheres_root, f"spherical_cardioid_signal_{f}Hz.pkl")
                save_pickle(sphere_data, out_path)


       
if __name__ == "__main__":
    main()

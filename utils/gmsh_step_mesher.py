import os
import pickle
import numpy as np
import gmsh
from mpi4py import MPI
from utils.mesh_utils import create_mesh


def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)
    # print(f"Saved pickle to {path}")


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_surface_normal(surface_tag):
    try:
        uv = [0.5, 0.5]
        normal = gmsh.model.getNormal(surface_tag, uv)
        norm_array = np.array(normal)
        norm_array /= np.linalg.norm(norm_array)
        return norm_array
    except Exception as e:
        print(f"Warning: Could not get normal for surface {surface_tag}: {e}")
        return None


def classify_orientation(normal):
    if normal is None:
        return "unknown"
    if np.isclose(np.abs(normal[2]), 1.0, atol=0.1):
        return "horizontal"
    elif np.isclose(np.abs(normal[2]), 0.0, atol=0.1):
        return "vertical"
    else:
        return "angled"


def _initialize_gmsh(show_terminal, optimize, threshold, num_threads=12, algorithm2d=None, algorithm3d=None):
    if not gmsh.isInitialized():
        gmsh.initialize()
    if num_threads is not None:
        gmsh.option.setNumber("General.NumThreads", int(num_threads))
    gmsh.option.setString("Geometry.OCCTargetUnit", "M")
    gmsh.option.setNumber("General.Terminal", int(show_terminal))
    gmsh.option.setNumber("Mesh.Optimize", int(optimize))
    gmsh.option.setNumber("Mesh.OptimizeThreshold", threshold)

    if algorithm2d is not None:
        gmsh.option.setNumber("Mesh.Algorithm", float(algorithm2d))

    if algorithm3d is not None:
        if isinstance(algorithm3d, str):
            alg = algorithm3d.strip().lower()
            if alg in {"hxt", "hxtdelaunay", "hxt-delaunay"}:
                algorithm3d = 10
            elif alg in {"delaunay", "del", "tetgen"}:
                algorithm3d = 1
            elif alg in {"frontal", "frontal-delaunay", "frontal_delaunay"}:
                algorithm3d = 4
            else:
                raise ValueError(
                    f"Unknown algorithm3d='{algorithm3d}'. Use an int (e.g. 10) or one of: 'hxt', 'delaunay', 'frontal'."
                )
        gmsh.option.setNumber("Mesh.Algorithm3D", float(algorithm3d))


def _finalize_gmsh():
    if gmsh.isInitialized():
        gmsh.finalize()


def _extract_geometry_metadata(volumes):
    volume_to_surfaces = {}
    surface_orientations = {}

    for dim, vol_tag in volumes:
        gmsh.model.addPhysicalGroup(dim, [vol_tag], vol_tag)
        gmsh.model.setPhysicalName(dim, vol_tag, f"Volume_{vol_tag}")
        boundaries = gmsh.model.getBoundary([(dim, vol_tag)], oriented=False)
        surface_phys_ids = []

        for surf_dim, surf_tag in boundaries:
            phys_tag = 100000 + surf_tag
            gmsh.model.addPhysicalGroup(surf_dim, [surf_tag], phys_tag)
            gmsh.model.setPhysicalName(surf_dim, phys_tag, f"Surface_{surf_tag}_of_Volume_{vol_tag}")
            surface_phys_ids.append(phys_tag)

            normal = get_surface_normal(surf_tag)
            orientation = classify_orientation(normal)
            surface_orientations[phys_tag] = {
                "normal": normal.tolist() if normal is not None else None,
                "orientation": orientation
            }

        volume_to_surfaces[vol_tag] = surface_phys_ids

    return volume_to_surfaces, surface_orientations


def _convert_msh_to_xdmf(step_base, msh_files_by_freq, OUTPATH):
    xdmf_paths = {}
    for freq_tag, msh_path in msh_files_by_freq.items():
        xdmf_path = os.path.join(OUTPATH, f"{step_base}_{freq_tag}Hz.xdmf")
        if os.path.exists(xdmf_path):
            print(f"XDMF already exists for {freq_tag} Hz: {xdmf_path}")
        else:
            try:
                if create_mesh(MPI.COMM_SELF, meshPath=msh_path, xdmfPath=xdmf_path, gdim=3, mode="w", name="mesh"):
                    print(f"XDMF written for {freq_tag} Hz: {xdmf_path}")
                else:
                    print(f"Failed to convert {msh_path} to XDMF")
            except Exception as e:
                print(f"Exception while converting {msh_path} to XDMF: {e}")
        xdmf_paths[freq_tag] = xdmf_path

    return xdmf_paths


def mesh_range(step_file, frequencies, show_meshing_info=False, data_pkl_path=None,
               outpath=None, c=343, optimize=False, opt_threshold=0.4, div=10,
               num_threads=12, algorithm3d=None, algorithm2d=None):
    step_dir = os.path.dirname(step_file)
    step_base = os.path.splitext(os.path.basename(step_file))[0]
    OUTPATH = outpath or step_dir
    os.makedirs(OUTPATH, exist_ok=True)
    DATA_PKL = data_pkl_path or os.path.join(OUTPATH, f"{step_base}_mesh_data.pkl")

    msh_files_by_freq = {}

    try:
        _initialize_gmsh(
            show_meshing_info,
            optimize,
            opt_threshold,
            num_threads=num_threads,
            algorithm2d=algorithm2d,
            algorithm3d=algorithm3d,
        )
        gmsh.model.add(step_base)
        gmsh.model.occ.importShapes(step_file)
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(3)
        if not volumes:
            raise RuntimeError("No volume entities found in the STEP file.")

        volume_to_surfaces, surface_orientations = _extract_geometry_metadata(volumes)

        for freq in frequencies:
            msh_path = os.path.join(OUTPATH, f"{step_base}_{freq}Hz.msh")
            
            mesh_size = c / freq / div
            print(f"Meshing for {freq} Hz : Target h = {mesh_size:.4f}")
            gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)

            gmsh.model.mesh.clear()
            
            if os.path.exists(msh_path):
                print(f"Mesh already exists for {freq} Hz: {msh_path}")
            else:
                gmsh.model.mesh.generate(3)
                gmsh.write(msh_path)
                print(f"Mesh saved: {msh_path}")
            msh_files_by_freq[freq] = msh_path

    except Exception as e:
        print(f"Meshing failed: {e}")
        return None

    finally:
        _finalize_gmsh()

    mesh_data = {
        "volume": volume_to_surfaces,
        "orientation": surface_orientations,
        "msh_paths": msh_files_by_freq,
        "frequencies": sorted(msh_files_by_freq.keys())
    }
    save_pickle(mesh_data, DATA_PKL)
    print(f"Mesh metadata saved to: {DATA_PKL}")

    return DATA_PKL


def mesh_range_from_planes(planes, frequencies, show_meshing_info=False, data_pkl_path=None,
                           outpath=None, c=343, optimize=False, opt_threshold=0.4, div=10,
                           num_threads=12, algorithm3d=None, algorithm2d=None,
                           wall_groups=None):
    """
    Create a closed volume from a list of triangle planes and mesh it for a
    set of frequencies. Each plane is a list of three XYZ points.

    wall_groups: optional list of lists of triangle indices per wall.
                 If None, assumes planes are ordered by walls in pairs.
    """
    OUTPATH = outpath or os.getcwd()
    os.makedirs(OUTPATH, exist_ok=True)
    DATA_PKL = data_pkl_path or os.path.join(OUTPATH, "planes_mesh_data.pkl")

    msh_files_by_freq = {}

    try:
        _initialize_gmsh(
            show_meshing_info,
            optimize,
            opt_threshold,
            num_threads=num_threads,
            algorithm2d=algorithm2d,
            algorithm3d=algorithm3d,
        )
        gmsh.model.add("planes_room")

        # Build unique point/line caches
        point_cache = {}
        line_cache = {}
        surface_tags = []

        def _get_point_tag(pt):
            key = tuple(np.round(np.asarray(pt, dtype=float), 12))
            if key in point_cache:
                return point_cache[key]
            tag = gmsh.model.occ.addPoint(float(key[0]), float(key[1]), float(key[2]))
            point_cache[key] = tag
            return tag

        def _get_line_tag(p1, p2):
            key = (min(p1, p2), max(p1, p2))
            if key in line_cache:
                tag = line_cache[key]
                return tag if (p1, p2) == key else -tag
            tag = gmsh.model.occ.addLine(p1, p2)
            line_cache[key] = tag
            return tag

        def _order_points(points):
            if len(points) <= 3:
                return points
            pts = np.array(points, dtype=float)
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)
            axis = np.argmax(np.abs(normal))
            if axis == 0:
                proj = pts[:, 1:]
            elif axis == 1:
                proj = pts[:, [0, 2]]
            else:
                proj = pts[:, :2]
            center = np.mean(proj, axis=0)
            angles = np.arctan2(proj[:, 1] - center[1], proj[:, 0] - center[0])
            order = np.argsort(angles)
            return [tuple(points[i]) for i in order]

        # Physical groups for walls (auto-group by plane if not provided)
        if wall_groups is None:
            tol = 1e-8
            plane_map = {}
            for tri_idx, tri in enumerate(planes):
                pts = np.array(tri, dtype=float)
                if pts.shape[0] != 3:
                    raise ValueError("Each plane must have exactly 3 points (triangle).")
                n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
                n_norm = np.linalg.norm(n)
                if n_norm == 0:
                    raise ValueError(f"Triangle {tri_idx} is degenerate.")
                n = n / n_norm
                # Consistent sign
                for k in range(3):
                    if abs(n[k]) > tol:
                        if n[k] < 0:
                            n = -n
                        break
                d = -np.dot(n, pts[0])
                key = (
                    round(n[0] / tol) * tol,
                    round(n[1] / tol) * tol,
                    round(n[2] / tol) * tol,
                    round(d / tol) * tol,
                )
                plane_map.setdefault(key, []).append(tri_idx)

            wall_groups = list(plane_map.values())

        wall_phys_map = {}
        for idx, tri_indices in enumerate(wall_groups, start=1):
            wall_points = []
            for tri_idx in tri_indices:
                tri = planes[tri_idx]
                if len(tri) != 3:
                    raise ValueError("Each plane must have exactly 3 points (triangle).")
                wall_points.extend(tri)

            unique = []
            seen = set()
            for pt in wall_points:
                key = tuple(np.round(np.asarray(pt, dtype=float), 12))
                if key in seen:
                    continue
                seen.add(key)
                unique.append(key)

            ordered = _order_points(unique)
            if len(ordered) < 3:
                raise ValueError(f"Wall {idx}: not enough unique points to create a surface.")

            # Check coplanarity
            pts = np.array(ordered, dtype=float)
            n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            n_norm = np.linalg.norm(n)
            if n_norm == 0:
                raise ValueError(f"Wall {idx}: degenerate points, cannot compute normal.")
            n = n / n_norm
            dists = np.abs((pts - pts[0]) @ n)
            if np.max(dists) > 1e-8:
                raise ValueError(f"Wall {idx}: points are not coplanar (max dist {np.max(dists)}).")

            p_tags = [_get_point_tag(p) for p in ordered]
            lines = [_get_line_tag(p_tags[i], p_tags[(i + 1) % len(p_tags)]) for i in range(len(p_tags))]
            cl = gmsh.model.occ.addCurveLoop(lines)
            try:
                s = gmsh.model.occ.addPlaneSurface([cl])
            except Exception as exc:
                raise RuntimeError(f"Wall {idx}: Could not create surface from points {ordered}") from exc
            surface_tags.append(s)

            wall_surf_tags = [s]
            wall_phys_map[idx] = wall_surf_tags

        # Create volume from all wall surfaces
        sl = gmsh.model.occ.addSurfaceLoop(surface_tags)
        vol_tag = gmsh.model.occ.addVolume([sl])

        # Cleanup duplicated entities to improve meshing performance
        gmsh.model.occ.removeAllDuplicates()
        try:
            gmsh.model.occ.healShapes()
            print("Geometry healed to remove duplicates and fix issues.")
        except Exception:
            print("Geometry healing failed, but continuing with meshing. Mesh quality may be affected.")
            pass

        gmsh.model.occ.synchronize()

        # Add wall physical groups after synchronize
        for idx, wall_surf_tags in wall_phys_map.items():
            gmsh.model.addPhysicalGroup(2, wall_surf_tags, idx)
            gmsh.model.setPhysicalName(2, idx, f"Wall_{idx}")

        volumes = [(3, vol_tag)]
        volume_to_surfaces, surface_orientations = _extract_geometry_metadata(volumes)

        for freq in frequencies:
            msh_path = os.path.join(OUTPATH, f"planes_{freq}Hz.msh")
            mesh_size = c / freq / div
            print(f"Meshing for {freq} Hz : Target h = {mesh_size:.4f}")
            gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)

            gmsh.model.mesh.clear()

            if os.path.exists(msh_path):
                print(f"Mesh already exists for {freq} Hz: {msh_path}")
            else:
                gmsh.model.mesh.generate(3)
                gmsh.write(msh_path)
                print(f"Mesh saved: {msh_path}")
            msh_files_by_freq[freq] = msh_path

    except Exception as e:
        print(f"Meshing failed: {e}")
        return None

    finally:
        _finalize_gmsh()

    # xdmf_paths = _convert_msh_to_xdmf("planes", msh_files_by_freq, OUTPATH)

    # mesh_data = {
    #     "volume": volume_to_surfaces,
    #     "orientation": surface_orientations,
    #     "xdmf_paths": xdmf_paths,
    #     "frequencies": sorted(xdmf_paths.keys()),
    #     "wall_groups": wall_phys_map,
    # }
    mesh_data = {
        "volume": volume_to_surfaces,
        "orientation": surface_orientations,
        "msh_paths": msh_files_by_freq,
        "frequencies": sorted(msh_files_by_freq.keys()),
        "wall_groups": wall_phys_map,
    }
    save_pickle(mesh_data, DATA_PKL)
    print(f"Mesh metadata saved to: {DATA_PKL}")

    return DATA_PKL


from mpi4py import MPI
from dolfinx.io import XDMFFile
import numpy as np

# dolfinx 0.10+: dolfinx.io.gmsh (read_from_msh returns MeshData)
# dolfinx <=0.9: dolfinx.io.gmshio (read_from_msh returns (mesh, ct, ft))
try:
    from dolfinx.io import gmsh as gmshio  # type: ignore
except Exception:  # pragma: no cover
    from dolfinx.io import gmshio  # type: ignore

# Functions to read Mesh
def mpi_print(s, comm: MPI.Comm):
    print(f"Rank {comm.rank}: {s}")

def read_xdmf_data(xdmfPath: str, comm: MPI.Comm,gdim:int, name:str = "mesh"):
    from dolfinx.io import XDMFFile
    from dolfinx.mesh import GhostMode
    
    with XDMFFile(comm, xdmfPath, "r") as xdmf:
        mesh = xdmf.read_mesh(name=name, ghost_mode=GhostMode.shared_facet)
        mesh.topology.create_connectivity(gdim-1, gdim)

        ct = xdmf.read_meshtags(mesh, name=f"{name}_cells")
        ft = xdmf.read_meshtags(mesh, name=f"{name}_facets")
    return mesh, ct, ft


def _read_from_msh_compat(meshPath: str, comm: MPI.Comm, gdim: int):
    """Read a .msh file across dolfinx versions.

    dolfinx 0.10 returns a MeshData namedtuple.
    Older versions return (mesh, cell_tags, facet_tags).
    """

    ret = gmshio.read_from_msh(meshPath, comm, gdim=gdim)
    if hasattr(ret, "mesh"):
        # dolfinx.io.gmsh.MeshData
        mesh = ret.mesh
        cell_tags = getattr(ret, "cell_tags", None)
        facet_tags = getattr(ret, "facet_tags", None)
        return mesh, cell_tags, facet_tags

    # dolfinx <=0.9
    try:
        mesh, cell_tags, facet_tags = ret
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Unexpected return from read_from_msh: {type(ret)}") from e
    return mesh, cell_tags, facet_tags

    
def create_mesh(comm: MPI.Comm, meshPath: str, xdmfPath: str, gdim: int,mode: str = "w", name: str = "mesh"):
    """Create a DOLFINx from a Gmsh model and output to file.

    Args:
        comm: MPI communicator top create the mesh on.
        meshPath: Gmsh .msh Path.
        xdmfPath: XDMF filename.
        gdim: Geometry dimension of the mesh.
        mode: XDMF file mode. "w" (write) or "a" (append).
        name: Name (identifier) of the mesh to add.
    """
    meshWritten = False
    msh, ct, ft = _read_from_msh_compat(meshPath, comm, gdim=gdim)

    # Try to enforce a stable mesh name when writing XDMF
    if name is not None and hasattr(msh, "name"):
        try:
            msh.name = name
        except Exception:
            pass

    if ft is not None:
        ft.name = f"{msh.name}_facets"
    if ct is not None:
        ct.name = f"{msh.name}_cells"
    with XDMFFile(msh.comm, xdmfPath, mode) as file:
        msh.topology.create_connectivity(gdim-1, gdim)

        file.write_mesh(msh)

        geometry_xpath = f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry"
        if ct is not None:
            file.write_meshtags(ct, msh.geometry, geometry_xpath=geometry_xpath)
        if ft is not None:
            file.write_meshtags(ft, msh.geometry, geometry_xpath=geometry_xpath)
        meshWritten = True

    return meshWritten


def refine_mesh(domain, ct, ft, comm: MPI.Comm, option: int = 3):
    """Refine a mesh using DOLFINx's refine function."""
    from dolfinx.mesh import refine, transfer_meshtag

    domain.topology.create_connectivity(domain.topology.dim, 1)

    refined_domain, parent_cells, parent_facets = refine(
        domain,
        edges=None,  # or specify edges for adaptive refinement
        partitioner=None,  # or leave default to repartition
        option=option
    )
    refined_domain.topology.create_connectivity(refined_domain.topology.dim, refined_domain.topology.dim - 1)
    refined_domain.topology.create_connectivity(0, refined_domain.topology.dim)

    ct_refined = transfer_meshtag(ct, refined_domain, parent_cells)
    ft_refined = transfer_meshtag(ft, refined_domain, parent_cells, parent_facets)

    return refined_domain, ct_refined, ft_refined


from dolfinx.geometry import compute_collisions_points, BoundingBoxTree
import numpy as np

def is_point_inside_mesh(domain, point, tol=1e-8):
    """
    Check if a given point lies within the mesh domain.
    """
    point = np.array(point, dtype=np.float64).reshape(-1, 1)  # Shape (gdim, 1)
    tree = BoundingBoxTree(domain, domain.topology.dim)
    collisions = compute_collisions_points(tree, point, tol)
    cell_indices = collisions.links(0)
    return len(cell_indices) > 0


def mesh_midpoint(domain):
    """
    Compute the geometric center (midpoint) of the mesh bounding box.
    Works in parallel using collective MPI communication.
    """
    local_coords = domain.geometry.x

    # Compute local bounding box
    local_min = np.min(local_coords, axis=0)
    local_max = np.max(local_coords, axis=0)

    # Reduce across all ranks to get global bounding box
    global_min = np.empty_like(local_min)
    global_max = np.empty_like(local_max)

    domain.comm.Allreduce(local_min, global_min, op=MPI.MIN)
    domain.comm.Allreduce(local_max, global_max, op=MPI.MAX)

    # Midpoint of bounding box
    midpoint = (global_min + global_max) / 2.0
    return midpoint


def safe_mesh_midpoint(domain, tol=1e-10):
    midpoint = mesh_midpoint(domain)

    tree = BoundingBoxTree(domain, domain.topology.dim)
    point = np.array(midpoint, dtype=np.float64).reshape(-1, 1)
    collisions = compute_collisions_points(tree, point, tol)
    cell_ids = collisions.links(0)

    if len(cell_ids) > 0:
        return midpoint  # Midpoint is inside the mesh
    else:
        raise ValueError("Mesh midpoint is outside the domain volume! Try another method.")
import numpy as np
from dolfinx import geometry, default_scalar_type
import numpy.typing as npt
from dolfinx.mesh import meshtags, locate_entities
import dolfinx.io
from mpi4py import MPI


class PointLocator:
    def __init__(self, domain, positions: npt.ArrayLike, *, allow_closest: bool = False):
        self._domain = domain
        self._allow_closest = bool(allow_closest)
        self._positions = np.asarray(positions, dtype=domain.geometry.x.dtype).reshape(-1, 3)
        self._local_cells, self._local_points, self._local_indices = self._locate_points()

    def _locate_points(self):
        mesh = self._domain
        dim = mesh.topology.dim

        # Only accept owned (non-ghost) cells to avoid duplicate point hits across ranks.
        cell_map = mesh.topology.index_map(dim)
        num_owned_cells = cell_map.size_local

        bb_tree = geometry.bb_tree(mesh, dim)
        cell_candidates = geometry.compute_collisions_points(bb_tree, self._positions)
        colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, self._positions)

        local_cells = []
        local_points = []
        local_indices = []

        # Optional fallback: nearest cell by midpoint (can be useful for debugging,
        # but will generally yield different cells per rank if the point is outside).
        midpoint_tree = None
        if self._allow_closest:
            entity_indices = locate_entities(mesh, dim, lambda x: np.full(x.shape[1], True))
            midpoint_tree = geometry.create_midpoint_tree(mesh, dim, entity_indices)

        for i, pt in enumerate(self._positions):
            candidate_cells = colliding_cells.links(i)
            if len(candidate_cells) > 0:
                # Prefer an owned cell if available (avoid ghost duplicates)
                owned = [c for c in candidate_cells if c < num_owned_cells]
                if owned:
                    cell = owned[0]
                else:
                    # Only ghost cells collide on this rank -> ignore to prevent duplicates
                    continue
            elif self._allow_closest:
                pt_np = np.asarray(pt, dtype=mesh.geometry.x.dtype)
                cell = int(geometry.compute_closest_entity(bb_tree, midpoint_tree, mesh, pt_np)[0])
                if cell >= num_owned_cells:
                    continue
            else:
                # Point not inside any local owned cell on this rank
                continue

            local_cells.append(cell)
            local_points.append(pt)
            local_indices.append(i)

        return (
            np.asarray(local_cells, dtype=np.int32),
            np.asarray(local_points, dtype=mesh.geometry.x.dtype),
            np.asarray(local_indices, dtype=np.int32)
        )

    def get(self):
        return self._local_cells, self._local_points, self._local_indices

    def evaluate(self, field):
        if len(self._local_cells) > 0:
            return field.eval(self._local_points, self._local_cells)
        else:
            value_size = field.function_space.value_size
            return np.zeros((0, value_size), dtype=field.dtype)
        



def mark_cells(domain, cell_indices, tag=1):
    """
    Create MeshTags object marking specific cells.
    
    Parameters:
        domain: dolfinx.mesh.Mesh
        cell_indices: array of cell indices (int32)
        tag: integer tag to apply

    Returns:
        dolfinx.mesh.meshtags
    """
    tags = np.full(len(cell_indices), tag, dtype=np.int32)
    return meshtags(domain, domain.topology.dim, cell_indices, tags)

def save_meshtags(filename, domain, cell_tags, field=None):
    """
    Save mesh and tags to file (e.g., for ParaView).

    Parameters:
        filename: path (without extension)
        domain: dolfinx.mesh.Mesh
        cell_tags: dolfinx.mesh.meshtags
        field: optional dolfinx.fem.Function to include
    """
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{filename}.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_meshtags(cell_tags, domain.geometry)
        if field:
            xdmf.write_function(field)
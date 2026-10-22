import numpy as np
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

from nn_predictors.misc import row_features


class InterfaceRowSampler:
    """
    Samples a P1 vector field at fixed depths below each interface node of the retained domain.

    For interface node i at (x_i, y_i, z_iface) the sample points are (x_i, y_i, z_iface - d) for each
    depth d, so every interface node gets a "column" of values and the input features stay one-to-one
    with the interface nodes. Points with d > 0 are pulled radially inwards by `radial_shrink` so they
    stay inside the faceted cylinder surface.

    Point values are exact P1 interpolation, precomputed as barycentric weights of the containing cell,
    so sampling is a gather and a weighted sum (and a linear map of the nodal values).
    """

    def __init__(self, V, interface_nodes: np.ndarray, depths, radial_shrink: float = 0.01) -> None:
        self.depths = np.asarray(depths, dtype=float)
        self.n_nodes = len(interface_nodes)
        mesh = V.mesh

        coords = V.tabulate_dof_coordinates()
        iface_xyz = coords[interface_nodes]

        points = []
        for d in self.depths:
            p = iface_xyz.copy()
            if d > 0:
                p[:, :2] *= 1.0 - radial_shrink
            p[:, 2] -= d
            points.append(p)
        self.points = np.concatenate(points)  # [n_depths * n_nodes, 3], depth-major

        tree = bb_tree(mesh, mesh.topology.dim)
        candidates = compute_collisions_points(tree, self.points)
        colliding = compute_colliding_cells(mesh, candidates, self.points)

        self.node_idx = np.zeros((len(self.points), 4), dtype=np.int64)
        self.weights = np.zeros((len(self.points), 4))
        for k, p in enumerate(self.points):
            cells = colliding.links(k)
            if len(cells) == 0:
                raise ValueError(f"Sample point {p} (depth {self.depths[k // self.n_nodes]} mm) is outside the mesh.")
            cell_nodes = V.dofmap.cell_dofs(cells[0])
            X = coords[cell_nodes]
            A = np.vstack([X.T, np.ones(4)])
            lam = np.linalg.solve(A, np.append(p, 1.0))
            self.node_idx[k] = cell_nodes
            self.weights[k] = lam

        if self.weights.min() < -1e-8:
            raise ValueError("Negative barycentric weight: a sample point is not inside its cell.")

    def sample(self, values: np.ndarray) -> np.ndarray:
        """Field values (flat P1 vector array) at the sample points, shape [n_depths, n_nodes, 3]."""
        nodal = values.reshape(-1, 3)
        sampled = np.einsum("pk,pkc->pc", self.weights, nodal[self.node_idx])
        return sampled.reshape(len(self.depths), self.n_nodes, 3)

    def features(self, u: np.ndarray, v: np.ndarray, velocity_at_interface: bool = True, include_velocity: bool = True) -> np.ndarray:
        """
        Node-major features [n_nodes * n_features]: per interface node the displacements at all depths,
        then the velocities at all depths (optionally without the velocity at depth 0).
        """
        return row_features(self.sample(u), self.sample(v), self.depths, velocity_at_interface, include_velocity)

    def n_features(self, velocity_at_interface: bool = True, include_velocity: bool = True) -> int:
        n_v = len(self.depths) if velocity_at_interface else int(np.sum(self.depths > 0))
        if not include_velocity:
            n_v = 0
        return 3 * (len(self.depths) + n_v)

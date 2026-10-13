import numpy as np
import logging
from typing import Literal

import gmsh
from mpi4py import MPI
from dolfinx.fem import functionspace
from dolfinx.mesh import create_submesh, exterior_facet_indices, compute_midpoints
from dolfinx.io import gmshio, XDMFFile

from fem_sim.structural_sims import StructuralSimulation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TCTSimulation3DDisc(StructuralSimulation):
    """
    3D TCT Simulation with a discontinuous interface.
    The domain represents a 3D column with a central hole/gap at the mid-plane (z = 50.0),
    creating two disjoint circular interface sections.
    """
    # Bottom BC: Sinusoidal displacement (Time-dependent)
    amplitude = 1.0

    # Domain and Mesh parameters
    column_radius = 25.0
    hole_radius = 10.0     # Horizontal circular hole bored through the center at z = 50
    height = 100.0
    z_split = 50.0

    use_top_half = False

    def __init__(
        self,
        frequency: int = 1000,
        constitutive_model: Literal["elastic", "plastic"] = "elastic",
        configuration: Literal["benchmark", "scaled"] = "scaled",
    ) -> None:
        super().__init__()
        self.omega = 2 * np.pi * frequency
        self.constitutive_model = constitutive_model
        if configuration != "scaled":
            raise ValueError(f"Only 'scaled' configuration is supported for 3D. Got {configuration}.")
        self.configuration = configuration

    def return_mesh(self, height: float):
        mesh_type = "unstructured"

        if height == 50.0:
            use_bottom_half = True
            height = 100.0
        else:
            use_bottom_half = False

        if mesh_type == "unstructured":
            filename = "mesh_files/scaled_problem_mesh_3d_disc.xdmf"

            try:
                # --- Load Mesh from XDMF ---
                mesh = None
                with XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
                    mesh = xdmf.read_mesh(name="mesh")
                logger.info("Using discontinuous 3D mesh file: %s", filename)
            except (RuntimeError, Exception) as e:
                logger.info(
                    "Generating 3D column mesh with horizontal through-hole disconnecting the interface... (%s)",
                    e,
                )
                gmsh.initialize()
                gmsh.model.add("column_with_hole_3d_disc")

                H = height
                Z_SPLIT = self.z_split
                R = self.column_radius
                R_HOLE = self.hole_radius

                # 1. Bottom cylinder: z in [0, 50]
                cyl_bot = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, Z_SPLIT, R)

                # 2. Top cylinder: z in [50, 100]
                cyl_top = gmsh.model.occ.addCylinder(0, 0, Z_SPLIT, 0, 0, H - Z_SPLIT, R)

                # 3. Horizontal cylindrical hole along the y-axis at z = Z_SPLIT
                # Bore through the entire column width (y from -(R+5) to +(R+5))
                y_start = -(R + 5.0)
                y_len = 2.0 * (R + 5.0)
                hole_bot = gmsh.model.occ.addCylinder(0, y_start, Z_SPLIT, 0, y_len, 0, R_HOLE)
                hole_top = gmsh.model.occ.addCylinder(0, y_start, Z_SPLIT, 0, y_len, 0, R_HOLE)

                # Cut hole out of bottom and top cylinders
                out_bot, _ = gmsh.model.occ.cut([(3, cyl_bot)], [(3, hole_bot)])
                out_top, _ = gmsh.model.occ.cut([(3, cyl_top)], [(3, hole_top)])

                # Fragment to ensure conformal shared interface at z = 50.0
                fragment_out, _ = gmsh.model.occ.fragment(out_bot, out_top)
                gmsh.model.occ.synchronize()

                # Physical volume containing both halves
                vols = [tag for (dim, tag) in fragment_out if dim == 3]
                gmsh.model.addPhysicalGroup(3, vols, 1)

                # Mesh sizing: refined around the through-hole and interface
                p1 = gmsh.model.occ.addPoint(0, -R, Z_SPLIT, 2.0)
                p2 = gmsh.model.occ.addPoint(0, 0.0, Z_SPLIT, 1.8)
                p3 = gmsh.model.occ.addPoint(0, R, Z_SPLIT, 2.0)
                gmsh.model.occ.synchronize()

                gmsh.model.mesh.field.add("Distance", 1)
                gmsh.model.mesh.field.setNumbers(1, "PointsList", [p1, p2, p3])
                gmsh.model.mesh.field.add("MathEval", 2)
                gmsh.model.mesh.field.setString(2, "F", "F1/25.0 * 3.5 + 2.0")
                gmsh.model.mesh.field.setAsBackgroundMesh(2)

                gmsh.model.mesh.generate(3)

                mesh, cell_tags, facet_tags = gmshio.model_to_mesh(
                    gmsh.model,
                    MPI.COMM_WORLD,
                    rank=0,
                    gdim=3,
                )
                gmsh.finalize()
                try:
                    with XDMFFile(MPI.COMM_WORLD, filename, "w") as xdmf:
                        xdmf.write_mesh(mesh)
                except Exception as ex:
                    logger.warning("Could not write XDMF mesh file %s: %s", filename, ex)

        return mesh, use_bottom_half

    def _define_mesh(self) -> None:
        self.mesh, use_bottom_half = self.return_mesh(self.height)

        y_interface = self.z_split
        tdim = self.mesh.topology.dim

        all_cell_indices = np.arange(self.mesh.topology.index_map(tdim).size_local, dtype=np.int32)
        cell_midpoints = compute_midpoints(self.mesh, tdim, all_cell_indices)
        z_coords = cell_midpoints[:, 2]

        bottom_mask = z_coords <= y_interface
        bottom_cells_final = all_cell_indices[bottom_mask]
        top_cells_final = all_cell_indices[~bottom_mask]

        total_assigned_cells = len(bottom_cells_final) + len(top_cells_final)
        all_cells_total = self.mesh.topology.index_map(tdim).size_local

        if total_assigned_cells != all_cells_total:
            raise RuntimeError(
                f"Internal Error: Could not assign all cells. Assigned {total_assigned_cells} of {all_cells_total}."
            )

        mesh_bottom, cell_map_bottom, vertex_map_bottom, node_map_bottom = create_submesh(
            self.mesh, tdim, bottom_cells_final
        )
        mesh_top, cell_map_top, vertex_map_top, node_map_top = create_submesh(
            self.mesh, tdim, top_cells_final
        )

        vdim = 0
        fdim = tdim - 1

        self.top_half_nodes = node_map_top
        self.top_half_cells = cell_map_top
        self.bottom_half_nodes = node_map_bottom

        interface_nodes_global_final = np.intersect1d(
            self.bottom_half_nodes,
            self.top_half_nodes,
            assume_unique=True,
        )

        self.interface_nodes = interface_nodes_global_final

        # Sort interface nodes consistently by physical coordinates (x, y, z)
        coords = np.around(self.mesh.geometry.x[self.interface_nodes], decimals=3)
        dt = [('x', coords.dtype), ('y', coords.dtype), ('z', coords.dtype)]
        ind = np.argsort(coords.ravel().view(dt), order=['x', 'y', 'z'])
        self.interface_nodes = self.interface_nodes[ind]

        # 1-to-1 mapping from sorted global interface nodes to local nodes in mesh_bottom:
        global_to_local_bottom = {gn: ln for ln, gn in enumerate(self.bottom_half_nodes)}
        self.interface_nodes_local = np.array(
            [global_to_local_bottom[gn] for gn in self.interface_nodes], dtype=np.int32
        )

        mesh_bottom.topology.create_connectivity(fdim, vdim)
        facet_to_vertex_bottom = mesh_bottom.topology.connectivity(fdim, vdim)
        mesh_bottom.topology.create_connectivity(fdim, tdim)

        bottom_boundary_facets_local = exterior_facet_indices(mesh_bottom.topology)
        interface_nodes_set = set(self.interface_nodes_local)

        interface_facets_local = []
        for facet_index in bottom_boundary_facets_local:
            connected_nodes = facet_to_vertex_bottom.links(facet_index)
            is_interface_facet = all(node in interface_nodes_set for node in connected_nodes)
            if is_interface_facet:
                interface_facets_local.append(facet_index)

        self.interface_facets = np.array(interface_facets_local, dtype=np.int32)

        self.local_overlap_nodes = np.arange(mesh_bottom.topology.index_map(vdim).size_local, dtype=np.int32)
        self.global_overlap_nodes = node_map_bottom

        N_cells_bottom = mesh_bottom.topology.index_map(tdim).size_local
        self.bottom_cells_local_indices = np.arange(N_cells_bottom, dtype=np.int32)
        self.bottom_cells_global_indices = cell_map_bottom

        self.mesh_t = mesh_bottom

        if use_bottom_half:
            self.mesh, _, _, _ = create_submesh(self.mesh, tdim, bottom_cells_final)
            self.interface_nodes = self.interface_nodes_local
            self.bottom_half_nodes = self.local_overlap_nodes
            self.global_overlap_nodes = self.local_overlap_nodes
            self.bottom_cells_global_indices = self.bottom_cells_local_indices

        if self.use_top_half:
            self.mesh, _, _, _ = create_submesh(self.mesh, tdim, top_cells_final)
            global_to_local_top = {gn: ln for ln, gn in enumerate(self.top_half_nodes)}
            self.interface_nodes_local = np.array(
                [global_to_local_top[gn] for gn in self.interface_nodes], dtype=np.int32
            )
            self.interface_nodes = self.interface_nodes_local

            self.mesh.topology.create_connectivity(fdim, vdim)
            facet_to_vertex_bottom = self.mesh.topology.connectivity(fdim, vdim)
            self.mesh.topology.create_connectivity(fdim, tdim)

            top_boundary_facets_local = exterior_facet_indices(self.mesh.topology)
            interface_nodes_set = set(self.interface_nodes_local)

            interface_facets_local = []
            for facet_index in top_boundary_facets_local:
                connected_nodes = facet_to_vertex_bottom.links(facet_index)
                is_interface_facet = all(node in interface_nodes_set for node in connected_nodes)
                if is_interface_facet:
                    interface_facets_local.append(facet_index)

            self.interface_facets = np.array(interface_facets_local, dtype=np.int32)
            self.top_half_nodes_t = np.arange(self.mesh.topology.index_map(vdim).size_local, dtype=np.int32)

    def get_interface_topology(self) -> tuple:
        """
        Returns (coords, connectivity) for the discontinuous interface.
        Coordinates are sorted, and connectivity contains the triangular facets
        from both circular interface sections.
        """
        coords = self.mesh.geometry.x[self.interface_nodes]

        fdim = self.mesh_t.topology.dim - 1
        vdim = 0
        self.mesh_t.topology.create_connectivity(fdim, vdim)
        facet_to_vertex = self.mesh_t.topology.connectivity(fdim, vdim)

        local_to_sorted_idx = {local_node: j for j, local_node in enumerate(self.interface_nodes_local)}

        connectivity = []
        for facet in self.interface_facets:
            local_nodes = facet_to_vertex.links(facet)
            sorted_indices = [local_to_sorted_idx[ln] for ln in local_nodes]
            connectivity.append(sorted_indices)

        return coords, np.array(connectivity, dtype=np.int32)

    def _preprocess(self) -> None:
        super()._preprocess()

        self.interface_dofs = self.get_dofs(self.interface_nodes)
        self.bottom_nodes = self.get_nodes(self.bottom_boundary)
        self.bottom_boundary_marker = 1111

        self.add_dirichlet_bc(self.bottom_boundary, self.bottom_boundary_marker)
        self.add_dirichlet_bc(self.top_boundary, 2222)

        self.traction_parameters = self.setup_traction_problem(
            self.mesh_t,
            self.interface_nodes_local,
            self.interface_facets,
            self.bottom_cells_global_indices,
            self.bottom_cells_local_indices,
            self.local_overlap_nodes,
            self.global_overlap_nodes,
        )

    def initial_velocity(self, x: np.ndarray) -> np.ndarray:
        res = np.zeros((3, len(x[1])))
        res[2, :] = (((100.0 - x[2]) / 100.0) * self.omega * self.amplitude * np.cos(self.omega * 0))
        return res

    def _define_differential_equations(self) -> None:
        super()._define_differential_equations()
        self.traction_problem = self.get_traction_problem(*self.traction_parameters)

    def bottom_displacement_function(self, t):
        return self.amplitude * np.sin(self.omega * t)

    @staticmethod
    def top_boundary(x):
        return np.isclose(x[2], 100.0)

    @staticmethod
    def bottom_boundary(x):
        return np.isclose(x[2], 0.0)

    @staticmethod
    def interface_boundary(x):
        return np.isclose(x[2], 50.0)

    @staticmethod
    def bottom_half(x):
        return x[2] < 50.4

    @staticmethod
    def top_half(x):
        return x[2] > 49.6

    def solve_time_step(self) -> None:
        bottom_bc_value = self.bottom_displacement_function(self.time)
        self.update_dirichlet_bc(
            np.array([0, 0, bottom_bc_value] * len(self.bottom_nodes), dtype=float),
            self.bottom_boundary_marker,
        )
        super().solve_time_step()


def tct_comp_3d_disc(
    extractor: StructuralSimulation, applicator: StructuralSimulation, filename: str
) -> None:
    extractor.run()
    extractor.postprocess("u", "u", "z", f"{filename}_full")

    applicator.run()
    applicator.postprocess("u", "u", "z", f"{filename}_applied")

    applicator.bottom_half_nodes = applicator.get_nodes(lambda x: x[2] < 50.4, sort=True)

    u_k_app_error = np.zeros(extractor.formatted_plot_results["u"].shape)
    u_k_app_error[:, extractor.bottom_half_nodes, :] = (
        extractor.formatted_plot_results["u"][:, extractor.bottom_half_nodes, :]
        - applicator.formatted_plot_results["u"][:, applicator.bottom_half_nodes, :]
    )
    extractor.postprocess(u_k_app_error, "u", "norm", f"{filename}_applied_error")

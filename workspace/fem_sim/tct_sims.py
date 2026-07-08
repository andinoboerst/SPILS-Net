import numpy as np
import logging
from typing import Literal

import pygmsh
import gmsh
from mpi4py import MPI
from dolfinx.fem import functionspace
from dolfinx.mesh import create_rectangle, CellType, create_submesh, exterior_facet_indices, compute_midpoints
from dolfinx.io import gmshio, XDMFFile

from fem_sim.structural_sims import StructuralSimulation


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TCTSimulation(StructuralSimulation):
    # Bottom BC: Sinusoidal displacement (Time-dependent)
    amplitude = 1.0

    # Domain and Mesh
    width = 100.0
    height = 50.0
    element_size_x = 5.0
    element_size_y = 5.0
    corner_point = (0.0, 0.0)

    use_top_half = False

    def __init__(self, frequency: int = 1000, constitutive_model: Literal["elastic", "plastic"] = "elastic", configuration: Literal["benchmark", "scaled"] = "scaled") -> None:
        super().__init__()

        self.omega = 2 * np.pi * frequency

        self.constitutive_model = constitutive_model
        if configuration not in ["benchmark", "scaled"]:
            raise ValueError(f"Unknown configuration: {configuration}. Needs to be one of ['benchmark', 'scaled'].")
        self.configuration = configuration

    def return_mesh(self, height: float, corner_point=None):
        if corner_point is None:
            corner_point = self.corner_point

        if self.configuration == "benchmark":
            mesh_type = "structured"
            self.c_damping = 0.0
        elif self.configuration == "scaled":
            mesh_type = "unstructured"
        else:
            raise ValueError(f"Unknown configuration: {self.configuration}. Needs to be one of ['benchmark', 'scaled'].")

        if height == 25.0:
            use_bottom_half = True
            height = 50.0
        else:
            use_bottom_half = False

        if mesh_type == "unstructured":

            filename = "mesh_files/scaled_problem_mesh.xdmf"

            try:
                # --- Load Mesh from XDMF ---
                # Initialize a dummy variable to hold the mesh object
                mesh = None
                with XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
                    mesh = xdmf.read_mesh(name="mesh")
                logger.info("Using mesh file.")
            except RuntimeError:

                # Define Geometry Parameters
                L, H = self.width, height  # Rectangle dimensions
                R = 5.0         # Hole radius
                center = [25.0, 15.0, 0.0]  # Center of the hole
                Y_SPLIT = 25.0  # The desired straight interface line

                # Set characteristic length (mesh size)
                lcar = 2

                # Define tags for the volumes
                VOLUME_TAG_TOP = 1
                VOLUME_TAG_BOTTOM = 2
                INTERFACE_TAG = 3  # Tag for the line itself

                with pygmsh.geo.Geometry() as geom:

                    # --- Define Points for the Boundary and Interface ---

                    # Corner points (Bottom-Left is self.corner_point)
                    p_bl = geom.add_point([self.corner_point[0], self.corner_point[1], 0.0], lcar)
                    p_br = geom.add_point([L, self.corner_point[1], 0.0], lcar)
                    p_tr = geom.add_point([L, H, 0.0], lcar)
                    p_tl = geom.add_point([self.corner_point[0], H, 0.0], lcar)

                    # Interface points at Y_SPLIT
                    p_il = geom.add_point([self.corner_point[0], Y_SPLIT, 0.0], lcar)
                    p_ir = geom.add_point([L, Y_SPLIT, 0.0], lcar)

                    # --- Define Lines ---

                    # Bottom boundary lines
                    line_b = geom.add_line(p_bl, p_br)
                    line_r_b = geom.add_line(p_br, p_ir)  # Right bottom
                    line_l_b = geom.add_line(p_il, p_bl)  # Left bottom

                    # Interface line (CRITICAL)
                    line_interface = geom.add_line(p_il, p_ir)

                    # Top boundary lines
                    line_t = geom.add_line(p_tr, p_tl)
                    line_r_t = geom.add_line(p_ir, p_tr)  # Right top
                    line_l_t = geom.add_line(p_tl, p_il)  # Left top

                    # --- Define Surfaces (Bottom and Top) ---

                    # Create the hole curve loop (same as before)
                    hole_loop = geom.add_circle(
                        x0=center,
                        radius=R,
                        mesh_size=lcar,
                        make_surface=False
                    ).curve_loop

                    # Bottom Surface
                    curve_loop_bottom = geom.add_curve_loop([
                        line_b,             # p_bl -> p_br
                        line_r_b,           # p_br -> p_ir
                        -line_interface,    # p_ir -> p_il (Reversed: Note the negative sign!)
                        line_l_b            # p_il -> p_bl
                    ])

                    # Top Surface (handle the hole here, assuming Y_SPLIT=25 is below center=35)
                    # Path: p_il -> p_ir -> p_tr -> p_tl -> p_il
                    curve_loop_top = geom.add_curve_loop([
                        line_interface,     # p_il -> p_ir (Original direction: Left-to-Right)
                        line_r_t,           # p_ir -> p_tr
                        line_t,            # p_tr -> p_tl (Need to reverse line_t if it was defined right-to-left)
                        line_l_t            # p_tl -> p_il
                    ])

                    # surface_bottom = geom.add_plane_surface(curve_loop_bottom)
                    # surface_top = geom.add_plane_surface(curve_loop_top)

                    if center[1] > Y_SPLIT:
                        surface_top = geom.add_plane_surface(curve_loop_top, holes=[hole_loop])
                        surface_bottom = geom.add_plane_surface(curve_loop_bottom)
                    else:
                        surface_top = geom.add_plane_surface(curve_loop_top)
                        surface_bottom = geom.add_plane_surface(curve_loop_bottom, holes=[hole_loop])

                    # --- Define Physical Groups ---

                    # Tag the interface line so you can apply boundary conditions later if needed
                    geom.add_physical([line_interface], label=f"Interface_{INTERFACE_TAG}")

                    # Tag the volumes (surfaces)
                    geom.add_physical([surface_top], label=f"Volume_{VOLUME_TAG_TOP}")
                    geom.add_physical([surface_bottom], label=f"Volume_{VOLUME_TAG_BOTTOM}")

                    # --- Generate Mesh ---
                    # This forces N equally spaced nodes on the line, but might constrain the overall mesh
                    gmsh.model.geo.mesh.setTransfiniteCurve(line_interface._id, int(L / lcar) + 1)

                    # gmsh.option.setNumber("Mesh.RecombineAll", 1)

                    geom.synchronize()
                    geom.generate_mesh(dim=2)

                    # Convert Gmsh's active model to DOLFINx mesh
                    mesh, cell_tags, facet_tags = gmshio.model_to_mesh(
                        gmsh.model,
                        MPI.COMM_WORLD,
                        rank=0,
                        gdim=2
                    )

                # Save mesh
                with XDMFFile(MPI.COMM_WORLD, filename, "w") as xdmf:
                    xdmf.write_mesh(mesh)

        elif mesh_type == "structured":
            # filename = "structured_mesh.xdmf"

            nx = int(self.width / self.element_size_x)
            ny = int((height - corner_point[1]) / self.element_size_y)

            mesh = create_rectangle(
                MPI.COMM_WORLD,
                cell_type=CellType.quadrilateral,
                points=(corner_point, (self.width, height)),
                n=(nx, ny)
            )
        else:
            raise ValueError(f"Mesh type {mesh_type} is not supported.")

        return mesh, use_bottom_half

    def _define_mesh(self) -> None:
        self.mesh, use_bottom_half = self.return_mesh(self.height)

        self.overlapping_func = self.bottom_half

        y_interface = 25.0
        tdim = self.mesh.topology.dim  # Cell dimension (2)

        # 1. Compute Midpoints
        tdim = self.mesh.topology.dim
        all_cell_indices = np.arange(self.mesh.topology.index_map(tdim).size_local, dtype=np.int32)
        cell_midpoints = compute_midpoints(self.mesh, tdim, all_cell_indices)
        y_coords = cell_midpoints[:, 1]  # y-coordinate of the midpoint

        # 2. Define Cell Indices using NumPy Mask (100% reliable)
        # Bottom is y <= 25.0
        bottom_mask = y_coords <= y_interface
        bottom_cells_final = all_cell_indices[bottom_mask]

        top_cells_final = all_cell_indices[~bottom_mask]

        # 3. Final Verification
        total_assigned_cells = len(bottom_cells_final) + len(top_cells_final)
        all_cells_total = self.mesh.topology.index_map(tdim).size_local

        if total_assigned_cells != all_cells_total:
            # This should now only fail if a centroid is exactly NaN or infinite, which is highly unlikely.
            raise RuntimeError(f"Internal Error: Could not assign all cells. Assigned {total_assigned_cells} of {all_cells_total}.")

        # --- Create Bottom Submesh ---
        mesh_bottom, cell_map_bottom, vertex_map_bottom, node_map_bottom = create_submesh(self.mesh, tdim, bottom_cells_final)

        # --- Create Top Submesh (Inverse) ---
        mesh_top, cell_map_top, vertex_map_top, node_map_top = create_submesh(self.mesh, tdim, top_cells_final)

        # Topological dimension of vertices
        vdim = 0
        fdim = tdim - 1  # Facet dimension (1)

        self.top_half_nodes = node_map_top
        self.top_half_cells = cell_map_top

        self.bottom_half_nodes = node_map_bottom

        interface_nodes_global_final = np.intersect1d(
            self.bottom_half_nodes,
            self.top_half_nodes,
            assume_unique=True
        )

        self.interface_nodes = interface_nodes_global_final

        interface_nodes_local_final = np.where(np.in1d(self.bottom_half_nodes, self.interface_nodes))[0]  # type: ignore

        self.interface_nodes_local = interface_nodes_local_final[interface_nodes_local_final >= 0]

        # Ensure connectivity from facet (fdim) to vertex (vdim) is built on the submesh
        mesh_bottom.topology.create_connectivity(fdim, vdim)
        facet_to_vertex_bottom = mesh_bottom.topology.connectivity(fdim, vdim)
        mesh_bottom.topology.create_connectivity(fdim, tdim)

        # Get all local facet indices (from 0 to N_facets-1)
        bottom_boundary_facets_local = exterior_facet_indices(mesh_bottom.topology)

        interface_nodes_set = set(self.interface_nodes_local)

        interface_facets_local = []

        # Loop over all facets in the bottom submesh
        for facet_index in bottom_boundary_facets_local:
            # Get the local nodes connected to this facet
            connected_nodes = facet_to_vertex_bottom.links(facet_index)

            # Check if ALL connected nodes are present in the interface_nodes_set
            is_interface_facet = all(node in interface_nodes_set for node in connected_nodes)

            if is_interface_facet:
                interface_facets_local.append(facet_index)

        # Convert the list to the final NumPy array
        self.interface_facets = np.array(interface_facets_local, dtype=np.int32)

        # 1. Get ALL local node indices in mesh_bottom
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

            # self.plot_mesh(self.mesh, self.interface_nodes, self.interface_facets, "bottom_mesh")

        if self.use_top_half:
            self.mesh, _, _, _ = create_submesh(self.mesh, tdim, top_cells_final)
            interface_nodes_local_final = np.where(np.in1d(self.top_half_nodes, self.interface_nodes))[0]  # type: ignore

            self.interface_nodes_local = interface_nodes_local_final[interface_nodes_local_final >= 0]
            self.interface_nodes = self.interface_nodes_local

            # Ensure connectivity from facet (fdim) to vertex (vdim) is built on the submesh
            self.mesh.topology.create_connectivity(fdim, vdim)
            facet_to_vertex_bottom = self.mesh.topology.connectivity(fdim, vdim)
            self.mesh.topology.create_connectivity(fdim, tdim)

            # Get all local facet indices (from 0 to N_facets-1)
            top_boundary_facets_local = exterior_facet_indices(self.mesh.topology)
            interface_nodes_set = set(self.interface_nodes_local)

            interface_facets_local = []

            # Loop over all facets in the bottom submesh
            for facet_index in top_boundary_facets_local:
                # Get the local nodes connected to this facet
                connected_nodes = facet_to_vertex_bottom.links(facet_index)

                # Check if ALL connected nodes are present in the interface_nodes_set
                is_interface_facet = all(node in interface_nodes_set for node in connected_nodes)

                if is_interface_facet:
                    interface_facets_local.append(facet_index)

            # Convert the list to the final NumPy array
            self.interface_facets = np.array(interface_facets_local, dtype=np.int32)

            self.top_half_nodes_t = np.arange(self.mesh.topology.index_map(vdim).size_local, dtype=np.int32)

            # self.plot_mesh(self.mesh, self.interface_nodes[:2], self.interface_facets, "top_mesh2")

        V = functionspace(self.mesh, (*self.element_type_disps, (2,)))
        coords = np.around(V.tabulate_dof_coordinates(), decimals=3)
        coords_dtype = coords.dtype
        dt = [('x', coords_dtype), ('y', coords_dtype), ('z', coords_dtype)]
        ind = np.argsort(coords[self.interface_nodes].ravel().view(dt), order=['x', 'y', 'z'])
        self.interface_nodes = self.interface_nodes[ind]

        # non_interface_nodes_local = np.array(list(set(self.local_overlap_nodes) - set(self.interface_nodes_local)))

        # self.plot_mesh(mesh_bottom, self.interface_nodes_local, self.interface_facets)
        # self.plot_mesh(self.mesh, self.interface_nodes, name="05_scaled_domain")

    def _preprocess(self) -> None:
        super()._preprocess()

        self.interface_dofs = self.get_dofs(self.interface_nodes)

        self.bottom_nodes = self.get_nodes(self.bottom_boundary)

        self.bottom_boundary_marker = 1111

        if self.configuration == "benchmark":
            self.add_dirichlet_bc(self.top_boundary, 2222, subspace=1)
            self.add_dirichlet_bc(self.left_boundary, 3333, subspace=0)
            self.add_dirichlet_bc(self.bottom_boundary, self.bottom_boundary_marker, subspace=1)
        else:
            self.add_dirichlet_bc(self.bottom_boundary, self.bottom_boundary_marker)
            self.add_dirichlet_bc(self.top_boundary, 2222)

        self.traction_parameters = self.setup_traction_problem(
            self.mesh_t,
            self.interface_nodes_local,
            self.interface_facets,
            self.bottom_cells_global_indices,
            self.bottom_cells_local_indices,
            self.local_overlap_nodes,
            self.global_overlap_nodes
        )

    def initial_velocity(self, x: np.ndarray) -> np.ndarray:
        res = np.zeros((2, len(x[1])))
        res[1, :] = (((50 - x[1]) / 50) * self.omega * self.amplitude * np.cos(self.omega * 0))
        return res

    def _define_differential_equations(self) -> None:
        super()._define_differential_equations()

        self.traction_problem = self.get_traction_problem(*self.traction_parameters)

    def bottom_displacement_function(self, t):
        if self.configuration == "benchmark":
            value = (self.amplitude * np.cos(self.omega * t) - self.amplitude) / 2
        else:
            value = self.amplitude * np.sin(self.omega * t)
        return value

    @staticmethod
    def top_boundary(x):
        return np.isclose(x[1], 50.0)

    @staticmethod
    def bottom_boundary(x):
        return np.isclose(x[1], 0.0)

    @staticmethod
    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    @staticmethod
    def interface_boundary(x):
        return np.isclose(x[1], 25.0)

    @staticmethod
    def bottom_half(x):
        return x[1] < 25.4

    @staticmethod
    def top_half(x):
        return x[1] > 24.6

    def solve_time_step(self) -> None:
        bottom_bc_value = self.bottom_displacement_function(self.time)

        if self.configuration == "benchmark":
            if self.step == 100:
                self.damping.value = 0
            self.update_dirichlet_bc(bottom_bc_value, self.bottom_boundary_marker)
        else:
            self.update_dirichlet_bc(np.array([0, bottom_bc_value] * len(self.bottom_nodes), dtype=float), self.bottom_boundary_marker)

        super().solve_time_step()


def tct_comp(extractor: StructuralSimulation, applicator: StructuralSimulation, filename: str) -> None:

    extractor.run()

    extractor.postprocess("u", "u", "y", f"{filename}_full")

    applicator.run()

    applicator.postprocess("u", "u", "y", f"{filename}_applied")

    applicator.bottom_half_nodes = applicator.get_nodes(lambda x: x[1] < 25.4, sort=True)  # type: ignore

    u_k_app_error = np.zeros(extractor.formatted_plot_results["u"].shape)
    u_k_app_error[:, extractor.bottom_half_nodes, :] = extractor.formatted_plot_results["u"][:, extractor.bottom_half_nodes, :] - applicator.formatted_plot_results["u"][:, applicator.bottom_half_nodes, :]  # type: ignore
    extractor.postprocess(u_k_app_error, "u", "norm", f"{filename}_applied_error")

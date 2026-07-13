import numpy as np
import logging
from typing import Literal

import pygmsh
import gmsh
from mpi4py import MPI
from dolfinx.fem import functionspace
from dolfinx.mesh import CellType, create_submesh, exterior_facet_indices, compute_midpoints
from dolfinx.io import gmshio, XDMFFile

from fem_sim.structural_sims import StructuralSimulation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TCTSimulation3D(StructuralSimulation):
    # Bottom BC: Sinusoidal displacement (Time-dependent)
    amplitude = 1.0

    # Domain and Mesh
    radius = 25.0
    height = 100.0
    
    use_top_half = False

    def __init__(self, frequency: int = 1000, constitutive_model: Literal["elastic", "plastic"] = "elastic", configuration: Literal["benchmark", "scaled"] = "scaled") -> None:
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
            filename = "mesh_files/scaled_problem_mesh_3d.xdmf"

            try:
                # --- Load Mesh from XDMF ---
                mesh = None
                with XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
                    mesh = xdmf.read_mesh(name="mesh")
                logger.info("Using mesh file.")
            except RuntimeError:
                logger.info("Generating new 3D mesh...")
                gmsh.initialize()
                gmsh.model.add("cylinder_mesh")
                
                R = self.radius
                H = height
                Z_SPLIT = 50.0
                
                # lcar at edge vs center
                lcar_center = 2.0
                lcar_edge = 6.0
                
                # Center points for bottom, middle, top
                p_b_c = gmsh.model.geo.addPoint(0, 0, 0, lcar_center)
                p_m_c = gmsh.model.geo.addPoint(0, 0, Z_SPLIT, lcar_center)
                p_t_c = gmsh.model.geo.addPoint(0, 0, H, lcar_center)
                
                # Edge points at bottom (4 points to make circle)
                p_b_1 = gmsh.model.geo.addPoint(R, 0, 0, lcar_edge)
                p_b_2 = gmsh.model.geo.addPoint(0, R, 0, lcar_edge)
                p_b_3 = gmsh.model.geo.addPoint(-R, 0, 0, lcar_edge)
                p_b_4 = gmsh.model.geo.addPoint(0, -R, 0, lcar_edge)
                
                # Edge points at middle
                p_m_1 = gmsh.model.geo.addPoint(R, 0, Z_SPLIT, lcar_edge)
                p_m_2 = gmsh.model.geo.addPoint(0, R, Z_SPLIT, lcar_edge)
                p_m_3 = gmsh.model.geo.addPoint(-R, 0, Z_SPLIT, lcar_edge)
                p_m_4 = gmsh.model.geo.addPoint(0, -R, Z_SPLIT, lcar_edge)
                
                # Edge points at top
                p_t_1 = gmsh.model.geo.addPoint(R, 0, H, lcar_edge)
                p_t_2 = gmsh.model.geo.addPoint(0, R, H, lcar_edge)
                p_t_3 = gmsh.model.geo.addPoint(-R, 0, H, lcar_edge)
                p_t_4 = gmsh.model.geo.addPoint(0, -R, H, lcar_edge)
                
                # Bottom arcs
                c_b_1 = gmsh.model.geo.addCircleArc(p_b_1, p_b_c, p_b_2)
                c_b_2 = gmsh.model.geo.addCircleArc(p_b_2, p_b_c, p_b_3)
                c_b_3 = gmsh.model.geo.addCircleArc(p_b_3, p_b_c, p_b_4)
                c_b_4 = gmsh.model.geo.addCircleArc(p_b_4, p_b_c, p_b_1)
                
                # Middle arcs
                c_m_1 = gmsh.model.geo.addCircleArc(p_m_1, p_m_c, p_m_2)
                c_m_2 = gmsh.model.geo.addCircleArc(p_m_2, p_m_c, p_m_3)
                c_m_3 = gmsh.model.geo.addCircleArc(p_m_3, p_m_c, p_m_4)
                c_m_4 = gmsh.model.geo.addCircleArc(p_m_4, p_m_c, p_m_1)
                
                # Top arcs
                c_t_1 = gmsh.model.geo.addCircleArc(p_t_1, p_t_c, p_t_2)
                c_t_2 = gmsh.model.geo.addCircleArc(p_t_2, p_t_c, p_t_3)
                c_t_3 = gmsh.model.geo.addCircleArc(p_t_3, p_t_c, p_t_4)
                c_t_4 = gmsh.model.geo.addCircleArc(p_t_4, p_t_c, p_t_1)
                
                # Vertical lines bottom to middle
                l_bm_1 = gmsh.model.geo.addLine(p_b_1, p_m_1)
                l_bm_2 = gmsh.model.geo.addLine(p_b_2, p_m_2)
                l_bm_3 = gmsh.model.geo.addLine(p_b_3, p_m_3)
                l_bm_4 = gmsh.model.geo.addLine(p_b_4, p_m_4)
                
                # Vertical lines middle to top
                l_mt_1 = gmsh.model.geo.addLine(p_m_1, p_t_1)
                l_mt_2 = gmsh.model.geo.addLine(p_m_2, p_t_2)
                l_mt_3 = gmsh.model.geo.addLine(p_m_3, p_t_3)
                l_mt_4 = gmsh.model.geo.addLine(p_m_4, p_t_4)
                
                # Curve loops and surfaces
                # Bottom circle
                cl_b = gmsh.model.geo.addCurveLoop([c_b_1, c_b_2, c_b_3, c_b_4])
                surf_b = gmsh.model.geo.addPlaneSurface([cl_b])
                
                # Middle circle
                cl_m = gmsh.model.geo.addCurveLoop([c_m_1, c_m_2, c_m_3, c_m_4])
                surf_m = gmsh.model.geo.addPlaneSurface([cl_m])
                
                # Top circle
                cl_t = gmsh.model.geo.addCurveLoop([c_t_1, c_t_2, c_t_3, c_t_4])
                surf_t = gmsh.model.geo.addPlaneSurface([cl_t])
                
                # Side surfaces bottom
                cl_s_b1 = gmsh.model.geo.addCurveLoop([c_b_1, l_bm_2, -c_m_1, -l_bm_1])
                surf_s_b1 = gmsh.model.geo.addSurfaceFilling([cl_s_b1])
                cl_s_b2 = gmsh.model.geo.addCurveLoop([c_b_2, l_bm_3, -c_m_2, -l_bm_2])
                surf_s_b2 = gmsh.model.geo.addSurfaceFilling([cl_s_b2])
                cl_s_b3 = gmsh.model.geo.addCurveLoop([c_b_3, l_bm_4, -c_m_3, -l_bm_3])
                surf_s_b3 = gmsh.model.geo.addSurfaceFilling([cl_s_b3])
                cl_s_b4 = gmsh.model.geo.addCurveLoop([c_b_4, l_bm_1, -c_m_4, -l_bm_4])
                surf_s_b4 = gmsh.model.geo.addSurfaceFilling([cl_s_b4])
                
                # Side surfaces top
                cl_s_t1 = gmsh.model.geo.addCurveLoop([c_m_1, l_mt_2, -c_t_1, -l_mt_1])
                surf_s_t1 = gmsh.model.geo.addSurfaceFilling([cl_s_t1])
                cl_s_t2 = gmsh.model.geo.addCurveLoop([c_m_2, l_mt_3, -c_t_2, -l_mt_2])
                surf_s_t2 = gmsh.model.geo.addSurfaceFilling([cl_s_t2])
                cl_s_t3 = gmsh.model.geo.addCurveLoop([c_m_3, l_mt_4, -c_t_3, -l_mt_3])
                surf_s_t3 = gmsh.model.geo.addSurfaceFilling([cl_s_t3])
                cl_s_t4 = gmsh.model.geo.addCurveLoop([c_m_4, l_mt_1, -c_t_4, -l_mt_4])
                surf_s_t4 = gmsh.model.geo.addSurfaceFilling([cl_s_t4])
                
                # Volumes
                sl_b = gmsh.model.geo.addSurfaceLoop([surf_b, surf_s_b1, surf_s_b2, surf_s_b3, surf_s_b4, surf_m])
                vol_b = gmsh.model.geo.addVolume([sl_b])
                
                sl_t = gmsh.model.geo.addSurfaceLoop([surf_m, surf_s_t1, surf_s_t2, surf_s_t3, surf_s_t4, surf_t])
                vol_t = gmsh.model.geo.addVolume([sl_t])
                
                gmsh.model.geo.synchronize()
                
                gmsh.model.addPhysicalGroup(3, [vol_b, vol_t], 1)
                
                # Add size field to refine mesh in center
                gmsh.model.mesh.field.add("Distance", 1)
                gmsh.model.mesh.field.setNumbers(1, "PointsList", [p_b_c, p_m_c, p_t_c])
                gmsh.model.mesh.field.add("MathEval", 2)
                gmsh.model.mesh.field.setString(2, "F", "F1/25 * 4 + 2")
                gmsh.model.mesh.field.setAsBackgroundMesh(2)
                
                gmsh.model.mesh.generate(3)
                
                # Convert to dolfinx mesh
                mesh, cell_tags, facet_tags = gmshio.model_to_mesh(
                    gmsh.model,
                    MPI.COMM_WORLD,
                    rank=0,
                    gdim=3
                )
                
                gmsh.finalize()

                with XDMFFile(MPI.COMM_WORLD, filename, "w") as xdmf:
                    xdmf.write_mesh(mesh)

        return mesh, use_bottom_half

    def _define_mesh(self) -> None:
        self.mesh, use_bottom_half = self.return_mesh(self.height)
        
        y_interface = 50.0
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
            raise RuntimeError(f"Internal Error: Could not assign all cells. Assigned {total_assigned_cells} of {all_cells_total}.")

        mesh_bottom, cell_map_bottom, vertex_map_bottom, node_map_bottom = create_submesh(self.mesh, tdim, bottom_cells_final)
        mesh_top, cell_map_top, vertex_map_top, node_map_top = create_submesh(self.mesh, tdim, top_cells_final)

        vdim = 0
        fdim = tdim - 1

        self.top_half_nodes = node_map_top
        self.top_half_cells = cell_map_top

        self.bottom_half_nodes = node_map_bottom

        interface_nodes_global_final = np.intersect1d(
            self.bottom_half_nodes,
            self.top_half_nodes,
            assume_unique=True
        )

        self.interface_nodes = interface_nodes_global_final

        interface_nodes_local_final = np.where(np.in1d(self.bottom_half_nodes, self.interface_nodes))[0]

        self.interface_nodes_local = interface_nodes_local_final[interface_nodes_local_final >= 0]

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
            interface_nodes_local_final = np.where(np.in1d(self.top_half_nodes, self.interface_nodes))[0]

            self.interface_nodes_local = interface_nodes_local_final[interface_nodes_local_final >= 0]
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

        V = functionspace(self.mesh, (*self.element_type_disps, (3,)))
        coords = np.around(V.tabulate_dof_coordinates(), decimals=3)
        coords_dtype = coords.dtype
        dt = [('x', coords_dtype), ('y', coords_dtype), ('z', coords_dtype)]
        ind = np.argsort(coords[self.interface_nodes].ravel().view(dt), order=['x', 'y', 'z'])
        self.interface_nodes = self.interface_nodes[ind]


    def _preprocess(self) -> None:
        super()._preprocess()

        self.interface_dofs = self.get_dofs(self.interface_nodes)
        self.bottom_nodes = self.get_nodes(self.bottom_boundary)
        self.bottom_boundary_marker = 1111

        # We constrain bottom (z=0) entirely except we drive it in z direction
        self.add_dirichlet_bc(self.bottom_boundary, self.bottom_boundary_marker)
        
        # We constrain top (z=100) entirely
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
        res = np.zeros((3, len(x[1])))
        res[2, :] = (((100.0 - x[2]) / 100.0) * self.omega * self.amplitude * np.cos(self.omega * 0))
        return res

    def _define_differential_equations(self) -> None:
        super()._define_differential_equations()
        self.traction_problem = self.get_traction_problem(*self.traction_parameters)

    def bottom_displacement_function(self, t):
        value = self.amplitude * np.sin(self.omega * t)
        return value

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
        self.update_dirichlet_bc(np.array([0, 0, bottom_bc_value] * len(self.bottom_nodes), dtype=float), self.bottom_boundary_marker)

        super().solve_time_step()


def tct_comp_3d(extractor: StructuralSimulation, applicator: StructuralSimulation, filename: str) -> None:

    extractor.run()
    extractor.postprocess("u", "u", "z", f"{filename}_full")

    applicator.run()
    applicator.postprocess("u", "u", "z", f"{filename}_applied")

    applicator.bottom_half_nodes = applicator.get_nodes(lambda x: x[2] < 50.4, sort=True)  # type: ignore

    u_k_app_error = np.zeros(extractor.formatted_plot_results["u"].shape)
    u_k_app_error[:, extractor.bottom_half_nodes, :] = extractor.formatted_plot_results["u"][:, extractor.bottom_half_nodes, :] - applicator.formatted_plot_results["u"][:, applicator.bottom_half_nodes, :]  # type: ignore
    extractor.postprocess(u_k_app_error, "u", "norm", f"{filename}_applied_error")

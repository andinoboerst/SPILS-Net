import abc
import numpy as np
import logging
from functools import partial
from PIL import Image, ImageDraw

import pyvista
from mpi4py import MPI
from dolfinx import default_scalar_type
from dolfinx.fem import Constant, Function, functionspace, dirichletbc, locate_dofs_topological
from dolfinx.mesh import meshtags, locate_entities
from dolfinx.plot import vtk_mesh
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from ufl import TestFunction, TrialFunction, Identity, Measure, grad, inner, dot, tr, sqrt, conditional, sym, gt, eq, replace, lhs, rhs
from petsc4py import PETSc

from fem_sim.plotting import format_vectors_from_flat, create_mesh_animation
from fem_sim.progress_bar import progressbar

logger = logging.getLogger("structural_sims")


class FenicsxSimulation(metaclass=abc.ABCMeta):
    # Time Stepping
    time = 0.0
    time_total = 3e-3
    dt = 5e-7

    export_interval = 100

    linear_petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "superlu_dist",  # "mumps"
    }

    def __init__(self) -> None:
        pass

    def _plot_variables(self) -> dict:
        return {}

    @property
    def plot_variables_options(self) -> list:
        try:
            return self.plot_results.keys()
        except AttributeError:
            logger.warning("Simulation has not been set up yet.")
            return []

    def _preprocess(self) -> None:
        pass

    @abc.abstractmethod
    def _solve_time_step(self) -> None:
        raise NotImplementedError("Need to implement _solve_time_step()")

    @abc.abstractmethod
    def _define_mesh(self) -> None:
        raise NotImplementedError("Need to implement _define_mesh()")

    @abc.abstractmethod
    def _define_functionspace(self) -> None:
        raise NotImplementedError("Need to implement _define_functionspace()")

    @abc.abstractmethod
    def _init_variables(self) -> None:
        raise NotImplementedError("Need to implement _init_variables()")

    @abc.abstractmethod
    def _define_differential_equations(self) -> None:
        raise NotImplementedError("Need to implement _define_differential_equations()")

    # def save(self, path: str) -> None:
    #     with open(path, "wb") as f:
    #         dill.dump(self, f)

    # def load(self, path: str) -> None:
    #     with open(path, "rb") as f:
    #         return dill.load(f)

    def setup(self) -> None:
        self._dirichlet_bcs_list = []
        self._neumann_bcs_list = []

        self.num_steps = int((self.time_total - self.time) / self.dt)

        self._define_mesh()
        self.dim = self.mesh.geometry.dim

        self._define_functionspace()

        self._init_variables()

        self._preprocess()

        self._setup_bcs()

        self._define_differential_equations()

        self.plot_results = {key: [] for key in self._plot_variables().keys()}

    def get_nodes(self, marker, V=None, points: bool = True) -> np.array:
        if V is None:
            V = self.V

        mesh = V.mesh
        mesh_dim = mesh.topology.dim

        if points:
            fdim = 0
        else:
            fdim = mesh_dim

        coords = np.around(V.tabulate_dof_coordinates(), decimals=3)

        # Find the facets on the top boundary
        entities = locate_entities(mesh, fdim, marker)

        mesh.topology.create_connectivity(fdim, 2)
        nodes = locate_dofs_topological(V, fdim, entities)

        coords_dtype = coords.dtype
        dt = [('x', coords_dtype), ('y', coords_dtype), ('z', coords_dtype)]
        ind = np.argsort(coords[nodes].ravel().view(dt), order=['x', 'y', 'z'])
        return nodes[ind]

    def get_dofs(self, nodes, value_dim: int = 1, subspace: int = None) -> np.array:
        dofs = np.zeros(len(nodes) * self.dim * value_dim, dtype=np.int32)
        for i, node in enumerate(nodes):
            start = node * self.dim * value_dim
            node_dofs = range(start, start + (self.dim * value_dim))
            dofs[self.dim * value_dim * i:self.dim * value_dim * i + (self.dim * value_dim)] = node_dofs

        if subspace is not None:
            dofs = np.array(dofs[subspace::self.dim], dtype=np.int32)

        return dofs

    def _setup_bcs(self) -> None:
        self._setup_dirichlet_bcs()
        self._setup_neumann_bcs()

    def _setup_dirichlet_bcs(self) -> None:
        self._applied_dirichlet_bcs = ([], [])

        self._dirichlet_bcs = {}

        for boundary, marker, V, subspace in self._dirichlet_bcs_list:
            if isinstance(boundary, np.ndarray):
                nodes = boundary
            else:
                nodes = self.get_nodes(boundary, V=V)
            dofs = self.get_dofs(nodes, subspace=subspace)
            if subspace is None:
                val = Function(V)
            else:
                val = Constant(V.mesh, default_scalar_type(0.0))
            self._dirichlet_bcs[marker] = (val, dofs)

            if V not in self._applied_dirichlet_bcs[0]:
                self._applied_dirichlet_bcs[0].append(V.mesh)
                self._applied_dirichlet_bcs[1].append([])

            i = self._applied_dirichlet_bcs[0].index(V.mesh)
            if subspace is None:
                self._applied_dirichlet_bcs[1][i].append(dirichletbc(val, nodes))
            else:
                self._applied_dirichlet_bcs[1][i].append(dirichletbc(val, dofs, V.sub(subspace)))

    def get_dirichlet_bcs(self, mesh=None):
        if mesh is None:
            mesh = self.mesh

        try:
            index = self._applied_dirichlet_bcs[0].index(mesh)
            return self._applied_dirichlet_bcs[1][index]
        except ValueError:
            return []

    def _setup_neumann_bcs(self) -> None:
        self._neumann_bcs, facet_info = {}, {}
        for boundary, marker, V, facets in self._neumann_bcs_list:
            mesh = V.mesh
            fdim = mesh.topology.dim - 1
            if isinstance(boundary, np.ndarray):
                nodes = boundary
                if facets is None:
                    raise ValueError("Facets must be provided if boundary is an array of nodes instead of a function.")
            else:
                nodes = self.get_nodes(boundary, V=V)
                if facets is None:
                    facets = locate_entities(mesh, fdim, boundary)

            dofs = self.get_dofs(nodes)
            self._neumann_bcs[marker] = (Function(V), dofs)
            if mesh not in facet_info:
                facet_info[mesh] = (
                    [],
                    [],
                    [],
                )

            facet_info[mesh][0].append(facets)
            facet_info[mesh][1].append(np.full_like(facets, marker))
            facet_info[mesh][2].append(marker)

        self._applied_neumann_bcs = ([], [])
        for mesh, facet_info in facet_info.items():
            dx_ = Measure("dx", domain=mesh)
            v = TestFunction(V)
            L = inner(Constant(mesh, np.array([0.0] * self.dim)), v) * dx_
            facet_indices = np.hstack(facet_info[0]).astype(np.int32)
            facet_markers = np.hstack(facet_info[1]).astype(np.int32)
            sorted_facets = np.argsort(facet_indices)
            facet_tag = meshtags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

            ds = Measure("ds", domain=mesh, subdomain_data=facet_tag)

            for marker in facet_info[2]:
                L += dot(self._neumann_bcs[marker][0], v) * ds(marker)

            self._applied_neumann_bcs[0].append(mesh)
            self._applied_neumann_bcs[1].append(L)

    def apply_neumann_bcs(self, L, mesh=None):
        if mesh is None:
            mesh = self.mesh

        try:
            index = self._applied_neumann_bcs[0].index(mesh)
            L += self._applied_neumann_bcs[1][index]
        except ValueError:
            logger.info("No Neumann BCs applied to this function space.")

        return L

    def add_dirichlet_bc(self, boundary, marker: int, V=None, subspace: int = None) -> None:
        if V is None:
            V = self.V

        self._dirichlet_bcs_list.append((boundary, marker, V, subspace))

    def add_neumann_bc(self, boundary, marker: int, facets: np.ndarray = None, V=None) -> None:
        if V is None:
            V = self.V

        self._neumann_bcs_list.append((boundary, marker, V, facets))

    def update_dirichlet_bc(self, values, marker: int) -> None:
        dirichlet = self._dirichlet_bcs[marker]
        if isinstance(dirichlet[0], Function):
            dirichlet[0].x.array[dirichlet[1]] = values
        else:
            dirichlet[0].value = default_scalar_type(values)

    def update_neumann_bc(self, values, marker: int) -> None:
        neumann = self._neumann_bcs[marker]
        neumann[0].x.array[neumann[1]] = values

    def get_projection_problem(self, u_projected, u_result) -> tuple:
        V_proj = u_projected.function_space

        v = TestFunction(V_proj)
        du = TrialFunction(V_proj)

        dx_ = Measure("dx", domain=V_proj.mesh)

        a = inner(du, v) * dx_
        L = inner(u_result, v) * dx_

        return self.get_linear_problem(u_projected, a - L)

    def get_linear_problem(self, u, residual, bcs=[]):
        V = u.function_space
        du = TrialFunction(V)
        Residual_du = replace(residual, {u: du})
        a_form = lhs(Residual_du)
        L_form = rhs(Residual_du)

        problem = LinearProblem(
            a_form,
            L_form,
            bcs=bcs,
            petsc_options=self.linear_petsc_options,
        )

        class LinearSolver:
            def __init__(self, u, problem) -> None:
                self.u = u
                self.problem = problem

            def solve(self):
                self.u.x.array[:] = self.problem.solve().x.array[:]
        return LinearSolver(u, problem)

    def get_nonlinear_problem(self, u, residual, bcs=[]):
        problem = NonlinearProblem(
            residual,
            u,
            bcs=bcs,
        )
        solver = NewtonSolver(MPI.COMM_WORLD, problem)

        solver.max_it = 1000  # Increase max iterations
        solver.relaxation_parameter = 0.8  # Can reduce to 0.8 if needed
        solver.damping = 0.5  # Reduce the step size to stabilize convergence
        solver.ls = "bt"  # Use backtracking line search
        solver.convergence_criterion = "residual"
        solver.atol = 1e-8  # Absolute tolerance (adjust based on problem size)
        solver.rtol = 1e-6  # Relative tolerance
        ksp = solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "gmres"
        opts[f"{option_prefix}pc_type"] = "gamg"
        # opts[f"{option_prefix}pc_factork_mat_solver_type"] = "mumps"
        # opts[f"{option_prefix}pc_factor_levels"] = 5
        ksp.setFromOptions()

        solver.solve = partial(solver.solve, u)

        return solver

    def check_export_results(self) -> bool:
        return self.step % self.export_interval == 0

    def update_prev_values(self) -> None:
        pass

    def solve_time_step(self) -> None:
        self._solve_time_step()

        if self.check_export_results():
            self.export_results()

    def advance_time(self) -> None:
        self.time += self.dt

    def run(self) -> None:
        self.setup()

        self.export_results()

        for self.step in progressbar(range(self.num_steps)):
            self.advance_time()

            self.solve_time_step()

        self.format_results()

    def export_results(self) -> None:
        for key, var in self._plot_variables().items():
            if var[1] == "node":
                proj_func_space = functionspace(self.mesh, ("CG", 1, (2,)))
            elif var[1] == "element":
                proj_func_space = functionspace(self.mesh, ("DG", 0))

            proj_variable = Function(proj_func_space)

            problem = self.get_projection_problem(proj_variable, var[0])

            problem.solve()

            self.plot_results[key].append(proj_variable.x.array.copy())

    def format_results(self) -> None:
        plot_vars = self._plot_variables()
        self.formatted_plot_results = {key: format_vectors_from_flat(res, n_dim=self.dim) if plot_vars[key][1] == "node" else np.array(res) for key, res in self.plot_results.items()}

    def postprocess(self, scalars: str | np.ndarray | None = None, vectors: str = None, scalar_process: str = None, mesh=None, **kwargs) -> None:
        if mesh is None:
            mesh = self.mesh

        variables = {}
        variable_names = []
        if scalars is not None:
            if isinstance(scalars, str):
                scalar_variable = scalars
                variable_names.append(scalar_variable)
            else:
                scalar_variable = "scalar_variable"
                variables[scalar_variable] = scalars
        else:
            scalar_variable = None
            scalar_process = None

        if vectors is not None:
            if isinstance(vectors, str):
                vector_variable = vectors
                variable_names.append(vector_variable)
            else:
                vector_variable = "vector_variable"
                variables[vector_variable] = vectors
        else:
            vector_variable = None

        variable_names = set(variable_names)

        try:
            self.formatted_plot_results
        except AttributeError:
            self.format_results()

        for var in variable_names:
            if var in self.formatted_plot_results:
                variables[var] = self.formatted_plot_results[var]  # format_vectors_from_flat(self.plot_results[var], n_dim=self.dim)
            else:
                logger.warning(f"Variable {var} not found in plot results.")

        if scalar_process is None:
            scalar_value = variables.get(scalar_variable)
        elif scalar_process == "x":
            scalar_value = variables[scalar_variable][:, :, 0]
        elif scalar_process == "y":
            scalar_value = variables[scalar_variable][:, :, 1]
        elif scalar_process == "z":
            scalar_value = variables[scalar_variable][:, :, 2]
        elif scalar_process == "norm":
            scalar_value = np.linalg.norm(variables[scalar_variable], axis=-1)

        plot_mesh = vtk_mesh(mesh)
        create_mesh_animation(plot_mesh, scalar_value, variables.get(vector_variable), **kwargs)

    def plot_mesh(self, mesh, nodes=None, facets=None, name: str = "mesh") -> None:
        mesh_to_plot = mesh
        interface_nodes_local = nodes
        interface_facets_local = facets
        tdim = mesh_to_plot.topology.dim
        fdim = tdim - 1

        mesh_to_plot.topology.create_connectivity(fdim, tdim)

        grid_components = vtk_mesh(mesh_to_plot, tdim)
        grid = pyvista.UnstructuredGrid(*grid_components)

        # 2. Extract and Split Coordinates of Interface Nodes
        if nodes is not None:
            submesh_coords = mesh_to_plot.geometry.x
            interface_coords = submesh_coords[interface_nodes_local]
            num_points = len(interface_coords)  # Should be 51

            # --- NEW: Split the indices based on your range ---
            # Indices for the green subset: 0, 8, 16, 24, 32, 40, 48
            green_indices = np.arange(0, num_points, 8)

            # Indices for the remaining red nodes (everything NOT in green_indices)
            red_indices = np.setdiff1d(np.arange(num_points), green_indices)

            # Create two separate PolyData objects
            green_points = pyvista.PolyData(interface_coords[green_indices])
            red_points = pyvista.PolyData(interface_coords[red_indices])

        if facets is not None:
            interface_marker = 99
            facets_marker = np.full_like(interface_facets_local, interface_marker, dtype=np.int32)
            facet_tags = meshtags(mesh_to_plot, fdim, interface_facets_local, facets_marker)

        # --- High-DPI Scaling Setup ---
        hires_scale = 4
        scaled_point_size = 15 * hires_scale
        scaled_line_width = 2 * hires_scale

        p = pyvista.Plotter(off_screen=True)
        p.set_background('white')
        p.enable_anti_aliasing('ssaa')

        # --- Layer 1: Plot the Submesh ---
        p.add_mesh(
            grid,
            show_edges=True,
            color='skyblue',
            lighting=False,
            label='Bottom Submesh Cells',
            line_width=scaled_line_width,
        )

        if nodes is not None:
            # --- Layer 2a: Plot the Standard Interface Nodes (Red) ---
            p.add_mesh(
                red_points,
                color='red',
                render_points_as_spheres=True,
                point_size=scaled_point_size,
                label='Standard Interface Nodes',
            )

            # --- Layer 2b: Plot the Subset Interface Nodes (Green) ---
            p.add_mesh(
                green_points,
                color='green',
                render_points_as_spheres=True,
                # Optional: multiply scaled_point_size by 1.5 here if you want
                # the green spheres slightly larger so they pop out even more
                point_size=scaled_point_size,
                label='Subset Interface Nodes'
            )

        if facets is not None:
            # --- Layer 3: Plot the Interface Facets ---
            facet_grid_components = vtk_mesh(mesh_to_plot, fdim, entities=interface_facets_local)
            facet_grid = pyvista.UnstructuredGrid(*facet_grid_components)

            p.add_mesh(
                facet_grid,
                color='lime',
                style='wireframe',
                line_width=scaled_line_width,
                lighting=False,
                label='Interface Facets'
            )

        p.view_xy()

        # 5. Save as an ultra-high-resolution TIFF
        filename = f"figures/{name}.png"
        p.screenshot(filename, scale=hires_scale)
        p.close()

        # Open the high-res image we just saved
        img = Image.open(filename)

        # Convert to grayscale to evaluate pixel brightness
        gray = img.convert("L")

        # Threshold: if a pixel is almost white (> 250), make it background (0).
        # Otherwise, make it foreground (255). This cleans up any edge artifacts.
        bw = gray.point(lambda x: 0 if x > 250 else 255)

        draw = ImageDraw.Draw(bw)
        w, h = bw.size
        border = 3  # Wipe out the outermost 3 pixels

        # Draw background (0) over the absolute edges of the mask
        draw.rectangle([0, 0, w, border], fill=0)           # Top edge
        draw.rectangle([0, h - border, w, h], fill=0)       # Bottom edge
        draw.rectangle([0, 0, border, h], fill=0)           # Left edge
        draw.rectangle([w - border, 0, w, h], fill=0)       # Right edge

        # Get the strict bounding box of the domain elements
        bbox = bw.getbbox()

        if bbox:
            # Optional: Add a small pixel padding so the mesh doesn't touch the absolute edge
            # Because we scaled the image up so much, 50-100 pixels is a good padding size
            pad = 20
            padded_bbox = (
                max(0, bbox[0] - pad),
                max(0, bbox[1] - pad),
                min(img.size[0], bbox[2] + pad),
                min(img.size[1], bbox[3] + pad)
            )

            # Crop the image and overwrite the original file
            cropped_img = img.crop(padded_bbox)
            cropped_img.save(filename)


class StructuralSimulation(FenicsxSimulation):

    E = 200.0e3
    nu = 0.3
    rho = 7.85e-9
    c_damping = 0.02  # 0.02
    sigma_yield_0 = 250  # Yield stress (MPa)

    # E = 210.0e3
    # nu = 0.45
    # rho = 0.1e-9
    # c_damping = 0
    # sigma_yield_0 = 500  # Yield stress (MPa)

    body_force = np.array([0.0, 0.0])

    tol = 1e-10

    beta = 0.5
    gamma = 1

    element_type_disps = ("CG", 1)
    element_type_stress = ("DG", 0)

    constitutive_model_options = ["elastic", "plastic"]
    constitutive_model = "elastic"

    def __init__(self) -> None:
        super().__init__()

        if self.constitutive_model not in self.constitutive_model_options:
            raise ValueError(f"Unknown constitutive model: {self.constitutive_model}. Needs to be one of {self.constitutive_model_options}.")

    def _plot_variables(self) -> dict:
        return {
            "u": (self.u_next, "node"),
            "v": (self.v_next, "node"),
            "a": (self.a_next, "node"),
            "sigma": (self.sigma_vm(self.sig), "element"),
            "sigma_yield": (self.phi(self.sigma_elastic(self.u_next), self.eq_epsilon_p_k), "element"),
        }

    def _define_functionspace(self) -> None:
        self.V = functionspace(self.mesh, (*self.element_type_disps, (2,)))
        self.W = functionspace(self.mesh, self.element_type_stress)
        self.We = functionspace(self.mesh, (*self.element_type_stress, (2, 2)))
        self.Z = functionspace(self.mesh, (*self.element_type_stress, (2, 2)))

    def _init_variables(self) -> None:
        # Material parameters
        # x = SpatialCoordinate(self.mesh)
        # x_condition = And(gt(x[0], 50.0), lt(x[0], 80.0))
        # y_condition = And(gt(x[1], 35.0), lt(x[1], 45.0))
        # is_in_inclusion = And(x_condition, y_condition)

        # self.E = conditional(is_in_inclusion, 50.0e3, 200.0e3)
        self.E = Constant(self.mesh, 200.0e3)
        self.nu = Constant(self.mesh, self.nu)

        self.damping = Constant(self.mesh, self.c_damping)

        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lmbda = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

        self.H = self.E * 0.05

        # PDE variables
        self.f = Constant(self.mesh, self.body_force)  # Force term

        self.u_k = Function(self.V, name="Displacement")
        self.v_k = Function(self.V, name="Velocity")
        self.a_k = Function(self.V, name="Acceleration")

        self.u_next = Function(self.V)
        self.v_next = Function(self.V)
        self.a_next = Function(self.V)

        # Initialize variables to zero
        self.u_k.x.array[:] = 0.0
        self.v_k.x.array[:] = 0.0
        self.a_k.x.array[:] = 0.0

        # Plastic variables
        self.eq_epsilon_p_k = Function(self.W)
        self.eq_epsilon_p_next = Function(self.W)
        self.eq_epsilon_p_k.x.array[:] = 0.0

        self.sigma_k = Function(self.We)
        self.sigma_next = Function(self.We)
        self.sigma_k.x.array[:] = 0.0

        self.eps = Function(self.Z, name="Strain")
        self.sig = Function(self.Z, name="Stress")

    def setup_traction_problem(
        self,
        mesh_t,
        interface_nodes_t,
        interface_facets_t,
        overlap_cells_global,
        overlap_cells_local,
        overlap_nodes_local,
        overlap_nodes_global,
    ) -> tuple:

        V_t = functionspace(mesh_t, (*self.element_type_disps, (2,)))
        W_t = functionspace(mesh_t, self.element_type_stress)
        We_t = functionspace(mesh_t, (*self.element_type_stress, (2, 2)))

        self.f_res = Function(V_t)
        self.u_next_t = Function(V_t)
        self.u_k_t = Function(V_t)
        self.v_k_t = Function(V_t)
        self.a_k_t = Function(V_t)
        self.f_t = Constant(mesh_t, self.body_force)  # Force term
        self.eq_epsilon_p_k_t = Function(W_t)
        self.sigma_k_t = Function(We_t)

        # Full simulation
        self.overlapping_dofs_V = self.get_dofs(overlap_nodes_global)
        self.overlapping_dofs_W = overlap_cells_global
        self.overlapping_dofs_We = self.get_dofs(overlap_cells_global, value_dim=2)

        # Traction extraction
        self.overlapping_dofs_V_t = self.get_dofs(overlap_nodes_local)
        self.overlapping_dofs_W_t = overlap_cells_local
        self.overlapping_dofs_We_t = self.get_dofs(overlap_cells_local, value_dim=2)

        self.interface_dofs_t = self.get_dofs(interface_nodes_t)

        interface_marker_t = 88

        fdim = mesh_t.topology.dim - 1
        facets_marker = np.full_like(interface_facets_t, interface_marker_t).astype(np.int32)
        facet_tag = meshtags(mesh_t, fdim, interface_facets_t, facets_marker)

        ds_t = Measure("ds", domain=mesh_t, subdomain_data=facet_tag)(interface_marker_t)

        non_interface_nodes_local = np.array(list(set(overlap_nodes_local) - set(interface_nodes_t)))

        self.add_dirichlet_bc(non_interface_nodes_local, 1234, V_t)

        return self.f_res, self.u_next_t, self.u_k_t, self.v_k_t, self.a_k_t, self.f_t, ds_t, self.constitutive_model, self.eq_epsilon_p_k_t, self.sigma_k_t

    @staticmethod
    def epsilon(u):
        return sym(grad(u))

    def sigma_elastic(self, u):
        epsilon = self.epsilon(u)
        return self.lmbda * tr(epsilon) * Identity(self.dim) + 2 * self.mu * epsilon

    def sigma_trial(self, u, sigma_k, u_k):
        return sigma_k + self.sigma_elastic(u) - self.sigma_elastic(u_k)

    def sigma_dev(self, sigma):
        return sigma - (1 / 3) * tr(sigma) * Identity(self.dim)

    def sigma_vm(self, sigma):
        sigma_dev = self.sigma_dev(sigma)
        return sqrt((3 / 2) * inner(sigma_dev, sigma_dev))

    def sigma_yield(self, eq_epsilon_p):
        sig_0 = Function(eq_epsilon_p.function_space)
        sig_0.x.array[:] = self.sigma_yield_0
        return sig_0 + self.H * eq_epsilon_p

    def phi(self, sigma, eq_epsilon_p):
        return self.sigma_vm(sigma) - self.sigma_yield(eq_epsilon_p)

    def yield_condition(self, sigma, eq_epsilon_p):
        return conditional(gt(self.phi(sigma, eq_epsilon_p), self.tol), 1, 0)

    def delta_lambda(self, sigma, eq_epsilon_p):
        return self.phi(sigma, eq_epsilon_p) / (3 * self.mu + self.H)

    def delta_sigma(self, sigma, eq_epsilon_p):
        return - 3 * self.mu * self.delta_lambda(sigma, eq_epsilon_p) * self.sigma_dev(sigma) / self.sigma_vm(sigma)

    def eq_epsilon_p(self, u, eq_epsilon_p, sigma_k, u_k):
        sigma_trial = self.sigma_trial(u, sigma_k, u_k)
        return conditional(eq(self.yield_condition(sigma_trial, eq_epsilon_p), 1), eq_epsilon_p + self.delta_lambda(sigma_trial, eq_epsilon_p), eq_epsilon_p)

    def sigma_plastic(self, u, eq_epsilon_p, sigma_k, u_k):
        sigma_trial = self.sigma_trial(u, sigma_k, u_k)
        return conditional(eq(self.yield_condition(sigma_trial, eq_epsilon_p), 1), sigma_trial + self.delta_sigma(sigma_trial, eq_epsilon_p), sigma_trial)

    def velocity(self, u_next, u_k, v_k, a_k):
        return v_k + (1 - self.gamma) * self.dt * a_k + self.gamma * self.dt * self.acceleration(u_next, u_k, v_k, a_k)

    def acceleration(self, u_next, u_k, v_k, a_k):
        return (1 / self.beta) * ((self.beta - 0.5) * a_k + (1 / self.dt**2) * (u_next - u_k - self.dt * v_k))

    def get_constitutive_functions(self, constitutive_model, eq_epsilon_p, sigma_k, u_k) -> tuple:
        if constitutive_model == "elastic":
            sigma_func = self.sigma_elastic
            get_problem_func = self.get_linear_problem
            sigma_kwargs = {}
        elif constitutive_model == "plastic":
            if eq_epsilon_p is None:
                raise ValueError("alpha_k must be provided for plastic consitutive model.")
            sigma_func = self.sigma_plastic
            get_problem_func = self.get_nonlinear_problem
            sigma_kwargs = {
                "eq_epsilon_p": eq_epsilon_p,
                "sigma_k": sigma_k,
                "u_k": u_k,
            }

        return sigma_func, get_problem_func, sigma_kwargs

    def get_problem_equations(self, u_next, u_k, v_k, a_k, f, sigma_func, sigma_kwargs) -> tuple:
        V = u_next.function_space
        v = TestFunction(V)
        dx_ = Measure("dx", domain=V.mesh)

        stiffness_term = inner(sigma_func(u_next, **sigma_kwargs), self.epsilon(v)) * dx_
        # a = stiffness_term
        damping_term = self.damping * inner(self.velocity(u_next, u_k, v_k, a_k), v) * dx_
        mass_term = self.rho * inner(self.acceleration(u_next, u_k, v_k, a_k), v) * dx_
        a = mass_term + damping_term + stiffness_term

        L_body = dot(f, v) * dx_
        L = self.apply_neumann_bcs(L_body, V.mesh)

        return u_next, a - L, self.get_dirichlet_bcs(V.mesh)

    def get_traction_problem(self, f_interface, u_t_next, u_t_k, v_t_k, a_t_k, f_t, ds_t, constitutive_model: str = None, eq_epsilon_p_k=None, sigma_k=None) -> tuple:
        sigma_func, _, sigma_kwargs = self.get_constitutive_functions(constitutive_model, eq_epsilon_p_k, sigma_k, u_t_k)

        V_t = u_t_next.function_space
        v_t = TestFunction(V_t)

        dx_t = Measure("dx", domain=V_t.mesh)

        u_t_next, residual, bcs = self.get_problem_equations(u_t_next, u_t_k, v_t_k, a_t_k, f_t, sigma_func, sigma_kwargs)

        # return self.get_linear_problem(f_interface, residual)

        # residual -= dot(f_interface, v_t) * ds_t + dot(f_interface, v_t) * dx_t - dot(f_interface, v_t) * dx_t

        residual -= dot(f_interface, v_t) * ds_t + 1.0e-20 * dot(f_interface, v_t) * dx_t

        return self.get_linear_problem(f_interface, residual, bcs)

    def get_main_problems(self, u_next, v_next, a_next, u_k, v_k, a_k, f, constitutive_model: str = None, eq_epsilon_p_next=None, eq_epsilon_p_k=None, sigma_next=None, sigma_k=None):
        if eq_epsilon_p_k is not None and eq_epsilon_p_next is None:
            raise ValueError("eq_epsilon_p_next must be provided if eq_epsilon_p_k is provided.")

        sigma_func, get_problem_func, sigma_kwargs = self.get_constitutive_functions(constitutive_model, eq_epsilon_p_k, sigma_k, u_k)

        u_problem = get_problem_func(*self.get_problem_equations(u_next, u_k, v_k, a_k, f, sigma_func, sigma_kwargs))

        acceleration_problem = self.get_projection_problem(a_next, self.acceleration(u_next, u_k=u_k, v_k=v_k, a_k=a_k))
        velocity_problem = self.get_projection_problem(v_next, self.velocity(u_next=u_next, u_k=u_k, v_k=v_k, a_k=a_k))

        eps_problem = self.get_projection_problem(self.eps, self.epsilon(u_next))
        sig_problem = self.get_projection_problem(self.sig, sigma_func(u_next, **sigma_kwargs))

        if constitutive_model == "elastic":
            return u_problem, acceleration_problem, velocity_problem, eps_problem, sig_problem
        else:
            sigma_problem = self.get_projection_problem(sigma_next, sigma_func(u_next, **sigma_kwargs))
            eq_epsilon_p_problem = self.get_projection_problem(eq_epsilon_p_next, self.eq_epsilon_p(u_next, **sigma_kwargs))
            return u_problem, acceleration_problem, velocity_problem, eq_epsilon_p_problem, sigma_problem, eps_problem, sig_problem

    def calculate_interface_tractions(self) -> None:
        self.sigma_k_t.x.array[self.overlapping_dofs_We_t] = self.sigma_k.x.array[self.overlapping_dofs_We].copy()
        self.eq_epsilon_p_k_t.x.array[self.overlapping_dofs_W_t] = self.eq_epsilon_p_k.x.array[self.overlapping_dofs_W].copy()
        self.u_next_t.x.array[self.overlapping_dofs_V_t] = self.u_next.x.array[self.overlapping_dofs_V].copy()
        self.u_k_t.x.array[self.overlapping_dofs_V_t] = self.u_k.x.array[self.overlapping_dofs_V].copy()
        self.v_k_t.x.array[self.overlapping_dofs_V_t] = self.v_k.x.array[self.overlapping_dofs_V].copy()
        self.a_k_t.x.array[self.overlapping_dofs_V_t] = self.a_k.x.array[self.overlapping_dofs_V].copy()

        self.traction_problem.solve()

        return self.f_res.x.array[self.interface_dofs_t].copy()

    def _define_differential_equations(self):
        self.main_problems = self.get_main_problems(self.u_next, self.v_next, self.a_next, self.u_k, self.v_k, self.a_k, self.f, self.constitutive_model, self.eq_epsilon_p_next, self.eq_epsilon_p_k, self.sigma_next, self.sigma_k)

    def solve_u(self) -> None:

        if self.step == 50:
            self.beta = 0.25
            self.gamma = 0.5

        self.update_prev_values()

        for problem in self.main_problems:
            problem.solve()

        # logger.info(f"Solved step {self.step}, ||u_|| = {np.linalg.norm(self.u_next.x.array):.2f}, ||epsilon_|| = {np.linalg.norm(self.eq_epsilon_p_next.x.array):.2f}")

    def update_prev_values(self) -> None:
        self.u_k.x.array[:] = self.u_next.x.array[:]
        self.v_k.x.array[:] = self.v_next.x.array[:]
        self.a_k.x.array[:] = self.a_next.x.array[:]

        self.eq_epsilon_p_k.x.array[:] = self.eq_epsilon_p_next.x.array[:]
        self.sigma_k.x.array[:] = self.sigma_next.x.array[:]

    def _solve_time_step(self) -> None:
        self.solve_u()

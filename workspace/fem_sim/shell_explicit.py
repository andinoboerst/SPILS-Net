"""
Explicit Reissner-Mindlin shell model on linear triangles (flat facets), for crash-like shell problems.

Kinematics per node: displacement u and rotation vector theta (both global, 3 components each). On every
(flat) element with orthonormal frame (e1, e2, e3), e3 the consistently oriented normal, a fibre at
thickness coordinate z moves by z * beta with beta = theta x e3, and in local in-plane components

    membrane strain   eps_ab   = sym(e_a . grad(u) e_b)
    curvature         kappa_ab = sym(e_a . grad(beta) e_b)
    transverse shear  gamma_a  = e3 . grad(u) e_a + beta . e_a        (evaluated at the element centre)

With P1 fields membrane strain and curvature are constant per element; evaluating the shear at the centre
is the usual reduced integration against shear locking. The rotation about the normal (drilling) has no
physical stiffness and gets a small penalty.

Material: plane-stress J2 plasticity with linear isotropic hardening, integrated with Gauss points through
the thickness (return mapping in numpy); transverse shear is elastic. The stress resultants (membrane
forces N, moments M, shear forces Q) are assembled as nodal forces and moments with UFL.

Time integration: central differences with lumped translational and rotary masses, mass-proportional
damping treated semi-implicitly, prescribed dofs by divided differences (stable for varying dt). The time
step may change every step.
"""

import numpy as np
import ufl
from basix.ufl import element as basix_element
from dolfinx import default_scalar_type
from dolfinx.fem import Constant, Expression, Function, assemble_vector, form, functionspace
from dolfinx.la import InsertMode
from dolfinx.mesh import create_mesh
from mpi4py import MPI
from scipy.spatial import cKDTree
from ufl import as_vector, cross, dot, grad, inner


def square_tube_mesh(side: float = 50.0, height: float = 100.0, h: float = 2.5, z_min: float = 0.0):
    """
    Structured triangulation of the mid-surface of a square tube [-side/2, side/2]^2 x [z_min, height] with
    node rings every h in z and nodes every h along the perimeter (side / h must be an integer). Returns
    (points [n, 3], triangles [m, 3], ring_s [n] perimeter coordinate of each node).
    """
    n_side = int(round(side / h))
    if abs(n_side * h - side) > 1e-9:
        raise ValueError("side / h must be an integer")
    n_ring = 4 * n_side
    n_z = int(round((height - z_min) / h))
    s = np.arange(n_ring) * h  # perimeter coordinate, counter-clockwise from (-side/2, -side/2)

    def perimeter_xy(s):
        a = side / 2
        q, r = np.divmod(s, side)
        xy = np.zeros((len(s), 2))
        for k, (x0, y0, dx, dy) in enumerate([(-a, -a, 1, 0), (a, -a, 0, 1), (a, a, -1, 0), (-a, a, 0, -1)]):
            m = q == k
            xy[m, 0] = x0 + dx * r[m]
            xy[m, 1] = y0 + dy * r[m]
        return xy

    xy = perimeter_xy(s)
    z = z_min + np.arange(n_z + 1) * h
    points = np.array([[x, y, zz] for zz in z for x, y in xy])
    ring_s = np.tile(s, n_z + 1)

    cells = []
    for j in range(n_z):
        for i in range(n_ring):
            a, b = j * n_ring + i, j * n_ring + (i + 1) % n_ring
            c, d = a + n_ring, b + n_ring
            # Alternate the diagonal so the mesh has no preferred direction
            if (i + j) % 2 == 0:
                cells += [[a, b, d], [a, d, c]]
            else:
                cells += [[a, b, c], [b, d, c]]
    return points, np.array(cells, dtype=np.int64), ring_s


class ExplicitShell:
    """
    Explicit RM shell on a triangle mesh in 3D. Call setup(points, cells), then set prescribed dofs with
    prescribe_u / prescribe_theta (masks on [n_nodes, 3]) and advance with step(dt, u_prescribed).

    Units as the solid models: mm, s, t (tonne), N, MPa.
    """

    E = 200.0e3
    nu = 0.3
    rho = 7.85e-9
    sigma_yield_0 = 250.0
    hardening_ratio = 0.05  # H = hardening_ratio * E, as the solid model
    thickness = 1.5
    c_damping = 0.02  # mass-proportional damping per volume, as the solid model (alpha = c / rho)
    shear_factor = 5.0 / 6.0
    drilling_ratio = 1e-3  # drilling stiffness relative to the transverse shear stiffness
    n_thickness_points = 5
    rotary_inertia_scale = 1.0  # explicit codes scale rotary inertia up to relax the time-step limit
    plastic = True

    def __init__(self, normal_hint=None) -> None:
        # normal_hint(centroids [m, 3]) -> reference directions [m, 3] used to orient the element normals
        # consistently (the bending kinematics need it); default: +z
        self.normal_hint = normal_hint
        self.time = 0.0

    # ------------------------------------------------------------------ setup
    def setup(self, points: np.ndarray, cells: np.ndarray) -> None:
        coord_el = basix_element("Lagrange", "triangle", 1, shape=(3,))
        self.mesh = create_mesh(MPI.COMM_SELF, cells, points, ufl.Mesh(coord_el))
        mesh = self.mesh
        self.V = functionspace(mesh, ("Lagrange", 1, (3,)))
        self.n_nodes = self.V.dofmap.index_map.size_local
        # Node order of the input points -> dof (block) index of V
        dof_xyz = self.V.tabulate_dof_coordinates()[: self.n_nodes]
        dist, self.node_to_dof = cKDTree(dof_xyz).query(points)
        if dist.max() > 1e-8:
            raise RuntimeError("Could not match the input points to dofs")
        self.points = points

        # Element frames from the geometry, normals oriented with the hint
        W0 = functionspace(mesh, ("DG", 0, (3,)))
        self._cell_dofs = W0.dofmap.list.reshape(-1)  # DG0 dof of each cell
        X = mesh.geometry.x[mesh.geometry.dofmap]  # [m, 3, 3]
        e1 = X[:, 1] - X[:, 0]
        e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        n = np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0])
        self.cell_area = 0.5 * np.linalg.norm(n, axis=1)
        e3 = n / (2 * self.cell_area[:, None])
        centroids = X.mean(axis=1)
        hint = self.normal_hint(centroids) if self.normal_hint is not None else np.tile([0.0, 0.0, 1.0], (len(X), 1))
        e3 *= np.sign(np.einsum("ij,ij->i", e3, hint))[:, None]
        e2 = np.cross(e3, e1)
        self.frames = np.stack([e1, e2, e3], axis=1)  # rows e1, e2, e3
        self.centroids = centroids
        self.n_cells = len(X)

        self._e = [Function(W0) for _ in range(3)]
        for k in range(3):
            self._set_cell_values(self._e[k], self.frames[:, k])
        e1f, e2f, e3f = self._e

        # Fields
        self.u = Function(self.V)
        self.theta = Function(self.V)
        self.v_u = np.zeros(3 * self.n_nodes)
        self.v_t = np.zeros(3 * self.n_nodes)
        self.a_u = np.zeros(3 * self.n_nodes)
        self.a_t = np.zeros(3 * self.n_nodes)

        # Generalised strains at the element centre: [eps11, eps22, gam12, kap11, kap22, kap12(eng), gam1, gam2]
        Gu = grad(self.u)
        beta = cross(self.theta, e3f)
        Gb = grad(beta)
        strains = as_vector([
            dot(e1f, Gu * e1f),
            dot(e2f, Gu * e2f),
            dot(e1f, Gu * e2f) + dot(e2f, Gu * e1f),
            dot(e1f, Gb * e1f),
            dot(e2f, Gb * e2f),
            dot(e1f, Gb * e2f) + dot(e2f, Gb * e1f),
            dot(e3f, Gu * e1f) + dot(beta, e1f),
            dot(e3f, Gu * e2f) + dot(beta, e2f),
        ])
        self._W8 = functionspace(mesh, ("DG", 0, (8,)))
        self._strain_fn = Function(self._W8)
        pts = self._W8.element.interpolation_points
        self._strain_expr = Expression(strains, pts() if callable(pts) else pts)

        # Stress resultants in global form (N_g = T^T N T, M_g = T^T M T, Q_g = T^T Q) and the drilling rotation
        T9 = functionspace(mesh, ("DG", 0, (3, 3)))
        self._N = Function(T9)
        self._M = Function(T9)
        self._Q = Function(W0)
        G = self.E / (2 * (1 + self.nu))
        t = self.thickness
        k_shear = self.shear_factor * G * t
        self._k_drill = Constant(mesh, default_scalar_type(self.drilling_ratio * k_shear))

        wu, wt = ufl.TestFunction(self.V), ufl.TestFunction(self.V)
        dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 2})
        self._R_u = form((inner(self._N, grad(wu)) + dot(self._Q, grad(wu).T * e3f)) * dx)
        self._R_t = form((inner(self._M, grad(cross(wt, e3f))) + dot(self._Q, cross(wt, e3f))
                          + self._k_drill * dot(self.theta, e3f) * dot(wt, e3f)) * dx)

        # Lumped masses (row sums; positive for P1 triangles)
        ones = Constant(mesh, np.ones(3, dtype=default_scalar_type))
        area = assemble_vector(form(inner(ones, wu) * dx))
        area.scatter_reverse(InsertMode.add)
        nodal_area = area.array[: 3 * self.n_nodes].copy()
        self.m_u = self.rho * t * nodal_area
        self.m_t = self.rotary_inertia_scale * self.rho * t**3 / 12.0 * nodal_area
        alpha = self.c_damping / self.rho
        self.c_u = alpha * self.m_u
        self.c_t = alpha * self.m_t

        # Plastic state per cell and thickness point: plastic strain [eps11, eps22, gam12] and equivalent strain
        gp, gw = np.polynomial.legendre.leggauss(self.n_thickness_points)
        self._z = gp * t / 2
        self._wz = gw * t / 2
        self.eps_p = np.zeros((self.n_cells, self.n_thickness_points, 3))
        self.eq_eps_p = np.zeros((self.n_cells, self.n_thickness_points))
        self._eps_p_trial = self.eps_p.copy()
        self._eq_trial = self.eq_eps_p.copy()

        self.fixed_u = np.zeros(3 * self.n_nodes, dtype=bool)
        self.fixed_t = np.zeros(3 * self.n_nodes, dtype=bool)
        self._prev_rate = {}

    def _set_cell_values(self, fn: Function, values: np.ndarray) -> None:
        bs = fn.function_space.dofmap.index_map_bs
        arr = fn.x.array.reshape(-1, bs)
        arr[self._cell_dofs] = values.reshape(len(self._cell_dofs), bs)

    def node_dofs(self, nodes: np.ndarray) -> np.ndarray:
        """Flat dof indices [len(nodes) * 3] (node-major, x y z) of input-point nodes."""
        d = self.node_to_dof[np.asarray(nodes)]
        return (3 * d[:, None] + np.arange(3)).ravel()

    def prescribe_u(self, dofs: np.ndarray) -> None:
        self.fixed_u[dofs] = True

    def prescribe_theta(self, dofs: np.ndarray) -> None:
        self.fixed_t[dofs] = True

    # ------------------------------------------------------------------ material
    def _elastic_matrix(self) -> np.ndarray:
        c = self.E / (1 - self.nu**2)
        return c * np.array([[1, self.nu, 0], [self.nu, 1, 0], [0, 0, (1 - self.nu) / 2]])

    def _return_mapping(self, eps: np.ndarray) -> None:
        """Plane-stress J2 return mapping; eps [m, k, 3] -> self._sigma, trial plastic state."""
        C = self._elastic_matrix()
        sig_tr = (eps - self.eps_p) @ C.T
        self._eps_p_trial[:] = self.eps_p
        self._eq_trial[:] = self.eq_eps_p
        if not self.plastic:
            self._sigma = sig_tr
            return
        E, nu = self.E, self.nu
        G = E / (2 * (1 + nu))
        H = self.hardening_ratio * E
        c1 = E / (3 * (1 - nu))
        # Components in the common eigenbasis of C and P: sigma(dg) = a_i / d_i(dg)
        a1 = (sig_tr[..., 0] + sig_tr[..., 1]) / np.sqrt(2)
        a2 = (sig_tr[..., 1] - sig_tr[..., 0]) / np.sqrt(2)
        a3 = sig_tr[..., 2]
        q0 = a1**2 / 3 + a2**2 + 2 * a3**2
        f0 = 0.5 * q0 - (self.sigma_yield_0 + H * self.eq_eps_p) ** 2 / 3.0
        yielding = f0 > 1e-10 * self.sigma_yield_0**2
        self._sigma = sig_tr
        if not np.any(yielding):
            return
        # Newton on the consistency condition phi(dg) = q/2 - ybar^2/3 = 0 for the yielding points only
        A1, A2, A3, ep = a1[yielding], a2[yielding], a3[yielding], self.eq_eps_p[yielding]
        dg = np.zeros_like(A1)
        for _ in range(50):
            d1, d2 = 1 + c1 * dg, 1 + 2 * G * dg
            b1, b2, b3 = A1 / d1, A2 / d2, A3 / d2
            q = b1**2 / 3 + b2**2 + 2 * b3**2
            dq = -2 * (b1**2 / 3 * c1 / d1 + (b2**2 + 2 * b3**2) * 2 * G / d2)
            r = np.sqrt(2.0 / 3.0 * q)
            ybar = self.sigma_yield_0 + H * (ep + dg * r)
            dy = H * (r + dg * dq / (3 * np.maximum(r, 1e-300)))
            phi = 0.5 * q - ybar**2 / 3.0
            dphi = 0.5 * dq - 2.0 / 3.0 * ybar * dy
            step = phi / dphi
            dg = np.maximum(dg - step, 0.5 * dg)  # phi is decreasing; keep dg positive
            if np.all(np.abs(phi) <= 1e-12 * self.sigma_yield_0**2):
                break
        d1, d2 = 1 + c1 * dg, 1 + 2 * G * dg
        b1, b2, b3 = A1 / d1, A2 / d2, A3 / d2
        q = b1**2 / 3 + b2**2 + 2 * b3**2
        sigma = np.stack([(b1 - b2) / np.sqrt(2), (b1 + b2) / np.sqrt(2), b3], axis=-1)
        # Flow: d eps_p = dg * P sigma (engineering shear), d eq = dg * sqrt(2/3 sigma^T P sigma)
        P_sigma = np.stack([(2 * sigma[:, 0] - sigma[:, 1]) / 3, (2 * sigma[:, 1] - sigma[:, 0]) / 3, 2 * sigma[:, 2]], axis=-1)
        self._sigma = sig_tr.copy()
        self._sigma[yielding] = sigma
        self._eps_p_trial[yielding] = self.eps_p[yielding] + dg[:, None] * P_sigma
        self._eq_trial[yielding] = ep + dg * np.sqrt(2.0 / 3.0 * q)

    def _internal_forces(self) -> tuple:
        """Residual vectors (R_u, R_theta) at the current u, theta (plastic state not committed)."""
        self._strain_fn.interpolate(self._strain_expr)
        s = self._strain_fn.x.array.reshape(-1, 8)[self._cell_dofs]
        eps_m, kappa, gamma = s[:, 0:3], s[:, 3:6], s[:, 6:8]
        eps = eps_m[:, None, :] + self._z[None, :, None] * kappa[:, None, :]
        self._return_mapping(eps)
        N = np.einsum("k,mkc->mc", self._wz, self._sigma)
        M = np.einsum("k,mkc->mc", self._wz * self._z, self._sigma)
        G = self.E / (2 * (1 + self.nu))
        Q = self.shear_factor * G * self.thickness * gamma
        T = self.frames[:, :2, :]  # [m, 2, 3]

        def to_global(R):  # Voigt [11, 22, 12] -> T^T R T
            R2 = np.stack([np.stack([R[:, 0], R[:, 2]], -1), np.stack([R[:, 2], R[:, 1]], -1)], 1)
            return np.einsum("mai,mab,mbj->mij", T, R2, T)

        self._set_cell_values(self._N, to_global(N))
        self._set_cell_values(self._M, to_global(M))
        self._set_cell_values(self._Q, np.einsum("mai,ma->mi", T, Q))
        out = []
        for f in (self._R_u, self._R_t):
            r = assemble_vector(f)
            r.scatter_reverse(InsertMode.add)
            out.append(r.array[: 3 * self.n_nodes].copy())
        return tuple(out)

    # ------------------------------------------------------------------ time stepping
    def critical_time_step(self, iterations: int = 60, seed: int = 0) -> float:
        """2 / omega_max of the elastic, undamped system (power iteration on M^-1 K), with constraints."""
        plastic, self.plastic = self.plastic, False
        rng = np.random.default_rng(seed)
        free = np.concatenate([~self.fixed_u, ~self.fixed_t])
        m = np.concatenate([self.m_u, self.m_t])
        x = rng.standard_normal(len(m)) * free
        lam = 0.0
        u0, t0 = self.u.x.array.copy(), self.theta.x.array.copy()
        for _ in range(iterations):
            x /= np.sqrt(np.sum(m * x**2))
            self.u.x.array[: 3 * self.n_nodes] = x[: 3 * self.n_nodes] * 1e-6
            self.theta.x.array[: 3 * self.n_nodes] = x[3 * self.n_nodes :] * 1e-6
            Ru, Rt = self._internal_forces()
            Kx = np.concatenate([Ru, Rt]) / 1e-6 * free
            lam = np.sum(x * Kx)
            x = Kx / m
        self.u.x.array[:], self.theta.x.array[:] = u0, t0
        self.plastic = plastic
        return 2.0 / np.sqrt(lam)

    def step(self, dt: float, u_prescribed: np.ndarray | None = None, theta_prescribed: np.ndarray | None = None,
             f_ext: np.ndarray | None = None) -> None:
        """
        Advance by dt. u_prescribed / theta_prescribed: full flat arrays whose entries at the prescribed dofs
        are the values at the end of the step (other entries ignored); zero if None. f_ext: external nodal
        forces (flat, as u).
        """
        n3 = 3 * self.n_nodes
        u, th = self.u.x.array[:n3], self.theta.x.array[:n3]
        u_old, th_old = u.copy(), th.copy()
        u_new = u + dt * self.v_u + 0.5 * dt**2 * self.a_u
        th_new = th + dt * self.v_t + 0.5 * dt**2 * self.a_t
        u_new[self.fixed_u] = 0.0 if u_prescribed is None else u_prescribed[self.fixed_u]
        th_new[self.fixed_t] = 0.0 if theta_prescribed is None else theta_prescribed[self.fixed_t]
        self.u.x.array[:n3], self.theta.x.array[:n3] = u_new, th_new

        Ru, Rt = self._internal_forces()
        if f_ext is not None:
            Ru = Ru - f_ext
        for name, x_old, x_new, v, a, m, c, fixed, R in (
            ("u", u_old, u_new, self.v_u, self.a_u, self.m_u, self.c_u, self.fixed_u, Ru),
            ("t", th_old, th_new, self.v_t, self.a_t, self.m_t, self.c_t, self.fixed_t, Rt),
        ):
            v_star = v + 0.5 * dt * a
            a_new = (-R - c * v_star) / (m + 0.5 * dt * c)
            v_new = v_star + 0.5 * dt * a_new
            # Prescribed dofs: divided differences of the prescribed values (no recursion; see structural_sims)
            rate = (x_new[fixed] - x_old[fixed]) / dt
            prev = self._prev_rate.get(name)
            a_new[fixed] = 2.0 * (rate - v[fixed]) / dt if prev is None else 2.0 * (rate - prev[0]) / (dt + prev[1])
            v_new[fixed] = rate + 0.5 * dt * a_new[fixed]
            self._prev_rate[name] = (rate, dt)
            a[:], v[:] = a_new, v_new
        # Commit the plastic state of the accepted step
        self.eps_p[:] = self._eps_p_trial
        self.eq_eps_p[:] = self._eq_trial
        self.time += dt

    # ------------------------------------------------------------------ output helpers
    def nodal(self, values: np.ndarray, nodes: np.ndarray) -> np.ndarray:
        """Values of a flat nodal array at input-point nodes, [len(nodes), 3]."""
        return values[: 3 * self.n_nodes].reshape(-1, 3)[self.node_to_dof[np.asarray(nodes)]]
